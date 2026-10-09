"""
画面（skill_transfer.html）のテスト。実際にサーバーを起動し、スマホの画面幅のブラウザ
（Playwright + Chromium）で操作する。音声認識・LLMはモック。
Playwrightが無い環境ではスキップする。
"""
import os
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conftest import SEED_CSV, FakeLLM, FakeWhisperModel

playwright_api = pytest.importorskip("playwright.sync_api")

PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True}
SCREENSHOT_DIR = os.environ.get("SKILL_SCREENSHOT_DIR")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(skill, conn, tmp_path):
    app = FastAPI()
    llm = FakeLLM()
    ctx = skill.register_skill_transfer(app, conn=conn, whisper_model=FakeWhisperModel(), llm_fn=llm,
                                        media_dir=str(tmp_path / "media"), seed_csv=str(SEED_CSV),
                                        run_in_background=True)
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", ctx, llm
    srv.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def page():
    with playwright_api.sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(**PHONE)
        pg = context.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.errors = errors
        yield pg
        browser.close()


def screenshot(page, name):
    if SCREENSHOT_DIR:
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, f"{name}.png"), full_page=True)


def no_horizontal_scroll(page):
    return page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def upload_through_screen(page, base, video_path, title="型枠の建て込み"):
    page.goto(f"{base}/skill#/upload")
    page.set_input_files("#file-input", str(video_path))
    page.fill("#title", title)
    page.fill("#explainer", "山田")
    page.click("#upload-button")
    try:
        page.wait_for_selector("text=動画を見る", timeout=60000)
    except playwright_api.TimeoutError:
        # どの段階で止まったかを失敗の理由に出す
        status = page.inner_text("#status-area") if page.locator("#status-area").count() else ""
        pytest.fail(f"投稿の処理が終わりませんでした。画面の状態: {status!r}")


def test_page_is_served_from_the_cell(skill, conn, tmp_path):
    app = FastAPI()
    skill.register_skill_transfer(app, conn=conn, llm_fn=FakeLLM(), media_dir=str(tmp_path))
    res = TestClient(app).get("/skill")
    assert res.status_code == 200 and res.text == skill.SKILL_PAGE_HTML
    assert "技能伝承動画" in res.text


def test_upload_screen_has_camera_and_file_inputs(server, page):
    base, _, _ = server
    page.goto(f"{base}/skill#/upload")
    assert page.get_attribute("#camera-input", "capture") == "environment"
    assert page.get_attribute("#camera-input", "accept") == "video/*"
    assert page.get_attribute("#file-input", "capture") is None  # ファイル選択はカメラに限定しない
    assert no_horizontal_scroll(page)
    page.on("dialog", lambda d: d.accept())
    page.click("#upload-button")  # 動画もタイトルも無い → 送信しない
    assert page.locator("#upload-progress").is_hidden()
    assert page.errors == []


def test_upload_shows_progress_then_plays_with_subtitles_and_scene_jump(server, page, sample_video):
    base, ctx, _ = server
    upload_through_screen(page, base, sample_video)
    assert page.locator(".steps-status li.done").count() == 4
    screenshot(page, "01_upload_done")

    page.click("text=動画を見る")
    page.wait_for_selector("#player")
    assert page.inner_text("h2 >> nth=0") == "型枠の建て込み"
    assert page.get_attribute("#player", "src").endswith("/skill/api/videos/1/media")
    assert page.get_attribute("#subtitles", "src").endswith("/skill/api/videos/1/subtitles.vtt")
    assert page.locator(".scene-bar .mark").count() == 2
    assert page.locator(".step-card").count() == 2
    assert "コツ" in page.inner_text("#step-1") or "コツ" in page.inner_text(".step-card >> nth=0")
    assert page.locator(".scene-tag", has_text="セパレーター").count() == 1
    assert page.get_attribute("#pdf-link", "href").endswith("/skill/api/videos/1/procedure.pdf")
    assert no_horizontal_scroll(page)
    screenshot(page, "02_player")

    # 場面（手順2・場面タグ・再生バーの印）をタップすると、その開始時間に移動する
    page.click(".step-card >> nth=1 >> .step-head")
    assert page.evaluate("document.getElementById('player').currentTime") == pytest.approx(2.0, abs=0.05)
    page.click(".scene-tag >> text=セパレーター")
    assert page.evaluate("document.getElementById('player').currentTime") == pytest.approx(2.0, abs=0.05)
    page.click(".scene-bar .mark >> nth=0")
    assert page.evaluate("document.getElementById('player').currentTime") == pytest.approx(0.0, abs=0.05)

    # 字幕の表示・非表示
    assert page.evaluate("document.getElementById('player').textTracks[0].mode") == "showing"
    page.click("#subtitle-toggle")
    assert page.evaluate("document.getElementById('player').textTracks[0].mode") == "hidden"
    assert page.inner_text("#subtitle-toggle") == "字幕: 非表示"
    page.click("#subtitle-toggle")
    assert page.evaluate("document.getElementById('player').textTracks[0].mode") == "showing"
    assert page.errors == []


def test_subtitle_cues_are_loaded_by_browser(server, page, sample_video):
    base, _, _ = server
    upload_through_screen(page, base, sample_video)
    page.goto(f"{base}/skill#/video/1")
    page.wait_for_function(
        "document.getElementById('player') && document.getElementById('player').textTracks[0].cues"
        " && document.getElementById('player').textTracks[0].cues.length > 0", timeout=10000)
    texts = page.evaluate("Array.from(document.getElementById('player').textTracks[0].cues).map(c => c.text)")
    assert texts == ["まず型枠を立てます。", "次はセパを入れます。"]


def test_failed_processing_can_be_retried_from_screen(server, page, sample_video):
    base, ctx, llm = server
    llm.responses["procedure"] = ["だめ", "だめ", llm.responses["procedure"][0]]
    page.goto(f"{base}/skill#/upload")
    page.set_input_files("#file-input", str(sample_video))
    page.fill("#title", "型枠")
    page.click("#upload-button")
    page.wait_for_selector("text=失敗したところから再実行", timeout=30000)
    assert page.inner_text(".steps-status li.failed") == "手順書"
    screenshot(page, "03_upload_failed")
    page.click("text=失敗したところから再実行")
    page.wait_for_selector("text=動画を見る", timeout=30000)
    assert page.errors == []


def test_text_from_ai_is_escaped(server, page, sample_video):
    base, ctx, llm = server
    evil = {"steps": [{"title": "<img src=x onerror=window.hacked=1>", "description": "<b>太字</b>",
                       "start_segment": 0, "end_segment": 1}]}
    llm.responses["procedure"] = [evil]
    llm.responses["tag_terms"] = [{"steps": [{"step_index": 0, "terms": [{"word": "<i>型枠</i>", "category": "作業"}]}]}]
    upload_through_screen(page, base, sample_video, title="<script>window.hacked=1</script>")
    page.click("text=動画を見る")
    page.wait_for_selector(".step-card")
    assert page.evaluate("window.hacked") is None
    assert "<b>太字</b>" in page.inner_text(".step-card")


def add_done_video(skill, ctx, title, tag_names):
    video_id = skill.create_skill_video(ctx.conn, title, "", "a.mp4")
    skill.update_skill_video(ctx.conn, video_id, status="done", duration=90)
    tags = {t["name"]: t for t in skill.list_skill_tags(ctx.conn)}
    for name in tag_names:
        ctx.conn.execute("INSERT INTO skill_annotations (video_id, layer, label, tag_id) VALUES (?,?,?,?)",
                         (video_id, "video_tag", name, tags[name]["tag_id"]))
    ctx.conn.commit()
    return video_id


def video_titles(page):
    return sorted(page.locator("#video-list .video-card .title").all_inner_texts())


def test_home_walks_work_type_hierarchy_to_list(skill, server, page, sample_video):
    base, ctx, _ = server
    upload_through_screen(page, base, sample_video)
    add_done_video(skill, ctx, "鉄筋の結束", ["鉄筋結束", "ハッカー"])
    add_done_video(skill, ctx, "ボード張り", ["ボード張り"])

    page.goto(f"{base}/skill#/")
    page.wait_for_selector(".work-grid a")
    assert page.locator(".work-grid a").all_inner_texts() == [
        "躯体工事\nさらに選ぶ ▶", "仕上工事\nさらに選ぶ ▶", "仮設工事\nさらに選ぶ ▶"]
    assert no_horizontal_scroll(page)
    screenshot(page, "04_home")

    page.click(".work-grid a >> text=躯体工事")
    page.wait_for_selector("text=型枠工事")
    assert "コンクリート打設\n動画を見る" in page.locator(".work-grid a").all_inner_texts()
    page.click(".work-grid a >> text=型枠工事")
    page.wait_for_selector("text=型枠組立")
    assert page.inner_text(".breadcrumb") == "ホーム ＞ 躯体工事 ＞ 型枠工事"

    page.click("text=「型枠工事」の動画をすべて見る")
    page.wait_for_selector("#video-list")
    assert video_titles(page) == ["型枠の建て込み"]

    page.goto(f"{base}/skill#/?work=1")  # 躯体工事（下位の分類の動画もすべて）
    page.click("text=「躯体工事」の動画をすべて見る")
    page.wait_for_selector("#video-list")
    assert video_titles(page) == ["型枠の建て込み", "鉄筋の結束"]

    page.click("#video-list .video-card >> text=型枠の建て込み")
    page.wait_for_selector("#player")
    assert page.errors == []


def test_list_tag_buttons_filter_with_and(skill, server, page):
    base, ctx, _ = server
    add_done_video(skill, ctx, "型枠の建て込み", ["型枠組立", "インパクトドライバー", "コンパネ"])
    add_done_video(skill, ctx, "型枠のばらし", ["型枠解体", "インパクトドライバー"])
    add_done_video(skill, ctx, "鉄筋の結束", ["鉄筋結束", "ハッカー"])

    page.goto(f"{base}/skill#/list")
    page.wait_for_selector("#video-list")
    assert len(video_titles(page)) == 3
    assert no_horizontal_scroll(page)

    page.click(".tag-filter >> text=インパクトドライバー")
    page.wait_for_function("location.hash.includes('tags=')")
    page.wait_for_selector("text=（2件）")
    assert video_titles(page) == ["型枠のばらし", "型枠の建て込み"]
    page.click(".tag-filter >> text=コンパネ")
    page.wait_for_selector("text=（1件）")
    assert video_titles(page) == ["型枠の建て込み"]
    assert page.locator(".tag-filter.selected").count() == 2
    screenshot(page, "05_list_filtered")

    page.click(".tag-filter.selected >> text=インパクトドライバー")  # 選択を外す
    page.wait_for_function("document.querySelectorAll('.tag-filter.selected').length === 1")
    assert page.locator(".tag-filter.selected").all_inner_texts() == ["コンパネ ✕"]
    page.click("text=絞り込みを解除")
    page.wait_for_selector("text=（3件）")
    assert page.errors == []


def test_tag_management_screen(skill, server, page, sample_video):
    base, ctx, _ = server
    upload_through_screen(page, base, sample_video)  # 「Pコン」が新タグ候補になる
    page.set_viewport_size({"width": 1280, "height": 900})
    page.on("dialog", lambda d: d.accept())

    page.goto(f"{base}/skill#/tags")
    page.wait_for_selector("text=新タグ候補")
    assert "Pコン" in page.inner_text("table >> nth=0")

    # 候補を新しいタグとして採用 → 全動画のタグを付け直す
    page.click("text=新しいタグにする")
    page.select_option("select[id^=adopt-category-]", "資材")
    page.click("text=採用する")
    page.wait_for_selector("text=採用しました")
    page.click("button:has-text('全動画のタグを付け直す')")
    page.wait_for_selector("text=1本の動画のタグを付け直しました")
    video = TestClient_get(base, "/skill/api/videos/1")
    assert "Pコン" in [t["name"] for t in video["video_tags"]]

    # 追加（別名つき）
    page.fill("#tag-name", "バール")
    page.select_option("#tag-category", "道具")
    page.fill("#tag-aliases", "かじや|釘抜き")
    page.click("#tag-save")
    page.wait_for_selector("text=タグを追加しました")
    assert "かじや、釘抜き" in page.inner_text("text=バール >> xpath=ancestor::tr")

    # 編集（作業の階層の親を付け替え）
    tags = TestClient_get(base, "/skill/api/tags")["tags"]
    pid = next(t["tag_id"] for t in tags if t["name"] == "バール")
    page.click(f"#tag-row-{pid} >> text=編集")
    page.fill("#tag-name", "バール（釘抜き）")
    page.click("#tag-save")
    page.wait_for_selector("text=タグを保存しました")
    assert page.locator(f"#tag-row-{pid}").inner_text().startswith("バール（釘抜き）")
    screenshot(page, "06_tags")

    # 削除（下位のタグがあるものは消せない）
    page.click(f"#tag-row-{pid} >> text=削除")
    page.wait_for_selector("text=タグを削除しました")
    assert page.locator(f"#tag-row-{pid}").count() == 0
    assert page.errors == []


def TestClient_get(base, path):
    import json
    import urllib.request
    with urllib.request.urlopen(base + path) as res:
        return json.loads(res.read())


def test_recent_uploads_show_status_and_failed_video_can_be_opened(server, page, sample_video):
    base, ctx, llm = server
    llm.responses["procedure"] = ["だめ", "だめ"]
    page.goto(f"{base}/skill#/upload")
    page.set_input_files("#file-input", str(sample_video))
    page.fill("#title", "失敗する動画")
    page.click("#upload-button")
    page.wait_for_selector("text=失敗したところから再実行", timeout=30000)
    # 処理が終わると、同じ画面の「最近の投稿」も更新される
    page.wait_for_selector("#recent-uploads .status-badge.error")
    assert page.inner_text("#recent-uploads .status-badge") == "失敗"
    page.click("#recent-uploads .video-card")
    page.wait_for_selector("text=失敗したところから再実行")
    assert page.errors == []


def test_search_from_home_and_jump_to_scene(skill, server, page, sample_video):
    base, ctx, _ = server
    upload_through_screen(page, base, sample_video)
    page.goto(f"{base}/skill#/")
    page.wait_for_selector("#search-input")
    page.fill("#search-input", "セパ")
    page.press("#search-input", "Enter")
    page.wait_for_selector("#search-results .search-result")
    assert "型枠の建て込み" in page.inner_text("#search-results")
    assert page.locator("#search-results mark").count() > 0
    assert no_horizontal_scroll(page)
    screenshot(page, "07_search")

    # 字幕でヒットした場面（2秒〜）をタップすると、その時間から再生する
    page.click("#search-results a.hit:has-text('字幕')")
    page.wait_for_selector("#player")
    page.wait_for_function("Math.abs(document.getElementById('player').currentTime - 2.0) < 0.05")
    assert page.errors == []


def test_search_with_no_results_and_header_link(server, page):
    base, _, _ = server
    page.goto(f"{base}/skill#/")
    page.click("header a[title='検索']")
    page.wait_for_selector("#search-input")
    page.fill("#search-input", "クレーン")
    page.click(".search-form button")
    page.wait_for_selector("text=見つかりませんでした")
    assert page.errors == []
