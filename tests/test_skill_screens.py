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

from conftest import SEED_CSV, SKILL_HTML, FakeLLM, FakeWhisperModel

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
                                        html_path=str(SKILL_HTML), run_in_background=True)
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
    page.wait_for_selector("text=動画を見る", timeout=30000)


def test_page_is_served_and_missing_html_is_explained(skill, conn, tmp_path):
    app = FastAPI()
    skill.register_skill_transfer(app, conn=conn, llm_fn=FakeLLM(), media_dir=str(tmp_path),
                                  seed_csv=str(SEED_CSV), html_path=str(tmp_path / "none.html"))
    res = TestClient(app).get("/skill")
    assert res.status_code == 404 and "skill_transfer.html" in res.text

    app2 = FastAPI()
    skill.register_skill_transfer(app2, conn=conn, llm_fn=FakeLLM(), media_dir=str(tmp_path),
                                  seed_csv=str(SEED_CSV), html_path=str(SKILL_HTML))
    res = TestClient(app2).get("/skill")
    assert res.status_code == 200 and "技能伝承動画" in res.text


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
