"""
検索機能のテスト（タイトル・タグ・手順書・字幕から探し、ヒットした場面の時間を返す）。
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conftest import FakeLLM


@pytest.fixture
def api(skill, conn, tmp_path):
    app = FastAPI()
    ctx = skill.register_skill_transfer(app, conn=conn, llm_fn=FakeLLM(), media_dir=str(tmp_path),
                                        run_in_background=False)
    return TestClient(app), ctx


def add_video(skill, ctx, title, segments, steps, tag_names=(), status="done"):
    """文字起こし・手順・動画タグ付きの動画を直接作る。"""
    video_id = skill.create_skill_video(ctx.conn, title, "山田", "a.mp4")
    skill.update_skill_video(ctx.conn, video_id, status=status, duration=60)
    skill.save_segments(ctx.conn, video_id, [{"start": s, "end": s + 3, "text": t} for s, t in segments])
    skill.save_steps(ctx.conn, video_id, [
        {"start": st["start"], "end": st["start"] + 10, "title": st["title"], "description": st.get("description", ""),
         "tools": st.get("tools", []), "materials": st.get("materials", []), "cautions": st.get("cautions", []),
         "tips": st.get("tips", []), "segment_ids": []} for st in steps], "v1")
    tags = {t["name"]: t for t in skill.list_skill_tags(ctx.conn)}
    for name in tag_names:
        ctx.conn.execute("INSERT INTO skill_annotations (video_id, layer, label, tag_id) VALUES (?,?,?,?)",
                         (video_id, "video_tag", name, tags[name]["tag_id"]))
    ctx.conn.commit()
    return video_id


@pytest.fixture
def library(skill, api):
    client, ctx = api
    katawaku = add_video(skill, ctx, "型枠の建て込み",
                         [(0, "まず墨を確認します。"), (12, "次はセパを通します。"), (30, "インパクトで締めます。")],
                         [{"start": 0, "title": "墨の確認"},
                          {"start": 12, "title": "セパを通す", "materials": ["セパ"]},
                          {"start": 30, "title": "締め付け", "tools": ["インパクト"],
                           "cautions": ["締めすぎるとセパが切れる"], "tips": ["下から順番に締める"]}],
                         ["型枠組立", "セパレーター", "インパクトドライバー"])
    board = add_video(skill, ctx, "ボード張り",
                      [(0, "石膏ボードを当てます。"), (20, "インパクトドライバーでビスを留めます。")],
                      [{"start": 0, "title": "ボードを当てる", "cautions": ["仮止めを忘れない"]},
                       {"start": 20, "title": "ビス留め", "tools": ["インパクトドライバー"]}],
                      ["ボード張り", "石膏ボード", "インパクトドライバー"])
    add_video(skill, ctx, "処理中の動画", [(0, "インパクトで締めます。")], [], status="transcribe")
    return client, {"katawaku": katawaku, "board": board}


def titles(res):
    return [v["title"] for v in res.json()["videos"]]


def test_search_finds_videos_by_transcript_steps_and_tags(library):
    client, ids = library
    res = client.get("/skill/api/search", params={"q": "仮止め"})
    assert titles(res) == ["ボード張り"]
    [hit] = res.json()["videos"][0]["hits"]
    assert hit == {"kind": "手順", "label": "手順1の注意点", "text": "仮止めを忘れない", "start": 0.0}


def test_search_returns_scene_times_in_order(library):
    client, ids = library
    video = client.get("/skill/api/search", params={"q": "セパ"}).json()["videos"][0]
    assert video["video_id"] == ids["katawaku"]
    starts = [h["start"] for h in video["hits"] if h["start"] is not None]
    assert starts == sorted(starts) and starts[0] == 12.0
    assert video["hits"][0]["kind"] == "タグ"  # タグ（セパレーター）が先、場面は時間順


def test_hits_are_limited_but_counted(skill, library):
    client, ids = library
    ctx = client.app.state.skill_transfer
    [video] = skill.search_skill_videos(ctx.conn, "セパ", max_hits=2)["videos"]
    assert len(video["hits"]) == 2 and video["hit_count"] == 5


def test_search_uses_tag_aliases(library):
    """「インパクト」でも、正式名「インパクトドライバー」としか書かれていない動画が見つかる。"""
    client, _ = library
    data = client.get("/skill/api/search", params={"q": "インパクト"}).json()
    assert sorted(v["title"] for v in data["videos"]) == ["ボード張り", "型枠の建て込み"]
    assert "インパクトドライバー" in data["words"]
    data = client.get("/skill/api/search", params={"q": "PB"}).json()  # 石膏ボードの別名
    assert [v["title"] for v in data["videos"]] == ["ボード張り"]


def test_multiple_words_are_and_search(library):
    client, _ = library
    assert titles(client.get("/skill/api/search", params={"q": "インパクト　セパ"})) == ["型枠の建て込み"]
    assert titles(client.get("/skill/api/search", params={"q": "セパ ビス"})) == []


def test_search_ignores_width_and_case(library):
    client, _ = library
    assert titles(client.get("/skill/api/search", params={"q": "ｾﾊﾟ"})) == ["型枠の建て込み"]


def test_results_are_ordered_by_number_of_hits(library):
    client, _ = library
    res = client.get("/skill/api/search", params={"q": "インパクトドライバー"}).json()["videos"]
    assert [v["hit_count"] for v in res] == sorted([v["hit_count"] for v in res], reverse=True)


def test_unfinished_videos_are_not_searched(library):
    client, _ = library
    assert "処理中の動画" not in titles(client.get("/skill/api/search", params={"q": "締め"}))


def test_empty_query_is_400_and_no_match_is_empty(library):
    client, _ = library
    assert client.get("/skill/api/search", params={"q": "  "}).status_code == 400
    assert client.get("/skill/api/search", params={"q": "クレーン"}).json()["videos"] == []
