"""
④ 閲覧（作業の種類・タグでの絞り込み）と、タグ一覧の管理（追加・編集・削除・別名・新タグ候補の採用）のテスト。
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conftest import SEED_CSV, FakeLLM, FakeWhisperModel


@pytest.fixture
def api(skill, conn, tmp_path):
    app = FastAPI()
    llm = FakeLLM()
    ctx = skill.register_skill_transfer(app, conn=conn, whisper_model=FakeWhisperModel(), llm_fn=llm,
                                        media_dir=str(tmp_path / "media"), seed_csv=str(SEED_CSV),
                                        run_in_background=False)
    return TestClient(app), ctx, llm


def tag_id(client, name):
    return next(t["tag_id"] for t in client.get("/skill/api/tags").json()["tags"] if t["name"] == name)


def add_video(skill, ctx, title, tag_names, status="done"):
    """動画と動画タグを直接作る（絞り込みのテスト用）。"""
    video_id = skill.create_skill_video(ctx.conn, title, "", "a.mp4")
    skill.update_skill_video(ctx.conn, video_id, status=status)
    tags = {t["name"]: t for t in skill.list_skill_tags(ctx.conn)}
    for name in tag_names:
        ctx.conn.execute(
            "INSERT INTO skill_annotations (video_id, layer, label, tag_id) VALUES (?,?,?,?)",
            (video_id, "video_tag", name, tags[name]["tag_id"]))
    ctx.conn.commit()
    return video_id


@pytest.fixture
def library(skill, api):
    client, ctx, _ = api
    ids = {
        "型枠": add_video(skill, ctx, "型枠の建て込み", ["型枠組立", "インパクトドライバー", "コンパネ"]),
        "解体": add_video(skill, ctx, "型枠のばらし", ["型枠解体", "インパクトドライバー"]),
        "結束": add_video(skill, ctx, "鉄筋の結束", ["鉄筋結束", "ハッカー"]),
        "ボード": add_video(skill, ctx, "ボード張り", ["ボード張り", "インパクトドライバー", "石膏ボード"]),
        "処理中": add_video(skill, ctx, "処理中の動画", ["型枠組立"], status="transcribe"),
    }
    return client, ctx, ids


def titles(res):
    return sorted(v["title"] for v in res.json()["videos"])


# --- 作業の種類（下位の分類を含む） ---
def test_work_type_includes_lower_categories(library):
    client, _, _ = library
    assert titles(client.get(f"/skill/api/videos?work_tag={tag_id(client, '型枠工事')}")) == ["型枠のばらし", "型枠の建て込み"]
    assert titles(client.get(f"/skill/api/videos?work_tag={tag_id(client, '躯体工事')}")) == [
        "型枠のばらし", "型枠の建て込み", "鉄筋の結束"]
    assert titles(client.get(f"/skill/api/videos?work_tag={tag_id(client, '型枠解体')}")) == ["型枠のばらし"]
    assert titles(client.get(f"/skill/api/videos?work_tag={tag_id(client, '仮設工事')}")) == []


def test_all_videos_without_filters_excludes_unfinished(library):
    client, _, _ = library
    assert len(client.get("/skill/api/videos").json()["videos"]) == 4
    all_videos = client.get("/skill/api/videos?include_unfinished=true").json()["videos"]
    assert {v["status"] for v in all_videos} == {"done", "transcribe"}


# --- タグのAND検索 ---
def test_tags_are_and_search(library):
    client, _, _ = library
    impact = tag_id(client, "インパクトドライバー")
    panel = tag_id(client, "コンパネ")
    assert titles(client.get(f"/skill/api/videos?tags={impact}")) == ["ボード張り", "型枠のばらし", "型枠の建て込み"]
    assert titles(client.get(f"/skill/api/videos?tags={impact},{panel}")) == ["型枠の建て込み"]
    assert titles(client.get(f"/skill/api/videos?tags={panel},{tag_id(client, 'ハッカー')}")) == []


def test_work_type_and_tags_together(library):
    client, _, _ = library
    res = client.get(f"/skill/api/videos?work_tag={tag_id(client, '型枠工事')}&tags={tag_id(client, 'インパクトドライバー')}")
    assert titles(res) == ["型枠のばらし", "型枠の建て込み"]


def test_facets_are_tags_of_listed_videos_with_counts(library):
    client, _, _ = library
    res = client.get(f"/skill/api/videos?work_tag={tag_id(client, '型枠工事')}").json()
    facets = {f["name"]: f["count"] for f in res["facets"]}
    assert facets == {"型枠組立": 1, "型枠解体": 1, "インパクトドライバー": 2, "コンパネ": 1}
    assert [f["category"] for f in res["facets"]][:2] == ["作業", "作業"]  # 分類順に並ぶ


def test_bad_tags_parameter_is_400(api):
    client, _, _ = api
    assert client.get("/skill/api/videos?tags=abc").status_code == 400


# --- タグ一覧の管理 ---
def test_tag_list_has_video_counts(library):
    client, _, _ = library
    tags = {t["name"]: t for t in client.get("/skill/api/tags").json()["tags"]}
    assert tags["インパクトドライバー"]["video_count"] == 3
    assert tags["型枠組立"]["video_count"] == 1  # 処理中の動画は数えない
    assert tags["ハッカー"]["aliases"] == ["結束ハッカー"]


def test_create_update_delete_tag(api):
    client, _, _ = api
    parent = tag_id(client, "型枠工事")
    res = client.post("/skill/api/tags", json={"name": "型枠の補強", "category": "作業", "parent_id": parent,
                                               "aliases": ["補強", " 補強 ", "", "型枠の補強"]})
    assert res.status_code == 200
    new_id = res.json()["tag_id"]
    tag = next(t for t in client.get("/skill/api/tags").json()["tags"] if t["tag_id"] == new_id)
    assert tag["aliases"] == ["補強"] and tag["parent_id"] == parent

    assert client.put(f"/skill/api/tags/{new_id}", json={"name": "型枠補強", "category": "作業",
                                                          "parent_id": parent, "aliases": ["補強", "バタ角"]}).status_code == 200
    tag = next(t for t in client.get("/skill/api/tags").json()["tags"] if t["tag_id"] == new_id)
    assert tag["name"] == "型枠補強" and tag["aliases"] == ["補強", "バタ角"]

    assert client.delete(f"/skill/api/tags/{new_id}").status_code == 200
    assert all(t["tag_id"] != new_id for t in client.get("/skill/api/tags").json()["tags"])


@pytest.mark.parametrize("body,message", [
    ({"name": " ", "category": "道具"}, "名前"),
    ({"name": "新しい道具", "category": "工具"}, "分類"),
    ({"name": "ハッカー", "category": "道具"}, "既にあります"),
    ({"name": "新しい道具", "category": "道具", "parent_id": 9999}, "親のタグ"),
])
def test_invalid_tag_is_rejected(api, body, message):
    client, _, _ = api
    res = client.post("/skill/api/tags", json=body)
    assert res.status_code == 400 and message in res.json()["detail"]


def test_parent_must_have_same_category_and_no_cycles(api):
    client, _, _ = api
    res = client.post("/skill/api/tags", json={"name": "新しい道具", "category": "道具",
                                               "parent_id": tag_id(client, "型枠工事")})
    assert res.status_code == 400 and "同じ分類" in res.json()["detail"]
    body = {"name": "躯体工事", "category": "作業", "parent_id": tag_id(client, "型枠組立")}
    res = client.put(f"/skill/api/tags/{tag_id(client, '躯体工事')}", json=body)
    assert res.status_code == 400 and "下位" in res.json()["detail"]


def test_tag_with_children_cannot_be_deleted(api):
    client, _, _ = api
    res = client.delete(f"/skill/api/tags/{tag_id(client, '型枠工事')}")
    assert res.status_code == 409 and "下位のタグ" in res.json()["detail"]


def test_deleting_tag_removes_it_from_videos(library):
    client, ctx, ids = library
    assert client.delete(f"/skill/api/tags/{tag_id(client, 'ハッカー')}").status_code == 200
    names = [t["name"] for t in client.get("/skill/api/videos").json()["videos"][0]["video_tags"]]
    assert "ハッカー" not in names
    assert ctx.conn.execute("SELECT COUNT(*) FROM skill_annotations WHERE label='ハッカー'").fetchone()[0] == 0


def test_renaming_tag_updates_video_tag_labels(library):
    client, ctx, _ = library
    tid = tag_id(client, "コンパネ")
    client.put(f"/skill/api/tags/{tid}", json={"name": "型枠用合板", "category": "資材", "aliases": ["コンパネ"]})
    assert ctx.conn.execute("SELECT DISTINCT label FROM skill_annotations WHERE tag_id=?", (tid,)).fetchall() == [("型枠用合板",)]


# --- 新タグ候補の採用・却下と、付け直し ---
def upload(client, video):
    with open(video, "rb") as f:
        return client.post("/skill/api/videos", data={"title": "型枠の建て込み"},
                           files={"file": ("a.mp4", f)}).json()["video_id"]


def test_candidate_adopted_as_new_tag_then_rematch_tags_videos_without_llm(api, sample_video):
    client, ctx, llm = api
    video_id = upload(client, sample_video)
    [candidate] = client.get("/skill/api/tag_candidates").json()["candidates"]
    assert candidate["word"] == "Pコン" and candidate["video_title"] == "型枠の建て込み"

    res = client.post(f"/skill/api/tag_candidates/{candidate['candidate_id']}/adopt",
                      json={"mode": "new", "name": "Pコン", "category": "資材"})
    assert res.status_code == 200
    assert client.get("/skill/api/tag_candidates").json()["candidates"] == []
    assert "Pコン" not in [t["name"] for t in client.get(f"/skill/api/videos/{video_id}").json()["video_tags"]]

    n_prompts = len(llm.prompts)
    assert client.post("/skill/api/rematch").json() == {"status": "done", "videos": 1}
    assert len(llm.prompts) == n_prompts  # LLMは呼ばない
    video = client.get(f"/skill/api/videos/{video_id}").json()
    assert "Pコン" in [t["name"] for t in video["video_tags"]]
    assert [t["start"] for t in video["scene_tags"] if t["name"] == "Pコン"] == [2.0]


def test_candidate_adopted_as_alias_of_existing_tag(api, sample_video):
    client, _, _ = api
    video_id = upload(client, sample_video)
    [candidate] = client.get("/skill/api/tag_candidates").json()["candidates"]
    sep = tag_id(client, "セパレーター")
    assert client.post(f"/skill/api/tag_candidates/{candidate['candidate_id']}/adopt",
                       json={"mode": "alias", "tag_id": sep}).status_code == 200
    tag = next(t for t in client.get("/skill/api/tags").json()["tags"] if t["tag_id"] == sep)
    assert tag["aliases"] == ["セパ", "Pコン"]
    client.post("/skill/api/rematch")
    assert client.get(f"/skill/api/videos/{video_id}").json()["video_tags"]


def test_adopted_with_different_name_keeps_word_as_alias(api, sample_video):
    client, _, _ = api
    upload(client, sample_video)
    [candidate] = client.get("/skill/api/tag_candidates").json()["candidates"]
    res = client.post(f"/skill/api/tag_candidates/{candidate['candidate_id']}/adopt",
                      json={"mode": "new", "name": "プラスチックコーン", "category": "資材"})
    tag = next(t for t in client.get("/skill/api/tags").json()["tags"] if t["tag_id"] == res.json()["tag_id"])
    assert tag["aliases"] == ["Pコン"]


def test_reject_candidate_and_double_processing(api, sample_video):
    client, _, _ = api
    upload(client, sample_video)
    [candidate] = client.get("/skill/api/tag_candidates").json()["candidates"]
    cid = candidate["candidate_id"]
    assert client.post(f"/skill/api/tag_candidates/{cid}/reject").status_code == 200
    assert client.get("/skill/api/tag_candidates?status=rejected").json()["candidates"][0]["word"] == "Pコン"
    assert client.post(f"/skill/api/tag_candidates/{cid}/reject").status_code == 404
    res = client.post(f"/skill/api/tag_candidates/{cid}/adopt", json={"mode": "new"})
    assert res.status_code == 400 and "処理済み" in res.json()["detail"]
    assert client.post("/skill/api/tag_candidates/999/adopt", json={"mode": "new"}).status_code == 404
