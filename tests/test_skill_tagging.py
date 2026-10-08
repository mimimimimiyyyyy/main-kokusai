"""
③ タグ付けのテスト。LLMはモック（FakeLLM）に差し替える。タグ一覧は seed CSV を使う。
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conftest import SEED_CSV, FakeLLM, FakeWhisperModel


@pytest.fixture
def tags(skill, seeded_conn):
    return skill.list_skill_tags(seeded_conn)


def matched_name(skill, word, tags, threshold=85):
    m = skill.match_tag(word, tags, threshold)
    return (m["tag"]["name"], m["method"]) if m else None


# --- 照合（完全一致 → 別名一致 → 類似度） ---
def test_exact_name_match(skill, tags):
    assert matched_name(skill, "インパクトドライバー", tags) == ("インパクトドライバー", "name")


def test_alias_match(skill, tags):
    assert matched_name(skill, "セパ", tags) == ("セパレーター", "alias")
    assert matched_name(skill, "建て込み", tags) == ("型枠組立", "alias")
    assert matched_name(skill, "丸ノコ", tags) == ("電動丸のこ", "alias")


def test_width_and_case_differences_are_ignored(skill, tags):
    assert matched_name(skill, "ＬＧＳ", tags) == ("軽量鉄骨下地", "alias")
    assert matched_name(skill, "lgs", tags) == ("軽量鉄骨下地", "alias")
    assert matched_name(skill, "インパクト ドライバー", tags) == ("インパクトドライバー", "name")


def test_name_match_wins_over_alias_match(skill):
    tags = [
        {"tag_id": 1, "name": "レベル", "category": "道具", "aliases": []},
        {"tag_id": 2, "name": "レーザー墨出し器", "category": "道具", "aliases": ["レベル"]},
    ]
    assert matched_name(skill, "レベル", tags) == ("レベル", "name")


def test_trailing_long_vowel_and_dash_variants_are_the_same_word(skill, tags):
    assert matched_name(skill, "インパクトドライバ―", tags) == ("インパクトドライバー", "name")  # ―(U+2015)
    assert matched_name(skill, "セパレータ", tags) == ("セパレーター", "name")


def test_fuzzy_match_above_threshold(skill, tags):
    # 送り仮名の違いなど、表記が揺れた言葉
    assert matched_name(skill, "レーザー墨出器", tags) == ("レーザー墨出し器", "fuzzy")
    assert matched_name(skill, "コンクリ―ト打説", tags) == ("コンクリート打設", "fuzzy")  # 誤変換


def test_fuzzy_match_respects_threshold(skill, tags):
    assert matched_name(skill, "コンクリ―ト打説", tags, threshold=99) is None
    assert matched_name(skill, "コンクリート", tags) is None  # 「コンクリート打設」より広い意味の言葉は一致させない
    assert matched_name(skill, "墨", tags) is None            # 1文字の言葉も一致させない


def test_word_containing_tag_name_matches_most_specific_tag(skill, tags):
    # 話し言葉の「型枠の建て込み」には「型枠」（型枠工事の別名）と「建て込み」（型枠組立の別名）が入っている
    assert matched_name(skill, "型枠の建て込み", tags) == ("型枠組立", "partial")
    assert matched_name(skill, "インパクトで締める", tags) == ("インパクトドライバー", "partial")
    assert matched_name(skill, "型枠解体工", tags)[0] == "型枠解体"


def test_similarity_is_checked_before_containment(skill, tags):
    # 「石膏ボード」（資材）を含むが、文字列全体は作業の「石膏ボード張り」（ボード張りの別名）に近い
    assert matched_name(skill, "石膏ボードはり", tags) == ("ボード張り", "fuzzy")


def test_difficulty_tags_are_not_matched_from_words(skill, tags):
    assert skill.match_tag("上級", tags) is None


# --- 言葉の抽出の検証 ---
@pytest.mark.parametrize("bad", [
    {"steps": [{"step_index": 5, "terms": []}]},
    {"steps": [{"step_index": 0, "terms": []}, {"step_index": 0, "terms": []}]},
    {"steps": [{"step_index": 0, "terms": [{"word": "型枠", "category": "工法"}]}]},
    {"steps": [{"step_index": 0, "terms": [{"word": " ", "category": "作業"}]}]},
    {"items": []},
])
def test_invalid_terms_are_rejected(skill, bad):
    with pytest.raises(Exception):
        skill.validate_tag_terms(bad, n_steps=2)


def test_steps_without_terms_may_be_omitted(skill):
    result = skill.validate_tag_terms({"steps": [{"step_index": 1, "terms": []}]}, n_steps=2)
    assert result.steps[0].step_index == 1


# --- 場面タグ・動画タグ・新タグ候補 ---
def make_steps(skill, conn):
    video_id = skill.create_skill_video(conn, "型枠", "", "a.mp4")
    steps = [
        {"start": 0.0, "end": 10.0, "title": "型枠を立てる", "description": "d", "tools": ["インパクト"],
         "materials": [], "cautions": ["倒れ注意", "強風"], "tips": [], "segment_ids": []},
        {"start": 10.0, "end": 25.0, "title": "セパを入れる", "description": "d", "tools": [],
         "materials": [], "cautions": [], "tips": [], "segment_ids": []},
    ]
    skill.save_steps(conn, video_id, steps, "v1")
    return video_id, skill.get_steps(conn, video_id)


def test_scene_tags_follow_step_times_and_video_tags_are_their_union(skill, seeded_conn):
    video_id, steps = make_steps(skill, seeded_conn)
    terms = {
        0: [{"word": "型枠", "category": "作業"}, {"word": "建て込み", "category": "作業"},
            {"word": "インパクト", "category": "道具"}],
        1: [{"word": "セパ", "category": "資材"}, {"word": "インパクトドライバー", "category": "道具"}],
    }
    skill.assign_tags(seeded_conn, video_id, steps, terms, "v1", difficulty=False)

    scene = {(t["name"], t["start"], t["end"], t["step_id"]) for t in skill.get_scene_tags(seeded_conn, video_id)}
    assert scene == {
        ("型枠工事", 0.0, 10.0, steps[0]["step_id"]),
        ("型枠組立", 0.0, 10.0, steps[0]["step_id"]),
        ("インパクトドライバー", 0.0, 10.0, steps[0]["step_id"]),
        ("セパレーター", 10.0, 25.0, steps[1]["step_id"]),
        ("インパクトドライバー", 10.0, 25.0, steps[1]["step_id"]),
    }
    assert {t["name"] for t in skill.get_video_tags(seeded_conn, video_id)} == {
        "型枠工事", "型枠組立", "インパクトドライバー", "セパレーター"}


def test_same_tag_from_two_words_in_a_step_is_one_scene_tag(skill, seeded_conn):
    video_id, steps = make_steps(skill, seeded_conn)
    terms = {0: [{"word": "インパクト", "category": "道具"}, {"word": "インパクトドライバー", "category": "道具"}]}
    skill.assign_tags(seeded_conn, video_id, steps, terms, "v1", difficulty=False)
    assert [t["name"] for t in skill.get_scene_tags(seeded_conn, video_id)] == ["インパクトドライバー"]


def test_unknown_words_become_pending_candidates_not_tags(skill, seeded_conn):
    before = len(skill.list_skill_tags(seeded_conn))
    video_id, steps = make_steps(skill, seeded_conn)
    result = skill.assign_tags(seeded_conn, video_id, steps, {0: [{"word": "Pコン", "category": "資材"}],
                                                             1: [{"word": "ｐコン", "category": "資材"}]},
                               "v1", difficulty=False)
    assert result["new_candidates"] == 1  # 表記違いの同じ言葉は1件にまとめる
    rows = seeded_conn.execute("SELECT word, category, status, video_id FROM skill_tag_candidates").fetchall()
    assert rows == [("Pコン", "資材", "pending", video_id)]
    assert len(skill.list_skill_tags(seeded_conn)) == before  # 自動では登録しない
    assert skill.get_video_tags(seeded_conn, video_id) == []


def test_rejected_candidate_is_not_suggested_again(skill, seeded_conn):
    video_id, steps = make_steps(skill, seeded_conn)
    skill.assign_tags(seeded_conn, video_id, steps, {0: [{"word": "Pコン", "category": "資材"}]}, "v1")
    seeded_conn.execute("UPDATE skill_tag_candidates SET status='rejected'")
    result = skill.assign_tags(seeded_conn, video_id, steps, {0: [{"word": "Pコン", "category": "資材"}]}, "v2")
    assert result["new_candidates"] == 0


def test_retagging_replaces_previous_tags(skill, seeded_conn):
    video_id, steps = make_steps(skill, seeded_conn)
    skill.assign_tags(seeded_conn, video_id, steps, {0: [{"word": "型枠", "category": "作業"}]}, "v1")
    skill.assign_tags(seeded_conn, video_id, steps, {0: [{"word": "セパ", "category": "資材"}]}, "v2")
    assert [t["name"] for t in skill.get_video_tags(seeded_conn, video_id)] == ["セパレーター"]


# --- 難易度（設定でON/OFF） ---
@pytest.mark.parametrize("n_steps,n_cautions,expected", [(2, 0, "初級"), (2, 2, "初級"), (3, 2, "中級"),
                                                         (5, 3, "中級"), (5, 4, "上級")])
def test_judge_difficulty(skill, n_steps, n_cautions, expected):
    steps = [{"cautions": []} for _ in range(n_steps)]
    steps[0]["cautions"] = ["c"] * n_cautions
    assert skill.judge_difficulty(steps)[0] == expected


def test_difficulty_tag_only_when_enabled(skill, seeded_conn):
    video_id, steps = make_steps(skill, seeded_conn)
    skill.assign_tags(seeded_conn, video_id, steps, {}, "v1", difficulty=False)
    assert skill.get_video_tags(seeded_conn, video_id) == []
    skill.assign_tags(seeded_conn, video_id, steps, {}, "v2", difficulty=True)
    assert skill.get_video_tags(seeded_conn, video_id) == [
        {"tag_id": skill.list_skill_tags(seeded_conn, "難易度")[0]["tag_id"], "name": "初級", "category": "難易度"}]


# --- アップロードからの通し ---
@pytest.fixture
def api(skill, conn, tmp_path):
    app = FastAPI()
    llm = FakeLLM()
    ctx = skill.register_skill_transfer(app, conn=conn, whisper_model=FakeWhisperModel(), llm_fn=llm,
                                        media_dir=str(tmp_path / "media"), seed_csv=str(SEED_CSV),
                                        run_in_background=False)
    return TestClient(app), ctx, llm


def upload(client, video):
    with open(video, "rb") as f:
        return client.post("/skill/api/videos", data={"title": "型枠の建て込み"},
                           files={"file": ("a.mp4", f)}).json()["video_id"]


def test_upload_creates_scene_and_video_tags(api, sample_video):
    client, ctx, llm = api
    video = client.get(f"/skill/api/videos/{upload(client, sample_video)}").json()
    assert video["status"] == "done"
    # 1つ目の手順: LLMの「型枠」「建て込み」＋手順書の道具「インパクトドライバー」・資材「コンパネ」
    first = {t["name"] for t in video["scene_tags"] if t["step_id"] == video["steps"][0]["step_id"]}
    assert first == {"型枠工事", "型枠組立", "インパクトドライバー", "コンパネ"}
    second = [t for t in video["scene_tags"] if t["step_id"] == video["steps"][1]["step_id"]]
    assert {t["name"] for t in second} == {"セパレーター"}
    assert all((t["start"], t["end"]) == (2.0, 3.0) for t in second)
    assert {t["name"] for t in video["video_tags"]} == {
        "型枠工事", "型枠組立", "インパクトドライバー", "コンパネ", "セパレーター"}
    # 一覧に無い「Pコン」は新タグ候補に
    assert ctx.conn.execute("SELECT word FROM skill_tag_candidates").fetchall() == [("Pコン",)]
    # タグ抽出のプロンプトには手順ごとの文字起こしが入る
    prompt = llm.prompts_for("tag_terms")[0]
    assert "[0] 型枠を立てる\nまず型枠を立てます。" in prompt


def test_tagging_failure_can_be_retried_from_tagging(api, sample_video):
    client, ctx, llm = api
    llm.responses["tag_terms"] = ["だめ", "だめ", llm.responses["tag_terms"][0]]
    video_id = upload(client, sample_video)
    video = client.get(f"/skill/api/videos/{video_id}").json()
    assert video["status"] == "error" and video["failed_step"] == "tagging"
    assert video["has_pdf"]  # 手順書までは出来ている
    n_procedure = len(llm.prompts_for("procedure"))
    client.post(f"/skill/api/videos/{video_id}/retry")
    video = client.get(f"/skill/api/videos/{video_id}").json()
    assert video["status"] == "done" and video["video_tags"]
    assert len(llm.prompts_for("procedure")) == n_procedure  # 手順書はやり直さない
