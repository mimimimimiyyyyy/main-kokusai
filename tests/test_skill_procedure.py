"""
② 手順書作成のテスト。LLMはモック（FakeLLM）に差し替える。
"""
import json
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conftest import DEFAULT_PROCEDURE, SEED_CSV, FakeLLM, FakeWhisperModel

SEGMENTS = [
    {"segment_id": 11, "start": 0.0, "end": 2.0, "text": "まず型枠を立てます。"},
    {"segment_id": 12, "start": 2.0, "end": 3.5, "text": "倒れないように仮止めします。"},
    {"segment_id": 13, "start": 4.0, "end": 6.0, "text": "次はセパを入れます。"},
]


def step(title="手順", start=0, end=0, **extra):
    return {"title": title, "description": "説明", "start_segment": start, "end_segment": end, **extra}


# --- LLMの出力の検証 ---
def test_times_come_from_segments_not_from_llm(skill):
    llm = FakeLLM(procedure=[{"steps": [step("型枠を立てる", 0, 1, tips=["下から締める"]), step("セパ", 2, 2)]}])
    steps, raw = skill.generate_procedure(llm, "型枠", SEGMENTS)
    assert [(s["start"], s["end"]) for s in steps] == [(0.0, 3.5), (4.0, 6.0)]
    assert steps[0]["segment_ids"] == [11, 12]
    assert steps[0]["tips"] == ["下から締める"]
    assert steps[1]["tools"] == [] and steps[1]["cautions"] == []
    assert json.loads(raw)["steps"][0]["title"] == "型枠を立てる"


def test_prompt_contains_numbered_segments_and_separator_words(skill):
    llm = FakeLLM()
    skill.generate_procedure(llm, "型枠の建て込み", SEGMENTS[:2])
    prompt = llm.prompts[0]
    assert "[0] (0.0〜2.0秒) まず型枠を立てます。" in prompt
    assert "「まず」" in prompt and "「次は」" in prompt
    assert "型枠の建て込み" in prompt


@pytest.mark.parametrize("bad", [
    "これはJSONではありません",
    {"steps": [step(start=0, end=5)]},                      # 範囲外
    {"steps": [step(start=2, end=1)]},                      # 開始 > 終了
    {"steps": [step(start=0, end=1), step(start=1, end=2)]},  # 重なり
    {"steps": [step(start=2, end=2), step(start=0, end=0)]},  # 順番が逆
    {"steps": [{"title": "見出しだけ", "start_segment": 0, "end_segment": 0}]},  # 説明が無い
    {"steps": [step(title="  ", start=0, end=0)]},          # 空の見出し
    {"steps": [step(start=0, end=0, tools="インパクト")]},  # 配列でない
])
def test_invalid_output_is_retried_once_then_succeeds(skill, bad):
    llm = FakeLLM(procedure=[bad, {"steps": [step("型枠", 0, 2)]}])
    steps, _ = skill.generate_procedure(llm, "型枠", SEGMENTS)
    assert len(llm.prompts) == 2
    assert "前回の出力は次の理由で不正でした" in llm.prompts[1]
    assert steps[0]["title"] == "型枠"


def test_two_invalid_outputs_raise(skill):
    llm = FakeLLM(procedure=["だめ", {"steps": [step(start=0, end=9)]}])
    with pytest.raises(ValueError, match="2回とも"):
        skill.generate_procedure(llm, "型枠", SEGMENTS)
    assert len(llm.prompts) == 2


def test_no_steps_from_llm_becomes_one_step_for_whole_video(skill):
    """説明が短いなどで LLM が手順に分けられなかったときは、止めずに動画全体を1つの手順にする。"""
    llm = FakeLLM(procedure=[{"steps": []}])
    steps, _ = skill.generate_procedure(llm, "型枠の建て込み", SEGMENTS)
    assert len(llm.prompts) == 1
    assert [(s["title"], s["start"], s["end"]) for s in steps] == [("型枠の建て込み", 0.0, 6.0)]
    assert steps[0]["segment_ids"] == [11, 12, 13]
    assert "手順を区切れなかった" in steps[0]["description"]


def test_prompt_asks_for_at_least_one_step(skill):
    llm = FakeLLM()
    skill.generate_procedure(llm, "型枠", SEGMENTS)
    assert "手順は必ず1つ以上返す" in llm.prompts[0]


def test_json_inside_code_fence_is_accepted(skill):
    fenced = "```json\n" + json.dumps({"steps": [step("型枠", 0, 0)]}, ensure_ascii=False) + "\n```"
    steps, _ = skill.generate_procedure(FakeLLM(procedure=[fenced]), "型枠", SEGMENTS)
    assert steps[0]["title"] == "型枠"


# --- 保存（区間アノテーション） ---
def test_steps_are_saved_as_interval_annotations_and_replaced(skill, conn):
    video_id = skill.create_skill_video(conn, "型枠", "", "a.mp4")
    steps, _ = skill.generate_procedure(FakeLLM(), "型枠", SEGMENTS)
    skill.save_steps(conn, video_id, steps, "v1")
    skill.save_steps(conn, video_id, steps[:1], "v2")
    rows = conn.execute("SELECT layer, start, end, label, method_version FROM skill_annotations").fetchall()
    assert rows == [("step", 0.0, 2.0, "型枠を立てる", "v2")]
    saved = skill.get_steps(conn, video_id)[0]
    assert saved["tools"] == ["インパクトドライバー"] and saved["index"] == 0


# --- 手順書（HTML・PDF） ---
def test_html_has_steps_times_and_escapes_text(skill):
    video = {"title": "型枠<script>", "explainer": "山田", "duration": 125, "created": "2026-10-08 10:00:00"}
    steps = [{"start": 61, "end": 95, "title": "型枠を立てる", "description": "A&B", "tools": ["インパクト"],
              "materials": [], "cautions": ["倒れに注意"], "tips": ["下から締める"], "photo": None}]
    html = skill.render_procedure_html(video, steps)
    assert "型枠&lt;script&gt;" in html and "<script>" not in html
    assert "手順1　型枠を立てる" in html
    assert "01:01〜01:35" in html
    assert "倒れに注意" in html and "下から締める" in html
    assert "A&amp;B" in html


def test_pdf_is_written_with_photo(skill, sample_video, tmp_path):
    steps = skill.add_step_photos(str(sample_video), [{"start": 0, "end": 2}], str(tmp_path / "steps"))
    assert os.path.exists(steps[0]["photo"])
    video = {"title": "型枠の建て込み", "explainer": "", "duration": 3, "created": "2026-10-08"}
    full = [{**steps[0], "title": "型枠を立てる", "description": "説明", "tools": [], "materials": [],
             "cautions": [], "tips": []}]
    pdf = skill.write_procedure_pdf(skill.render_procedure_html(video, full), str(tmp_path / "p.pdf"))
    data = open(pdf, "rb").read()
    assert data.startswith(b"%PDF")
    assert b"/Image" in data  # 写真が埋め込まれている


def test_photo_failure_does_not_stop_procedure(skill, tmp_path):
    steps = skill.add_step_photos(str(tmp_path / "missing.mp4"), [{"start": 0, "end": 1}], str(tmp_path / "s"))
    assert steps[0]["photo"] is None


# --- LLMの差し替え（1か所にまとめた呼び出し） ---
def test_make_llm_fn_openai_uses_json_mode_and_existing_client(skill):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            msg = type("M", (), {"content": '{"steps": []}'})
            return type("R", (), {"choices": [type("C", (), {"message": msg})]})

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})})
    llm = skill.make_llm_fn(provider="openai", openai_client=client)
    assert llm("プロンプト") == '{"steps": []}'
    assert calls[0]["model"] == "gpt-4o"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert llm.method_version == "openai:gpt-4o"


def test_make_llm_fn_anthropic_default_model(skill, monkeypatch):
    import anthropic
    calls = []

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            block = type("B", (), {"type": "text", "text": '{"steps": []}'})
            return type("R", (), {"stop_reason": "end_turn", "content": [block]})

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.beta = type("Beta", (), {"messages": Messages()})

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    llm = skill.make_llm_fn(provider="anthropic")
    assert llm("プロンプト") == '{"steps": []}'
    assert calls[0]["model"] == "claude-opus-5-5"
    assert calls[0]["messages"] == [{"role": "user", "content": "プロンプト"}]
    assert llm.method_version == "anthropic:claude-opus-5-5"


def test_make_llm_fn_rejects_unknown_provider(skill):
    with pytest.raises(ValueError):
        skill.make_llm_fn(provider="gemini")


def test_provider_and_model_from_env():
    from conftest import load_skill_cell
    cell = load_skill_cell(env={"SKILL_LLM_PROVIDER": "anthropic", "SKILL_LLM_MODEL": "claude-sonnet-5-5"})
    assert cell.make_llm_fn().method_version == "anthropic:claude-sonnet-5-5"


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
        res = client.post("/skill/api/videos", data={"title": "型枠の建て込み"}, files={"file": ("a.mp4", f)})
    return res.json()["video_id"]


def test_upload_creates_steps_photos_and_pdf(api, sample_video):
    client, ctx, llm = api
    video_id = upload(client, sample_video)
    video = client.get(f"/skill/api/videos/{video_id}").json()
    assert video["status"] == "done" and video["has_pdf"]
    assert [s["title"] for s in video["steps"]] == ["型枠を立てる", "セパを入れる"]
    assert video["steps"][0]["start"] == 0.0 and video["steps"][0]["end"] == 2.0
    assert video["steps"][0]["photo_url"] == f"/skill/api/videos/{video_id}/steps/1.jpg"
    assert "photo" not in video["steps"][0]

    assert client.get(video["steps"][0]["photo_url"]).content[:2] == b"\xff\xd8"
    pdf = client.get(f"/skill/api/videos/{video_id}/procedure.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert "attachment" in pdf.headers["content-disposition"]

    # LLMの生の出力は版つきで残す
    kind, version, result = ctx.conn.execute(
        "SELECT kind, method_version, result FROM skill_llm_results WHERE video_id=?", (video_id,)).fetchone()
    assert kind == "procedure" and version.startswith("fake:llm@")
    assert json.loads(json.loads(result)["raw"]) == DEFAULT_PROCEDURE


def test_procedure_failure_is_recorded_and_retry_starts_from_procedure(api, sample_video):
    client, ctx, llm = api
    llm.responses["procedure"] = ["だめ", "まだだめ", DEFAULT_PROCEDURE]
    video_id = upload(client, sample_video)
    video = client.get(f"/skill/api/videos/{video_id}").json()
    assert video["status"] == "error" and video["failed_step"] == "procedure"
    assert "2回とも" in video["error"]
    assert len(video["segments"]) == 2  # 文字起こしは残っている
    assert client.get(f"/skill/api/videos/{video_id}/procedure.pdf").status_code == 404

    whisper_calls = len(ctx.whisper_model.calls)
    assert client.post(f"/skill/api/videos/{video_id}/retry").status_code == 200
    video = client.get(f"/skill/api/videos/{video_id}").json()
    assert video["status"] == "done" and len(video["steps"]) == 2
    assert len(ctx.whisper_model.calls) == whisper_calls  # 文字起こしはやり直さない


def test_missing_step_photo_is_404(api, sample_video):
    client, _, _ = api
    video_id = upload(client, sample_video)
    assert client.get(f"/skill/api/videos/{video_id}/steps/9.jpg").status_code == 404
