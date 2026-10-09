"""
アップロード → ① 文字起こし・字幕 の流れを、APIから通しで確かめる。
音声認識はモック。処理は裏のスレッドではなくその場で実行する（run_in_background=False）。
"""
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conftest import SEED_CSV, FakeLLM, FakeWhisperModel


@pytest.fixture
def app_ctx(skill, conn, tmp_path):
    app = FastAPI()
    model = FakeWhisperModel()
    ctx = skill.register_skill_transfer(
        app, conn=conn, whisper_model=model, llm_fn=FakeLLM(), media_dir=str(tmp_path / "media"),
        seed_csv=str(SEED_CSV), run_in_background=False,
    )
    return app, ctx, model


def upload(client, video_path, title="型枠の建て込み", explainer="山田", filename="IMG_0001.MOV"):
    with open(video_path, "rb") as f:
        return client.post(
            "/skill/api/videos",
            data={"title": title, "explainer": explainer},
            files={"file": (filename, f, "video/quicktime")},
        )


def test_register_seeds_tags_when_empty(skill, app_ctx):
    _, ctx, _ = app_ctx
    assert len(skill.list_skill_tags(ctx.conn)) > 0


def test_upload_transcribes_and_serves_subtitles_and_media(app_ctx, sample_video):
    app, ctx, model = app_ctx
    client = TestClient(app)
    res = upload(client, sample_video)
    assert res.status_code == 200
    video_id = res.json()["video_id"]

    video = client.get(f"/skill/api/videos/{video_id}").json()
    assert video["status"] == "done"
    assert video["title"] == "型枠の建て込み"
    assert video["duration"] == pytest.approx(3.0, abs=0.2)
    assert [s["text"] for s in video["segments"]] == ["まず型枠を立てます。", "次はセパを入れます。"]
    assert {s["speaker"] for s in video["segments"]} == {"山田"}
    assert "media_path" not in video

    # 認識には騒音除去後のWAVを渡し、日本語指定とタグ一覧の用語ヒントを付ける
    call = model.calls[0]
    assert call["audio"].endswith("audio.wav") and os.path.exists(call["audio"])
    assert call["language"] == "ja"
    assert "インパクトドライバー" in call["initial_prompt"] and "セパレーター" in call["initial_prompt"]

    vtt = client.get(f"/skill/api/videos/{video_id}/subtitles.vtt")
    assert vtt.headers["content-type"].startswith("text/vtt")
    assert vtt.text.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.000" in vtt.text

    media = client.get(f"/skill/api/videos/{video_id}/media")
    assert media.status_code == 200 and media.headers["content-type"] == "video/mp4"
    partial = client.get(f"/skill/api/videos/{video_id}/media", headers={"Range": "bytes=0-99"})
    assert partial.status_code == 206 and len(partial.content) == 100

    thumb = client.get(f"/skill/api/videos/{video_id}/thumbnail.jpg")
    assert thumb.status_code == 200 and thumb.content[:2] == b"\xff\xd8"


def test_uploaded_file_name_is_not_used_as_path(app_ctx, sample_video, tmp_path):
    app, ctx, _ = app_ctx
    video_id = upload(TestClient(app), sample_video, filename="../../evil.MOV").json()["video_id"]
    folder = os.path.join(ctx.media_dir, "videos", str(video_id))
    assert "original.mov" in os.listdir(folder)
    assert not os.path.exists(tmp_path / "evil.MOV")


def test_upload_requires_title(app_ctx, sample_video):
    app, _, _ = app_ctx
    assert upload(TestClient(app), sample_video, title="  ").status_code == 400


def test_failure_keeps_step_and_error_and_can_be_retried(app_ctx, sample_video):
    app, ctx, model = app_ctx
    client = TestClient(app)
    model.error = RuntimeError("GPUのメモリが足りません")
    video_id = upload(client, sample_video).json()["video_id"]

    video = client.get(f"/skill/api/videos/{video_id}").json()
    assert video["status"] == "error"
    assert video["failed_step"] == "transcribe"
    assert "GPUのメモリが足りません" in video["error"]

    model.error = None
    res = client.post(f"/skill/api/videos/{video_id}/retry")
    assert res.status_code == 200
    video = client.get(f"/skill/api/videos/{video_id}").json()
    assert video["status"] == "done" and video["error"] is None
    assert len(video["segments"]) == 2
    # 失敗した段階からやり直す（動画の変換はやり直さない）ので、認識は合計2回だけ
    assert len(model.calls) == 2


def test_retry_is_rejected_unless_failed(app_ctx, sample_video):
    app, _, _ = app_ctx
    client = TestClient(app)
    video_id = upload(client, sample_video).json()["video_id"]
    assert client.post(f"/skill/api/videos/{video_id}/retry").status_code == 409


def test_video_without_audio_is_reported_as_error(app_ctx, silent_video):
    app, _, _ = app_ctx
    client = TestClient(app)
    video = client.get(f"/skill/api/videos/{upload(client, silent_video).json()['video_id']}").json()
    assert video["status"] == "error" and video["failed_step"] == "transcribe"


def test_nothing_recognized_is_reported_as_error(app_ctx, sample_video):
    app, _, model = app_ctx
    model.segments = []
    client = TestClient(app)
    video = client.get(f"/skill/api/videos/{upload(client, sample_video).json()['video_id']}").json()
    assert video["status"] == "error"
    assert "認識できませんでした" in video["error"]


def test_unknown_video_is_404(app_ctx):
    app, _, _ = app_ctx
    client = TestClient(app)
    assert client.get("/skill/api/videos/999").status_code == 404
    assert client.get("/skill/api/videos/999/subtitles.vtt").status_code == 404


def test_anonymize_uses_mask_function_only_when_enabled(skill, conn, tmp_path, sample_video):
    calls = []

    def mask_fn(turns):
        calls.append(turns)
        return [{"word": "セパ"}]

    app = FastAPI()
    ctx = skill.register_skill_transfer(app, conn=conn, whisper_model=FakeWhisperModel(), mask_fn=mask_fn,
                                        llm_fn=FakeLLM(),
                                        media_dir=str(tmp_path / "m"), seed_csv=str(SEED_CSV),
                                        run_in_background=False)
    client = TestClient(app)
    video_id = upload(client, sample_video).json()["video_id"]
    assert calls == []  # 既定はOFF

    ctx.anonymize = True
    video_id = upload(client, sample_video).json()["video_id"]
    texts = [s["text"] for s in client.get(f"/skill/api/videos/{video_id}").json()["segments"]]
    assert texts[1] == "次は[MASK]を入れます。"


def test_anonymize_default_uses_llm_to_find_words(skill, conn, tmp_path, sample_video):
    app = FastAPI()
    llm = FakeLLM(mask=[{"mask_list": [{"word": "セパ"}]}])
    ctx = skill.register_skill_transfer(app, conn=conn, whisper_model=FakeWhisperModel(), llm_fn=llm,
                                        media_dir=str(tmp_path / "m"), run_in_background=False)
    ctx.anonymize = True
    client = TestClient(app)
    video_id = upload(client, sample_video).json()["video_id"]
    texts = [s["text"] for s in client.get(f"/skill/api/videos/{video_id}").json()["segments"]]
    assert texts[1] == "次は[MASK]を入れます。"
    assert "まず型枠を立てます。" in llm.prompts_for("mask")[0]
