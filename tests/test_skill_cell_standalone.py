"""
技能伝承のセル（skill_transfer_cell.py）が、1つのセルだけで完結することのテスト。

- 対話研究のパイプライン（audio_analysis_pipeline.py）は使わず、編集もしない
- セルを丸ごと実行すると、ドライブのマウント → Whisperの読み込み → ngrok → サーバー起動 まで行い、
  技能伝承の画面とAPIが使える状態になる（Colab専用の部分だけを差し替えて確かめる）
"""
import csv
import io
import os
import subprocess

from fastapi.testclient import TestClient

from conftest import (ACCURACY_CELL, PIPELINE_CELL, SEED_CSV, SKILL_CELL, ColabStubs, FakeLLM,
                      FakeWhisperModel, run_whole_cell)


def test_existing_python_files_are_not_edited():
    changed = subprocess.run(
        ["git", "diff", "--name-only", "60aa9a7", "--", PIPELINE_CELL.name, ACCURACY_CELL.name],
        capture_output=True, text=True, cwd=PIPELINE_CELL.parent, check=True,
    ).stdout.split()
    assert changed == []
    assert "skill" not in PIPELINE_CELL.read_text(encoding="utf-8").lower()


def test_cell_does_not_depend_on_dialogue_pipeline():
    """他のセルの変数（app・conn・whisper_model・client など）を当てにしていないこと。"""
    source = SKILL_CELL.read_text(encoding="utf-8")
    for name in ["transcribe_full_audio(", "extract_mask_targets_with_gpt", "globals()", "corpus.db",
                 "install_skill_transfer_hook"]:
        assert name not in source, name


class RecordingServe:
    """uvicorn.Server.serve の代わり。起動しようとしたappと設定を記録する（待ち受けはしない）。"""

    def __init__(self):
        self.calls = []

    def __call__(self_outer):
        async def serve(server, *args, **kwargs):
            self_outer.calls.append(server.config)
        return serve


def run_cell(tmp_path, llm_client=None, **env):
    stubs = ColabStubs(whisper_model=FakeWhisperModel(), openai_client=llm_client or object())
    recorder = RecordingServe()
    namespace = run_whole_cell(stubs, {"SKILL_MEDIA_DIR": str(tmp_path / "skill_transfer"),
                                       "NGROK_AUTH_TOKEN": "token", **env}, serve=recorder())
    return namespace, stubs, recorder


def test_running_the_cell_alone_starts_everything(tmp_path, capsys):
    namespace, stubs, recorder = run_cell(tmp_path)

    assert stubs.mounted == ["/content/drive"]                  # ドライブのマウント
    assert stubs.loaded_models == [("medium", "cuda")]          # Whisperの読み込み（GPU）
    assert stubs.ngrok_commands == [["ngrok", "http", "8000"]]  # ngrok
    [config] = recorder.calls                                   # サーバー起動
    assert config.app is namespace["skill_app"] and config.port == 8000 and config.host == "0.0.0.0"
    assert "https://example.ngrok-free.dev/skill" in capsys.readouterr().out

    ctx = namespace["skill_app"].state.skill_transfer
    assert ctx.whisper_model is stubs.whisper_model
    assert ctx.conn.raw.execute("PRAGMA database_list").fetchone()[2] == str(tmp_path / "skill_transfer" / "skill_transfer.db")

    client = TestClient(namespace["skill_app"])
    page = client.get("/skill")
    assert page.status_code == 200 and "技能伝承動画" in page.text and "<script>" in page.text
    assert "/skill" in client.get("/").text
    tags = client.get("/skill/api/tags").json()["tags"]
    assert len(tags) == len(list(csv.DictReader(open(SEED_CSV, encoding="utf-8"))))
    ctx.conn.close()


def test_cell_settings_from_environment(tmp_path):
    namespace, stubs, recorder = run_cell(tmp_path, SKILL_WHISPER_MODEL="large-v3", SKILL_PORT="8100")
    assert stubs.loaded_models == [("large-v3", "cuda")]
    assert stubs.ngrok_commands == [["ngrok", "http", "8100"]]
    assert recorder.calls[0].port == 8100
    namespace["skill_app"].state.skill_transfer.conn.close()


def test_rerunning_the_cell_keeps_data(tmp_path):
    first, _, _ = run_cell(tmp_path)
    ctx = first["skill_app"].state.skill_transfer
    first["create_skill_video"](ctx.conn, "型枠の建て込み", "", "a.mp4")
    ctx.conn.execute("UPDATE skill_tags SET aliases='[\"インパクト\", \"インパクトレンチ\"]' WHERE name='インパクトドライバー'")
    ctx.conn.commit()
    ctx.conn.close()

    second, _, _ = run_cell(tmp_path)
    client = TestClient(second["skill_app"])
    assert client.get("/skill/api/videos/1").json()["title"] == "型枠の建て込み"
    tag = next(t for t in client.get("/skill/api/tags").json()["tags"] if t["name"] == "インパクトドライバー")
    assert tag["aliases"] == ["インパクト", "インパクトレンチ"]  # 初期データで上書きしない
    second["skill_app"].state.skill_transfer.conn.close()


def test_embedded_seed_is_same_as_seed_csv(skill):
    with open(SEED_CSV, encoding="utf-8") as f:
        assert skill.SKILL_SEED_TAGS_CSV == f.read()


def test_seed_csv_setting_overrides_embedded(skill, conn, tmp_path):
    path = tmp_path / "tags.csv"
    path.write_text("name,category,parent,aliases\nバール,道具,,かじや\n", encoding="utf-8")
    skill.seed_skill_tags(conn, str(path))
    assert [t["name"] for t in skill.list_skill_tags(conn)] == ["バール"]
