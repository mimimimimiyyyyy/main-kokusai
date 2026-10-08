"""
既存の対話研究向け機能（audio_analysis_pipeline.py）が、技能伝承機能の追加で
変わっていないことを確かめる回帰テスト。
"""
import asyncio
import json
import types

import pytest
import uvicorn
from fastapi.testclient import TestClient

from conftest import PIPELINE_CELL, SKILL_CELL, load_cells_in_colab_order, load_pipeline_cell, load_skill_cell

EXISTING_TABLES = ["sessions", "turns", "addin_results", "human_annotations", "reference_transcripts"]
EXISTING_ROUTES = {
    ("POST", "/upload"), ("GET", "/jobs/{job_id}"), ("POST", "/search"),
    ("GET", "/corpus"), ("GET", "/corpus/{session_id}"), ("GET", "/corpus/{session_id}/raw_turns"),
    ("POST", "/corpus/{session_id}/retag"), ("POST", "/corpus/{session_id}/addin"),
    ("POST", "/corpus/{session_id}/annotations"), ("GET", "/corpus/{session_id}/annotations"),
    ("GET", "/corpus/{session_id}/agreement"),
    ("POST", "/corpus/{session_id}/reference_transcripts"),
    ("GET", "/corpus/{session_id}/transcription_accuracy"),
}


def schema(conn, tables):
    return {
        name: sql for name, sql in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        if name in tables
    }


def routes(app):
    """appのルート一覧。新しいFastAPIはinclude_routerしたルーターを包んで持つので中まで見る。"""
    def walk(items):
        for r in items:
            if hasattr(r, "original_router"):
                yield from walk(r.original_router.routes)
            elif hasattr(r, "methods"):
                yield from ((m, r.path) for m in r.methods)
    return set(walk(app.routes))


@pytest.fixture
def pipeline(tmp_path):
    p = load_pipeline_cell(tmp_path / "corpus.db")
    yield p
    p.conn.close()


def insert_dialogue_session(p):
    cur = p.conn.execute(
        "INSERT INTO sessions (filename, created, summary, metadata, roles) VALUES (?,?,?,?,?)",
        ("input_meeting.mp4", "2026-01-01 10:00:00", "summary",
         json.dumps({"scene": "s", "task": "t"}), json.dumps({"SPEAKER_00": "Leader"}))
    )
    p.conn.execute(
        "INSERT INTO turns (session_id, speaker, start, end, text, phase, intent, role) VALUES (?,?,?,?,?,?,?,?)",
        (cur.lastrowid, "SPEAKER_00", 0.0, 1.5, "はじめましょう", "Introduction", "Proposal", "Leader")
    )
    p.conn.commit()
    return cur.lastrowid


def test_existing_cell_alone_keeps_its_routes(pipeline):
    assert EXISTING_ROUTES <= routes(pipeline.app)
    assert not any(path.startswith("/skill") for _, path in routes(pipeline.app))


def test_skill_tables_do_not_change_existing_tables_or_api(pipeline):
    session_id = insert_dialogue_session(pipeline)
    client = TestClient(pipeline.app)
    schema_before = schema(pipeline.conn, EXISTING_TABLES)
    corpus_before = client.get("/corpus").json()
    detail_before = client.get(f"/corpus/{session_id}").json()

    skill = load_skill_cell()
    skill.init_skill_db(pipeline.conn)
    pipeline.conn.execute("INSERT INTO skill_videos (title) VALUES ('型枠の組立')")
    pipeline.conn.commit()

    assert schema(pipeline.conn, EXISTING_TABLES) == schema_before
    assert len(schema_before) == len(EXISTING_TABLES)
    assert client.get("/corpus").json() == corpus_before
    assert client.get(f"/corpus/{session_id}").json() == detail_before


def test_existing_python_files_are_not_edited():
    """既存の研究用ファイルは編集しない方針。技能伝承用の行が入っていないことを確かめる。"""
    import subprocess
    changed = subprocess.run(
        ["git", "diff", "--name-only", "60aa9a7", "--", "audio_analysis_pipeline.py", "transcription_accuracy_tool.py"],
        capture_output=True, text=True, cwd=PIPELINE_CELL.parent, check=True,
    ).stdout.split()
    assert changed == []
    assert "skill" not in PIPELINE_CELL.read_text(encoding="utf-8").lower()


@pytest.fixture
def fake_uvicorn_serve():
    """
    uvicorn.Server.serve を「実際には待ち受けず、呼ばれたappを記録するだけ」に差し替える。
    技能伝承セルの差し込みは、この差し替えたserveを包む。テスト後に元へ戻す。
    """
    original = uvicorn.Server.serve
    served = []

    async def fake_serve(self, *args, **kwargs):
        served.append(self.config.app)

    uvicorn.Server.serve = fake_serve
    yield served
    uvicorn.Server.serve = original
    if "_skill_transfer_original_serve" in vars(uvicorn.Server):
        del uvicorn.Server._skill_transfer_original_serve


def start_server_without_listening(shared):
    """既存セルの最後（uvicorn.Server(config).serve()）と同じ呼び出しを行う。"""
    shared.install_skill_transfer_hook(namespace=shared.__dict__)
    server = uvicorn.Server(uvicorn.Config(shared.app, host="0.0.0.0", port=8000, loop="asyncio"))
    asyncio.run(server.serve())


def test_colab_order_registers_skill_api_at_server_start(tmp_path, fake_uvicorn_serve):
    shared = load_cells_in_colab_order(tmp_path / "corpus.db")
    session_id = insert_dialogue_session(shared)
    client = TestClient(shared.app)
    corpus_before = client.get("/corpus").json()
    # Colabでは既定のDrive上のパス。テストでは一時フォルダに向ける
    shared.SKILL_DB_PATH = str(tmp_path / "corpus.db")
    shared.SKILL_MEDIA_DIR = str(tmp_path / "skill_transfer")

    start_server_without_listening(shared)

    assert fake_uvicorn_serve == [shared.app]  # 既存のサーバー起動処理はそのまま呼ばれる
    ctx = shared.app.state.skill_transfer
    # 既存セルが読み込んだWhisperモデル・OpenAIクライアント・匿名化関数を共用する
    assert ctx.whisper_model is shared.whisper_model
    assert ctx.openai_client is shared.client
    assert ctx.mask_fn is shared.extract_mask_targets_with_gpt
    assert EXISTING_ROUTES <= routes(shared.app)
    assert ("POST", "/skill/api/videos") in routes(shared.app)
    assert client.get("/corpus").json() == corpus_before
    assert client.get(f"/corpus/{session_id}").status_code == 200
    # 技能伝承の表は同じcorpus.dbに、既存の表とは別に作られる
    tables = {r[0] for r in shared.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sessions", "turns", "skill_videos", "skill_segments"} <= tables
    ctx.conn.close()


def test_registration_failure_does_not_stop_existing_server(tmp_path, fake_uvicorn_serve, capsys):
    shared = load_cells_in_colab_order(tmp_path / "corpus.db")

    def broken(*args, **kwargs):
        raise RuntimeError("Driveがマウントされていません")

    shared.register_skill_transfer = broken
    start_server_without_listening(shared)
    assert fake_uvicorn_serve == [shared.app]
    assert EXISTING_ROUTES <= routes(shared.app)
    assert "既存の機能はそのまま起動します" in capsys.readouterr().out


def test_skill_cell_does_not_overwrite_names_of_existing_cell(tmp_path):
    """同じ変数の置き場で実行するので、既存セルの関数・変数を上書きしないこと。"""
    skill_names = {k for k, v in vars(load_skill_cell()).items()
                   if not k.startswith("__") and not isinstance(v, types.ModuleType)}
    pipeline = load_pipeline_cell(tmp_path / "corpus.db")
    pipeline_names = {k for k, v in vars(pipeline).items()
                      if not k.startswith("__") and not isinstance(v, types.ModuleType)}
    pipeline.conn.close()
    # 同じものをimportしているだけの名前（クラス・関数）は共有してよい
    shared_imports = {"datetime", "Optional", "BaseModel", "FastAPI", "UploadFile", "File"}
    assert (skill_names & pipeline_names) - shared_imports == set()
