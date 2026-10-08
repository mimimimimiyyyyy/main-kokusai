"""
既存の対話研究向け機能（audio_analysis_pipeline.py）が、技能伝承機能の追加で
変わっていないことを確かめる回帰テスト。
"""
import json

import pytest
from fastapi.testclient import TestClient

from conftest import load_pipeline_cell, load_skill_cell

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
    return {(m, r.path) for r in app.routes if hasattr(r, "methods") for m in r.methods}


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
