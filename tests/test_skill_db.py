import csv
import json
import sqlite3

from conftest import SEED_CSV

SKILL_TABLES = {
    "skill_videos", "skill_segments", "skill_annotations",
    "skill_llm_results", "skill_tags", "skill_tag_candidates",
}


def table_names(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_init_creates_only_skill_tables(skill):
    conn = sqlite3.connect(":memory:")
    skill.init_skill_db(conn)
    assert table_names(conn) - {"sqlite_sequence"} == SKILL_TABLES


def test_init_is_idempotent(skill, conn):
    conn.execute("INSERT INTO skill_videos (title) VALUES ('既存の動画')")
    conn.commit()
    skill.init_skill_db(conn)
    assert conn.execute("SELECT title FROM skill_videos").fetchall() == [("既存の動画",)]


def test_skill_segments_has_same_core_columns_as_turns(conn):
    columns = [r[1] for r in conn.execute("PRAGMA table_info(skill_segments)")]
    assert columns == ["id", "video_id", "speaker", "start", "end", "text"]


def test_seed_inserts_all_rows_with_parents_and_aliases(skill, conn):
    result = skill.seed_skill_tags(conn, str(SEED_CSV))
    with open(SEED_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert result["inserted"] == len(rows)
    assert result["unknown_parents"] == []

    tags = {t["name"]: t for t in skill.list_skill_tags(conn)}
    assert len(tags) == len(rows)
    for r in rows:
        tag = tags[r["name"]]
        assert tag["category"] == r["category"]
        assert tag["aliases"] == skill.parse_aliases(r["aliases"])
        if r["parent"]:
            assert tag["parent_id"] == tags[r["parent"]]["tag_id"]
        else:
            assert tag["parent_id"] is None


def test_seed_categories_are_known(skill, seeded_conn):
    for tag in skill.list_skill_tags(seeded_conn):
        assert tag["category"] in skill.TAG_CATEGORIES


def test_seed_twice_does_not_duplicate(skill, seeded_conn):
    before = skill.list_skill_tags(seeded_conn)
    result = skill.seed_skill_tags(seeded_conn, str(SEED_CSV))
    assert result["inserted"] == 0
    assert skill.list_skill_tags(seeded_conn) == before


def test_seed_does_not_overwrite_edited_tags(skill, seeded_conn):
    seeded_conn.execute(
        "UPDATE skill_tags SET aliases=? WHERE name='インパクトドライバー'",
        (json.dumps(["インパクト", "インパクトレンチ"], ensure_ascii=False),)
    )
    seeded_conn.commit()
    skill.seed_skill_tags(seeded_conn, str(SEED_CSV))
    tag = [t for t in skill.list_skill_tags(seeded_conn) if t["name"] == "インパクトドライバー"][0]
    assert tag["aliases"] == ["インパクト", "インパクトレンチ"]


def test_seed_resolves_parent_defined_later_and_reports_unknown(skill, conn, tmp_path):
    path = tmp_path / "tags.csv"
    path.write_text(
        "name,category,parent,aliases\n"
        "子の作業,作業,親の作業,\n"
        "親の作業,作業,,\n"
        "迷子,作業,存在しない親,\n",
        encoding="utf-8",
    )
    result = skill.seed_skill_tags(conn, str(path))
    tags = {t["name"]: t for t in skill.list_skill_tags(conn)}
    assert tags["子の作業"]["parent_id"] == tags["親の作業"]["tag_id"]
    assert result["unknown_parents"] == [{"name": "迷子", "parent": "存在しない親"}]


def test_parse_aliases_trims_and_skips_empty(skill):
    assert skill.parse_aliases(" 丸のこ | |丸ノコ ") == ["丸のこ", "丸ノコ"]
    assert skill.parse_aliases("") == []
    assert skill.parse_aliases(None) == []


def test_list_tags_by_category(skill, seeded_conn):
    tools = skill.list_skill_tags(seeded_conn, category="道具")
    assert tools and all(t["category"] == "道具" for t in tools)


def test_env_overrides_settings():
    from conftest import load_skill_cell
    cell = load_skill_cell(env={"SKILL_SUBTITLE_MAX_CHARS": "16", "SKILL_DIFFICULTY_TAG": "true"})
    assert cell.SKILL_SUBTITLE_MAX_CHARS == 16
    assert cell.SKILL_DIFFICULTY_TAG is True


def test_defaults(skill):
    assert skill.SKILL_SUBTITLE_MAX_CHARS == 20
    assert skill.SKILL_SUBTITLE_MAX_LINES == 2
    assert skill.SKILL_DIFFICULTY_TAG is False
    assert skill.SKILL_ANONYMIZE is False
    assert skill.SKILL_LLM_PROVIDER == "openai"


def test_serialized_connection_survives_concurrent_queries(skill, conn):
    """処理スレッドと画面からの問い合わせが同じ動画を同時に読んでも、空の結果にならないこと。"""
    import threading
    db = skill.SerializedConnection(conn)
    video_id = skill.create_skill_video(db, "型枠", "", "a.mp4")
    failures = []

    def worker():
        for _ in range(300):
            if skill.get_skill_video(db, video_id) is None:
                failures.append(1)
            skill.update_skill_video(db, video_id, status="transcribe")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert failures == []


def test_serialized_connection_behaves_like_cursor(skill, conn):
    db = skill.SerializedConnection(conn)
    cur = db.execute("INSERT INTO skill_tags (name, category) VALUES ('a', '道具')")
    assert cur.lastrowid == 1 and cur.rowcount == 1
    db.commit()
    assert db.execute("SELECT name FROM skill_tags").fetchone() == ("a",)
    assert db.execute("SELECT name FROM skill_tags WHERE name='x'").fetchone() is None
    assert [r for r in db.execute("SELECT name FROM skill_tags")] == [("a",)]
