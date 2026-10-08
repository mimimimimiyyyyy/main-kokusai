# --- 別セル: 技能伝承動画共有機能 ---
# 建設業向けに、熟練者の作業説明動画から文字起こし・字幕・手順書・タグを自動で作り、
# 若手がタグで動画を探して見られるようにする機能。
#
# 実行順: このセル → audio_analysis_pipeline.py のセル
# audio_analysis_pipeline.py はセルの最後でサーバーを起動したまま止まる
# （await server.serve()）ため、後から別セルでAPIを追加できない。そこでこのセルでは
# 関数と設定の定義だけを行い、既存セルの「5. サーバー起動」の直前で
# register_skill_transfer() を呼んでもらう。このセルを実行していなければ既存セルは
# 何もしないので、対話研究向けの既存機能の動作は変わらない。
#
# 対話研究のデータ（sessions / turns など）とは混ぜず、同じcorpus.dbの中に
# skill_ で始まる表を別に作って保存する（docs/skill-transfer-mapping.md のQ1）。

!pip install noisereduce rapidfuzz weasyprint -q
!apt-get -qq install -y fonts-noto-cjk > /dev/null

import os, csv, json, sqlite3
from datetime import datetime


# --- 1. 設定 ---
# 秘密情報・設定値は環境変数から読む。ローカルでは .env、Colabではシークレット
# （既存セルと同じ google.colab.userdata）から環境変数に入れる。
def load_dotenv(path=".env"):
    """KEY=VALUE 形式の .env を読み、まだ設定されていない環境変数だけを入れる。"""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_colab_secrets(names):
    try:
        from google.colab import userdata
    except ImportError:
        return
    for name in names:
        if name in os.environ:
            continue
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            os.environ[name] = value


load_dotenv()
load_colab_secrets(["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])


def _env_bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


SKILL_MEDIA_DIR = os.environ.get("SKILL_MEDIA_DIR", "/content/drive/MyDrive/skill_transfer")
SKILL_SEED_CSV = os.environ.get("SKILL_SEED_CSV", os.path.join(SKILL_MEDIA_DIR, "seed", "skill_transfer_tags.csv"))
SKILL_HTML_PATH = os.environ.get("SKILL_HTML_PATH", os.path.join(SKILL_MEDIA_DIR, "skill_transfer.html"))
SKILL_SUBTITLE_MAX_CHARS = int(os.environ.get("SKILL_SUBTITLE_MAX_CHARS", "20"))
SKILL_SUBTITLE_MAX_LINES = int(os.environ.get("SKILL_SUBTITLE_MAX_LINES", "2"))
SKILL_TAG_FUZZY_THRESHOLD = float(os.environ.get("SKILL_TAG_FUZZY_THRESHOLD", "85"))
SKILL_DIFFICULTY_TAG = _env_bool("SKILL_DIFFICULTY_TAG", False)
SKILL_ANONYMIZE = _env_bool("SKILL_ANONYMIZE", False)
SKILL_WHISPER_LANGUAGE = os.environ.get("SKILL_WHISPER_LANGUAGE", "ja")
SKILL_LLM_PROVIDER = os.environ.get("SKILL_LLM_PROVIDER", "openai")
SKILL_LLM_MODEL = os.environ.get("SKILL_LLM_MODEL", "")

TAG_CATEGORIES = ["作業", "道具", "資材", "安全", "難易度"]


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- 2. DB（skill_ で始まる表だけを追加する。既存の表には触れない） ---
def init_skill_db(conn):
    # skill_videos: 作業動画1本（既存のsessionsに相当）。既存のsessionsは
    # 元の動画ファイルを保存しないが、技能伝承では再生が必要なので保存先を持つ。
    # 処理状態もここに永続化し、失敗した段階から再実行できるようにする
    # （既存のjobs辞書はメモリ上だけなので、再起動で消えてしまう）。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skill_videos (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT,
            explainer   TEXT,
            filename    TEXT,
            media_path  TEXT,
            duration    REAL,
            status      TEXT,
            failed_step TEXT,
            error       TEXT,
            vtt_path    TEXT,
            pdf_path    TEXT,
            created     TEXT,
            updated     TEXT
        )
    """)
    # skill_segments: 時間付きの文字起こし（既存のturnsと同じ列構成）。
    # 対話研究用のphase/intent/role/embeddingは持たない。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skill_segments (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            speaker  TEXT,
            start    REAL,
            end      REAL,
            text     TEXT,
            FOREIGN KEY (video_id) REFERENCES skill_videos(id)
        )
    """)
    # skill_annotations: 時間範囲つきのラベル。layerで種類を分ける
    # （'step'=手順、'scene_tag'=場面タグ、'video_tag'=動画タグ）。
    # start/endがNULLなら動画全体につくラベル。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skill_annotations (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id       INTEGER,
            layer          TEXT,
            start          REAL,
            end            REAL,
            label          TEXT,
            tag_id         INTEGER,
            parent_id      INTEGER,
            payload        TEXT,
            source         TEXT,
            method_version TEXT,
            created        TEXT,
            FOREIGN KEY (video_id) REFERENCES skill_videos(id)
        )
    """)
    # skill_llm_results: LLMの生の出力（既存のaddin_resultsと同じ考え方）。
    # 手順分割やタグ抽出をやり直しても上書きせず、method_version付きで全部残す。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skill_llm_results (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id       INTEGER,
            kind           TEXT,
            method_version TEXT,
            created        TEXT,
            result         TEXT,
            FOREIGN KEY (video_id) REFERENCES skill_videos(id)
        )
    """)
    # skill_tags: タグ一覧。作業の種類はcategory='作業'の親子関係で階層を表す。
    # aliasesは別名のJSON配列（現場や人による呼び方の違いを吸収するため）。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skill_tags (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT UNIQUE,
            category  TEXT,
            parent_id INTEGER,
            aliases   TEXT,
            created   TEXT,
            FOREIGN KEY (parent_id) REFERENCES skill_tags(id)
        )
    """)
    # skill_tag_candidates: タグ一覧に無い言葉。自動では登録せず、管理画面で
    # 人が採用・却下を決める（status: pending / adopted / rejected）。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skill_tag_candidates (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            word     TEXT,
            category TEXT,
            status   TEXT,
            created  TEXT,
            FOREIGN KEY (video_id) REFERENCES skill_videos(id)
        )
    """)
    conn.commit()


def parse_aliases(text):
    """CSVの '|' 区切りの別名を、空白を除いたリストにする。"""
    return [a.strip() for a in (text or "").split("|") if a.strip()]


def seed_skill_tags(conn, csv_path=SKILL_SEED_CSV):
    """
    タグ一覧の初期データ（列: name, category, parent, aliases）を投入する。
    同じ名前のタグが既にあれば上書きしない（管理画面で編集した内容を、
    セルの再実行で初期データに戻してしまわないため）。
    親は名前で指定し、CSV内の行の順番に関係なく解決する。
    """
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("name") or "").strip()]

    now = now_str()
    inserted = 0
    for r in rows:
        cur = conn.execute(
            "INSERT OR IGNORE INTO skill_tags (name, category, parent_id, aliases, created) VALUES (?,?,?,?,?)",
            (r["name"].strip(), (r.get("category") or "").strip(), None,
             json.dumps(parse_aliases(r.get("aliases")), ensure_ascii=False), now)
        )
        inserted += cur.rowcount

    unknown_parents = []
    for r in rows:
        parent = (r.get("parent") or "").strip()
        if not parent:
            continue
        parent_row = conn.execute("SELECT id FROM skill_tags WHERE name=?", (parent,)).fetchone()
        if parent_row is None:
            unknown_parents.append({"name": r["name"].strip(), "parent": parent})
            continue
        # 親が未設定のものだけ埋める（管理画面で付け替えた親は上書きしない）
        conn.execute(
            "UPDATE skill_tags SET parent_id=? WHERE name=? AND parent_id IS NULL",
            (parent_row[0], r["name"].strip())
        )
    conn.commit()
    return {"inserted": inserted, "total_rows": len(rows), "unknown_parents": unknown_parents}


def tag_row_to_dict(row):
    tag_id, name, category, parent_id, aliases = row
    return {
        "tag_id": tag_id, "name": name, "category": category,
        "parent_id": parent_id, "aliases": json.loads(aliases or "[]")
    }


def list_skill_tags(conn, category=None):
    sql = "SELECT id, name, category, parent_id, aliases FROM skill_tags"
    params = ()
    if category:
        sql += " WHERE category=?"
        params = (category,)
    return [tag_row_to_dict(r) for r in conn.execute(sql + " ORDER BY id", params).fetchall()]
