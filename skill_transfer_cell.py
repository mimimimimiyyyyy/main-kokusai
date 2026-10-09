# --- 技能伝承動画共有機能（建設業向け）: このセル1つで完結 ---
# 熟練者の作業説明動画から文字起こし・字幕・手順書・タグを自動で作り、
# 若手がタグで動画を探して見られるようにする。
#
# 使い方: このセルを実行するだけ（Googleドライブのマウント → Whisperの読み込み →
# サーバー起動 → ngrokのURL表示 まで行う）。表示されたURLの末尾に /skill を付けて
# スマホのブラウザで開く。事前にColabのシークレットに OPENAI_API_KEY と
# NGROK_AUTH_TOKEN を登録しておく（Claudeを使う場合は ANTHROPIC_API_KEY も）。
#
# 対話研究のパイプライン（audio_analysis_pipeline.py）とは独立して動く。DBも別の
# ファイル（skill_transfer.db）に保存し、対話研究のデータには一切触れない。
# 技術構成・書き方（FastAPI＋ngrok、SQLite、Whisper、ジョブを裏で動かして画面から
# ポーリングする方式など）は、対話研究のパイプラインに合わせている。

!pip install fastapi uvicorn pyngrok nest_asyncio python-multipart pydub openai-whisper noisereduce rapidfuzz weasyprint openai anthropic -q
!apt-get -qq install -y fonts-noto-cjk > /dev/null

import os, io, re, csv, json, time, shutil, sqlite3, inspect, threading, traceback, subprocess
from datetime import datetime
from typing import Optional

import numpy as np


# --- 1. 設定 ---
# 秘密情報・設定値は環境変数から読む。ローカルでは .env、Colabではシークレット
# （google.colab.userdata）から環境変数に入れる。
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
load_colab_secrets(["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NGROK_AUTH_TOKEN"])


def _env_bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# 動画・写真・PDFとDBは、Colabを閉じても残るようにGoogleドライブに保存する
SKILL_MEDIA_DIR = os.environ.get("SKILL_MEDIA_DIR", "/content/drive/MyDrive/skill_transfer")
SKILL_DB_PATH = os.environ.get("SKILL_DB_PATH", os.path.join(SKILL_MEDIA_DIR, "skill_transfer.db"))
# タグ一覧の初期データ。空欄ならこのセルに入っている初期データ（seed/skill_transfer_tags.csv と同じ）を使う
SKILL_SEED_CSV = os.environ.get("SKILL_SEED_CSV", "")
SKILL_WHISPER_MODEL = os.environ.get("SKILL_WHISPER_MODEL", "medium")
SKILL_PORT = int(os.environ.get("SKILL_PORT", "8000"))
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


# --- 2. DB（技能伝承専用の skill_transfer.db） ---
def init_skill_db(conn):
    # skill_videos: 作業動画1本（対話研究のsessionsに相当）。対話研究のsessionsは
    # 元の動画ファイルを保存しないが、技能伝承では再生が必要なので保存先を持つ。
    # 処理状態もここに永続化し、失敗した段階から再実行できるようにする
    # （メモリ上だけに持つと、Colabの再起動で消えてしまう）。
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
    # skill_segments: 時間付きの文字起こし（対話研究のturnsと同じ列構成）。
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
    # skill_llm_results: LLMの生の出力（対話研究のaddin_resultsと同じ考え方）。
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


def seed_skill_tags(conn, csv_path=None):
    """
    タグ一覧の初期データ（列: name, category, parent, aliases）を投入する。
    csv_pathを省略すると、このセルに入っている初期データ（SKILL_SEED_TAGS_CSV）を使う。
    同じ名前のタグが既にあれば上書きしない（管理画面で編集した内容を、
    セルの再実行で初期データに戻してしまわないため）。
    親は名前で指定し、CSV内の行の順番に関係なく解決する。
    """
    if csv_path:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            text = f.read()
    else:
        text = SKILL_SEED_TAGS_CSV
    rows = [r for r in csv.DictReader(io.StringIO(text.lstrip("\ufeff"))) if (r.get("name") or "").strip()]

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


class QueryResult:
    """SerializedConnection.execute の戻り値（結果は取り出し済み）。sqlite3のカーソルと同じ使い方ができる。"""

    def __init__(self, rows, lastrowid, rowcount):
        self._rows = rows
        self.lastrowid = lastrowid
        self.rowcount = rowcount

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows

    def __iter__(self):
        return iter(self.fetchall())


class SerializedConnection:
    """
    1つのSQLite接続を、処理のスレッドと画面からの問い合わせ（状況の確認など）で同時に使うと、
    同じSQL文の準備済みステートメントを取り合って、片方に空の結果が返ることがある。
    問い合わせ1回分（実行〜結果の取り出し）をロックの中で行い、順番に処理する。
    """

    def __init__(self, conn):
        self.raw = conn
        self._lock = threading.RLock()

    def execute(self, sql, params=()):
        with self._lock:
            cur = self.raw.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
            return QueryResult(rows, cur.lastrowid, cur.rowcount)

    def commit(self):
        with self._lock:
            self.raw.commit()

    def close(self):
        with self._lock:
            self.raw.close()


def open_skill_db(path=None):
    """技能伝承のDB（skill_transfer.db）を開き、表を用意する。"""
    path = path or SKILL_DB_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    init_skill_db(conn)
    return conn


# --- 3. 動画・音声の下ごしらえ ---
def video_dir(video_id, media_dir=SKILL_MEDIA_DIR):
    path = os.path.join(media_dir, "videos", str(video_id))
    os.makedirs(path, exist_ok=True)
    return path


def run_ffmpeg(args):
    """ffmpeg/ffprobe を実行し、失敗したら標準エラーの末尾を含めて例外にする。"""
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{args[0]} が失敗しました: {proc.stderr.strip()[-500:]}")
    return proc.stdout


def probe_duration(path):
    out = run_ffmpeg(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=noprint_wrappers=1:nokey=1", path])
    return round(float(out.strip()), 2)


def convert_for_playback(src, dst):
    """
    スマホで撮った動画（iPhoneのHEVC/MOVなど）は、閲覧する端末によっては再生できない
    ことがあるため、どの端末でも再生しやすい H.264/AAC のMP4に変換して保存する。
    faststartは、ダウンロードが終わる前に再生を始められるようにするため。
    """
    run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", dst])


def extract_frame(video_path, seconds, dst, width=640):
    """指定時刻の1コマを静止画（JPEG）として切り出す。サムネイルと手順書の写真に使う。"""
    run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(seconds, 0):.2f}", "-i", video_path,
                "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", dst])
    return dst


def extract_audio(video_path, wav_path, denoise=True):
    """
    ① の1〜2: 動画から音声を取り出し、騒音を減らして16kHz・モノラルのWAVにする。
    16kHz・モノラル化と音量正規化は、対話研究のパイプラインの音声変換と同じ処理。
    重機や工具の音は一定ではないため、noisereduceは非定常（stationary=False）で使う。
    """
    from pydub import AudioSegment, effects as audio_effects
    audio = AudioSegment.from_file(video_path).set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio = audio_effects.normalize(audio)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    if denoise and len(samples) > 0:
        import noisereduce as nr
        samples = nr.reduce_noise(y=samples, sr=16000, stationary=False)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    AudioSegment(pcm.tobytes(), frame_rate=16000, sample_width=2, channels=1).export(wav_path, format="wav")
    return wav_path


# --- 4. ① 文字起こし ---
# 対話研究のパイプライン（audio_analysis_pipeline.py の transcribe_full_audio）で実データに
# 合わせて調整した設定・幻覚フィルタと同じ値に、言語指定と専門用語ヒントを足している。
# このセルは独立して動くので、値はここに持っている。
SKILL_WHISPER_NO_SPEECH_THRESHOLD = 0.6
SKILL_WHISPER_LOGPROB_THRESHOLD = -1.0
SKILL_WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4
SKILL_WHISPER_HALLUCINATION_PHRASES = {
    "thanks for watching", "thank you for watching", "thank you for your support",
    "please subscribe", "like and subscribe", "see you next time",
    "don't forget to subscribe", "bye", "bye bye",
    # 日本語の無音・騒音区間で出やすい定型文（技能伝承は日本語で認識するため追加）
    "ご視聴ありがとうございました", "ご視聴ありがとうございました。",
    "チャンネル登録よろしくお願いします", "チャンネル登録お願いします",
}
SENTENCE_END_CHARS = "。！？!?"


VOCABULARY_CATEGORY_ORDER = ["道具", "資材", "作業", "安全"]


def build_vocabulary_prompt(tags, max_chars=150):
    """
    タグ一覧の名前と別名を、Whisperへの語彙ヒント（initial_prompt）にする。
    専門用語（ハッカー、セパ、コンパネなど）が一般的な言葉に誤認識されるのを減らすため。
    Whisperはプロンプトが長すぎると先頭側を切り捨てるので、文字数で打ち切る。
    1つの分類だけで埋まらないよう、分類ごとに1語ずつ交互に入れる
    （誤認識されやすい道具・資材の呼び名を優先し、正式名を別名より先に入れる）。
    """
    names, aliases = {}, {}
    for tag in tags:
        if tag.get("category") == "難易度":
            continue
        names.setdefault(tag.get("category"), []).append(tag["name"])
        aliases.setdefault(tag.get("category"), []).extend(tag.get("aliases", []))
    order = VOCABULARY_CATEGORY_ORDER + [c for c in names if c not in VOCABULARY_CATEGORY_ORDER]
    words, seen = [], set()
    # 正式名を全分類から交互に入れ、余った文字数で別名を入れる
    for by_category in (names, aliases):
        queues = [by_category[c] for c in order if c in by_category]
        for i in range(max((len(q) for q in queues), default=0)):
            for q in queues:
                if i < len(q) and q[i] not in seen:
                    seen.add(q[i])
                    words.append(q[i])
    prompt = "建設現場の作業説明。用語: "
    for word in words:
        if len(prompt) + len(word) + 1 > max_chars:
            break
        prompt += word + "、"
    return prompt.rstrip("、") + "。"


def transcribe_skill_audio(audio_path, whisper_model, language=SKILL_WHISPER_LANGUAGE, initial_prompt=None):
    """
    音声全体をWhisperに1回で通し、セグメント（単語のタイムスタンプ付き）を返す。
    temperature=0.0 / word_timestamps=True / condition_on_previous_text=False と
    幻覚フィルタは対話研究のパイプラインと同じ（理由は audio_analysis_pipeline.py のコメント参照）。
    condition_on_previous_text=False のままだと initial_prompt は最初の30秒にしか
    効かないため、carry_initial_prompt に対応した版では全区間に付ける。
    """
    options = {}
    if language:
        options["language"] = language
    if initial_prompt:
        options["initial_prompt"] = initial_prompt
        if "carry_initial_prompt" in inspect.signature(whisper_model.transcribe).parameters:
            options["carry_initial_prompt"] = True
    result = whisper_model.transcribe(
        audio_path, temperature=0.0, word_timestamps=True, condition_on_previous_text=False, **options
    )
    kept = []
    for seg in result.get("segments") or []:
        no_speech_prob = seg.get("no_speech_prob", 0.0)
        avg_logprob = seg.get("avg_logprob", 0.0)
        if no_speech_prob > SKILL_WHISPER_NO_SPEECH_THRESHOLD and avg_logprob < SKILL_WHISPER_LOGPROB_THRESHOLD:
            continue
        if seg.get("compression_ratio", 0.0) > SKILL_WHISPER_COMPRESSION_RATIO_THRESHOLD:
            continue
        text = seg.get("text", "").strip()
        if not text:
            continue
        if text.lower().strip(" .!?") in SKILL_WHISPER_HALLUCINATION_PHRASES:
            continue
        words = seg.get("words") or [{"word": text, "start": seg["start"], "end": seg["end"]}]
        kept.append({"start": seg["start"], "end": seg["end"], "text": text, "words": words})
    return kept


def split_sentences(whisper_segments):
    """
    ① の3: Whisperのセグメントを「文」に分け、文ごとに開始・終了時間を付ける。
    Whisperの1セグメントに複数の文が入ることがあるため、単語のタイムスタンプを使って
    「。」「？」などの文末で区切る。文末記号が無い場合はセグメントの切れ目を文の切れ目とする。
    """
    sentences = []
    for seg in whisper_segments:
        buf = []
        for w in seg["words"]:
            buf.append(w)
            if w["word"].strip().endswith(tuple(SENTENCE_END_CHARS)):
                sentences.append(buf)
                buf = []
        if buf:
            sentences.append(buf)

    result = []
    for words in sentences:
        text = "".join(w["word"] for w in words).strip()
        if not text:
            continue
        result.append({"start": round(words[0]["start"], 2), "end": round(words[-1]["end"], 2), "text": text})
    return result


def mask_sentences(sentences, mask_fn):
    """
    設定でONのときだけ、匿名化すべき言葉（人名・会社名・現場名・連絡先など）を
    [MASK]に置き換える。既定はOFF（道具やメーカーの名前まで伏せてしまうため）。
    mask_fn(sentences) は [{"word": ...}, ...] を返す関数（既定は extract_mask_words）。
    """
    for target in mask_fn(sentences):
        word = target.get("word")
        if word:
            for s in sentences:
                s["text"] = s["text"].replace(word, "[MASK]")
    return sentences


def save_segments(conn, video_id, sentences, speaker="SPEAKER_00"):
    """文字起こしを skill_segments に保存する（やり直し時は前の結果を消してから入れる）。"""
    conn.execute("DELETE FROM skill_segments WHERE video_id=?", (video_id,))
    for s in sentences:
        conn.execute(
            "INSERT INTO skill_segments (video_id, speaker, start, end, text) VALUES (?,?,?,?,?)",
            (video_id, speaker, s["start"], s["end"], s["text"])
        )
    conn.commit()


def get_segments(conn, video_id):
    rows = conn.execute(
        "SELECT id, speaker, start, end, text FROM skill_segments WHERE video_id=? ORDER BY start, id",
        (video_id,)
    ).fetchall()
    return [{"segment_id": r[0], "speaker": r[1], "start": r[2], "end": r[3], "text": r[4]} for r in rows]


# --- 5. ① 字幕（WebVTT） ---
def wrap_subtitle_text(text, max_chars=SKILL_SUBTITLE_MAX_CHARS):
    """
    字幕1行あたり最大max_chars文字で区切る。行の後半に読点・句点があれば、
    そこで改行して読みやすくする（無ければ文字数ちょうどで切る）。
    """
    text = text.strip()
    lines = []
    while len(text) > max_chars:
        cut = max_chars
        for i in range(max_chars, max_chars // 2, -1):
            if text[i - 1] in "、。！？,.!? ":
                cut = i
                break
        lines.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        lines.append(text)
    return lines


def build_subtitle_cues(segments, max_chars=SKILL_SUBTITLE_MAX_CHARS, max_lines=SKILL_SUBTITLE_MAX_LINES):
    """
    ① の4: 各文を「1行max_chars文字×最大max_lines行」の字幕に分け、文の表示時間を
    字幕ごとの文字数に比例して割り当てる。
    """
    cues = []
    for seg in segments:
        lines = wrap_subtitle_text(seg["text"], max_chars)
        if not lines:
            continue
        groups = [lines[i:i + max_lines] for i in range(0, len(lines), max_lines)]
        total_chars = sum(len(l) for l in lines)
        start = seg["start"]
        duration = max(seg["end"] - seg["start"], 0.01)
        for g_index, group in enumerate(groups):
            if g_index == len(groups) - 1:
                end = seg["end"] if seg["end"] > start else start + 0.01
            else:
                end = start + duration * sum(len(l) for l in group) / total_chars
            cues.append({"start": round(start, 3), "end": round(end, 3), "lines": group})
            start = end
    return cues


def format_vtt_time(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def build_webvtt(segments, max_chars=SKILL_SUBTITLE_MAX_CHARS, max_lines=SKILL_SUBTITLE_MAX_LINES):
    out = ["WEBVTT", ""]
    for i, cue in enumerate(build_subtitle_cues(segments, max_chars, max_lines), start=1):
        out.append(str(i))
        out.append(f"{format_vtt_time(cue['start'])} --> {format_vtt_time(cue['end'])}")
        out.extend(cue["lines"])
        out.append("")
    return "\n".join(out) + "\n"


# --- 6. 処理の流れ（状態の記録と、失敗した段階からの再実行） ---
# 処理の状態をメモリ上だけに持つと、Colabの再起動で消えてしまう。技能伝承では
# 状態・失敗した段階・エラー内容をskill_videosに残し、画面から失敗した段階だけを
# やり直せるようにする。
VIDEO_COLUMNS = ["id", "title", "explainer", "filename", "media_path", "duration", "status",
                 "failed_step", "error", "vtt_path", "pdf_path", "created", "updated"]


class SkillContext:
    """技能伝承の処理に必要なもの一式（DB、保存先、Whisperモデル、LLMなど）。"""

    def __init__(self, conn, media_dir=SKILL_MEDIA_DIR, whisper_model=None, llm_fn=None,
                 mask_fn=None, anonymize=SKILL_ANONYMIZE, difficulty=SKILL_DIFFICULTY_TAG,
                 run_in_background=True):
        self.conn = conn
        self.media_dir = media_dir
        self.whisper_model = whisper_model
        self.llm_fn = llm_fn
        self.mask_fn = mask_fn
        self.anonymize = anonymize
        self.difficulty = difficulty
        self.run_in_background = run_in_background
        # GPU上のWhisperを複数の動画で同時に使うとメモリが足りなくなるため、
        # 技能伝承の処理は1本ずつ順番に行う。
        self.pipeline_lock = threading.Lock()
        self.running = set()


def create_skill_video(conn, title, explainer, filename):
    now = now_str()
    cur = conn.execute(
        "INSERT INTO skill_videos (title, explainer, filename, status, created, updated) VALUES (?,?,?,?,?,?)",
        (title, explainer, filename, "uploaded", now, now)
    )
    conn.commit()
    return cur.lastrowid


def update_skill_video(conn, video_id, **fields):
    fields["updated"] = now_str()
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE skill_videos SET {sets} WHERE id=?", (*fields.values(), video_id))
    conn.commit()


def get_skill_video(conn, video_id):
    row = conn.execute(f"SELECT {', '.join(VIDEO_COLUMNS)} FROM skill_videos WHERE id=?", (video_id,)).fetchone()
    return dict(zip(VIDEO_COLUMNS, row)) if row else None


def original_path(ctx, video_id):
    folder = video_dir(video_id, ctx.media_dir)
    for name in os.listdir(folder):
        if name.startswith("original"):
            return os.path.join(folder, name)
    raise FileNotFoundError("アップロードされた元の動画が見つかりません")


def step_media(ctx, video_id):
    """再生用のMP4への変換、長さの取得、一覧用のサムネイル作成。"""
    folder = video_dir(video_id, ctx.media_dir)
    src = original_path(ctx, video_id)
    dst = os.path.join(folder, "video.mp4")
    convert_for_playback(src, dst)
    duration = probe_duration(dst)
    try:
        extract_frame(dst, min(1.0, duration / 2), os.path.join(folder, "thumbnail.jpg"))
    except RuntimeError:
        traceback.print_exc()  # サムネイルが無くても閲覧はできるので、処理は止めない
    update_skill_video(ctx.conn, video_id, media_path=dst, duration=duration)


def step_transcribe(ctx, video_id):
    """① 文字起こし・字幕作成（音声取り出し → 騒音除去 → 音声認識 → 文分割 → 字幕）。"""
    video = get_skill_video(ctx.conn, video_id)
    folder = video_dir(video_id, ctx.media_dir)
    wav_path = extract_audio(video["media_path"], os.path.join(folder, "audio.wav"))
    prompt = build_vocabulary_prompt(list_skill_tags(ctx.conn))
    whisper_segments = transcribe_skill_audio(wav_path, ctx.whisper_model, initial_prompt=prompt)
    sentences = split_sentences(whisper_segments)
    if not sentences:
        raise RuntimeError("音声から説明の言葉を認識できませんでした（音声が無い、または騒音が大きすぎる可能性があります）")
    if ctx.anonymize and ctx.mask_fn:
        sentences = mask_sentences(sentences, ctx.mask_fn)
    save_segments(ctx.conn, video_id, sentences, speaker=video["explainer"] or "SPEAKER_00")
    vtt_path = os.path.join(folder, "subtitles.vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write(build_webvtt(get_segments(ctx.conn, video_id)))
    update_skill_video(ctx.conn, video_id, vtt_path=vtt_path)




def run_skill_pipeline(ctx, video_id, start_step=None):
    """
    動画1本を、start_step（省略時は最初）から最後の段階まで順に処理する。
    失敗したら status='error' と失敗した段階・エラー内容を残して止まる。
    """
    names = SKILL_STEP_NAMES
    start_index = names.index(start_step) if start_step else 0
    with ctx.pipeline_lock:
        try:
            for name, func in SKILL_STEPS[start_index:]:
                update_skill_video(ctx.conn, video_id, status=name, failed_step=None, error=None)
                try:
                    func(ctx, video_id)
                except Exception as e:
                    traceback.print_exc()
                    update_skill_video(ctx.conn, video_id, status="error", failed_step=name,
                                       error=f"{type(e).__name__}: {e}")
                    return False
            update_skill_video(ctx.conn, video_id, status="done")
            return True
        finally:
            ctx.running.discard(video_id)


def start_skill_pipeline(ctx, video_id, start_step=None):
    ctx.running.add(video_id)
    if ctx.run_in_background:
        threading.Thread(target=run_skill_pipeline, args=(ctx, video_id, start_step), daemon=True).start()
    else:
        run_skill_pipeline(ctx, video_id, start_step)


# --- 7. LLMの呼び出し（1か所にまとめる） ---
# 既定は対話研究のパイプラインと同じOpenAI（gpt-4o・JSONモード）。SKILL_LLM_PROVIDER=anthropic で
# Claudeに切り替えられる。どちらも「プロンプトを渡してJSONの文字列を受け取る」関数
# （llm_fn）として扱い、手順書・タグ付けはこの関数だけを使う。テストではこの関数を
# モックに差し替える。
SKILL_LLM_DEFAULT_MODELS = {"openai": "gpt-4o", "anthropic": "claude-opus-5-5"}


def make_llm_fn(provider=None, model=None, openai_client=None):
    provider = provider or SKILL_LLM_PROVIDER
    if provider not in SKILL_LLM_DEFAULT_MODELS:
        raise ValueError(f"SKILL_LLM_PROVIDER は openai か anthropic を指定してください: {provider}")
    model = model or SKILL_LLM_MODEL or SKILL_LLM_DEFAULT_MODELS[provider]
    clients = {}

    def call_openai(prompt):
        if "openai" not in clients:
            import openai
            clients["openai"] = openai_client or openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        res = clients["openai"].chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return res.choices[0].message.content

    def call_anthropic(prompt):
        if "anthropic" not in clients:
            import anthropic
            clients["anthropic"] = anthropic.Anthropic()
        options = {}
        if model == "claude-opus-5-5":
            # 安全判定で断られた場合に、同じリクエストを別のモデルで続けるサーバー側の仕組み
            options = {"betas": ["server-side-fallback-2026-06-01"], "fallbacks": [{"model": "claude-opus-4-8"}]}
        res = clients["anthropic"].beta.messages.create(
            model=model, max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            **options
        )
        if res.stop_reason == "refusal":
            raise RuntimeError("LLMが応答を断りました")
        return "".join(b.text for b in res.content if b.type == "text")

    call = call_openai if provider == "openai" else call_anthropic
    call.method_version = f"{provider}:{model}"
    return call


def parse_json_text(text):
    """LLMの出力からJSONを取り出す（```json で囲まれていても読めるようにする）。"""
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSONが見つかりません")
    return json.loads(text[start:end + 1])


def call_llm_json(llm_fn, prompt, validate):
    """
    LLMに固定スキーマのJSONを出させ、validate(dict) で検証する。検証に失敗したら、
    理由を添えて1回だけ再試行する（2回とも失敗したらエラーにして状態に残す）。
    戻り値: (検証済みの結果, LLMの生の出力)
    """
    errors = []
    current = prompt
    for attempt in range(2):
        raw = llm_fn(current)
        try:
            return validate(parse_json_text(raw)), raw
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
            current = (prompt + "\n\n# 前回の出力は次の理由で不正でした。指定どおりのJSONだけを返してください。\n"
                       + errors[-1][:1000])
    raise ValueError("LLMの出力が2回とも指定の形式になりませんでした: " + " / ".join(e[:300] for e in errors))


def save_llm_result(conn, video_id, kind, method_version, result):
    conn.execute(
        "INSERT INTO skill_llm_results (video_id, kind, method_version, created, result) VALUES (?,?,?,?,?)",
        (video_id, kind, method_version, now_str(), json.dumps(result, ensure_ascii=False))
    )
    conn.commit()


def extract_mask_words(llm_fn, sentences):
    """匿名化すべき言葉をLLMで全文から抜き出す（SKILL_ANONYMIZE=true のときだけ使う）。"""
    lines = "\n".join(s["text"] for s in sentences)
    prompt = f"""# task: mask
次の作業説明の文字起こしから、匿名化が必要な言葉（人名、会社名、現場名・工事名、電話番号、
メールアドレス、住所など）を抜き出してください。道具・資材・工法などの一般的な言葉は含めないでください。

{lines}

次の形のJSONだけを返してください: {{"mask_list": [{{"word": "..."}}]}}
"""

    def validate(data):
        words = data.get("mask_list")
        if not isinstance(words, list):
            raise ValueError("mask_list がありません")
        return [w for w in words if isinstance(w, dict) and str(w.get("word", "")).strip()]

    result, _ = call_llm_json(llm_fn, prompt, validate)
    return result


# --- 8. ② 手順書 ---
from pydantic import BaseModel, field_validator


class ProcedureStep(BaseModel):
    title: str
    description: str
    start_segment: int
    end_segment: int
    tools: list[str] = []
    materials: list[str] = []
    cautions: list[str] = []
    tips: list[str] = []

    @field_validator("title", "description")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("空欄です")
        return v.strip()


class ProcedureResult(BaseModel):
    steps: list[ProcedureStep]


def validate_procedure(data, n_segments):
    """
    スキーマに加えて、セグメント番号が範囲内・順番どおり・重ならないことを確かめる。
    手順が0個でも不正にはしない（説明が短い動画などで、LLMが手順に分けられないことがある。
    その場合は generate_procedure で動画全体を1つの手順にする）。
    """
    result = ProcedureResult.model_validate(data)
    prev_end = -1
    for i, step in enumerate(result.steps):
        if not (0 <= step.start_segment <= step.end_segment < n_segments):
            raise ValueError(f"手順{i + 1}のセグメント番号が範囲外です（0〜{n_segments - 1}）")
        if step.start_segment <= prev_end:
            raise ValueError(f"手順{i + 1}が前の手順と重なっているか、順番が逆です")
        prev_end = step.end_segment
    return result


def build_procedure_prompt(title, segments):
    lines = "\n".join(
        f"[{i}] ({s['start']:.1f}〜{s['end']:.1f}秒) {s['text']}" for i, s in enumerate(segments)
    )
    return f"""# task: procedure
あなたは建設現場の技能伝承を手伝う専門家です。熟練者が作業しながら説明した動画の
文字起こし（番号付きの文）から、若手向けの手順書を作ります。

ルール:
- 「まず」「最初に」「次に」「次は」「それから」「続いて」「最後に」などの区切りの言葉を手がかりに、作業の手順に分ける
- 各手順は、連続した文の範囲（start_segment〜end_segment、番号は下の[ ]の数字）で表す。手順同士は重ねず、順番どおりに並べる
- 作業に関係ない雑談やあいさつの文は、どの手順にも含めなくてよい
- 手順は必ず1つ以上返す。区切りの言葉が無くても、内容のまとまりごとに分ける。説明が短い場合は、全体を1つの手順にまとめてよい
- title: 手順の短い見出し（例: 型枠を建て込む）
- description: その手順で何をするかを、若手が読んで分かるように1〜3文で
- tools: 使う道具 / materials: 使う資材 / cautions: 注意点・危険 / tips: コツ・勘所（「〜するのがコツ」「感覚としては〜」など、熟練者ならではの説明）
- 文字起こしに出てこないことは書かない。該当がなければ空の配列にする
- 文字起こしには音声認識の誤りが含まれることがある。文脈から明らかな誤りは正しい用語で書いてよい

動画のタイトル: {title}

文字起こし:
{lines}

次の形のJSONだけを返してください:
{{"steps": [{{"title": "...", "description": "...", "start_segment": 0, "end_segment": 3,
  "tools": ["..."], "materials": ["..."], "cautions": ["..."], "tips": ["..."]}}]}}
"""


def generate_procedure(llm_fn, title, segments):
    """② の1〜2: 手順に分け、手順ごとの道具・資材・注意点・コツを取り出す。"""
    prompt = build_procedure_prompt(title, segments)
    result, raw = call_llm_json(llm_fn, prompt, lambda d: validate_procedure(d, len(segments)))
    if not result.steps:
        # LLMが手順に分けられなかったときは、エラーで止めずに動画全体を1つの手順にする
        result.steps = [ProcedureStep(
            title=title, start_segment=0, end_segment=len(segments) - 1,
            description="説明の文字起こしから手順を区切れなかったため、動画全体を1つの手順にしています。")]
    steps = []
    for step in result.steps:
        covered = segments[step.start_segment:step.end_segment + 1]
        steps.append({
            # 開始・終了時間はLLMに決めさせず、元の文字起こしの時間から決める
            "start": covered[0]["start"], "end": covered[-1]["end"],
            "title": step.title, "description": step.description,
            "tools": step.tools, "materials": step.materials,
            "cautions": step.cautions, "tips": step.tips,
            "segment_ids": [s["segment_id"] for s in covered],
        })
    return steps, raw


def save_steps(conn, video_id, steps, method_version):
    """手順を区間アノテーション（layer='step'）として保存する。手順に付く場面タグ・動画タグも作り直すため消す。"""
    conn.execute(
        "DELETE FROM skill_annotations WHERE video_id=? AND layer IN ('step', 'scene_tag', 'video_tag')",
        (video_id,)
    )
    now = now_str()
    for i, step in enumerate(steps):
        payload = {k: step[k] for k in ("description", "tools", "materials", "cautions", "tips", "segment_ids")}
        payload["index"] = i
        payload["photo"] = step.get("photo")
        conn.execute(
            """INSERT INTO skill_annotations
               (video_id, layer, start, end, label, payload, source, method_version, created)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (video_id, "step", step["start"], step["end"], step["title"],
             json.dumps(payload, ensure_ascii=False), "llm", method_version, now)
        )
    conn.commit()


def get_steps(conn, video_id):
    rows = conn.execute(
        """SELECT id, start, end, label, payload FROM skill_annotations
           WHERE video_id=? AND layer='step' ORDER BY start, id""",
        (video_id,)
    ).fetchall()
    steps = []
    for annotation_id, start, end, label, payload in rows:
        data = json.loads(payload or "{}")
        steps.append({"step_id": annotation_id, "start": start, "end": end, "title": label, **data})
    return steps


def format_clock(seconds):
    seconds = int(seconds or 0)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


PROCEDURE_CSS = """
@page { size: A4; margin: 16mm 14mm; @bottom-center { content: counter(page) " / " counter(pages); font-size: 9pt; color: #666; } }
body { font-family: "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", sans-serif; font-size: 10.5pt; color: #222; line-height: 1.55; }
h1 { font-size: 18pt; margin: 0 0 4px; border-bottom: 3px solid #e67e00; padding-bottom: 4px; }
.meta { color: #555; font-size: 9.5pt; margin-bottom: 12px; }
.summary { border: 1px solid #ccc; padding: 6px 10px; margin-bottom: 12px; font-size: 9.5pt; }
.step { border: 1px solid #bbb; border-radius: 4px; margin-bottom: 10px; page-break-inside: avoid; }
.step-head { background: #fff3e0; padding: 5px 10px; font-weight: bold; font-size: 12pt; }
.step-head .time { float: right; font-weight: normal; font-size: 9.5pt; color: #555; }
.step-body { display: flex; gap: 10px; padding: 8px 10px; }
.photo { width: 62mm; flex: none; text-align: center; }
/* スマホで縦向きに撮った動画でも写真が大きくなりすぎないよう、高さにも上限を付ける */
.photo img { max-width: 100%; max-height: 55mm; border: 1px solid #ccc; }
.text { flex: 1; }
.label { display: inline-block; font-weight: bold; min-width: 4.5em; }
.caution { color: #b00020; }
.tip { color: #0b5394; }
ul { margin: 2px 0 4px 1.2em; padding: 0; }
"""


def render_procedure_html(video, steps):
    """② の3: 決まった書式（HTML）に当てはめる。PDFはこのHTMLから作る。"""
    from html import escape

    def items(values, css=""):
        if not values:
            return "<span>－</span>"
        return "<ul>" + "".join(f'<li class="{css}">{escape(v)}</li>' for v in values) + "</ul>"

    all_tools = sorted({t for s in steps for t in s.get("tools", [])})
    all_materials = sorted({m for s in steps for m in s.get("materials", [])})
    blocks = []
    for i, step in enumerate(steps, start=1):
        photo = step.get("photo")
        photo_html = (f'<div class="photo"><img src="file://{escape(photo)}"></div>'
                      if photo and os.path.exists(photo) else "")
        blocks.append(f"""
<div class="step">
  <div class="step-head">手順{i}　{escape(step['title'])}<span class="time">動画 {format_clock(step['start'])}〜{format_clock(step['end'])}</span></div>
  <div class="step-body">{photo_html}
    <div class="text">
      <p>{escape(step.get('description', ''))}</p>
      <div><span class="label">道具</span>{items(step.get('tools'))}</div>
      <div><span class="label">資材</span>{items(step.get('materials'))}</div>
      <div><span class="label caution">注意点</span>{items(step.get('cautions'), 'caution')}</div>
      <div><span class="label tip">コツ</span>{items(step.get('tips'), 'tip')}</div>
    </div>
  </div>
</div>""")
    explainer = f"説明者: {escape(video['explainer'])}　" if video.get("explainer") else ""
    return f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8"><style>{PROCEDURE_CSS}</style></head>
<body>
<h1>{escape(video['title'])}</h1>
<div class="meta">{explainer}動画の長さ: {format_clock(video.get('duration'))}　作成日: {escape((video.get('created') or '')[:10])}　手順数: {len(steps)}</div>
<div class="summary"><span class="label">道具</span>{escape('、'.join(all_tools) or '－')}<br>
<span class="label">資材</span>{escape('、'.join(all_materials) or '－')}</div>
{''.join(blocks)}
<p class="meta">この手順書は、動画の説明音声からAIが自動で作成しました。内容は動画と合わせて確認してください。</p>
</body></html>"""


def write_procedure_pdf(html_text, pdf_path):
    from weasyprint import HTML
    HTML(string=html_text, base_url="/").write_pdf(pdf_path)
    return pdf_path


def add_step_photos(video_path, steps, folder):
    """各手順の時間帯の中ほどの1コマを、手順書の写真として切り出す（失敗しても手順書は作る）。"""
    os.makedirs(folder, exist_ok=True)
    for i, step in enumerate(steps, start=1):
        try:
            step["photo"] = extract_frame(video_path, (step["start"] + step["end"]) / 2,
                                          os.path.join(folder, f"step_{i}.jpg"))
        except RuntimeError:
            traceback.print_exc()
            step["photo"] = None
    return steps


def step_procedure(ctx, video_id):
    """② 手順書作成（手順分割 → 道具・資材・注意点・コツの抽出 → 写真 → PDF）。"""
    if ctx.llm_fn is None:
        raise RuntimeError("LLMが設定されていません")
    video = get_skill_video(ctx.conn, video_id)
    segments = get_segments(ctx.conn, video_id)
    if not segments:
        raise RuntimeError("文字起こしがありません")
    method_version = f"{getattr(ctx.llm_fn, 'method_version', 'llm')}@{datetime.now().strftime('%Y%m%d%H%M%S')}"
    steps, raw = generate_procedure(ctx.llm_fn, video["title"], segments)
    save_llm_result(ctx.conn, video_id, "procedure", method_version, {"raw": raw, "steps": steps})
    folder = video_dir(video_id, ctx.media_dir)
    add_step_photos(video["media_path"], steps, os.path.join(folder, "steps"))
    save_steps(ctx.conn, video_id, steps, method_version)
    pdf_path = write_procedure_pdf(render_procedure_html(video, get_steps(ctx.conn, video_id)),
                                   os.path.join(folder, "procedure.pdf"))
    update_skill_video(ctx.conn, video_id, pdf_path=pdf_path)


# --- 9. ③ タグ付け ---
import unicodedata
from rapidfuzz import fuzz

TERM_CATEGORIES = ["作業", "道具", "資材", "安全"]
# 難易度の目安（手順の数＋注意点の数）。これ以下なら初級、次の値以下なら中級、それより多ければ上級
SKILL_DIFFICULTY_THRESHOLDS = [("初級", 4), ("中級", 8), ("上級", None)]


class StepTerms(BaseModel):
    step_index: int
    terms: list[dict]


class TagTermsResult(BaseModel):
    steps: list[StepTerms]


def validate_tag_terms(data, n_steps):
    result = TagTermsResult.model_validate(data)
    seen = set()
    for item in result.steps:
        if not 0 <= item.step_index < n_steps:
            raise ValueError(f"step_index {item.step_index} が範囲外です（0〜{n_steps - 1}）")
        if item.step_index in seen:
            raise ValueError(f"step_index {item.step_index} が重複しています")
        seen.add(item.step_index)
        for term in item.terms:
            if not str(term.get("word", "")).strip():
                raise ValueError("word が空です")
            if term.get("category") not in TERM_CATEGORIES:
                raise ValueError(f"category は {TERM_CATEGORIES} のいずれかにしてください: {term.get('category')}")
    return result


def build_tag_terms_prompt(title, steps, segments):
    by_id = {s["segment_id"]: s["text"] for s in segments}
    blocks = []
    for i, step in enumerate(steps):
        text = "".join(by_id.get(sid, "") for sid in step.get("segment_ids", []))
        blocks.append(f"[{i}] {step['title']}\n{text}")
    return f"""# task: tag_terms
建設現場の作業説明動画の文字起こしを、手順ごとに示します。動画を探しやすくするためのタグの元になる言葉を、手順ごとに抜き出してください。

- 抜き出すのは、作業名（例: 型枠組立、配筋）、道具（例: インパクトドライバー）、資材（例: コンパネ）、安全に関わる言葉（例: 墜落防止）だけ
- category は「作業」「道具」「資材」「安全」のいずれか
- 話された言葉をそのまま（略称・現場での呼び方でもよい）書く。言い換えや一般的すぎる言葉（作業、道具、ここ など）は入れない
- 該当する言葉が無い手順は terms を空の配列にする

動画のタイトル: {title}

手順と文字起こし:
{chr(10).join(blocks)}

次の形のJSONだけを返してください:
{{"steps": [{{"step_index": 0, "terms": [{{"word": "...", "category": "道具"}}]}}]}}
"""


def normalize_term(text):
    """
    全角・半角、大文字・小文字、空白や記号の違いをそろえて比べるための形にする。
    カタカナ語の末尾の長音（ドライバ／ドライバー）も表記ゆれとして取り除く。
    """
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[―‐—–]", "ー", text)
    text = re.sub(r"[\s・「」『』()（）\[\]、。,.]", "", text)
    return re.sub(r"ー+$", "", text)


def match_tag(word, tags, threshold=SKILL_TAG_FUZZY_THRESHOLD):
    """
    ③ の2: タグ一覧と照合する。完全一致（名前）→ 別名一致 → 文字列の類似度が閾値以上 →
    言葉の中に名前・別名が入っている、の順。
    難易度タグは言葉からは付けない。見つからなければ None。
    """
    key = normalize_term(word)
    if not key:
        return None
    candidates = [t for t in tags if t["category"] != "難易度"]
    for tag in candidates:
        if normalize_term(tag["name"]) == key:
            return {"tag": tag, "method": "name", "score": 100.0}
    for tag in candidates:
        if key in (normalize_term(a) for a in tag["aliases"]):
            return {"tag": tag, "method": "alias", "score": 100.0}
    # 表記ゆれ（送り仮名・誤認識など）は、文字列全体の類似度で比べる。言葉の方が短く、
    # タグ名に含まれるだけの場合（コンクリート ⊂ コンクリート打設）は意味が広いので一致させない。
    best = None
    for tag in candidates:
        for name in [tag["name"]] + tag["aliases"]:
            other = normalize_term(name)
            if key in other:
                continue
            score = fuzz.ratio(key, other)
            if score >= threshold and (best is None or score > best["score"]):
                best = {"tag": tag, "method": "fuzzy", "score": round(score, 1)}
    if best:
        return best
    # 最後に、言葉の中にタグの名前・別名がそのまま入っている場合（型枠の建て込み ⊃ 建て込み）は、
    # より詳しい言い方なので、そのタグとみなす。複数入っていれば一番長い（具体的な）名前を採る
    # （「型枠」より「建て込み」）。
    best_len = 0
    for tag in candidates:
        for name in [tag["name"]] + tag["aliases"]:
            other = normalize_term(name)
            if len(other) >= 2 and other in key and len(other) > best_len:
                best, best_len = {"tag": tag, "method": "partial", "score": 100.0}, len(other)
    return best
    for tag in candidates:
        for name in [tag["name"]] + tag["aliases"]:
            other = normalize_term(name)
            if key in other:
                continue
            score = fuzz.ratio(key, other)
            if score >= threshold and (best is None or score > best["score"]):
                best = {"tag": tag, "method": "fuzzy", "score": round(score, 1)}
    return best


def judge_difficulty(steps, thresholds=SKILL_DIFFICULTY_THRESHOLDS):
    """③ の4（任意）: 手順の数と注意点の数から、初級・中級・上級を決める。"""
    score = len(steps) + sum(len(s.get("cautions", [])) for s in steps)
    for name, limit in thresholds:
        if limit is None or score <= limit:
            return name, score


def add_tag_candidate(conn, video_id, word, category):
    """
    タグ一覧に無い言葉を「新タグ候補」として残す（自動では登録しない）。
    同じ言葉が保留中・却下済みの候補として既にあれば、重ねて登録しない。
    """
    key = normalize_term(word)
    for existing_word, status in conn.execute("SELECT word, status FROM skill_tag_candidates").fetchall():
        if normalize_term(existing_word) == key and status in ("pending", "rejected"):
            return False
    conn.execute(
        "INSERT INTO skill_tag_candidates (video_id, word, category, status, created) VALUES (?,?,?,?,?)",
        (video_id, word.strip(), category, "pending", now_str())
    )
    return True


def assign_tags(conn, video_id, steps, step_terms, method_version, difficulty=SKILL_DIFFICULTY_TAG,
                threshold=SKILL_TAG_FUZZY_THRESHOLD):
    """
    ③ の2〜5: 言葉をタグ一覧と照合し、手順の時間帯に場面タグを付け、場面タグをまとめて動画タグにする。
    step_terms: {手順の番号: [{"word", "category"}, ...]}
    """
    tags = list_skill_tags(conn)
    conn.execute("DELETE FROM skill_annotations WHERE video_id=? AND layer IN ('scene_tag', 'video_tag')", (video_id,))
    now = now_str()
    video_tag_ids = {}
    new_candidates = 0
    for i, step in enumerate(steps):
        matched = {}
        for term in step_terms.get(i, []):
            m = match_tag(term["word"], tags, threshold)
            if m is None:
                new_candidates += add_tag_candidate(conn, video_id, term["word"], term.get("category"))
                continue
            entry = matched.setdefault(m["tag"]["tag_id"], {"tag": m["tag"], "words": [], "method": m["method"]})
            entry["words"].append({"word": term["word"], "method": m["method"], "score": m["score"]})
        for tag_id, entry in matched.items():
            conn.execute(
                """INSERT INTO skill_annotations
                   (video_id, layer, start, end, label, tag_id, parent_id, payload, source, method_version, created)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (video_id, "scene_tag", step["start"], step["end"], entry["tag"]["name"], tag_id, step["step_id"],
                 json.dumps({"words": entry["words"]}, ensure_ascii=False), "llm", method_version, now)
            )
            video_tag_ids[tag_id] = {"tag": entry["tag"], "source": "scene"}

    if difficulty and steps:
        level, score = judge_difficulty(steps)
        level_tag = next((t for t in tags if t["category"] == "難易度" and t["name"] == level), None)
        if level_tag:
            video_tag_ids[level_tag["tag_id"]] = {"tag": level_tag, "source": "rule", "score": score}

    for tag_id, entry in video_tag_ids.items():
        conn.execute(
            """INSERT INTO skill_annotations
               (video_id, layer, label, tag_id, payload, source, method_version, created)
               VALUES (?,?,?,?,?,?,?,?)""",
            (video_id, "video_tag", entry["tag"]["name"], tag_id,
             json.dumps({k: v for k, v in entry.items() if k != "tag"}, ensure_ascii=False),
             "rule" if entry["source"] == "rule" else "llm", method_version, now)
        )
    conn.commit()
    return {"video_tags": len(video_tag_ids), "new_candidates": new_candidates}


def get_scene_tags(conn, video_id):
    rows = conn.execute(
        """SELECT a.id, a.start, a.end, a.tag_id, a.parent_id, t.name, t.category
           FROM skill_annotations a JOIN skill_tags t ON t.id = a.tag_id
           WHERE a.video_id=? AND a.layer='scene_tag' ORDER BY a.start, t.category, t.name""",
        (video_id,)
    ).fetchall()
    return [{"scene_tag_id": r[0], "start": r[1], "end": r[2], "tag_id": r[3], "step_id": r[4],
             "name": r[5], "category": r[6]} for r in rows]


def get_video_tags(conn, video_id):
    rows = conn.execute(
        """SELECT t.id, t.name, t.category FROM skill_annotations a JOIN skill_tags t ON t.id = a.tag_id
           WHERE a.video_id=? AND a.layer='video_tag' ORDER BY t.category, t.name""",
        (video_id,)
    ).fetchall()
    return [{"tag_id": r[0], "name": r[1], "category": r[2]} for r in rows]


def step_tagging(ctx, video_id):
    """③ タグ付け（言葉の抽出 → タグ一覧と照合 → 場面タグ → 難易度 → 動画タグ）。"""
    if ctx.llm_fn is None:
        raise RuntimeError("LLMが設定されていません")
    video = get_skill_video(ctx.conn, video_id)
    steps = get_steps(ctx.conn, video_id)
    if not steps:
        raise RuntimeError("手順がありません（先に手順書を作る必要があります）")
    prompt = build_tag_terms_prompt(video["title"], steps, get_segments(ctx.conn, video_id))
    result, raw = call_llm_json(ctx.llm_fn, prompt, lambda d: validate_tag_terms(d, len(steps)))
    step_terms = {item.step_index: list(item.terms) for item in result.steps}
    # 手順書で取り出した道具・資材も、タグの元になる言葉として使う
    for i, step in enumerate(steps):
        extra = [{"word": w, "category": "道具"} for w in step.get("tools", [])]
        extra += [{"word": w, "category": "資材"} for w in step.get("materials", [])]
        step_terms.setdefault(i, []).extend(extra)
    method_version = f"{getattr(ctx.llm_fn, 'method_version', 'llm')}@{datetime.now().strftime('%Y%m%d%H%M%S')}"
    save_llm_result(ctx.conn, video_id, "tag_terms", method_version, {"raw": raw})
    assign_tags(ctx.conn, video_id, steps, step_terms, method_version, difficulty=ctx.difficulty)


SKILL_STEPS = [("media", step_media), ("transcribe", step_transcribe), ("procedure", step_procedure),
               ("tagging", step_tagging)]
SKILL_STEP_NAMES = [name for name, _ in SKILL_STEPS]


# --- 10. ④ 閲覧（作業の種類・タグでの絞り込み）とタグ一覧の管理 ---
def descendant_tag_ids(conn, tag_id):
    """そのタグと、下位の分類（子・孫…）のタグのID。"""
    rows = conn.execute(
        """WITH RECURSIVE sub(id) AS (
               SELECT ? UNION SELECT t.id FROM skill_tags t JOIN sub ON t.parent_id = sub.id)
           SELECT id FROM sub""",
        (tag_id,)
    ).fetchall()
    return [r[0] for r in rows]


def search_videos(conn, work_tag_id=None, tag_ids=(), include_unfinished=False):
    """
    ④ の1〜2: 作業の種類が選ばれたら、その分類（下位の分類を含む）の動画タグを持つ動画。
    タグが選ばれたら、選ばれたタグをすべて持つ動画だけ（AND検索）。
    """
    sql = f"SELECT {', '.join(VIDEO_COLUMNS)} FROM skill_videos WHERE 1=1"
    params = []
    if not include_unfinished:
        sql += " AND status='done'"
    if work_tag_id is not None:
        ids = descendant_tag_ids(conn, work_tag_id)
        sql += f""" AND id IN (SELECT video_id FROM skill_annotations
                               WHERE layer='video_tag' AND tag_id IN ({','.join('?' * len(ids))}))"""
        params += ids
    tag_ids = sorted(set(tag_ids))
    if tag_ids:
        sql += f""" AND id IN (SELECT video_id FROM skill_annotations
                               WHERE layer='video_tag' AND tag_id IN ({','.join('?' * len(tag_ids))})
                               GROUP BY video_id HAVING COUNT(DISTINCT tag_id) = ?)"""
        params += tag_ids + [len(tag_ids)]
    rows = conn.execute(sql + " ORDER BY created DESC, id DESC", params).fetchall()
    return [dict(zip(VIDEO_COLUMNS, r)) for r in rows]


class TagBody(BaseModel):
    name: str
    category: str
    parent_id: Optional[int] = None
    aliases: list[str] = []


def clean_tag_body(conn, body, tag_id=None):
    """タグの追加・編集の内容を確かめて整える。問題があれば ValueError。"""
    name = body.name.strip()
    if not name:
        raise ValueError("タグの名前を入力してください")
    if body.category not in TAG_CATEGORIES:
        raise ValueError(f"分類は {'・'.join(TAG_CATEGORIES)} のいずれかにしてください")
    same = conn.execute("SELECT id FROM skill_tags WHERE name=?", (name,)).fetchone()
    if same and same[0] != tag_id:
        raise ValueError(f"「{name}」は既にあります")
    if body.parent_id is not None:
        parent = conn.execute("SELECT category FROM skill_tags WHERE id=?", (body.parent_id,)).fetchone()
        if parent is None:
            raise ValueError("親のタグが見つかりません")
        if parent[0] != body.category:
            raise ValueError("親には同じ分類のタグを選んでください")
        if tag_id is not None and body.parent_id in descendant_tag_ids(conn, tag_id):
            raise ValueError("自分自身や下位のタグを親にはできません")
    aliases = []
    for a in body.aliases:
        a = a.strip()
        if a and a != name and a not in aliases:
            aliases.append(a)
    return name, body.category, body.parent_id, aliases


def save_tag(conn, body, tag_id=None):
    name, category, parent_id, aliases = clean_tag_body(conn, body, tag_id)
    aliases_json = json.dumps(aliases, ensure_ascii=False)
    if tag_id is None:
        tag_id = conn.execute(
            "INSERT INTO skill_tags (name, category, parent_id, aliases, created) VALUES (?,?,?,?,?)",
            (name, category, parent_id, aliases_json, now_str())
        ).lastrowid
    else:
        conn.execute("UPDATE skill_tags SET name=?, category=?, parent_id=?, aliases=? WHERE id=?",
                     (name, category, parent_id, aliases_json, tag_id))
        # 名前を変えたら、付いている場面タグ・動画タグの表示名もそろえる
        conn.execute("UPDATE skill_annotations SET label=? WHERE tag_id=?", (name, tag_id))
    conn.commit()
    return tag_id


def delete_tag(conn, tag_id):
    """下位のタグがあるときは消さない（先に下位のタグを消すか付け替える）。付いている場面タグ・動画タグも消す。"""
    if conn.execute("SELECT COUNT(*) FROM skill_tags WHERE parent_id=?", (tag_id,)).fetchone()[0]:
        raise ValueError("下位のタグがあるため削除できません")
    conn.execute("DELETE FROM skill_annotations WHERE tag_id=?", (tag_id,))
    conn.execute("DELETE FROM skill_tags WHERE id=?", (tag_id,))
    conn.commit()


def tag_video_counts(conn):
    return dict(conn.execute(
        """SELECT a.tag_id, COUNT(DISTINCT a.video_id) FROM skill_annotations a
           JOIN skill_videos v ON v.id = a.video_id
           WHERE a.layer='video_tag' AND v.status='done' GROUP BY a.tag_id"""
    ).fetchall())


def list_tag_candidates(conn, status="pending"):
    rows = conn.execute(
        """SELECT c.id, c.word, c.category, c.status, c.created, c.video_id, v.title
           FROM skill_tag_candidates c LEFT JOIN skill_videos v ON v.id = c.video_id
           WHERE c.status=? ORDER BY c.id""",
        (status,)
    ).fetchall()
    return [{"candidate_id": r[0], "word": r[1], "category": r[2], "status": r[3], "created": r[4],
             "video_id": r[5], "video_title": r[6]} for r in rows]


class AdoptBody(BaseModel):
    mode: str                      # "new"（新しいタグにする）/ "alias"（既存タグの別名にする）
    name: Optional[str] = None
    category: Optional[str] = None
    parent_id: Optional[int] = None
    tag_id: Optional[int] = None


def adopt_candidate(conn, candidate_id, body):
    row = conn.execute("SELECT word, category, status FROM skill_tag_candidates WHERE id=?",
                       (candidate_id,)).fetchone()
    if row is None:
        raise LookupError("候補が見つかりません")
    word, category, status = row
    if status != "pending":
        raise ValueError("この候補は処理済みです")
    if body.mode == "new":
        tag_id = save_tag(conn, TagBody(name=body.name or word, category=body.category or category or "作業",
                                        parent_id=body.parent_id,
                                        aliases=[word] if body.name and body.name != word else []))
    elif body.mode == "alias":
        tag = conn.execute("SELECT id, name, category, parent_id, aliases FROM skill_tags WHERE id=?",
                           (body.tag_id,)).fetchone()
        if tag is None:
            raise ValueError("別名を追加するタグを選んでください")
        current = tag_row_to_dict(tag)
        tag_id = save_tag(conn, TagBody(name=current["name"], category=current["category"],
                                        parent_id=current["parent_id"], aliases=current["aliases"] + [word]),
                          tag_id=current["tag_id"])
    else:
        raise ValueError("mode は new か alias を指定してください")
    conn.execute("UPDATE skill_tag_candidates SET status='adopted' WHERE id=?", (candidate_id,))
    conn.commit()
    return tag_id


def rematch_video_tags(ctx, video_id):
    """
    タグ一覧を変えた後に、保存済みの抽出結果（LLMの出力）を使ってタグを付け直す。
    LLMはもう一度呼ばない（費用と時間がかからない）。
    """
    row = ctx.conn.execute(
        "SELECT result FROM skill_llm_results WHERE video_id=? AND kind='tag_terms' ORDER BY id DESC LIMIT 1",
        (video_id,)
    ).fetchone()
    steps = get_steps(ctx.conn, video_id)
    if row is None or not steps:
        return False
    result = validate_tag_terms(parse_json_text(json.loads(row[0])["raw"]), len(steps))
    step_terms = {item.step_index: list(item.terms) for item in result.steps}
    for i, step in enumerate(steps):
        step_terms.setdefault(i, []).extend(
            [{"word": w, "category": "道具"} for w in step.get("tools", [])]
            + [{"word": w, "category": "資材"} for w in step.get("materials", [])])
    assign_tags(ctx.conn, video_id, steps, step_terms, f"rematch@{datetime.now().strftime('%Y%m%d%H%M%S')}",
                difficulty=ctx.difficulty)
    return True


# --- 11. API（/skill/api/...） ---
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

ALLOWED_VIDEO_EXT = re.compile(r"^\.[a-z0-9]{1,5}$")


def video_summary(video):
    """画面に返す動画の情報（Drive上のファイルパスは返さない）。"""
    keys = ["id", "title", "explainer", "filename", "duration", "status", "failed_step", "error", "created", "updated"]
    summary = {k: video[k] for k in keys}
    summary["video_id"] = summary.pop("id")
    summary["has_media"] = bool(video["media_path"])
    summary["has_pdf"] = bool(video["pdf_path"])
    return summary


def public_step(video_id, step):
    """画面に返す手順（写真はDrive上のパスでなく、取得用のURLにする）。"""
    step = dict(step)
    number = step.get("index", 0) + 1
    step["photo_url"] = f"/skill/api/videos/{video_id}/steps/{number}.jpg" if step.pop("photo", None) else None
    return step


def create_skill_router(ctx):
    router = APIRouter()

    def require_video(video_id):
        video = get_skill_video(ctx.conn, video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="動画が見つかりません")
        return video

    @router.get("/skill", response_class=HTMLResponse)
    def api_skill_page():
        # スマホからは「ngrokのURL/skill」で開く（同じ所から配信すると、動画や字幕も
        # ngrokの警告ページに止められずに読み込める）。画面はこのセルの中のHTML（12.）
        return HTMLResponse(SKILL_PAGE_HTML)

    @router.get("/")
    def api_root():
        return HTMLResponse("<meta charset='utf-8'><p>技能伝承動画の画面は <a href='/skill'>/skill</a> です。</p>")

    @router.post("/skill/api/videos")
    # 対話研究の/uploadと同じく、重い処理を待たずにIDをすぐ返し、処理は裏で行う
    # （ngrok無料枠のタイムアウト対策）。画面は /skill/api/videos/{id} で状態を確認する。
    def api_skill_upload(file: UploadFile = File(...), title: str = Form(...), explainer: str = Form("")):
        title = title.strip()
        if not title:
            raise HTTPException(status_code=400, detail="タイトルを入力してください")
        ext = os.path.splitext(file.filename or "")[1].lower()
        if not ALLOWED_VIDEO_EXT.match(ext):
            ext = ".mp4"
        video_id = create_skill_video(ctx.conn, title, explainer.strip(), file.filename)
        # アップロードされたファイル名はパスに使わない（../ などを含められるため）
        with open(os.path.join(video_dir(video_id, ctx.media_dir), "original" + ext), "wb") as f:
            shutil.copyfileobj(file.file, f)
        start_skill_pipeline(ctx, video_id)
        return {"video_id": video_id, "status": get_skill_video(ctx.conn, video_id)["status"]}

    @router.get("/skill/api/videos")
    def api_skill_videos(work_tag: Optional[int] = None, tags: str = "", include_unfinished: bool = False):
        try:
            tag_ids = [int(t) for t in tags.split(",") if t.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="tags はタグIDのカンマ区切りで指定してください")
        videos = search_videos(ctx.conn, work_tag, tag_ids, include_unfinished)
        result = []
        facet_counts = {}
        for v in videos:
            video_tags = get_video_tags(ctx.conn, v["id"])
            for t in video_tags:
                facet_counts.setdefault(t["tag_id"], {**t, "count": 0})["count"] += 1
            thumb = os.path.join(video_dir(v["id"], ctx.media_dir), "thumbnail.jpg")
            result.append({**video_summary(v), "video_tags": video_tags,
                           "thumbnail_url": f"/skill/api/videos/{v['id']}/thumbnail.jpg" if os.path.exists(thumb) else None})
        # 絞り込みボタン用: いま表示している動画に付いているタグと、その動画数
        facets = sorted(facet_counts.values(), key=lambda t: (TAG_CATEGORIES.index(t["category"])
                                                              if t["category"] in TAG_CATEGORIES else 99, -t["count"], t["name"]))
        return {"videos": result, "facets": facets}

    @router.get("/skill/api/tags")
    def api_skill_tags():
        counts = tag_video_counts(ctx.conn)
        return {"categories": TAG_CATEGORIES,
                "tags": [{**t, "video_count": counts.get(t["tag_id"], 0)} for t in list_skill_tags(ctx.conn)]}

    @router.post("/skill/api/tags")
    def api_skill_tag_create(body: TagBody):
        try:
            return {"tag_id": save_tag(ctx.conn, body)}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.put("/skill/api/tags/{tag_id}")
    def api_skill_tag_update(tag_id: int, body: TagBody):
        if ctx.conn.execute("SELECT id FROM skill_tags WHERE id=?", (tag_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="タグが見つかりません")
        try:
            return {"tag_id": save_tag(ctx.conn, body, tag_id)}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.delete("/skill/api/tags/{tag_id}")
    def api_skill_tag_delete(tag_id: int):
        if ctx.conn.execute("SELECT id FROM skill_tags WHERE id=?", (tag_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="タグが見つかりません")
        try:
            delete_tag(ctx.conn, tag_id)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"status": "deleted", "tag_id": tag_id}

    @router.get("/skill/api/tag_candidates")
    def api_skill_tag_candidates(status: str = "pending"):
        return {"candidates": list_tag_candidates(ctx.conn, status)}

    @router.post("/skill/api/tag_candidates/{candidate_id}/adopt")
    def api_skill_candidate_adopt(candidate_id: int, body: AdoptBody):
        try:
            return {"status": "adopted", "tag_id": adopt_candidate(ctx.conn, candidate_id, body)}
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/skill/api/tag_candidates/{candidate_id}/reject")
    def api_skill_candidate_reject(candidate_id: int):
        cur = ctx.conn.execute("UPDATE skill_tag_candidates SET status='rejected' WHERE id=? AND status='pending'",
                               (candidate_id,))
        ctx.conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="保留中の候補が見つかりません")
        return {"status": "rejected"}

    @router.post("/skill/api/rematch")
    # タグ一覧を変えた後、全動画のタグを付け直す（保存済みの抽出結果を使い、LLMは呼ばない）
    def api_skill_rematch():
        done = [v["id"] for v in search_videos(ctx.conn) if v["id"] not in ctx.running]
        updated = sum(1 for video_id in done if rematch_video_tags(ctx, video_id))
        return {"status": "done", "videos": updated}

    @router.get("/skill/api/videos/{video_id}")
    def api_skill_video(video_id: int):
        video = require_video(video_id)
        return {**video_summary(video), "segments": get_segments(ctx.conn, video_id),
                "steps": [public_step(video_id, s) for s in get_steps(ctx.conn, video_id)],
                "scene_tags": get_scene_tags(ctx.conn, video_id),
                "video_tags": get_video_tags(ctx.conn, video_id)}

    @router.post("/skill/api/videos/{video_id}/retry")
    def api_skill_retry(video_id: int):
        video = require_video(video_id)
        if video_id in ctx.running:
            raise HTTPException(status_code=409, detail="処理中です")
        if video["status"] != "error":
            raise HTTPException(status_code=409, detail="失敗した動画だけ再実行できます")
        start_skill_pipeline(ctx, video_id, start_step=video["failed_step"] or SKILL_STEP_NAMES[0])
        return {"video_id": video_id, "status": get_skill_video(ctx.conn, video_id)["status"]}

    @router.get("/skill/api/videos/{video_id}/media")
    def api_skill_media(video_id: int):
        # FileResponseはRangeリクエストに対応しているので、スマホでもシークできる
        video = require_video(video_id)
        if not video["media_path"] or not os.path.exists(video["media_path"]):
            raise HTTPException(status_code=404, detail="動画の準備ができていません")
        return FileResponse(video["media_path"], media_type="video/mp4")

    @router.get("/skill/api/videos/{video_id}/thumbnail.jpg")
    def api_skill_thumbnail(video_id: int):
        require_video(video_id)
        path = os.path.join(video_dir(video_id, ctx.media_dir), "thumbnail.jpg")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="サムネイルがありません")
        return FileResponse(path, media_type="image/jpeg")

    @router.get("/skill/api/videos/{video_id}/procedure.pdf")
    def api_skill_procedure_pdf(video_id: int):
        video = require_video(video_id)
        if not video["pdf_path"] or not os.path.exists(video["pdf_path"]):
            raise HTTPException(status_code=404, detail="手順書はまだできていません")
        return FileResponse(video["pdf_path"], media_type="application/pdf",
                            filename=f"手順書_{video['title']}.pdf")

    @router.get("/skill/api/videos/{video_id}/steps/{step_number}.jpg")
    def api_skill_step_photo(video_id: int, step_number: int):
        require_video(video_id)
        steps = get_steps(ctx.conn, video_id)
        if not 1 <= step_number <= len(steps) or not steps[step_number - 1].get("photo") \
                or not os.path.exists(steps[step_number - 1]["photo"]):
            raise HTTPException(status_code=404, detail="写真がありません")
        return FileResponse(steps[step_number - 1]["photo"], media_type="image/jpeg")

    @router.get("/skill/api/videos/{video_id}/subtitles.vtt")
    def api_skill_subtitles(video_id: int):
        require_video(video_id)
        return Response(build_webvtt(get_segments(ctx.conn, video_id)), media_type="text/vtt; charset=utf-8")

    return router


def register_skill_transfer(app, conn=None, whisper_model=None, openai_client=None, mask_fn=None,
                            llm_fn=None, media_dir=None, seed_csv=None, run_in_background=True):
    """
    技能伝承の画面とAPIをFastAPIアプリに登録する。
    タグ一覧が空のときだけ初期データを入れる。
    保存先などを省略したときは、呼び出した時点の設定（SKILL_MEDIA_DIR など）を使う。
    """
    media_dir = media_dir or SKILL_MEDIA_DIR
    seed_csv = seed_csv or SKILL_SEED_CSV or None
    if conn is None:
        conn = open_skill_db()
    else:
        init_skill_db(conn)
    if not isinstance(conn, SerializedConnection):
        conn = SerializedConnection(conn)
    os.makedirs(media_dir, exist_ok=True)
    if conn.execute("SELECT COUNT(*) FROM skill_tags").fetchone()[0] == 0:
        print("技能伝承: タグ一覧の初期データを投入しました", seed_skill_tags(conn, seed_csv))
    if llm_fn is None:
        llm_fn = make_llm_fn(openai_client=openai_client)
    if mask_fn is None:
        mask_fn = lambda sentences: extract_mask_words(llm_fn, sentences)
    ctx = SkillContext(conn, media_dir=media_dir, whisper_model=whisper_model, llm_fn=llm_fn,
                       mask_fn=mask_fn, run_in_background=run_in_background)
    ctx.openai_client = openai_client
    app.include_router(create_skill_router(ctx))
    app.state.skill_transfer = ctx
    return ctx


def build_skill_app(whisper_model, openai_client=None, conn=None, **kwargs):
    """技能伝承だけのFastAPIアプリを作る（CORSの設定は対話研究のパイプラインと同じ）。"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    register_skill_transfer(app, conn=conn, whisper_model=whisper_model, openai_client=openai_client, **kwargs)
    return app


# --- 12. 画面（HTML）とタグ一覧の初期データ ---
# 1つのセルで完結させるため、画面とタグ一覧の初期データもこのセルに入れている。
# タグ一覧の初期データは seed/skill_transfer_tags.csv と同じ内容（列: name, category, parent, aliases）。
# 運用で変えたいときは、画面の「タグ管理」で編集する（セルを実行し直しても上書きしない）。
SKILL_SEED_TAGS_CSV = """name,category,parent,aliases
躯体工事,作業,,
型枠工事,作業,躯体工事,型枠|かたわく
型枠組立,作業,型枠工事,型枠の組立|建て込み|建込み
型枠解体,作業,型枠工事,ばらし|解体
鉄筋工事,作業,躯体工事,
配筋,作業,鉄筋工事,鉄筋組立|鉄筋の組立
鉄筋結束,作業,鉄筋工事,結束|結束作業
コンクリート打設,作業,躯体工事,打設|生コン打設|コン打ち
仕上工事,作業,,
内装工事,作業,仕上工事,内装
軽量鉄骨下地,作業,内装工事,LGS|軽鉄下地|軽天
ボード張り,作業,内装工事,石膏ボード張り|PB張り|ボード貼り
左官工事,作業,仕上工事,左官
モルタル塗り,作業,左官工事,モル塗り
タイル張り,作業,仕上工事,タイル貼り|タイル
仮設工事,作業,,
足場組立,作業,仮設工事,足場|足場の組立
墨出し,作業,仮設工事,墨打ち|墨付け
インパクトドライバー,道具,,インパクト|インパクトドライバ
電動丸のこ,道具,,丸のこ|丸ノコ|マルノコ
ハッカー,道具,,結束ハッカー
レベル,道具,,水平器|水準器|水平
墨つぼ,道具,,墨壺|すみつぼ
レーザー墨出し器,道具,,レーザー|墨出し器
コテ,道具,,鏝|左官ごて
バイブレーター,道具,,バイブ|振動機
メジャー,道具,,コンベックス|スケール
石膏ボード,資材,,プラスターボード|PB
コンパネ,資材,,合板|型枠用合板|ベニヤ
セパレーター,資材,,セパ
鉄筋,資材,,異形鉄筋|D13|D10
結束線,資材,,番線|なまし鉄線
軽量鉄骨,資材,,スタッド|ランナー
ビス,資材,,ねじ|ネジ
モルタル,資材,,
墜落防止,安全,,墜落|転落|安全帯|フルハーネス
保護具,安全,,ヘルメット|保護メガネ|手袋
初級,難易度,,
中級,難易度,,
上級,難易度,,
"""

SKILL_PAGE_HTML = r'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>技能伝承動画</title>
    <style>
        :root {
            --accent: #e67e00; --accent-dark: #b86400; --bg: #f6f6f4; --card: #ffffff;
            --text: #222; --sub: #666; --line: #ddd; --danger: #c62828; --tip: #0b5394; --ok: #2e7d32;
        }
        * { box-sizing: border-box; }
        body { margin: 0; font-family: sans-serif; line-height: 1.6; color: var(--text); background: var(--bg); }
        header { position: sticky; top: 0; z-index: 10; background: #333; color: #fff; padding: 8px 12px;
                 display: flex; align-items: center; gap: 8px; }
        header h1 { font-size: 1.05em; margin: 0; flex: 1; white-space: nowrap; }
        header a { color: #fff; text-decoration: none; background: #555; padding: 8px 12px; border-radius: 6px; font-size: 0.9em; }
        header a.upload { background: var(--accent); }
        .container { max-width: 960px; margin: 0 auto; padding: 12px; }
        .section { background: var(--card); padding: 12px; border-radius: 8px; margin-bottom: 12px; border: 1px solid var(--line); }
        h2 { font-size: 1.1em; margin: 0 0 8px; border-left: 5px solid var(--accent); padding-left: 8px; }
        .muted { color: var(--sub); font-size: 0.9em; }
        .error { color: var(--danger); }
        button, .button { font-size: 1em; border: none; border-radius: 8px; padding: 12px 14px; cursor: pointer;
                          background: #e0e0e0; color: var(--text); text-decoration: none; display: inline-block; text-align: center; }
        button.primary, .button.primary { background: var(--accent); color: #fff; font-weight: bold; }
        button.small { padding: 6px 10px; font-size: 0.85em; }
        button:disabled { opacity: 0.5; }
        input[type=text], select { font-size: 1em; padding: 10px; width: 100%; border: 1px solid #bbb; border-radius: 6px; }
        label { display: block; font-weight: bold; margin: 10px 0 4px; }
        .big-buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .big-buttons button, .big-buttons .button { padding: 18px 8px; font-size: 1.05em; }
        .chips { display: flex; flex-wrap: wrap; gap: 6px; }
        .chip { border: 1px solid #bbb; background: #fff; border-radius: 16px; padding: 4px 12px; font-size: 0.9em; cursor: pointer; }
        a.chip { text-decoration: none; color: var(--text); }
        .chip.selected { background: var(--accent); color: #fff; border-color: var(--accent); }
        .chip .cat { font-size: 0.75em; color: var(--sub); margin-right: 4px; }
        .chip.selected .cat { color: #ffe0b2; }

        /* アップロード */
        .file-buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .file-buttons label { margin: 0; padding: 18px 8px; text-align: center; border-radius: 8px; background: #e0e0e0; cursor: pointer; font-weight: bold; }
        .file-buttons label.camera { background: var(--accent); color: #fff; }
        .file-buttons input { display: none; }
        progress { width: 100%; height: 14px; }
        .steps-status { list-style: none; padding: 0; margin: 8px 0; }
        .steps-status li { padding: 6px 8px; border-bottom: 1px solid var(--line); }
        .steps-status li.done::before { content: "✔ "; color: var(--ok); }
        .steps-status li.running { font-weight: bold; color: var(--accent-dark); }
        .steps-status li.running::before { content: "▶ "; }
        .steps-status li.failed { color: var(--danger); font-weight: bold; }
        .steps-status li.failed::before { content: "✖ "; }
        .steps-status li.waiting { color: var(--sub); }
        .steps-status li.waiting::before { content: "・ "; }

        /* 再生 */
        .player-wrap { background: #000; border-radius: 8px; overflow: hidden; }
        video { width: 100%; max-height: 60vh; display: block; background: #000; }
        video::cue { font-size: 1.1em; background: rgba(0,0,0,0.75); }
        .scene-bar { position: relative; height: 26px; background: #444; cursor: pointer; }
        .scene-bar .mark { position: absolute; top: 3px; bottom: 3px; background: var(--accent); border-left: 2px solid #fff;
                           color: #fff; font-size: 0.7em; overflow: hidden; white-space: nowrap; padding-left: 2px; }
        .scene-bar .mark:nth-child(even) { background: #ffa726; }
        .scene-bar .playhead { position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; pointer-events: none; }
        .player-tools { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
        .step-card { border: 1px solid var(--line); border-radius: 8px; margin-bottom: 8px; overflow: hidden; }
        .step-card.current { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent) inset; }
        .step-head { display: flex; align-items: center; gap: 8px; background: #fff3e0; padding: 8px 10px; cursor: pointer; }
        .step-head .time { font-family: monospace; color: var(--accent-dark); font-weight: bold; }
        .step-head .title { font-weight: bold; flex: 1; }
        .step-body { padding: 8px 10px; font-size: 0.95em; }
        .step-body .label { font-weight: bold; margin-right: 4px; }
        .step-body .caution { color: var(--danger); }
        .step-body .tip { color: var(--tip); }
        .transcript p { margin: 0; padding: 4px 0; border-bottom: 1px dotted var(--line); cursor: pointer; }
        .transcript .time { font-family: monospace; color: var(--sub); margin-right: 6px; }
        details summary { cursor: pointer; font-weight: bold; padding: 4px 0; }

        /* ホーム・一覧 */
        .breadcrumb { font-size: 0.9em; margin-bottom: 8px; }
        .breadcrumb a { color: var(--accent-dark); }
        .work-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .work-grid a { display: block; background: #fff; border: 2px solid var(--accent); color: var(--text); text-decoration: none;
                       border-radius: 10px; padding: 16px 8px; text-align: center; font-weight: bold; font-size: 1.05em; }
        .work-grid a .more { display: block; font-size: 0.75em; color: var(--sub); font-weight: normal; }
        .filter-group { margin-bottom: 6px; }
        .filter-group .group-label { font-size: 0.8em; color: var(--sub); margin-bottom: 2px; }
        .video-card { display: flex; gap: 10px; padding: 8px 0; border-bottom: 1px solid var(--line); text-decoration: none; color: var(--text); }
        .video-card img, .video-card .noimg { width: 120px; height: 68px; object-fit: cover; border-radius: 6px; background: #ccc; flex: none; }
        .video-card .info { flex: 1; min-width: 0; }
        .video-card .title { font-weight: bold; }
        .video-card .chips .chip { font-size: 0.75em; padding: 1px 8px; cursor: default; }
        .status-badge { font-size: 0.75em; padding: 1px 8px; border-radius: 10px; background: #eee; }
        .status-badge.error { background: #ffebee; color: var(--danger); }

        /* タグ管理（PC想定） */
        table { width: 100%; border-collapse: collapse; font-size: 0.92em; }
        th, td { border-bottom: 1px solid var(--line); padding: 6px; text-align: left; vertical-align: top; }
        th { background: #fafafa; }
        td.actions { white-space: nowrap; }
        .form-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 8px; align-items: end; }
        .inline-form { background: #fff8e1; padding: 8px; border-radius: 6px; margin-top: 6px; }
        .notice { background: #e8f5e9; padding: 8px; border-radius: 6px; margin-bottom: 8px; }
    </style>
</head>
<body>
    <header>
        <h1>🦺 技能伝承動画</h1>
        <a href="#/">ホーム</a>
        <a href="#/upload" class="upload">撮影・投稿</a>
    </header>
    <div class="container" id="app"></div>

    <script>
        // Colab の ngrok URL の末尾に /skill を付けて開く（配信元をそのまま使う）。
        // この画面をファイルとして保存してPCで直接開く場合は、ここに ngrok URL を貼り付ける。
        const BASE_URL = location.protocol.startsWith("http") ? location.origin : "https://xxxx.ngrok-free.dev";
        const NGROK_HEADERS = { 'ngrok-skip-browser-warning': 'true' };

        const STEP_LABELS = [
            ["media", "動画の変換"],
            ["transcribe", "文字起こし・字幕"],
            ["procedure", "手順書"],
            ["tagging", "タグ付け"],
        ];
        const app = document.getElementById('app');
        let pollTimer = null;

        function sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }

        // 文字起こしやタグ名は利用者の入力・AIの出力なので、必ずエスケープしてから表示する
        function esc(text) {
            return String(text ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
        }

        function clock(seconds) {
            const s = Math.floor(seconds || 0);
            return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
        }

        async function api(path, options = {}) {
            const response = await fetch(`${BASE_URL}${path}`, {
                ...options,
                headers: { ...NGROK_HEADERS, ...(options.body && !(options.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}), ...(options.headers || {}) },
            });
            const text = await response.text();
            const data = text ? JSON.parse(text) : {};
            if (!response.ok) {
                throw new Error(data.detail || `サーバーエラー(${response.status})`);
            }
            return data;
        }

        // --- 画面の切り替え（#/〜 で画面を表す。タップで戻る・進むができるように） ---
        function route() {
            clearTimeout(pollTimer);
            const [path, query] = location.hash.replace(/^#/, "").split("?");
            const params = new URLSearchParams(query || "");
            const parts = (path || "/").split("/").filter(Boolean);
            window.scrollTo(0, 0);
            if (parts[0] === "upload") return renderUpload();
            if (parts[0] === "video" && parts[1]) return renderVideo(Number(parts[1]), params);
            if (parts[0] === "list") return renderList(params);
            if (parts[0] === "tags") return renderTags();
            return renderHome(params);
        }
        window.addEventListener("hashchange", route);

        function showError(err) {
            app.innerHTML = `<div class="section error">エラー: ${esc(err.message)}</div>`;
        }

        // --- タグ一覧（作業の種類の階層をたどるために使う） ---
        async function loadTags() {
            const data = await api("/skill/api/tags");
            const byId = Object.fromEntries(data.tags.map(t => [t.tag_id, t]));
            const children = id => data.tags.filter(t => t.parent_id === id);
            const ancestors = id => {
                const chain = [];
                for (let t = byId[id]; t; t = byId[t.parent_id]) chain.unshift(t);
                return chain;
            };
            return { ...data, byId, children, ancestors };
        }

        function breadcrumb(tags, workId, linkLast) {
            const chain = workId ? tags.ancestors(workId) : [];
            const items = [`<a href="#/">ホーム</a>`].concat(chain.map((t, i) =>
                i === chain.length - 1 && !linkLast ? esc(t.name) : `<a href="#/?work=${t.tag_id}">${esc(t.name)}</a>`));
            return `<div class="breadcrumb">${items.join(" ＞ ")}</div>`;
        }

        // --- ホーム（作業の種類のボタンを、階層をタップでたどる） ---
        async function renderHome(params) {
            let tags;
            try {
                tags = await loadTags();
            } catch (err) {
                return showError(err);
            }
            const workId = params.get("work") ? Number(params.get("work")) : null;
            const current = workId ? tags.byId[workId] : null;
            const items = tags.children(workId).filter(t => t.category === "作業");
            const buttons = items.map(t => {
                const hasChildren = tags.children(t.tag_id).length > 0;
                // 下位の分類があればさらにたどり、無ければその作業の動画一覧へ
                const href = hasChildren ? `#/?work=${t.tag_id}` : `#/list?work=${t.tag_id}`;
                return `<a href="${href}" class="work-button">${esc(t.name)}<span class="more">${hasChildren ? "さらに選ぶ ▶" : "動画を見る"}</span></a>`;
            }).join("");
            const allLink = current
                ? `<a class="button primary" href="#/list?work=${current.tag_id}">「${esc(current.name)}」の動画をすべて見る</a>`
                : `<a class="button primary" href="#/list">すべての動画を見る</a>`;
            app.innerHTML = `
                <div class="section">
                    ${current ? breadcrumb(tags, workId, false) : ""}
                    <h2>${current ? esc(current.name) : "作業の種類を選ぶ"}</h2>
                    <div class="work-grid">${buttons || '<p class="muted">下位の分類はありません</p>'}</div>
                    <p>${allLink}</p>
                </div>
                ${current ? "" : `
                <div class="section">
                    <h2>動画を増やす</h2>
                    <p class="muted">熟練者が作業しながら説明する様子を撮影して投稿すると、字幕・手順書・タグが自動で作られます。</p>
                    <a class="button primary" href="#/upload">📹 撮影・投稿する</a>
                    <p class="muted" style="margin-top:12px"><a href="#/tags">タグ一覧の管理（PC向け）</a></p>
                </div>`}`;
        }

        // --- 動画一覧（上部のタグボタンで絞り込み。複数選ぶと、すべてを持つ動画だけ） ---
        function listHash(workId, tagIds) {
            const q = new URLSearchParams();
            if (workId) q.set("work", workId);
            if (tagIds.length) q.set("tags", tagIds.join(","));
            const qs = q.toString();
            return `#/list${qs ? "?" + qs : ""}`;
        }

        function videoCard(v) {
            const thumb = v.thumbnail_url ? `<img src="${BASE_URL}${v.thumbnail_url}" alt="" loading="lazy">` : `<div class="noimg"></div>`;
            const chips = (v.video_tags || []).slice(0, 6).map(t => `<span class="chip">${esc(t.name)}</span>`).join("");
            return `<a class="video-card" href="#/video/${v.video_id}">${thumb}
                <div class="info"><div class="title">${esc(v.title)}</div>
                <div class="muted">${clock(v.duration)}${v.explainer ? "　" + esc(v.explainer) : ""}</div>
                <div class="chips">${chips}</div></div></a>`;
        }

        async function renderList(params) {
            const workId = params.get("work") ? Number(params.get("work")) : null;
            const selected = (params.get("tags") || "").split(",").filter(Boolean).map(Number);
            let tags, data;
            try {
                tags = await loadTags();
                data = await api(`/skill/api/videos?${new URLSearchParams({ ...(workId ? { work_tag: workId } : {}), tags: selected.join(",") })}`);
            } catch (err) {
                return showError(err);
            }
            const work = workId ? tags.byId[workId] : null;
            // 絞り込みボタン: 表示中の動画に付いているタグ（＋選択中のタグ）を分類ごとに並べる
            const facets = [...data.facets];
            for (const id of selected) {
                if (!facets.some(f => f.tag_id === id) && tags.byId[id]) facets.push({ ...tags.byId[id], count: 0 });
            }
            const groups = {};
            for (const f of facets) {
                if (f.tag_id === workId) continue;
                (groups[f.category] ||= []).push(f);
            }
            const filterHtml = tags.categories.filter(c => groups[c]).map(c => `
                <div class="filter-group"><div class="group-label">${esc(c)}</div><div class="chips">
                ${groups[c].map(f => {
                    const on = selected.includes(f.tag_id);
                    const next = on ? selected.filter(id => id !== f.tag_id) : [...selected, f.tag_id];
                    return `<a class="chip tag-filter ${on ? "selected" : ""}" href="${listHash(workId, next)}">${esc(f.name)}${on ? " ✕" : ` (${f.count})`}</a>`;
                }).join("")}</div></div>`).join("");
            app.innerHTML = `
                <div class="section">
                    ${breadcrumb(tags, workId, true)}
                    <h2>${work ? esc(work.name) + " の動画" : "すべての動画"}（${data.videos.length}件）</h2>
                    ${filterHtml ? `<p class="muted" style="margin:0 0 4px">タグで絞り込み（複数選ぶと、すべてに当てはまる動画だけ）</p>${filterHtml}` : ""}
                    ${selected.length ? `<p><a class="button small" href="${listHash(workId, [])}">絞り込みを解除</a></p>` : ""}
                </div>
                <div class="section" id="video-list">
                    ${data.videos.map(videoCard).join("") || '<p class="muted">該当する動画がありません</p>'}
                </div>`;
        }

        // --- アップロード ---
        function renderUpload() {
            app.innerHTML = `
                <div class="section">
                    <h2>動画を撮影・投稿</h2>
                    <div class="file-buttons">
                        <label class="camera">📹 撮影する<input type="file" id="camera-input" accept="video/*" capture="environment"></label>
                        <label>📁 ファイル選択<input type="file" id="file-input" accept="video/*"></label>
                    </div>
                    <p id="file-name" class="muted">動画が選ばれていません</p>
                    <label for="title">タイトル（作業の内容）</label>
                    <input type="text" id="title" placeholder="例: 型枠の建て込み">
                    <label for="explainer">説明者（任意）</label>
                    <input type="text" id="explainer" placeholder="例: 山田">
                    <p><button class="primary" id="upload-button" onclick="uploadVideo()">投稿する</button></p>
                    <div id="upload-progress" style="display:none">
                        <p class="muted" id="upload-text">アップロード中...</p>
                        <progress id="upload-bar" max="100" value="0"></progress>
                    </div>
                    <div id="status-area"></div>
                </div>
                <div class="section">
                    <h2>最近の投稿（処理の状況）</h2>
                    <div id="recent-uploads" class="muted">読み込み中...</div>
                </div>`;
            loadRecentUploads();
            let selected = null;
            for (const id of ["camera-input", "file-input"]) {
                document.getElementById(id).addEventListener("change", e => {
                    selected = e.target.files[0] || selected;
                    document.getElementById("file-name").textContent = selected ? `選んだ動画: ${selected.name}` : "動画が選ばれていません";
                });
            }
            window.selectedFile = () => selected;
        }

        const STATUS_TEXT = { uploaded: "受付", media: "動画の変換中", transcribe: "文字起こし中", procedure: "手順書の作成中",
                              tagging: "タグ付け中", done: "完了", error: "失敗" };

        async function loadRecentUploads() {
            const box = document.getElementById("recent-uploads");
            try {
                const data = await api("/skill/api/videos?include_unfinished=true");
                box.innerHTML = data.videos.slice(0, 10).map(v =>
                    `<a class="video-card" href="#/video/${v.video_id}"><div class="info"><div class="title">${esc(v.title)}</div>
                     <span class="status-badge ${v.status === "error" ? "error" : ""}">${STATUS_TEXT[v.status] || esc(v.status)}</span>
                     <span class="muted">${esc(v.created)}</span></div></a>`).join("") || "まだ投稿はありません";
                box.classList.remove("muted");
            } catch (err) {
                box.textContent = "取得できませんでした: " + err.message;
            }
        }

        function uploadVideo() {
            const file = window.selectedFile();
            const title = document.getElementById("title").value.trim();
            if (!file) return alert("動画を撮影するか、ファイルを選んでください");
            if (!title) return alert("タイトルを入力してください");

            const formData = new FormData();
            formData.append("file", file);
            formData.append("title", title);
            formData.append("explainer", document.getElementById("explainer").value.trim());

            // fetch ではアップロードの進み具合が分からないため、XMLHttpRequest を使う
            // （スマホの動画は大きく、送信に時間がかかるため）
            const xhr = new XMLHttpRequest();
            xhr.open("POST", `${BASE_URL}/skill/api/videos`);
            xhr.setRequestHeader("ngrok-skip-browser-warning", "true");
            document.getElementById("upload-button").disabled = true;
            document.getElementById("upload-progress").style.display = "block";
            xhr.upload.onprogress = e => {
                if (e.lengthComputable) {
                    const pct = Math.round(e.loaded / e.total * 100);
                    document.getElementById("upload-bar").value = pct;
                    document.getElementById("upload-text").textContent = `アップロード中... ${pct}%`;
                }
            };
            xhr.onload = () => {
                let data = {};
                try { data = JSON.parse(xhr.responseText); } catch (e) { }
                if (xhr.status !== 200) {
                    document.getElementById("upload-button").disabled = false;
                    return alert("アップロードに失敗しました: " + (data.detail || xhr.status));
                }
                document.getElementById("upload-text").textContent = "アップロード完了。自動処理を行っています（数分かかることがあります）";
                pollStatus(data.video_id);
            };
            xhr.onerror = () => {
                document.getElementById("upload-button").disabled = false;
                alert("アップロードに失敗しました。通信状態を確認してください。");
            };
            xhr.send(formData);
        }

        function renderStatus(video) {
            const names = STEP_LABELS.map(s => s[0]);
            const current = video.status === "error" ? video.failed_step : video.status;
            const currentIndex = video.status === "done" ? names.length : names.indexOf(current);
            const items = STEP_LABELS.map(([name, label], i) => {
                let cls = "waiting";
                if (i < currentIndex) cls = "done";
                else if (i === currentIndex) cls = video.status === "error" ? "failed" : "running";
                return `<li class="${cls}">${label}</li>`;
            }).join("");
            let footer = "";
            if (video.status === "done") {
                footer = `<a class="button primary" href="#/video/${video.video_id}">動画を見る</a>`;
            } else if (video.status === "error") {
                footer = `<p class="error">失敗しました: ${esc(video.error)}</p>
                          <button class="primary" onclick="retryVideo(${video.video_id})">失敗したところから再実行</button>`;
            }
            return `<h2>処理の状況: ${esc(video.title)}</h2><ul class="steps-status">${items}</ul>${footer}`;
        }

        async function pollStatus(videoId) {
            const area = document.getElementById("status-area");
            if (!area) return;
            try {
                const video = await api(`/skill/api/videos/${videoId}`);
                area.innerHTML = renderStatus(video);
                if (video.status !== "done" && video.status !== "error") {
                    pollTimer = setTimeout(() => pollStatus(videoId), 3000);
                } else if (document.getElementById("recent-uploads")) {
                    loadRecentUploads();
                }
            } catch (err) {
                area.innerHTML = `<p class="error">状況を取得できませんでした: ${esc(err.message)}</p>`;
                pollTimer = setTimeout(() => pollStatus(videoId), 3000);
            }
        }

        async function retryVideo(videoId) {
            try {
                await api(`/skill/api/videos/${videoId}/retry`, { method: "POST" });
                pollStatus(videoId);
            } catch (err) {
                alert("再実行できませんでした: " + err.message);
            }
        }

        // --- タグ一覧の管理（PC想定）: 追加・編集・削除、別名の登録、新タグ候補の採用 ---
        let tagState = null;

        function parseAliases(text) {
            return text.split(/[|｜、,，\n]/).map(a => a.trim()).filter(Boolean);
        }

        function tagOptions(filter, selectedId, emptyLabel) {
            const rows = tagState.tags.filter(filter).map(t =>
                `<option value="${t.tag_id}" ${t.tag_id === selectedId ? "selected" : ""}>${esc(tagState.ancestors(t.tag_id).map(a => a.name).join(" ＞ "))}</option>`);
            return (emptyLabel ? [`<option value="">${emptyLabel}</option>`] : []).concat(rows).join("");
        }

        function categoryOptions(selected) {
            return tagState.categories.map(c => `<option ${c === selected ? "selected" : ""}>${esc(c)}</option>`).join("");
        }

        async function renderTags(message) {
            try {
                tagState = await loadTags();
                tagState.candidates = (await api("/skill/api/tag_candidates")).candidates;
            } catch (err) {
                return showError(err);
            }
            const depth = t => tagState.ancestors(t.tag_id).length - 1;
            const ordered = [];
            const walk = (parentId, category) => {
                for (const t of tagState.tags.filter(t => t.parent_id === parentId && t.category === category)) {
                    ordered.push(t);
                    walk(t.tag_id, category);
                }
            };
            const tables = tagState.categories.map(c => {
                ordered.length = 0;
                walk(null, c);
                const rows = ordered.map(t => `<tr id="tag-row-${t.tag_id}">
                    <td>${"　".repeat(depth(t))}${depth(t) ? "└ " : ""}${esc(t.name)}</td>
                    <td>${t.aliases.map(esc).join("、")}</td><td>${t.video_count}</td>
                    <td class="actions"><button class="small" onclick="editTag(${t.tag_id})">編集</button>
                    <button class="small" onclick="deleteTag(${t.tag_id})">削除</button></td></tr>`).join("");
                return `<h3>${esc(c)}</h3><table><tr><th>名前</th><th>別名</th><th>動画数</th><th></th></tr>${rows}</table>`;
            }).join("");
            const candidates = tagState.candidates.map(c => `<tr id="candidate-${c.candidate_id}">
                <td><b>${esc(c.word)}</b></td><td>${esc(c.category || "")}</td><td>${esc(c.video_title || "")}</td>
                <td class="actions"><button class="small primary" onclick="showAdopt(${c.candidate_id}, 'new')">新しいタグにする</button>
                <button class="small" onclick="showAdopt(${c.candidate_id}, 'alias')">既存タグの別名にする</button>
                <button class="small" onclick="rejectCandidate(${c.candidate_id})">却下</button>
                <div id="adopt-${c.candidate_id}"></div></td></tr>`).join("");
            app.innerHTML = `
                ${message ? `<div class="notice">${esc(message)}</div>` : ""}
                <div class="section">
                    <h2>新タグ候補（タグ一覧に無い言葉）</h2>
                    ${candidates ? `<table><tr><th>言葉</th><th>分類</th><th>動画</th><th></th></tr>${candidates}</table>`
                                 : '<p class="muted">保留中の候補はありません</p>'}
                    <p><button onclick="rematchAll()">全動画のタグを付け直す</button>
                    <span class="muted">タグや別名を変えた後に押すと、これまでの動画にも反映されます（AIは使いません）</span></p>
                </div>
                <div class="section" id="tag-form-section">
                    <h2 id="tag-form-title">タグを追加</h2>
                    <div class="form-row">
                        <div><label for="tag-name">名前</label><input type="text" id="tag-name"></div>
                        <div><label for="tag-category">分類</label><select id="tag-category" onchange="refreshParentOptions()">${categoryOptions("作業")}</select></div>
                        <div><label for="tag-parent">親（上位の分類）</label><select id="tag-parent"></select></div>
                    </div>
                    <label for="tag-aliases">別名（「|」や「、」で区切る）</label>
                    <input type="text" id="tag-aliases" placeholder="例: 型枠の組立|建て込み">
                    <p><button class="primary" id="tag-save" onclick="saveTag()">追加する</button>
                    <button id="tag-cancel" style="display:none" onclick="renderTags()">編集をやめる</button></p>
                    <input type="hidden" id="tag-id">
                </div>
                <div class="section"><h2>タグ一覧</h2>${tables}</div>`;
            refreshParentOptions();
        }

        function refreshParentOptions(selectedId) {
            const category = document.getElementById("tag-category").value;
            const editingId = Number(document.getElementById("tag-id").value) || null;
            document.getElementById("tag-parent").innerHTML =
                tagOptions(t => t.category === category && t.tag_id !== editingId, selectedId, "（なし）");
        }

        function editTag(tagId) {
            const t = tagState.byId[tagId];
            document.getElementById("tag-id").value = tagId;
            document.getElementById("tag-form-title").textContent = `タグを編集: ${t.name}`;
            document.getElementById("tag-name").value = t.name;
            document.getElementById("tag-category").value = t.category;
            document.getElementById("tag-aliases").value = t.aliases.join("|");
            refreshParentOptions(t.parent_id);
            document.getElementById("tag-save").textContent = "保存する";
            document.getElementById("tag-cancel").style.display = "inline-block";
            document.getElementById("tag-form-section").scrollIntoView({ behavior: "smooth" });
        }

        async function saveTag() {
            const tagId = document.getElementById("tag-id").value;
            const parent = document.getElementById("tag-parent").value;
            const body = JSON.stringify({
                name: document.getElementById("tag-name").value,
                category: document.getElementById("tag-category").value,
                parent_id: parent ? Number(parent) : null,
                aliases: parseAliases(document.getElementById("tag-aliases").value),
            });
            try {
                await api(tagId ? `/skill/api/tags/${tagId}` : "/skill/api/tags", { method: tagId ? "PUT" : "POST", body });
                renderTags(tagId ? "タグを保存しました" : "タグを追加しました");
            } catch (err) {
                alert(err.message);
            }
        }

        async function deleteTag(tagId) {
            const t = tagState.byId[tagId];
            if (!confirm(`「${t.name}」を削除しますか？（${t.video_count}本の動画から外れます）`)) return;
            try {
                await api(`/skill/api/tags/${tagId}`, { method: "DELETE" });
                renderTags("タグを削除しました");
            } catch (err) {
                alert(err.message);
            }
        }

        function showAdopt(candidateId, mode) {
            const c = tagState.candidates.find(c => c.candidate_id === candidateId);
            const box = document.getElementById(`adopt-${candidateId}`);
            if (mode === "new") {
                const category = tagState.categories.includes(c.category) ? c.category : "作業";
                box.innerHTML = `<div class="inline-form form-row">
                    <div><label>名前</label><input type="text" id="adopt-name-${candidateId}" value="${esc(c.word)}"></div>
                    <div><label>分類</label><select id="adopt-category-${candidateId}">${categoryOptions(category)}</select></div>
                    <div><label>親</label><select id="adopt-parent-${candidateId}">${tagOptions(t => t.category === category, null, "（なし）")}</select></div>
                    <div><button class="primary small" onclick="adopt(${candidateId}, 'new')">採用する</button></div></div>`;
                document.getElementById(`adopt-category-${candidateId}`).addEventListener("change", e => {
                    document.getElementById(`adopt-parent-${candidateId}`).innerHTML = tagOptions(t => t.category === e.target.value, null, "（なし）");
                });
            } else {
                box.innerHTML = `<div class="inline-form form-row">
                    <div><label>別名を追加するタグ</label><select id="adopt-tag-${candidateId}">${tagOptions(t => t.category !== "難易度", null, "選んでください")}</select></div>
                    <div><button class="primary small" onclick="adopt(${candidateId}, 'alias')">別名にする</button></div></div>`;
            }
        }

        async function adopt(candidateId, mode) {
            const value = id => document.getElementById(`${id}-${candidateId}`).value;
            const body = mode === "new"
                ? { mode, name: value("adopt-name"), category: value("adopt-category"), parent_id: value("adopt-parent") ? Number(value("adopt-parent")) : null }
                : { mode, tag_id: value("adopt-tag") ? Number(value("adopt-tag")) : null };
            try {
                await api(`/skill/api/tag_candidates/${candidateId}/adopt`, { method: "POST", body: JSON.stringify(body) });
                renderTags("採用しました。「全動画のタグを付け直す」を押すと、これまでの動画にも反映されます");
            } catch (err) {
                alert(err.message);
            }
        }

        async function rejectCandidate(candidateId) {
            try {
                await api(`/skill/api/tag_candidates/${candidateId}/reject`, { method: "POST" });
                renderTags("却下しました（同じ言葉は今後候補に出ません）");
            } catch (err) {
                alert(err.message);
            }
        }

        async function rematchAll() {
            try {
                const res = await api("/skill/api/rematch", { method: "POST" });
                renderTags(`${res.videos}本の動画のタグを付け直しました`);
            } catch (err) {
                alert(err.message);
            }
        }

        // --- 再生 ---
        async function renderVideo(videoId, params) {
            app.innerHTML = `<div class="section muted">読み込み中...</div>`;
            let video;
            try {
                video = await api(`/skill/api/videos/${videoId}`);
            } catch (err) {
                return showError(err);
            }
            if (video.status !== "done") {
                app.innerHTML = `<div class="section" id="status-area">${renderStatus(video)}</div>`;
                if (video.status !== "error") pollTimer = setTimeout(() => pollStatus(videoId), 3000);
                return;
            }

            const tagsByStep = {};
            for (const t of video.scene_tags) (tagsByStep[t.step_id] ||= []).push(t);
            const list = (label, values, cls = "") =>
                values && values.length ? `<div class="${cls}"><span class="label">${label}</span>${values.map(esc).join("、")}</div>` : "";
            const stepCards = video.steps.map((s, i) => `
                <div class="step-card" id="step-${s.step_id}" data-start="${s.start}" data-end="${s.end}">
                    <div class="step-head" onclick="seekTo(${s.start})">
                        <span class="time">${clock(s.start)}</span>
                        <span class="title">手順${i + 1}　${esc(s.title)}</span>
                        <span>▶</span>
                    </div>
                    <div class="step-body">
                        <div>${esc(s.description)}</div>
                        ${list("道具", s.tools)}${list("資材", s.materials)}
                        ${list("注意点", s.cautions, "caution")}${list("コツ", s.tips, "tip")}
                        <div class="chips" style="margin-top:6px">
                            ${(tagsByStep[s.step_id] || []).map(t => `<span class="chip scene-tag" onclick="seekTo(${t.start})"><span class="cat">${esc(t.category)}</span>${esc(t.name)}</span>`).join("")}
                        </div>
                    </div>
                </div>`).join("");
            const marks = video.steps.map((s, i) => {
                const left = s.start / video.duration * 100;
                const width = Math.max((s.end - s.start) / video.duration * 100, 1);
                return `<div class="mark" style="left:${left}%;width:${width}%" title="${esc(s.title)}" onclick="event.stopPropagation();seekTo(${s.start})">${i + 1}</div>`;
            }).join("");
            const transcript = video.segments.map(s =>
                `<p onclick="seekTo(${s.start})"><span class="time">${clock(s.start)}</span>${esc(s.text)}</p>`).join("");

            app.innerHTML = `
                <div class="section">
                    <h2>${esc(video.title)}</h2>
                    <div class="muted">${video.explainer ? `説明者: ${esc(video.explainer)}　` : ""}長さ: ${clock(video.duration)}</div>
                    <div class="chips" style="margin:6px 0">${video.video_tags.map(t => `<span class="chip"><span class="cat">${esc(t.category)}</span>${esc(t.name)}</span>`).join("")}</div>
                    <div class="player-wrap">
                        <video id="player" controls playsinline preload="metadata" src="${BASE_URL}/skill/api/videos/${videoId}/media">
                            <track id="subtitles" kind="subtitles" srclang="ja" label="日本語" default src="${BASE_URL}/skill/api/videos/${videoId}/subtitles.vtt">
                        </video>
                        <div class="scene-bar" id="scene-bar" title="場面の印（タップでその場面へ）">${marks}<div class="playhead" id="playhead"></div></div>
                    </div>
                    <div class="player-tools">
                        <button id="subtitle-toggle" onclick="toggleSubtitles()">字幕: 表示中</button>
                        ${video.has_pdf ? `<a class="button primary" id="pdf-link" href="${BASE_URL}/skill/api/videos/${videoId}/procedure.pdf" download>📄 手順書PDF</a>` : ""}
                    </div>
                </div>
                <div class="section">
                    <h2>手順と場面（タップでその場面から再生）</h2>
                    ${stepCards || '<p class="muted">手順がありません</p>'}
                </div>
                <div class="section transcript">
                    <details><summary>文字起こし全文</summary>${transcript}</details>
                </div>`;

            const player = document.getElementById("player");
            document.getElementById("scene-bar").addEventListener("click", e => {
                const rect = e.currentTarget.getBoundingClientRect();
                seekTo((e.clientX - rect.left) / rect.width * video.duration);
            });
            player.addEventListener("timeupdate", () => {
                document.getElementById("playhead").style.left = `${player.currentTime / video.duration * 100}%`;
                for (const card of document.querySelectorAll(".step-card")) {
                    const inStep = player.currentTime >= Number(card.dataset.start) && player.currentTime < Number(card.dataset.end);
                    card.classList.toggle("current", inStep);
                }
            });
            if (params.get("t")) seekTo(Number(params.get("t")));
        }

        function seekTo(seconds) {
            const player = document.getElementById("player");
            if (!player) return;
            player.currentTime = seconds;
            player.play().catch(() => { });  // 自動再生が止められても、位置の移動はできている
            player.scrollIntoView({ behavior: "smooth", block: "start" });
        }

        function toggleSubtitles() {
            const track = document.getElementById("player").textTracks[0];
            const showing = track.mode === "showing";
            track.mode = showing ? "hidden" : "showing";
            document.getElementById("subtitle-toggle").textContent = showing ? "字幕: 非表示" : "字幕: 表示中";
        }

        route();
    </script>
</body>
</html>
'''


# --- 13. サーバー起動（Googleドライブのマウント → Whisperの読み込み → ngrok → サーバー） ---
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(SKILL_MEDIA_DIR, exist_ok=True)

import torch, whisper, openai, requests, nest_asyncio, uvicorn
skill_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Whisper（{SKILL_WHISPER_MODEL}）を読み込み中...")
skill_whisper_model = whisper.load_model(SKILL_WHISPER_MODEL, device=skill_device)
skill_openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) if SKILL_LLM_PROVIDER == "openai" else None
skill_app = build_skill_app(skill_whisper_model, skill_openai_client)

!pkill -f uvicorn
!pkill -f ngrok
time.sleep(2)

NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")
!ngrok config add-authtoken {NGROK_AUTH_TOKEN}
skill_ngrok_process = subprocess.Popen(["ngrok", "http", str(SKILL_PORT)], stdout=subprocess.PIPE)
time.sleep(5)

try:
    public_url = requests.get("http://localhost:4040/api/tunnels").json()["tunnels"][0]["public_url"]
    print("\n✅ 接続成功！")
    print(f"📱 スマホで開くURL: {public_url}/skill")
    print("↑ 初回だけ『Visit Site』を押してください。")
except Exception as e:
    print(f"\n❌ URL取得失敗: {e}")

nest_asyncio.apply()
skill_config = uvicorn.Config(skill_app, host="0.0.0.0", port=SKILL_PORT, loop="asyncio")
skill_server = uvicorn.Server(skill_config)
await skill_server.serve()
