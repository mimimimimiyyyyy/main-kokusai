# --- 別セル: 技能伝承動画共有機能 ---
# 建設業向けに、熟練者の作業説明動画から文字起こし・字幕・手順書・タグを自動で作り、
# 若手がタグで動画を探して見られるようにする機能。
#
# 実行順: このセル → audio_analysis_pipeline.py のセル（いつもどおり）
# 既存セルは最後にサーバーを起動したまま止まる（await server.serve()）ため、後から
# 別セルでAPIを追加できない。また既存セルは編集しない方針なので、このセルでは
# uvicornのサーバー起動処理に「起動直前に技能伝承のAPIをappへ登録する」処理を
# 差し込んでおく（install_skill_transfer_hook）。Colabではセル同士が同じ変数の置き場を
# 共有しているので、起動時点で既存セルが作ったapp・whisper_model・clientを使える。
# このセルを実行していなければ何も差し込まれないので、既存の動作は変わらない。
#
# 対話研究のデータ（sessions / turns など）とは混ぜず、同じcorpus.dbの中に
# skill_ で始まる表を別に作って保存する（docs/skill-transfer-mapping.md のQ1）。

!pip install fastapi uvicorn python-multipart pydub noisereduce rapidfuzz weasyprint openai anthropic -q
!apt-get -qq install -y fonts-noto-cjk > /dev/null

import os, re, csv, json, shutil, sqlite3, inspect, threading, traceback, subprocess
from datetime import datetime
from typing import Optional

import numpy as np


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


# DBは既存セルと同じcorpus.db（skill_ の表だけを読み書きする）
SKILL_DB_PATH = os.environ.get("SKILL_DB_PATH", "/content/drive/MyDrive/corpus.db")
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
    """
    既存セルと同じcorpus.dbを、技能伝承用の別の接続で開く。既存セルの conn を
    共有しないのは、技能伝承側の書き込み途中のトランザクションを既存の処理の
    commitに巻き込まないため（逆も同じ）。設定は既存セルのinit_dbと揃えている。
    """
    conn = sqlite3.connect(path or SKILL_DB_PATH, check_same_thread=False, timeout=30)
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
    16kHz・モノラル化と音量正規化は既存セルのrun_full_analysis（A. 音声変換）と同じ処理。
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
# 既存セルの transcribe_full_audio と同じ設定・同じフィルタ。既存セルは編集しない方針の
# ため、言語指定と専門用語ヒントを足したものをここに複製している。既存側のフィルタを
# 変更した場合はこちらにも反映すること（transcription_accuracy_tool.py と同じ扱い）。
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
    幻覚フィルタは既存セルと同じ（理由は既存セルのコメント参照）。
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
    設定でONのときだけ、既存セルの匿名化（extract_mask_targets_with_gpt）を使って
    固有名詞を[MASK]に置き換える。既定はOFF（道具やメーカーの名前まで伏せてしまうため）。
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
# 既存セルのjobs辞書はメモリ上だけなので、Colabの再起動で消えてしまう。技能伝承では
# 状態・失敗した段階・エラー内容をskill_videosに残し、画面から失敗した段階だけを
# やり直せるようにする。
VIDEO_COLUMNS = ["id", "title", "explainer", "filename", "media_path", "duration", "status",
                 "failed_step", "error", "vtt_path", "pdf_path", "created", "updated"]


class SkillContext:
    """技能伝承の処理に必要なもの一式。Colabでは既存セルのモデル・クライアントを入れる。"""

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
# 既定は既存セルと同じOpenAI（gpt-4o・JSONモード）。SKILL_LLM_PROVIDER=anthropic で
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
    """スキーマに加えて、セグメント番号が範囲内・順番どおり・重ならないことを確かめる。"""
    result = ProcedureResult.model_validate(data)
    if not result.steps:
        raise ValueError("手順が1つもありません")
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
.photo { width: 62mm; flex: none; }
.photo img { width: 100%; border: 1px solid #ccc; }
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
    ③ の2: タグ一覧と照合する。完全一致（名前）→ 別名一致 → 文字列の類似度が閾値以上、の順。
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
    # 類似度は表記ゆれ（送り仮名・誤認識など）を拾うためのもの。一方がもう一方を丸ごと
    # 含む言葉（コンクリート／コンクリート打設）は意味の広さが違うので、ここでは一致させない。
    best = None
    for tag in candidates:
        for name in [tag["name"]] + tag["aliases"]:
            other = normalize_term(name)
            if key in other or other in key:
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
        # ngrokの警告ページに止められずに読み込める）
        if not os.path.exists(ctx.html_path):
            return HTMLResponse(
                "<meta charset='utf-8'><p>画面のファイル（skill_transfer.html）が見つかりません。"
                f"Google Driveの {ctx.html_path} に置いてから、セルを実行し直してください。</p>",
                status_code=404)
        with open(ctx.html_path, encoding="utf-8") as f:
            return HTMLResponse(f.read())

    @router.post("/skill/api/videos")
    # 既存の/uploadと同じく、重い処理を待たずにIDをすぐ返し、処理は裏で行う
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
                            llm_fn=None, media_dir=None, seed_csv=None, html_path=None, run_in_background=True):
    """
    技能伝承のAPIを既存のFastAPIアプリに追加する。既存のルートには触れない。
    タグ一覧が空のときだけ初期データ（seed CSV）を入れる。
    保存先などを省略したときは、呼び出した時点の設定（SKILL_MEDIA_DIR など）を使う。
    """
    media_dir = media_dir or SKILL_MEDIA_DIR
    seed_csv = seed_csv or SKILL_SEED_CSV
    html_path = html_path or SKILL_HTML_PATH
    if conn is None:
        conn = open_skill_db()
    else:
        init_skill_db(conn)
    if not isinstance(conn, SerializedConnection):
        conn = SerializedConnection(conn)
    os.makedirs(media_dir, exist_ok=True)
    if conn.execute("SELECT COUNT(*) FROM skill_tags").fetchone()[0] == 0 and os.path.exists(seed_csv):
        print("技能伝承: タグ一覧の初期データを投入しました", seed_skill_tags(conn, seed_csv))
    if llm_fn is None:
        llm_fn = make_llm_fn(openai_client=openai_client)
    ctx = SkillContext(conn, media_dir=media_dir, whisper_model=whisper_model, llm_fn=llm_fn,
                       mask_fn=mask_fn, run_in_background=run_in_background)
    ctx.openai_client = openai_client
    ctx.html_path = html_path
    app.include_router(create_skill_router(ctx))
    app.state.skill_transfer = ctx
    return ctx


# --- 12. 既存セルのサーバー起動時に自動で登録する仕組み ---
def install_skill_transfer_hook(namespace=None):
    """
    uvicorn.Server.serve を包み、サーバー起動の直前に register_skill_transfer を呼ぶ。
    既存セルを編集せずに技能伝承のAPIを追加するための仕組み。namespace（省略時はこの
    セルの変数の置き場＝Colabでは全セル共通）から、既存セルが作ったWhisperモデル・
    OpenAIクライアント・匿名化関数を受け取る。登録に失敗しても既存のサーバーは起動する。
    """
    import uvicorn
    from fastapi import FastAPI

    original = getattr(uvicorn.Server, "_skill_transfer_original_serve", None) or uvicorn.Server.serve
    ns = namespace if namespace is not None else globals()

    async def serve_with_skill_transfer(self, *args, **kwargs):
        app = self.config.app
        if isinstance(app, FastAPI) and getattr(app.state, "skill_transfer", None) is None:
            try:
                register_skill_transfer(
                    app,
                    whisper_model=ns.get("whisper_model"),
                    openai_client=ns.get("client"),
                    mask_fn=ns.get("extract_mask_targets_with_gpt"),
                )
                print("✅ 技能伝承機能を登録しました（画面: 表示されたURLの末尾に /skill を付けて開く）")
            except Exception:
                traceback.print_exc()
                print("⚠️ 技能伝承機能の登録に失敗しました。既存の機能はそのまま起動します。")
        return await original(self, *args, **kwargs)

    uvicorn.Server._skill_transfer_original_serve = original
    uvicorn.Server.serve = serve_with_skill_transfer


if __name__ == "__main__":
    # Colabのセルとして実行されたときだけ差し込む（pytestから読み込むときは差し込まない）
    install_skill_transfer_hook()
    print("技能伝承セルの準備ができました。続けて audio_analysis_pipeline.py のセルを実行してください。")
