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

!pip install fastapi uvicorn python-multipart pydub noisereduce rapidfuzz weasyprint -q
!apt-get -qq install -y fonts-noto-cjk > /dev/null

import os, re, csv, json, shutil, sqlite3, inspect, threading, traceback, subprocess
from datetime import datetime

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
                 mask_fn=None, anonymize=SKILL_ANONYMIZE, run_in_background=True):
        self.conn = conn
        self.media_dir = media_dir
        self.whisper_model = whisper_model
        self.llm_fn = llm_fn
        self.mask_fn = mask_fn
        self.anonymize = anonymize
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


SKILL_STEPS = [("media", step_media), ("transcribe", step_transcribe)]
SKILL_STEP_NAMES = [name for name, _ in SKILL_STEPS]


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


# --- 7. API（/skill/api/...） ---
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response

ALLOWED_VIDEO_EXT = re.compile(r"^\.[a-z0-9]{1,5}$")


def video_summary(video):
    """画面に返す動画の情報（Drive上のファイルパスは返さない）。"""
    keys = ["id", "title", "explainer", "filename", "duration", "status", "failed_step", "error", "created", "updated"]
    summary = {k: video[k] for k in keys}
    summary["video_id"] = summary.pop("id")
    summary["has_media"] = bool(video["media_path"])
    summary["has_pdf"] = bool(video["pdf_path"])
    return summary


def create_skill_router(ctx):
    router = APIRouter()

    def require_video(video_id):
        video = get_skill_video(ctx.conn, video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="動画が見つかりません")
        return video

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

    @router.get("/skill/api/videos/{video_id}")
    def api_skill_video(video_id: int):
        video = require_video(video_id)
        return {**video_summary(video), "segments": get_segments(ctx.conn, video_id)}

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

    @router.get("/skill/api/videos/{video_id}/subtitles.vtt")
    def api_skill_subtitles(video_id: int):
        require_video(video_id)
        return Response(build_webvtt(get_segments(ctx.conn, video_id)), media_type="text/vtt; charset=utf-8")

    return router


def register_skill_transfer(app, conn=None, whisper_model=None, openai_client=None, mask_fn=None,
                            llm_fn=None, media_dir=None, seed_csv=None, run_in_background=True):
    """
    技能伝承のAPIを既存のFastAPIアプリに追加する。既存のルートには触れない。
    タグ一覧が空のときだけ初期データ（seed CSV）を入れる。
    保存先などを省略したときは、呼び出した時点の設定（SKILL_MEDIA_DIR など）を使う。
    """
    media_dir = media_dir or SKILL_MEDIA_DIR
    seed_csv = seed_csv or SKILL_SEED_CSV
    if conn is None:
        conn = open_skill_db()
    else:
        init_skill_db(conn)
    os.makedirs(media_dir, exist_ok=True)
    if conn.execute("SELECT COUNT(*) FROM skill_tags").fetchone()[0] == 0 and os.path.exists(seed_csv):
        print("技能伝承: タグ一覧の初期データを投入しました", seed_skill_tags(conn, seed_csv))
    ctx = SkillContext(conn, media_dir=media_dir, whisper_model=whisper_model, llm_fn=llm_fn,
                       mask_fn=mask_fn, run_in_background=run_in_background)
    ctx.openai_client = openai_client
    app.include_router(create_skill_router(ctx))
    app.state.skill_transfer = ctx
    return ctx


# --- 8. 既存セルのサーバー起動時に自動で登録する仕組み ---
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
