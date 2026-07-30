# --- 1. ライブラリのインストール ---
!pip install fastapi uvicorn pyngrok nest_asyncio python-multipart japanize-matplotlib pydub whisper pyannote.audio openai -q

import os, openai, torch, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
import japanize_matplotlib, time, warnings, whisper, shutil, io, base64, sqlite3
from datetime import datetime
from typing import Optional
from pydub import AudioSegment
from pyannote.audio import Pipeline
from google.colab import userdata
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn, nest_asyncio, pyngrok.ngrok as ngrok

# --- 2. 初期設定 ---
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
nest_asyncio.apply()
client = openai.OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

TARGET_NUM_SPEAKERS = 4
TEMP_SEGMENT_FILE = "temp_segment.wav"

PHASE_ORDER = ['Introduction', 'Information Sharing', 'Conflict', 'Conclusion', 'Agreement']
INTENT_LABELS = ["Proposal", "Question", "Agreement", "Disagreement", "Confirmation", "Acknowledge", "Explanation"]

PHASE_MAP = {
    '導入': 'Introduction', '情報共有': 'Information Sharing',
    '意見対立': 'Conflict', '収束': 'Conclusion', '合意': 'Agreement',
    '不明': 'Information Sharing'
}
INTENT_MAP = {
    '説明': 'Explanation', '相槌': 'Acknowledge', '同意': 'Agreement',
    '提案': 'Proposal', '質問': 'Question', '確認': 'Confirmation',
    '不明': 'Explanation', '終了': 'Explanation', '挨拶': 'Explanation',
    '意見対立': 'Disagreement'
}
ROLE_MAP = {
    '進行役': 'Leader', '議長': 'Leader',
    '参加者': 'Participant', '不明': 'Participant', 'unknown': 'Participant'
}


def normalize_phase(p):
    p = str(p)
    return PHASE_MAP.get(p, p if p in PHASE_ORDER else 'Information Sharing')


def normalize_intent(i):
    i = str(i)
    return INTENT_MAP.get(i, i if i in INTENT_LABELS else 'Explanation')


def normalize_role(r):
    r = str(r).strip()
    return ROLE_MAP.get(r, r if r not in ["", "None", "Unknown"] else "Participant")


def init_db():
    conn = sqlite3.connect("/content/drive/MyDrive/corpus.db", check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            filename  TEXT,
            created   TEXT,
            summary   TEXT,
            metadata  TEXT,
            roles     TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            speaker    TEXT,
            start      REAL,
            end        REAL,
            text       TEXT,
            phase      TEXT,
            intent     TEXT,
            role       TEXT,
            embedding  BLOB,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    # addin_results: 新しい分析手法の結果をセッションに紐づけて何度でも追加できる置き場。
    # method_version を分けることで、同じ addin_name でも手法を変えて再分析した結果を
    # 上書きせずに全部残し、後から比較・再利用できるようにする。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS addin_results (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     INTEGER,
            addin_name     TEXT,
            method_version TEXT,
            created        TEXT,
            result         TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    return conn


try:
    conn.close()
except NameError:
    pass

conn = init_db()

print("AIモデルをロード中...")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diarization_pipeline.to(device)
whisper_model = whisper.load_model("small", device=device)
warnings.filterwarnings("ignore")

# --- 3. フル解析ロジック ---

def save_to_corpus(filename, result, all_turns, embeddings):
    cur = conn.execute(
        "INSERT INTO sessions (filename, created, summary, metadata, roles) VALUES (?,?,?,?,?)",
        (
            filename,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result["summary"],
            json.dumps(result["metadata"]),
            json.dumps(result["roles"])
        )
    )
    session_id = cur.lastrowid
    for i, turn in enumerate(all_turns):
        conn.execute(
            """INSERT INTO turns
               (session_id, speaker, start, end, text, phase, intent, role, embedding)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                session_id,
                turn['speaker'], turn['start'], turn['end'],
                turn['text'],
                turn.get('phase', 'Information Sharing'),
                turn.get('intent', 'Explanation'),
                turn.get('role', 'Participant'),
                embeddings[i].astype(np.float64).tobytes()
            )
        )
    conn.commit()
    return session_id


def tag_turns_with_gpt(turns, roles_dict, block_size=40):
    """
    speaker/start/end/text だけを持つ発話リストに phase/intent を付与する。
    diarization/Whisperをやり直さずに、既に文字起こし済みのturnsへ何度でも
    新しいタグ付けロジック（プロンプト変更・モデル変更）を適用し直せるように、
    フル解析の中の全編タグ付け部分を独立させたもの。/upload の初回分析と
    /corpus/{id}/retag による再分析の両方から共有される。
    """
    tagged = [dict(t) for t in turns]
    for i in range(0, len(tagged), block_size):
        block = tagged[i: i + block_size]
        tag_prompt = f"""
You are a dialogue analysis expert. Label each utterance with a phase and intent.

=== PHASE DEFINITIONS ===
- "Introduction"       : Opening, greetings, topic setting, agenda announcement
- "Information Sharing": Presenting facts, explaining background, reporting status
- "Conflict"           : Disagreement, opposing opinions, challenging someone's idea, tension
- "Conclusion"         : Summarizing what was discussed, wrapping up a topic
- "Agreement"          : Reaching consensus, confirming a shared decision

CRITICAL RULES:
* A real conversation MUST contain multiple phases. Do NOT assign "Information Sharing" to everything.
* Early utterances  → likely "Introduction"
* Middle utterances → "Information Sharing", "Conflict", or "Agreement" depending on content
* Final utterances  → "Conclusion" or "Agreement"
* If there is TENSION or OPPOSITION in the exchange -> use "Conflict"
* A neutral question for clarification is NOT "Conflict"
* If someone says ok / agreed / let's do that → use "Agreement"

=== INTENT DEFINITIONS ===
- "Proposal"      : Suggesting a new idea or action
- "Question"      : Asking for information or clarification
- "Agreement"     : Expressing approval or consent
- "Disagreement"  : Expressing opposition or doubt
- "Confirmation"  : Verifying or checking understanding
- "Acknowledge"   : Short responses showing attention (yeah, I see, mm-hmm)
- "Explanation"   : Elaborating or clarifying a point

Data (utterances {i} to {i + len(block) - 1} of {len(tagged)} total):
{json.dumps(block, ensure_ascii=False)}

Return JSON with key 'analysis' containing a list of {{"phase": ..., "intent": ...}} for each utterance in order.
        """
        tag_res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": tag_prompt}],
            response_format={"type": "json_object"}
        )
        tags = json.loads(tag_res.choices[0].message.content).get('analysis', [])
        for j, tag in enumerate(tags):
            if i + j < len(tagged):
                turn = tagged[i + j]
                turn["phase"] = normalize_phase(tag.get('phase', 'Information Sharing'))
                turn["intent"] = normalize_intent(tag.get('intent', 'Explanation'))
                turn["role"] = roles_dict.get(turn['speaker'], "Participant")
    return tagged


def extract_mask_targets_with_gpt(turns, block_size=40):
    """
    匿名化すべき単語（個人名・会社名・機密プロジェクト名・連絡先など）を
    全発話を対象に抽出する。冒頭サンプルだけを見ると、会話後半にしか
    登場しない固有名詞が見逃されるため、tag_turns_with_gptと同様に
    全発話をブロックに分けてGPT-4oに走査させ、結果を統合する。
    """
    mask_words = {}
    for i in range(0, len(turns), block_size):
        block = turns[i: i + block_size]
        block_data = json.dumps(
            [{"speaker": t["speaker"], "text": t["text"]} for t in block],
            ensure_ascii=False
        )
        mask_prompt = f"""
Extract words or phrases that require anonymization (personal names, company
names, confidential project names, phone numbers, email addresses, addresses,
etc.) from the dialogue below. Do not flag common nouns or generic terms.

Return JSON with key 'mask_list': a list of objects like {{"word": "..."}}.
If nothing needs anonymization in this excerpt, return {{"mask_list": []}}.

Data (utterances {i} to {i + len(block) - 1}):
{block_data}
        """
        mask_res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": mask_prompt}],
            response_format={"type": "json_object"}
        )
        for target in json.loads(mask_res.choices[0].message.content).get('mask_list', []):
            word = target.get('word')
            if word:
                mask_words[word] = True
    return [{"word": w} for w in mask_words]


def render_analysis_charts(all_turns):
    df = pd.DataFrame(all_turns)
    df['duration'] = df['end'] - df['start']
    df['phase'] = df['phase'].apply(normalize_phase)
    df['intent'] = df['intent'].apply(normalize_intent)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    df['speaker'].value_counts().sort_index().plot(
        kind='pie', ax=axes[0,0], autopct='%1.1f%%', cmap='Set3',
        title="Utterance Count Ratio by Speaker"
    )
    axes[0,0].set_ylabel("")

    df.groupby('speaker')['duration'].sum().sort_index().plot(
        kind='bar', ax=axes[0,1], color='skyblue',
        title="Total Utterance Duration by Speaker (sec)"
    )
    axes[0,1].tick_params(axis='x', rotation=45)

    df_plot = df[df['phase'].isin(PHASE_ORDER)].copy()
    if not df_plot.empty:
        axes[1,0].step(df_plot['start'], df_plot['phase'], where='post', marker='o', color='purple')
        axes[1,0].set_yticks(PHASE_ORDER)
        axes[1,0].set_title("Timeline of Consensus Building Phases")
    else:
        axes[1,0].text(0.5, 0.5, "No valid phase data", ha='center')

    df['intent'].value_counts().head(10).plot(
        kind='bar', ax=axes[1,1], color='orange',
        title="Frequency of Utterance Intents (TOP 10)"
    )
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def run_full_analysis(video_path):
    # A. 音声変換
    full_audio = AudioSegment.from_file(video_path).set_frame_rate(16000).set_channels(1)
    samples = np.array(full_audio.get_array_of_samples()).astype(np.float32) / 32768.0
    audio_data_dict = {'waveform': torch.tensor(samples).unsqueeze(0), 'sample_rate': 16000}

    # B. 話者分離 & 文字起こし
    print("話者分離と文字起こしを実行中...")
    diarization = diarization_pipeline(audio_data_dict, num_speakers=TARGET_NUM_SPEAKERS)
    speaker_results = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        seg = full_audio[int(turn.start*1000):int(turn.end*1000)]
        seg.export(TEMP_SEGMENT_FILE, format="wav")
        text = whisper_model.transcribe(TEMP_SEGMENT_FILE)['text'].strip()
        if text:
            speaker_results.setdefault(speaker, []).append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "text": text
            })

    all_turns = []
    for spk, utts in speaker_results.items():
        for u in utts:
            all_turns.append({"speaker": spk, **u})
    all_turns.sort(key=lambda x: x['start'])

    # C. メタデータ・要約判定
    # summary/metadata/speaker_rolesは全体像の把握が目的なので冒頭サンプルで十分だが、
    # 匿名化対象語の抽出は見逃しが個人情報漏洩に直結するため、ここでは含めない
    # （全発話を対象に extract_mask_targets_with_gpt で別途行う）。
    print("AIによるメタデータ・要約・役割判定中...")
    all_speakers = sorted(list(set(t['speaker'] for t in all_turns)))
    prompt = f"""Extract the following from the dialogue data and return it in JSON.
    1. summary: Overall summary in English
    2. metadata: {{"scene": "Scene description in English", "task": "Task description in English"}}
    3. speaker_roles: {{"SPEAKER_ID": "Role in English"}} for all members. Never use Japanese or 'Unknown'. Example: {{"SPEAKER_00": "Leader"}}
    Data (Sample): {json.dumps(all_turns[:30], ensure_ascii=False)}"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    analysis_res = json.loads(res.choices[0].message.content)

    raw_roles = analysis_res.get('speaker_roles', {})
    roles_dict = {}
    if isinstance(raw_roles, dict):
        roles_dict = raw_roles
    elif isinstance(raw_roles, list):
        for i, r in enumerate(raw_roles):
            if i < len(all_speakers):
                roles_dict[all_speakers[i]] = r
    for spk in all_speakers:
        roles_dict[spk] = normalize_role(roles_dict.get(spk, "Participant"))

    # C-2. テキストデータの自動置換処理
    print("匿名化対象語を全発話から抽出中...")
    mask_list = extract_mask_targets_with_gpt(all_turns)
    print("文字起こしテキストの自動アノニマイズを実行中...")
    for target in mask_list:
        secret_word = target.get('word')
        if secret_word:
            for turn in all_turns:
                if secret_word in turn['text']:
                    turn['text'] = turn['text'].replace(secret_word, "[MASK]")
            print(f"   [Text Masked] '{secret_word}' -> '[MASK]'")

    for turn in all_turns:
        turn.setdefault('phase', 'Information Sharing')
        turn.setdefault('intent', 'Explanation')
        turn.setdefault('role', roles_dict.get(turn['speaker'], 'Participant'))

    # D. 全編タグ付け（既存turnsの再タグ付けとロジックを共有）
    print("AIによる全編タグ付け中...")
    all_turns = tag_turns_with_gpt(all_turns, roles_dict)

    # E. 4画面グラフの作成
    print("Generating analysis charts...")
    img_base64 = render_analysis_charts(all_turns)

    # F. ベクトル検索用埋め込みの生成（DBに保存し、/searchは常にDBから読む）
    print("埋め込みベクトルを生成中...")
    texts = [t['text'] for t in all_turns]
    emb_res = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings_array = np.array([r.embedding for r in emb_res.data])

    result = {
        "summary": analysis_res.get("summary", "No summary available"),
        "metadata": analysis_res.get("metadata", {"scene": "Unknown", "task": "Unknown"}),
        "roles": roles_dict,
        "chart": img_base64
    }

    # G. DBに永続化
    session_id = save_to_corpus(video_path, result, all_turns, embeddings_array)
    result["session_id"] = session_id
    print(f"✅ セッション {session_id} としてコーパスに保存しました")

    return result


# --- 4. API ---
class SearchRequest(BaseModel):
    query: str

class AddinRequest(BaseModel):
    addin_name: str
    result: dict
    method_version: Optional[str] = None

class RetagRequest(BaseModel):
    method_version: Optional[str] = None


@app.post("/upload")
async def api_upload(file: UploadFile = File(...)):
    temp_file = f"input_{file.filename}"
    with open(temp_file, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return run_full_analysis(temp_file)


@app.post("/search")
async def api_search(req: SearchRequest):
    # 直近アップロード分だけのインメモリ状態ではなく、DBに永続化された
    # 全セッションのturnsを対象に検索する。過去に保存した分析結果を
    # セッション横断で再利用できるようにするための変更。
    # 「直近アップロードした動画」= sessions.id が最大のセッションとして扱い、
    # それ以外の全セッションと分けて返す。
    latest = conn.execute(
        "SELECT id, filename FROM sessions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    latest_session_id, latest_filename = latest if latest else (None, None)

    rows = conn.execute(
        """SELECT t.session_id, s.filename, t.speaker, t.start, t.text, t.phase, t.role, t.embedding
           FROM turns t JOIN sessions s ON s.id = t.session_id
           WHERE t.embedding IS NOT NULL"""
    ).fetchall()
    empty_response = {
        "latest_session": {"session_id": latest_session_id, "filename": latest_filename, "results": []},
        "past_sessions": {"results": []}
    }
    if not rows:
        return empty_response

    q_vec = np.array(
        client.embeddings.create(input=[req.query], model="text-embedding-3-small").data[0].embedding
    )
    emb_matrix = np.stack([np.frombuffer(r[7], dtype=np.float64) for r in rows])
    scores = emb_matrix @ q_vec

    latest_results = []
    past_results = []
    for row, score in zip(rows, scores):
        if score >= 0.4:
            session_id, filename, speaker, start, text, phase, role, _ = row
            item = {
                "session_id": session_id,
                "filename": filename,
                "time": f"{int(start//60):02d}:{int(start%60):02d}",
                "speaker": speaker,
                "role": normalize_role(role),
                "text": text,
                "phase": normalize_phase(phase),
                "score": float(score)
            }
            if session_id == latest_session_id:
                latest_results.append(item)
            else:
                past_results.append(item)

    latest_results.sort(key=lambda x: x['score'], reverse=True)
    past_results.sort(key=lambda x: x['score'], reverse=True)
    return {
        "latest_session": {"session_id": latest_session_id, "filename": latest_filename, "results": latest_results},
        "past_sessions": {"results": past_results}
    }


@app.get("/corpus")
async def api_corpus_list():
    rows = conn.execute(
        "SELECT id, filename, created, summary, metadata FROM sessions ORDER BY created DESC"
    ).fetchall()
    return {"sessions": [
        {
            "session_id": r[0], "filename": r[1], "created": r[2],
            "summary": r[3], "metadata": json.loads(r[4])
        }
        for r in rows
    ]}


@app.get("/corpus/{session_id}")
async def api_corpus_get(session_id: int):
    session = conn.execute(
        "SELECT id, filename, created, summary, metadata, roles FROM sessions WHERE id=?",
        (session_id,)
    ).fetchone()
    if not session:
        return {"error": "Session not found"}
    turns = conn.execute(
        "SELECT speaker, start, end, text, phase, intent, role FROM turns WHERE session_id=? ORDER BY start",
        (session_id,)
    ).fetchall()
    addins = conn.execute(
        "SELECT addin_name, method_version, created, result FROM addin_results WHERE session_id=? ORDER BY created",
        (session_id,)
    ).fetchall()
    return {
        "session_id": session[0],
        "filename":   session[1],
        "created":    session[2],
        "summary":    session[3],
        "metadata":   json.loads(session[4]),
        "roles":      json.loads(session[5]),
        "turns": [
            {
                "speaker": t[0], "start": t[1], "end": t[2],
                "text": t[3], "phase": t[4], "intent": t[5], "role": t[6]
            }
            for t in turns
        ],
        "addin_results": [
            {"addin_name": a[0], "method_version": a[1], "created": a[2], "result": json.loads(a[3])}
            for a in addins
        ]
    }


@app.get("/corpus/{session_id}/raw_turns")
async def api_raw_turns(session_id: int):
    # 新しい分析手法を外部で試すためのエンドポイント。diarization/Whisperを
    # やり直さずに、既に文字起こし済みの speaker/start/end/text だけを取り出せる。
    session = conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not session:
        return {"error": "Session not found"}
    turns = conn.execute(
        "SELECT id, speaker, start, end, text FROM turns WHERE session_id=? ORDER BY start",
        (session_id,)
    ).fetchall()
    return {
        "session_id": session_id,
        "turns": [
            {"turn_id": t[0], "speaker": t[1], "start": t[2], "end": t[3], "text": t[4]}
            for t in turns
        ]
    }


@app.post("/corpus/{session_id}/retag")
async def api_retag(session_id: int, req: RetagRequest):
    # 既存の文字起こし結果に対して、新しいphase/intentタグ付けロジックを
    # 適用し直す。turnsは上書きせず、結果はaddin_resultsにバージョン付きで
    # 追加保存するので、旧手法の結果と比較・再利用できる。
    session = conn.execute(
        "SELECT roles FROM sessions WHERE id=?", (session_id,)
    ).fetchone()
    if not session:
        return {"error": "Session not found"}
    roles_dict = json.loads(session[0])

    turns = conn.execute(
        "SELECT speaker, start, end, text FROM turns WHERE session_id=? ORDER BY start",
        (session_id,)
    ).fetchall()
    turn_dicts = [
        {"speaker": t[0], "start": t[1], "end": t[2], "text": t[3]}
        for t in turns
    ]

    retagged = tag_turns_with_gpt(turn_dicts, roles_dict)
    version = req.method_version or datetime.now().strftime("%Y%m%d%H%M%S")

    conn.execute(
        "INSERT INTO addin_results (session_id, addin_name, method_version, created, result) VALUES (?,?,?,?,?)",
        (
            session_id, "phase_intent_tagging", version,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            json.dumps({"turns": retagged})
        )
    )
    conn.commit()
    return {"status": "saved", "session_id": session_id, "addin_name": "phase_intent_tagging", "method_version": version}


@app.post("/corpus/{session_id}/addin")
async def api_addin_save(session_id: int, req: AddinRequest):
    session = conn.execute(
        "SELECT id FROM sessions WHERE id=?", (session_id,)
    ).fetchone()
    if not session:
        return {"error": "Session not found"}
    version = req.method_version or datetime.now().strftime("%Y%m%d%H%M%S")
    conn.execute(
        "INSERT INTO addin_results (session_id, addin_name, method_version, created, result) VALUES (?,?,?,?,?)",
        (
            session_id, req.addin_name, version,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            json.dumps(req.result)
        )
    )
    conn.commit()
    return {"status": "saved", "session_id": session_id, "addin_name": req.addin_name, "method_version": version}


# --- 5. サーバー起動 ---
!pkill -f uvicorn
!pkill -f ngrok
import time
time.sleep(2)

import subprocess, requests

NGROK_AUTH_TOKEN = userdata.get('NGROK_AUTH_TOKEN')
!ngrok config add-authtoken {NGROK_AUTH_TOKEN}

process = subprocess.Popen(['ngrok', 'http', '8000'], stdout=subprocess.PIPE)
time.sleep(5)

try:
    ngrok_data = requests.get("http://localhost:4040/api/tunnels").json()
    public_url = ngrok_data['tunnels'][0]['public_url']
    print(f"\n✅ 接続成功！")
    print(f"🔗 新しいURL: {public_url}")
    print("↑ このURLをブラウザで一度開き、『Visit Site』を押してください。")
except Exception as e:
    print(f"\n❌ URL取得失敗: {e}")

nest_asyncio.apply()
config = uvicorn.Config(app, host="0.0.0.0", port=8000, loop="asyncio")
server = uvicorn.Server(config)
await server.serve()
