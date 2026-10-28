# --- 別セル: フェーズ・意図タグ付け精度（人手 vs GPT-4o の一致率）測定ツール ---
# audio_analysis_pipeline.py のサーバーを起動していなくても、Google Driveがマウントされて
# いればこのセル単体で動く（transcription_accuracy_tool.py と同じ考え方）。
#
# 手軽な流れ（おすすめ・CSV不要）:
#   1. interactive_annotate() を実行すると、発話テキストが1件ずつ表示され、
#      番号を入力してEnterを押すだけでphase/intentを選べる
#      （Ctrl+Cで中断しても、それまでの回答は保存される）
#   2. print_agreement_report() で、GPT-4oのタグと人手ラベルの一致率・Cohen's kappaを表示する
#
# CSV経由の流れ（スプレッドシートで編集したい場合）:
#   1'. export_annotation_template() で、対象セッションの発話をランダムサンプリングし、
#       phase/intent列が空のCSVテンプレートをGoogle Driveに保存する
#   2'. そのCSVをGoogleスプレッドシート等で開き、phase/intent列を手で埋める
#       （有効な値はコンソールに表示されるPHASE_ORDER/INTENT_LABELSの中から選ぶ）
#   3'. import_annotations_from_csv() で、埋めたCSVをhuman_annotationsテーブルに保存する
#   4'. print_agreement_report() で一致率を表示する
#
# PHASE_ORDER/INTENT_LABELSとCohen's kappaの計算ロジックは、サーバー側
# （audio_analysis_pipeline.py）と同じものをここに複製している。ロジックを
# 変更した場合は両方に反映すること。

import csv
import json
import random
import sqlite3
from datetime import datetime

from google.colab import userdata

DB_PATH = "/content/drive/MyDrive/corpus.db"
SAVE_DIR = "/content/drive/MyDrive"

PHASE_ORDER = ['Introduction', 'Information Sharing', 'Conflict', 'Conclusion', 'Agreement']
INTENT_LABELS = ["Proposal", "Question", "Agreement", "Disagreement", "Confirmation", "Acknowledge", "Explanation"]


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    # サーバーを一度も起動していない状態でこのセルだけ使う場合に備えてテーブルを保証する
    conn.execute("""
        CREATE TABLE IF NOT EXISTS human_annotations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id      INTEGER,
            annotator_id TEXT,
            phase        TEXT,
            intent       TEXT,
            confidence   INTEGER,
            note         TEXT,
            created      TEXT,
            UNIQUE(turn_id, annotator_id),
            FOREIGN KEY (turn_id) REFERENCES turns(id)
        )
    """)
    conn.commit()
    return conn


def cohens_kappa(labels_a, labels_b):
    """
    2人の評定者（人手アノテーター、またはGPT-4o）が同じ発話につけたラベル列から、
    偶然の一致を差し引いたCohen's kappaを計算する。labels_a/labels_bは
    同じ長さ・同じ順序（同一turnの並び）である前提。
    """
    n = len(labels_a)
    if n == 0:
        return None
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    categories = set(labels_a) | set(labels_b)
    pe = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories)
    if pe >= 1:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def export_annotation_template(session_id: int, n: int = 30, seed: int = 42, save_dir: str = SAVE_DIR):
    """
    対象セッションの発話からn件をランダムサンプリングし、phase/intent列が空の
    CSVテンプレートを保存する。人手ラベル付けを特定の時間帯や話者に偏らせない
    ため、全発話から一様ランダムに選ぶ（GPT-4oの予測は見せない＝先入観を避ける
    ため、CSVにはAIタグを含めていない）。
    """
    conn = get_connection()
    turns = conn.execute(
        "SELECT id, speaker, start, text FROM turns WHERE session_id=? ORDER BY start",
        (session_id,)
    ).fetchall()
    conn.close()
    if not turns:
        print(f"session_id={session_id} のturnsが見つかりません。")
        return None

    rng = random.Random(seed)
    sample = turns if len(turns) <= n else rng.sample(turns, n)
    sample = sorted(sample, key=lambda t: t[2])  # 時系列順に並べ直す

    template_path = f"{save_dir}/session_{session_id}_annotation_template.csv"
    with open(template_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["turn_id", "speaker", "start", "text", "phase", "intent"])
        for turn_id, speaker, start, text in sample:
            writer.writerow([turn_id, speaker, start, text, "", ""])

    print(f"テンプレートを保存しました: {template_path}（{len(sample)}件）")
    print(f"\nphase列に入力できる値: {', '.join(PHASE_ORDER)}")
    print(f"intent列に入力できる値: {', '.join(INTENT_LABELS)}")
    print("\nこのCSVをGoogleスプレッドシート等で開き、phase/intent列を埋めてから")
    print("import_annotations_from_csv() で読み込んでください。")
    return template_path


def _prompt_choice(options):
    while True:
        raw = input(f"番号(1-{len(options)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"1〜{len(options)}の数字を入力してください。")


def interactive_annotate(session_id: int, n: int = 15, seed: int = 42, annotator_id: str = "me"):
    """
    CSV・スプレッドシートを経由せず、Colabのセル内でinput()を使って直接
    人手ラベル付けができる簡易版。発話のテキストを1件ずつ表示し、phase/intent
    を番号で選んでもらう（数字を入力してEnterを押すだけ）。ダウンロード・
    再アップロードの手間を省きたい場合はこちらを使う。

    途中でCtrl+C（中断）しても、それまでに答えた分は保存される。
    """
    conn = get_connection()
    turns = conn.execute(
        "SELECT id, speaker, start, text FROM turns WHERE session_id=? ORDER BY start",
        (session_id,)
    ).fetchall()
    conn.close()
    if not turns:
        print(f"session_id={session_id} のturnsが見つかりません。")
        return None

    rng = random.Random(seed)
    sample = turns if len(turns) <= n else rng.sample(turns, n)
    sample = sorted(sample, key=lambda t: t[2])

    print(f"{len(sample)}件の発話にラベルを付けます。各質問で番号を入力してEnterを押してください。")
    print("途中でやめたい場合はCtrl+Cで中断できます（それまでに答えた分は保存されます）。")

    annotations = []
    try:
        for idx, (turn_id, speaker, start, text) in enumerate(sample, 1):
            print(f"\n--- [{idx}/{len(sample)}] turn_id={turn_id} ({speaker}, {start:.1f}秒) ---")
            print(f"発話内容: {text}")

            print("\nフェーズを選んでください:")
            for i, label in enumerate(PHASE_ORDER, 1):
                print(f"  {i}. {label}")
            phase = _prompt_choice(PHASE_ORDER)

            print("\n意図を選んでください:")
            for i, label in enumerate(INTENT_LABELS, 1):
                print(f"  {i}. {label}")
            intent = _prompt_choice(INTENT_LABELS)

            annotations.append({"turn_id": turn_id, "phase": phase, "intent": intent})
    except KeyboardInterrupt:
        print(f"\n中断しました。ここまでの{len(annotations)}件を保存します。")

    if not annotations:
        print("保存するラベルがありません。")
        return None

    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for a in annotations:
        conn.execute(
            """INSERT INTO human_annotations (turn_id, annotator_id, phase, intent, created)
               VALUES (?,?,?,?,?)
               ON CONFLICT(turn_id, annotator_id) DO UPDATE SET
                   phase=excluded.phase, intent=excluded.intent, created=excluded.created""",
            (a["turn_id"], annotator_id, a["phase"], a["intent"], now)
        )
    conn.commit()
    conn.close()

    print(f"\n{len(annotations)}件のラベルを保存しました（annotator_id={annotator_id}）。")
    return annotations


def import_annotations_from_csv(session_id: int, csv_path: str, annotator_id: str):
    """
    export_annotation_template() で作ったCSV（phase/intent列を埋めたもの）を
    読み込み、human_annotationsテーブルに保存する。phase/intentがPHASE_ORDER/
    INTENT_LABELSに含まれない値や空欄の行はスキップする（有効な値だけを使って
    一致率を計算するため）。
    """
    conn = get_connection()
    valid_turn_ids = {
        row[0] for row in conn.execute(
            "SELECT id FROM turns WHERE session_id=?", (session_id,)
        ).fetchall()
    }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    saved = 0
    skipped = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            turn_id = int(row["turn_id"])
            phase = row.get("phase", "").strip()
            intent = row.get("intent", "").strip()
            if turn_id not in valid_turn_ids:
                skipped.append((turn_id, "turn_idが存在しない"))
                continue
            if phase not in PHASE_ORDER or intent not in INTENT_LABELS:
                skipped.append((turn_id, f"phase/intentが未入力または不正な値(phase={phase!r}, intent={intent!r})"))
                continue
            conn.execute(
                """INSERT INTO human_annotations (turn_id, annotator_id, phase, intent, created)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(turn_id, annotator_id) DO UPDATE SET
                       phase=excluded.phase, intent=excluded.intent, created=excluded.created""",
                (turn_id, annotator_id, phase, intent, now)
            )
            saved += 1
    conn.commit()
    conn.close()

    print(f"{saved}件を保存しました（annotator_id={annotator_id}）。")
    if skipped:
        print(f"{len(skipped)}件をスキップしました:")
        for turn_id, reason in skipped:
            print(f"  turn_id={turn_id}: {reason}")
    return {"saved": saved, "skipped": skipped}


def get_agreement(session_id: int):
    """
    GPT-4o（turnsテーブルのphase/intent）と、human_annotationsに登録された
    各アノテーターとの一致率（%一致 + Cohen's kappa）を、phase/intentそれぞれ
    について算出する。両者がラベル付けした発話の共通部分だけを比較する。
    """
    conn = get_connection()
    ai_rows = conn.execute(
        "SELECT id, phase, intent FROM turns WHERE session_id=?", (session_id,)
    ).fetchall()
    human_rows = conn.execute(
        """SELECT ha.annotator_id, ha.turn_id, ha.phase, ha.intent
           FROM human_annotations ha
           JOIN turns t ON t.id = ha.turn_id
           WHERE t.session_id = ?""",
        (session_id,)
    ).fetchall()
    conn.close()

    raters = {"GPT-4o": {r[0]: (r[1], r[2]) for r in ai_rows}}
    for annotator_id, turn_id, phase, intent in human_rows:
        raters.setdefault(annotator_id, {})[turn_id] = (phase, intent)

    rater_ids = list(raters.keys())
    pairwise_agreement = []
    for i in range(len(rater_ids)):
        for j in range(i + 1, len(rater_ids)):
            rater_a, rater_b = rater_ids[i], rater_ids[j]
            for label_index, label_type in [(0, "phase"), (1, "intent")]:
                shared_turns = [
                    tid for tid in raters[rater_a]
                    if tid in raters[rater_b]
                    and raters[rater_a][tid][label_index] is not None
                    and raters[rater_b][tid][label_index] is not None
                ]
                if not shared_turns:
                    continue
                labels_a = [raters[rater_a][t][label_index] for t in shared_turns]
                labels_b = [raters[rater_b][t][label_index] for t in shared_turns]
                percent_agreement = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / len(shared_turns)
                pairwise_agreement.append({
                    "rater_1": rater_a,
                    "rater_2": rater_b,
                    "label_type": label_type,
                    "n": len(shared_turns),
                    "percent_agreement": round(percent_agreement, 3),
                    "cohens_kappa": round(cohens_kappa(labels_a, labels_b), 3)
                })

    return {"session_id": session_id, "raters": rater_ids, "pairwise_agreement": pairwise_agreement}


# Cohen's kappaの目安（Landis & Koch, 1977 の慣用的な解釈）
_KAPPA_GUIDE = [
    (0.81, "ほぼ完全に一致"),
    (0.61, "かなり一致"),
    (0.41, "中程度に一致"),
    (0.21, "わずかに一致"),
    (0.00, "ほとんど一致しない"),
]


def _kappa_label(kappa):
    for threshold, label in _KAPPA_GUIDE:
        if kappa >= threshold:
            return label
    return "一致以下（偶然より悪い）"


def get_retag_consistency(session_id: int, method_version: str):
    """
    人手ラベルを用意しなくても自動で測れる、GPT-4oの自己一貫性(test-retest
    reliability)チェック。サーバーの POST /corpus/{session_id}/retag を
    実行して同じ発話にもう一度タグ付けさせ(addin_resultsに保存される)、
    その結果と1回目のタグ付け(turnsテーブル)をCohen's kappaで比較する。

    注意: これは「正解と合っているか」ではなく「同じモデルが毎回同じ判断を
    するか」を測るものに過ぎない。一致率が低ければ「このタスクは判断が
    曖昧・不安定」という警告になるが、一致率が高くても両方とも間違っている
    可能性は否定できない(本当の正解率を知りたい場合は人手ラベルが必要)。

    事前にColab外(curl等)かrequestsで下記を実行し、method_versionを
    控えておくこと:
        POST {ngrok_url}/corpus/{session_id}/retag
        -> レスポンスの "method_version" を引数に渡す
    """
    conn = get_connection()
    original = conn.execute(
        "SELECT id, start, phase, intent FROM turns WHERE session_id=? ORDER BY start",
        (session_id,)
    ).fetchall()
    retag_row = conn.execute(
        """SELECT result FROM addin_results
           WHERE session_id=? AND addin_name='phase_intent_tagging' AND method_version=?""",
        (session_id, method_version)
    ).fetchone()
    conn.close()

    if not retag_row:
        print(f"method_version={method_version} の再タグ付け結果が見つかりません。"
              f"先に POST /corpus/{session_id}/retag を実行してください。")
        return None

    retagged_turns = json.loads(retag_row[0])["turns"]
    if len(original) != len(retagged_turns):
        print(f"件数が一致しません(turns={len(original)}件, retag結果={len(retagged_turns)}件)。"
              "turnsが変化した後の再タグ付けの可能性があるため比較できません。")
        return None

    labels_a_phase = [t[2] for t in original]
    labels_b_phase = [t["phase"] for t in retagged_turns]
    labels_a_intent = [t[3] for t in original]
    labels_b_intent = [t["intent"] for t in retagged_turns]

    n = len(original)
    phase_kappa = cohens_kappa(labels_a_phase, labels_b_phase)
    intent_kappa = cohens_kappa(labels_a_intent, labels_b_intent)
    phase_agree = sum(1 for a, b in zip(labels_a_phase, labels_b_phase) if a == b) / n
    intent_agree = sum(1 for a, b in zip(labels_a_intent, labels_b_intent) if a == b) / n

    print(f"=== セッション{session_id} GPT-4o自己一貫性（1回目タグ付け vs retag） ===")
    print(f"（n={n}発話、1回目と{method_version}の2回のタグ付け結果を比較）\n")
    print(f"phase : 一致率={phase_agree:.1%}, Cohen's kappa={phase_kappa:.3f} ({_kappa_label(phase_kappa)})")
    print(f"intent: 一致率={intent_agree:.1%}, Cohen's kappa={intent_kappa:.3f} ({_kappa_label(intent_kappa)})")
    return {
        "n": n,
        "phase": {"percent_agreement": round(phase_agree, 3), "cohens_kappa": round(phase_kappa, 3)},
        "intent": {"percent_agreement": round(intent_agree, 3), "cohens_kappa": round(intent_kappa, 3)},
    }


def print_agreement_report(session_id: int):
    data = get_agreement(session_id)
    if not data["pairwise_agreement"]:
        print("比較できるデータがありません。人手ラベルを登録してから実行してください。")
        return data

    print(f"=== セッション{session_id} フェーズ・意図タグ付け 一致率レポート ===\n")
    for m in data["pairwise_agreement"]:
        print(
            f"[{m['label_type']}] {m['rater_1']} vs {m['rater_2']} "
            f"(n={m['n']}): 一致率={m['percent_agreement']:.1%}, "
            f"Cohen's kappa={m['cohens_kappa']} ({_kappa_label(m['cohens_kappa'])})"
        )
    return data


# --- 使い方の例（実行する場合はコメントを外す） ---

# --- 手軽な流れ(おすすめ・CSV不要) ---
#
# 1. 最新セッションを対象に、Colab上で直接ラベル付けする
#    （テキストが1件ずつ表示され、番号を入力してEnterを押すだけ）
# _conn = sqlite3.connect(DB_PATH)
# _latest_session_id = _conn.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1").fetchone()[0]
# _conn.close()
# interactive_annotate(session_id=_latest_session_id, n=15, annotator_id="researcher_A")

# 2. GPT-4oと人手ラベルの一致率・Cohen's kappaを表示する
# print_agreement_report(session_id=_latest_session_id)

# --- CSV経由の流れ(スプレッドシートで編集したい場合) ---
#
# 1'. 最新セッションから30件をランダムサンプリングしてCSVテンプレートを出力する
# _conn = sqlite3.connect(DB_PATH)
# _latest_session_id = _conn.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1").fetchone()[0]
# _conn.close()
# export_annotation_template(session_id=_latest_session_id, n=30)

# 2'. 出力されたCSVをGoogleスプレッドシート等で開き、phase/intent列を手で埋める
#    （有効な値は上記の出力に表示される）

# 3'. 埋めたCSVを読み込んで保存する（annotator_idは自分の名前など）
# import_annotations_from_csv(
#     session_id=_latest_session_id,
#     csv_path="/content/drive/MyDrive/session_X_annotation_template.csv",
#     annotator_id="researcher_A"
# )

# 4'. GPT-4oと人手ラベルの一致率・Cohen's kappaを表示する
# print_agreement_report(session_id=_latest_session_id)

# --- 人手を使わない自動チェック: GPT-4oの自己一貫性 ---
#
# 5. まずColab外(ブラウザ等)かrequestsで、対象セッションの再タグ付けを実行する
#    （ngrok_urlはColabに表示されている公開URLに置き換える）
# import requests
# retag_res = requests.post(f"{ngrok_url}/corpus/{_latest_session_id}/retag").json()
# print(retag_res)  # {"status": "saved", ..., "method_version": "..."} のmethod_versionを控える

# 6. 1回目のタグ付け結果とretag結果の一致率・Cohen's kappaを表示する
#    （人手ラベルは一切不要。ただし「正解と合っているか」ではなく
#    「GPT-4oが毎回同じ判断をするか」を測るものである点に注意）
# get_retag_consistency(session_id=_latest_session_id, method_version="（5.で控えたmethod_version）")

