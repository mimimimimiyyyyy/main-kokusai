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
# プロンプト改善の効果を測る流れ（サーバー/ngrok不要）:
#   run_standalone_retag() で、audio_analysis_pipeline.pyのサーバーを起動・再起動
#   せずにこのセルから直接phase/intentの再タグ付けができる（corpus.dbを直接読み書き
#   するだけなので、ngrokのURLずれやポート競合といったトラブルを回避できる）。
#   結果はaddin_resultsにmethod_version付きで保存されるので、
#   get_agreement_for_retag/print_retag_agreement_reportでそのまま比較できる。
#
# PHASE_ORDER/INTENT_LABELSとCohen's kappaの計算ロジックは、サーバー側
# （audio_analysis_pipeline.py）と同じものをここに複製している。ロジックを
# 変更した場合は両方に反映すること。

import csv
import json
import random
import sqlite3
from datetime import datetime

import openai
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


def _sample_turns_with_context(session_id, n, seed, min_chars, exclude_turn_ids=None):
    """
    対象セッションの発話を取得し、min_chars文字未満の短すぎる発話
    （相槌の断片など、それだけでは意図・フェーズを判断しづらいもの）と、
    exclude_turn_idsに含まれる発話（既にラベル付け済みのものなど）を
    候補から除外したうえでn件をランダムサンプリングする。
    戻り値: (サンプル[(turn_id, speaker, start, text), ...], 全発話の順序付きID,
             turn_id -> (turn_id, speaker, start, text) の辞書)
    GPT-4o自身は実際にはブロック単位(前後の発話も含む)で判断しているため、
    文脈表示用に全発話も一緒に返す。
    """
    conn = get_connection()
    all_turns = conn.execute(
        "SELECT id, speaker, start, text FROM turns WHERE session_id=? ORDER BY start",
        (session_id,)
    ).fetchall()
    conn.close()
    if not all_turns:
        return None, None, None

    ordered_ids = [t[0] for t in all_turns]
    turns_by_id = {t[0]: t for t in all_turns}

    candidates = [t for t in all_turns if len(t[3]) >= min_chars]
    if exclude_turn_ids:
        candidates = [t for t in candidates if t[0] not in exclude_turn_ids]
    if not candidates:
        return [], ordered_ids, turns_by_id

    rng = random.Random(seed)
    sample = candidates if len(candidates) <= n else rng.sample(candidates, n)
    sample = sorted(sample, key=lambda t: t[2])
    return sample, ordered_ids, turns_by_id


def _context_window(turn_id, ordered_ids, turns_by_id, context_size):
    """turn_idの前後context_size件ずつの発話を(speaker, text, is_target)のリストで返す。"""
    pos = ordered_ids.index(turn_id)
    window_start = max(0, pos - context_size)
    window_end = min(len(ordered_ids), pos + context_size + 1)
    return [
        (turns_by_id[wid][1], turns_by_id[wid][3], wid == turn_id)
        for wid in ordered_ids[window_start:window_end]
    ]


def export_annotation_template(session_id: int, n: int = 30, seed: int = 42, save_dir: str = SAVE_DIR,
                                min_chars: int = 8, context_size: int = 2, annotator_id: str = None):
    """
    対象セッションの発話からn件をランダムサンプリングし、phase/intent列が空の
    CSVテンプレートを保存する。人手ラベル付けを特定の時間帯や話者に偏らせない
    ため、全発話から一様ランダムに選ぶ（GPT-4oの予測は見せない＝先入観を避ける
    ため、CSVにはAIタグを含めていない）。

    "is"のような1語だけの発話は、それだけでは意図・フェーズを判断しづらいため、
    min_chars文字未満の発話はサンプリング候補から除外する。また、GPT-4o自身は
    実際にはブロック単位(前後の発話も含む)で判断しているため、前後
    context_size件ずつの発話をcontext列にまとめて載せ、人手でも同じように
    文脈込みで判断できるようにする（ラベル付けの対象はtext列の発話のみ）。

    annotator_idを指定すると、そのアノテーターが既にラベル付け済みの発話は
    候補から除外する（サンプル数を段階的に増やしていく際に、同じ発話を
    何度も見せないようにするため）。
    """
    exclude_turn_ids = None
    if annotator_id:
        conn = get_connection()
        exclude_turn_ids = {
            row[0] for row in conn.execute(
                """SELECT ha.turn_id FROM human_annotations ha JOIN turns t ON t.id = ha.turn_id
                   WHERE ha.annotator_id=? AND t.session_id=?""",
                (annotator_id, session_id)
            ).fetchall()
        }
        conn.close()

    sample, ordered_ids, turns_by_id = _sample_turns_with_context(session_id, n, seed, min_chars, exclude_turn_ids)
    if sample is None:
        print(f"session_id={session_id} のturnsが見つかりません。")
        return None
    if not sample:
        print(f"min_chars={min_chars}文字以上の未ラベルの発話がありません。")
        return None

    template_path = f"{save_dir}/session_{session_id}_annotation_template.csv"
    with open(template_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["turn_id", "speaker", "start", "text", "context", "phase", "intent"])
        for turn_id, speaker, start, text in sample:
            window = _context_window(turn_id, ordered_ids, turns_by_id, context_size)
            context = " / ".join(
                f"{'[対象]' if is_target else ''}{w_speaker}: {w_text}"
                for w_speaker, w_text, is_target in window
            )
            writer.writerow([turn_id, speaker, start, text, context, "", ""])

    print(f"テンプレートを保存しました: {template_path}（{len(sample)}件、{min_chars}文字未満の発話は除外済み）")
    print(f"\nphase列に入力できる値: {', '.join(PHASE_ORDER)}")
    print(f"intent列に入力できる値: {', '.join(INTENT_LABELS)}")
    print("\nこのCSVをGoogleスプレッドシート等で開き、context列（前後の発話）も参考にしながら")
    print("phase/intent列を埋めてから import_annotations_from_csv() で読み込んでください。")
    return template_path


def _prompt_choice(options):
    while True:
        raw = input(f"番号(1-{len(options)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"1〜{len(options)}の数字を入力してください。")


def interactive_annotate(session_id: int, n: int = 15, seed: int = 42, annotator_id: str = "me",
                          min_chars: int = 8, context_size: int = 2, skip_already_labeled: bool = True):
    """
    CSV・スプレッドシートを経由せず、Colabのセル内でinput()を使って直接
    人手ラベル付けができる簡易版。発話のテキストを1件ずつ表示し、phase/intent
    を番号で選んでもらう（数字を入力してEnterを押すだけ）。ダウンロード・
    再アップロードの手間を省きたい場合はこちらを使う。

    "is"のような1語だけの発話は、それだけでは意図・フェーズを判断しづらいため、
    min_chars文字未満の発話はサンプリング候補から除外する。また、GPT-4o自身は
    実際にはブロック単位(前後の発話も含む)で判断しているため、前後
    context_size件ずつの発話も文脈として一緒に表示する（ラベル付けの対象は
    【対象】と表示された発話のみ）。

    skip_already_labeled=True（デフォルト）の場合、annotator_idが既に
    ラベル付け済みの発話は候補から除外する。これにより、nを大きくしながら
    同じ関数を繰り返し呼び出すだけで、サンプル数を段階的に増やしていける
    （例: 最初にn=18で呼び、後からn=50で呼べば、新たに約50件の未ラベル発話
    だけが対象になる＝合計で約68件まで増える）。

    途中でCtrl+C（中断）しても、それまでに答えた分は保存される。
    """
    exclude_turn_ids = None
    if skip_already_labeled:
        conn = get_connection()
        exclude_turn_ids = {
            row[0] for row in conn.execute(
                """SELECT ha.turn_id FROM human_annotations ha JOIN turns t ON t.id = ha.turn_id
                   WHERE ha.annotator_id=? AND t.session_id=?""",
                (annotator_id, session_id)
            ).fetchall()
        }
        conn.close()

    sample, ordered_ids, turns_by_id = _sample_turns_with_context(session_id, n, seed, min_chars, exclude_turn_ids)
    if sample is None:
        print(f"session_id={session_id} のturnsが見つかりません。")
        return None
    if not sample:
        print(f"min_chars={min_chars}文字以上の未ラベルの発話がありません。")
        return None

    print(f"{len(sample)}件の発話にラベルを付けます（{min_chars}文字未満の短い発話・既にラベル済みの発話は除外済み）。")
    print("前後の発話も文脈として表示します（ラベル付けの対象は【対象】の行だけです）。")
    print("各質問で番号を入力してEnterを押してください。Ctrl+Cで中断してもそれまでの回答は保存されます。")

    annotations = []
    try:
        for idx, (turn_id, speaker, start, text) in enumerate(sample, 1):
            window = _context_window(turn_id, ordered_ids, turns_by_id, context_size)

            print(f"\n--- [{idx}/{len(sample)}] turn_id={turn_id} ({start:.1f}秒) ---")
            for w_speaker, w_text, is_target in window:
                marker = "【対象】" if is_target else "　　　"
                print(f"{marker} {w_speaker}: {w_text}")

            print("\nフェーズを選んでください（【対象】の発話について）:")
            for i, label in enumerate(PHASE_ORDER, 1):
                print(f"  {i}. {label}")
            phase = _prompt_choice(PHASE_ORDER)

            print("\n意図を選んでください（【対象】の発話について）:")
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


def _load_retag_labels(session_id: int, method_version: str):
    """
    addin_resultsのturnsはspeaker/start/end/textのみで保存されており、turn_idを
    含まないため、turnsテーブルの並び順(start昇順)と対応付けて逆引きする。
    /retagはturnsを順番通りそのまま再タグ付けするため、件数・順序が一致する前提。
    見つからない・件数が合わない場合はNoneを返す(エラーメッセージは表示する)。
    """
    conn = get_connection()
    retag_row = conn.execute(
        """SELECT result FROM addin_results
           WHERE session_id=? AND addin_name='phase_intent_tagging' AND method_version=?""",
        (session_id, method_version)
    ).fetchone()
    original_order = conn.execute(
        "SELECT id FROM turns WHERE session_id=? ORDER BY start", (session_id,)
    ).fetchall()
    conn.close()

    if not retag_row:
        print(f"method_version={method_version} の再タグ付け結果が見つかりません。"
              f"先に run_standalone_retag(method_version={method_version!r}, ...) を実行してください。")
        return None

    retagged_turns = json.loads(retag_row[0])["turns"]
    original_ids = [r[0] for r in original_order]
    if len(original_ids) != len(retagged_turns):
        print(f"件数が一致しません(turns={len(original_ids)}件, retag結果={len(retagged_turns)}件)。"
              "turnsが変化した後の再タグ付けの可能性があるため比較できません。")
        return None

    return {
        turn_id: (t.get("phase"), t.get("intent"))
        for turn_id, t in zip(original_ids, retagged_turns)
    }


def get_agreement_for_retag(session_id: int, method_version: str, label: str = None):
    """
    get_agreement()はturnsテーブルの最初のタグ付け結果とhuman_annotationsを
    比較するが、/corpus/{id}/retagの結果はturnsを上書きせずaddin_resultsに
    別バージョンとして保存される。そのため、プロンプト改善(chain-of-thought等)
    を試した後の結果を人手ラベルと比較するには、この関数でaddin_resultsの
    指定バージョンを読み込んで比較する必要がある。

    labelは比較結果に付けるラベル（例: "CoT追加後"）。省略時はmethod_versionを使う。
    """
    retagged = _load_retag_labels(session_id, method_version)
    if retagged is None:
        return None

    conn = get_connection()
    human_rows = conn.execute(
        """SELECT ha.annotator_id, ha.turn_id, ha.phase, ha.intent
           FROM human_annotations ha
           JOIN turns t ON t.id = ha.turn_id
           WHERE t.session_id = ?""",
        (session_id,)
    ).fetchall()
    conn.close()

    retag_label = label or method_version
    raters = {retag_label: retagged}
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


def _ensure_addin_results_table(conn):
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
    cols = {row[1] for row in conn.execute("PRAGMA table_info(addin_results)")}
    if "method_version" not in cols:
        conn.execute("ALTER TABLE addin_results ADD COLUMN method_version TEXT")


def _normalize_phase(p):
    p = str(p)
    return p if p in PHASE_ORDER else 'Information Sharing'


def _normalize_intent(i):
    i = str(i)
    return i if i in INTENT_LABELS else 'Explanation'


_PHASE_INTENT_DEFS = """
=== PHASE DEFINITIONS ===
- "Introduction"       : Opening, greetings, topic setting, agenda announcement
- "Information Sharing": Presenting facts, explaining background, reporting status
- "Conflict"           : Disagreement or tension BETWEEN THE SPEAKERS in this conversation
- "Conclusion"         : Summarizing what was discussed, wrapping up a topic
- "Agreement"          : Reaching consensus, confirming a shared decision

CRITICAL RULES:
* A real conversation MUST contain multiple phases. Do NOT assign "Information Sharing" to everything.
* Early utterances  → likely "Introduction"
* Middle utterances → "Information Sharing", "Conflict", or "Agreement" depending on content
* Final utterances  → "Conclusion" or "Agreement"
* "Conflict" requires one speaker pushing back against, or disagreeing with, ANOTHER
  SPEAKER'S idea in this conversation. Describing opposing teams, "sides", or the rules
  of a game/activity (e.g. "we're all on one side, you're all on the other side") is NOT
  Conflict by itself — that is Information Sharing, unless a speaker is also disagreeing
  with another speaker about it.
* A neutral question for clarification is NOT "Conflict"
* If an utterance OPENS with an agreement marker (yeah, that's true, I agree, sounds
  good, exactly, right) → classify it as "Agreement", even if the speaker then adds more
  detail or explanation afterward in the same utterance.
* If someone says ok / agreed / let's do that → use "Agreement"

=== INTENT DEFINITIONS ===
- "Proposal"      : Suggesting a new idea or action FOR THE GROUP TO DO
  (signals: "let's...", "we should...", "what if we...", "how about...")
- "Question"      : Asking for information or clarification
- "Agreement"     : Expressing approval or consent
- "Disagreement"  : Expressing opposition or doubt
- "Confirmation"  : Verifying or checking understanding
- "Acknowledge"   : Short responses showing attention (yeah, I see, mm-hmm)
- "Explanation"   : Describing facts, rules, or background WITHOUT suggesting any new
  action (e.g. explaining how a game works, or adding detail to something already
  proposed earlier) — if no action is being suggested, prefer "Explanation" over "Proposal"
"""

_PHASE_INTENT_DEFS_WITH_EXAMPLES = """
=== PHASE DEFINITIONS AND EXAMPLES ===
- "Introduction"       : Opening, greetings, topic setting, agenda announcement
  e.g. "Okay, let's get started. Today we need to finalize the Q3 roadmap and the client proposal."
- "Information Sharing": Presenting facts, explaining background, reporting status
  e.g. "The current conversion rate on the landing page is about 3.2%, slightly below our 4% target."
- "Conflict"           : Disagreement, opposing opinions, challenging someone's idea, tension
  e.g. "I don't think that's the right approach. We tried something similar last quarter and it didn't work."
- "Conclusion"         : Summarizing what was discussed, wrapping up a topic
  e.g. "So to summarize, we're moving forward with option B and the deadline is next Friday."
- "Agreement"          : Reaching consensus, confirming a shared decision
  e.g. "Yeah, that works for me. Let's go with that plan."

CRITICAL RULES:
* A real conversation MUST contain multiple phases. Do NOT assign "Information Sharing" to everything.
* Early utterances  → likely "Introduction"
* Middle utterances → "Information Sharing", "Conflict", or "Agreement" depending on content
* Final utterances  → "Conclusion" or "Agreement"
* If there is TENSION or OPPOSITION in the exchange -> use "Conflict"
* A neutral question for clarification is NOT "Conflict"
* If someone says ok / agreed / let's do that → use "Agreement"

=== INTENT DEFINITIONS AND EXAMPLES ===
- "Proposal"      : Suggesting a new idea or action
  e.g. "What if we moved the deadline up by a week to give us more buffer time?"
- "Question"      : Asking for information or clarification
  e.g. "How many users have we onboarded so far this month?"
- "Agreement"     : Expressing approval or consent
  e.g. "Sounds good, I'm on board with that."
- "Disagreement"  : Expressing opposition or doubt
  e.g. "I'm not sure I agree with that assessment, honestly."
- "Confirmation"  : Verifying or checking understanding
  e.g. "So just to confirm, we're sending the revised draft by Thursday, right?"
- "Acknowledge"   : Short responses showing attention (yeah, I see, mm-hmm)
  e.g. "Yeah, okay, got it."
- "Explanation"   : Elaborating or clarifying a point
  e.g. "The reason the numbers dropped is because we changed the pricing tier in August."

Use the examples above only as a guide to each category's judgment criteria —
do not default to "Information Sharing" / "Explanation" just because you are
unsure; actively compare the utterance against ALL categories' examples first.
"""


_GENERIC_PHASE_EXAMPLES = {
    'Introduction': "Okay, let's get started. Today we need to finalize the Q3 roadmap and the client proposal.",
    'Information Sharing': "The current conversion rate on the landing page is about 3.2%, slightly below our 4% target.",
    'Conflict': "I don't think that's the right approach. We tried something similar last quarter and it didn't work.",
    'Conclusion': "So to summarize, we're moving forward with option B and the deadline is next Friday.",
    'Agreement': "Yeah, that works for me. Let's go with that plan.",
}
_GENERIC_INTENT_EXAMPLES = {
    'Proposal': "What if we moved the deadline up by a week to give us more buffer time?",
    'Question': "How many users have we onboarded so far this month?",
    'Agreement': "Sounds good, I'm on board with that.",
    'Disagreement': "I'm not sure I agree with that assessment, honestly.",
    'Confirmation': "So just to confirm, we're sending the revised draft by Thursday, right?",
    'Acknowledge': "Yeah, okay, got it.",
    'Explanation': "The reason the numbers dropped is because we changed the pricing tier in August.",
}


def _get_indomain_examples(session_id: int):
    """
    汎用的な(業務会議風の)few-shot例文が、このセッションの実際の内容
    (雑談・ゲームの相談など)と噛み合わず精度を下げていた可能性がある
    (fewshot手法がベースラインより悪化した一因と見られる)。そこで、
    このセッション自身の人手ラベル済み発話から、カテゴリごとに実例を
    1件ずつ抽出し、ドメインが一致したfew-shot例文として使う。
    戻り値: (phase_examples: {label: text}, intent_examples: {label: text})
    """
    conn = get_connection()
    rows = conn.execute(
        """SELECT ha.phase, ha.intent, t.text
           FROM human_annotations ha JOIN turns t ON t.id = ha.turn_id
           WHERE t.session_id = ?""",
        (session_id,)
    ).fetchall()
    conn.close()

    phase_examples, intent_examples = {}, {}
    for phase, intent, text in rows:
        text = (text or "").strip()
        if len(text) < 15:
            continue
        if phase and phase not in phase_examples:
            phase_examples[phase] = text
        if intent and intent not in intent_examples:
            intent_examples[intent] = text
    return phase_examples, intent_examples


def _build_indomain_defs(phase_examples, intent_examples):
    phase_lines = []
    for label in PHASE_ORDER:
        text = phase_examples.get(label) or _GENERIC_PHASE_EXAMPLES.get(label, "")
        source = "" if label in phase_examples else "(汎用例)"
        phase_lines.append(f'- "{label}"\n  e.g. "{text}" {source}'.rstrip())
    intent_lines = []
    for label in INTENT_LABELS:
        text = intent_examples.get(label) or _GENERIC_INTENT_EXAMPLES.get(label, "")
        source = "" if label in intent_examples else "(汎用例)"
        intent_lines.append(f'- "{label}"\n  e.g. "{text}" {source}'.rstrip())
    return (
        "\n=== PHASE EXAMPLES (from THIS conversation where available) ===\n"
        + "\n".join(phase_lines)
        + "\n\n=== INTENT EXAMPLES (from THIS conversation where available) ===\n"
        + "\n".join(intent_lines) + "\n"
    )


def _build_tag_prompt(prompt_variant, i, block, total, examples_block=None):
    """
    アブレーション用に、プロンプトのバリエーションをここで切り替える。
    - "baseline": 元のプロンプト(reasoning無し、例文無し)。人手ラベルn=68との
      比較で、判断に迷うと"Information Sharing"/"Explanation"に逃げる傾向が
      確認された(phase kappa=0.146)。
    - "fewshot" : 汎用的な(業務会議風の)具体例をプロンプトに埋め込む方式。
      試した結果、ベースラインより悪化した(phase kappa=0.01)。このセッションの
      実際の内容(雑談)と例文のドメインが噛み合っていなかったことが一因と見られる。
    - "cot"     : ラベルを決める前に短い理由(reasoning)を書かせる
      chain-of-thought方式。JSONはキーの出現順に生成されるため、reasoningを
      phase/intentより先に置くことで、対立・結論・合意の兆候を実際に検討
      させてからラベルを選ばせる狙い。
    - "cot_indomain": cotに加えて、このセッション自身の人手ラベル済み発話から
      抽出したin-domainのfew-shot例文(examples_block、_get_indomain_examples
      参照)を埋め込む。fewshotの失敗がドメイン不一致によるものなら、
      ドメインを一致させることで改善するはずだという仮説の検証。
    """
    if prompt_variant == "cot_indomain":
        return f"""
You are a dialogue analysis expert. Label each utterance with a phase and intent.
{_PHASE_INTENT_DEFS}
{examples_block or ""}
Data (utterances {i} to {i + len(block) - 1} of {total} total):
{json.dumps(block, ensure_ascii=False)}

For each utterance, first briefly explain (1 short sentence) what in the content
points to a specific phase and intent — e.g. is there tension/pushback (Conflict),
a wrap-up (Conclusion), a yes/agreed (Agreement)? Only after that reasoning,
commit to the phase and intent labels. Do not default to "Information Sharing" /
"Explanation" just because you are unsure — use the reasoning step to actively
check for signs of the other categories first.

Return JSON with key 'analysis' containing a list of
{{"reasoning": "...", "phase": ..., "intent": ...}}
for each utterance in order (reasoning must come first in each object).
        """
    if prompt_variant == "cot":
        return f"""
You are a dialogue analysis expert. Label each utterance with a phase and intent.
{_PHASE_INTENT_DEFS}
Data (utterances {i} to {i + len(block) - 1} of {total} total):
{json.dumps(block, ensure_ascii=False)}

For each utterance, first briefly explain (1 short sentence) what in the content
points to a specific phase and intent — e.g. is there tension/pushback (Conflict),
a wrap-up (Conclusion), a yes/agreed (Agreement)? Only after that reasoning,
commit to the phase and intent labels. Do not default to "Information Sharing" /
"Explanation" just because you are unsure — use the reasoning step to actively
check for signs of the other categories first.

Return JSON with key 'analysis' containing a list of
{{"reasoning": "...", "phase": ..., "intent": ...}}
for each utterance in order (reasoning must come first in each object).
        """
    if prompt_variant == "fewshot":
        return f"""
You are a dialogue analysis expert. Label each utterance with a phase and intent.
{_PHASE_INTENT_DEFS_WITH_EXAMPLES}
Data (utterances {i} to {i + len(block) - 1} of {total} total):
{json.dumps(block, ensure_ascii=False)}

Return JSON with key 'analysis' containing a list of
{{"phase": ..., "intent": ...}}
for each utterance in order.
        """
    # baseline
    return f"""
You are a dialogue analysis expert. Label each utterance with a phase and intent.
{_PHASE_INTENT_DEFS}
Data (utterances {i} to {i + len(block) - 1} of {total} total):
{json.dumps(block, ensure_ascii=False)}

Return JSON with key 'analysis' containing a list of {{"phase": ..., "intent": ...}} for each utterance in order.
        """


def _merge_adjacent_turns(turns, max_gap_sec):
    """
    話者分離由来のturnは、同じ話者が短い間を挟んで続けて話しているだけでも
    別々のturnに分割されてしまう。これが「to 10 people.」のような数語の断片に
    なり、GPT・人間の双方にとって文脈が読み取りにくい原因になっていた
    (print_confusion_breakdownの不一致例の多くが断片的な発話だった)。

    タグ付けの判断材料として、同じ話者・短い間隔(max_gap_sec以内)で連続する
    turnを1つのまとまりに結合してから渡す。結合後のまとまり単位で1つの
    phase/intentを決め、そのまとまりに含まれる元のturn全部に同じラベルを
    適用することで、addin_resultsの件数・順序は元のturns件数のまま保つ。

    戻り値: [{"speaker", "start", "end", "text", "member_indices": [...]}, ...]
    member_indicesは、このグループに含まれる元のturnsリスト内でのインデックス。
    """
    groups = []
    current = None
    for idx, t in enumerate(turns):
        if (current is not None
                and current["speaker"] == t["speaker"]
                and t["start"] - current["end"] <= max_gap_sec):
            current["text"] = (current["text"] + " " + t["text"]).strip()
            current["end"] = t["end"]
            current["member_indices"].append(idx)
        else:
            if current:
                groups.append(current)
            current = {
                "speaker": t["speaker"], "start": t["start"], "end": t["end"],
                "text": t["text"], "member_indices": [idx]
            }
    if current:
        groups.append(current)
    return groups


def run_standalone_retag(session_id: int, method_version: str, prompt_variant: str = "fewshot",
                          block_size: int = 40, merge_gap_sec: float = None):
    """
    audio_analysis_pipeline.pyのサーバー(ngrok経由)を一切使わず、このセルから
    直接corpus.dbを読み書きしてphase/intentの再タグ付けを行う。サーバー起動・
    ngrokのURLずれ・ポート競合といったColab特有のトラブルを避けるために、
    /corpus/{id}/retagと同じロジックをここに複製した。

    prompt_variantで"baseline"/"fewshot"/"cot"/"cot_indomain"を切り替えられる
    (_build_tag_prompt参照)。複数の手法を同じ人手ラベルと比較するアブレーション
    実験のためのもの。"cot_indomain"は、このセッション自身の人手ラベル済み発話から
    few-shot例文を自動抽出するため、_get_indomain_examplesを事前に呼ぶ
    (人手ラベルがまだ無いとカテゴリによっては汎用例文にフォールバックする)。

    merge_gap_secを指定すると、_merge_adjacent_turnsで同じ話者の断片的なturnを
    結合してからタグ付けする(例: 1.5なら1.5秒以内の間隔で続く同話者のturnを結合)。
    Noneなら結合せず、turnsテーブルの単位のままタグ付けする(従来の挙動)。

    結果はturnsテーブルを上書きせず、addin_resultsにmethod_version付きで
    保存する。既存のget_agreement_for_retag/print_retag_agreement_reportで
    そのまま比較できる。
    """
    conn = get_connection()
    _ensure_addin_results_table(conn)

    turns = conn.execute(
        "SELECT speaker, start, end, text FROM turns WHERE session_id=? ORDER BY start",
        (session_id,)
    ).fetchall()
    if not turns:
        conn.close()
        print(f"session_id={session_id} の発話が見つかりません。")
        return None

    raw_turns = [{"speaker": t[0], "start": t[1], "end": t[2], "text": t[3]} for t in turns]

    if merge_gap_sec is not None:
        groups = _merge_adjacent_turns(raw_turns, merge_gap_sec)
        print(f"  {len(raw_turns)}件のturnを、話者継続・間隔{merge_gap_sec}秒以内で{len(groups)}件にまとめました")
    else:
        groups = [{**t, "member_indices": [idx]} for idx, t in enumerate(raw_turns)]

    tagged_groups = [{"speaker": g["speaker"], "start": g["start"], "end": g["end"], "text": g["text"]} for g in groups]
    client = openai.OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

    examples_block = None
    if prompt_variant == "cot_indomain":
        phase_examples, intent_examples = _get_indomain_examples(session_id)
        examples_block = _build_indomain_defs(phase_examples, intent_examples)
        print(f"  in-domain例文: phase {len(phase_examples)}/{len(PHASE_ORDER)}カテゴリ分, "
              f"intent {len(intent_examples)}/{len(INTENT_LABELS)}カテゴリ分をこのセッションから抽出")

    for i in range(0, len(tagged_groups), block_size):
        block = tagged_groups[i: i + block_size]
        tag_prompt = _build_tag_prompt(prompt_variant, i, block, len(tagged_groups), examples_block=examples_block)
        tag_res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": tag_prompt}],
            response_format={"type": "json_object"}
        )
        tags = json.loads(tag_res.choices[0].message.content).get('analysis', [])
        for j, tag in enumerate(tags):
            if i + j < len(tagged_groups):
                tagged_groups[i + j]["phase"] = _normalize_phase(tag.get('phase', 'Information Sharing'))
                tagged_groups[i + j]["intent"] = _normalize_intent(tag.get('intent', 'Explanation'))
        print(f"  タグ付け中... {min(i + block_size, len(tagged_groups))}/{len(tagged_groups)}")

    # グループ単位のラベルを、元のturns件数・順序のまま展開する
    tagged = [dict(t) for t in raw_turns]
    for group, tagged_group in zip(groups, tagged_groups):
        for idx in group["member_indices"]:
            tagged[idx]["phase"] = tagged_group.get("phase", "Information Sharing")
            tagged[idx]["intent"] = tagged_group.get("intent", "Explanation")

    conn.execute(
        "INSERT INTO addin_results (session_id, addin_name, method_version, created, result) VALUES (?,?,?,?,?)",
        (
            session_id, "phase_intent_tagging", method_version,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            json.dumps({"turns": tagged})
        )
    )
    conn.commit()
    conn.close()
    print(f"✅ session_id={session_id} の再タグ付け結果を method_version={method_version} として保存しました")
    return {"status": "saved", "session_id": session_id, "method_version": method_version}


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


def _print_agreement(data):
    if not data["pairwise_agreement"]:
        print("比較できるデータがありません。人手ラベルを登録してから実行してください。")
        return data

    for m in data["pairwise_agreement"]:
        print(
            f"[{m['label_type']}] {m['rater_1']} vs {m['rater_2']} "
            f"(n={m['n']}): 一致率={m['percent_agreement']:.1%}, "
            f"Cohen's kappa={m['cohens_kappa']} ({_kappa_label(m['cohens_kappa'])})"
        )
    return data


def print_agreement_report(session_id: int):
    data = get_agreement(session_id)
    print(f"=== セッション{session_id} フェーズ・意図タグ付け 一致率レポート ===\n")
    return _print_agreement(data)


def print_retag_agreement_report(session_id: int, method_version: str, label: str = None):
    """
    print_agreement_report()のretag版。/corpus/{id}/retagの結果
    （addin_resultsに保存された指定method_version）と人手ラベルの一致率を表示する。
    プロンプト改善(chain-of-thought等)の前後で、同じ人手ラベルに対する
    一致率がどう変わったかを比較するために使う。
    """
    data = get_agreement_for_retag(session_id, method_version, label)
    if data is None:
        return None
    print(f"=== セッション{session_id} フェーズ・意図タグ付け 一致率レポート（{label or method_version}） ===\n")
    return _print_agreement(data)


def print_confusion_breakdown(session_id: int, method_version: str, label_type: str,
                               annotator_id: str = None, top_n: int = 10, example_chars: int = 80):
    """
    GPT-4o(retag結果)と人手ラベルの不一致を、(人手ラベル, GPTラベル)の組み合わせ別に
    集計して表示する。κが低い理由を「とりあえずプロンプトを変える」で探るのではなく、
    どのカテゴリがどのカテゴリに誤判定されやすいかを具体的に見てから改善の的を絞るために使う。

    label_type: "phase" または "intent"。
    annotator_idを省略すると、human_annotationsの全アノテーターを対象にする。
    """
    if label_type not in ("phase", "intent"):
        print('label_typeは"phase"か"intent"を指定してください。')
        return None

    retagged = _load_retag_labels(session_id, method_version)
    if retagged is None:
        return None

    conn = get_connection()
    query = """SELECT ha.annotator_id, ha.turn_id, ha.phase, ha.intent, t.text
               FROM human_annotations ha
               JOIN turns t ON t.id = ha.turn_id
               WHERE t.session_id = ?"""
    params = [session_id]
    if annotator_id:
        query += " AND ha.annotator_id = ?"
        params.append(annotator_id)
    human_rows = conn.execute(query, params).fetchall()
    conn.close()

    label_index = 0 if label_type == "phase" else 1
    mismatch_counts = {}
    mismatch_examples = {}
    n_total, n_match = 0, 0
    for _, turn_id, phase, intent, text in human_rows:
        human_label = phase if label_type == "phase" else intent
        if turn_id not in retagged or human_label is None:
            continue
        gpt_label = retagged[turn_id][label_index]
        if gpt_label is None:
            continue
        n_total += 1
        if gpt_label == human_label:
            n_match += 1
            continue
        key = (human_label, gpt_label)
        mismatch_counts[key] = mismatch_counts.get(key, 0) + 1
        mismatch_examples.setdefault(key, []).append(text)

    print(f"=== {label_type} 不一致パターン (method_version={method_version}, "
          f"n={n_total}, 一致={n_match}件, 不一致={n_total - n_match}件) ===\n")
    ranked = sorted(mismatch_counts.items(), key=lambda kv: -kv[1])[:top_n]
    for (human_label, gpt_label), count in ranked:
        print(f"人間「{human_label}」→ GPT「{gpt_label}」: {count}件")
        for example_text in mismatch_examples[(human_label, gpt_label)][:2]:
            shown = example_text[:example_chars]
            print(f"    例: {shown}")
        print()

    return {
        "session_id": session_id, "method_version": method_version, "label_type": label_type,
        "n": n_total, "n_match": n_match,
        "mismatches": [
            {"human_label": h, "gpt_label": g, "count": c}
            for (h, g), c in ranked
        ]
    }


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
# 5. run_standalone_retag()で、サーバー/ngrokを使わずこのセルから直接再タグ付けする
# retag_res = run_standalone_retag(session_id=_latest_session_id, method_version="consistency_check_1", prompt_variant="baseline")
# print(retag_res)

# 6. 1回目のタグ付け結果とretag結果の一致率・Cohen's kappaを表示する
#    （人手ラベルは一切不要。ただし「正解と合っているか」ではなく
#    「GPT-4oが毎回同じ判断をするか」を測るものである点に注意）
# get_retag_consistency(session_id=_latest_session_id, method_version="consistency_check_1")

# --- プロンプト改善(few-shot/CoT等)の効果をアブレーションで測る ---
#
# 7. prompt_variantで"fewshot"/"cot"/"baseline"を切り替えて再タグ付けする
#    (サーバーの起動・再起動は一切不要)
# run_standalone_retag(session_id=_latest_session_id, method_version="fewshot_v1", prompt_variant="fewshot")
# run_standalone_retag(session_id=_latest_session_id, method_version="cot_v1", prompt_variant="cot")

# 8. ベースライン(turnsテーブルの元のタグ付け、n=68で測定済み)と、
#    各手法のretag結果について、同じ人手ラベルとの一致率を比較する
# print_agreement_report(session_id=_latest_session_id)  # ベースライン
# print_retag_agreement_report(session_id=_latest_session_id, method_version="fewshot_v1", label="Few-shot")
# print_retag_agreement_report(session_id=_latest_session_id, method_version="cot_v1", label="CoT")

# --- κが低い理由を具体的に見る: 不一致パターンの分析 ---
#
# 9. どのカテゴリがどのカテゴリに誤判定されやすいかを、具体的な発話例付きで表示する
# print_confusion_breakdown(session_id=_latest_session_id, method_version="cot_block10", label_type="intent")
# print_confusion_breakdown(session_id=_latest_session_id, method_version="cot_v1", label_type="phase")

# --- 発話の断片化を解消する: 同話者の連続turnを結合してからタグ付け ---
#
# 10. merge_gap_secで、同じ話者が短い間隔(秒)以内で続けて話しているturnを
#     結合してからタグ付けする。数語の断片（例:「to 10 people.」）が
#     もっと長い・判断しやすい単位になる。
# run_standalone_retag(session_id=_latest_session_id, method_version="cot_merged_block10",
#                       prompt_variant="cot", block_size=10, merge_gap_sec=1.5)
# print_retag_agreement_report(session_id=_latest_session_id, method_version="cot_merged_block10", label="CoT+結合")

# --- ドメインが一致したfew-shot例文: このセッション自身の人手ラベルから抽出 ---
#
# 11. 汎用的な例文(業務会議風)がこのセッション(雑談)と噛み合わず精度を下げていた
#     可能性があるため、このセッション自身の人手ラベル済み発話を例文として使う
# run_standalone_retag(session_id=_latest_session_id, method_version="cot_indomain_block10",
#                       prompt_variant="cot_indomain", block_size=10)
# print_retag_agreement_report(session_id=_latest_session_id, method_version="cot_indomain_block10", label="CoT+in-domain例文")

