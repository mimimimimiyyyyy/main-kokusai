# --- 別セル: 文字起こし精度（WER/CER）測定ツール（DB直接アクセス版） ---
# audio_analysis_pipeline.py のサーバーを起動していなくても、Google Driveがマウントされて
# いればこのセル単体で動く。ngrokのURLは不要（corpus.dbに直接読み書きする）。
# WER/CER計算ロジックはサーバー側（audio_analysis_pipeline.py）と同じものをここに
# 複製している。ロジックを変更した場合は両方に反映すること。

import difflib
import re
import sqlite3
from datetime import datetime

import numpy as np
import openai
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from google.colab import userdata

DB_PATH = "/content/drive/MyDrive/corpus.db"
SAVE_DIR = "/content/drive/MyDrive"

# 意味的類似度（コサイン類似度）の算出用。CERは文字列としての一致度しか見ないため、
# 「文字はだいぶ違うが意味はほぼ同じ」なケース（言い換え・フィラー違い等）を見分けるために使う。
client = openai.OpenAI(api_key=userdata.get('OPENAI_API_KEY'))


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    # サーバーを一度も起動していない状態でこのセルだけ使う場合に備えてテーブルを保証する
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reference_transcripts (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id        INTEGER,
            transcriber_id TEXT,
            reference_text TEXT,
            created        TEXT,
            UNIQUE(turn_id, transcriber_id),
            FOREIGN KEY (turn_id) REFERENCES turns(id)
        )
    """)
    conn.commit()
    return conn


def edit_distance(ref_tokens, hyp_tokens):
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def word_error_rate(reference, hypothesis):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens:
        return None
    return edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def char_error_rate(reference, hypothesis):
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    if not ref_chars:
        return None
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def semantic_similarity(reference, hypothesis):
    """
    正解テキストとWhisperテキストの意味的な近さをコサイン類似度(0〜1、高いほど近い)で返す。
    CERが高く出ていても、これが高ければ「文字は違うが言っている内容はほぼ同じ」と判断できる。
    text-embedding-3-smallは出力ベクトルが正規化済みなので単純な内積で類似度になる
    （run_full_analysis/searchの埋め込み計算と同じ考え方）。
    """
    if not reference.strip() or not hypothesis.strip():
        return None
    res = client.embeddings.create(input=[reference, hypothesis], model="text-embedding-3-small")
    v1 = np.array(res.data[0].embedding)
    v2 = np.array(res.data[1].embedding)
    return float(np.dot(v1, v2))


def text_diff(reference, hypothesis):
    """
    正解テキストとWhisperテキストを文字単位で比較し、一致していない箇所
    （置換・削除・追加）のリストを返す。分かち書きしない日本語にも対応するため
    単語区切りではなく文字単位で差分を取る。
    """
    matcher = difflib.SequenceMatcher(None, reference, hypothesis)
    diffs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        diffs.append({
            "type": tag,  # replace / delete / insert
            "reference": reference[i1:i2],
            "whisper": hypothesis[j1:j2],
        })
    return diffs


_TIME_RANGE_RE = re.compile(
    r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})'
)


def _to_seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_reference_file(path):
    """
    "00:32:54.000 --> 00:33:00.000" のようなタイムスタンプ区間の後にテキストが続く
    ファイル（zenhann_hyouka.txtのような形式）を読み込み、
    [{"start": 秒, "end": 秒, "text": ...}, ...] を時系列で返す。
    区間番号の行があってもなくても、タイムスタンプ行を探して処理するので影響しない。
    """
    content = open(path, encoding="utf-8").read()
    blocks = re.split(r"\n\s*\n", content.strip())
    segments = []
    for block in blocks:
        lines = [l for l in block.strip().split("\n") if l.strip() != ""]
        ts_idx = None
        for i, line in enumerate(lines):
            if _TIME_RANGE_RE.search(line):
                ts_idx = i
                break
        if ts_idx is None:
            continue
        m = _TIME_RANGE_RE.search(lines[ts_idx])
        start = _to_seconds(*m.groups()[0:4])
        end = _to_seconds(*m.groups()[4:8])
        text = " ".join(lines[ts_idx + 1:]).strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def evaluate_against_reference_file(session_id: int, reference_file_path: str,
                                     cer_threshold: float = 0.3, save_dir: str = SAVE_DIR):
    """
    turn単位の正解ではなく、タイムスタンプ区間ごとのテキストファイル
    （zenhann_hyouka.txtのような形式）を正解として評価する。
    1つの区間が複数のturnにまたがることが多いため、submit_reference_transcripts
    （turn単位）は使わず、区間の時間窓に重なるturnのテキストを連結して比較する。

    正解ファイルの時刻は、turnsの開始時刻(0秒付近)とはズレている前提
    （例: 元の長い録画の32:54〜42:58を切り出したクリップがzenhan.mp4になっている場合、
    正解ファイル側は32:54始まりのままになる）。正解ファイルの最初のセグメント開始時刻と
    turnsの最初の開始時刻の差を自動的にオフセットとして推定する。
    """
    segments = parse_reference_file(reference_file_path)
    if not segments:
        print("正解ファイルからセグメントを読み取れませんでした。")
        return None

    conn = get_connection()
    turns = conn.execute(
        "SELECT id, speaker, start, end, text FROM turns WHERE session_id=? ORDER BY start",
        (session_id,)
    ).fetchall()
    conn.close()
    if not turns:
        print(f"session_id={session_id} のturnsが見つかりません。")
        return None

    offset = segments[0]["start"] - turns[0][2]
    print(f"推定オフセット: {offset:.1f}秒（正解ファイルの先頭とturnsの先頭を揃えています）")

    rows = []
    for seg in segments:
        win_start = seg["start"] - offset
        win_end = seg["end"] - offset
        overlapped = [t for t in turns if t[2] < win_end and t[3] > win_start]
        whisper_text = " ".join(t[4] for t in overlapped)
        cer = char_error_rate(seg["text"], whisper_text)
        sim = semantic_similarity(seg["text"], whisper_text)
        rows.append({
            "ref_start": seg["start"],
            "ref_end": seg["end"],
            "reference_text": seg["text"],
            "whisper_text": whisper_text,
            "n_matched_turns": len(overlapped),
            "contains_mask": "[MASK]" in whisper_text,
            "cer": round(cer, 3) if cer is not None else None,
            "semantic_similarity": round(sim, 3) if sim is not None else None,
            "diffs": text_diff(seg["text"], whisper_text)
        })

    # 「文字起こしの精度が低い」のか「そもそも対応turnが無い(検出漏れ)」のかは
    # 別の問題なので分けて扱う。検出漏れ(n_matched_turns=0)はCER=1.0で平均に
    # 混ぜず、別枠として件数だけ報告する。
    undetected = [r for r in rows if r["n_matched_turns"] == 0]
    detected = [r for r in rows if r["n_matched_turns"] > 0]
    for r in detected:
        r["is_significant"] = r["cer"] is not None and r["cer"] >= cer_threshold

    SIM_HIGH_THRESHOLD = 0.85  # これ以上なら「文字は違うが意味はほぼ同じ」とみなす目安

    # --- 可視化: 検出できたセグメントのみ対象。緑=一致、赤=低一致 で色分けする ---
    x = list(range(len(detected)))
    accuracy = [max(0, 1 - r["cer"]) for r in detected]
    COLOR_LOW, COLOR_OK = "#d62728", "#2ca02c"
    colors = [COLOR_LOW if r["is_significant"] else COLOR_OK for r in detected]

    fig, ax = plt.subplots(figsize=(max(10, len(x) * 0.3), 5))
    ax.bar(x, accuracy, color=colors)
    ax.set_xlabel("正解セグメント順（検出漏れを除く）")
    ax.set_ylabel("文字一致率 (1 - CER)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"セッション{session_id}: 正解ファイルとDB文字起こしの文字一致率")
    ax.axhline(1 - cer_threshold, color="gray", linestyle="--", linewidth=1)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_OK, label=f"一致率 ≥ {1 - cer_threshold:.0%}"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_LOW, label=f"一致率 < {1 - cer_threshold:.0%}"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()

    chart_path = f"{save_dir}/session_{session_id}_reference_file_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"チャートを保存しました: {chart_path}")

    # --- CSV保存 ---
    report_path = f"{save_dir}/session_{session_id}_reference_file_report.csv"
    report_df = pd.DataFrame([
        {
            "ref_start": r["ref_start"], "ref_end": r["ref_end"],
            "cer": r["cer"], "semantic_similarity": r["semantic_similarity"],
            "n_matched_turns": r["n_matched_turns"], "undetected": r["n_matched_turns"] == 0,
            "below_threshold": r.get("is_significant"), "contains_mask": r["contains_mask"],
            "reference_text": r["reference_text"], "whisper_text": r["whisper_text"],
            "diff": " / ".join(f"{d['type']}:「{d['reference']}」→「{d['whisper']}」" for d in r["diffs"])
        }
        for r in rows
    ])
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"レポートを保存しました: {report_path}")

    overall_cer = sum(r["cer"] for r in detected) / len(detected) if detected else None
    overall_sim = (
        sum(r["semantic_similarity"] for r in detected if r["semantic_similarity"] is not None)
        / len([r for r in detected if r["semantic_similarity"] is not None])
        if detected else None
    )
    print(f"\n検出漏れ（対応turnが無い区間）: {len(undetected)}件 / {len(rows)}件中"
          "　※これらはCER計算から除外しています")
    print(f"検出できた{len(detected)}件の平均CER: {overall_cer:.3f}" if overall_cer is not None else "")
    print(f"検出できた{len(detected)}件の平均意味的類似度: {overall_sim:.3f}" if overall_sim is not None else "")

    if undetected:
        print(f"\n=== 検出漏れ区間: {len(undetected)}件 ===\n")
        for r in undetected:
            print(f"[{r['ref_start']}s〜{r['ref_end']}s] 正解: {r['reference_text']}")
        print()

    # --- 一致率が低いセグメントの一覧（検出できたもののみ対象） ---
    significant = [r for r in detected if r["is_significant"]]
    print(f"\n=== 文字一致率が{1 - cer_threshold:.0%}未満のセグメント: {len(significant)}件 / {len(detected)}件中（検出できた分） ===\n")
    for r in sorted(significant, key=lambda r: r["cer"], reverse=True):
        mask_note = "　※匿名化([MASK])による差分の可能性あり" if r["contains_mask"] else ""
        sim = r["semantic_similarity"]
        sim_note = "　※意味的には近い可能性あり（言い換え等）" if sim is not None and sim >= SIM_HIGH_THRESHOLD else ""
        print(f"[{r['ref_start']}s] CER={r['cer']}  意味的類似度={sim}  対応turn数={r['n_matched_turns']}{mask_note}{sim_note}")
        print(f"  正解    : {r['reference_text']}")
        print(f"  DB      : {r['whisper_text']}")
        print()

    return rows


def get_raw_turns(session_id: int):
    """正解テキストを付けたいセッションの発話一覧（turn_id/speaker/text）を確認する。"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, speaker, start, end, text FROM turns WHERE session_id=? ORDER BY start",
        (session_id,)
    ).fetchall()
    conn.close()
    return {
        "session_id": session_id,
        "turns": [
            {"turn_id": r[0], "speaker": r[1], "start": r[2], "end": r[3], "text": r[4]}
            for r in rows
        ]
    }


def submit_reference_transcripts(session_id: int, transcriber_id: str, transcripts: list):
    """
    人手で書き起こした正解テキストをcorpus.dbに保存する。
    transcripts: [{"turn_id": 12, "reference_text": "実際に話された正しいテキスト"}, ...]
    同じturn×同じtranscriber_idで再送すると上書きされる。
    """
    conn = get_connection()
    valid_turn_ids = {
        row[0] for row in conn.execute(
            "SELECT id FROM turns WHERE session_id=?", (session_id,)
        ).fetchall()
    }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    saved = 0
    skipped_turn_ids = []
    for item in transcripts:
        turn_id = item["turn_id"]
        if turn_id not in valid_turn_ids:
            skipped_turn_ids.append(turn_id)
            continue
        conn.execute(
            """INSERT INTO reference_transcripts (turn_id, transcriber_id, reference_text, created)
               VALUES (?,?,?,?)
               ON CONFLICT(turn_id, transcriber_id) DO UPDATE SET
                   reference_text=excluded.reference_text, created=excluded.created""",
            (turn_id, transcriber_id, item["reference_text"], now)
        )
        saved += 1
    conn.commit()
    conn.close()
    return {
        "status": "saved", "session_id": session_id, "transcriber_id": transcriber_id,
        "saved": saved, "skipped_turn_ids": skipped_turn_ids
    }


def get_transcription_accuracy(session_id: int):
    """保存済みの正解テキストとWhisper出力を突き合わせたWER/CERの集計結果を返す。"""
    conn = get_connection()
    rows = conn.execute(
        """SELECT rt.turn_id, rt.transcriber_id, rt.reference_text, t.speaker, t.text
           FROM reference_transcripts rt
           JOIN turns t ON t.id = rt.turn_id
           WHERE t.session_id = ?
           ORDER BY t.start""",
        (session_id,)
    ).fetchall()
    conn.close()

    per_turn = []
    by_speaker = {}
    total_wer, total_cer, total_sim, total_n = 0.0, 0.0, 0.0, 0
    for turn_id, transcriber_id, reference_text, speaker, whisper_text in rows:
        wer = word_error_rate(reference_text, whisper_text)
        cer = char_error_rate(reference_text, whisper_text)
        sim = semantic_similarity(reference_text, whisper_text)
        per_turn.append({
            "turn_id": turn_id,
            "speaker": speaker,
            "transcriber_id": transcriber_id,
            "contains_mask": "[MASK]" in whisper_text,
            "reference_text": reference_text,
            "whisper_text": whisper_text,
            "wer": round(wer, 3) if wer is not None else None,
            "cer": round(cer, 3) if cer is not None else None,
            "semantic_similarity": round(sim, 3) if sim is not None else None
        })
        if wer is not None and cer is not None and sim is not None:
            bucket = by_speaker.setdefault(speaker, {"n": 0, "wer_sum": 0.0, "cer_sum": 0.0, "sim_sum": 0.0})
            bucket["n"] += 1
            bucket["wer_sum"] += wer
            bucket["cer_sum"] += cer
            bucket["sim_sum"] += sim
            total_n += 1
            total_wer += wer
            total_cer += cer
            total_sim += sim

    return {
        "session_id": session_id,
        "per_turn": per_turn,
        "by_speaker": [
            {
                "speaker": speaker,
                "n_turns": b["n"],
                "mean_wer": round(b["wer_sum"] / b["n"], 3),
                "mean_cer": round(b["cer_sum"] / b["n"], 3),
                "mean_semantic_similarity": round(b["sim_sum"] / b["n"], 3)
            }
            for speaker, b in by_speaker.items()
        ],
        "overall": {
            "n_turns": total_n,
            "mean_wer": round(total_wer / total_n, 3) if total_n else None,
            "mean_cer": round(total_cer / total_n, 3) if total_n else None,
            "mean_semantic_similarity": round(total_sim / total_n, 3) if total_n else None
        }
    }


def print_accuracy_report(session_id: int):
    data = get_transcription_accuracy(session_id)

    overall = data["overall"]
    print(f"=== セッション{session_id} 文字起こし精度レポート ===")
    print(f"全体: 発話数={overall['n_turns']}, 平均WER={overall['mean_wer']}, 平均CER={overall['mean_cer']}, "
          f"平均意味的類似度={overall['mean_semantic_similarity']}")

    print("\n話者別:")
    for row in data["by_speaker"]:
        print(f"  {row['speaker']}: n={row['n_turns']}, 平均WER={row['mean_wer']}, 平均CER={row['mean_cer']}, "
              f"平均意味的類似度={row['mean_semantic_similarity']}")

    masked = [t for t in data["per_turn"] if t["contains_mask"]]
    if masked:
        print(f"\n[注意] {len(masked)}件の発話が匿名化（[MASK]置換）されており、"
              f"誤差率が本来のWhisperの認識精度より高く出ている可能性があります。")
        print("       正確な精度検証をしたい場合は、これらのturn_idを除外して再集計してください。")

    return data


def analyze_and_visualize(session_id: int, cer_threshold: float = 0.15, save_dir: str = SAVE_DIR):
    """
    正解テキスト(reference_text) と Whisper出力(turns.text) の文字一致率(1-CER)を
    発話ごとに可視化し、CERが閾値を超える（＝文字の一致率が低い）発話だけを
    差分付きで一覧表示する。
    グラフ（PNG）と発話ごとの結果（CSV）はsave_dir配下にファイルとして保存する。
    """
    data = get_transcription_accuracy(session_id)
    per_turn = data["per_turn"]
    if not per_turn:
        print("正解テキストが1件も登録されていません。先にsubmit_reference_transcriptsで登録してください。")
        return data

    for t in per_turn:
        t["diffs"] = text_diff(t["reference_text"], t["whisper_text"])
        t["is_significant"] = t["cer"] is not None and t["cer"] >= cer_threshold

    # --- 可視化: 発話ごとの文字一致率（1-CER）。赤=閾値超過、緑=閾値内 で色分けする ---
    turns_sorted = sorted(per_turn, key=lambda t: t["turn_id"])
    x = list(range(len(turns_sorted)))
    accuracy = [1 - t["cer"] if t["cer"] is not None else 0 for t in turns_sorted]

    COLOR_LOW = "#d62728"
    COLOR_OK = "#2ca02c"
    colors = [COLOR_LOW if t["is_significant"] else COLOR_OK for t in turns_sorted]

    fig, ax = plt.subplots(figsize=(max(8, len(x) * 0.35), 5))
    ax.bar(x, accuracy, color=colors)
    ax.set_xlabel("発話順（turn）")
    ax.set_ylabel("文字一致率 (1 - CER)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"セッション{session_id}: 正解テキストとWhisper出力の文字一致率")
    ax.axhline(1 - cer_threshold, color="gray", linestyle="--", linewidth=1)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_OK, label=f"一致率 ≥ {1 - cer_threshold:.0%}"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_LOW, label=f"一致率 < {1 - cer_threshold:.0%}"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    plt.tight_layout()

    chart_path = f"{save_dir}/session_{session_id}_accuracy_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"チャートを保存しました: {chart_path}")

    # --- 発話ごとの結果をCSVとして保存 ---
    report_path = f"{save_dir}/session_{session_id}_accuracy_report.csv"
    report_df = pd.DataFrame([
        {
            "turn_id": t["turn_id"],
            "speaker": t["speaker"],
            "cer": t["cer"],
            "wer": t["wer"],
            "semantic_similarity": t["semantic_similarity"],
            "below_threshold": t["is_significant"],
            "contains_mask": t["contains_mask"],
            "reference_text": t["reference_text"],
            "whisper_text": t["whisper_text"],
            "diff": " / ".join(f"{d['type']}:「{d['reference']}」→「{d['whisper']}」" for d in t["diffs"])
        }
        for t in turns_sorted
    ])
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"レポートを保存しました: {report_path}")

    # --- 一致率が低い発話の一覧を出力 ---
    SIM_HIGH_THRESHOLD = 0.85  # これ以上なら「文字は違うが意味はほぼ同じ」とみなす目安
    significant = [t for t in turns_sorted if t["is_significant"]]
    print(f"\n=== 文字一致率が{1 - cer_threshold:.0%}未満の発話: {len(significant)}件 / 正解あり{len(turns_sorted)}件中 ===\n")
    for t in significant:
        mask_note = "　※匿名化([MASK])による差分の可能性あり" if t["contains_mask"] else ""
        sim = t["semantic_similarity"]
        sim_note = ""
        if sim is not None and sim >= SIM_HIGH_THRESHOLD:
            sim_note = "　※意味的には近い可能性あり（言い換え等）"
        print(f"[turn_id={t['turn_id']}] 話者={t['speaker']}  CER={t['cer']}  意味的類似度={sim}{mask_note}{sim_note}")
        print(f"  正解    : {t['reference_text']}")
        print(f"  Whisper : {t['whisper_text']}")
        for d in t["diffs"]:
            print(f"    - {d['type']}: 「{d['reference']}」 → 「{d['whisper']}」")
        print()

    return per_turn


# --- 使い方の例（実行する場合はコメントを外す） ---

# 1. 対象セッションの発話一覧を確認し、どのturn_idに正解テキストを付けるか決める
# print(get_raw_turns(session_id=1))

# 2. 人手で書き起こした正解テキストを送信する（transcriber_idはアノテーターの識別名）
# submit_reference_transcripts(
#     session_id=1,
#     transcriber_id="researcher_A",
#     transcripts=[
#         {"turn_id": 12, "reference_text": "会議の予定について話しましょう"},
#         {"turn_id": 13, "reference_text": "はい、来週の火曜日はどうですか"},
#     ]
# )

# 3. 話者別のWER/CERレポートを表示する
# print_accuracy_report(session_id=1)

# 4. 文字一致率を可視化し、一致率が低い発話だけを差分付きで一覧表示する
# analyze_and_visualize(session_id=1, cer_threshold=0.15)

# 5. turn単位の正解を1件ずつ登録する代わりに、タイムスタンプ区間テキストファイル
#    （zenhann_hyouka.txtのような形式）をまとめて正解として評価する場合
# evaluate_against_reference_file(
#     session_id=3,
#     reference_file_path="/content/drive/MyDrive/zenhann_hyouka.txt",
#     cer_threshold=0.3
# )

# 6. 最新セッション（直近アップロード分）に対して自動的に評価する
_conn = sqlite3.connect(DB_PATH)
latest_session_id, latest_filename = _conn.execute(
    "SELECT id, filename FROM sessions ORDER BY id DESC LIMIT 1"
).fetchone()
_conn.close()
print(f"最新セッション: session_id={latest_session_id} ({latest_filename})")

evaluate_against_reference_file(
    session_id=latest_session_id,
    reference_file_path="/content/drive/MyDrive/zenhann_hyouka.txt",
    cer_threshold=0.3
)
