# --- 別セル: 文字起こし精度（WER/CER）測定ツール ---
# 前提: audio_analysis_pipeline.py のサーバーが同じColab上で起動していること。
# サーバー起動セルの出力に表示された public_url を、下のBASE_URLに貼り付けてから実行する。
# ロジック（正解テキストの保存・WER/CER計算）はサーバー側の
# POST /corpus/{id}/reference_transcripts と GET /corpus/{id}/transcription_accuracy
# をそのまま呼び出しているだけなので、計算処理を二重実装していない。

import difflib
import requests
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

BASE_URL = "https://xxxx-xx-xx-xx-xx.ngrok-free.app"  # ← サーバー起動時に表示されたURLに置き換える
SAVE_DIR = "/content/drive/MyDrive"  # ← チャート・レポートの保存先（corpus.dbと同じDrive）


def get_raw_turns(session_id: int):
    """正解テキストを付けたいセッションの発話一覧（turn_id/speaker/text）を確認する。"""
    res = requests.get(f"{BASE_URL}/corpus/{session_id}/raw_turns")
    res.raise_for_status()
    return res.json()


def submit_reference_transcripts(session_id: int, transcriber_id: str, transcripts: list):
    """
    人手で書き起こした正解テキストをサーバーに保存する。
    transcripts: [{"turn_id": 12, "reference_text": "実際に話された正しいテキスト"}, ...]
    """
    res = requests.post(
        f"{BASE_URL}/corpus/{session_id}/reference_transcripts",
        json={"transcriber_id": transcriber_id, "transcripts": transcripts}
    )
    res.raise_for_status()
    return res.json()


def get_transcription_accuracy(session_id: int):
    """保存済みの正解テキストとWhisper出力を突き合わせたWER/CERの集計結果を取得する。"""
    res = requests.get(f"{BASE_URL}/corpus/{session_id}/transcription_accuracy")
    res.raise_for_status()
    return res.json()


def print_accuracy_report(session_id: int):
    data = get_transcription_accuracy(session_id)

    overall = data["overall"]
    print(f"=== セッション{session_id} 文字起こし精度レポート ===")
    print(f"全体: 発話数={overall['n_turns']}, 平均WER={overall['mean_wer']}, 平均CER={overall['mean_cer']}")

    print("\n話者別:")
    for row in data["by_speaker"]:
        print(f"  {row['speaker']}: n={row['n_turns']}, 平均WER={row['mean_wer']}, 平均CER={row['mean_cer']}")

    masked = [t for t in data["per_turn"] if t["contains_mask"]]
    if masked:
        print(f"\n[注意] {len(masked)}件の発話が匿名化（[MASK]置換）されており、"
              f"誤差率が本来のWhisperの認識精度より高く出ている可能性があります。")
        print("       正確な精度検証をしたい場合は、これらのturn_idを除外して再集計してください。")

    return data


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
    significant = [t for t in turns_sorted if t["is_significant"]]
    print(f"\n=== 文字一致率が{1 - cer_threshold:.0%}未満の発話: {len(significant)}件 / 正解あり{len(turns_sorted)}件中 ===\n")
    for t in significant:
        mask_note = "　※匿名化([MASK])による差分の可能性あり" if t["contains_mask"] else ""
        print(f"[turn_id={t['turn_id']}] 話者={t['speaker']}  CER={t['cer']}{mask_note}")
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
