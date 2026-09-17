# --- 別セル: 文字起こし精度（WER/CER）測定ツール ---
# 前提: audio_analysis_pipeline.py のサーバーが同じColab上で起動していること。
# サーバー起動セルの出力に表示された public_url を、下のBASE_URLに貼り付けてから実行する。
# ロジック（正解テキストの保存・WER/CER計算）はサーバー側の
# POST /corpus/{id}/reference_transcripts と GET /corpus/{id}/transcription_accuracy
# をそのまま呼び出しているだけなので、計算処理を二重実装していない。

import difflib
import requests
import matplotlib.pyplot as plt
import japanize_matplotlib

BASE_URL = "https://xxxx-xx-xx-xx-xx.ngrok-free.app"  # ← サーバー起動時に表示されたURLに置き換える

# 意図ラベリング（Proposal/Question/Agreement/Disagreement/Confirmation/Acknowledge/Explanation）
# の判定を左右しそうなキーワード。ここに単語が含まれる差分は「意味が変わりうる」とみなす。
# あくまでヒューリスティック（機械的なキーワード一致）であり、意味的に完全な判定ではない点に注意。
INTENT_CRITICAL_WORDS = {
    "否定(negation)": [
        "not", "n't", "never", "no", "don't", "isn't", "wasn't", "can't", "won't", "doesn't",
        "ない", "ません", "拒否", "無理", "だめ"
    ],
    "同意(agreement)": [
        "agree", "ok", "okay", "sure", "yes", "fine", "sounds good",
        "同意", "賛成", "いいですね", "そうですね", "オッケー", "はい"
    ],
    "反対(disagreement)": [
        "disagree", "but", "however", "actually", "not sure",
        "反対", "違う", "でも", "しかし", "微妙"
    ],
    "疑問(question)": [
        "what", "why", "how", "could you", "would you", "?",
        "か？", "ですか", "でしょうか", "なぜ", "どう", "？"
    ],
}


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


def find_intent_critical_categories(reference, whisper):
    """
    正解テキストとWhisperテキストそれぞれについて、意図ラベリングに関わる
    キーワードカテゴリの出現有無を調べ、両者で結果が異なるカテゴリ
    （＝誤りによってキーワードが消えた/現れた）を返す。
    diffの文字断片同士を突き合わせるのではなく全文で判定することで、
    「違う」のようなキーワードが1文字だけの差分に分断されて検出漏れするのを防ぐ。
    """
    ref_lower = reference.lower()
    hyp_lower = whisper.lower()
    changed_categories = set()
    for category, words in INTENT_CRITICAL_WORDS.items():
        ref_hit = any(w.lower() in ref_lower for w in words)
        hyp_hit = any(w.lower() in hyp_lower for w in words)
        if ref_hit != hyp_hit:
            changed_categories.add(category)
    return changed_categories


def analyze_and_visualize(session_id: int, cer_threshold: float = 0.15):
    """
    正解テキスト(reference_text) と Whisper出力(turns.text) の一致率を発話ごとに可視化し、
    「大きな違い」（CERが閾値を超える、または意図ラベリングに影響しそうなキーワードが
    変化した）発話だけを一覧表示する。
    """
    data = get_transcription_accuracy(session_id)
    per_turn = data["per_turn"]
    if not per_turn:
        print("正解テキストが1件も登録されていません。先にsubmit_reference_transcriptsで登録してください。")
        return data

    for t in per_turn:
        t["diffs"] = text_diff(t["reference_text"], t["whisper_text"])
        t["hit_categories"] = find_intent_critical_categories(t["reference_text"], t["whisper_text"])
        # 「大きな違い」= 誤り率が閾値を超える、または意図に関わるキーワードが変化した場合
        t["is_significant"] = bool(t["diffs"]) and (
            (t["cer"] is not None and t["cer"] >= cer_threshold) or len(t["hit_categories"]) > 0
        )

    # --- 可視化: 発話ごとの一致率（1-CER）。赤=意図に影響しうる重大な差分、
    #     オレンジ=軽微な差分、緑=完全一致 で色分けする ---
    turns_sorted = sorted(per_turn, key=lambda t: t["turn_id"])
    x = list(range(len(turns_sorted)))
    accuracy = [1 - t["cer"] if t["cer"] is not None else 0 for t in turns_sorted]

    COLOR_SIGNIFICANT = "#d62728"
    COLOR_MINOR = "#ff9f40"
    COLOR_EXACT = "#2ca02c"
    colors = []
    for t in turns_sorted:
        if t["is_significant"]:
            colors.append(COLOR_SIGNIFICANT)
        elif t["cer"] and t["cer"] > 0:
            colors.append(COLOR_MINOR)
        else:
            colors.append(COLOR_EXACT)

    fig, ax = plt.subplots(figsize=(max(8, len(x) * 0.35), 5))
    ax.bar(x, accuracy, color=colors)
    ax.set_xlabel("発話順（turn）")
    ax.set_ylabel("一致率 (1 - CER)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"セッション{session_id}: 正解テキストとWhisper出力の一致率")
    ax.axhline(1 - cer_threshold, color="gray", linestyle="--", linewidth=1)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_EXACT, label="完全一致"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MINOR, label="軽微な差分"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_SIGNIFICANT, label="意図に影響しうる重大な差分"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    plt.tight_layout()
    plt.show()

    # --- 重大な差分の一覧を出力 ---
    significant = [t for t in turns_sorted if t["is_significant"]]
    print(f"\n=== 意図ラベリングに影響しうる重大な差分: {len(significant)}件 / 正解あり{len(turns_sorted)}件中 ===\n")
    for t in significant:
        mask_note = "　※匿名化([MASK])による差分の可能性あり" if t["contains_mask"] else ""
        reason = sorted(t["hit_categories"]) if t["hit_categories"] else [f"CER閾値({cer_threshold})超過"]
        print(f"[turn_id={t['turn_id']}] 話者={t['speaker']}  CER={t['cer']}  理由={reason}{mask_note}")
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

# 4. 一致率を可視化し、意図ラベリングに影響しうる大きな差分だけを抽出する
# analyze_and_visualize(session_id=1, cer_threshold=0.15)
