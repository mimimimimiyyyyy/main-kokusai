# --- 別セル: 文字起こし精度（WER/CER）測定ツール ---
# 前提: audio_analysis_pipeline.py のサーバーが同じColab上で起動していること。
# サーバー起動セルの出力に表示された public_url を、下のBASE_URLに貼り付けてから実行する。
# ロジック（正解テキストの保存・WER/CER計算）はサーバー側の
# POST /corpus/{id}/reference_transcripts と GET /corpus/{id}/transcription_accuracy
# をそのまま呼び出しているだけなので、計算処理を二重実装していない。

import requests

BASE_URL = "https://xxxx-xx-xx-xx-xx.ngrok-free.app"  # ← サーバー起動時に表示されたURLに置き換える


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
