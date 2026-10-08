# 対話・コミュニケーション研究用 統合データベースプラットフォーム

会話の音声・映像と書き起こしをまとめて保存し、ラベルを付けて整理・検索できる研究用プラットフォーム。
Google Colab のセルとして動かし、ngrok で公開した API を画面（`kokusai.html`）から使う。

| ファイル | 内容 |
| --- | --- |
| `audio_analysis_pipeline.py` | 既存。対話の解析（話者分離・Whisper・GPT-4o によるタグ付け）と API（`/upload`、`/search`、`/corpus` など） |
| `transcription_accuracy_tool.py` | 既存。文字起こし精度（WER/CER）を測る別セル |
| `skill_transfer_cell.py` | 追加。建設業向け **技能伝承動画共有機能** の別セル |
| `skill_transfer.html` | 追加。技能伝承の画面（スマホ優先） |
| `seed/skill_transfer_tags.csv` | 追加。技能伝承のタグ一覧の初期データ |
| `docs/skill-transfer-mapping.md` | 追加。既存の仕組みとの対応づけと実装計画・結果 |
| `tests/` | 追加。pytest のテスト |

既存の 2 つのファイルは編集していない。技能伝承のセルを実行しなければ、これまでと同じように動く。

---

## 技能伝承動画共有機能

熟練者が作業しながら説明する動画をスマホで撮影・投稿すると、文字起こし・字幕・手順書（写真付き PDF）・タグを自動で作り、
若手が作業の種類やタグで動画を探して、見たい場面から再生できるようにする。

- データは既存と同じ `corpus.db` に、`skill_` で始まる表として別に保存する（対話研究の `sessions`・`turns` などには入れない）
- 既存セルが読み込んだ Whisper モデル・OpenAI クライアントをそのまま共用する（モデルを二重に読み込まない）

### セットアップ（Google Colab）

1. Google Drive に次の 2 つのファイルを置く（フォルダは無ければ作る）
   - `skill_transfer.html` → `/content/drive/MyDrive/skill_transfer/skill_transfer.html`
   - `seed/skill_transfer_tags.csv` → `/content/drive/MyDrive/skill_transfer/seed/skill_transfer_tags.csv`
2. Colab のシークレット（左の鍵アイコン）に `OPENAI_API_KEY`（既存と同じ）が登録されていることを確認する。
   Claude を使う場合は `ANTHROPIC_API_KEY` も登録する
3. 新しいセルに `skill_transfer_cell.py` の中身を貼って実行する（関数と設定を定義するだけで、すぐ終わる）
4. いつもどおり `audio_analysis_pipeline.py` のセルを実行する。
   サーバーの起動時に「✅ 技能伝承機能を登録しました」と表示される
5. 表示された ngrok の URL の末尾に `/skill` を付けてスマホのブラウザで開く（初回だけ「Visit Site」を押す）

**実行順が大事**：技能伝承のセル → 既存のセル の順に実行する。既存のセルは最後にサーバーを起動したまま止まるため、
後から技能伝承のセルを実行しても登録されない。その場合は既存のセルを止めて、もう一度実行する。

技能伝承のセルは、uvicorn のサーバー起動処理に「起動の直前に技能伝承の API を登録する」処理を差し込んでいる。
登録に失敗しても（例: Drive がマウントされていない）、エラーを表示したうえで既存の機能はそのまま起動する。

### 設定

秘密情報と設定は環境変数で変えられる（`.env.example` 参照）。Colab ではシークレットから自動で読み込み、
それ以外の値は技能伝承のセルの先頭で `os.environ["..."] = "..."` と書くか、`/content/.env` に書く。

| 環境変数 | 既定値 | 内容 |
| --- | --- | --- |
| `SKILL_LLM_PROVIDER` | `openai` | `openai`（既存と同じ gpt-4o）または `anthropic`（Claude） |
| `SKILL_LLM_MODEL` | 空欄 | 空欄なら openai は `gpt-4o`、anthropic は `claude-opus-5-5` |
| `SKILL_DB_PATH` | `/content/drive/MyDrive/corpus.db` | 既存と同じ DB |
| `SKILL_MEDIA_DIR` | `/content/drive/MyDrive/skill_transfer` | 動画・写真・PDF の保存先 |
| `SKILL_SUBTITLE_MAX_CHARS` / `SKILL_SUBTITLE_MAX_LINES` | `20` / `2` | 字幕 1 行の文字数と行数 |
| `SKILL_WHISPER_LANGUAGE` | `ja` | 音声認識の言語 |
| `SKILL_TAG_FUZZY_THRESHOLD` | `85` | タグ一覧との文字列の類似度（0〜100）。これ以上なら同じタグとみなす |
| `SKILL_DIFFICULTY_TAG` | `false` | 手順の数・注意点の数から難易度タグ（初級・中級・上級）を付ける |
| `SKILL_ANONYMIZE` | `false` | 既存の匿名化（[MASK] 置換）を技能伝承にもかける |

### 使い方（画面）

| 画面 | 操作 |
| --- | --- |
| ホーム | 作業の種類のボタンをタップして階層をたどる（例: 躯体工事 → 型枠工事 → 型枠組立）。「〜の動画をすべて見る」で下位の分類も含めた一覧へ |
| 動画一覧 | 上部のタグボタンをタップして絞り込む。複数選ぶと、選んだタグをすべて持つ動画だけになる |
| 動画再生 | 字幕の表示・非表示、手順と場面タグ（タップでその時間から再生）、再生バーの下の場面の印、手順書 PDF のダウンロード、文字起こし全文 |
| 撮影・投稿 | 「撮影する」でカメラが開く。「ファイル選択」で撮影済みの動画も選べる。タイトルを入れて投稿すると処理状況が表示される。失敗したら「失敗したところから再実行」 |
| タグ管理（PC 向け、ホームの下のリンク） | タグの追加・編集・削除、別名の登録、新タグ候補（タグ一覧に無かった言葉）の採用・却下、「全動画のタグを付け直す」 |

### 自動処理の流れ

投稿すると、次の 4 段階を順に行う。状態・失敗した段階・エラー内容は `skill_videos` 表に残り、失敗した段階からやり直せる。

1. **動画の変換**：再生しやすい H.264 の MP4 に変換し、サムネイルを作る
2. **① 文字起こし・字幕**：音声を取り出し（16kHz・モノラル・音量正規化）、noisereduce で騒音を減らし、Whisper で認識する。
   言語は日本語に固定し、タグ一覧の用語を語彙のヒントとして渡す。文ごとに区切って `skill_segments` に保存し、字幕（WebVTT）を作る
3. **② 手順書**：LLM が「まず」「次は」などを手がかりに手順に分け、道具・資材・注意点・コツを取り出す（固定の形の JSON を検証し、だめなら 1 回だけ再試行）。
   手順の時間は文字起こしの時間から決め、各手順の中ほどの 1 コマを写真にして PDF を作る
4. **③ タグ付け**：LLM が手順ごとに作業名・道具・資材などの言葉を取り出し、タグ一覧と照合する
   （名前の完全一致 → 別名 → 文字列の類似度 → 言葉の中にタグ名が入っている、の順）。
   一覧に無い言葉は新タグ候補として残し、自動では登録しない。手順の時間帯に場面タグを付け、まとめて動画タグにする

タグ一覧を変えた後は、タグ管理の「全動画のタグを付け直す」を押す（保存済みの抽出結果を使うので LLM は呼ばない）。

### ①②③ を単独で実行する

各段階は関数として分かれている。サーバーを起動していない Colab のセッションで、既存セルのモデル読み込みまで済んでいれば、次のように 1 本ずつ実行できる。

```python
ctx = SkillContext(SerializedConnection(open_skill_db()), whisper_model=whisper_model,
                   llm_fn=make_llm_fn(openai_client=client), run_in_background=False)
step_transcribe(ctx, video_id)   # ① 文字起こし・字幕
step_procedure(ctx, video_id)    # ② 手順書
step_tagging(ctx, video_id)      # ③ タグ付け
run_skill_pipeline(ctx, video_id, start_step="procedure")  # 指定した段階から最後まで
```

字幕（`build_webvtt`）、手順の検証（`validate_procedure`）、タグの照合（`match_tag`）などは、DB やモデルが無くても単独で使える。

### テスト

```bash
pip install -r requirements-dev.txt
PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 pip install playwright   # 画面のテスト用（任意）
pip install pyopenjtalk-plus                                  # サンプル動画の通しテスト用（任意）
pytest
```

- 音声認識（Whisper）と LLM はモックに差し替える。音声の取り出し・騒音除去・動画の変換・PDF 作成は本物（ffmpeg が必要）
- 既存のセルは編集せず、`tests/conftest.py` のローダーで `!` の行を除き、GPU モデルや Colab 専用のライブラリをスタブにして読み込む。
  既存の API がすべて残ること、対話研究のデータの見え方が変わらないことを確かめる
- `tests/test_skill_sample_end_to_end.py`：日本語の音声合成に騒音を混ぜた HEVC の .mov を作り、Colab と同じ実行順でサーバーを起動して、
  スマホ幅のブラウザで投稿 → 再生 → 手順書 → 一覧 → 絞り込み まで通しで確かめる
- Playwright・pyopenjtalk が無い環境では、該当するテストはスキップされる

### 注意点

- 大きな動画は ngrok 経由のアップロードに時間がかかる（分割アップロードは未対応）
- GPU 上の Whisper を共用するため、技能伝承の処理は 1 本ずつ順番に行う（対話研究の解析と同時に動かすとメモリが足りなくなることがある）
- 用語ヒントを全区間に効かせる `carry_initial_prompt` は新しい版の openai-whisper にある。古い版では最初の 30 秒だけに効く（どちらでも動く）
- PDF の日本語フォントのため、技能伝承のセルで `fonts-noto-cjk` を入れている
- ログイン・権限、承認フロー、顔や図面の自動ぼかしは MVP の範囲外（ngrok の URL を知っていれば誰でも見られる点は既存と同じ）
