# 技能伝承動画共有機能：既存リポジトリとの対応づけと実装計画

CLAUDE.md「作業0」の成果物。コードはまだ書いていない。
調査対象は `main` 相当の現行コード（コミット `60aa9a7` 時点）。

---

## 1. 技術構成

| 項目 | 現状 | 根拠 |
| --- | --- | --- |
| 言語 | Python 3（Google Colab のセルとして書かれた `.py`） | `audio_analysis_pipeline.py` 1行目 `!pip install ...`、末尾のトップレベル `await server.serve()` |
| Web フレームワーク | FastAPI + uvicorn（`nest_asyncio` で Colab 上に起動）、pydantic でリクエスト定義 | `audio_analysis_pipeline.py:29-44, 743-772, 1242-1245` |
| 公開方法 | ngrok で 8000 番をトンネル公開。CORS は全許可 | `audio_analysis_pipeline.py:36-43, 1227-1240` |
| DB | SQLite（`/content/drive/MyDrive/corpus.db`、WAL、`check_same_thread=False` で全スレッド共有の `conn`） | `audio_analysis_pipeline.py:97-180` |
| 音声処理 | pydub（内部で ffmpeg）で 16kHz・モノラル化＋音量正規化 | `audio_analysis_pipeline.py:603-606` |
| 話者分離 | pyannote `speaker-diarization-3.1`（`TARGET_NUM_SPEAKERS=4` 固定、閾値を 0.85 倍に調整） | `audio_analysis_pipeline.py:50, 183-202` |
| 音声認識 | openai-whisper `medium`（faster-whisper ではない）。幻覚除去フィルタつき | `audio_analysis_pipeline.py:204, 467-538` |
| LLM | OpenAI `gpt-4o`（JSON モード）、埋め込みは `text-embedding-3-small` | `audio_analysis_pipeline.py:45, 284-289, 724-725` |
| 秘密情報 | `google.colab.userdata`（Colab のシークレット）。`.env` は無い | `audio_analysis_pipeline.py:45, 1227` |
| フロントエンド | リポジトリ外の単一 HTML ファイル `kokusai.html`（提供いただいたもの）。CSS・JavaScript をファイル内に直書き、ライブラリ・ビルド無し、`const BASE_URL`（ngrok URL を貼る）へ `fetch`、ヘッダ `ngrok-skip-browser-warning` を付与、アップロード後は `/jobs/{id}` を 3 秒ごとにポーリング。VS Code の配色変数を使っており PC で開く前提 | `kokusai.html` |
| 起動方法 | Colab のセルに `audio_analysis_pipeline.py` を貼って実行 → 表示された ngrok URL を `kokusai.html` の `BASE_URL` に貼る（精度ツールは別セル） | `audio_analysis_pipeline.py:1219-1245` |
| テスト | **無い**（テストフレームワーク・`tests/`・CI いずれも無し）。`transcription_accuracy_tool.py` は文字起こし精度（WER/CER）を測る研究用ツールで、ソフトウェアのテストではない | — |
| 依存関係の管理 | `requirements.txt` などは無く、セル冒頭の `!pip install` のみ | — |
| その他 | `README`、`.env.example`、`seed/` も無い | — |

**重要な制約**：両ファイルとも Colab セル専用の書き方（`!` コマンド、トップレベル `await`、import 時にモデル読み込み・Drive の DB 接続・サーバー起動）のため、**普通の Python モジュールとして import できない**。
そのため、既存コードをそのまま単体テストから呼ぶことはできない（テストの方法は 5.2 参照）。

---

## 2. データモデル

| テーブル | 主な列 | 役割 |
| --- | --- | --- |
| `sessions` | id, filename, created, summary, metadata(JSON), roles(JSON) | 対話セッション（記録単位）。`filename` はアップロード時の一時ファイル名 `input_xxx` |
| `turns` | id, session_id, speaker, start, end, text, phase, intent, role, embedding(BLOB) | 発話＝時間同期つき書き起こし。phase / intent は GPT-4o の自動ラベル |
| `addin_results` | id, session_id, addin_name, method_version, created, result(JSON) | 新しい分析手法の結果をセッションに版つきで何度でも追加できる汎用置き場 |
| `human_annotations` | turn_id, annotator_id, phase, intent, confidence, note | 発話単位の人手ラベル（turn×annotator で一意） |
| `reference_transcripts` | turn_id, transcriber_id, reference_text | 発話単位の正解書き起こし（WER/CER 用） |

存在しないもの：

- **話者・参加者のテーブル**：無い。`turns.speaker`（`SPEAKER_00` 等）と `sessions.roles`（JSON）のみ
- **メディアファイルの管理**：無い。アップロード動画は作業ディレクトリに一時保存されるだけで、保存場所も記録されず、再生用に配信もしていない
- **ラベルの定義・ラベルセット（マスタ）**：無い。phase / intent はコード中の定数（`PHASE_ORDER`, `INTENT_LABELS`）
- **区間アノテーション（任意の時間範囲につくラベル）**：無い。ラベルはすべて「発話 1 件」単位
- **セッション単位のタグ**：無い（`sessions.metadata` に scene / task の自由文があるだけ）
- **処理状態の永続化**：無い。ジョブ状態はメモリ上の `jobs` 辞書のみで、再起動で消える

---

## 3. 既にある処理

| 処理 | 実装 | 技能伝承での再利用性 |
| --- | --- | --- |
| 動画アップロード → 非同期ジョブ | `POST /upload` → スレッドで処理、`GET /jobs/{job_id}` でポーリング（ngrok のタイムアウト対策） | **方式を流用**（すぐ ID を返し、裏で処理、画面はポーリング）。状態はメモリでなく `skill_videos` に保存し、再起動後も確認・再実行できるようにする |
| 音声取り出し | pydub で 16kHz・モノラル・正規化 → WAV 書き出し | **流用**（ffmpeg を直接呼ぶ必要なし）。騒音除去だけ追加 |
| 文字起こし | `transcribe_full_audio()`：全体を 1 回で Whisper に通す、単語タイムスタンプ、幻覚・繰り返し除去 | **読み込み済みの Whisper モデルを共用し、同じ設定・同じフィルタを技能伝承セルに複製**（言語指定・用語ヒントを足すため。既存関数は編集しない）。精度ツールと同じく「変更したら両方に反映」と注記する |
| 話者分離 | pyannote（4 人固定） | **使わない**。説明者 1 人の作業動画なので不要（4 人固定だと誤分割する） |
| 匿名化 | `extract_mask_targets_with_gpt()` で固有名詞を `[MASK]` 置換 | 使うかどうか要判断（Q5） |
| 発話ラベル付け | `tag_turns_with_gpt()`（phase / intent） | 使わない（対話研究用のラベル体系） |
| 要約・メタデータ | `run_full_analysis()` 内の GPT-4o 呼び出し | 使わない（手順書生成で代替） |
| 埋め込み・意味検索 | `POST /search`（全セッションの turns を横断してコサイン類似度） | 使わない。技能伝承は別の表に保存するので、研究用の検索結果は変わらない（Q1＝B 案） |
| 一覧・詳細取得 | `GET /corpus`, `GET /corpus/{id}`, `GET /corpus/{id}/raw_turns` | API の形（一覧／詳細／`raw_turns` 相当）を真似る。既存の一覧には技能伝承は出ない（Q1＝B 案） |
| 版つき再分析 | `POST /corpus/{id}/retag`, `/addin` | **考え方を流用**（手順書・タグ付けの再実行時に `method_version` を残す） |
| 時間区間テキストの読み込み | `transcription_accuracy_tool.py` の `parse_reference_file()`（`00:00:00.000 --> ...` 形式） | WebVTT と同じ時刻表記。字幕の出力形式を揃える参考にする |
| 精度評価（WER/CER） | `transcription_accuracy_tool.py` | 技能伝承は別の表なので、そのままでは使えない（Q1＝B 案を選んだことによる制約。必要になれば別途対応） |
| エクスポート | グラフ PNG（base64）、精度レポート CSV | 字幕（WebVTT）・PDF は新規 |

---

## 4. 対応づけ（調査結果で修正）

| 技能伝承での概念 | 想定 | 調査結果 | 方針 |
| --- | --- | --- | --- |
| 作業動画 | 対話セッション＋メディアファイル | `sessions` はあるが、メディアファイル管理は無い | **新規（`sessions` を手本に）**：`skill_videos` に動画情報・動画ファイル・処理状態を持つ。`sessions` には入れない |
| 説明者（熟練者） | 話者・参加者 | 話者テーブルは無い。`turns.speaker` の文字列のみ | **同じ形で新規**：`skill_segments.speaker` に説明者名（未入力なら `SPEAKER_00`）。話者分離はしない |
| 文字起こしセグメント（開始・終了時間付き） | 発話・書き起こし | `turns`（start / end / text） | **同じ形で新規**：`turns` と同じ列構成の `skill_segments` に保存 |
| 字幕 | 書き起こしの時間情報から生成 | 生成処理は無い | **新規**：`skill_segments` から WebVTT を生成 |
| 手順（時間帯つき） | 区間アノテーション | 区間アノテーションは無い（ラベルは発話単位のみ） | **新規**：区間アノテーション表 `skill_annotations` を作り、`layer='step'` で保存 |
| タグ一覧（タグマスタ） | ラベルの定義・ラベルセット | ラベル定義は無い（コード中の定数） | **新規**：`skill_tags`（名前・分類・親・別名） |
| 動画タグ／場面タグ | セッション単位／区間単位のアノテーション | どちらも無い | **新規**：`skill_annotations` の `layer='scene_tag'`（時間あり）／`'video_tag'`（時間なし＝動画単位） |
| 新タグ候補 | — | 無い | **新規**：`skill_tag_candidates` |
| タグでの絞り込み | 既存の検索・フィルタ | 既存は埋め込みによる意味検索のみ。タグ・分類での絞り込みは無い | **新規**：AND 絞り込み・分類の下位を含む一覧の API |
| 手順書 | 新規（LLM → PDF） | — | **新規**。LLM 結果の原本は `addin_results` と同じ形の `skill_llm_results`（版つき）に残す＝既存の再分析の考え方を流用 |
| 処理状況・再実行 | — | メモリ上の `jobs` のみ | **新規**：処理状態・失敗した段階・エラー内容を `skill_videos` に保存し、画面から再実行できるようにする |

---

## 5. 実装計画

### 5.1 追加するテーブル（既存テーブルには一切触れない＝Q1 は B 案）

同じ `corpus.db` に、`skill_` で始まる表だけを追加する。既存の 5 つの表・既存 API の SQL は変更しないので、研究用の一覧・検索・精度ツールの結果は今と同じ。

```sql
-- 作業動画（sessions に相当）。動画ファイルと処理状態も持つ
CREATE TABLE IF NOT EXISTS skill_videos (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT,
    explainer    TEXT,
    filename     TEXT,      -- アップロード時の元のファイル名
    media_path   TEXT,      -- 保存した動画（再生用）
    duration     REAL,
    status       TEXT,      -- uploaded / transcribing / transcribed / procedure / tagging / done / error
    failed_step  TEXT,      -- どの段階で失敗したか（再実行の起点）
    error        TEXT,
    vtt_path     TEXT,
    pdf_path     TEXT,
    created      TEXT,
    updated      TEXT
);

-- 文字起こしセグメント（turns と同じ列構成。研究用の phase/intent/role/embedding は持たない）
CREATE TABLE IF NOT EXISTS skill_segments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id   INTEGER,
    speaker    TEXT,
    start      REAL,
    end        REAL,
    text       TEXT,
    FOREIGN KEY (video_id) REFERENCES skill_videos(id)
);

-- 区間アノテーション（layer で種類を分ける。start/end が NULL なら動画単位）
CREATE TABLE IF NOT EXISTS skill_annotations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id       INTEGER,
    layer          TEXT,    -- 'step' / 'scene_tag' / 'video_tag'
    start          REAL,
    end            REAL,
    label          TEXT,    -- 手順タイトル、タグ名など
    tag_id         INTEGER, -- タグのとき skill_tags.id
    parent_id      INTEGER, -- 場面タグ → 手順（skill_annotations.id）
    payload        TEXT,    -- JSON（手順の説明・道具・資材・注意点・写真など）
    source         TEXT,    -- 'llm' / 'human'
    method_version TEXT,
    created        TEXT,
    FOREIGN KEY (video_id) REFERENCES skill_videos(id)
);

-- LLM の生の結果（addin_results と同じ考え方。手順分割・タグ抽出をやり直しても上書きせず版で残す）
CREATE TABLE IF NOT EXISTS skill_llm_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id       INTEGER,
    kind           TEXT,    -- 'procedure' / 'tag_terms'
    method_version TEXT,
    created        TEXT,
    result         TEXT,
    FOREIGN KEY (video_id) REFERENCES skill_videos(id)
);

-- タグマスタ（作業の種類は category='作業' の階層として表す）
CREATE TABLE IF NOT EXISTS skill_tags (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT UNIQUE,
    category  TEXT,          -- 作業 / 道具 / 資材 / 安全 / 難易度 など
    parent_id INTEGER,
    aliases   TEXT,          -- JSON 配列
    created   TEXT
);

-- 新タグ候補（自動登録はしない。管理画面で採用／却下）
CREATE TABLE IF NOT EXISTS skill_tag_candidates (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id   INTEGER,
    word       TEXT,
    category   TEXT,
    status     TEXT,         -- pending / adopted / rejected
    created    TEXT
);
```

既存の表の書き方（`id INTEGER PRIMARY KEY AUTOINCREMENT`、`created TEXT`、JSON は TEXT 列）に合わせる。

### 5.2 ファイル構成と Colab での動かし方

> **変更あり**：この節の「既存セルのサーバー起動時に登録する」方式はやめ、技能伝承のセル1つで完結する形に変えた（9章）。DB も `corpus.db` ではなく専用の `skill_transfer.db` にした。

既存の運用（Colab はセルに貼り付け、画面は単一 HTML）に合わせる。

```
skill_transfer_cell.py       # 新しい Colab セル（技能伝承の処理と API を 1 ファイルに。精度ツールと同じ「別セル」形式）
skill_transfer.html          # 新しい画面（kokusai.html と同じ書き方：単一ファイル、CSS/JS 直書き、ライブラリ無し）
seed/skill_transfer_tags.csv # タグ一覧の初期データ
tests/                       # pytest（新規導入）。LLM・音声認識はモック
  conftest.py                # セルファイルを読み込むローダー（後述）
.env.example
README.md
```

**既存ファイルは編集しない**（ご指示）。`audio_analysis_pipeline.py`・`transcription_accuracy_tool.py` は 1 行も変えず、技能伝承の処理はすべて新しいファイルに書く。

**Colab での実行順**

1. `skill_transfer_cell.py` を貼ったセルを実行（関数と設定の定義だけ。モデル読み込みやサーバー起動はしない）
2. 既存の `audio_analysis_pipeline.py` のセルを **いつもどおり** 実行

既存セルは最後にサーバーを起動したまま止まる（`await server.serve()`）ため、後から別セルで API を足すことはできない。
そこで技能伝承セルは、uvicorn のサーバー起動処理（`uvicorn.Server.serve`）に「起動の直前に技能伝承の API を `app` に登録する」処理を差し込んでおく。
Colab ではセル同士が同じ変数の置き場を共有しているので、起動の時点で既存セルが作った `app`・`whisper_model`・`client`・`extract_mask_targets_with_gpt` をそのまま受け取れる。

- 技能伝承セルを実行しなければ何も差し込まれないので、**既存セルだけを動かしたときの動作は今と同じ**
- 登録に失敗しても、エラーを表示したうえで既存のサーバーはそのまま起動する
- 既存の Whisper モデル（GPU に載っているもの）・OpenAI クライアントを共用するので、モデルを二重に読み込まない
- DB は既存と同じ `corpus.db` に、技能伝承セル側で別の接続を開いて `skill_` の表だけを読み書きする

**画面の配信**：`kokusai.html` のようにファイルを開いて ngrok URL に `fetch` する方式は、スマホでは HTML ファイルを開く手段が無いうえ、`<video>` タグには `ngrok-skip-browser-warning` ヘッダを付けられず ngrok の警告ページで再生が止まる。
そのため `skill_transfer.html` は書き方を `kokusai.html` に揃えたうえで、**FastAPI から `/skill` で配信**し、スマホでは「ngrok URL/skill」を開く（初回だけ「Visit Site」を押せば以後は動画も再生できる）。
`BASE_URL` は既定で配信元（`location.origin`）を使い、`kokusai.html` と同じく PC でファイルを直接開いて URL を貼る使い方もできるようにする。HTML ファイルは Drive（`SKILL_MEDIA_DIR` と同じ場所）に置き、セルからそのパスを読む。

**テストの方法**：セルファイルは `!pip install` 行があるため、そのままは import できない。`tests/conftest.py` に「`!` で始まる行を除いて読み込むローダー」を用意し、
- `skill_transfer_cell.py`：そのまま読み込んでテスト（重い処理は引数で注入するため、スタブは LLM・音声認識だけ）
- `audio_analysis_pipeline.py`（既存 API の回帰確認）：同じローダーで、torch・whisper・pyannote・colab などをスタブにし、DB パスを一時ファイルに置き換え、「5. サーバー起動」以降を除いて読み込む。技能伝承セルと同じ変数の置き場で読み込むことで Colab の実行順を再現し、既存の API がすべて残ること、`/corpus` などが技能伝承の追加前と同じ結果を返すことを確かめる

この方法なら、テストのために既存コードを書き換える必要はない（前回提案した `CORPUS_DB_PATH` の変更は取り下げる）。

### 5.3 流用／拡張／新規の一覧

| 区分 | 内容 |
| --- | --- |
| 流用 | `sessions`・`turns`・`addin_results` の列構成（新しい表の手本）、読み込み済みの Whisper モデルと `transcribe_full_audio()` の設定・フィルタ、pydub による音声変換・正規化、ID を返してポーリングする方式、OpenAI クライアント、匿名化関数（設定で ON のとき）、pydantic、FastAPI の `app` |
| 拡張 | なし。既存ファイル・既存の表・既存 API はいずれも変更しない（5.2、Q1＝B 案） |
| 新規 | `skill_transfer_cell.py`・`skill_transfer.html`、上記 6 テーブル、騒音除去、文分割、WebVTT、LLM 呼び出しの集約、手順分割＋検証＋再試行、PDF、タグ照合（rapidfuzz）、タグ絞り込み API、動画ファイル配信、画面一式、seed CSV、pytest、`.env.example`、README |

### 5.4 各段階の作業とテスト

| 段階 | 作業 | テスト（すべて pytest、一時 SQLite、LLM・音声認識はモック） |
| --- | --- | --- |
| 1 | 6 テーブル作成、seed CSV と投入処理 | テーブル作成が冪等であること、既存の表に変更が無いこと、CSV の親子・別名が正しく入ること、再投入で重複しないこと |
| 2 | アップロード API（`/skill/upload`）、動画保存、音声取り出し・騒音除去、文分割→`turns` 保存、WebVTT | 文分割と時刻、1 行 N 文字×最大 2 行での区切り、文字数比例の時間配分、VTT の書式、失敗時に `status=error` とエラー内容が残り再実行できること |
| 3 | LLM で手順分割（固定スキーマ JSON・検証・1 回だけ再試行）、セグメント番号→時間の決定、区間アノテーション保存、各手順の代表フレームを ffmpeg で静止画に切り出し、写真付き PDF | 正常系、不正 JSON → 再試行成功／再試行も失敗で error、時間がセグメントから決まること、PDF が生成されること（WeasyPrint が無い環境ではスキップ表示） |
| 4 | 用語抽出、完全一致→別名一致→類似度の順で照合、新タグ候補、場面タグ、難易度（設定で ON/OFF）、動画タグ集約 | 照合の優先順位と閾値、候補が自動登録されないこと、場面タグの時間が手順と一致、動画タグ＝場面タグの和集合 |
| 5 | 画面（`skill_transfer.html`、`/skill` で配信）：アップロード（`capture="environment"`、処理状況表示）、再生（字幕 ON/OFF、場面タグ一覧で頭出し、再生バーに印、PDF ダウンロード）、動画配信（Range 対応） | API のテスト（TestClient）。画面は Playwright（環境にある Chromium）でスマホ幅の表示確認 |
| 6 | 画面：ホーム（作業の種類を階層でたどる）、一覧（タグボタンで AND 絞り込み）、タグ管理（追加・編集・削除・別名・候補の採用） | 下位分類を含む一覧、AND 絞り込み、タグ CRUD・候補採用の API テスト |
| 7 | サンプル動画（ffmpeg で合成した音声付き動画、または用意していただく動画）で通し確認 | ①〜③を通しで実行（ここだけ実モデルを使うかは環境次第。Colab での確認手順を README に記載） |

各段階で「既存の対話研究の機能が変わっていないこと」を確認する回帰テストも走らせる（5.2 のローダーによる）。

### 5.5 設定（`.env.example` に追記予定）

`OPENAI_API_KEY`、（Q2 次第で）`ANTHROPIC_API_KEY`・`SKILL_LLM_PROVIDER`・`SKILL_LLM_MODEL`、`SKILL_MEDIA_DIR`（既定 `/content/drive/MyDrive/skill_transfer`）、
`SKILL_SUBTITLE_MAX_CHARS=20`、`SKILL_SUBTITLE_MAX_LINES=2`、`SKILL_TAG_FUZZY_THRESHOLD=85`、`SKILL_DIFFICULTY_TAG=false`、`SKILL_WHISPER_LANGUAGE=ja`。
Colab では従来どおりシークレットから `os.environ` に入れる形にし、ローカルでは `.env` を読む。

### 5.6 方針資料（「建設業向け 技能伝承動画共有システム 方針」）との照合

CLAUDE.md と方針資料を突き合わせ、計画に次の点を反映・確認する。

| 方針資料の記述 | CLAUDE.md | 計画での扱い |
| --- | --- | --- |
| ② 出力は「**写真付き**手順書（PDF）」、入力に撮影動画を含む | 「PDF出力」のみ | **反映**：各手順の時間帯の中ほどから ffmpeg で静止画を 1 枚切り出し、手順ごとに PDF に載せる。画像のパスは手順の `payload` に保存（段階 3） |
| 課題「文字起こしの精度（騒音・専門用語・方言）」 | 騒音除去のみ | **反映**：騒音除去に加え、タグ一覧の名前・別名を Whisper の `initial_prompt`（語彙のヒント）に渡す案（Q6）。方言は特別な処理はせず、既存の精度ツール（WER/CER）で効果を測れるようにする |
| 課題「コツの言語化」 | 手順から「タイトル・説明・道具・資材・注意点」 | **提案**：手順のスキーマに「コツ・ポイント」欄を足す（Q8） |
| 課題「タグのばらつき」 | 別名・類似度照合・新タグ候補 | **計画どおり**：別名登録、類似度での照合、新タグ候補を管理画面で採用 |
| 難易度判定は「必要であれば」 | 設定で ON/OFF | **計画どおり**：既定 OFF |
| 「承認された動画と手順書を共有」 | 承認フローは作らない | **MVP 外のまま**（Q9）。後で足せるよう `skill_videos.status` に公開状態を追加しやすい形にしておく |
| 対策「自動ぼかし」「公開範囲の設定」「撮影ルール」 | 自動ぼかしは作らない、権限は新設しない | **MVP 外のまま**（Q9）。撮影ルールは運用で対応 |
| 閲覧「タップ操作だけ」 | スマホ優先、タップ中心 | **計画どおり** |

---

## 6. 判断が必要だった点（回答済み：Q1＝B 案、Q2〜Q9＝推奨どおり）

**Q1. 技能伝承の動画を既存の `sessions`/`turns` に入れるか** → **回答済み：B 案（別の表）**。
既存の表・既存 API には触れず、`skill_` で始まる表を新しく作る（5.1）。研究用の一覧・検索・精度ツールには技能伝承の動画は出てこない。
その代わり、精度ツール（WER/CER）は技能伝承の文字起こしには使えない。

**Q2. LLM はどれを使うか**
既存は OpenAI `gpt-4o`。CLAUDE.md の「既存にあるものが優先」に従うと OpenAI になる。
推奨：呼び出しを技能伝承セル内の 1 つの関数に集約し、**既定は既存と同じ OpenAI**、環境変数 `SKILL_LLM_PROVIDER=anthropic` で Claude に切り替えられるようにする。Claude を既定にしたい場合はお知らせください。

**Q3. 画面（フロントエンド）** → **回答済み**：既存は単一 HTML（`kokusai.html`）。同じ書き方で `skill_transfer.html` を作り、スマホから開けるよう FastAPI から配信する（5.2）。`kokusai.html` には手を入れない。

**Q4. コードの置き方** → **回答済み**：Colab はセルに貼り付け。技能伝承は別セル `skill_transfer_cell.py` にし、既存セルは編集しない（技能伝承セルを先に実行すると、既存セルのサーバー起動時に自動で登録される。5.2）。
テストは pytest を新たに導入し、セルファイルを読み込むローダーで既存 API の回帰確認も行う（既存コードの書き換えは不要）。

**Q5. 匿名化を技能伝承動画にもかけるか**
既存の `[MASK]` 置換は、道具の商品名・メーカー名・現場名まで伏せる可能性があり、手順書やタグ付けの質が落ちます。推奨は**既定 OFF（設定で ON 可）**。個人名を伏せる必要があれば ON にします。

**Q6. Whisper の言語指定と専門用語ヒント** → **回答済み（既存ファイルは編集しない形に変更）**
既存の `transcribe_full_audio()` には引数を足さず、技能伝承セルに同じ設定・同じフィルタの関数を作り、そこで言語（`ja`）とタグ一覧の用語ヒント（`initial_prompt`）を渡す。Whisper モデルは既存セルで読み込んだものを共用する。
既存は `condition_on_previous_text=False` なので、openai-whisper の版によっては `initial_prompt` が最初の 30 秒にしか効かない。全体に効かせる `carry_initial_prompt` がある版では自動で使う。

**Q7. タグ一覧の初期データ**
実際に使うタグ（作業の分類・道具・資材など）の一覧があれば `seed/skill_transfer_tags.csv` に使います。無ければ、型枠・鉄筋・内装などのサンプルを私が作ります。

**Q8. 手順書に「コツ・ポイント」欄を足すか**（方針資料の課題「コツの言語化」より）
LLM に「〜するのがコツ」「感覚としては〜」のような説明を手順ごとに抜き出させ、手順書に注意点とは別の欄として載せる案です。推奨は**追加する**。

**Q9. 承認・自動ぼかし・公開範囲**
方針資料には「承認された動画を共有」「自動ぼかし」「公開範囲の設定」がありますが、CLAUDE.md では MVP 外です。**MVP では作らず**、アップロード後は自動処理が終わればすぐ閲覧できる形で進めてよいか確認させてください。

---

## 7. 気づいた点・リスク（今回は既存コードに手を入れない）

- **動画サイズと ngrok**：スマホ動画は数百 MB になりうる。分割アップロードは MVP 外なので、まずは 1 回送信で、大きすぎる場合の上限とエラー表示を設ける。
- **再生互換性**：iPhone の HEVC/MOV は Android の Chrome で再生できないことがある。保存時に ffmpeg で H.264/AAC の MP4（faststart）に変換する案を段階 2 に含める。
- **GPU の取り合い**：既存の対話解析と技能伝承の処理が同時に走ると Whisper を共有するため、技能伝承側で処理を直列化するロックを置く。
- **PDF の日本語フォント**：Colab で WeasyPrint を使うには日本語フォント（`fonts-noto-cjk`）の導入が必要。README に手順を書く。
- **ngrok の警告ページ**：無料枠では最初に「Visit Site」画面が出る（既存の案内と同じ）。画面を同じオリジンから配信すれば、1 回押せば以後は動く。
- **既存コードで目についた点（参考、今回は変更しない）**：`sessions.filename` が一時ファイル名（`input_xxx`）で元動画が残らない／アップロードファイル名をそのままパスに使っている／ジョブ状態が再起動で消える／WER・CER の計算がサーバーと精度ツールで重複。
- **既存画面の表示**：`kokusai.html` は検索結果などのテキストを `innerHTML` にそのまま入れている。新しい画面では文字起こしやタグ名を必ずエスケープして表示する（既存画面は変更しない）。
- **認証**：既存にログイン・権限は無いため、CLAUDE.md に従い新設しない（ngrok URL を知っていれば誰でも見られる点は既存と同じ）。

---

## 8. 実装結果（段階 1〜7）

計画からの変更点と、実装中に見つかって直した問題。

| 項目 | 計画 | 実装 |
| --- | --- | --- |
| 既存ファイル | サーバー起動直前に登録の数行を追加 | **一切編集しない**（ご指示）。技能伝承セルが `uvicorn.Server.serve` に登録処理を差し込む（5.2） |
| 文字起こし | `transcribe_full_audio()` に引数を追加 | 既存関数は編集せず、同じ設定・同じフィルタの関数を技能伝承セルに複製。Whisper モデルは共用 |
| 処理状況 | 既存の `jobs`/`/jobs/{id}` を共用 | `skill_videos` に状態を保存し、画面は `/skill/api/videos/{id}` をポーリング（再起動後も残る） |
| 手順書 | 道具・資材・注意点 | 方針資料に合わせて「コツ」欄を追加（Q8）、各手順の写真付き |
| タグ照合 | 完全一致 → 別名 → 類似度 | 末尾の長音や全角半角の違いをそろえたうえで、完全一致 → 別名 → 類似度 → 言葉の中にタグ名が入っている（一番長い名前）の順 |
| 付け直し | — | タグ一覧の変更後に、保存済みの抽出結果で全動画を付け直す機能を追加（LLM は呼ばない） |

**直した問題**

- **同時アクセスで DB の結果が空になる**：1 つの SQLite 接続を処理スレッドと画面の状況確認で同時に使うと、同じ SQL 文を取り合って空の結果や例外になる（4 スレッドで 300 回ずつ読み書きした再現では、空の結果が 121 回・例外が 334 回）。問い合わせ 1 回分を順番に行う `SerializedConnection` で解消
- **語彙ヒントが作業の用語だけで埋まる**：分類ごとに交互に入れ、正式名を別名より先に入れるように変更
- **照合の行き過ぎと取りこぼし**：「コンクリート」が「コンクリート打設」に一致していたのを防ぎ、話し言葉の「型枠の建て込み」が「型枠組立」（別名: 建て込み）に一致するようにした
- **縦向き動画の写真が大きすぎる**：手順書の写真の高さに上限を付けた

**テスト**：134 件（既存 API の回帰、①②③ の単体、API の通し、スマホ幅のブラウザ操作、サンプル動画の通し）。

**この環境で確かめられなかったこと**：Whisper のモデルと LLM の API は外部への通信が遮断されているため呼んでいない（テストではモック）。
実際の認識精度・LLM の出力の質は、Colab で実際の現場動画を使って確かめる必要がある。

---

## 9. 方針変更：技能伝承のセル1つで完結させる

ご指示：「audio_analysis_pipeline.py は今回なにも使わない。触らない。1つのセルを動かすだけで完結できるようにする」。

| 項目 | 変更前 | 変更後 |
| --- | --- | --- |
| 実行方法 | 技能伝承セル → 既存セルの順に実行し、既存セルのサーバー起動時に登録 | `skill_transfer_cell.py` のセル1つを実行するだけ。ドライブのマウント → Whisper の読み込み → ngrok → サーバー起動までセルの中で行う |
| Whisper・OpenAI | 既存セルが読み込んだものを共用 | セルの中で読み込む（モデルは同じ `medium`、`SKILL_WHISPER_MODEL` で変更可） |
| DB | 既存の `corpus.db` に `skill_` の表を追加 | ドライブの `skill_transfer/skill_transfer.db`（対話研究のデータには一切触れない） |
| 画面・タグの初期データ | ドライブに `skill_transfer.html` と CSV を置く | どちらもセルの中に入れた（CSV は `seed/skill_transfer_tags.csv` と同じ内容で、テストで一致を確認） |
| 匿名化（設定で ON） | 既存セルの関数を使う | セルの中の LLM 呼び出しで同じことをする |
| 既存セルとの関係 | サーバー起動処理に差し込み | なし。技術構成・書き方（FastAPI＋ngrok、SQLite、Whisper の設定と幻覚フィルタの値、ポーリング方式）を合わせただけ |

テスト：Colab 専用の部分（ドライブ・GPU・Whisper のモデル・ngrok）だけを差し替えてセルを丸ごと実行し、
セル1つで起動まで行われること、設定で変えられること、実行し直してもデータが残ることを確かめる。
サンプル動画の通しテストも、セル自身が起動したサーバーに対して行う。既存の 2 つのファイルが編集されていないこともテストで確かめる。全 136 件。
