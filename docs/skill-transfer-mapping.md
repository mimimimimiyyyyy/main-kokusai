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
| フロントエンド | **リポジトリ内には無い**。CORS 全許可と ngrok の案内文から、別の場所にある画面から API を呼んでいると推測 | — |
| 起動方法 | Colab で `audio_analysis_pipeline.py` をセルとして実行 → 表示された ngrok URL を開く | `audio_analysis_pipeline.py:1219-1245` |
| テスト | **無い**（テストフレームワーク・`tests/`・CI いずれも無し）。`transcription_accuracy_tool.py` は文字起こし精度（WER/CER）を測る研究用ツールで、ソフトウェアのテストではない | — |
| 依存関係の管理 | `requirements.txt` などは無く、セル冒頭の `!pip install` のみ | — |
| その他 | `README`、`.env.example`、`seed/` も無い | — |

**重要な制約**：両ファイルとも Colab セル専用の書き方（`!` コマンド、トップレベル `await`、import 時にモデル読み込み・Drive の DB 接続・サーバー起動）のため、**普通の Python モジュールとして import できない**。
そのため、既存コードをそのまま単体テストから呼ぶことはできない（後述の「判断が必要な点」Q4）。

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
| 動画アップロード → 非同期ジョブ | `POST /upload` → スレッドで処理、`GET /jobs/{job_id}` でポーリング（ngrok のタイムアウト対策） | **方式をそのまま流用**。`jobs` 辞書と `/jobs/{job_id}` も共用できる |
| 音声取り出し | pydub で 16kHz・モノラル・正規化 → WAV 書き出し | **流用**（ffmpeg を直接呼ぶ必要なし）。騒音除去だけ追加 |
| 文字起こし | `transcribe_full_audio()`：全体を 1 回で Whisper に通す、単語タイムスタンプ、幻覚・繰り返し除去 | **流用**。実データで調整済みのフィルタをそのまま活かす |
| 話者分離 | pyannote（4 人固定） | **使わない**。説明者 1 人の作業動画なので不要（4 人固定だと誤分割する） |
| 匿名化 | `extract_mask_targets_with_gpt()` で固有名詞を `[MASK]` 置換 | 使うかどうか要判断（Q5） |
| 発話ラベル付け | `tag_turns_with_gpt()`（phase / intent） | 使わない（対話研究用のラベル体系） |
| 要約・メタデータ | `run_full_analysis()` 内の GPT-4o 呼び出し | 使わない（手順書生成で代替） |
| 埋め込み・意味検索 | `POST /search`（全セッションの turns を横断してコサイン類似度） | MVP では使わない。ただし技能伝承の動画が混ざると研究用の検索結果が変わる（Q1） |
| 一覧・詳細取得 | `GET /corpus`, `GET /corpus/{id}`, `GET /corpus/{id}/raw_turns` | 詳細取得の考え方は流用。一覧は技能伝承が混ざらないよう配慮が必要（Q1） |
| 版つき再分析 | `POST /corpus/{id}/retag`, `/addin` | **考え方を流用**（手順書・タグ付けの再実行時に `method_version` を残す） |
| 時間区間テキストの読み込み | `transcription_accuracy_tool.py` の `parse_reference_file()`（`00:00:00.000 --> ...` 形式） | WebVTT と同じ時刻表記。字幕の出力形式を揃える参考にする |
| 精度評価（WER/CER） | `transcription_accuracy_tool.py` | 技能伝承の文字起こしも `turns` に入れれば、**騒音下での認識精度を同じツールで評価できる**（研究上の利点） |
| エクスポート | グラフ PNG（base64）、精度レポート CSV | 字幕（WebVTT）・PDF は新規 |

---

## 4. 対応づけ（調査結果で修正）

| 技能伝承での概念 | 想定 | 調査結果 | 方針 |
| --- | --- | --- | --- |
| 作業動画 | 対話セッション＋メディアファイル | `sessions` はあるが、メディアファイル管理は無い | **流用＋新規**：`sessions` に 1 行作り、動画ファイル・処理状態は新テーブル `skill_videos` で持つ |
| 説明者（熟練者） | 話者・参加者 | 話者テーブルは無い。`turns.speaker` の文字列のみ | **流用**：`turns.speaker` に説明者名（未入力なら `SPEAKER_00`）を入れる。話者分離はしない |
| 文字起こしセグメント（開始・終了時間付き） | 発話・書き起こし | `turns`（start / end / text）がそのまま使える | **流用**：`turns` に保存（phase / intent / embedding は NULL） |
| 字幕 | 書き起こしの時間情報から生成 | 生成処理は無い | **新規**：`turns` から WebVTT を生成（保存せず毎回生成、またはファイル保存） |
| 手順（時間帯つき） | 区間アノテーション | 区間アノテーションは無い（ラベルは発話単位のみ） | **新規**：汎用の区間アノテーション表 `segment_annotations` を作り、`layer='skill_step'` で保存 |
| タグ一覧（タグマスタ） | ラベルの定義・ラベルセット | ラベル定義は無い（コード中の定数） | **新規**：`skill_tags`（名前・分類・親・別名） |
| 動画タグ／場面タグ | セッション単位／区間単位のアノテーション | どちらも無い | **新規**：`segment_annotations` の `layer='skill_scene_tag'`（時間あり）／`'skill_video_tag'`（時間なし＝セッション単位） |
| 新タグ候補 | — | 無い | **新規**：`skill_tag_candidates` |
| タグでの絞り込み | 既存の検索・フィルタ | 既存は埋め込みによる意味検索のみ。タグ・分類での絞り込みは無い | **新規**：AND 絞り込み・分類の下位を含む一覧の API |
| 手順書 | 新規（LLM → PDF） | — | **新規**。LLM 結果の原本は `addin_results`（`addin_name='skill_procedure'`, 版つき）にも残す＝既存の再分析の考え方を流用 |
| 処理状況・再実行 | — | メモリ上の `jobs` のみ | **流用＋拡張**：進捗表示は既存 `jobs`/`/jobs/{id}` を共用、失敗状態とエラーは `skill_videos` に永続化して再実行可能にする |

---

## 5. 実装計画

### 5.1 追加するテーブル（既存テーブルは原則そのまま）

```sql
-- 作業動画（sessions 1 行に対応）。メディアファイルと処理状態を持つ
CREATE TABLE IF NOT EXISTS skill_videos (
    session_id   INTEGER PRIMARY KEY,
    title        TEXT,
    explainer    TEXT,
    media_path   TEXT,      -- 保存した動画（再生用）
    duration     REAL,
    status       TEXT,      -- uploaded / transcribing / transcribed / procedure / tagging / done / error
    failed_step  TEXT,      -- どの段階で失敗したか（再実行の起点）
    error        TEXT,
    vtt_path     TEXT,
    pdf_path     TEXT,
    created      TEXT,
    updated      TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- 区間アノテーション（汎用。layer で種類を分ける。start/end が NULL ならセッション単位）
CREATE TABLE IF NOT EXISTS segment_annotations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER,
    layer          TEXT,    -- 'skill_step' / 'skill_scene_tag' / 'skill_video_tag'
    start          REAL,
    end            REAL,
    label          TEXT,    -- 手順タイトル、タグ名など
    tag_id         INTEGER, -- タグのとき skill_tags.id
    parent_id      INTEGER, -- 場面タグ → 手順（segment_annotations.id）
    payload        TEXT,    -- JSON（手順の説明・道具・資材・注意点など）
    source         TEXT,    -- 'llm' / 'human'
    method_version TEXT,
    created        TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
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
    session_id INTEGER,
    word       TEXT,
    category   TEXT,
    status     TEXT,         -- pending / adopted / rejected
    created    TEXT
);
```

`segment_annotations` は技能伝承専用にせず汎用の名前にしておく。対話研究でも「発話をまたぐ区間ラベル」が必要になったときに同じ表を使える。

### 5.2 ディレクトリ構成（案）

```
skill_transfer/              # 新規パッケージ（Colab 非依存・import 可能・単体テスト可能）
  __init__.py
  config.py                  # 環境変数の読み込み（N文字、閾値、保存先、LLM 設定など）
  db.py                      # 上記テーブルの作成・保存・取得（conn を引数で受け取る）
  seed.py                    # seed/skill_transfer_tags.csv の投入
  audio.py                   # 音声取り出し（pydub）＋騒音除去（noisereduce）
  transcribe.py              # ① 文分割・turns 保存（認識関数は引数で注入）
  subtitles.py               # ① WebVTT 生成
  llm.py                     # LLM 呼び出しを 1 か所に集約（差し替え可能）
  procedure.py               # ② 手順分割・スキーマ検証・再試行・区間アノテーション保存
  procedure_pdf.py           # ② HTML テンプレート → PDF（WeasyPrint）
  templates/procedure.html
  tagging.py                 # ③ 抽出・照合（rapidfuzz）・場面タグ・動画タグ・難易度
  pipeline.py                # ①②③ を順に実行、状態の記録と失敗段階からの再実行
  api.py                     # FastAPI の APIRouter（/skill/...）
  static/                    # 画面（スマホ優先、HTML + 素の JavaScript）
seed/skill_transfer_tags.csv
tests/                       # pytest。LLM・音声認識はモック
.env.example
README.md
```

既存の 2 ファイルと同様に Colab で動かすため、`audio_analysis_pipeline.py` の「5. サーバー起動」の直前に **数行だけ追加** し、
既存の `app`・`conn`・`jobs`・`transcribe_full_audio`・`client` を渡して技能伝承のルーターを登録する（既存の処理・API には手を入れない）。

```python
# --- 4.5 技能伝承機能 ---（追加イメージ）
import sys; sys.path.insert(0, SKILL_TRANSFER_REPO_DIR)
from skill_transfer.api import create_router
app.include_router(create_router(conn=conn, jobs=jobs, jobs_lock=jobs_lock,
                                 transcribe_fn=transcribe_full_audio, openai_client=client))
```

既存関数を引数で注入する形にすることで、**既存の Whisper 設定・幻覚フィルタをそのまま使いつつ、テストではモックに差し替えられる**。

### 5.3 流用／拡張／新規の一覧

| 区分 | 内容 |
| --- | --- |
| 流用 | `sessions`・`turns`（記録単位・時間付き書き起こし）、`transcribe_full_audio()`、pydub による音声変換・正規化、ジョブ ID＋ポーリング方式と `jobs`/`/jobs/{id}`、`addin_results`（LLM 生出力の版つき保存）、OpenAI クライアント（Q2 次第）、pydantic、FastAPI の `app`、`transcription_accuracy_tool.py`（技能伝承動画の認識精度評価にそのまま使える） |
| 拡張 | `audio_analysis_pipeline.py` にルーター登録の数行を追加。Q1 で A 案の場合は `sessions` に `domain` 列を追加し、`/corpus`・`/search`・精度ツールの「最新セッション」取得で対話研究分だけを対象にする（Q6 は任意） |
| 新規 | 上記 4 テーブル、騒音除去、文分割、WebVTT、LLM 呼び出しの集約、手順分割＋検証＋再試行、PDF、タグ照合（rapidfuzz）、タグ絞り込み API、動画ファイル配信、画面一式、seed CSV、pytest、`.env.example`、README |

### 5.4 各段階の作業とテスト

| 段階 | 作業 | テスト（すべて pytest、一時 SQLite、LLM・音声認識はモック） |
| --- | --- | --- |
| 1 | 4 テーブル作成（＋Q1 の列追加）、seed CSV と投入処理 | テーブル作成が冪等であること、既存テーブルの列が変わらないこと、CSV の親子・別名が正しく入ること、再投入で重複しないこと |
| 2 | アップロード API（`/skill/upload`）、動画保存、音声取り出し・騒音除去、文分割→`turns` 保存、WebVTT | 文分割と時刻、1 行 N 文字×最大 2 行での区切り、文字数比例の時間配分、VTT の書式、失敗時に `status=error` とエラー内容が残り再実行できること |
| 3 | LLM で手順分割（固定スキーマ JSON・検証・1 回だけ再試行）、セグメント番号→時間の決定、区間アノテーション保存、PDF | 正常系、不正 JSON → 再試行成功／再試行も失敗で error、時間がセグメントから決まること、PDF が生成されること（WeasyPrint が無い環境ではスキップ表示） |
| 4 | 用語抽出、完全一致→別名一致→類似度の順で照合、新タグ候補、場面タグ、難易度（設定で ON/OFF）、動画タグ集約 | 照合の優先順位と閾値、候補が自動登録されないこと、場面タグの時間が手順と一致、動画タグ＝場面タグの和集合 |
| 5 | 画面：アップロード（`capture="environment"`、処理状況表示）、再生（字幕 ON/OFF、場面タグ一覧で頭出し、再生バーに印、PDF ダウンロード）、動画配信（Range 対応） | API のテスト（TestClient）。画面は Playwright（環境にある Chromium）でスマホ幅の表示確認 |
| 6 | 画面：ホーム（作業の種類を階層でたどる）、一覧（タグボタンで AND 絞り込み）、タグ管理（追加・編集・削除・別名・候補の採用） | 下位分類を含む一覧、AND 絞り込み、タグ CRUD・候補採用の API テスト |
| 7 | サンプル動画（ffmpeg で合成した音声付き動画、または用意していただく動画）で通し確認 | ①〜③を通しで実行（ここだけ実モデルを使うかは環境次第。Colab での確認手順を README に記載） |

各段階で「既存の対話研究の機能が変わっていないこと」を確認する回帰テストも走らせる（Q4 の方法による）。

### 5.5 設定（`.env.example` に追記予定）

`OPENAI_API_KEY`、（Q2 次第で）`ANTHROPIC_API_KEY`・`SKILL_LLM_PROVIDER`・`SKILL_LLM_MODEL`、`SKILL_MEDIA_DIR`（既定 `/content/drive/MyDrive/skill_transfer`）、
`SKILL_SUBTITLE_MAX_CHARS=20`、`SKILL_SUBTITLE_MAX_LINES=2`、`SKILL_TAG_FUZZY_THRESHOLD=85`、`SKILL_DIFFICULTY_TAG=false`、`SKILL_WHISPER_LANGUAGE=ja`。
Colab では従来どおりシークレットから `os.environ` に入れる形にし、ローカルでは `.env` を読む。

---

## 6. 判断が必要な点（確認をお願いします）

**Q1. 技能伝承の動画を既存の `sessions`/`turns` に入れるか**

- **A 案（推奨）**：`sessions`/`turns` に入れ、`sessions` に `domain TEXT DEFAULT 'dialogue'` 列を追加（既存行は自動で `'dialogue'`）。
  `/corpus` 一覧・`/search`・精度ツールの「最新セッション」取得に `domain='dialogue'` の条件を足し、**対話研究側の出力は今と同じに保つ**。
  - 利点：データモデルを本当に流用できる。`raw_turns`・正解書き起こし・精度ツールが技能伝承動画にもそのまま使える
  - 影響：既存テーブル 1 つに列追加、既存 API 3 か所の SQL に条件追加（＝CLAUDE.md の「既存の変更」に当たるため確認が必要）
- **B 案**：既存テーブルには一切触れず、技能伝承用に独立した表（動画・セグメント）を作る。既存への影響はゼロだが、`turns` と同じ構造を二重に持つことになり、精度ツールも使えない。

**Q2. LLM はどれを使うか**
既存は OpenAI `gpt-4o`。CLAUDE.md の「既存にあるものが優先」に従うと OpenAI になる。
推奨：呼び出しを `skill_transfer/llm.py` に集約し、**既定は既存と同じ OpenAI**、環境変数 `SKILL_LLM_PROVIDER=anthropic` で Claude に切り替えられるようにする。Claude を既定にしたい場合はお知らせください。

**Q3. 画面（フロントエンド）**
リポジトリ内に画面のコードがありません。既存の画面が別の場所にあれば、その技術（素の HTML か、React 等か）と置き場所を教えてください。
無ければ、ビルド不要の **HTML＋素の JavaScript を FastAPI から配信**（`/skill/` 配下）する案を推奨します（Colab＋ngrok のままスマホから開ける）。

**Q4. コードの置き方とテスト**
既存はセル貼り付け型で import できないため、新機能は `skill_transfer/` パッケージとして書き、Colab では「リポジトリを Drive などに clone → `sys.path` に追加」して読み込む形を推奨します。Colab にコードをどう持ち込んでいるか（セルに貼り付けか、clone か）を教えてください。
テストは pytest を新たに導入します（既存にテストが無いため）。既存 API の回帰確認は、`audio_analysis_pipeline.py` を Colab 専用行と重い依存（torch・whisper・pyannote・colab）をスタブにして読み込むテスト用ローダーで行う想定です。この際、DB の場所を環境変数 `CORPUS_DB_PATH`（既定は現在と同じパス）で上書きできるよう **1 行だけ変更** してよいか確認させてください。

**Q5. 匿名化を技能伝承動画にもかけるか**
既存の `[MASK]` 置換は、道具の商品名・メーカー名・現場名まで伏せる可能性があり、手順書やタグ付けの質が落ちます。推奨は**既定 OFF（設定で ON 可）**。個人名を伏せる必要があれば ON にします。

**Q6.（小）Whisper の言語指定**
既存の `transcribe_full_audio()` は言語自動判定です。騒音の多い現場音声では誤判定のおそれがあるため、**省略可能な引数 `language=None` を追加**（既定は今と同じ動作）して、技能伝承からは `"ja"` を渡したいです。不可なら自動判定のまま使います。

**Q7. タグ一覧の初期データ**
実際に使うタグ（作業の分類・道具・資材など）の一覧があれば `seed/skill_transfer_tags.csv` に使います。無ければ、型枠・鉄筋・内装などのサンプルを私が作ります。

---

## 7. 気づいた点・リスク（今回は既存コードに手を入れない）

- **動画サイズと ngrok**：スマホ動画は数百 MB になりうる。分割アップロードは MVP 外なので、まずは 1 回送信で、大きすぎる場合の上限とエラー表示を設ける。
- **再生互換性**：iPhone の HEVC/MOV は Android の Chrome で再生できないことがある。保存時に ffmpeg で H.264/AAC の MP4（faststart）に変換する案を段階 2 に含める。
- **GPU の取り合い**：既存の対話解析と技能伝承の処理が同時に走ると Whisper を共有するため、技能伝承側で処理を直列化するロックを置く。
- **PDF の日本語フォント**：Colab で WeasyPrint を使うには日本語フォント（`fonts-noto-cjk`）の導入が必要。README に手順を書く。
- **ngrok の警告ページ**：無料枠では最初に「Visit Site」画面が出る（既存の案内と同じ）。画面を同じオリジンから配信すれば、1 回押せば以後は動く。
- **既存コードで目についた点（参考、今回は変更しない）**：`sessions.filename` が一時ファイル名（`input_xxx`）で元動画が残らない／アップロードファイル名をそのままパスに使っている／ジョブ状態が再起動で消える／WER・CER の計算がサーバーと精度ツールで重複。
- **認証**：既存にログイン・権限は無いため、CLAUDE.md に従い新設しない（ngrok URL を知っていれば誰でも見られる点は既存と同じ）。
