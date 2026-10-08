"""
Colabのセルとして書かれた .py を pytest から読み込むための仕組み。

既存の audio_analysis_pipeline.py も skill_transfer_cell.py も、`!pip install` のような
Colab専用の行を含むため普通には import できない。ここでは
- `!` で始まる行を取り除き、
- 既存セルについては「5. サーバー起動」以降（ngrok起動・await server.serve()）を除き、
  GPUモデルやColab専用のライブラリをスタブに差し替え、DBを一時ファイルに向けて
読み込む。テストのために既存コードを書き換えないための仕組み。
"""
import os
import sqlite3
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CELL = REPO_ROOT / "audio_analysis_pipeline.py"
SKILL_CELL = REPO_ROOT / "skill_transfer_cell.py"
SEED_CSV = REPO_ROOT / "seed" / "skill_transfer_tags.csv"
SKILL_HTML = REPO_ROOT / "skill_transfer.html"

PIPELINE_DB_PATH = "/content/drive/MyDrive/corpus.db"
PIPELINE_SERVER_MARKER = "# --- 5. サーバー起動 ---"


def cell_source(path, stop_marker=None, replacements=None):
    src = Path(path).read_text(encoding="utf-8")
    if stop_marker is not None:
        src = src.split(stop_marker)[0]
    for old, new in (replacements or {}).items():
        assert old in src, f"{old!r} が {path} に見つかりません"
        src = src.replace(old, new)
    lines = ["" if line.lstrip().startswith("!") else line for line in src.splitlines()]
    return "\n".join(lines) + "\n"


def exec_cell(source, name, namespace=None, module=None):
    """module を渡すと、そのモジュールの変数の置き場で実行する（Colabでセル同士が変数を共有するのと同じ）。"""
    if module is None:
        module = types.ModuleType(name)
    if namespace:
        module.__dict__.update(namespace)
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def load_skill_cell(env=None, namespace=None):
    with mock.patch.dict(os.environ, env or {}):
        return exec_cell(cell_source(SKILL_CELL), "skill_transfer_cell", namespace)


def _pipeline_stub_modules():
    """既存セルが読み込むGPUモデル・Colab専用ライブラリの代わり。"""
    torch = types.ModuleType("torch")
    torch.device = lambda name: name
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.tensor = lambda data: mock.MagicMock(name="tensor")
    serialization = types.ModuleType("torch.serialization")
    serialization.add_safe_globals = lambda classes: None
    torch_version = types.ModuleType("torch.torch_version")
    torch_version.TorchVersion = type("TorchVersion", (), {})
    torch.serialization = serialization
    torch.torch_version = torch_version

    diarization = mock.MagicMock(name="diarization_pipeline")
    diarization.parameters.return_value = {}
    pyannote = types.ModuleType("pyannote")
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = types.SimpleNamespace(from_pretrained=lambda name: diarization)
    pyannote_core = types.ModuleType("pyannote.audio.core")
    pyannote_task = types.ModuleType("pyannote.audio.core.task")
    for cls in ("Specifications", "Problem", "Resolution"):
        setattr(pyannote_task, cls, type(cls, (), {}))

    whisper = types.ModuleType("whisper")
    whisper.load_model = lambda name, device=None: mock.MagicMock(name="whisper_model")

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda name: "dummy-" + name)
    google.colab = colab

    openai = types.ModuleType("openai")
    openai.OpenAI = lambda api_key=None: mock.MagicMock(name="openai_client")

    nest_asyncio = types.ModuleType("nest_asyncio")
    nest_asyncio.apply = lambda: None
    pyngrok = types.ModuleType("pyngrok")
    pyngrok.ngrok = types.ModuleType("pyngrok.ngrok")

    return {
        "torch": torch, "torch.serialization": serialization, "torch.torch_version": torch_version,
        "pyannote": pyannote, "pyannote.audio": pyannote_audio,
        "pyannote.audio.core": pyannote_core, "pyannote.audio.core.task": pyannote_task,
        "whisper": whisper, "google": google, "google.colab": colab, "openai": openai,
        "japanize_matplotlib": types.ModuleType("japanize_matplotlib"),
        "nest_asyncio": nest_asyncio, "pyngrok": pyngrok, "pyngrok.ngrok": pyngrok.ngrok,
    }


def load_pipeline_cell(db_path, namespace=None, module=None):
    """
    既存セルを、サーバー起動の手前まで読み込む。namespaceに関数を入れておくと、
    Colabで先に別セルを実行した状態（例: register_skill_transfer が定義済み）を再現できる。
    """
    source = cell_source(
        PIPELINE_CELL,
        stop_marker=PIPELINE_SERVER_MARKER,
        replacements={PIPELINE_DB_PATH: str(db_path)},
    )
    # sys.modules全体を元に戻すと、読み込み中にimportされた本物のnumpy等まで消えて
    # 二重読み込みエラーになるため、差し替えたスタブのキーだけを戻す。
    stubs = _pipeline_stub_modules()
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        return exec_cell(source, "audio_analysis_pipeline", namespace, module)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def skill():
    return load_skill_cell()


@pytest.fixture
def conn(skill):
    c = sqlite3.connect(":memory:", check_same_thread=False)
    skill.init_skill_db(c)
    yield c
    c.close()


@pytest.fixture
def seeded_conn(skill, conn):
    skill.seed_skill_tags(conn, str(SEED_CSV))
    return conn


def load_cells_in_colab_order(db_path):
    """Colabと同じく「技能伝承セル → 既存セル」の順に、同じ変数の置き場で実行する。"""
    shared = load_skill_cell()
    load_pipeline_cell(db_path, module=shared)
    return shared


def make_sample_video(path, seconds=3, audio=True):
    """テスト用の短い動画（カラーバー＋440Hzの音）を ffmpeg で作る。"""
    args = ["ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=15:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}",
                 "-c:a", "aac"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(args, check=True)
    return path


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory):
    return make_sample_video(tmp_path_factory.mktemp("media") / "sample.mp4")


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory):
    return make_sample_video(tmp_path_factory.mktemp("media") / "no_audio.mp4", audio=False)


class FakeWhisperModel:
    """
    openai-whisper のモデルの代わり。transcribe の引数を記録し、決まったセグメントを返す。
    引数の形は carry_initial_prompt に対応した新しい版の whisper.transcribe と同じ。
    """

    def __init__(self, segments=None, error=None):
        self.segments = segments if segments is not None else default_whisper_segments()
        self.error = error
        self.calls = []

    def transcribe(self, audio, *, temperature=(0.0,), word_timestamps=False,
                   condition_on_previous_text=True, initial_prompt=None,
                   carry_initial_prompt=False, **decode_options):
        self.calls.append({
            "audio": audio, "temperature": temperature, "word_timestamps": word_timestamps,
            "condition_on_previous_text": condition_on_previous_text, "initial_prompt": initial_prompt,
            "carry_initial_prompt": carry_initial_prompt, **decode_options,
        })
        if self.error:
            raise self.error
        return {"segments": self.segments}


class OldFakeWhisperModel(FakeWhisperModel):
    """carry_initial_prompt が無い古い版の whisper の代わり。"""

    def transcribe(self, audio, *, temperature=(0.0,), word_timestamps=False,
                   condition_on_previous_text=True, initial_prompt=None, **decode_options):
        return super().transcribe(audio, temperature=temperature, word_timestamps=word_timestamps,
                                  condition_on_previous_text=condition_on_previous_text,
                                  initial_prompt=initial_prompt, **decode_options)


def whisper_segment(start, end, words, **extra):
    """words: [(単語, 開始, 終了), ...] からWhisperのセグメントを作る。"""
    return {
        "start": start, "end": end, "text": "".join(w for w, _, _ in words),
        "words": [{"word": w, "start": s, "end": e} for w, s, e in words],
        "no_speech_prob": 0.01, "avg_logprob": -0.2, "compression_ratio": 1.2, **extra,
    }


def default_whisper_segments():
    return [
        whisper_segment(0.0, 2.0, [("まず", 0.0, 0.4), ("型枠を", 0.4, 1.0), ("立てます。", 1.0, 2.0)]),
        whisper_segment(2.0, 3.0, [("次は", 2.0, 2.4), ("セパを", 2.4, 2.7), ("入れます。", 2.7, 3.0)]),
    ]


DEFAULT_PROCEDURE = {
    "steps": [
        {"title": "型枠を立てる", "description": "墨に合わせて型枠を立てる。", "start_segment": 0, "end_segment": 0,
         "tools": ["インパクトドライバー"], "materials": ["コンパネ"], "cautions": ["倒れないように仮止めする"],
         "tips": ["下から順に締めるのがコツ"]},
        {"title": "セパを入れる", "description": "セパレーターを入れて間隔を保つ。", "start_segment": 1, "end_segment": 1,
         "tools": [], "materials": ["セパ"], "cautions": [], "tips": []},
    ]
}


DEFAULT_TAG_TERMS = {
    "steps": [
        {"step_index": 0, "terms": [{"word": "型枠", "category": "作業"}, {"word": "建て込み", "category": "作業"}]},
        {"step_index": 1, "terms": [{"word": "セパ", "category": "資材"}, {"word": "Pコン", "category": "資材"}]},
    ]
}


class FakeLLM:
    """
    LLM（OpenAI / Claude）の代わり。プロンプト先頭の「# task: 〜」で用途を見分け、
    用意した応答を順番に返す。応答は dict（JSONにして返す）か文字列（そのまま返す）。
    """

    def __init__(self, **responses):
        self.responses = {"procedure": [DEFAULT_PROCEDURE], "tag_terms": [DEFAULT_TAG_TERMS], **responses}
        self.prompts = []
        self.method_version = "fake:llm"

    def __call__(self, prompt):
        import json
        self.prompts.append(prompt)
        task = prompt.split("\n", 1)[0].replace("# task:", "").strip()
        queue = self.responses[task]
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)

    def prompts_for(self, task):
        return [p for p in self.prompts if p.startswith(f"# task: {task}")]
