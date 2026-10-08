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


def exec_cell(source, name, namespace=None):
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


def load_pipeline_cell(db_path, namespace=None):
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
        return exec_cell(source, "audio_analysis_pipeline", namespace)
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
