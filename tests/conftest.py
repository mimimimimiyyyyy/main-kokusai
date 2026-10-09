"""
Colabのセルとして書かれた skill_transfer_cell.py を pytest から読み込むための仕組み。

セルは `!pip install` のようなColab専用の行や、Googleドライブのマウント・GPUでのモデル読み込み・
ngrok・トップレベルの await を含むため、普通には import できない。ここでは
- 関数のテスト用: `!` で始まる行を取り除き、「13. サーバー起動」より前だけを読み込む
- セル全体のテスト用: Colab専用の部分（ドライブ・GPU・Whisperのモデル・ngrok）だけを差し替えて、
  セルを最後（await skill_server.serve()）まで丸ごと実行する
"""
import ast
import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CELL = REPO_ROOT / "audio_analysis_pipeline.py"
ACCURACY_CELL = REPO_ROOT / "transcription_accuracy_tool.py"
SKILL_CELL = REPO_ROOT / "skill_transfer_cell.py"
SEED_CSV = REPO_ROOT / "seed" / "skill_transfer_tags.csv"
SKILL_SERVER_MARKER = "# --- 13. サーバー起動"


def cell_source(path, stop_marker=None):
    src = Path(path).read_text(encoding="utf-8")
    if stop_marker is not None:
        assert stop_marker in src, f"{stop_marker!r} が {path} に見つかりません"
        src = src.split(stop_marker)[0]
    lines = ["" if line.lstrip().startswith("!") else line for line in src.splitlines()]
    return "\n".join(lines) + "\n"


def load_skill_cell(env=None, namespace=None):
    """サーバー起動より前（関数と設定の定義）だけを読み込む。"""
    with mock.patch.dict(os.environ, env or {}):
        module = types.ModuleType("skill_transfer_cell")
        if namespace:
            module.__dict__.update(namespace)
        exec(compile(cell_source(SKILL_CELL, SKILL_SERVER_MARKER), "skill_transfer_cell", "exec"), module.__dict__)
        return module


class ColabStubs:
    """
    セルを丸ごと実行するときに差し替える、Colab・GPU・ngrok まわりのもの。
    whisper_model / openai_client を渡すと、セルが読み込むモデル・クライアントの代わりに使われる。
    serve=False なら uvicorn のサーバーは待ち受けず、起動しようとしたappを記録するだけ。
    """

    def __init__(self, whisper_model, openai_client, public_url="https://example.ngrok-free.dev"):
        self.whisper_model = whisper_model
        self.openai_client = openai_client
        self.public_url = public_url
        self.mounted = []
        self.loaded_models = []
        self.ngrok_commands = []

    def modules(self):
        stubs = self
        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(mount=lambda path: stubs.mounted.append(path))
        colab.userdata = types.SimpleNamespace(get=lambda name: None)
        google.colab = colab

        torch = types.ModuleType("torch")
        torch.device = lambda name: name
        torch.cuda = types.SimpleNamespace(is_available=lambda: True)

        whisper = types.ModuleType("whisper")

        def load_model(name, device=None):
            stubs.loaded_models.append((name, device))
            return stubs.whisper_model
        whisper.load_model = load_model

        openai = types.ModuleType("openai")
        openai.OpenAI = lambda api_key=None: stubs.openai_client

        requests = types.ModuleType("requests")
        tunnels = {"tunnels": [{"public_url": self.public_url}]}
        requests.get = lambda url: types.SimpleNamespace(json=lambda: tunnels)

        nest_asyncio = types.ModuleType("nest_asyncio")
        nest_asyncio.apply = lambda: None
        return {"google": google, "google.colab": colab, "torch": torch, "whisper": whisper,
                "openai": openai, "requests": requests, "nest_asyncio": nest_asyncio}

    def popen(self, real_popen):
        def fake(args, *a, **kw):
            if isinstance(args, (list, tuple)) and args and args[0] == "ngrok":
                self.ngrok_commands.append(list(args))
                return mock.MagicMock(name="ngrok_process")
            return real_popen(args, *a, **kw)
        return fake


def run_whole_cell(stubs, env, serve=None, namespace=None):
    """
    セル全体（!の行を除く）を、Colabと同じくトップレベルの await を許して実行する。
    serve: uvicorn.Server.serve の代わりに呼ぶ async 関数（省略すると本物のサーバーが待ち受ける）。
    namespace: セルの変数の置き場にする辞書（別スレッドで動かすとき、外から skill_server を止めるために渡す）
    戻り値: セルの変数の置き場（skill_app などが入っている）
    """
    import uvicorn
    source = cell_source(SKILL_CELL)
    code = compile(source, "skill_transfer_cell", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    namespace = {} if namespace is None else namespace
    namespace["__name__"] = "__main__"
    # noisereduce は torch があると torch 版の処理も読み込む。スタブの torch を本物と取り違えないよう、
    # スタブを入れる前に（このテスト環境の torch なしの状態で）読み込んでおく。Colabでは本物の torch を使う。
    import noisereduce  # noqa: F401
    saved = {name: sys.modules.get(name) for name in stubs.modules()}
    sys.modules.update(stubs.modules())
    patches = [mock.patch.dict(os.environ, env), mock.patch("subprocess.Popen", stubs.popen(subprocess.Popen)),
               mock.patch("time.sleep", lambda s: None)]
    if serve is not None:
        patches.append(mock.patch.object(uvicorn.Server, "serve", serve))
    try:
        for p in patches:
            p.start()
        coroutine = eval(code, namespace)
        if coroutine is not None:
            asyncio.run(coroutine)
    finally:
        for p in reversed(patches):
            p.stop()
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return namespace


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
