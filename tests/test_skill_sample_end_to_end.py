"""
実装の順番 7: サンプル動画で一通り動作確認する通しテスト。

- 日本語の音声合成（pyopenjtalk）で熟練者の説明を作り、重機・工具のような騒音を混ぜ、
  iPhoneの動画と同じ HEVC の .mov にする
- 技能伝承のセルを丸ごと実行し（Colab専用のドライブ・GPU・ngrokだけ差し替え）、セル自身が
  起動したサーバーに対して操作する
- スマホの画面幅のブラウザで、投稿 → 自動処理 → 再生（字幕・場面ジャンプ）→ 手順書PDF →
  作業の種類からの一覧 → タグ絞り込み までを操作する
- 音声の取り出し・騒音除去・動画の変換・PDF作成は本物を使う。音声認識（Whisper）とLLMはモック
  （モデルの取得・APIの呼び出しはテストで行わない）。LLMはOpenAIクライアントの形で差し替える

pyopenjtalk / Playwright / libx265 が無い環境ではスキップする。
"""
import json
import os
import subprocess
import threading
import time
import urllib.request
import wave

import numpy as np
import pytest
from conftest import ColabStubs, FakeWhisperModel, run_whole_cell, whisper_segment

pyopenjtalk = pytest.importorskip("pyopenjtalk")
playwright_api = pytest.importorskip("playwright.sync_api")
from test_skill_screens import PHONE, free_port  # noqa: E402

SCRIPT = [
    "それでは型枠の建て込みを説明します。",
    "まず、墨に合わせてコンパネを立てていきます。",
    "倒れないように、必ず仮止めをしてください。",
    "次は、セパを通してPコンで留めます。",
    "インパクトで締めるときは、下から順に締めるのがコツです。",
    "最後に、レベルで垂直を確認して終わりです。",
]
SAMPLE_RATE = 16000


def synthesize_speech():
    """台本を読み上げ、文ごとの開始・終了時間とともに16kHzの音声を返す。"""
    pieces, times, t = [], [], 0.5
    pieces.append(np.zeros(int(0.5 * SAMPLE_RATE)))
    for line in SCRIPT:
        x, sr = pyopenjtalk.tts(line)
        x = np.interp(np.arange(0, len(x), sr / SAMPLE_RATE), np.arange(len(x)), x) / 32768.0
        pieces.append(x)
        times.append((round(t, 2), round(t + len(x) / SAMPLE_RATE, 2)))
        t += len(x) / SAMPLE_RATE
        gap = np.zeros(int(0.8 * SAMPLE_RATE))
        pieces.append(gap)
        t += len(gap) / SAMPLE_RATE
    return np.concatenate(pieces), times


def construction_noise(n, rng):
    """エンジンのうなり（低音）と、断続的な電動工具の音（高めの帯域の雑音）。"""
    t = np.arange(n) / SAMPLE_RATE
    rumble = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) / k for k, f in enumerate([55, 110, 165, 220], 1))
    hiss = rng.normal(0, 1, n)
    tool = np.convolve(rng.normal(0, 1, n), np.ones(3) / 3, mode="same") * (np.sin(2 * np.pi * 0.4 * t) > 0.3)
    noise = 0.5 * rumble + 0.3 * hiss + 0.8 * tool
    return noise / np.sqrt(np.mean(noise ** 2))


def make_sample_video(folder):
    rng = np.random.default_rng(0)
    speech, times = synthesize_speech()
    speech_rms = np.sqrt(np.mean(speech[np.abs(speech) > 0.01] ** 2))
    noise = construction_noise(len(speech), rng) * speech_rms / (10 ** (5 / 20))  # SN比 約5dB
    mixed = np.clip(speech + noise, -1, 1)
    wav_path = os.path.join(folder, "mixed.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((mixed * 32767).astype(np.int16).tobytes())
    duration = len(mixed) / SAMPLE_RATE
    video_path = os.path.join(folder, "IMG_0420.MOV")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=720x1280:rate=30:duration={duration:.2f}",
        "-i", wav_path, "-c:v", "libx265", "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", video_path], check=True)
    return video_path, times, mixed, noise


def gap_rms_db(samples, times):
    """文と文のあいだ（話していない区間）の音の大きさ。騒音がどれだけ残っているかの目安。"""
    gaps = [(e + 0.2, s - 0.2) for (_, e), (s, _) in zip(times, times[1:])]
    parts = [samples[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)] for a, b in gaps]
    return 20 * np.log10(np.sqrt(np.mean(np.concatenate(parts) ** 2)) + 1e-9)


class ScriptedWhisper(FakeWhisperModel):
    """台本の文と時間を、Whisperの認識結果（単語のタイムスタンプ付き）の形で返す。"""

    def __init__(self, times):
        segments = []
        for (start, end), text in zip(times, SCRIPT):
            cut = len(text) // 2
            mid = start + (end - start) * cut / len(text)
            segments.append(whisper_segment(start, end, [(text[:cut], start, mid), (text[cut:], mid, end)]))
        super().__init__(segments)


class FakeOpenAIClient:
    """openai.OpenAI と同じ形。プロンプトの用途に応じた応答を返す。"""

    PROCEDURE = {"steps": [
        {"title": "コンパネを立てる", "description": "墨に合わせてコンパネを立て、仮止めする。",
         "start_segment": 1, "end_segment": 2, "tools": [], "materials": ["コンパネ"],
         "cautions": ["倒れないように必ず仮止めする"], "tips": []},
        {"title": "セパを通して締める", "description": "セパを通してPコンで留め、インパクトで締める。",
         "start_segment": 3, "end_segment": 4, "tools": ["インパクト"], "materials": ["セパ", "Pコン"],
         "cautions": [], "tips": ["下から順に締める"]},
        {"title": "垂直を確認する", "description": "レベルで垂直を確認する。",
         "start_segment": 5, "end_segment": 5, "tools": ["レベル"], "materials": [], "cautions": [], "tips": []},
    ]}
    TAG_TERMS = {"steps": [
        {"step_index": 0, "terms": [{"word": "型枠の建て込み", "category": "作業"}, {"word": "墨", "category": "作業"}]},
        {"step_index": 1, "terms": [{"word": "セパ", "category": "資材"}, {"word": "Pコン", "category": "資材"}]},
        {"step_index": 2, "terms": [{"word": "レベル", "category": "道具"}]},
    ]}

    def __init__(self):
        self.calls = []
        client = self

        class Completions:
            def create(self, **kwargs):
                client.calls.append(kwargs)
                prompt = kwargs["messages"][0]["content"]
                data = client.PROCEDURE if prompt.startswith("# task: procedure") else client.TAG_TERMS
                message = type("Message", (), {"content": json.dumps(data, ensure_ascii=False)})
                return type("Response", (), {"choices": [type("Choice", (), {"message": message})]})

        self.chat = type("Chat", (), {"completions": Completions()})


def get_json(base, path):
    with urllib.request.urlopen(base + path) as res:
        return json.loads(res.read())


def test_sample_video_end_to_end_with_the_cell_alone(tmp_path):
    shots = os.environ.get("SKILL_SCREENSHOT_DIR")
    video_path, times, mixed, noise = make_sample_video(str(tmp_path))

    # --- 技能伝承のセルを丸ごと実行する（セル自身がサーバーを起動する） ---
    stubs = ColabStubs(whisper_model=ScriptedWhisper(times), openai_client=FakeOpenAIClient())
    port = free_port()
    env = {"SKILL_MEDIA_DIR": str(tmp_path / "skill_transfer"), "SKILL_PORT": str(port), "NGROK_AUTH_TOKEN": "t"}
    cell = {}
    thread = threading.Thread(target=run_whole_cell, args=(stubs, env), kwargs={"namespace": cell}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 60
        while not (cell.get("skill_server") and cell["skill_server"].started):
            assert time.monotonic() < deadline and thread.is_alive(), "セルのサーバーが起動しませんでした"
            time.sleep(0.05)
        base = f"http://127.0.0.1:{port}"
        assert stubs.mounted == ["/content/drive"] and stubs.ngrok_commands == [["ngrok", "http", str(port)]]
        media_dir = env["SKILL_MEDIA_DIR"]

        with playwright_api.sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_context(**PHONE).new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            # 投稿 → 自動処理
            page.goto(f"{base}/skill#/upload")
            page.set_input_files("#camera-input", video_path)
            page.fill("#title", "型枠の建て込み")
            page.fill("#explainer", "山田")
            page.click("#upload-button")
            page.wait_for_selector("text=動画を見る", timeout=120000)
            if shots:
                page.screenshot(path=os.path.join(shots, "e2e_01_done.png"), full_page=True)

            video = get_json(base, "/skill/api/videos/1")
            assert video["status"] == "done", video["error"]

            # ① 文字起こし・字幕: 文ごとの時間、日本語指定と用語ヒント、騒音除去
            assert [s["text"] for s in video["segments"]] == SCRIPT
            assert [(s["start"], s["end"]) for s in video["segments"]] == times
            call = stubs.whisper_model.calls[0]
            assert call["language"] == "ja" and call["carry_initial_prompt"] is True
            assert "セパレーター" in call["initial_prompt"]
            with wave.open(call["audio"]) as w:
                denoised = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
                assert w.getframerate() == 16000 and w.getnchannels() == 1
            # 話していない区間の騒音が、元の音声より小さくなっている（正規化の分を差し引いて比べる）
            gain = np.max(np.abs(denoised)) / np.max(np.abs(mixed))
            reduction = gap_rms_db(mixed * gain, times) - gap_rms_db(denoised, times)
            print(f"\n騒音除去: 話していない区間の音が {reduction:.1f} dB 小さくなった")
            assert reduction > 6
            vtt = urllib.request.urlopen(f"{base}/skill/api/videos/1/subtitles.vtt").read().decode()
            assert all(len(line) <= 20 for line in vtt.split("\n") if line and "-->" not in line and not line.isdigit())

            # 再生用にH.264へ変換されている（iPhoneのHEVCのままだと再生できない端末がある）
            media = os.path.join(media_dir, "videos", "1", "video.mp4")
            codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=codec_name", "-of", "csv=p=0", media], capture_output=True, text=True).stdout
            assert codec.strip() == "h264"

            # ② 手順書: 手順の時間は文字起こしから、PDFに写真
            assert [s["title"] for s in video["steps"]] == ["コンパネを立てる", "セパを通して締める", "垂直を確認する"]
            assert video["steps"][0]["start"] == times[1][0] and video["steps"][0]["end"] == times[2][1]
            pdf = urllib.request.urlopen(f"{base}/skill/api/videos/1/procedure.pdf").read()
            assert pdf.startswith(b"%PDF") and pdf.count(b"/Subtype /Image") >= 3
            if shots:
                open(os.path.join(shots, "e2e_procedure.pdf"), "wb").write(pdf)

            # ③ タグ: 別名（建て込み・セパ・インパクト）で一覧のタグに、Pコンは新タグ候補に
            assert {t["name"] for t in video["video_tags"]} == {
                "型枠組立", "コンパネ", "セパレーター", "インパクトドライバー", "レベル"}
            scene = {(t["name"], t["start"]) for t in video["scene_tags"]}
            assert ("セパレーター", times[3][0]) in scene and ("レベル", times[5][0]) in scene
            assert [c["word"] for c in get_json(base, "/skill/api/tag_candidates")["candidates"]] == ["墨", "Pコン"]
            assert len(stubs.openai_client.calls) == 2  # 手順書とタグ抽出の2回だけLLMを呼ぶ

            # ④ 閲覧: 作業の種類をたどって一覧 → タグで絞り込み → 場面から再生
            page.goto(f"{base}/skill#/")
            page.click(".work-grid a >> text=躯体工事")
            page.click(".work-grid a >> text=型枠工事")
            page.click(".work-grid a >> text=型枠組立")
            page.wait_for_selector("#video-list .video-card")
            page.click(".tag-filter >> text=セパレーター")
            page.wait_for_selector(".tag-filter.selected")
            assert page.locator("#video-list .video-card .title").all_inner_texts() == ["型枠の建て込み"]
            page.click("#video-list .video-card")
            page.wait_for_selector(".step-card")
            page.wait_for_function("document.getElementById('player').textTracks[0].cues"
                                   " && document.getElementById('player').textTracks[0].cues.length > 0")
            page.click(".step-card >> nth=2 >> .step-head")
            assert page.evaluate("document.getElementById('player').currentTime") == pytest.approx(times[5][0], abs=0.05)
            if shots:
                page.screenshot(path=os.path.join(shots, "e2e_02_player.png"), full_page=True)
            assert errors == []
            browser.close()

        # データはドライブ上の技能伝承専用のDBに入っている
        assert os.path.exists(os.path.join(media_dir, "skill_transfer.db"))
    finally:
        if cell.get("skill_server"):
            cell["skill_server"].should_exit = True
        thread.join(timeout=10)
