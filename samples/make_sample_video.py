"""
技能伝承機能の動作確認用サンプル動画を作る（ローカルで実行するスクリプト）。

「壁の型枠の建て込み」の動画を、日本語の音声合成（pyopenjtalk）と説明のスライドで作る。
スマホで縦に撮った動画と同じ縦長（720x1280）・H.264/AAC の MP4。台本は2種類。

- katawaku : 熟練者が1人で説明する
- dialogue : 上司が後輩に教えている会話（声の高さと速さを変えて2人を区別する）

それぞれ騒音なし（*_clean.mp4）と、重機のうなりと電動工具の音を混ぜたもの（*_noisy.mp4）を作る。

使い方:
    pip install pyopenjtalk-plus pillow numpy
    python samples/make_sample_video.py [出力フォルダ] [katawaku|dialogue|all]
（ffmpeg と日本語フォント（IPAゴシックなど）が必要）
"""
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
import pyopenjtalk
from PIL import Image, ImageDraw, ImageFont

SAMPLE_RATE = 48000
WIDTH, HEIGHT = 720, 1280
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

# 話す人ごとの声（声の高さ[半音]と速さ）と、画面での表示
VOICES = {
    "説明": {"half_tone": 0.0, "speed": 1.0, "label": None, "color": None},
    "上司": {"half_tone": -2.0, "speed": 0.92, "label": "上司", "color": "#e67e00"},
    "後輩": {"half_tone": 4.0, "speed": 1.08, "label": "後輩", "color": "#1e88e5"},
}

# (話す人, スライドの見出し, 読み上げる文)
SCRIPT_KATAWAKU = [("説明", h, t) for h, t in [
    ("はじめに", "それでは、壁の型枠の建て込みを説明します。"),
    ("手順1　墨の確認", "まず、床に打ってある墨を確認します。"),
    ("手順1　墨の確認", "レーザー墨出し器で通りを見ておくと、間違いがありません。"),
    ("手順2　コンパネを立てる", "次に、墨に合わせてコンパネを立てていきます。"),
    ("手順2　コンパネを立てる", "立てたら、倒れないように必ず仮止めをしてください。風が強い日は特に注意が必要です。"),
    ("手順3　セパを通す", "次は、セパレーターを通します。セパの長さは、壁の厚さに合わせて選びます。"),
    ("手順3　セパを通す", "セパの両端にピーコンを付けて、反対側のコンパネも立てます。"),
    ("手順4　締め付け", "続いて、インパクトドライバーで締め付けていきます。"),
    ("手順4　締め付け", "締めるときは、下から順番に締めるのがコツです。上から締めると、型枠がゆがんでしまいます。"),
    ("手順4　締め付け", "締めすぎるとセパが切れることがあるので、力加減に気をつけてください。"),
    ("手順5　垂直の確認", "最後に、レベルで垂直を確認します。ずれていたら、ここで直しておきます。"),
    ("安全", "足場の上で作業するときは、必ずフルハーネスを使ってください。"),
    ("おわり", "以上で、型枠の建て込みは終わりです。"),
]]

SCRIPT_DIALOGUE = [
    ("上司", "はじめに", "よし、今日は壁の型枠の建て込みをやってもらう。まず俺がやるから見ててくれ。"),
    ("後輩", "はじめに", "はい、よろしくお願いします。"),
    ("上司", "手順1　墨の確認", "まず、床の墨を確認する。レーザー墨出し器で通りを見ておけば間違いない。"),
    ("後輩", "手順1　墨の確認", "墨はどこを見ればいいんですか？"),
    ("上司", "手順1　墨の確認", "この線が壁の位置だ。コンパネはこの線に合わせて立てる。"),
    ("上司", "手順2　コンパネを立てる", "次に、コンパネを立てる。立てたらすぐに仮止めしろ。倒れたら危ないからな。"),
    ("後輩", "手順2　コンパネを立てる", "風が強い日はどうしますか？"),
    ("上司", "手順2　コンパネを立てる", "風が強い日は特に注意だ。仮止めを増やして、無理なら作業を止める。"),
    ("上司", "手順3　セパを通す", "次は、セパを通す。セパの長さは壁の厚さで決まる。"),
    ("後輩", "手順3　セパを通す", "ピーコンは両側に付けるんですよね？"),
    ("上司", "手順3　セパを通す", "そうだ。両端にピーコンを付けてから、反対側のコンパネを立てる。"),
    ("上司", "手順4　締め付け", "続いて、インパクトで締めていく。下から順番に締めるのがコツだ。"),
    ("後輩", "手順4　締め付け", "上から締めるとだめなんですか？"),
    ("上司", "手順4　締め付け", "上から締めると型枠がゆがむ。それと、締めすぎるとセパが切れるから、力加減に気をつけろ。"),
    ("後輩", "手順4　締め付け", "わかりました。"),
    ("上司", "手順5　垂直の確認", "最後に、レベルで垂直を確認する。ずれていたら、ここで直しておく。"),
    ("上司", "安全", "足場に上がるときは、必ずフルハーネスを使えよ。"),
    ("後輩", "おわり", "はい。次は自分でやってみます。"),
]

SCRIPTS = {
    "katawaku": ("作業説明（サンプル）型枠の建て込み", SCRIPT_KATAWAKU),
    "dialogue": ("上司から後輩へ（サンプル）型枠の建て込み", SCRIPT_DIALOGUE),
}
GAP_SECONDS = 0.8
COLORS = ["#37474f", "#455a64", "#4e342e", "#3e2723", "#263238"]


def find_font():
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    raise SystemExit("日本語フォントが見つかりません。IPAゴシックなどを入れてください。")


def wrap(draw, text, font, width):
    lines, line = [], ""
    for ch in text:
        if draw.textlength(line + ch, font=font) > width:
            lines.append(line)
            line = ch
        else:
            line += ch
    return lines + ([line] if line else [])


def make_slide(path, banner, speaker, heading, text, index, font_path):
    img = Image.new("RGB", (WIDTH, HEIGHT), COLORS[index % len(COLORS)])
    draw = ImageDraw.Draw(img)
    small = ImageFont.truetype(font_path, 30)
    big = ImageFont.truetype(font_path, 54)
    body = ImageFont.truetype(font_path, 40)
    draw.rectangle([0, 0, WIDTH, 90], fill="#e67e00")
    draw.text((30, 25), banner, font=small, fill="white")
    draw.text((40, 200), heading, font=big, fill="#ffcc80")
    y = 360
    voice = VOICES[speaker]
    if voice["label"]:
        # 話している人を色つきの札で示す
        draw.rounded_rectangle([40, 300, 190, 350], radius=12, fill=voice["color"])
        draw.text((62, 306), voice["label"], font=small, fill="white")
        y = 380
    for line in wrap(draw, text, body, WIDTH - 80):
        draw.text((40, y), line, font=body, fill="white")
        y += 62
    # 型枠らしい簡単な絵（コンパネとセパ）
    top = 900
    draw.rectangle([120, top, 200, top + 300], fill="#c8a165", outline="#8d6e3f", width=4)
    draw.rectangle([520, top, 600, top + 300], fill="#c8a165", outline="#8d6e3f", width=4)
    for k in range(3):
        yy = top + 60 + k * 90
        draw.line([200, yy, 520, yy], fill="#b0bec5", width=6)
        draw.ellipse([186, yy - 10, 214, yy + 10], fill="#90a4ae")
        draw.ellipse([506, yy - 10, 534, yy + 10], fill="#90a4ae")
    draw.text((30, HEIGHT - 60), "※ 音声は合成音声です", font=small, fill="#bbbbbb")
    img.save(path)


def synthesize(script):
    """台本を読み上げ、(音声, 文ごとの長さ[秒]) を返す。話す人ごとに声の高さと速さを変える。"""
    pieces, durations = [], []
    for speaker, _, text in script:
        voice = VOICES[speaker]
        x, sr = pyopenjtalk.tts(text, speed=voice["speed"], half_tone=voice["half_tone"])
        if sr != SAMPLE_RATE:
            x = np.interp(np.arange(0, len(x), sr / SAMPLE_RATE), np.arange(len(x)), x)
        x = x / 32768.0
        gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE))
        pieces += [x, gap]
        durations.append((len(x) + len(gap)) / SAMPLE_RATE)
    lead = np.zeros(int(0.5 * SAMPLE_RATE))
    durations[0] += 0.5
    return np.concatenate([lead] + pieces), durations


def construction_noise(n, rng):
    """重機のエンジンのうなり（低い音）と、断続的に鳴る電動工具の音。"""
    t = np.arange(n) / SAMPLE_RATE
    rumble = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) / k for k, f in enumerate([48, 96, 144, 192], 1))
    tool = rng.normal(0, 1, n) * (np.sin(2 * np.pi * 0.25 * t) > 0.6)       # 時々鳴る工具
    tool *= 0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 22 * t))                 # インパクトの打撃のような断続
    hiss = rng.normal(0, 1, n)
    noise = 0.6 * rumble + 0.2 * hiss + 0.9 * tool
    return noise / np.sqrt(np.mean(noise ** 2))


def write_wav(path, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


def encode(slides_list, wav_path, out_path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", slides_list,
                    "-i", wav_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", out_path], check=True)


def make_video(out_dir, name, font_path):
    banner, script = SCRIPTS[name]
    speech, durations = synthesize(script)
    with tempfile.TemporaryDirectory() as tmp:
        lines = []
        for i, ((speaker, heading, text), duration) in enumerate(zip(script, durations)):
            slide = os.path.join(tmp, f"slide_{i:02d}.png")
            make_slide(slide, banner, speaker, heading, text, i, font_path)
            lines += [f"file '{slide}'", f"duration {duration:.3f}"]
        lines.append(f"file '{slide}'")  # concat では最後の画像をもう一度書く決まり
        slides_list = os.path.join(tmp, "slides.txt")
        with open(slides_list, "w") as f:
            f.write("\n".join(lines) + "\n")

        clean_wav = os.path.join(tmp, "clean.wav")
        write_wav(clean_wav, speech * 0.9)
        encode(slides_list, clean_wav, os.path.join(out_dir, f"sample_{name}_clean.mp4"))

        rng = np.random.default_rng(0)
        speech_rms = np.sqrt(np.mean(speech[np.abs(speech) > 0.01] ** 2))
        noise = construction_noise(len(speech), rng) * speech_rms / (10 ** (10 / 20))  # SN比 約10dB
        noisy = speech + noise
        noisy_wav = os.path.join(tmp, "noisy.wav")
        write_wav(noisy_wav, noisy / np.max(np.abs(noisy)) * 0.9)
        encode(slides_list, noisy_wav, os.path.join(out_dir, f"sample_{name}_noisy.mp4"))
    print(f"作成しました: {out_dir}/sample_{name}_clean.mp4, sample_{name}_noisy.mp4（約{sum(durations):.0f}秒）")


def main(out_dir, which="all"):
    os.makedirs(out_dir, exist_ok=True)
    font_path = find_font()
    for name in (SCRIPTS if which == "all" else [which]):
        make_video(out_dir, name, font_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples", sys.argv[2] if len(sys.argv) > 2 else "all")
