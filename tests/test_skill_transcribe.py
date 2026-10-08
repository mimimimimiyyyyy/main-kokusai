"""
① 文字起こし・字幕作成のテスト。音声認識（Whisper）はモックに差し替える。
音声の取り出し・騒音除去・動画の変換は、ffmpegで作った短い動画で実際に動かす。
"""
import re
import wave

import pytest

from conftest import FakeWhisperModel, OldFakeWhisperModel, whisper_segment

VTT_TIME = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> (\d{2}):(\d{2}):(\d{2})\.(\d{3})$")


def parse_vtt(text):
    """WebVTTを [(開始秒, 終了秒, [行...]), ...] に戻す（テストでの検証用）。"""
    blocks = text.strip().split("\n\n")
    assert blocks[0] == "WEBVTT"
    cues = []
    for block in blocks[1:]:
        lines = block.split("\n")
        m = VTT_TIME.match(lines[1])
        assert m, lines[1]
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        cues.append((start, end, lines[2:]))
    return cues


# --- 語彙ヒント ---
def test_vocabulary_prompt_uses_names_and_aliases_but_not_difficulty(skill, seeded_conn):
    prompt = skill.build_vocabulary_prompt(skill.list_skill_tags(seeded_conn), max_chars=1000)
    assert "型枠組立" in prompt and "セパ" in prompt
    assert "上級" not in prompt


def test_vocabulary_prompt_is_cut_at_max_chars(skill, seeded_conn):
    prompt = skill.build_vocabulary_prompt(skill.list_skill_tags(seeded_conn), max_chars=40)
    assert len(prompt) <= 41


def test_vocabulary_prompt_mixes_categories_and_puts_names_before_aliases(skill):
    tags = [
        {"name": "型枠組立", "category": "作業", "aliases": ["建て込み"]},
        {"name": "配筋", "category": "作業", "aliases": []},
        {"name": "ハッカー", "category": "道具", "aliases": ["結束ハッカー"]},
        {"name": "セパレーター", "category": "資材", "aliases": ["セパ"]},
    ]
    prompt = skill.build_vocabulary_prompt(tags, max_chars=1000)
    assert prompt == "建設現場の作業説明。用語: ハッカー、セパレーター、型枠組立、配筋、結束ハッカー、セパ、建て込み。"


# --- 音声認識の呼び出し ---
def test_transcribe_uses_same_settings_as_existing_cell_plus_language_and_prompt(skill):
    model = FakeWhisperModel()
    skill.transcribe_skill_audio("a.wav", model, language="ja", initial_prompt="用語: セパ")
    call = model.calls[0]
    assert call["temperature"] == 0.0
    assert call["word_timestamps"] is True
    assert call["condition_on_previous_text"] is False
    assert call["language"] == "ja"
    assert call["initial_prompt"] == "用語: セパ"
    assert call["carry_initial_prompt"] is True


def test_transcribe_works_with_whisper_without_carry_initial_prompt(skill):
    model = OldFakeWhisperModel()
    skill.transcribe_skill_audio("a.wav", model, initial_prompt="用語: セパ")
    assert "carry_initial_prompt" not in model.calls[0] or model.calls[0]["carry_initial_prompt"] is False


def test_transcribe_filters_hallucinations_like_existing_cell(skill):
    ok = whisper_segment(0, 1, [("まず", 0, 1)])
    segments = [
        ok,
        whisper_segment(1, 2, [("ご視聴ありがとうございました", 1, 2)]),
        whisper_segment(2, 3, [("Thanks for watching!", 2, 3)]),
        whisper_segment(3, 4, [("ああ", 3, 4)], no_speech_prob=0.9, avg_logprob=-1.5),
        whisper_segment(4, 5, [("ん" * 50, 4, 5)], compression_ratio=3.0),
        whisper_segment(5, 6, [("  ", 5, 6)]),
    ]
    kept = skill.transcribe_skill_audio("a.wav", FakeWhisperModel(segments))
    assert [k["text"] for k in kept] == ["まず"]


# --- 文分割 ---
def test_split_sentences_at_sentence_end_inside_a_segment(skill):
    seg = whisper_segment(0, 4, [("まず型枠を", 0, 1), ("立てます。", 1, 2), ("次は", 2, 3), ("セパです", 3, 4)])
    assert skill.split_sentences([seg]) == [
        {"start": 0, "end": 2, "text": "まず型枠を立てます。"},
        {"start": 2, "end": 4, "text": "次はセパです"},
    ]


def test_split_sentences_uses_segment_boundary_when_no_punctuation(skill):
    segs = [whisper_segment(0, 1, [("まず", 0, 1)]), whisper_segment(1, 2, [("次", 1, 2)])]
    assert [s["text"] for s in skill.split_sentences(segs)] == ["まず", "次"]


# --- 字幕 ---
def test_wrap_keeps_lines_within_max_chars_and_loses_no_text(skill):
    text = "型枠の建て込みでは、セパレーターの位置を墨に合わせてから、インパクトでしっかり締め付けます。"
    lines = skill.wrap_subtitle_text(text, 20)
    assert all(len(l) <= 20 for l in lines)
    assert "".join(lines) == text


def test_wrap_prefers_breaking_after_punctuation(skill):
    lines = skill.wrap_subtitle_text("まず最初に墨出しを行い、次に型枠を立てていきます", 20)
    assert lines[0] == "まず最初に墨出しを行い、"


def test_cues_have_at_most_max_lines_and_cover_the_sentence_time(skill):
    seg = {"start": 10.0, "end": 20.0, "text": "あ" * 100}
    cues = skill.build_subtitle_cues([seg], max_chars=20, max_lines=2)
    assert len(cues) == 3  # 100文字 → 5行 → 2行・2行・1行
    assert all(len(c["lines"]) <= 2 for c in cues)
    assert cues[0]["start"] == 10.0 and cues[-1]["end"] == 20.0
    for a, b in zip(cues, cues[1:]):
        assert a["end"] == pytest.approx(b["start"])


def test_cue_duration_is_proportional_to_characters(skill):
    seg = {"start": 0.0, "end": 10.0, "text": "あ" * 60}  # 40文字(2行) + 20文字(1行)
    cues = skill.build_subtitle_cues([seg], max_chars=20, max_lines=2)
    assert cues[0]["end"] - cues[0]["start"] == pytest.approx(10.0 * 40 / 60, abs=0.01)
    assert cues[1]["end"] - cues[1]["start"] == pytest.approx(10.0 * 20 / 60, abs=0.01)


def test_format_vtt_time(skill):
    assert skill.format_vtt_time(0) == "00:00:00.000"
    assert skill.format_vtt_time(3661.5) == "01:01:01.500"


def test_webvtt_round_trip(skill):
    segments = [
        {"start": 0.0, "end": 2.0, "text": "まず型枠を立てます。"},
        {"start": 2.5, "end": 9.0, "text": "次はセパレーターを入れて、インパクトドライバーで締め付けていきます。"},
    ]
    vtt = skill.build_webvtt(segments, max_chars=20, max_lines=2)
    cues = parse_vtt(vtt)
    assert cues[0] == (0.0, 2.0, ["まず型枠を立てます。"])
    assert all(len(line) <= 20 for _, _, lines in cues for line in lines)
    assert "".join("".join(lines) for _, _, lines in cues) == "".join(s["text"] for s in segments)
    assert cues[-1][1] == 9.0


def test_webvtt_settings_from_env():
    from conftest import load_skill_cell
    cell = load_skill_cell(env={"SKILL_SUBTITLE_MAX_CHARS": "10", "SKILL_SUBTITLE_MAX_LINES": "1"})
    cues = parse_vtt(cell.build_webvtt([{"start": 0, "end": 3, "text": "あ" * 25}]))
    assert [len(lines) for _, _, lines in cues] == [1, 1, 1]
    assert [len(lines[0]) for _, _, lines in cues] == [10, 10, 5]


# --- 音声の取り出し・騒音除去（ffmpegで作った動画で実際に動かす） ---
def test_extract_audio_makes_16k_mono_wav(skill, sample_video, tmp_path):
    wav_path = skill.extract_audio(str(sample_video), str(tmp_path / "a.wav"))
    with wave.open(wav_path) as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getnframes() / 16000 == pytest.approx(3.0, abs=0.2)


def test_extract_audio_fails_for_video_without_audio(skill, silent_video, tmp_path):
    with pytest.raises(Exception):
        skill.extract_audio(str(silent_video), str(tmp_path / "a.wav"))


def test_convert_and_probe_and_frame(skill, sample_video, tmp_path):
    dst = tmp_path / "video.mp4"
    skill.convert_for_playback(str(sample_video), str(dst))
    assert skill.probe_duration(str(dst)) == pytest.approx(3.0, abs=0.2)
    frame = skill.extract_frame(str(dst), 1.0, str(tmp_path / "f.jpg"))
    assert open(frame, "rb").read(2) == b"\xff\xd8"


# --- 文字起こしの保存 ---
def test_save_segments_replaces_previous_result(skill, conn):
    video_id = skill.create_skill_video(conn, "型枠", "", "a.mp4")
    skill.save_segments(conn, video_id, [{"start": 0, "end": 1, "text": "古い"}])
    skill.save_segments(conn, video_id, [{"start": 0, "end": 1, "text": "新しい"}], speaker="山田")
    assert [(s["speaker"], s["text"]) for s in skill.get_segments(conn, video_id)] == [("山田", "新しい")]


def test_mask_sentences_uses_existing_mask_function(skill):
    sentences = [{"start": 0, "end": 1, "text": "山田さんの現場です"}]
    masked = skill.mask_sentences(sentences, lambda turns: [{"word": "山田"}])
    assert masked[0]["text"] == "[MASK]さんの現場です"
