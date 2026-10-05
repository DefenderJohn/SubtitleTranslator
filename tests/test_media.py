"""media.py 测试：解析逻辑用预录 stderr 样本，不真跑 ffmpeg；

末尾的集成测试在找到 ffmpeg 时真跑（生成合成音频），否则 skip。
"""

from __future__ import annotations

import sys
import wave

import pytest

from subtitle_translator import media
from subtitle_translator.media import (
    FFmpegNotFoundError,
    find_ffmpeg,
    parse_duration,
    parse_silence_intervals,
)

FFMPEG_I_STDERR = """ffmpeg version 7.0.2-static https://johnvansickle.com/ffmpeg/
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'movie.mp4':
  Metadata:
    major_brand     : isom
  Duration: 01:23:45.67, start: 0.000000, bitrate: 1234 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1920x1080 [SAR 1:1 DAR 16:9], 1000 kb/s, 23.98 fps, 23.98 tbr, 16k tbn (default)
At least one output file must be specified
"""

SILENCEDETECT_STDERR = """Input #0, wav, from 'audio.wav':
  Duration: 00:00:12.00, bitrate: 256 kb/s
Stream mapping:
  Stream #0:0 -> #0:0 (pcm_s16le (native) -> pcm_s16le (native))
[silencedetect @ 0x5ac1d6a0] silence_start: 2.5
[silencedetect @ 0x5ac1d6a0] silence_end: 4.0 | silence_duration: 1.5
[silencedetect @ 0x5ac1d6a0] silence_start: 7.25
[silencedetect @ 0x5ac1d6a0] silence_end: 8.0 | silence_duration: 0.75
Output #0, null, to 'pipe:':
"""

SILENCEDETECT_TRAILING_STDERR = """[silencedetect @ 0x5ac1d6a0] silence_start: 9.5
Output #0, null, to 'pipe:':
"""


class TestParseDuration:
    def test_normal(self):
        assert parse_duration(FFMPEG_I_STDERR) == pytest.approx(5025.67)

    def test_short(self):
        assert parse_duration("Duration: 00:00:05.50, start: 0.0") == pytest.approx(5.5)

    def test_no_duration_raises(self):
        with pytest.raises(ValueError, match="时长"):
            parse_duration("no duration here")


class TestParseSilenceIntervals:
    def test_paired(self):
        assert parse_silence_intervals(SILENCEDETECT_STDERR) == [(2.5, 4.0), (7.25, 8.0)]

    def test_trailing_silence_extended_by_duration(self):
        assert parse_silence_intervals(
            SILENCEDETECT_TRAILING_STDERR, duration=12.0
        ) == [(9.5, 12.0)]

    def test_trailing_silence_without_duration_dropped(self):
        assert parse_silence_intervals(SILENCEDETECT_TRAILING_STDERR) == []

    def test_empty(self):
        assert parse_silence_intervals("nothing\n") == []


class TestFindFFmpeg:
    def test_configured_valid(self, tmp_path):
        fake = tmp_path / "ffmpeg"
        fake.write_text("#!/bin/sh\n")
        assert find_ffmpeg(str(fake)) == str(fake)

    def test_configured_missing_raises(self, tmp_path):
        with pytest.raises(FFmpegNotFoundError, match="不存在"):
            find_ffmpeg(str(tmp_path / "nope"))

    def test_auto_detection_finds_something(self):
        # 本环境装有 imageio-ffmpeg，自动探测应成功
        assert find_ffmpeg()

    def test_not_found_error_has_install_hint(self, monkeypatch):
        monkeypatch.setattr(media.shutil, "which", lambda name: None)
        # sys.modules 里置 None 会让 import 抛 ImportError
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
        with pytest.raises(FFmpegNotFoundError, match="imageio-ffmpeg"):
            find_ffmpeg()


def _ffmpeg_or_skip() -> str:
    try:
        return find_ffmpeg()
    except FFmpegNotFoundError:
        pytest.skip("未安装 ffmpeg，跳过集成测试")


def _make_wav_with_silence(ffmpeg: str, dst, total=5.5, silence=(2.0, 3.5)) -> None:
    """生成 [0,2) 正弦 + [2,3.5) 静音 + [3.5,5.5) 正弦 的 16kHz 单声道 wav。"""
    expr = (
        "sin(2*PI*440*t)*(1-between(t\\,%s\\,%s))" % (silence[0], silence[1])
    )
    media._run(
        ffmpeg,
        [
            "-y",
            "-f", "lavfi",
            "-i", f"aevalsrc={expr}:d={total}:s=16000",
            "-ac", "1",
            str(dst),
        ],
    )


@pytest.mark.integration
class TestWithRealFFmpeg:
    def test_probe_duration(self, tmp_path):
        ffmpeg = _ffmpeg_or_skip()
        wav = tmp_path / "tone.wav"
        _make_wav_with_silence(ffmpeg, wav)
        assert media.probe_duration(wav, ffmpeg=ffmpeg) == pytest.approx(5.5, abs=0.1)

    def test_detect_silences(self, tmp_path):
        ffmpeg = _ffmpeg_or_skip()
        wav = tmp_path / "tone.wav"
        _make_wav_with_silence(ffmpeg, wav)
        silences = media.detect_silences(wav, ffmpeg=ffmpeg)
        assert len(silences) == 1
        (s, e) = silences[0]
        assert s == pytest.approx(2.0, abs=0.2)
        assert e == pytest.approx(3.5, abs=0.2)

    def test_extract_audio_chunk(self, tmp_path):
        ffmpeg = _ffmpeg_or_skip()
        src = tmp_path / "tone.wav"
        _make_wav_with_silence(ffmpeg, src)
        dst = tmp_path / "chunk.wav"
        media.extract_audio_chunk(src, 1.0, 3.0, dst, ffmpeg=ffmpeg)
        with wave.open(str(dst), "rb") as w:
            assert w.getframerate() == 16000
            assert w.getnchannels() == 1
            assert w.getnframes() / w.getframerate() == pytest.approx(2.0, abs=0.1)

    def test_extract_audio_chunk_invalid_range(self, tmp_path):
        ffmpeg = _ffmpeg_or_skip()
        with pytest.raises(ValueError):
            media.extract_audio_chunk("x.wav", 3.0, 1.0, tmp_path / "o.wav", ffmpeg=ffmpeg)
