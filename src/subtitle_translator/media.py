"""ffmpeg 封装：探测时长、静音检测、抽取音频片段。

ffmpeg 是硬依赖（转录切块、将来剪辑都要用）。二进制路径解析顺序：
配置项 ``asr.ffmpeg_path`` → PATH 里的 ffmpeg → imageio-ffmpeg 自带的静态
二进制；都找不到时抛 FFmpegNotFoundError 并给出安装指引。

没有 ffprobe 时，时长获取用 ``ffmpeg -i`` 的 stderr 解析兜底。
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union


class FFmpegNotFoundError(RuntimeError):
    """找不到可用的 ffmpeg 二进制。"""


_INSTALL_HINT = (
    "找不到 ffmpeg。可选安装方式（任选其一）：\n"
    "  1. pip install imageio-ffmpeg   （自带静态 ffmpeg 二进制，无需 sudo）\n"
    "  2. conda install -y -c conda-forge ffmpeg\n"
    "  3. sudo apt install ffmpeg\n"
    "也可在 config.yaml 的 asr.ffmpeg_path 中直接指定 ffmpeg 二进制路径。"
)


def find_ffmpeg(configured: Optional[str] = None) -> str:
    """解析 ffmpeg 二进制路径：配置项 → PATH → imageio-ffmpeg；都找不到抛错。"""
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        raise FFmpegNotFoundError(
            f"配置的 ffmpeg 路径不存在: {configured!r}\n{_INSTALL_HINT}"
        )
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except ImportError:
        pass
    raise FFmpegNotFoundError(_INSTALL_HINT)


def _run(ffmpeg: str, args: list[str]) -> subprocess.CompletedProcess:
    """跑 ffmpeg 并返回结果。ffmpeg 把探测信息打到 stderr，调用方自行解析。"""
    return subprocess.run(
        [ffmpeg, "-hide_banner", *args],
        capture_output=True,
        text=True,
        timeout=600,
    )


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")

# silencedetect 输出样例：
#   [silencedetect @ 0x...] silence_start: 2.34
#   [silencedetect @ 0x...] silence_end: 4.12 | silence_duration: 1.78
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


def parse_duration(stderr_text: str) -> float:
    """从 ``ffmpeg -i`` 的 stderr 解析 Duration，返回秒数。"""
    m = _DURATION_RE.search(stderr_text)
    if not m:
        raise ValueError("无法从 ffmpeg 输出中解析媒体时长")
    hours, minutes, seconds = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return hours * 3600 + minutes * 60 + seconds


def parse_silence_intervals(
    stderr_text: str,
    duration: Optional[float] = None,
) -> list[tuple[float, float]]:
    """解析 silencedetect 的 stderr，返回 [(start, end), ...] 静音区间。

    末尾出现没有配对的 silence_start（静音持续到文件结束）时，
    若给了 duration 则以 duration 收尾，否则丢弃该段。
    """
    intervals: list[tuple[float, float]] = []
    pending: Optional[float] = None
    for line in stderr_text.splitlines():
        m = _SILENCE_START_RE.search(line)
        if m:
            pending = float(m.group(1))
            continue
        m = _SILENCE_END_RE.search(line)
        if m and pending is not None:
            intervals.append((pending, float(m.group(1))))
            pending = None
    if pending is not None and duration is not None:
        intervals.append((pending, duration))
    return intervals


def probe_duration(
    path: Union[str, Path],
    ffmpeg: Optional[str] = None,
) -> float:
    """媒体时长（秒）。有 ffprobe 用 ffprobe，否则解析 ``ffmpeg -i`` 的 stderr。"""
    path = str(path)
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            try:
                return float(result.stdout.strip())
            except ValueError:
                pass  # 兜底走 ffmpeg -i
    ffmpeg = ffmpeg or find_ffmpeg()
    result = _run(ffmpeg, ["-i", path])
    # ffmpeg -i 不带输出文件时 returncode=1 属正常，时长信息在 stderr
    return parse_duration(result.stderr)


def detect_silences(
    path: Union[str, Path],
    noise_db: float = -35.0,
    min_silence: float = 0.4,
    ffmpeg: Optional[str] = None,
) -> list[tuple[float, float]]:
    """用 silencedetect 找出静音区间 [(start, end), ...]。

    noise_db：低于该分贝视为静音；min_silence：最短静音时长（秒）。
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    af = f"silencedetect=noise={noise_db}dB:d={min_silence}"
    result = _run(ffmpeg, ["-nostats", "-i", str(path), "-af", af, "-f", "null", "-"])
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg silencedetect 失败: {result.stderr[-500:]}")
    try:
        duration = probe_duration(path, ffmpeg=ffmpeg)
    except ValueError:
        duration = None
    return parse_silence_intervals(result.stderr, duration)


def extract_audio_chunk(
    src: Union[str, Path],
    start: float,
    end: float,
    dst_wav: Union[str, Path],
    ffmpeg: Optional[str] = None,
) -> Path:
    """抽取 [start, end) 片段为 16kHz 单声道 wav（ASR 输入规格）。"""
    if end <= start:
        raise ValueError(f"非法片段区间: start={start}, end={end}")
    ffmpeg = ffmpeg or find_ffmpeg()
    dst = Path(dst_wav)
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        ffmpeg,
        [
            "-y",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", str(src),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            str(dst),
        ],
    )
    if result.returncode != 0 or not dst.is_file():
        raise RuntimeError(f"ffmpeg 抽取音频片段失败: {result.stderr[-500:]}")
    return dst
