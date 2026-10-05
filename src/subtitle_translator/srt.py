"""SRT 读写导出。

SRT 是导出物，不是存储格式；唯一事实来源是 models.py 的 JSON 工程文件。

- format_timestamp / parse_timestamp：时间戳转换
- export_srt：把 SubtitleProject.cues 导出为单语 / 双语 SRT（双语时译文在上、原文在下）
- parse_srt：读回标准 SRT，用于导入外部字幕 / 断点恢复
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

from .models import Cue, SubtitleProject

_TIMESTAMP_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def format_timestamp(seconds: float) -> str:
    """秒数 -> "HH:MM:SS,mmm"。毫秒四舍五入到整体毫秒数后进位，保证毫秒位不会溢出为 60。"""
    if seconds < 0:
        raise ValueError(f"时间戳不能为负: {seconds!r}")
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_timestamp(timestamp: str) -> float:
    """"HH:MM:SS,mmm"（也容忍小数点分隔）-> 秒数。"""
    m = re.match(r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*$", timestamp)
    if not m:
        raise ValueError(f"非法 SRT 时间戳: {timestamp!r}")
    hours, minutes, secs, millis = (int(g) for g in m.groups())
    return hours * 3600 + minutes * 60 + secs + millis / 1000.0


def export_srt(
    project: SubtitleProject,
    path: Union[str, Path],
    bilingual: bool = False,
) -> None:
    """导出 SRT。bilingual=True 时译文在上、原文在下；无译文的条目退化为只有原文。"""
    blocks = []
    for index, cue in enumerate(project.cues, start=1):
        lines = [
            str(index),
            f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}",
        ]
        if bilingual and cue.translation:
            lines.append(cue.translation)
            lines.append(cue.text)
        elif cue.translation is not None and not bilingual:
            lines.append(cue.translation)
        else:
            lines.append(cue.text)
        blocks.append("\n".join(lines))
    content = "\n\n".join(blocks)
    if content:
        content += "\n"
    Path(path).write_text(content, encoding="utf-8")


def parse_srt(path: Union[str, Path]) -> list[Cue]:
    """读回标准 SRT。容忍 BOM、\\r\\n、多余空行；跳过没有时间轴行的块。"""
    raw = Path(path).read_bytes().decode("utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        ts_match = None
        ts_index = -1
        for i, line in enumerate(lines[:2]):
            ts_match = _TIMESTAMP_RE.match(line)
            if ts_match:
                ts_index = i
                break
        if not ts_match:
            continue
        g = ts_match.groups()
        start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000.0
        end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000.0
        text = "\n".join(lines[ts_index + 1 :]).strip()
        index = len(cues) + 1
        if ts_index == 1:
            try:
                index = int(lines[0].strip())
            except ValueError:
                pass
        cues.append(Cue(id=index, start=start, end=end, text=text))
    return cues
