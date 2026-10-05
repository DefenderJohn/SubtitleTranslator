"""全片摘要生成（翻译三步走第 ① 步）。

整片字幕一次调用；文本超过 ``SUMMARY_MAX_CHARS`` 时做简单分块
map-reduce：逐块摘要后再合并成一份总摘要。
"""

from __future__ import annotations

from typing import Protocol

from ..models import SubtitleProject
from . import prompts

# 摘要输入长度上限（字符）：超过则分块 map-reduce
SUMMARY_MAX_CHARS = 100_000


class _ChatFn(Protocol):
    def __call__(self, messages: list[dict[str, str]]) -> str: ...


def _full_text(project: SubtitleProject) -> str:
    return "\n".join(c.text for c in project.cues)


def _chunk_cues(project: SubtitleProject) -> list[str]:
    """按 cue 边界把全文切成不超过 SUMMARY_MAX_CHARS 的块。"""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for cue in project.cues:
        line_len = len(cue.text) + 1
        if current and current_len + line_len > SUMMARY_MAX_CHARS:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(cue.text)
        current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def generate_summary(chat: _ChatFn, project: SubtitleProject) -> str:
    """生成全片摘要。``chat`` 为 ``client.chat`` 风格的调用。"""
    text = _full_text(project)
    if len(text) <= SUMMARY_MAX_CHARS:
        return chat(prompts.summary_messages(text)).strip()
    chunks = _chunk_cues(project)
    partials = [
        chat(prompts.summary_chunk_messages(chunk, i + 1, len(chunks))).strip()
        for i, chunk in enumerate(chunks)
    ]
    return chat(prompts.summary_merge_messages(partials)).strip()
