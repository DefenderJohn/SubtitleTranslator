"""全片摘要生成（翻译三步走第 ① 步）。

整片字幕一次调用；文本超过 ``SUMMARY_MAX_CHARS`` 时做简单分块
map-reduce：逐块摘要后再合并成一份总摘要。模型输出由
``prompts.SUMMARY_RESPONSE_SCHEMA`` 约束为 ``{"summary": str}``。
"""

from __future__ import annotations

from typing import Protocol

from ..models import SubtitleProject
from . import prompts
from .client import InvalidModelJsonError

# 摘要输入长度上限（字符）：超过则分块 map-reduce
SUMMARY_MAX_CHARS = 100_000


class _ChatJsonFn(Protocol):
    def __call__(
        self, messages: list[dict[str, str]], *, schema: dict, name: str
    ) -> dict: ...


def _summarize(chat_json: _ChatJsonFn, messages: list[dict[str, str]]) -> str:
    data = chat_json(
        messages, schema=prompts.SUMMARY_RESPONSE_SCHEMA, name="summary"
    )
    summary = data.get("summary")
    if not isinstance(summary, str):
        raise InvalidModelJsonError(f"摘要输出缺少 summary 字段: {str(data)[:200]!r}")
    return summary.strip()


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


def generate_summary(chat_json: _ChatJsonFn, project: SubtitleProject) -> str:
    """生成全片摘要。``chat_json`` 为 ``client.chat_json`` 风格的调用。"""
    text = _full_text(project)
    if len(text) <= SUMMARY_MAX_CHARS:
        return _summarize(chat_json, prompts.summary_messages(text))
    chunks = _chunk_cues(project)
    partials = [
        _summarize(chat_json, prompts.summary_chunk_messages(chunk, i + 1, len(chunks)))
        for i, chunk in enumerate(chunks)
    ]
    return _summarize(chat_json, prompts.summary_merge_messages(partials))
