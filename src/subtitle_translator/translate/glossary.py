"""术语表提取与解析（翻译三步走第 ② 步）。

模型按固定格式输出 ``原文 | 译文 | 出现次数``，逐行解析成
:class:`~subtitle_translator.models.GlossaryEntry`。解析失败时换提示
（追加严格格式说明）并降批（条数上限减半）重试，最多
``cfg.glossary_max_retries`` 次，仍失败抛 :class:`GlossaryExtractError`。
"""

from __future__ import annotations

from typing import Protocol

from ..config import TranslateConfig
from ..models import GlossaryEntry, SubtitleProject
from . import prompts


class _ChatFn(Protocol):
    def __call__(self, messages: list[dict[str, str]]) -> str: ...


class GlossaryParseError(ValueError):
    """模型输出无法解析出任何有效术语条目。"""


class GlossaryExtractError(RuntimeError):
    """术语表提取在重试耗尽后仍失败。"""


def parse_glossary(text: str) -> list[GlossaryEntry]:
    """解析 ``原文 | 译文 | 出现次数`` 行；一条都解析不出时抛 GlossaryParseError。"""
    entries: list[GlossaryEntry] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        src, dst = parts[0], parts[1]
        if not src or not dst:
            continue
        count = 0
        if len(parts) >= 3:
            try:
                count = int(parts[2])
            except ValueError:
                count = 0
        entries.append(GlossaryEntry(src=src, dst=dst, count=count))
    if not entries:
        raise GlossaryParseError(f"术语表输出无法解析: {text[:200]!r}")
    return entries


def extract_glossary(
    chat: _ChatFn,
    project: SubtitleProject,
    cfg: TranslateConfig,
) -> list[GlossaryEntry]:
    """提取术语表：解析失败时换提示/降批重试，耗尽后抛 GlossaryExtractError。"""
    text = "\n".join(c.text for c in project.cues)
    last_error: Exception | None = None
    for attempt in range(cfg.glossary_max_retries):
        # 降批：重试时把条数上限减半，降低模型输出负担
        max_entries = max(5, cfg.glossary_max_entries // (2**attempt))
        messages = prompts.glossary_messages(
            project.meta.summary,
            text,
            max_entries=max_entries,
            target_language=cfg.target_language,
            strict=attempt > 0,
        )
        try:
            entries = parse_glossary(chat(messages))
        except GlossaryParseError as exc:
            last_error = exc
            continue
        return entries[: cfg.glossary_max_entries]
    raise GlossaryExtractError(
        f"术语表提取失败：{cfg.glossary_max_retries} 次尝试均无法解析模型输出"
    ) from last_error
