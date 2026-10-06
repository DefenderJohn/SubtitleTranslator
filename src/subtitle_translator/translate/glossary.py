"""术语表提取与解析（翻译三步走第 ② 步）。

模型按 ``prompts.GLOSSARY_RESPONSE_SCHEMA`` 输出 ``{"entries": [{src, dst,
count}]}``，直接读 JSON 字段解析成 :class:`~subtitle_translator.models.GlossaryEntry`。
JSON 无效或 entries 为空时重试，最多 ``cfg.glossary_max_retries`` 次，
仍失败抛 :class:`GlossaryExtractError`。
"""

from __future__ import annotations

from typing import Any, Protocol

from ..config import TranslateConfig
from ..models import GlossaryEntry, SubtitleProject
from . import prompts
from .client import InvalidModelJsonError


class _ChatJsonFn(Protocol):
    def __call__(
        self, messages: list[dict[str, str]], *, schema: dict, name: str
    ) -> dict: ...


class GlossaryParseError(ValueError):
    """模型输出无法解析出任何有效术语条目。"""


class GlossaryExtractError(RuntimeError):
    """术语表提取在重试耗尽后仍失败。"""


def parse_glossary_payload(data: dict[str, Any]) -> list[GlossaryEntry]:
    """从 ``{"entries": [...]}`` payload 解析术语条目；一条都没有时抛 GlossaryParseError。"""
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list):
        raise GlossaryParseError(f"术语表输出缺少 entries 数组: {str(data)[:200]!r}")
    entries: list[GlossaryEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        src = str(item.get("src") or "").strip()
        dst = str(item.get("dst") or "").strip()
        if not src or not dst:
            continue
        try:
            count = int(item.get("count") or 0)
        except (TypeError, ValueError):
            count = 0
        entries.append(GlossaryEntry(src=src, dst=dst, count=count))
    if not entries:
        raise GlossaryParseError(f"术语表输出没有有效条目: {str(data)[:200]!r}")
    return entries


def extract_glossary(
    chat_json: _ChatJsonFn,
    project: SubtitleProject,
    cfg: TranslateConfig,
) -> list[GlossaryEntry]:
    """提取术语表：输出无效时重试，耗尽后抛 GlossaryExtractError。"""
    text = "\n".join(c.text for c in project.cues)
    messages = prompts.glossary_messages(
        project.meta.summary,
        text,
        max_entries=cfg.glossary_max_entries,
        target_language=cfg.target_language,
    )
    last_error: Exception | None = None
    for _attempt in range(cfg.glossary_max_retries):
        try:
            entries = parse_glossary_payload(
                chat_json(
                    messages,
                    schema=prompts.GLOSSARY_RESPONSE_SCHEMA,
                    name="glossary",
                )
            )
        except (InvalidModelJsonError, GlossaryParseError) as exc:
            last_error = exc
            continue
        return entries[: cfg.glossary_max_entries]
    raise GlossaryExtractError(
        f"术语表提取失败：{cfg.glossary_max_retries} 次尝试均无法解析模型输出"
    ) from last_error
