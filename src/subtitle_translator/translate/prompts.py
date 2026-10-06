"""翻译层 prompt 模板与结构化输出 schema。

三类 prompt（摘要 / 术语表 / 逐句翻译）全部走 JSON schema 结构化输出
（``client.chat_json``），每个模板旁边是对应的 response schema——prompt
里的格式说明与 schema 是同一份约定的两份表达，改动时必须同步。
"""

from __future__ import annotations

from ..config import TranslateConfig
from ..models import GlossaryEntry

# ---------------------------------------------------------------- 摘要

SUMMARY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

SUMMARY_SYSTEM = (
    "你是一名专业的影视内容分析助手。请阅读用户给出的整片字幕文本，"
    "输出一份简洁的内容摘要，供后续翻译参考。摘要需涵盖：作品体裁与题材、"
    "语言风格与语域（正式/口语/术语密度）、主要人物及其关系、关键专有名词。"
    '以 JSON 对象 {"summary": "摘要正文"} 输出，摘要正文不要包含标题或额外说明。'
)

SUMMARY_CHUNK_SYSTEM = (
    "你是一名专业的影视内容分析助手。用户给出的是一部作品字幕的【部分内容】，"
    "请对这部分做简要摘要：情节进展、出场人物、出现的专有名词与术语。"
    '以 JSON 对象 {"summary": "摘要正文"} 输出，摘要正文不要包含标题或额外说明。'
)

SUMMARY_MERGE_SYSTEM = (
    "你是一名专业的影视内容分析助手。用户给出的是一部作品各分段字幕摘要的集合，"
    "请合并为一份全片摘要，涵盖：作品体裁与题材、语言风格与语域、"
    "主要人物及其关系、关键专有名词。"
    '以 JSON 对象 {"summary": "摘要正文"} 输出，摘要正文不要包含标题或额外说明。'
)


def summary_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": f"以下是整片字幕文本：\n\n{text}"},
    ]


def summary_chunk_messages(text: str, index: int, total: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SUMMARY_CHUNK_SYSTEM},
        {"role": "user", "content": f"以下是第 {index}/{total} 段字幕文本：\n\n{text}"},
    ]


def summary_merge_messages(partials: list[str]) -> list[dict[str, str]]:
    body = "\n\n".join(f"【分段摘要 {i + 1}】\n{p}" for i, p in enumerate(partials))
    return [
        {"role": "system", "content": SUMMARY_MERGE_SYSTEM},
        {"role": "user", "content": body},
    ]


# ---------------------------------------------------------------- 术语表

GLOSSARY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["src", "dst", "count"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entries"],
    "additionalProperties": False,
}

GLOSSARY_SYSTEM = (
    "你是一名专业的字幕翻译术语管理助手。请从用户给出的全片字幕中，"
    "提取高频出现的专有名词（人名、地名、组织名、作品特有术语等），"
    "并给出目标语言的规范译法。"
)


def glossary_messages(
    summary: str,
    text: str,
    *,
    max_entries: int,
    target_language: str,
) -> list[dict[str, str]]:
    """术语表提取 prompt（输出由 GLOSSARY_RESPONSE_SCHEMA 约束）。"""
    user = (
        f"目标语言：{target_language}\n"
        '输出格式：JSON 对象 {"entries": [{"src": 原文, "dst": 译文, "count": 出现次数}]}，'
        f"按出现次数降序，最多 {max_entries} 条。\n\n"
        f"全片摘要：\n{summary or '（无）'}\n\n"
        f"全片字幕文本：\n{text}"
    )
    return [
        {"role": "system", "content": GLOSSARY_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------- 逐句翻译

TRANSLATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"translation": {"type": "string"}},
    "required": ["translation"],
    "additionalProperties": False,
}


def translation_system(
    cfg: TranslateConfig,
    glossary: list[GlossaryEntry],
    summary: str,
) -> str:
    """system prompt：角色 + 目标语言 + 术语表 + 摘要 + additional_prompt。"""
    parts = [
        f"你是一名专业字幕翻译员，请将字幕原文翻译成{cfg.target_language}。",
        '要求：以 JSON 对象 {"translation": "译文"} 输出，translation 字段只含'
        "译文本身，不要输出编号、原文、解释或任何额外内容；"
        "译文须符合字幕习惯，简洁口语化，单行不宜过长。",
    ]
    if cfg.additional_prompt:
        parts.append(f"补充要求：{cfg.additional_prompt}")
    if summary:
        parts.append(f"全片摘要（供理解上下文）：\n{summary}")
    confirmed = [g for g in glossary if g.confirmed]
    if confirmed:
        lines = "\n".join(f"{g.src} | {g.dst}" for g in confirmed)
        parts.append(f"术语表（必须严格采用对应译法）：\n{lines}")
    return "\n\n".join(parts)


def translation_user(text: str, forward: list[str]) -> str:
    """user prompt：待译原文 + 前瞻原文（明确标注不翻译）。"""
    user = f"请翻译以下字幕原文：\n{text}"
    if forward:
        forward_block = "\n".join(forward)
        user += (
            f"\n\n以下为后续原文，仅供上下文参考，【不要翻译】：\n{forward_block}"
        )
    return user
