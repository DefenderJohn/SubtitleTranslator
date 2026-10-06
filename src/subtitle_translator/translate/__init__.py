"""翻译层：OpenAI 兼容端点，三步走，可插拔策略。

三步走：① 全片摘要 → ② 术语表（人工确认检查点，confirmed=true 才开译）
→ ③ 滑动窗口逐句翻译（历史对 + 前瞻 + 术语表/摘要注入 system prompt）。
三步全部走 JSON schema 结构化输出（``ChatClient.chat_json``），不做格式降级。

公开 API：
- :class:`ChatClient`：chat.completions 封装，429/5xx/超时指数退避；
  ``chat`` 纯文本，``chat_json`` JSON schema 结构化输出；
- :class:`TranslationStrategy` ABC + :class:`SlidingWindowStrategy`（第一版实现）；
- :func:`generate_summary` / :func:`extract_glossary` / :func:`parse_glossary_payload`；
- 异常：:class:`ChatError` / :class:`InvalidModelJsonError` /
  :class:`GlossaryParseError` / :class:`GlossaryExtractError` /
  :class:`GlossaryNotConfirmedError`。
"""

from .client import ChatClient, ChatError, InvalidModelJsonError
from .context import SUMMARY_MAX_CHARS, generate_summary
from .glossary import (
    GlossaryExtractError,
    GlossaryParseError,
    extract_glossary,
    parse_glossary_payload,
)
from .strategy import (
    GlossaryNotConfirmedError,
    SlidingWindowStrategy,
    TranslationStrategy,
)

__all__ = [
    "SUMMARY_MAX_CHARS",
    "ChatClient",
    "ChatError",
    "GlossaryExtractError",
    "GlossaryNotConfirmedError",
    "GlossaryParseError",
    "InvalidModelJsonError",
    "SlidingWindowStrategy",
    "TranslationStrategy",
    "extract_glossary",
    "generate_summary",
    "parse_glossary_payload",
]
