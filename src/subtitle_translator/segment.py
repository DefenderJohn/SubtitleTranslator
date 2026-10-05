"""分段器（纯函数）：词级时间戳 -> 字幕行（cue）。

断点优先级：句末标点 > 词间 gap > gap_threshold > 逗号等次级标点 > 超长硬切。
约束：单行不超过 max_chars（英文按字符含空格）与 max_duration 秒；
尽量不产生短于 min_duration 的行（过短则与相邻行合并，但不违反 max 约束）。

text 拼接规则：英文词间补空格；中文按字不补；标点符号前不补空格。
撇号词（don't）、连字符词由对齐器作为整体给出，分段只在词边界下刀，不会拆坏。
"""

from __future__ import annotations

import logging

from .models import Cue, WordTiming

logger = logging.getLogger(__name__)

_SENT_END = frozenset(".!?。！？…")
_SECONDARY = frozenset(",;:，；：、")
_NO_SPACE_BEFORE = frozenset(".,!?;:%)]}，。！？；：、…％）】」』")
_NO_SPACE_AFTER = frozenset("([{（【「『")


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF  # CJK 统一表意文字
        or 0x3040 <= o <= 0x30FF  # 日文假名 + CJK 标点
        or 0xAC00 <= o <= 0xD7AF  # 韩文音节
    )


def _needs_space(prev: str, cur: str) -> bool:
    """两个词之间是否补空格：中文/日文/韩文不补，标点符号前不补。"""
    if not prev or not cur:
        return False
    if cur[0] in _NO_SPACE_BEFORE or prev[-1] in _NO_SPACE_AFTER:
        return False
    if _is_cjk(prev[-1]) or _is_cjk(cur[0]):
        return False
    return True


def _join_texts(texts: list[str]) -> str:
    out = ""
    for text in texts:
        if _needs_space(out, text):
            out += " "
        out += text
    return out


def _boundary_scores(
    words: list[WordTiming], gap_threshold: float
) -> list[int]:
    """每个词前边界的断点分数：3 句末标点 / 2 大 gap / 1 次级标点 / 0 不主动断。"""
    scores = [0] * len(words)
    for j in range(1, len(words)):
        prev, cur = words[j - 1], words[j]
        tail = prev.text[-1:] if prev.text else ""
        if tail in _SENT_END:
            scores[j] = 3
        elif cur.start - prev.end > gap_threshold:
            scores[j] = 2
        elif tail in _SECONDARY:
            scores[j] = 1
    return scores


def segment_words(
    words: list[WordTiming],
    *,
    max_chars: int = 42,
    max_duration: float = 7.0,
    min_duration: float = 1.0,
    gap_threshold: float = 0.6,
) -> list[Cue]:
    """词级时间戳 -> Cue 列表（id 从 1 起，words 保留该行词级时间戳）。"""
    if not words:
        return []
    n = len(words)
    texts = [w.text for w in words]
    scores = _boundary_scores(words, gap_threshold)

    def fits(a: int, b: int) -> bool:
        """words[a:b] 是否满足 max 约束（单词超限只能保留，视为满足）。"""
        if b - a <= 1:
            return True
        if len(_join_texts(texts[a:b])) > max_chars:
            return False
        return words[b - 1].end - words[a].start <= max_duration

    # 贪心切段：尽量延长当前行；触限时在行内回看最佳断点
    spans: list[tuple[int, int]] = []
    s = 0
    while s < n:
        e = s + 1
        hit_limit = False
        while e < n:
            if scores[e] >= 2:
                break  # 句末标点 / 大 gap：主动断
            if not fits(s, e + 1):
                hit_limit = True
                break
            e += 1
        if hit_limit:
            # 回看 (s, e] 内最佳断点：分数高者优先，同分取靠后的（行更饱满）
            best = max(range(s + 1, e + 1), key=lambda j: (scores[j], j))
            if scores[best] == 0:
                logger.warning(
                    "分段硬切：第 %d 个词附近无标点/停顿可断（%.2fs）",
                    e,
                    words[e - 1].end,
                )
            cut = best
        else:
            cut = e
        spans.append((s, cut))
        s = cut

    # min_duration 合并：过短的行优先并入下一行，末尾的并入上一行
    def merged_fits(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return fits(a[0], b[1])

    i = 0
    while i < len(spans) - 1:
        a, b = spans[i]
        if words[b - 1].end - words[a].start < min_duration and merged_fits(
            spans[i], spans[i + 1]
        ):
            spans[i : i + 2] = [(a, spans[i + 1][1])]
            continue
        i += 1
    if len(spans) >= 2:
        a, b = spans[-1]
        if words[b - 1].end - words[a].start < min_duration and merged_fits(
            spans[-2], spans[-1]
        ):
            spans[-2:] = [(spans[-2][0], b)]

    cues: list[Cue] = []
    for idx, (a, b) in enumerate(spans, start=1):
        cues.append(
            Cue(
                id=idx,
                start=words[a].start,
                end=words[b - 1].end,
                text=_join_texts(texts[a:b]),
                words=list(words[a:b]),
            )
        )
    return cues
