"""segment.py 测试：构造合成 WordTiming 序列验证约束与断点选择。"""

from __future__ import annotations

import pytest

from subtitle_translator.models import WordTiming
from subtitle_translator.segment import segment_words


def make_words(pairs, start=0.0, step=0.4, dur=0.35):
    """pairs: 词文本列表；均匀排布，词间 gap = step - dur。"""
    words = []
    t = start
    for text in pairs:
        words.append(WordTiming(text=text, start=t, end=t + dur))
        t += step
    return words


class TestBasic:
    def test_empty(self):
        assert segment_words([]) == []

    def test_single_word(self):
        cues = segment_words(make_words(["hello"]))
        assert len(cues) == 1
        assert cues[0].text == "hello"
        assert cues[0].id == 1
        assert cues[0].start == pytest.approx(0.0)

    def test_english_joined_with_spaces(self):
        cues = segment_words(make_words(["I", "don't", "know", "well-known", "words."]))
        assert cues[0].text == "I don't know well-known words."

    def test_chinese_joined_without_spaces(self):
        cues = segment_words(make_words(["我", "今", "天", "很", "高", "兴", "。"]))
        assert cues[0].text == "我今天很高兴。"

    def test_no_space_before_punctuation_token(self):
        cues = segment_words(make_words(["你好", "，", "世界", "。"]))
        assert cues[0].text == "你好，世界。"

    def test_words_preserved_and_ids_sequential(self):
        words = make_words(["a", "b.", "c", "d."])
        cues = segment_words(words, min_duration=0.0)
        assert [c.id for c in cues] == [1, 2]
        assert cues[0].words == words[:2]
        assert cues[1].words == words[2:]


class TestBreakpointPriority:
    def test_sentence_punctuation_breaks(self):
        words = make_words(["第一句。", "第二句。", "第三句。"])
        cues = segment_words(words, min_duration=0.0)
        assert [c.text for c in cues] == ["第一句。", "第二句。", "第三句。"]

    def test_gap_breaks(self):
        words = make_words(["hello", "world", "again"], step=0.4)
        # 在 world 与 again 之间制造 1.2s 的 gap
        words[2] = WordTiming(text="again", start=words[1].end + 1.2, end=words[1].end + 1.5)
        cues = segment_words(words, min_duration=0.0, gap_threshold=0.6)
        assert [c.text for c in cues] == ["hello world", "again"]

    def test_small_gap_does_not_break(self):
        words = make_words(["hello", "world"], step=0.4)  # gap = 0.05
        cues = segment_words(words, gap_threshold=0.6)
        assert len(cues) == 1

    def test_comma_only_used_when_forced(self):
        # 逗号不主动断行
        words = make_words(["说点什么，", "然后继续。"])
        cues = segment_words(words, min_duration=0.0)
        assert len(cues) == 1


class TestMaxConstraints:
    def test_max_chars_prefers_comma_over_hard_cut(self):
        # 长句超限时应回看在逗号处断
        words = make_words(
            ["alpha", "beta", "gamma,", "delta", "epsilon", "zeta."], step=0.5, dur=0.45
        )
        cues = segment_words(words, max_chars=20, min_duration=0.0)
        assert all(len(c.text) <= 20 for c in cues)
        assert cues[0].text == "alpha beta gamma,"  # 在逗号处断，而非硬切
        assert cues[1].text == "delta epsilon zeta."

    def test_max_chars_hard_cut_without_punctuation(self, caplog):
        words = make_words(["aaaa", "bbbb", "cccc", "dddd"], step=0.5, dur=0.45)
        with caplog.at_level("WARNING", logger="subtitle_translator.segment"):
            cues = segment_words(words, max_chars=10, min_duration=0.0)
        assert all(len(c.text) <= 10 for c in cues)
        assert "硬切" in caplog.text

    def test_max_duration_enforced(self):
        words = make_words(["w%d" % i for i in range(20)], step=0.5, dur=0.45)
        cues = segment_words(words, max_duration=3.0, min_duration=0.0)
        assert len(cues) > 1
        assert all(c.end - c.start <= 3.0 + 1e-9 for c in cues)
        # 覆盖完整且顺序连续
        assert cues[0].words[0].text == "w0"
        assert cues[-1].words[-1].text == "w19"

    def test_single_overlong_word_kept(self):
        words = [WordTiming(text="supercalifragilisticexpialidocious", start=0.0, end=2.0)]
        cues = segment_words(words, max_chars=10)
        assert cues[0].text == "supercalifragilisticexpialidocious"


class TestMinDurationMerge:
    def test_short_line_merges_forward(self):
        # "好。" 只有 0.35s，应与后行合并
        words = make_words(["好。", "接", "下", "来", "是", "正", "文", "。"], step=0.5, dur=0.45)
        cues = segment_words(words, min_duration=1.0)
        assert cues[0].text.startswith("好。接")
        assert all(c.end - c.start >= 1.0 for c in cues)

    def test_trailing_short_line_merges_backward(self):
        words = make_words(["前", "面", "是", "长", "句", "子", "。", "嗯。"], step=0.5, dur=0.45)
        cues = segment_words(words, min_duration=1.0)
        assert cues[-1].text.endswith("嗯。")
        assert all(c.end - c.start >= 1.0 for c in cues)

    def test_merge_never_violates_max(self):
        # 短行无法合并（合并会超 max_chars）时保持原样
        words = make_words(["aaaa", "bbbb", "ccc.", "dddd", "eeee", "ffff"], step=0.5, dur=0.45)
        cues = segment_words(words, max_chars=14, min_duration=1.0)
        assert all(len(c.text) <= 14 for c in cues)
