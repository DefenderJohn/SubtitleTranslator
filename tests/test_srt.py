"""srt.py 测试：时间戳边界、双语/单语导出、parse round-trip、BOM/CRLF 容错。"""

import pytest

from subtitle_translator.models import Cue, SubtitleProject
from subtitle_translator.srt import (
    export_srt,
    format_timestamp,
    parse_srt,
    parse_timestamp,
)


def make_project() -> SubtitleProject:
    project = SubtitleProject()
    project.cues = [
        Cue(id=1, start=0.0, end=1.5, text="hello", translation="你好"),
        Cue(id=2, start=61.0, end=62.5, text="world"),
    ]
    return project


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (61.0, "00:01:01,000"),
        (3661.001, "01:01:01,001"),
        (59.9996, "00:01:00,000"),  # 毫秒进位到秒，毫秒位不得出现 60
        (3599.9996, "01:00:00,000"),
        (0.0004, "00:00:00,000"),
    ],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected
    # 毫秒位永不溢出
    assert expected.split(",")[1] != "60"


def test_format_timestamp_negative():
    with pytest.raises(ValueError):
        format_timestamp(-0.1)


@pytest.mark.parametrize(
    "timestamp,expected",
    [
        ("00:00:00,000", 0.0),
        ("00:01:01,000", 61.0),
        ("01:01:01,001", 3661.001),
        ("00:00:01.500", 1.5),  # 容忍小数点分隔
    ],
)
def test_parse_timestamp(timestamp, expected):
    assert parse_timestamp(timestamp) == pytest.approx(expected)


def test_parse_timestamp_invalid():
    with pytest.raises(ValueError, match="非法 SRT 时间戳"):
        parse_timestamp("1:2:3")


def test_export_bilingual(tmp_path):
    path = tmp_path / "out.srt"
    export_srt(make_project(), path, bilingual=True)
    content = path.read_text(encoding="utf-8")
    assert content == (
        "1\n00:00:00,000 --> 00:00:01,500\n你好\nhello\n"
        "\n"
        "2\n00:01:01,000 --> 00:01:02,500\nworld\n"
    )  # 双语时译文在上原文在下；无译文退化为原文


def test_export_monolingual_translation(tmp_path):
    path = tmp_path / "out.srt"
    export_srt(make_project(), path, bilingual=False)
    content = path.read_text(encoding="utf-8")
    assert "你好\n" in content
    assert "hello" not in content
    # 无译文条目导出原文
    assert "world" in content


def test_export_parse_round_trip(tmp_path):
    project = make_project()
    path = tmp_path / "out.srt"
    export_srt(project, path, bilingual=False)
    cues = parse_srt(path)
    assert len(cues) == 2
    assert cues[0].id == 1
    assert cues[0].start == pytest.approx(0.0)
    assert cues[0].end == pytest.approx(1.5)
    assert cues[0].text == "你好"
    assert cues[1].start == pytest.approx(61.0)
    assert cues[1].text == "world"


def test_parse_bom_and_crlf(tmp_path):
    path = tmp_path / "bom.srt"
    content = "1\r\n00:00:00,000 --> 00:00:01,000\r\n第一行\r\n\r\n\r\n2\r\n00:00:02,000 --> 00:00:03,000\r\n第二行\r\n"
    path.write_bytes(("\ufeff" + content).encode("utf-8"))
    cues = parse_srt(path)
    assert len(cues) == 2
    assert cues[0].text == "第一行"
    assert cues[1].id == 2
    assert cues[1].end == pytest.approx(3.0)


def test_parse_multiline_text_and_malformed_block(tmp_path):
    path = tmp_path / "mix.srt"
    path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n两行\n文本\n\n"
        "这是坏块，没有时间轴\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nOK\n",
        encoding="utf-8",
    )
    cues = parse_srt(path)
    assert len(cues) == 2
    assert cues[0].text == "两行\n文本"
    assert cues[1].text == "OK"
