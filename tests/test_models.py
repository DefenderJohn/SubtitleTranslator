"""models.py 测试：序列化 round-trip、缺字段容错、stage 校验、version 校验。"""

import json

import pytest

from subtitle_translator.models import (
    SCHEMA_VERSION,
    Cue,
    GlossaryEntry,
    SchemaVersionError,
    Stage,
    SubtitleProject,
    WordTiming,
)


def make_project() -> SubtitleProject:
    project = SubtitleProject()
    project.meta.source.file = "demo.mp4"
    project.meta.source.duration = 12.5
    project.meta.source.language = "en"
    project.meta.models.asr = "Qwen/Qwen3-ASR-1.7B"
    project.meta.models.aligner = "Qwen/Qwen3-ForcedAligner-0.6B"
    project.meta.models.translator = "Qwen/Qwen3-32B"
    project.meta.summary = "一段演示"
    project.stage = Stage.TRANSCRIBED
    project.glossary.append(GlossaryEntry(src="Transformer", dst="变形金刚", count=3, confirmed=True))
    project.cues.append(
        Cue(
            id=1,
            start=0.0,
            end=1.5,
            text="hello world",
            translation="你好世界",
            words=[
                WordTiming(text="hello", start=0.0, end=0.6),
                WordTiming(text="world", start=0.7, end=1.5),
            ],
        )
    )
    project.cues.append(Cue(id=2, start=2.0, end=3.0, text="bye"))
    return project


def test_round_trip(tmp_path):
    project = make_project()
    path = tmp_path / "demo.json"
    project.save(path)
    loaded = SubtitleProject.load(path)
    assert loaded.to_dict() == project.to_dict()
    assert loaded.cues[0].words[1].text == "world"
    assert loaded.glossary[0].confirmed is True
    assert loaded.cues[1].translation is None


def test_saved_json_format(tmp_path):
    path = tmp_path / "demo.json"
    make_project().save(path)
    raw = path.read_text(encoding="utf-8")
    assert "你好世界" in raw  # ensure_ascii=False
    assert "\n  " in raw  # indent=2
    data = json.loads(raw)
    assert data["version"] == SCHEMA_VERSION
    assert data["stage"] == "transcribed"


def test_from_dict_missing_fields_get_defaults():
    project = SubtitleProject.from_dict({"version": SCHEMA_VERSION})
    assert project.meta.source.file == ""
    assert project.meta.models.asr == ""
    assert project.meta.summary == ""
    assert project.glossary == []
    assert project.cues == []
    assert project.stage is Stage.EMPTY


def test_from_dict_partial_cue():
    project = SubtitleProject.from_dict(
        {"version": SCHEMA_VERSION, "cues": [{"text": "hi"}]}
    )
    cue = project.cues[0]
    assert cue.text == "hi"
    assert cue.start == 0.0
    assert cue.words == []


def test_version_mismatch_raises():
    with pytest.raises(SchemaVersionError, match="version=99"):
        SubtitleProject.from_dict({"version": 99})


def test_stage_validation():
    project = SubtitleProject()
    project.stage = "translated"
    assert project.stage is Stage.TRANSLATED
    with pytest.raises(ValueError, match="非法 stage"):
        project.stage = "done"
    with pytest.raises(ValueError, match="非法 stage"):
        SubtitleProject.from_dict({"version": SCHEMA_VERSION, "stage": "bogus"})


def test_stage_order_values():
    assert [s.value for s in Stage] == ["empty", "transcribed", "contexted", "translated"]


def test_load_rejects_non_dict(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TypeError):
        SubtitleProject.load(path)
