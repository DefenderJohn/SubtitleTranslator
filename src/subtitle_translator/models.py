"""数据结构与 JSON schema（唯一事实来源）。

每个视频一个 JSON 文件，是唯一事实来源；SRT 只是导出物。

Schema 字段：
- version
- source{file, duration, language}
- models{asr, aligner, translator}
- summary
- glossary[{src, dst, count, confirmed}]
- cues[{id, start, end, text, translation, words[{text, start, end}], flags}]
- stage（empty -> transcribed -> contexted -> translated，驱动断点续传）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Union

SCHEMA_VERSION = 1


class Stage(str, Enum):
    """流水线阶段，驱动断点续传。empty 为初始态（尚未转录）。"""

    EMPTY = "empty"
    TRANSCRIBED = "transcribed"
    CONTEXTED = "contexted"
    TRANSLATED = "translated"

    @classmethod
    def parse(cls, value: Any) -> "Stage":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError:
            valid = ", ".join(s.value for s in cls)
            raise ValueError(f"非法 stage: {value!r}，合法取值为: {valid}") from None


class SchemaVersionError(ValueError):
    """JSON 文件 version 与当前 SCHEMA_VERSION 不匹配。"""


@dataclass
class WordTiming:
    text: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordTiming":
        return cls(
            text=str(data.get("text", "")),
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
        )


@dataclass
class Cue:
    id: int
    start: float
    end: float
    text: str
    translation: Union[str, None] = None
    words: list[WordTiming] = field(default_factory=list)
    # 翻译层标记，如 "glossary_miss:<src>"（术语后校验未命中）、
    # "translation_failed"（重试后仍失败，保留原文占位）
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "translation": self.translation,
            "words": [w.to_dict() for w in self.words],
            "flags": list(self.flags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Cue":
        words = data.get("words") or []
        return cls(
            id=int(data.get("id", 0)),
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=str(data.get("text", "")),
            translation=data.get("translation"),
            words=[WordTiming.from_dict(w) for w in words],
            flags=[str(f) for f in data.get("flags") or []],
        )


@dataclass
class GlossaryEntry:
    src: str
    dst: str
    count: int = 0
    confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "count": self.count,
            "confirmed": self.confirmed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GlossaryEntry":
        return cls(
            src=str(data.get("src", "")),
            dst=str(data.get("dst", "")),
            count=int(data.get("count", 0)),
            confirmed=bool(data.get("confirmed", False)),
        )


@dataclass
class SourceInfo:
    file: str = ""
    duration: float = 0.0
    language: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "duration": self.duration, "language": self.language}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceInfo":
        return cls(
            file=str(data.get("file", "")),
            duration=float(data.get("duration", 0.0)),
            language=str(data.get("language", "")),
        )


@dataclass
class ModelInfo:
    asr: str = ""
    aligner: str = ""
    translator: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"asr": self.asr, "aligner": self.aligner, "translator": self.translator}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        return cls(
            asr=str(data.get("asr", "")),
            aligner=str(data.get("aligner", "")),
            translator=str(data.get("translator", "")),
        )


@dataclass
class ProjectMeta:
    version: int = SCHEMA_VERSION
    source: SourceInfo = field(default_factory=SourceInfo)
    models: ModelInfo = field(default_factory=ModelInfo)
    summary: str = ""
    stage: Stage = Stage.EMPTY


@dataclass
class SubtitleProject:
    """一个视频的字幕工程：meta + glossary + cues。"""

    meta: ProjectMeta = field(default_factory=ProjectMeta)
    glossary: list[GlossaryEntry] = field(default_factory=list)
    cues: list[Cue] = field(default_factory=list)

    @property
    def stage(self) -> Stage:
        return self.meta.stage

    @stage.setter
    def stage(self, value: Union[Stage, str]) -> None:
        self.meta.stage = Stage.parse(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.meta.version,
            "source": self.meta.source.to_dict(),
            "models": self.meta.models.to_dict(),
            "summary": self.meta.summary,
            "glossary": [g.to_dict() for g in self.glossary],
            "cues": [c.to_dict() for c in self.cues],
            "stage": self.meta.stage.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubtitleProject":
        """容错反序列化：缺字段给默认值；version 不匹配抛 SchemaVersionError。"""
        if not isinstance(data, dict):
            raise TypeError(f"工程数据必须是 dict，得到 {type(data).__name__}")
        version = int(data.get("version", SCHEMA_VERSION))
        if version != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"工程文件 version={version}，当前支持的 SCHEMA_VERSION={SCHEMA_VERSION}"
            )
        meta = ProjectMeta(
            version=version,
            source=SourceInfo.from_dict(data.get("source") or {}),
            models=ModelInfo.from_dict(data.get("models") or {}),
            summary=str(data.get("summary", "")),
            stage=Stage.parse(data.get("stage", Stage.EMPTY.value)),
        )
        return cls(
            meta=meta,
            glossary=[GlossaryEntry.from_dict(g) for g in data.get("glossary") or []],
            cues=[Cue.from_dict(c) for c in data.get("cues") or []],
        )

    def save(self, path: Union[str, Path]) -> None:
        """保存为 UTF-8 JSON（indent=2，ensure_ascii=False）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SubtitleProject":
        path = Path(path)
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return cls.from_dict(data)
