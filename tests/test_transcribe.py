"""transcribe.py 测试：chunk_plan 纯函数 + fake backend/mock media 的整流程。

不碰真模型、不碰真 ffmpeg。
"""

from __future__ import annotations

import logging

import pytest

from subtitle_translator import transcribe
from subtitle_translator.config import AsrConfig, Config
from subtitle_translator.models import Stage, WordTiming
from subtitle_translator.transcribe import (
    AsrBackend,
    TransformersBackend,
    VllmBackend,
    chunk_plan,
    create_backend,
    transcribe_media,
)


class TestChunkPlan:
    def test_short_media_single_chunk(self):
        assert chunk_plan(10.0, [], 290.0) == [(0.0, 10.0)]

    def test_zero_duration(self):
        assert chunk_plan(0.0, [(1.0, 2.0)], 290.0) == []

    def test_evenly_distributed_silences(self):
        silences = [(28.0, 29.0), (58.0, 59.0), (88.0, 89.0)]
        chunks = chunk_plan(100.0, silences, 30.0)
        assert chunks == [(0.0, 28.5), (28.5, 58.5), (58.5, 88.5), (88.5, 100.0)]

    def test_prefers_midpoint_closest_to_target(self):
        # 目标切点 30：静音中点 10 与 29 之间应选 29
        silences = [(9.9, 10.1), (28.9, 29.1)]
        chunks = chunk_plan(60.0, silences, 30.0)
        assert chunks[0] == (0.0, 29.0)

    def test_no_silence_hard_cut_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="subtitle_translator.transcribe"):
            chunks = chunk_plan(100.0, [], 30.0)
        assert chunks == [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 100.0)]
        assert sum("硬切" in r.message for r in caplog.records) == 3

    def test_leading_and_trailing_silence(self):
        silences = [(0.0, 1.0), (99.0, 100.0)]
        chunks = chunk_plan(100.0, silences, 30.0)
        # 唯一可用的静音中点 0.5 落在第一刀
        assert chunks[0] == (0.0, 0.5)
        # 末尾静音中点 99.5 用不上：最后一块必然 ≤ max_seconds
        assert chunks[-1][1] == 100.0
        assert all(e - s <= 30.0 + 1e-9 for s, e in chunks)

    def test_full_coverage_and_max_constraint(self):
        silences = [(float(i), float(i) + 0.4) for i in range(13, 600, 37)]
        for duration, max_s in [(600.0, 290.0), (97.3, 20.0), (50.0, 7.5)]:
            chunks = chunk_plan(duration, silences, max_s)
            assert chunks[0][0] == 0.0
            assert chunks[-1][1] == duration
            for (s1, e1), (s2, _) in zip(chunks, chunks[1:]):
                assert e1 == s2  # 无缝覆盖
            assert all(e - s <= max_s + 1e-9 for s, e in chunks)


class TestBackends:
    def test_create_backend(self):
        assert isinstance(create_backend(AsrConfig(backend="vllm")), VllmBackend)
        assert isinstance(
            create_backend(AsrConfig(backend="transformers")), TransformersBackend
        )
        assert isinstance(create_backend(Config()), VllmBackend)

    def test_create_backend_invalid(self):
        # 绕过 __post_init__ 校验塞入非法 backend
        cfg = AsrConfig()
        object.__setattr__(cfg, "backend", "bogus")
        with pytest.raises(ValueError, match="非法"):
            create_backend(cfg)

    def test_load_without_qwen_asr_gives_clear_error(self):
        with pytest.raises(RuntimeError, match="qwen-asr"):
            VllmBackend(AsrConfig()).load()
        with pytest.raises(RuntimeError, match="qwen-asr"):
            TransformersBackend(AsrConfig()).load()

    def test_transcribe_before_load_raises(self):
        with pytest.raises(RuntimeError, match="load"):
            VllmBackend(AsrConfig()).transcribe_chunk("x.wav")

    def test_module_imports_without_qwen_asr(self):
        # 本环境未安装 qwen-asr；能走到这里即证明模块可 import
        assert transcribe.AsrBackend is AsrBackend


class TestExtractResult:
    def test_attr_style(self):
        class R:
            text = "你好"
            time_stamps = [WordTiming(text="你", start=0.0, end=0.3)]

        text, words = transcribe._extract_result(R())
        assert text == "你好"
        assert words == [WordTiming(text="你", start=0.0, end=0.3)]

    def test_dict_style(self):
        text, words = transcribe._extract_result(
            {"text": "hi", "words": [{"text": "hi", "start": 1, "end": 2}]}
        )
        assert text == "hi"
        assert words == [WordTiming(text="hi", start=1.0, end=2.0)]


class FakeBackend(AsrBackend):
    """固定返回两个词的假 backend，记录调用参数。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.loaded = False
        self.unloaded = False
        self.calls: list[dict] = []

    def load(self):
        self.loaded = True
        self._model = object()

    def unload(self):
        self.unloaded = True
        self._model = None

    def transcribe_chunk(self, audio_path, language=None):
        self.calls.append({"audio": str(audio_path), "language": language})
        return "hello world.", [
            WordTiming(text="hello", start=0.1, end=0.6),
            WordTiming(text="world.", start=0.7, end=1.2),
        ]


@pytest.fixture
def mocked_media(monkeypatch, tmp_path):
    """把 media 层的 ffmpeg 调用全部换成假的。"""
    calls = {"extract": []}
    monkeypatch.setattr(transcribe.media, "find_ffmpeg", lambda configured=None: "fake-ffmpeg")
    monkeypatch.setattr(transcribe.media, "probe_duration", lambda path, ffmpeg=None: 10.0)
    monkeypatch.setattr(
        transcribe.media, "detect_silences", lambda path, ffmpeg=None: [(4.9, 5.1)]
    )

    def fake_extract(src, start, end, dst_wav, ffmpeg=None):
        calls["extract"].append((str(src), start, end))

    monkeypatch.setattr(transcribe.media, "extract_audio_chunk", fake_extract)
    return calls


class TestTranscribeMedia:
    def _cfg(self) -> Config:
        cfg = Config()
        cfg.asr.chunk_max_seconds = 6.0
        cfg.asr.language = "English"
        return cfg

    def test_offsets_and_project(self, mocked_media, tmp_path):
        cfg = self._cfg()
        backend = FakeBackend(cfg.asr)
        backend.load()
        progress = []
        project = transcribe_media(
            tmp_path / "movie.mp4",
            cfg,
            backend=backend,
            progress_cb=lambda done, total: progress.append((done, total)),
        )

        # 切块：静音中点 5.0 下刀 -> [(0,5),(5,10)]
        assert mocked_media["extract"] == [
            (str(tmp_path / "movie.mp4"), 0.0, 5.0),
            (str(tmp_path / "movie.mp4"), 5.0, 10.0),
        ]
        # 词级时间戳带上了块偏移量
        assert project.cues[0].text == "hello world."
        assert project.cues[0].start == pytest.approx(0.1)
        assert project.cues[0].end == pytest.approx(1.2)
        assert project.cues[1].start == pytest.approx(5.1)
        assert project.cues[1].end == pytest.approx(6.2)
        assert project.cues[1].words[0].start == pytest.approx(5.1)

        assert project.stage == Stage.TRANSCRIBED
        assert project.meta.source.duration == 10.0
        assert project.meta.source.language == "English"
        assert project.meta.models.asr == cfg.asr.model
        assert project.meta.models.aligner == cfg.asr.aligner_model

        # language 透传给 backend；进度回调每块一次
        assert [c["language"] for c in backend.calls] == ["English", "English"]
        assert progress == [(1, 2), (2, 2)]
        # 外部传入的 backend 不被 unload
        assert not backend.unloaded

    def test_own_backend_lifecycle(self, mocked_media, monkeypatch, tmp_path):
        cfg = self._cfg()
        fake = FakeBackend(cfg.asr)
        monkeypatch.setattr(transcribe, "create_backend", lambda c: fake)
        project = transcribe_media(tmp_path / "m.mp4", cfg)
        assert fake.loaded and fake.unloaded
        assert project.stage == Stage.TRANSCRIBED

    def test_roundtrip_save_load(self, mocked_media, tmp_path):
        cfg = self._cfg()
        backend = FakeBackend(cfg.asr)
        backend.load()
        project = transcribe_media(tmp_path / "m.mp4", cfg, backend=backend)
        out = tmp_path / "m.json"
        project.save(out)
        from subtitle_translator.models import SubtitleProject

        loaded = SubtitleProject.load(out)
        assert loaded.stage == Stage.TRANSCRIBED
        assert [c.text for c in loaded.cues] == ["hello world.", "hello world."]
        assert loaded.cues[1].words[0].start == pytest.approx(5.1)
