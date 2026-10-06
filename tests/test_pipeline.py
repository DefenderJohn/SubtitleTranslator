"""pipeline 测试：FakeBackend + fake chat client 端到端，media 层全 mock。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from subtitle_translator import pipeline, transcribe
from subtitle_translator.config import Config
from subtitle_translator.models import Cue, GlossaryEntry, Stage, SubtitleProject
from subtitle_translator.pipeline import (
    PIPELINE_STAGES,
    PipelineCancelledError,
    default_srt_path,
    find_media_files,
    project_json_path,
    run_batch,
    run_pipeline,
)
from subtitle_translator.srt import parse_srt
from subtitle_translator.transcribe import AsrBackend
from subtitle_translator.translate import GlossaryNotConfirmedError, SlidingWindowStrategy


class FakeBackend(AsrBackend):
    """固定返回两条词的假 ASR backend。"""

    def load(self):
        self._model = object()

    def transcribe_chunk(self, audio_path, language=None):
        from subtitle_translator.models import WordTiming

        return "hello world.", [
            WordTiming(text="hello", start=0.1, end=0.6),
            WordTiming(text="world.", start=0.7, end=1.2),
        ]


class FakeChatClient:
    """按 schema name 路由的 fake chat_json 客户端。"""

    def __init__(self, translation="默认译文"):
        self.calls: list[list[dict]] = []
        self.translation = translation

    def chat_json(self, messages, *, schema, name, temperature=None):
        self.calls.append([dict(m) for m in messages])
        if name == "summary":
            return {"summary": "测试摘要"}
        if name == "glossary":
            return {"entries": [{"src": "Erebus", "dst": "厄瑞玻斯", "count": 2}]}
        return {"translation": self.translation}


@pytest.fixture
def cfg() -> Config:
    cfg = Config()
    cfg.asr.chunk_max_seconds = 300.0
    cfg.translate.model = "test-model"
    cfg.translate.additional_prompt = ""
    return cfg


@pytest.fixture
def mocked_media(monkeypatch):
    """把 media 层的 ffmpeg 调用全部换成假的。"""
    monkeypatch.setattr(transcribe.media, "find_ffmpeg", lambda configured=None: "fake")
    monkeypatch.setattr(transcribe.media, "probe_duration", lambda path, ffmpeg=None: 10.0)
    monkeypatch.setattr(transcribe.media, "detect_silences", lambda path, ffmpeg=None: [])
    monkeypatch.setattr(
        transcribe.media, "extract_audio_chunk", lambda *a, **k: None
    )


def _fake_strategy(client=None) -> SlidingWindowStrategy:
    return SlidingWindowStrategy(client=client or FakeChatClient())


def _transcribed_project(tmp_path, name="movie.mp4") -> SubtitleProject:
    """造一个 stage=TRANSCRIBED 的工程。"""
    project = SubtitleProject()
    project.meta.source.file = str(tmp_path / name)
    project.meta.source.duration = 10.0
    project.cues = [
        Cue(id=1, start=0.1, end=1.2, text="hello world."),
        Cue(id=2, start=2.0, end=3.0, text="second line."),
    ]
    project.stage = Stage.TRANSCRIBED
    return project


class TestFindMediaFiles:
    def test_single_file_ok(self, tmp_path):
        f = tmp_path / "a.mkv"
        f.touch()
        assert find_media_files(f) == [f]

    def test_single_file_bad_ext(self, tmp_path):
        f = tmp_path / "a.txt"
        f.touch()
        with pytest.raises(ValueError, match="不支持的媒体格式"):
            find_media_files(f)

    def test_directory_recursive(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "b.mp4").touch()
        (tmp_path / "sub" / "a.wav").touch()
        (tmp_path / "sub" / "c.txt").touch()
        found = find_media_files(tmp_path)
        assert found == sorted([tmp_path / "b.mp4", tmp_path / "sub" / "a.wav"])

    def test_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_media_files(tmp_path / "nope")

    def test_unreadable_entries_skipped(self, tmp_path, monkeypatch):
        """遍历中遇到无权限/不可读的条目（stat 失败）跳过继续，不抛异常。"""
        (tmp_path / "good.mp4").touch()
        bad = tmp_path / "secret"
        bad.mkdir()
        (bad / "hidden.mp4").touch()

        real_is_file = Path.is_file

        def fake_is_file(self):
            if str(self).startswith(str(bad)):
                raise PermissionError(13, "Permission denied", str(self))
            return real_is_file(self)

        monkeypatch.setattr(Path, "is_file", fake_is_file)
        assert find_media_files(tmp_path) == [tmp_path / "good.mp4"]

    def test_unreadable_subdir_skipped(self, tmp_path, monkeypatch):
        """下降进入无权限子目录（scandir 抛错）时跳过整棵子树，不抛异常。"""
        (tmp_path / "good.mp4").touch()
        bad = tmp_path / "secret"
        bad.mkdir()
        (bad / "hidden.mp4").touch()

        real_scandir = os.scandir

        def fake_scandir(p):
            if str(p) == str(bad):
                raise PermissionError(13, "Permission denied", str(p))
            return real_scandir(p)

        monkeypatch.setattr(os, "scandir", fake_scandir)
        assert find_media_files(tmp_path) == [tmp_path / "good.mp4"]


class TestPaths:
    def test_project_json_path(self, tmp_path):
        assert project_json_path(tmp_path / "movie.mp4") == tmp_path / "movie.sub.json"

    def test_default_srt_path_media(self, tmp_path):
        assert default_srt_path(tmp_path / "movie.mp4") == tmp_path / "movie.srt"

    def test_default_srt_path_from_json(self, tmp_path):
        assert (
            default_srt_path(tmp_path / "movie.sub.json", from_json=True)
            == tmp_path / "movie.srt"
        )


class TestRunPipeline:
    def test_full_run(self, mocked_media, cfg, tmp_path):
        media = tmp_path / "movie.mp4"
        media.touch()
        backend = FakeBackend(cfg.asr)
        backend.load()
        events = []
        project = run_pipeline(
            media,
            cfg,
            auto_confirm=True,
            backend=backend,
            strategy=_fake_strategy(),
            progress_cb=events.append,
        )
        assert project.stage == Stage.TRANSLATED
        assert project.meta.summary == "测试摘要"
        assert all(g.confirmed for g in project.glossary)
        assert all(c.translation == "默认译文" for c in project.cues)

        # JSON 落盘（断点续传的事实来源）
        json_path = tmp_path / "movie.sub.json"
        assert json_path.exists()
        loaded = SubtitleProject.load(json_path)
        assert loaded.stage == Stage.TRANSLATED

        # 默认导出双语 SRT：译文在上、原文在下
        srt_path = tmp_path / "movie.srt"
        assert srt_path.exists()
        text = srt_path.read_text(encoding="utf-8")
        assert "00:00:00,100 --> 00:00:01,200" in text
        assert "默认译文\nhello world." in text
        # 导出的 SRT 能被 parse_srt 读回
        assert len(parse_srt(srt_path)) == 1

        # event 协议：键齐全、stage 合法、翻译进度推进到 total
        for e in events:
            assert set(e) == {"media", "stage", "done", "total", "message"}
            assert e["stage"] in PIPELINE_STAGES
            assert e["media"] == str(media)
        translate_progress = [
            (e["done"], e["total"]) for e in events if e["stage"] == "translate" and e["total"]
        ]
        assert translate_progress[-1] == (1, 1)

    def test_resume_skips_transcribe(self, mocked_media, cfg, tmp_path, monkeypatch):
        # 预置 stage=TRANSCRIBED 的工程 JSON
        _transcribed_project(tmp_path).save(tmp_path / "movie.sub.json")
        media = tmp_path / "movie.mp4"
        media.touch()
        # 转录一旦被调用就炸，证明它被跳过
        monkeypatch.setattr(
            pipeline,
            "transcribe_media",
            lambda *a, **k: pytest.fail("断点续传不应重新转录"),
        )
        project = run_pipeline(media, cfg, auto_confirm=True, strategy=_fake_strategy())
        assert project.stage == Stage.TRANSLATED
        assert [c.text for c in project.cues] == ["hello world.", "second line."]
        assert all(c.translation == "默认译文" for c in project.cues)

    def test_transcribe_only(self, mocked_media, cfg, tmp_path):
        media = tmp_path / "movie.mp4"
        media.touch()
        backend = FakeBackend(cfg.asr)
        backend.load()
        client = FakeChatClient()
        project = run_pipeline(
            media,
            cfg,
            stages=("transcribe", "export"),
            bilingual=False,
            backend=backend,
            strategy=_fake_strategy(client),
        )
        assert project.stage == Stage.TRANSCRIBED
        assert client.calls == []  # 没有碰翻译端点
        srt = (tmp_path / "movie.srt").read_text(encoding="utf-8")
        assert "hello world." in srt
        assert "默认译文" not in srt

    def test_glossary_checkpoint_saves_context(self, mocked_media, cfg, tmp_path):
        """不 auto_confirm：检查点抛错但 context 产物已落盘；确认后重跑续上。"""
        media = tmp_path / "movie.mp4"
        media.touch()
        backend = FakeBackend(cfg.asr)
        backend.load()
        with pytest.raises(GlossaryNotConfirmedError):
            run_pipeline(media, cfg, backend=backend, strategy=_fake_strategy())

        # 崩溃/中断后摘要 + 术语表已保存
        loaded = SubtitleProject.load(tmp_path / "movie.sub.json")
        assert loaded.stage == Stage.CONTEXTED
        assert loaded.meta.summary == "测试摘要"
        assert [g.src for g in loaded.glossary] == ["Erebus"]
        assert not loaded.glossary[0].confirmed

        # 人工确认后重跑：跳过摘要/术语表，直接逐句翻译
        loaded.glossary[0].confirmed = True
        loaded.save(tmp_path / "movie.sub.json")
        client = FakeChatClient()
        project = run_pipeline(media, cfg, strategy=_fake_strategy(client))
        assert project.stage == Stage.TRANSLATED
        # 只剩逐句翻译调用（无摘要/术语表调用）
        systems = [c[0]["content"] for c in client.calls]
        assert not any("内容摘要" in s or "术语管理助手" in s for s in systems)

    def test_from_json(self, cfg, tmp_path, monkeypatch):
        """--from-json 模式：不碰媒体层，直接对已有 JSON 跑翻译+导出。"""
        json_path = tmp_path / "movie.sub.json"
        _transcribed_project(tmp_path).save(json_path)
        monkeypatch.setattr(
            transcribe.media,
            "probe_duration",
            lambda *a, **k: pytest.fail("from-json 不应触碰媒体文件"),
        )
        project = run_pipeline(
            json_path, cfg, auto_confirm=True, from_json=True, strategy=_fake_strategy()
        )
        assert project.stage == Stage.TRANSLATED
        assert (tmp_path / "movie.srt").exists()

    def test_from_json_missing_file(self, cfg, tmp_path):
        with pytest.raises(FileNotFoundError, match="工程文件不存在"):
            run_pipeline(tmp_path / "nope.sub.json", cfg, from_json=True)

    def test_unknown_stage(self, cfg, tmp_path):
        with pytest.raises(ValueError, match="未知阶段"):
            run_pipeline(tmp_path / "m.mp4", cfg, stages=("bogus",))


class TestRunBatch:
    def test_failure_does_not_stop_batch(self, mocked_media, cfg, tmp_path, monkeypatch):
        good = tmp_path / "good.mp4"
        bad = tmp_path / "bad.mp4"
        good.touch()
        bad.touch()

        real_transcribe = pipeline.transcribe_media

        def flaky(path, *a, **k):
            if Path(path).name == "bad.mp4":
                raise RuntimeError("磁盘爆炸")
            return real_transcribe(path, *a, **k)

        monkeypatch.setattr(pipeline, "transcribe_media", flaky)
        backend = FakeBackend(cfg.asr)
        backend.load()
        result = run_batch(
            tmp_path,
            cfg,
            auto_confirm=True,
            backend=backend,
            strategy=_fake_strategy(),
        )
        assert result.succeeded == [good]
        assert len(result.failed) == 1
        assert result.failed[0][0] == bad
        assert "磁盘爆炸" in result.failed[0][1]
        # 好文件完整走完
        assert SubtitleProject.load(tmp_path / "good.sub.json").stage == Stage.TRANSLATED
        assert (tmp_path / "good.srt").exists()

    def test_empty_directory(self, cfg, tmp_path):
        result = run_batch(tmp_path, cfg)
        assert result.total == 0


class TestCancellation:
    """协作式取消：progress_cb 抛 PipelineCancelledError。"""

    def test_translate_cancel_saves_progress(self, mocked_media, cfg, tmp_path):
        """translate 阶段被取消：已翻译的 cue 落盘（stage 保持 contexted），重跑续上。"""
        project = _transcribed_project(tmp_path)
        project.cues.append(Cue(id=3, start=4.0, end=5.0, text="third line."))
        project.save(tmp_path / "movie.sub.json")
        media = tmp_path / "movie.mp4"
        media.touch()

        def cancel_at_second_cue(event):
            if event["stage"] == "translate" and event["done"] == 2:
                raise PipelineCancelledError("取消")

        with pytest.raises(PipelineCancelledError):
            run_pipeline(
                media,
                cfg,
                auto_confirm=True,
                strategy=_fake_strategy(),
                progress_cb=cancel_at_second_cue,
            )
        loaded = SubtitleProject.load(tmp_path / "movie.sub.json")
        assert loaded.stage == Stage.CONTEXTED  # 不谎报 translated
        assert [c.translation for c in loaded.cues] == ["默认译文", "默认译文", None]

        # 重跑断点续传：已翻译的跳过，只补第三条
        client = FakeChatClient()
        project = run_pipeline(media, cfg, auto_confirm=True, strategy=_fake_strategy(client))
        assert project.stage == Stage.TRANSLATED
        assert all(c.translation == "默认译文" for c in project.cues)
        assert len(client.calls) == 1

    def test_batch_cancel_stops(self, mocked_media, cfg, tmp_path):
        (tmp_path / "a.mp4").touch()
        (tmp_path / "b.mp4").touch()
        backend = FakeBackend(cfg.asr)
        backend.load()

        def cancel_immediately(event):
            raise PipelineCancelledError("取消")

        with pytest.raises(PipelineCancelledError):
            run_batch(
                tmp_path,
                cfg,
                auto_confirm=True,
                backend=backend,
                strategy=_fake_strategy(),
                progress_cb=cancel_immediately,
            )
        # 取消即停：第一个文件都没开始转录
        assert not (tmp_path / "a.sub.json").exists()
