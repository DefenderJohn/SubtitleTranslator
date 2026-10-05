"""CLI 测试：argparse 解析、config init、子命令冒烟（重活全 mock）。"""

from __future__ import annotations

import pytest

from subtitle_translator import cli, pipeline
from subtitle_translator.config import load_config
from subtitle_translator.models import Cue, GlossaryEntry, Stage, SubtitleProject


def _make_project(tmp_path, name="movie") -> tuple[SubtitleProject, object]:
    project = SubtitleProject()
    project.cues = [
        Cue(id=1, start=0.0, end=1.5, text="hello world.", translation="你好，世界。"),
        Cue(id=2, start=2.0, end=3.0, text="bye.", translation=None),
    ]
    project.glossary = [
        GlossaryEntry(src="Erebus", dst="厄瑞玻斯", count=2, confirmed=False),
        GlossaryEntry(src="Nyra", dst="妮拉", count=1, confirmed=True),
    ]
    project.stage = Stage.TRANSLATED
    json_path = tmp_path / f"{name}.sub.json"
    project.save(json_path)
    return project, json_path


class TestParser:
    def test_run_defaults(self):
        args = cli.build_parser().parse_args(["run", "video.mp4"])
        assert args.path == "video.mp4"
        assert args.config == "config.yaml"
        assert not args.transcribe_only
        assert not args.auto_confirm
        assert not args.no_bilingual
        assert not args.from_json
        assert args.language is None

    def test_run_all_flags(self):
        args = cli.build_parser().parse_args(
            [
                "run", "v.mp4",
                "--config", "c.yaml",
                "--transcribe-only",
                "--auto-confirm",
                "--no-bilingual",
                "--from-json",
                "--language", "en",
            ]
        )
        assert args.config == "c.yaml"
        assert args.transcribe_only and args.auto_confirm
        assert args.no_bilingual and args.from_json
        assert args.language == "en"

    def test_export_flags(self):
        args = cli.build_parser().parse_args(
            ["export", "a.sub.json", "--bilingual", "-o", "out.srt"]
        )
        assert args.json_path == "a.sub.json"
        assert args.bilingual
        assert args.output == "out.srt"

    def test_glossary_flags(self):
        args = cli.build_parser().parse_args(["glossary", "a.sub.json", "--confirm-all"])
        assert args.confirm_all and not args.show

    def test_config_init_default_path(self):
        args = cli.build_parser().parse_args(["config", "init"])
        assert args.path == "config.yaml"

    def test_no_subcommand_exits(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])


class TestConfigInit:
    def test_creates_default_config(self, tmp_path, capsys):
        path = tmp_path / "config.yaml"
        assert cli.main(["config", "init", str(path)]) == 0
        assert path.exists()
        cfg = load_config(path)  # 能读回
        assert cfg.asr.model
        assert cfg.translate.api_key is None  # 默认不落盘明文 key

    def test_refuses_overwrite(self, tmp_path, capsys):
        path = tmp_path / "config.yaml"
        path.write_text("asr: {}\n", encoding="utf-8")
        assert cli.main(["config", "init", str(path)]) == 1
        assert "已存在" in capsys.readouterr().err
        assert path.read_text(encoding="utf-8") == "asr: {}\n"  # 未被覆盖


class TestExport:
    def test_export_bilingual_default_output(self, tmp_path, capsys):
        _, json_path = _make_project(tmp_path)
        assert cli.main(["export", str(json_path), "--bilingual"]) == 0
        srt = tmp_path / "movie.srt"
        assert srt.exists()
        text = srt.read_text(encoding="utf-8")
        assert "你好，世界。\nhello world." in text  # 译文在上
        assert "bye." in text  # 无译文退化为原文

    def test_export_single_language_with_output(self, tmp_path):
        _, json_path = _make_project(tmp_path)
        out = tmp_path / "out.srt"
        assert cli.main(["export", str(json_path), "-o", str(out)]) == 0
        text = out.read_text(encoding="utf-8")
        assert "你好，世界。" in text and "hello world.\n\n" not in text

    def test_export_missing_json(self, tmp_path, capsys):
        assert cli.main(["export", str(tmp_path / "nope.sub.json")]) == 1
        assert "工程文件不存在" in capsys.readouterr().err


class TestGlossary:
    def test_show_by_default(self, tmp_path, capsys):
        _, json_path = _make_project(tmp_path)
        assert cli.main(["glossary", str(json_path)]) == 0
        out = capsys.readouterr().out
        assert "Erebus | 厄瑞玻斯 | 2" in out
        assert "[ ] Erebus" in out and "[✓] Nyra" in out

    def test_confirm_all(self, tmp_path, capsys):
        _, json_path = _make_project(tmp_path)
        assert cli.main(["glossary", str(json_path), "--confirm-all"]) == 0
        loaded = SubtitleProject.load(json_path)
        assert all(g.confirmed for g in loaded.glossary)
        assert "已确认全部 2 条" in capsys.readouterr().err

    def test_missing_json(self, tmp_path, capsys):
        assert cli.main(["glossary", str(tmp_path / "nope.sub.json")]) == 1
        assert "工程文件不存在" in capsys.readouterr().err


class TestRun:
    @pytest.fixture
    def mock_pipeline(self, monkeypatch, tmp_path):
        """mock 掉 pipeline 重活，记录调用参数；配好带 model 的 config.yaml。"""
        (tmp_path / "config.yaml").write_text(
            "translate:\n  model: test-model\n  api_key: k\n", encoding="utf-8"
        )
        calls = {}

        def fake_run_pipeline(path, cfg, **kwargs):
            calls["run"] = (path, cfg, kwargs)

        def fake_run_batch(path, cfg, **kwargs):
            calls["batch"] = (path, cfg, kwargs)
            return pipeline.BatchResult(succeeded=[path])

        monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(pipeline, "run_batch", fake_run_batch)
        return calls

    def test_run_single_file(self, mock_pipeline, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        media = tmp_path / "v.mp4"
        media.touch()
        assert cli.main(["run", str(media), "--language", "en"]) == 0
        path, cfg, kwargs = mock_pipeline["run"]
        assert path == media
        assert cfg.asr.language == "en"  # CLI 覆盖 yaml
        assert kwargs["stages"] == pipeline.PIPELINE_STAGES
        assert kwargs["bilingual"] is True

    def test_run_transcribe_only(self, mock_pipeline, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        media = tmp_path / "v.mp4"
        media.touch()
        assert cli.main(["run", str(media), "--transcribe-only"]) == 0
        _, _, kwargs = mock_pipeline["run"]
        assert kwargs["stages"] == ("transcribe", "export")
        assert kwargs["bilingual"] is False

    def test_run_directory_goes_batch(self, mock_pipeline, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert cli.main(["run", str(tmp_path)]) == 0
        assert "batch" in mock_pipeline and "run" not in mock_pipeline

    def test_run_from_json(self, mock_pipeline, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        json_path = tmp_path / "v.sub.json"
        json_path.write_text("{}", encoding="utf-8")
        assert cli.main(["run", str(json_path), "--from-json"]) == 0
        _, _, kwargs = mock_pipeline["run"]
        assert kwargs["from_json"] is True

    def test_missing_model_is_friendly_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)  # 无 config.yaml → 默认配置 translate.model=""
        media = tmp_path / "v.mp4"
        media.touch()
        assert cli.main(["run", str(media)]) == 1
        err = capsys.readouterr().err
        assert "translate.model 未配置" in err
        assert "config init" in err

    def test_missing_api_key_warns_but_runs(self, mock_pipeline, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yaml").write_text(
            "translate:\n  model: test-model\n", encoding="utf-8"
        )
        media = tmp_path / "v.mp4"
        media.touch()
        assert cli.main(["run", str(media)]) == 0
        assert "api_key" in capsys.readouterr().err

    def test_pipeline_error_returns_1(self, mock_pipeline, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yaml").write_text(
            "translate:\n  model: m\n  api_key: k\n", encoding="utf-8"
        )
        media = tmp_path / "v.mp4"
        media.touch()

        def boom(path, cfg, **kwargs):
            raise FileNotFoundError("路径不存在: xxx")

        monkeypatch.setattr(pipeline, "run_pipeline", boom)
        assert cli.main(["run", str(media)]) == 1
        assert "路径不存在" in capsys.readouterr().err
