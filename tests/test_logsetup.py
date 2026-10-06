"""logsetup 测试：脱敏过滤器、统一配置、per-task 日志隔离、log API 端点。

server 集成部分复用 test_server 的 fake pipeline 思路（FakeBackend + mock media），
不碰真模型与 ffmpeg。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from subtitle_translator import logsetup, pipeline, transcribe
from subtitle_translator.config import Config, save_config
from subtitle_translator.logsetup import (
    SanitizeFilter,
    create_task_log_handler,
    reset_task_log_context,
    setup_logging,
    shutdown_logging,
    task_log_context,
)
from subtitle_translator.server import create_app
from subtitle_translator.server import tasks as server_tasks
from subtitle_translator.transcribe import AsrBackend


@pytest.fixture(autouse=True)
def _clean_logging():
    """每个测试前后摘除 logsetup 安装的 root handler，避免互相污染。"""
    shutdown_logging()
    yield
    shutdown_logging()


# ---------------------------------------------------------------------------
# SanitizeFilter
# ---------------------------------------------------------------------------


class TestSanitizeFilter:
    def test_registered_secret_exact_replaced(self):
        f = SanitizeFilter()
        f.register_secret("sk-live-abcdef1234567890")
        assert f.sanitize("key 是 sk-live-abcdef1234567890 别外传") == "key 是 *** 别外传"

    def test_short_secret_not_registered(self):
        f = SanitizeFilter()
        f.register_secret("0")  # 占位符：精确替换会毁掉正常日志
        f.register_secret(None)
        assert f.sanitize("版本 0 到 1 升级") == "版本 0 到 1 升级"

    def test_sk_pattern(self):
        f = SanitizeFilter()
        assert f.sanitize("Authorization sk-a1b2c3d4e5f6") == "Authorization sk-***"

    def test_bearer_pattern(self):
        f = SanitizeFilter()
        assert f.sanitize("header: Bearer eyJhbGciOi.token") == "header: Bearer ***"

    def test_api_key_pattern(self):
        f = SanitizeFilter()
        assert f.sanitize("请求 api_key=secretvalue123 失败") == "请求 api_key=*** 失败"

    def test_url_query_key_token(self):
        f = SanitizeFilter()
        assert (
            f.sanitize("GET https://x.test/v1?model=a&token=tok123456&key=k987654321")
            == "GET https://x.test/v1?model=a&token=***&key=***"
        )

    def test_home_path_masked(self):
        f = SanitizeFilter()
        home = str(Path.home())
        assert f.sanitize(f"处理 {home}/videos/x.mp4 完成") == "处理 ~/videos/x.mp4 完成"

    def test_filter_rewrites_record(self):
        f = SanitizeFilter()
        f.register_secret("topsecretvalue")
        record = logging.LogRecord(
            "t", logging.INFO, __file__, 1, "key=%s 路径 %s", ("topsecretvalue", "x"), None
        )
        assert f.filter(record) is True
        assert record.getMessage() == "key=*** 路径 x"

    def test_key_value_pattern_keeps_following_params(self):
        f = SanitizeFilter()
        assert f.sanitize("api_key=abc123456&page=2") == "api_key=***&page=2"


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_creates_console_and_rotating_file(self, tmp_path):
        cfg = Config()
        cfg.log.dir = str(tmp_path / "logs")
        log_file = setup_logging(cfg)
        assert log_file == tmp_path / "logs" / "subtitle-translator.log"
        logging.getLogger("test.setup").info("落盘检查")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert "落盘检查" in log_file.read_text(encoding="utf-8")

    def test_idempotent_no_duplicate_handlers(self, tmp_path):
        cfg = Config()
        cfg.log.dir = str(tmp_path / "logs")
        setup_logging(cfg)
        before = len(logging.getLogger().handlers)
        setup_logging(cfg)
        assert len(logging.getLogger().handlers) == before

    def test_console_level_from_config(self, tmp_path):
        cfg = Config()
        cfg.log.dir = str(tmp_path / "logs")
        cfg.log.level = "WARNING"
        setup_logging(cfg)
        root = logging.getLogger()
        console = next(
            h
            for h in root.handlers
            if getattr(h, "_subtitle_translator_handler", False)
            and isinstance(h, logging.StreamHandler)
            and not hasattr(h, "maxBytes")
        )
        assert console.level == logging.WARNING

    def test_file_log_sanitized(self, tmp_path):
        cfg = Config()
        cfg.log.dir = str(tmp_path / "logs")
        log_file = setup_logging(cfg)
        logsetup.register_secret("sk-filetest-secret-99")
        logging.getLogger("test.sanitize").info("连接失败 key=sk-filetest-secret-99")
        for handler in logging.getLogger().handlers:
            handler.flush()
        content = log_file.read_text(encoding="utf-8")
        assert "sk-filetest-secret-99" not in content
        assert "***" in content


# ---------------------------------------------------------------------------
# per-task 日志隔离（handler 层）
# ---------------------------------------------------------------------------


class TestTaskLogHandler:
    def test_isolation_by_task_context(self, tmp_path):
        root = logging.getLogger()
        h1 = create_task_log_handler(tmp_path / "tasks" / "t1.log", "t1")
        h2 = create_task_log_handler(tmp_path / "tasks" / "t2.log", "t2")
        root.addHandler(h1)
        root.addHandler(h2)
        try:
            log = logging.getLogger("test.tasklog")
            token = task_log_context("t1")
            try:
                log.info("任务一的记录")
            finally:
                reset_task_log_context(token)
            token = task_log_context("t2")
            try:
                log.info("任务二的记录")
            finally:
                reset_task_log_context(token)
            log.info("不属于任何任务的记录")
        finally:
            root.removeHandler(h1)
            root.removeHandler(h2)
            h1.close()
            h2.close()
        t1 = (tmp_path / "tasks" / "t1.log").read_text(encoding="utf-8")
        t2 = (tmp_path / "tasks" / "t2.log").read_text(encoding="utf-8")
        assert "任务一的记录" in t1 and "任务二的记录" not in t1
        assert "任务二的记录" in t2 and "任务一的记录" not in t2
        assert "不属于任何任务的记录" not in t1 and "不属于任何任务的记录" not in t2


# ---------------------------------------------------------------------------
# server 集成：per-task 日志文件 + GET /api/tasks/{id}/log
# ---------------------------------------------------------------------------


class FakeBackend(AsrBackend):
    def load(self):
        self._model = object()

    def transcribe_chunk(self, audio_path, language=None):
        from subtitle_translator.models import WordTiming

        return "hello.", [WordTiming(text="hello.", start=0.1, end=0.6)]


@pytest.fixture
def config_path(tmp_path) -> Path:
    cfg = Config()
    cfg.log.dir = str(tmp_path / "logs")
    cfg.ui.upload_dir = str(tmp_path / "uploads")
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    return path


def _wait_status(client: TestClient, task_id: str, states, timeout=10.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/tasks/{task_id}").json()["status"]
        if status in states:
            return status
        time.sleep(0.05)
    raise AssertionError(f"任务 {task_id} 未在 {timeout}s 内进入 {states}")


class TestTaskLogApi:
    def test_success_task_log_endpoint(self, config_path, tmp_path, monkeypatch):
        monkeypatch.setattr(transcribe.media, "find_ffmpeg", lambda configured=None: "fake")
        monkeypatch.setattr(transcribe.media, "probe_duration", lambda path, ffmpeg=None: 10.0)
        monkeypatch.setattr(transcribe.media, "detect_silences", lambda path, ffmpeg=None: [])
        monkeypatch.setattr(transcribe.media, "extract_audio_chunk", lambda *a, **k: None)

        def _run(media_path, cfg, **kwargs):
            backend = FakeBackend(cfg.asr)
            backend.load()
            kwargs.setdefault("backend", backend)
            return pipeline.run_pipeline(media_path, cfg, **kwargs)

        monkeypatch.setattr(server_tasks, "run_pipeline", _run)

        media = tmp_path / "a.mp4"
        media.write_bytes(b"x")
        with TestClient(create_app(config_path=config_path)) as client:
            task_id = client.post(
                "/api/tasks", json={"path": str(media), "options": {"transcribe_only": True}}
            ).json()["id"]
            _wait_status(client, task_id, ("done", "failed"))
            assert client.get(f"/api/tasks/{task_id}").json()["status"] == "done"

            resp = client.get(f"/api/tasks/{task_id}/log")
            assert resp.status_code == 200
            assert f"任务 {task_id} 开始" in resp.text
            assert f"任务 {task_id} 完成" in resp.text
            assert resp.headers["X-Log-Truncated"] == "0"

            # 日志文件落在 log.dir/tasks/ 下
            log_file = tmp_path / "logs" / "tasks" / f"{task_id}.log"
            assert log_file.is_file()

    def test_failed_task_log_contains_traceback(self, config_path, tmp_path, monkeypatch):
        def _boom(media_path, cfg, **kwargs):
            raise RuntimeError("模拟流水线崩溃")

        monkeypatch.setattr(server_tasks, "run_pipeline", _boom)
        media = tmp_path / "b.mp4"
        media.write_bytes(b"x")
        with TestClient(create_app(config_path=config_path)) as client:
            task_id = client.post(
                "/api/tasks", json={"path": str(media), "options": {"transcribe_only": True}}
            ).json()["id"]
            _wait_status(client, task_id, ("failed",))
            resp = client.get(f"/api/tasks/{task_id}/log")
            assert resp.status_code == 200
            assert "Traceback" in resp.text
            assert "模拟流水线崩溃" in resp.text

    def test_log_404_for_pending_or_unknown(self, config_path, tmp_path, monkeypatch):
        # 第一个任务堵住 worker，第二个保持 pending → 无日志文件
        gate = threading.Event()

        def _slow(media_path, cfg, **kwargs):
            gate.wait(5)
            raise RuntimeError("不用真跑完")

        monkeypatch.setattr(server_tasks, "run_pipeline", _slow)
        media = tmp_path / "c.mp4"
        media.write_bytes(b"x")
        with TestClient(create_app(config_path=config_path)) as client:
            first = client.post(
                "/api/tasks", json={"path": str(media), "options": {"transcribe_only": True}}
            ).json()["id"]
            _wait_status(client, first, ("running",))
            second = client.post(
                "/api/tasks", json={"path": str(media), "options": {"transcribe_only": True}}
            ).json()["id"]
            assert client.get(f"/api/tasks/{second}/log").status_code == 404
            assert client.get("/api/tasks/no-such-id/log").status_code == 404
            gate.set()
