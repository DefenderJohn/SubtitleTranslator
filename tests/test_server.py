"""server 层测试：FastAPI TestClient + fake backend / fake chat，不碰真模型与 ffmpeg。

- 业务流走真 pipeline（FakeBackend + FakeChatClient 注入，media 层 mock）；
- SSE 用两种姿势断言：终态任务的历史回放（HTTP 层）、manager.subscribe
  内部队列接口（实时推送）；
- 取消 / waiting_confirm→resume 用慢速 fake pipeline 与真实检查点流程覆盖。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from subtitle_translator import pipeline, transcribe
from subtitle_translator.config import save_config, Config
from subtitle_translator.models import Cue, GlossaryEntry, Stage, SubtitleProject
from subtitle_translator.pipeline import PROJECT_SUFFIX, BatchResult, find_media_files
from subtitle_translator.server import app as server_app
from subtitle_translator.server import create_app, tasks as server_tasks
from subtitle_translator.transcribe import AsrBackend
from subtitle_translator.translate import SlidingWindowStrategy


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
    """按 system prompt 内容路由的 fake chat 客户端。"""

    def __init__(self, translation="默认译文"):
        self.translation = translation

    def chat(self, messages, temperature=None):
        system = messages[0]["content"]
        if "内容摘要" in system or "部分内容" in system or "合并为一份全片摘要" in system:
            return "测试摘要"
        if "术语管理助手" in system:
            return "Erebus | 厄瑞玻斯 | 2"
        return self.translation


@pytest.fixture
def config_path(tmp_path) -> Path:
    cfg = Config()
    cfg.translate.model = "test-model"
    cfg.translate.additional_prompt = ""
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    return path


@pytest.fixture
def mocked_media(monkeypatch):
    """把 media 层的 ffmpeg 调用全部换成假的。"""
    monkeypatch.setattr(transcribe.media, "find_ffmpeg", lambda configured=None: "fake")
    monkeypatch.setattr(transcribe.media, "probe_duration", lambda path, ffmpeg=None: 10.0)
    monkeypatch.setattr(transcribe.media, "detect_silences", lambda path, ffmpeg=None: [])
    monkeypatch.setattr(transcribe.media, "extract_audio_chunk", lambda *a, **k: None)


@pytest.fixture
def fake_pipeline(monkeypatch, mocked_media):
    """server.tasks 里的 run_pipeline / run_batch 换成注入 fake backend 的真 pipeline。"""

    def _run(media_path, cfg, **kwargs):
        backend = FakeBackend(cfg.asr)
        backend.load()
        kwargs.setdefault("backend", backend)
        kwargs.setdefault("strategy", SlidingWindowStrategy(client=FakeChatClient()))
        return pipeline.run_pipeline(media_path, cfg, **kwargs)

    def _batch(path, cfg, **kwargs):
        result = BatchResult()
        for media in find_media_files(path):
            try:
                _run(media, cfg, **kwargs)
            except Exception as exc:  # noqa: BLE001
                result.failed.append((media, str(exc)))
            else:
                result.succeeded.append(media)
        return result

    monkeypatch.setattr(server_tasks, "run_pipeline", _run)
    monkeypatch.setattr(server_tasks, "run_batch", _batch)


@pytest.fixture
def client(config_path, fake_pipeline):
    with TestClient(create_app(config_path=config_path)) as c:
        yield c


def _wait_status(client: TestClient, task_id: str, states, timeout=10.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/tasks/{task_id}").json()["status"]
        if status in states:
            return status
        time.sleep(0.05)
    raise AssertionError(f"任务 {task_id} 未在 {timeout}s 内进入 {states}")


def _sse_events(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


class TestTaskLifecycle:
    def test_create_and_run_to_done(self, client, tmp_path):
        media = tmp_path / "movie.mp4"
        media.touch()
        resp = client.post("/api/tasks", json={"path": str(media), "options": {"auto_confirm": True}})
        assert resp.status_code == 201
        task_id = resp.json()["id"]
        assert resp.json()["status"] == "pending"

        assert _wait_status(client, task_id, {"done"}) == "done"
        detail = client.get(f"/api/tasks/{task_id}").json()
        assert detail["media"] == [str(media)]
        assert detail["progress"]["status"] == "done"

        # 真 pipeline 的产物：JSON + 双语 SRT
        project = SubtitleProject.load(tmp_path / "movie.sub.json")
        assert project.stage == Stage.TRANSLATED
        assert (tmp_path / "movie.srt").exists()

        # 事件流遵循 pipeline 协议 + task_id/status
        keys = {"media", "stage", "done", "total", "message", "task_id", "status"}
        for event in client.app.state.manager.get(task_id).events:
            assert set(event) == keys

        # 列表接口包含该任务
        listing = client.get("/api/tasks").json()
        assert [t["id"] for t in listing] == [task_id]

    def test_directory_task(self, client, tmp_path):
        (tmp_path / "a.mp4").touch()
        (tmp_path / "b.mkv").touch()
        resp = client.post("/api/tasks", json={"path": str(tmp_path), "options": {"auto_confirm": True}})
        assert resp.status_code == 201
        task_id = resp.json()["id"]
        assert len(resp.json()["media"]) == 2
        _wait_status(client, task_id, {"done"})
        assert (tmp_path / "a.srt").exists() and (tmp_path / "b.srt").exists()

    def test_invalid_path(self, client, tmp_path):
        assert client.post("/api/tasks", json={"path": str(tmp_path / "nope")}).status_code == 404
        bad = tmp_path / "a.txt"
        bad.touch()
        assert client.post("/api/tasks", json={"path": str(bad)}).status_code == 400
        empty = tmp_path / "empty"
        empty.mkdir()
        assert client.post("/api/tasks", json={"path": str(empty)}).status_code == 400

    def test_unknown_task(self, client):
        assert client.get("/api/tasks/nope").status_code == 404
        assert client.post("/api/tasks/nope/cancel").status_code == 404
        assert client.get("/api/tasks/nope/events").status_code == 404


class TestSSE:
    def test_replay_after_done(self, client, tmp_path):
        media = tmp_path / "movie.mp4"
        media.touch()
        task_id = client.post(
            "/api/tasks", json={"path": str(media), "options": {"auto_confirm": True}}
        ).json()["id"]
        _wait_status(client, task_id, {"done"})

        # 终态任务的 SSE：回放全部历史事件后主动关流
        resp = client.get(f"/api/tasks/{task_id}/events")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _sse_events(resp.text)
        assert len(events) > 3
        assert events[0]["status"] == "running"
        assert events[-1]["status"] == "done"
        stages = {e["stage"] for e in events}
        assert {"transcribe", "translate", "export"} <= stages

    def test_live_subscription_queue(self, client, tmp_path):
        """内部订阅队列接口：实时收到与历史一致的事件序列。"""
        media = tmp_path / "movie.mp4"
        media.touch()
        manager = client.app.state.manager
        task_id = client.post(
            "/api/tasks", json={"path": str(media), "options": {"auto_confirm": True}}
        ).json()["id"]
        queue = manager.subscribe(task_id)
        _wait_status(client, task_id, {"done"})
        # 队列推送走 call_soon_threadsafe，略落后于 events 落账，轮询到终态事件
        pushed = []
        deadline = time.time() + 5
        while time.time() < deadline:
            while not queue.empty():
                pushed.append(queue.get_nowait())
            if pushed and pushed[-1]["status"] == "done":
                break
            time.sleep(0.02)
        manager.unsubscribe(task_id, queue)
        assert pushed
        assert pushed[-1]["status"] == "done"
        # 推送的事件是历史事件的子序列（订阅于任务开始之后）
        history = manager.get(task_id).events
        assert pushed[-1] == history[-1]


class TestCancel:
    def test_cancel_running_task(self, config_path, mocked_media, monkeypatch, tmp_path):
        """协作式取消：当前进度单元完成后停，任务进入 cancelled。"""
        started = threading.Event()

        def slow_run(media_path, cfg, *, progress_cb=None, **kwargs):
            for i in range(50):
                started.set()
                if progress_cb:
                    progress_cb(
                        {
                            "media": str(media_path),
                            "stage": "translate",
                            "done": i + 1,
                            "total": 50,
                            "message": f"翻译 {i + 1}/50",
                        }
                    )
                time.sleep(0.02)

        monkeypatch.setattr(server_tasks, "run_pipeline", slow_run)
        with TestClient(create_app(config_path=config_path)) as client:
            media = tmp_path / "movie.mp4"
            media.touch()
            task_id = client.post("/api/tasks", json={"path": str(media)}).json()["id"]
            assert started.wait(timeout=5)
            resp = client.post(f"/api/tasks/{task_id}/cancel")
            assert resp.status_code == 200
            _wait_status(client, task_id, {"cancelled"})
            detail = client.get(f"/api/tasks/{task_id}").json()
            assert detail["status"] == "cancelled"
            # 取消前已有部分进度事件
            manager = client.app.state.manager
            progress = [e for e in manager.get(task_id).events if e["total"] == 50]
            assert 0 < len(progress) < 50

    def test_cancel_pending_and_terminal(self, config_path, monkeypatch, tmp_path):
        """pending 立即取消（worker 跳过）；终态取消返回 409。"""
        gate = threading.Event()

        def blocking_run(media_path, cfg, **kwargs):
            gate.wait(timeout=10)

        monkeypatch.setattr(server_tasks, "run_pipeline", blocking_run)
        with TestClient(create_app(config_path=config_path)) as client:
            media = tmp_path / "movie.mp4"
            media.touch()
            first = client.post("/api/tasks", json={"path": str(media)}).json()["id"]
            second = client.post("/api/tasks", json={"path": str(media)}).json()["id"]
            _wait_status(client, first, {"running"})
            # 第二个任务在排队（单并发）
            assert client.get(f"/api/tasks/{second}").json()["status"] == "pending"
            client.post(f"/api/tasks/{second}/cancel")
            assert client.get(f"/api/tasks/{second}").json()["status"] == "cancelled"
            gate.set()
            _wait_status(client, first, {"done"})
            # 终态不可取消
            assert client.post(f"/api/tasks/{first}/cancel").status_code == 409


class TestGlossaryCheckpointFlow:
    def test_waiting_confirm_then_resume(self, client, tmp_path):
        """不 auto_confirm：任务停在 waiting_confirm → 网页确认术语 → resume 跑完。"""
        media = tmp_path / "movie.mp4"
        media.touch()
        task_id = client.post("/api/tasks", json={"path": str(media)}).json()["id"]
        _wait_status(client, task_id, {"waiting_confirm"})

        # 检查点产物已落盘：摘要 + 术语表，stage=contexted
        project = SubtitleProject.load(tmp_path / "movie.sub.json")
        assert project.stage == Stage.CONTEXTED
        assert project.meta.summary == "测试摘要"

        # 网页读取并确认术语表
        glossary = client.get("/api/project/glossary", params={"path": str(media)}).json()
        assert glossary["glossary"][0]["src"] == "Erebus"
        assert not glossary["glossary"][0]["confirmed"]
        client.patch(
            "/api/project/glossary",
            json={"path": str(media), "confirm_all": True},
        )

        resp = client.post(f"/api/tasks/{task_id}/resume")
        assert resp.status_code == 200
        _wait_status(client, task_id, {"done"})
        project = SubtitleProject.load(tmp_path / "movie.sub.json")
        assert project.stage == Stage.TRANSLATED
        assert all(c.translation == "默认译文" for c in project.cues)

    def test_resume_rejects_wrong_state(self, client, tmp_path):
        media = tmp_path / "movie.mp4"
        media.touch()
        task_id = client.post(
            "/api/tasks", json={"path": str(media), "options": {"auto_confirm": True}}
        ).json()["id"]
        _wait_status(client, task_id, {"done"})
        assert client.post(f"/api/tasks/{task_id}/resume").status_code == 409


class TestProjectEndpoints:
    @pytest.fixture
    def project_file(self, tmp_path) -> Path:
        project = SubtitleProject()
        project.meta.source.file = str(tmp_path / "movie.mp4")
        project.glossary = [
            GlossaryEntry(src="Erebus", dst="厄瑞玻斯", count=2),
            GlossaryEntry(src="Nyra", dst="尼拉", count=1),
        ]
        project.cues = [
            Cue(id=1, start=0.1, end=1.2, text="hello world.", translation="你好世界。"),
            Cue(id=2, start=2.0, end=3.0, text="second line."),
        ]
        project.stage = Stage.CONTEXTED
        path = tmp_path / "movie.sub.json"
        project.save(path)
        return path

    def test_read_project(self, client, project_file):
        resp = client.get("/api/project", params={"path": str(project_file)})
        assert resp.status_code == 200
        assert resp.json()["stage"] == "contexted"
        assert len(resp.json()["cues"]) == 2
        # 媒体文件路径也能解析到工程 JSON
        media_path = str(project_file).replace(PROJECT_SUFFIX, ".mp4")
        resp2 = client.get("/api/project", params={"path": media_path})
        assert resp2.json()["cues"] == resp.json()["cues"]
        assert client.get("/api/project", params={"path": "/nonexistent/x.mp4"}).status_code == 404

    def test_patch_cue(self, client, project_file):
        resp = client.patch(
            "/api/project/cues",
            json={"path": str(project_file), "cue_id": 2, "text": "改后原文", "translation": "改后译文"},
        )
        assert resp.status_code == 200
        assert resp.json()["text"] == "改后原文"
        loaded = SubtitleProject.load(project_file)
        assert loaded.cues[1].translation == "改后译文"
        assert client.patch(
            "/api/project/cues", json={"path": str(project_file), "cue_id": 99, "text": "x"}
        ).status_code == 404

    def test_glossary_edit_and_confirm(self, client, project_file):
        resp = client.patch(
            "/api/project/glossary",
            json={
                "path": str(project_file),
                "updates": [{"src": "Erebus", "dst": "幽冥之主"}, {"src": "Nyra", "new_src": "Nyraa"}],
                "confirm": ["Nyraa"],
            },
        )
        assert resp.status_code == 200
        entries = {g["src"]: g for g in resp.json()["glossary"]}
        assert entries["Erebus"]["dst"] == "幽冥之主"
        assert entries["Nyraa"]["confirmed"] is True
        # 落盘生效
        loaded = SubtitleProject.load(project_file)
        assert {g.src for g in loaded.glossary} == {"Erebus", "Nyraa"}
        # 未知术语 404
        assert client.patch(
            "/api/project/glossary",
            json={"path": str(project_file), "confirm": ["不存在"]},
        ).status_code == 404

    def test_export(self, client, project_file):
        resp = client.post(
            "/api/project/export", json={"path": str(project_file), "bilingual": True}
        )
        assert resp.status_code == 200
        srt_path = Path(resp.json()["srt_path"])
        assert srt_path.name == "movie.srt"
        text = srt_path.read_text(encoding="utf-8")
        assert "你好世界。\nhello world." in text


class TestConfigEndpoints:
    def test_api_key_masked(self, client, config_path):
        # 写入明文 key，GET 不得泄露
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data["translate"]["api_key"] = "sk-secret123"
        config_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        resp = client.get("/api/config")
        assert resp.status_code == 200
        assert "sk-secret123" not in resp.text
        assert resp.json()["translate"]["api_key"] == "sk-...***"
        assert resp.json()["translate"]["api_key_resolved"] is True

    def test_update_config(self, client, config_path):
        resp = client.put("/api/config", json={"asr": {"language": "en"}})
        assert resp.status_code == 200
        assert resp.json()["asr"]["language"] == "en"
        assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["asr"]["language"] == "en"

        # api_key 空字符串 = 不修改
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data["translate"]["api_key"] = "sk-old"
        config_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        client.put("/api/config", json={"translate": {"api_key": "", "temperature": 0.3}})
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))["translate"]
        assert saved["api_key"] == "sk-old"
        assert saved["temperature"] == 0.3
        # mask 值原样回传也不修改
        client.put("/api/config", json={"translate": {"api_key": "sk-...***"}})
        assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["translate"]["api_key"] == "sk-old"
        # 新 key 正常写入
        client.put("/api/config", json={"translate": {"api_key": "sk-new456"}})
        assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["translate"]["api_key"] == "sk-new456"

    def test_update_config_validation(self, client):
        assert client.put("/api/config", json={"bogus": {}}).status_code == 400
        assert client.put("/api/config", json={"asr": {"nope": 1}}).status_code == 400
        assert client.put("/api/config", json={"asr": {"backend": "bogus"}}).status_code == 400


class TestMediaAndVideo:
    def test_browse_media(self, client, tmp_path):
        (tmp_path / "a.mp4").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.wav").touch()
        resp = client.get("/api/media", params={"path": str(tmp_path)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["directories"] == ["sub"]
        assert set(Path(m).name for m in body["media"]) == {"a.mp4", "b.wav"}
        assert client.get("/api/media", params={"path": str(tmp_path / "nope")}).status_code == 404

    def test_browse_media_default_path(self, client):
        """path 省略/为空时后端给默认起点（用户主目录），不应报错。"""
        for params in ({}, {"path": ""}):
            resp = client.get("/api/media", params=params)
            assert resp.status_code == 200
            body = resp.json()
            assert Path(body["path"]).is_dir()
            assert isinstance(body["directories"], list)
            assert isinstance(body["media"], list)

    def test_browse_media_unreadable_dir(self, client, tmp_path, monkeypatch):
        """被浏览的目录本身不可读时容错：返回 200 + 空子目录清单，不 500。"""
        (tmp_path / "a.mp4").touch()

        def fake_iterdir(self):
            raise PermissionError(13, "Permission denied", str(self))

        monkeypatch.setattr(Path, "iterdir", fake_iterdir)
        resp = client.get("/api/media", params={"path": str(tmp_path)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["directories"] == []
        assert [Path(m).name for m in body["media"]] == ["a.mp4"]

    def test_browse_media_unreadable_child(self, client, tmp_path, monkeypatch):
        """单个子目录 stat 失败时跳过，其余子目录正常列出。"""
        (tmp_path / "ok").mkdir()
        bad = tmp_path / "secret"
        bad.mkdir()

        real_is_dir = Path.is_dir

        def fake_is_dir(self):
            if self == bad:
                raise PermissionError(13, "Permission denied", str(self))
            return real_is_dir(self)

        monkeypatch.setattr(Path, "is_dir", fake_is_dir)
        resp = client.get("/api/media", params={"path": str(tmp_path)})
        assert resp.status_code == 200
        assert resp.json()["directories"] == ["ok"]

    def test_video_full_and_range(self, client, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(bytes(range(256)) * 4)  # 1024 字节
        # 全量
        resp = client.get("/api/video", params={"path": str(video)})
        assert resp.status_code == 200
        assert resp.headers["accept-ranges"] == "bytes"
        assert len(resp.content) == 1024
        # 闭区间
        resp = client.get(
            "/api/video", params={"path": str(video)}, headers={"Range": "bytes=0-99"}
        )
        assert resp.status_code == 206
        assert resp.headers["content-range"] == "bytes 0-99/1024"
        assert resp.content == bytes(range(100))
        # 开放区间
        resp = client.get(
            "/api/video", params={"path": str(video)}, headers={"Range": "bytes=1000-"}
        )
        assert resp.status_code == 206
        assert resp.headers["content-range"] == "bytes 1000-1023/1024"
        assert len(resp.content) == 24
        # 后缀区间
        resp = client.get(
            "/api/video", params={"path": str(video)}, headers={"Range": "bytes=-10"}
        )
        assert resp.status_code == 206
        assert resp.headers["content-range"] == "bytes 1014-1023/1024"
        assert len(resp.content) == 10
        # 越界 / 非法
        resp = client.get(
            "/api/video", params={"path": str(video)}, headers={"Range": "bytes=2000-"}
        )
        assert resp.status_code == 416
        assert resp.headers["content-range"] == "bytes */1024"
        assert client.get("/api/video", params={"path": str(tmp_path / "nope.mp4")}).status_code == 404


class TestStaticHosting:
    def test_placeholder_without_dist(self, config_path, tmp_path):
        # 显式指向不存在的 dist，避免受仓库里真实 frontend/dist 是否构建影响
        with TestClient(
            create_app(config_path=config_path, frontend_dist=tmp_path / "no-dist")
        ) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            assert "前端尚未构建" in resp.text

    def test_serves_dist_when_present(self, config_path, tmp_path):
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html>spa</html>", encoding="utf-8")
        with TestClient(create_app(config_path=config_path, frontend_dist=dist)) as c:
            resp = c.get("/")
            assert resp.status_code == 200
            assert "spa" in resp.text
            # API 不被静态托管挡住
            assert c.get("/api/tasks").status_code == 200

    def test_spa_route_falls_back_to_index(self, config_path, tmp_path):
        """前端路由（react-router 的客户端路径）应回退到 index.html 而非 404。"""
        dist = tmp_path / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<html>spa</html>", encoding="utf-8")
        (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
        with TestClient(create_app(config_path=config_path, frontend_dist=dist)) as c:
            # 客户端路由 → index.html
            resp = c.get("/tasks/abc123")
            assert resp.status_code == 200
            assert "spa" in resp.text
            # 静态资源正常命中
            resp = c.get("/assets/app.js")
            assert resp.status_code == 200
            assert resp.text == "console.log(1)"
            # 目录穿越不泄露 dist 外的文件
            resp = c.get("/../config.yaml")
            assert resp.status_code in (200, 404)
            if resp.status_code == 200:
                assert "spa" in resp.text  # 回退到 index.html，而非读出文件
