"""preflight 启动预检测试：各检查项判定逻辑 + 报告聚合 + 端点连通性（本地 mock server）。

不碰真模型、真 HF 缓存与真翻译端点：模型目录 / HF 缓存用 tmp_path 伪造，
snapshot_download 与 OpenAI 请求全部 mock 或走本地 mock server。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from subtitle_translator import media, preflight
from subtitle_translator.config import Config
from subtitle_translator.preflight import (
    STATUS_FAIL,
    STATUS_OK,
    STATUS_WARN,
    CheckResult,
    EndpointTestResult,
    PreflightReport,
)


def _make_model_dir(path: Path, *, weights: bool = True) -> Path:
    """造一个关键文件齐全的假模型目录。"""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    if weights:
        (path / "model.safetensors").write_bytes(b"x")
    return path


def _make_hf_cache(cache: Path, model_id: str, *, complete: bool = True) -> Path:
    """伪造 HF 缓存快照：models--org--name/snapshots/<rev>/..."""
    snapshot = cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / "abc123"
    _make_model_dir(snapshot, weights=complete)
    return snapshot


def _cfg_with_local_models(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.asr.model = str(_make_model_dir(tmp_path / "asr"))
    cfg.asr.aligner_model = str(_make_model_dir(tmp_path / "aligner"))
    cfg.translate.model = "test-model"
    cfg.translate.api_key = "k"
    return cfg


class TestClassifyModelRef:
    def test_existing_dir_is_local(self, tmp_path):
        kind, ref = preflight._classify_model_ref(str(tmp_path))
        assert kind == "local" and ref == tmp_path

    def test_hub_id(self):
        kind, ref = preflight._classify_model_ref("Qwen/Qwen3-ASR-1.7B")
        assert kind == "hub" and ref == "Qwen/Qwen3-ASR-1.7B"

    def test_missing_absolute_path(self):
        kind, _ = preflight._classify_model_ref("/nonexistent/models/ASR")
        assert kind == "missing"

    def test_missing_dotted_path(self):
        kind, _ = preflight._classify_model_ref("./nonexistent-model")
        assert kind == "missing"

    def test_missing_multi_slash_path(self):
        kind, _ = preflight._classify_model_ref("models/sub/ASR-1.7B")
        assert kind == "missing"


class TestLocalModelCheck:
    def test_complete_dir_ok(self, tmp_path):
        model_dir = _make_model_dir(tmp_path / "m")
        result = preflight._check_one_model("ASR 模型", str(model_dir), download_missing=False, progress_cb=None)
        assert result.status == STATUS_OK

    def test_missing_weights_fails(self, tmp_path):
        model_dir = _make_model_dir(tmp_path / "m", weights=False)
        result = preflight._check_one_model("ASR 模型", str(model_dir), download_missing=False, progress_cb=None)
        assert result.status == STATUS_FAIL
        assert "*.safetensors" in result.message

    def test_missing_config_fails(self, tmp_path):
        model_dir = _make_model_dir(tmp_path / "m")
        (model_dir / "config.json").unlink()
        result = preflight._check_one_model("ASR 模型", str(model_dir), download_missing=False, progress_cb=None)
        assert result.status == STATUS_FAIL
        assert "config.json" in result.message

    def test_nonexistent_path_fails(self, tmp_path):
        result = preflight._check_one_model(
            "ASR 模型", str(tmp_path / "nope"), download_missing=True, progress_cb=None
        )
        assert result.status == STATUS_FAIL
        assert "路径不存在" in result.message
        assert "hub ID" in result.message  # 指引里提示 hub ID 形态

    def test_sharded_weights_ok(self, tmp_path):
        """分片权重（model-00001-of-00002.safetensors 形态）同样算齐全。"""
        model_dir = tmp_path / "m"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "tokenizer_config.json").write_text("{}")
        (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x")
        (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"x")
        result = preflight._check_one_model("ASR 模型", str(model_dir), download_missing=False, progress_cb=None)
        assert result.status == STATUS_OK


class TestHubCache:
    def test_cache_hit_ok(self, tmp_path, monkeypatch):
        cache = tmp_path / "hub"
        _make_hf_cache(cache, "Org/Model")
        monkeypatch.setattr(preflight, "_hf_cache_dir", lambda: cache)
        result = preflight._check_one_model("ASR 模型", "Org/Model", download_missing=True, progress_cb=None)
        assert result.status == STATUS_OK
        assert "缓存" in result.message

    def test_incomplete_snapshot_is_miss(self, tmp_path, monkeypatch):
        cache = tmp_path / "hub"
        _make_hf_cache(cache, "Org/Model", complete=False)
        monkeypatch.setattr(preflight, "_hf_cache_dir", lambda: cache)
        result = preflight._check_one_model("ASR 模型", "Org/Model", download_missing=False, progress_cb=None)
        assert result.status == STATUS_FAIL
        assert "huggingface-cli download" in result.message

    def test_miss_without_download_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(preflight, "_hf_cache_dir", lambda: tmp_path / "hub")
        result = preflight._check_one_model("ASR 模型", "Org/Model", download_missing=False, progress_cb=None)
        assert result.status == STATUS_FAIL
        assert "modelscope download" in result.message

    def test_miss_triggers_download(self, tmp_path, monkeypatch):
        monkeypatch.setattr(preflight, "_hf_cache_dir", lambda: tmp_path / "hub")
        calls = []
        monkeypatch.setattr(
            preflight, "_snapshot_download", lambda repo_id: calls.append(repo_id) or "/fake/snapshot"
        )
        events = []
        result = preflight._check_one_model(
            "ASR 模型", "Org/Model", download_missing=True, progress_cb=events.append
        )
        assert result.status == STATUS_OK
        assert calls == ["Org/Model"]
        assert events and "下载" in events[0]["message"]

    def test_download_failure_falls_back_to_modelscope(self, tmp_path, monkeypatch):
        """HF 下载失败 → 自动走 modelscope 兜底（此处 mock 兜底成功）。"""
        monkeypatch.setattr(preflight, "_hf_cache_dir", lambda: tmp_path / "hub")

        def boom(repo_id):
            raise RuntimeError("connection timeout")

        monkeypatch.setattr(preflight, "_snapshot_download", boom)
        monkeypatch.setattr(
            preflight, "_modelscope_download", lambda repo_id, local_dir: str(local_dir)
        )
        result = preflight._check_one_model("ASR 模型", "Org/Model", download_missing=True, progress_cb=None)
        assert result.status == STATUS_OK
        assert "modelscope" in result.message


class TestModelscopeFallback:
    """HF 失败后 modelscope 兜底的三分支：成功 / 未安装 / 下载失败。"""

    @pytest.fixture
    def hf_broken(self, tmp_path, monkeypatch):
        monkeypatch.setattr(preflight, "_hf_cache_dir", lambda: tmp_path / "hub")

        def boom(repo_id):
            raise RuntimeError("connection timeout")

        monkeypatch.setattr(preflight, "_snapshot_download", boom)

    def test_modelscope_success_resolves_local_path(self, hf_broken, tmp_path, monkeypatch):
        target = tmp_path / "models" / "Model"
        monkeypatch.setattr(
            preflight, "_modelscope_download", lambda repo_id, local_dir: str(target)
        )
        cfg = Config()
        cfg.asr.model = "Org/Model"
        cfg.asr.aligner_model = str(_make_model_dir(tmp_path / "aligner"))
        results = preflight.check_asr_models(cfg, download_missing=True)
        assert results[0].status == STATUS_OK
        assert results[0].resolved_path == str(target)
        assert "config.yaml" in results[0].message  # 提示用户把本地路径写进 config
        assert cfg.asr.model == str(target)  # 内存配置已改写，本次进程走本地路径

    def test_modelscope_not_installed_fails_with_pip_hint(self, hf_broken, monkeypatch):
        def not_installed(repo_id, local_dir):
            raise RuntimeError("未安装 modelscope（请先 pip install modelscope）")

        monkeypatch.setattr(preflight, "_modelscope_download", not_installed)
        result = preflight._check_one_model("ASR 模型", "Org/Model", download_missing=True, progress_cb=None)
        assert result.status == STATUS_FAIL
        assert "connection timeout" in result.message  # 两个错误都呈现
        assert "pip install modelscope" in result.message
        assert "HF_ENDPOINT" in result.message

    def test_modelscope_download_failure_fails(self, hf_broken, monkeypatch):
        def boom(repo_id, local_dir):
            raise RuntimeError("modelscope 网络错误")

        monkeypatch.setattr(preflight, "_modelscope_download", boom)
        result = preflight._check_one_model("ASR 模型", "Org/Model", download_missing=True, progress_cb=None)
        assert result.status == STATUS_FAIL
        assert "connection timeout" in result.message
        assert "modelscope 网络错误" in result.message
        assert "modelscope download --model" in result.message  # 手动下载指引


class TestFfmpegCheck:
    def test_found_ok(self, tmp_path, monkeypatch):
        monkeypatch.setattr(media, "find_ffmpeg", lambda configured=None: "/usr/bin/ffmpeg")
        assert preflight.check_ffmpeg(Config()).status == STATUS_OK

    def test_missing_fails_with_install_hint(self, monkeypatch):
        def raise_not_found(configured=None):
            raise media.FFmpegNotFoundError(media._INSTALL_HINT)

        monkeypatch.setattr(media, "find_ffmpeg", raise_not_found)
        result = preflight.check_ffmpeg(Config())
        assert result.status == STATUS_FAIL
        assert "imageio-ffmpeg" in result.message


class TestTranslateCheck:
    @pytest.fixture
    def stub_endpoint_ok(self, monkeypatch):
        monkeypatch.setattr(
            preflight,
            "test_translate_endpoint",
            lambda cfg, **kw: EndpointTestResult(True, latency_ms=12.3, response_preview="你好！"),
        )

    def test_missing_model_fails(self):
        cfg = Config()  # translate.model 默认为 ""
        results = preflight.check_translate(cfg)
        assert results[0].status == STATUS_FAIL
        assert "translate.model" in results[0].message
        assert "config init" in results[0].message
        assert len(results) == 1  # 配置不全时不做连通性测试

    def test_missing_api_key_warns(self, stub_endpoint_ok):
        cfg = Config()
        cfg.translate.model = "m"
        results = preflight.check_translate(cfg)
        by_name = {r.name: r for r in results}
        assert by_name["翻译 api_key"].status == STATUS_WARN
        assert by_name["翻译端点连通性"].status == STATUS_OK

    def test_connectivity_failure_is_hard_fail(self, monkeypatch):
        monkeypatch.setattr(
            preflight,
            "test_translate_endpoint",
            lambda cfg, **kw: EndpointTestResult(False, error="APIConnectionError: refused"),
        )
        cfg = Config()
        cfg.translate.model = "m"
        cfg.translate.api_key = "k"
        results = preflight.check_translate(cfg)
        conn = next(r for r in results if r.name == "翻译端点连通性")
        assert conn.status == STATUS_FAIL
        assert "refused" in conn.message and "排查" in conn.message


class _MockHandler(BaseHTTPRequestHandler):
    """最小 OpenAI 兼容端点：原样回显 user 内容。"""

    def do_POST(self) -> None:  # noqa: N802 - stdlib 约定
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        user = payload["messages"][-1]["content"]
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": f"回复：{user}"}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def mock_endpoint():
    server = HTTPServer(("127.0.0.1", 0), _MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()
    thread.join()


class TestEndpointTest:
    def test_ok(self, mock_endpoint):
        cfg = Config()
        cfg.translate.base_url = mock_endpoint
        cfg.translate.model = "m"
        result = preflight.test_translate_endpoint(cfg)
        assert result.ok
        assert result.latency_ms >= 0
        assert result.response_preview == "回复：你好"

    def test_connection_refused(self):
        cfg = Config()
        cfg.translate.base_url = "http://127.0.0.1:1/v1"  # 1 端口必然连不上
        cfg.translate.model = "m"
        result = preflight.test_translate_endpoint(cfg, timeout=2.0)
        assert not result.ok
        assert result.error

    def test_missing_model_shortcuts(self, mock_endpoint):
        cfg = Config()
        cfg.translate.base_url = mock_endpoint
        result = preflight.test_translate_endpoint(cfg)
        assert not result.ok and "model" in result.error


class TestGpuCheck:
    def test_no_gpu_warns(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        result = preflight.check_gpu()
        assert result.status == STATUS_WARN
        assert "CPU" in result.message

    def test_gpu_ok(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Fake GPU")
        result = preflight.check_gpu()
        assert result.status == STATUS_OK and "Fake GPU" in result.message


class TestReport:
    def test_aggregation(self):
        report = PreflightReport(
            [
                CheckResult("a", STATUS_OK),
                CheckResult("b", STATUS_WARN),
                CheckResult("c", STATUS_FAIL),
            ]
        )
        assert not report.ok
        assert [c.name for c in report.failed] == ["c"]
        data = report.to_dict()
        assert data["ok"] is False and len(data["checks"]) == 3
        text = report.format_text()
        assert "预检未通过" in text

    def test_all_ok(self):
        report = PreflightReport([CheckResult("a", STATUS_OK), CheckResult("b", STATUS_WARN)])
        assert report.ok  # 警告不阻塞
        assert "预检通过" in report.format_text()


class TestRunPreflight:
    @pytest.fixture
    def stub_all_ok(self, monkeypatch, tmp_path):
        """把重活全部 stub 成 ok，只留编排逻辑。"""
        monkeypatch.setattr(
            preflight, "check_ffmpeg", lambda cfg: CheckResult("ffmpeg", STATUS_OK, "fake")
        )
        monkeypatch.setattr(preflight, "check_gpu", lambda: CheckResult("GPU", STATUS_OK, "fake"))
        monkeypatch.setattr(
            preflight,
            "test_translate_endpoint",
            lambda cfg, **kw: EndpointTestResult(True, latency_ms=1.0, response_preview="x"),
        )

    def test_full_pass(self, stub_all_ok, tmp_path):
        cfg = _cfg_with_local_models(tmp_path)
        report = preflight.run_preflight(cfg)
        assert report.ok
        names = [c.name for c in report.checks]
        assert names == ["ffmpeg", "ASR 模型", "对齐模型", "GPU", "翻译端点配置", "翻译 api_key", "翻译端点连通性"]

    def test_need_translate_false_skips_translate(self, stub_all_ok, tmp_path):
        cfg = _cfg_with_local_models(tmp_path)
        cfg.translate.model = ""  # 不查翻译就不该失败
        report = preflight.run_preflight(cfg, need_translate=False)
        assert report.ok
        assert all("翻译" not in c.name for c in report.checks)

    def test_need_transcribe_false_skips_media_side(self, stub_all_ok):
        cfg = Config()  # ffmpeg/模型都没配，跳过转录侧就不该查
        cfg.translate.model = "m"
        cfg.translate.api_key = "k"
        report = preflight.run_preflight(cfg, need_transcribe=False)
        assert report.ok
        assert all(c.name.startswith("翻译") for c in report.checks)

    def test_failure_blocks(self, stub_all_ok):
        cfg = Config()
        cfg.asr.model = "/nonexistent/asr-model"
        cfg.asr.aligner_model = "/nonexistent/aligner"
        cfg.translate.model = "m"
        cfg.translate.api_key = "k"
        report = preflight.run_preflight(cfg)
        assert not report.ok
        assert {c.name for c in report.failed} == {"ASR 模型", "对齐模型"}


class TestRecheck:
    def test_all_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(media, "find_ffmpeg", lambda configured=None: "/usr/bin/ffmpeg")
        cfg = _cfg_with_local_models(tmp_path)
        assert preflight.recheck_model_paths(cfg) == []

    def test_deleted_model_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(media, "find_ffmpeg", lambda configured=None: "/usr/bin/ffmpeg")
        cfg = _cfg_with_local_models(tmp_path)
        cfg.asr.model = str(tmp_path / "asr" / "gone")  # 模拟运行期间被删
        problems = preflight.recheck_model_paths(cfg)
        assert len(problems) == 1 and "asr.model" in problems[0]

    def test_missing_weights_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(media, "find_ffmpeg", lambda configured=None: "/usr/bin/ffmpeg")
        cfg = _cfg_with_local_models(tmp_path)
        for f in Path(cfg.asr.model).glob("*.safetensors"):
            f.unlink()
        problems = preflight.recheck_model_paths(cfg)
        assert any("safetensors" in p for p in problems)

    def test_missing_ffmpeg_reported(self, tmp_path, monkeypatch):
        def raise_not_found(configured=None):
            raise media.FFmpegNotFoundError("找不到 ffmpeg")

        monkeypatch.setattr(media, "find_ffmpeg", raise_not_found)
        cfg = _cfg_with_local_models(tmp_path)
        problems = preflight.recheck_model_paths(cfg)
        assert any("ffmpeg" in p for p in problems)
