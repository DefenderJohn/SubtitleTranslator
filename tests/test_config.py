"""config.py 测试：默认值、加载保存 round-trip、api_key_env 优先级、密钥不落盘。"""

import pytest

from subtitle_translator.config import (
    AsrConfig,
    Config,
    TranslateConfig,
    UiConfig,
    default_config,
    load_config,
    resolve_api_key,
    resolve_upload_dir,
    save_config,
)


def test_default_values():
    cfg = default_config()
    assert cfg.asr.backend == "transformers"
    assert cfg.asr.model == "Qwen/Qwen3-ASR-1.7B"
    assert cfg.asr.aligner_model == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert cfg.asr.chunk_max_seconds <= 300.0
    assert cfg.translate.history_count == 10
    assert cfg.translate.forward_count == 1
    assert cfg.translate.glossary_max_entries == 50
    assert cfg.translate.target_language == "简体中文"
    assert cfg.translate.api_key is None
    assert cfg.ui.port > 0


def test_load_missing_file_returns_defaults(tmp_path):
    cfg = load_config(tmp_path / "not-exist.yaml")
    assert cfg.asr.model == "Qwen/Qwen3-ASR-1.7B"


def test_save_load_round_trip(tmp_path):
    cfg = default_config()
    cfg.asr.backend = "transformers"
    cfg.translate.base_url = "http://localhost:11434/v1"
    cfg.translate.model = "qwen3:32b"
    cfg.translate.temperature = 0.3
    cfg.ui.port = 9000
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.asr.backend == "transformers"
    assert loaded.translate.base_url == "http://localhost:11434/v1"
    assert loaded.translate.model == "qwen3:32b"
    assert loaded.translate.temperature == pytest.approx(0.3)
    assert loaded.ui.port == 9000


def test_load_partial_yaml_uses_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("translate:\n  model: foo\nunknown_section:\n  x: 1\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.translate.model == "foo"
    assert cfg.translate.history_count == 10
    assert cfg.asr.backend == "transformers"


def test_invalid_backend_rejected():
    with pytest.raises(ValueError, match="非法 asr.backend"):
        AsrConfig(backend="onnx")


def test_save_strips_plaintext_api_key(tmp_path):
    cfg = default_config()
    cfg.translate.api_key = "sk-secret"
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    assert "sk-secret" not in path.read_text(encoding="utf-8")
    assert load_config(path).translate.api_key is None


def test_save_can_include_api_key_explicitly(tmp_path):
    cfg = default_config()
    cfg.translate.api_key = "sk-secret"
    path = tmp_path / "config.yaml"
    save_config(cfg, path, include_api_key=True)
    assert load_config(path).translate.api_key == "sk-secret"


def test_resolve_upload_dir(tmp_path):
    # 默认：~/.subtitle_translator/uploads（展开后是绝对路径）
    default = resolve_upload_dir(Config())
    assert default.is_absolute() and default.name == "uploads"
    assert default.parent.name == ".subtitle_translator"
    # 显式配置优先，支持 ~ 展开
    configured = resolve_upload_dir(UiConfig(upload_dir=str(tmp_path / "up")))
    assert configured == tmp_path / "up"
    # 空白字符串等价于未配置
    assert resolve_upload_dir(UiConfig(upload_dir="  ")).name == "uploads"


def test_resolve_api_key_env_priority(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "from-env")
    cfg = TranslateConfig(api_key="from-yaml", api_key_env="MY_API_KEY")
    assert resolve_api_key(cfg) == "from-env"


def test_resolve_api_key_fallbacks(monkeypatch):
    monkeypatch.delenv("MY_API_KEY", raising=False)
    assert resolve_api_key(TranslateConfig(api_key="from-yaml", api_key_env="MY_API_KEY")) == "from-yaml"
    assert resolve_api_key(TranslateConfig(api_key="from-yaml")) == "from-yaml"
    assert resolve_api_key(TranslateConfig(api_key_env="MY_API_KEY")) is None
    assert resolve_api_key(Config()) is None
    # 空字符串视为未配置
    assert resolve_api_key(TranslateConfig(api_key="")) is None
