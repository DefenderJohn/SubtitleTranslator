"""YAML 配置加载与保存。

单份 config.yaml 为唯一配置存储，网页 / CLI / 脚本共用，分 asr / translate / ui 三节。

api_key 安全：``resolve_api_key`` 优先取 ``api_key_env`` 指向的环境变量，
其次才是 yaml 里的明文 api_key；``save_config`` 默认不写入明文密钥。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Union

import yaml

ASR_BACKENDS = ("vllm", "transformers")


@dataclass
class AsrConfig:
    backend: str = "vllm"  # vllm | transformers
    model: str = "Qwen/Qwen3-ASR-1.7B"
    aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    device: str = "cuda"
    dtype: str = "float16"  # RTX 2080 Ti（Turing）不支持 bf16 原生计算，用 float16
    chunk_max_seconds: float = 290.0  # 对齐器输入上限留余量（qwen-asr 内部再按 180s 切块）
    language: Optional[str] = None  # 源语言，None=自动检测
    ffmpeg_path: str = ""  # ffmpeg 二进制路径，空=自动探测（PATH → imageio-ffmpeg）

    def __post_init__(self) -> None:
        if self.backend not in ASR_BACKENDS:
            raise ValueError(
                f"非法 asr.backend: {self.backend!r}，合法取值为: {', '.join(ASR_BACKENDS)}"
            )


@dataclass
class TranslateConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: Optional[str] = None
    api_key_env: str = ""
    model: str = ""
    temperature: float = 0.7
    history_count: int = 10
    forward_count: int = 1
    glossary_max_entries: int = 50
    additional_prompt: str = "翻译当前字幕到简体中文"
    target_language: str = "简体中文"
    request_timeout: float = 120.0  # 单次 HTTP 请求超时（秒）
    max_retries: int = 4  # HTTP 层对 429/5xx/超时的指数退避重试次数
    glossary_max_retries: int = 3  # 术语表解析失败的重试次数（换提示/降批）


@dataclass
class UiConfig:
    host: str = "127.0.0.1"
    port: int = 7860


@dataclass
class Config:
    asr: AsrConfig = field(default_factory=AsrConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    ui: UiConfig = field(default_factory=UiConfig)


def _section_from_dict(cls: type, data: Any) -> Any:
    """用 dict 构造配置节：缺字段给默认值，忽略未知字段。"""
    if not isinstance(data, dict):
        return cls()
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def default_config() -> Config:
    return Config()


def load_config(path: Union[str, Path]) -> Config:
    """加载 config.yaml；文件不存在时返回默认值。"""
    path = Path(path)
    if not path.exists():
        return default_config()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config(
        asr=_section_from_dict(AsrConfig, data.get("asr")),
        translate=_section_from_dict(TranslateConfig, data.get("translate")),
        ui=_section_from_dict(UiConfig, data.get("ui")),
    )


def save_config(
    cfg: Config,
    path: Union[str, Path],
    include_api_key: bool = False,
) -> None:
    """保存 config.yaml。默认剥离明文 api_key，避免密钥落盘。"""
    data = {
        "asr": asdict(cfg.asr),
        "translate": asdict(cfg.translate),
        "ui": asdict(cfg.ui),
    }
    if not include_api_key:
        data["translate"]["api_key"] = None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def resolve_api_key(cfg: Union[Config, TranslateConfig]) -> Optional[str]:
    """解析 api_key：api_key_env 环境变量优先，yaml 明文次之，都没有返回 None。"""
    translate = cfg.translate if isinstance(cfg, Config) else cfg
    if translate.api_key_env:
        value = os.environ.get(translate.api_key_env)
        if value:
            return value
    return translate.api_key or None
