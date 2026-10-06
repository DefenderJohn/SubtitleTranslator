"""启动预检（preflight）：进入服务即 ready-to-use。

设计原则：**运行时炸不如启动时炸**。``run`` / ``serve`` 在干活之前先跑一遍
预检，把「ffmpeg 缺失、模型文件不齐、翻译端点连不通」这类问题在启动时
就以清晰报错暴露出来，而不是转录/翻译跑到一半才炸。

检查项：

- ffmpeg：:func:`media.find_ffmpeg` 探测，失败 = 硬错误（含安装指引）；
- ASR / 对齐模型：本地路径要求目录存在且关键文件齐全（config.json、
  tokenizer_config.json、至少一个 .safetensors）；hub ID 查 HF 缓存，
  未命中且 ``download_missing=True`` 时当场下载（尊重 HF_ENDPOINT，
  失败给出 modelscope 备选指引），下载失败 = 硬错误；
- 翻译端点（``need_translate=True`` 时）：model / base_url 未配置 = 硬错误；
  api_key 缺失 = 警告（本地端点不需要 key）；连通性测试发一条最小 chat
  请求（「你好」，max_tokens 极小，超时 15s），失败 = 硬错误。
  :func:`test_translate_endpoint` 同时被 ``POST /api/config/test`` 复用；
- GPU：torch.cuda.is_available() 探测，无 GPU = 警告（可跑 CPU，只是慢）。

结果聚合为 :class:`PreflightReport`（每项 {name, status, message}，
status ∈ ok / warn / fail），有任一项 fail 即整体不通过。
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from . import media
from .config import Config, TranslateConfig, resolve_api_key

logger = logging.getLogger(__name__)

# 检查状态取值
STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"

# 模型目录/快照的关键文件：config.json + tokenizer_config.json + 至少一个 .safetensors
# （以 Qwen3-ASR-1.7B / Qwen3-ForcedAligner-0.6B 实际结构为准）
MODEL_REQUIRED_FILES = ("config.json", "tokenizer_config.json")
MODEL_WEIGHTS_GLOB = "*.safetensors"

# 翻译端点连通性测试：最小请求 + 短超时（不能把启动卡在端点不通上）
ENDPOINT_TEST_TIMEOUT = 15.0
ENDPOINT_TEST_MAX_TOKENS = 8
ENDPOINT_TEST_PROMPT = "你好"
RESPONSE_PREVIEW_CHARS = 50

# hub ID 形如 "Qwen/Qwen3-ASR-1.7B"（恰好一段斜杠、两段合法字符）；
# 含更多路径特征（开头 / ~ .、多于一段斜杠、反斜杠）的按本地路径处理
_HUB_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")

# 预检进度回调：progress_cb({"check": 检查名, "message": 进度说明})
ProgressCb = Callable[[dict], None]


@dataclass
class CheckResult:
    """单项检查结果。message 在 ok 时通常是简短说明，warn/fail 时是排查指引。"""

    name: str
    status: str
    message: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "message": self.message}


@dataclass
class PreflightReport:
    """预检报告：全部检查项 + 整体判定（有 fail 即不通过）。"""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != STATUS_FAIL for c in self.checks)

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == STATUS_FAIL]

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [c.to_dict() for c in self.checks]}

    def format_text(self) -> str:
        """人可读报告（CLI / serve 启动打印用）。"""
        marks = {STATUS_OK: "✓", STATUS_WARN: "!", STATUS_FAIL: "✗"}
        lines = []
        for c in self.checks:
            lines.append(f"  [{marks.get(c.status, '?')}] {c.name}: {c.message or c.status}")
        verdict = "预检通过" if self.ok else "预检未通过（存在失败项，请先解决后再试）"
        return "\n".join([*lines, verdict])


@dataclass
class EndpointTestResult:
    """翻译端点连通性测试结果（POST /api/config/test 的返回载体）。"""

    ok: bool
    latency_ms: float = 0.0
    response_preview: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "response_preview": self.response_preview,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# 模型引用分类与 HF 缓存
# ---------------------------------------------------------------------------


def _classify_model_ref(value: str) -> tuple[str, Union[Path, str]]:
    """模型标识分类：("local", Path) / ("hub", repo_id) / ("missing", Path)。

    - 已存在的目录 → 本地路径；
    - 否则形如 hub ID（恰好 ``org/name`` 形态）→ hub ID；
    - 其余（绝对路径、~、./、多段斜杠等但目录不存在）→ 缺失的本地路径。
    """
    path = Path(str(value)).expanduser()
    try:
        if path.is_dir():
            return "local", path
    except OSError:
        pass
    text = str(value).strip()
    looks_pathy = (
        text.startswith(("/", "~", "."))
        or "\\" in text
        or text.count("/") != 1
    )
    if not looks_pathy and _HUB_ID_RE.match(text):
        return "hub", text
    return "missing", path


def _hf_cache_dir() -> Path:
    """HF 缓存根目录：优先 huggingface_hub 的解析（含 HF_HOME/HF_HUB_CACHE），
    未安装时退回 ``HF_HUB_CACHE`` 环境变量或 ``~/.cache/huggingface/hub``。"""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except ImportError:
        return Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()


def _find_complete_snapshot(model_id: str, cache_dir: Optional[Path] = None) -> Optional[Path]:
    """在 HF 缓存里找 model_id 的完整快照（关键文件齐全），找不到返回 None。

    缓存布局：``models--<org>--<name>/snapshots/<commit>/...``；任一快照
    目录满足关键文件要求即视为已缓存。
    """
    cache_dir = cache_dir or _hf_cache_dir()
    repo_dir = cache_dir / ("models--" + model_id.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    try:
        candidates = [d for d in snapshots.iterdir() if d.is_dir()]
    except OSError:
        return None
    for snapshot in candidates:
        if _missing_model_files(snapshot) == []:
            return snapshot
    return None


def _missing_model_files(model_dir: Path) -> list[str]:
    """模型目录缺哪些关键文件（空列表 = 齐全）。"""
    missing = [name for name in MODEL_REQUIRED_FILES if not (model_dir / name).is_file()]
    try:
        has_weights = any(model_dir.glob(MODEL_WEIGHTS_GLOB))
    except OSError:
        has_weights = False
    if not has_weights:
        missing.append(MODEL_WEIGHTS_GLOB)
    return missing


def _snapshot_download(model_id: str) -> str:
    """下载模型快照（huggingface_hub，尊重 HF_ENDPOINT）。单独成函数便于测试 mock。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "未安装 huggingface_hub，无法自动下载模型。请安装转录层依赖：\n"
            "  pip install 'subtitle-translator[asr]'\n"
            "或改用 modelscope 手动下载（见下方指引）。"
        ) from exc
    return snapshot_download(model_id)


# ---------------------------------------------------------------------------
# 各检查项
# ---------------------------------------------------------------------------


def check_ffmpeg(cfg: Config) -> CheckResult:
    """ffmpeg 探测：配置项 → PATH → imageio-ffmpeg；失败 = 硬错误（含安装指引）。"""
    try:
        path = media.find_ffmpeg(cfg.asr.ffmpeg_path or None)
    except media.FFmpegNotFoundError as exc:
        return CheckResult("ffmpeg", STATUS_FAIL, str(exc))
    return CheckResult("ffmpeg", STATUS_OK, f"已找到 {path}")


def _check_one_model(
    label: str,
    value: str,
    *,
    download_missing: bool,
    progress_cb: Optional[ProgressCb],
) -> CheckResult:
    """检查单个模型（ASR / 对齐器共用判定逻辑）。"""
    kind, ref = _classify_model_ref(value)

    if kind == "local":
        assert isinstance(ref, Path)
        missing = _missing_model_files(ref)
        if missing:
            return CheckResult(
                label,
                STATUS_FAIL,
                f"本地模型目录缺少关键文件 {missing}: {ref}\n"
                "模型可能下载不完整，请重新下载（modelscope 或 huggingface-cli，见文档 2.4 节）。",
            )
        return CheckResult(label, STATUS_OK, f"本地模型 {ref}")

    if kind == "missing":
        return CheckResult(
            label,
            STATUS_FAIL,
            f"模型路径不存在: {ref}\n"
            "请检查 config.yaml 中该配置项：本地路径拼写是否正确；"
            "想从 Hugging Face 下载请填 hub ID（如 Qwen/Qwen3-ASR-1.7B）。",
        )

    # hub ID：先查缓存，未命中按 download_missing 决定是否当场下载
    repo_id = str(ref)
    snapshot = _find_complete_snapshot(repo_id)
    if snapshot is not None:
        return CheckResult(label, STATUS_OK, f"HF 缓存已就绪 {repo_id}（{snapshot.name[:12]}）")

    if not download_missing:
        return CheckResult(
            label,
            STATUS_FAIL,
            f"HF 缓存中找不到 {repo_id} 的完整快照，且本次未允许自动下载。\n"
            f"可手动执行：huggingface-cli download {repo_id}\n"
            f"或改用 modelscope：modelscope download --model {repo_id} --local_dir <本地目录>\n"
            "并把 config.yaml 中对应项改为该本地路径。",
        )

    if progress_cb is not None:
        progress_cb({"check": label, "message": f"正在下载模型 {repo_id}（首次较慢）…"})
    logger.info("预检：HF 缓存未命中 %s，开始 snapshot_download", repo_id)
    try:
        path = _snapshot_download(repo_id)
    except Exception as exc:  # noqa: BLE001 - 下载失败统一转硬错误 + 备选指引
        logger.warning("预检：下载 %s 失败：%s", repo_id, exc)
        return CheckResult(
            label,
            STATUS_FAIL,
            f"模型 {repo_id} 自动下载失败: {exc}\n"
            "排查：网络能否访问 HF endpoint（可用 HF_ENDPOINT 环境变量换镜像，\n"
            "如 https://hf-mirror.com）；或改用 modelscope 手动下载：\n"
            f"  modelscope download --model {repo_id} --local_dir <本地目录>\n"
            "然后把 config.yaml 中对应项改为该本地路径。",
        )
    logger.info("预检：%s 下载完成 -> %s", repo_id, path)
    return CheckResult(label, STATUS_OK, f"已下载 {repo_id} -> {path}")


def check_asr_models(
    cfg: Config,
    *,
    download_missing: bool = True,
    progress_cb: Optional[ProgressCb] = None,
) -> list[CheckResult]:
    """ASR 模型 + 对齐模型两项检查。"""
    return [
        _check_one_model(
            "ASR 模型", cfg.asr.model,
            download_missing=download_missing, progress_cb=progress_cb,
        ),
        _check_one_model(
            "对齐模型", cfg.asr.aligner_model,
            download_missing=download_missing, progress_cb=progress_cb,
        ),
    ]


def test_translate_endpoint(
    cfg: Union[Config, TranslateConfig],
    *,
    timeout: float = ENDPOINT_TEST_TIMEOUT,
) -> EndpointTestResult:
    """给翻译端点发一条最小 chat 请求（「你好」，max_tokens 极小）测连通性。

    不打重试（max_retries=0，失败立刻返回）；被预检与
    ``POST /api/config/test`` 端点共同复用。
    """
    translate = cfg.translate if isinstance(cfg, Config) else cfg
    if not translate.base_url:
        return EndpointTestResult(False, error="translate.base_url 未配置")
    if not translate.model:
        return EndpointTestResult(False, error="translate.model 未配置")
    from openai import OpenAI

    client = OpenAI(
        base_url=translate.base_url,
        api_key=resolve_api_key(translate) or "EMPTY",
        timeout=timeout,
        max_retries=0,
    )
    start = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=translate.model,
            messages=[{"role": "user", "content": ENDPOINT_TEST_PROMPT}],
            max_tokens=ENDPOINT_TEST_MAX_TOKENS,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 - 连通性测试，一切异常都是「不通」
        return EndpointTestResult(False, error=f"{type(exc).__name__}: {exc}")
    latency_ms = (time.monotonic() - start) * 1000
    content = ""
    if resp.choices:
        content = resp.choices[0].message.content or ""
    return EndpointTestResult(
        True, latency_ms=round(latency_ms, 1), response_preview=content[:RESPONSE_PREVIEW_CHARS]
    )


def check_translate(cfg: Config) -> list[CheckResult]:
    """翻译端点检查：配置完整性 → api_key（警告级）→ 连通性（发「你好」）。"""
    results: list[CheckResult] = []
    translate = cfg.translate
    if not translate.model or not translate.base_url:
        missing = "、".join(
            name for name, val in (("translate.model", translate.model), ("translate.base_url", translate.base_url)) if not val
        )
        results.append(
            CheckResult(
                "翻译端点配置",
                STATUS_FAIL,
                f"{missing} 未配置。请在 config.yaml 的 translate 节填写，"
                "例如：\n  translate:\n    base_url: https://api.deepseek.com/v1\n"
                "    model: deepseek-chat\n"
                "（可用 `subtitle-translator config init` 生成模板）",
            )
        )
        return results  # 端点没配全，连通性无从测起

    results.append(CheckResult("翻译端点配置", STATUS_OK, f"{translate.model} @ {translate.base_url}"))

    if resolve_api_key(cfg) is None:
        results.append(
            CheckResult(
                "翻译 api_key",
                STATUS_WARN,
                "未配置 translate.api_key。若端点需要鉴权，请在 config.yaml 设置\n"
                "  api_key_env: 环境变量名   （推荐，密钥不落盘）\n"
                "本地 vLLM / Ollama 无鉴权时可忽略本警告。",
            )
        )
    else:
        results.append(CheckResult("翻译 api_key", STATUS_OK, "已配置"))

    test = test_translate_endpoint(cfg)
    if test.ok:
        results.append(
            CheckResult(
                "翻译端点连通性",
                STATUS_OK,
                f"{test.latency_ms:.0f}ms，响应预览: {test.response_preview!r}",
            )
        )
    else:
        results.append(
            CheckResult(
                "翻译端点连通性",
                STATUS_FAIL,
                f"连通性测试失败: {test.error}\n"
                "排查：① base_url 是否带 /v1 后缀、端口对不对，本地端点先确认服务已启动；\n"
                "② 401/鉴权失败：api_key_env 指向的环境变量是否设置（变量名≠密钥本身）；\n"
                "③ 超时：网络能否访问该端点（必要时调大 request_timeout）。",
            )
        )
    return results


def check_gpu() -> CheckResult:
    """GPU 探测：无 GPU 只是慢（CPU 可跑），警告不硬错。"""
    try:
        import torch
    except ImportError:
        return CheckResult(
            "GPU",
            STATUS_WARN,
            "未安装 torch，无法探测 GPU（转录需要 asr extra：pip install 'subtitle-translator[asr]'）。",
        )
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return CheckResult("GPU", STATUS_OK, f"CUDA 可用：{name}")
    except Exception as exc:  # noqa: BLE001 - 探测失败按无 GPU 处理
        logger.debug("CUDA 探测异常：%s", exc)
    return CheckResult("GPU", STATUS_WARN, "未检测到可用 GPU，将使用 CPU 转录（速度明显更慢）。")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def run_preflight(
    cfg: Config,
    *,
    need_translate: bool = True,
    need_transcribe: bool = True,
    download_missing: bool = True,
    progress_cb: Optional[ProgressCb] = None,
) -> PreflightReport:
    """跑完整预检，返回 :class:`PreflightReport`（``report.ok`` 为整体判定）。

    - ``need_translate=False``：跳过翻译端点检查（transcribe-only 场景）；
    - ``need_transcribe=False``：跳过 ffmpeg / 模型 / GPU 检查（--from-json
      重翻场景，不接触媒体与模型）；
    - ``download_missing=False``：hub ID 缓存未命中时不自动下载，直接判失败；
    - ``progress_cb({"check", "message"})``：耗时步骤（模型下载）的进度提示。
    """
    report = PreflightReport()
    if need_transcribe:
        report.checks.append(check_ffmpeg(cfg))
        report.checks.extend(
            check_asr_models(cfg, download_missing=download_missing, progress_cb=progress_cb)
        )
        report.checks.append(check_gpu())
    if need_translate:
        report.checks.extend(check_translate(cfg))
    for check in report.checks:
        log = logger.warning if check.status == STATUS_WARN else (
            logger.error if check.status == STATUS_FAIL else logger.info
        )
        log("预检 [%s] %s: %s", check.status, check.name, check.message.splitlines()[0] if check.message else "")
    return report


def recheck_model_paths(cfg: Config) -> list[str]:
    """任务执行前的轻量复核：模型路径 / 缓存仍存在（防运行期间文件被删）。

    只查存在性，不下载、不联网；返回问题清单（空 = 通过）。
    """
    problems: list[str] = []
    try:
        media.find_ffmpeg(cfg.asr.ffmpeg_path or None)
    except media.FFmpegNotFoundError as exc:
        problems.append(str(exc))
    for label, value in (("asr.model", cfg.asr.model), ("asr.aligner_model", cfg.asr.aligner_model)):
        kind, ref = _classify_model_ref(value)
        if kind == "missing":
            problems.append(f"{label} 路径不存在: {ref}")
        elif kind == "local":
            assert isinstance(ref, Path)
            missing = _missing_model_files(ref)
            if missing:
                problems.append(f"{label} 本地目录缺少关键文件 {missing}: {ref}")
        elif _find_complete_snapshot(str(ref)) is None:
            problems.append(f"{label} 的 HF 缓存快照已不存在: {ref}")
    return problems
