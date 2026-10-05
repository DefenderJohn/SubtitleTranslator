"""转录层：ASR 封装 + 静音切块拼接。

使用官方 ``qwen-asr`` 包（Qwen3ASRModel）：模型 Qwen/Qwen3-ASR-1.7B，
配合 Qwen/Qwen3-ForcedAligner-0.6B 出词级时间戳（return_time_stamps=True）。

- 两个 backend：VllmBackend 优先、TransformersBackend 兜底，配置项
  ``asr.backend`` 切换，对上层接口零差异。qwen_asr 在 load() 内部延迟
  import，未安装时本模块仍可正常 import，load() 报清晰错误。
- 硬约束：ForcedAligner 单次只支持 ≤5 分钟音频。chunk_plan 基于 ffmpeg
  silencedetect 的静音边界切块（不引入 silero-vad 等额外 torch 依赖），
  transcribe_media 逐块抽音频转录后偏移时间戳拼接。
- 分段策略：转录完成后直接流水线调用 segment.segment_words 生成 cues，
  词级时间戳保留在每个 cue 的 words 字段里（不存顶层 raw_words 冗余字段；
  展平所有 cue 的 words 即可恢复完整词序列）。
"""

from __future__ import annotations

import gc
import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Union

from . import media
from .config import ASR_BACKENDS, AsrConfig, Config
from .models import SourceInfo, Stage, SubtitleProject, WordTiming
from .segment import segment_words

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ASR backend 抽象
# ---------------------------------------------------------------------------


def _import_qwen_asr():
    """延迟 import qwen_asr，未安装时给出清晰指引。"""
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "未安装 qwen-asr，无法加载 ASR 模型。请安装转录层依赖：\n"
            "  pip install 'subtitle-translator[asr]'\n"
            "（或至少 pip install qwen-asr；vllm backend 还需 vllm，"
            "transformers backend 还需 torch+transformers）"
        ) from exc
    return Qwen3ASRModel


def _get(obj, name: str, default=None):
    """同时容忍属性访问与 dict 访问。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_result(result) -> tuple[str, list[WordTiming]]:
    """从 qwen-asr 的单条转录结果抽出 (text, words)。"""
    text = str(_get(result, "text", "") or "")
    raw_words = (
        _get(result, "time_stamps")
        or _get(result, "timestamps")
        or _get(result, "words")
        or []
    )
    words = [
        WordTiming(
            text=str(_get(w, "text", "")),
            start=float(_get(w, "start", 0.0)),
            end=float(_get(w, "end", 0.0)),
        )
        for w in raw_words
    ]
    return text, words


class AsrBackend(ABC):
    """ASR backend 抽象：load / transcribe_chunk / unload。"""

    def __init__(self, cfg: AsrConfig):
        self.cfg = cfg
        self._model = None

    @abstractmethod
    def load(self) -> None:
        """加载模型（含 forced_aligner）。重复调用应无副作用。"""

    @abstractmethod
    def transcribe_chunk(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> tuple[str, list[WordTiming]]:
        """转录单个音频片段（≤ chunk_max_seconds），返回 (text, 词级时间戳)（块内相对时间）。"""

    def unload(self) -> None:
        """释放模型引用。不 import torch，显存回收交给进程/GC。"""
        self._model = None
        gc.collect()

    def _ensure_loaded(self):
        if self._model is None:
            raise RuntimeError("backend 尚未 load()，请先调用 load()")


class VllmBackend(AsrBackend):
    """vLLM backend（优先）：Qwen3ASRModel.LLM(...)。"""

    def load(self) -> None:
        if self._model is not None:
            return
        Qwen3ASRModel = _import_qwen_asr()
        self._model = Qwen3ASRModel.LLM(
            model=self.cfg.model,
            forced_aligner=self.cfg.aligner_model,
            dtype=self.cfg.dtype,
        )

    def transcribe_chunk(self, audio_path, language=None):
        self._ensure_loaded()
        results = self._model.transcribe(
            audio=str(audio_path),
            language=language,
            return_time_stamps=True,
        )
        return _extract_result(results[0])


class TransformersBackend(AsrBackend):
    """transformers backend（兜底）：Qwen3ASRModel.from_pretrained(...)。"""

    def load(self) -> None:
        if self._model is not None:
            return
        Qwen3ASRModel = _import_qwen_asr()
        self._model = Qwen3ASRModel.from_pretrained(
            self.cfg.model,
            forced_aligner=self.cfg.aligner_model,
            dtype=self.cfg.dtype,
            device_map=self.cfg.device,
        )

    def transcribe_chunk(self, audio_path, language=None):
        self._ensure_loaded()
        results = self._model.transcribe(
            audio=str(audio_path),
            language=language,
            return_time_stamps=True,
        )
        return _extract_result(results[0])


def create_backend(cfg: Union[Config, AsrConfig]) -> AsrBackend:
    """按配置创建 backend（未 load，调用方负责 load/unload）。"""
    asr = cfg.asr if isinstance(cfg, Config) else cfg
    if asr.backend == "vllm":
        return VllmBackend(asr)
    if asr.backend == "transformers":
        return TransformersBackend(asr)
    raise ValueError(
        f"非法 asr.backend: {asr.backend!r}，合法取值为: {', '.join(ASR_BACKENDS)}"
    )


# ---------------------------------------------------------------------------
# 切块
# ---------------------------------------------------------------------------


def chunk_plan(
    duration: float,
    silences: list[tuple[float, float]],
    max_seconds: float,
) -> list[tuple[float, float]]:
    """把 [0, duration] 切成 ≤ max_seconds 的块，优先在静音中点下刀。

    切点选择：目标切点（当前块起点 + max_seconds）之前、最接近目标的静音
    中点；找不到合适静音时硬切在目标处并记录 warning。返回 [(start, end), ...]，
    完整覆盖 [0, duration]。
    """
    if duration <= 0 or max_seconds <= 0:
        return []
    midpoints = sorted(
        (s + e) / 2 for s, e in silences if e > 0.0 and s < duration
    )
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while duration - start > max_seconds:
        target = start + max_seconds
        candidates = [m for m in midpoints if start < m <= target]
        if candidates:
            cut = min(candidates, key=lambda m: target - m)
        else:
            cut = target
            logger.warning(
                "切块硬切：%.1fs 附近没有可用静音边界，在 %.1fs 处硬切",
                target,
                cut,
            )
        chunks.append((start, cut))
        start = cut
    chunks.append((start, duration))
    return chunks


# ---------------------------------------------------------------------------
# 整媒体转录
# ---------------------------------------------------------------------------


def transcribe_media(
    path: Union[str, Path],
    cfg: Config,
    backend: Optional[AsrBackend] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> SubtitleProject:
    """整条流水线：probe → 切块 → 逐块抽音频转录 → 偏移拼接 → 分段 → Project。

    - backend 为 None 时按 cfg.asr.backend 自建并负责 load/unload；
      传入外部 backend 时假定已 load，且不在此处 unload。
    - progress_cb(done_chunks, total_chunks)，每块完成后回调。
    - 产出的 Project：stage=TRANSCRIBED，cues 由 segment_words 生成，
      词级时间戳保留在各 cue 的 words 字段。
    """
    path = Path(path)
    ffmpeg = media.find_ffmpeg(cfg.asr.ffmpeg_path or None)
    duration = media.probe_duration(path, ffmpeg=ffmpeg)
    silences = media.detect_silences(path, ffmpeg=ffmpeg)
    chunks = chunk_plan(duration, silences, cfg.asr.chunk_max_seconds)
    logger.info(
        "转录 %s：时长 %.1fs，%d 个静音区间，切成 %d 块",
        path,
        duration,
        len(silences),
        len(chunks),
    )

    own_backend = backend is None
    if own_backend:
        backend = create_backend(cfg)
        backend.load()

    all_words: list[WordTiming] = []
    try:
        with tempfile.TemporaryDirectory(prefix="subtitle_translator_") as tmp:
            for i, (cstart, cend) in enumerate(chunks):
                wav = Path(tmp) / f"chunk_{i:04d}.wav"
                media.extract_audio_chunk(path, cstart, cend, wav, ffmpeg=ffmpeg)
                _, words = backend.transcribe_chunk(wav, language=cfg.asr.language)
                all_words.extend(
                    WordTiming(text=w.text, start=w.start + cstart, end=w.end + cstart)
                    for w in words
                )
                if progress_cb is not None:
                    progress_cb(i + 1, len(chunks))
    finally:
        if own_backend:
            backend.unload()

    cues = segment_words(all_words)

    project = SubtitleProject()
    project.meta.source = SourceInfo(
        file=str(path),
        duration=duration,
        language=cfg.asr.language or "",
    )
    project.meta.models.asr = cfg.asr.model
    project.meta.models.aligner = cfg.asr.aligner_model
    project.cues = cues
    project.stage = Stage.TRANSCRIBED
    return project
