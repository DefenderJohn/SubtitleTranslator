"""流水线编排 + 断点续传 + 批量处理。

音视频 -> 转录 -> 分段 ->（摘要 + 术语表，人工确认检查点）->
滑动窗口逐句翻译 -> 导出 SRT / 双语 SRT。

断点续传：每个媒体文件对应一个 ``xxx.sub.json`` 工程文件（唯一事实
来源），每完成一个阶段立即落盘；重跑时按 ``meta.stage`` 跳过已完成
阶段（empty -> transcribed -> contexted -> translated）。

进度回调协议（网页 SSE 直接复用）::

    progress_cb({"media": str, "stage": str, "done": int, "total": int, "message": str})

- ``stage`` 取值：``"transcribe"`` / ``"translate"`` / ``"export"``；
- 阶段开始 / 结束时 ``done=0, total=0``，信息在 ``message``；
- 阶段进行中 ``done/total`` 为实际进度（转录=块数，翻译=cue 条数）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from .config import Config
from .models import Stage, SubtitleProject
from .srt import export_srt
from .transcribe import AsrBackend, transcribe_media
from .translate import GlossaryNotConfirmedError, SlidingWindowStrategy, TranslationStrategy

logger = logging.getLogger(__name__)

# 媒体文件扩展名集合：旧项目的十个 + 常见视频容器
MEDIA_EXTENSIONS = frozenset(
    {
        "flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm",
        "mkv", "mov", "avi", "m4v",
    }
)

# 工程文件后缀：xxx.mp4 -> xxx.sub.json
PROJECT_SUFFIX = ".sub.json"

# 可编排的阶段名（stages 参数的合法取值）
PIPELINE_STAGES = ("transcribe", "translate", "export")

_STAGE_ORDER = (Stage.EMPTY, Stage.TRANSCRIBED, Stage.CONTEXTED, Stage.TRANSLATED)

# progress_cb 协议：event dict，键固定为 media/stage/done/total/message
ProgressCb = Callable[[dict], None]


def _stage_reached(stage: Stage, target: Stage) -> bool:
    """stage 是否已到达（>=）target 阶段。"""
    return _STAGE_ORDER.index(stage) >= _STAGE_ORDER.index(target)


def _emit(
    cb: Optional[ProgressCb],
    media: Union[str, Path],
    stage: str,
    done: int = 0,
    total: int = 0,
    message: str = "",
) -> None:
    if cb is not None:
        cb(
            {
                "media": str(media),
                "stage": stage,
                "done": done,
                "total": total,
                "message": message,
            }
        )


def project_json_path(media_path: Union[str, Path]) -> Path:
    """媒体文件对应的工程 JSON 路径：``xxx.mp4`` -> ``xxx.sub.json``。"""
    return Path(media_path).with_suffix(PROJECT_SUFFIX)


def default_srt_path(path: Union[str, Path], from_json: bool = False) -> Path:
    """默认 SRT 导出路径：``xxx.mp4`` / ``xxx.sub.json`` -> ``xxx.srt``。"""
    path = Path(path)
    if from_json and path.name.endswith(PROJECT_SUFFIX):
        return path.with_name(path.name[: -len(PROJECT_SUFFIX)] + ".srt")
    return path.with_suffix(".srt")


def find_media_files(path: Union[str, Path]) -> list[Path]:
    """找出 path 下的全部媒体文件。

    - path 是文件：校验扩展名后直接使用（不合法抛 ValueError）；
    - path 是目录：递归查找全部媒体文件，按路径排序；
    - path 不存在抛 FileNotFoundError。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    if path.is_file():
        if path.suffix.lower().lstrip(".") not in MEDIA_EXTENSIONS:
            raise ValueError(
                f"不支持的媒体格式: {path.name}"
                f"（支持: {', '.join(sorted(MEDIA_EXTENSIONS))}）"
            )
        return [path]
    return sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower().lstrip(".") in MEDIA_EXTENSIONS
    )


def run_pipeline(
    media_path: Union[str, Path],
    cfg: Config,
    *,
    stages: tuple[str, ...] = PIPELINE_STAGES,
    auto_confirm: bool = False,
    bilingual: bool = True,
    progress_cb: Optional[ProgressCb] = None,
    backend: Optional[AsrBackend] = None,
    strategy: Optional[TranslationStrategy] = None,
    from_json: bool = False,
) -> SubtitleProject:
    """对单个媒体文件（或已有工程 JSON）跑流水线，返回最终 project。

    - 断点续传：``xxx.sub.json`` 已存在时加载，按 stage 跳过已完成阶段；
      每完成一个阶段立即落盘（崩溃也不丢进度）。术语表检查点失败
      （GlossaryNotConfirmedError）时也会先落盘再抛出，确认术语后重跑
      即可从逐句翻译续上。
    - ``stages`` 控制跑哪些阶段，子集 + 顺序固定；``--transcribe-only``
      即 ``stages=("transcribe", "export")``，导出原文 SRT。
    - ``from_json=True``：media_path 直接是已有 .sub.json，跳过转录、
      不碰媒体文件（用于重翻 / 调术语后重跑）。
    - backend / strategy 可注入（测试与复用场景）；为 None 时按 cfg 自建。
    """
    unknown = set(stages) - set(PIPELINE_STAGES)
    if unknown:
        raise ValueError(
            f"未知阶段: {sorted(unknown)}，合法取值为: {', '.join(PIPELINE_STAGES)}"
        )
    media_path = Path(media_path)
    json_path = media_path if from_json else project_json_path(media_path)

    if json_path.exists():
        project = SubtitleProject.load(json_path)
        logger.info("加载已有工程 %s（stage=%s）", json_path, project.stage.value)
    else:
        if from_json:
            raise FileNotFoundError(f"--from-json 指定的工程文件不存在: {json_path}")
        project = SubtitleProject()

    # ---- 转录 ----------------------------------------------------------
    if "transcribe" in stages and not _stage_reached(project.stage, Stage.TRANSCRIBED):
        if from_json:
            raise ValueError(
                f"{json_path} 尚未转录（stage={project.stage.value}），"
                "--from-json 模式不接触媒体文件，请先对媒体文件跑转录"
            )
        _emit(progress_cb, media_path, "transcribe", message="开始转录")
        project = transcribe_media(
            media_path,
            cfg,
            backend=backend,
            progress_cb=lambda d, t: _emit(
                progress_cb, media_path, "transcribe", d, t, f"转录块 {d}/{t}"
            ),
        )
        project.save(json_path)
        _emit(progress_cb, media_path, "transcribe", message=f"转录完成，已保存 {json_path}")

    # ---- 摘要 + 术语表 + 逐句翻译 --------------------------------------
    if "translate" in stages and not _stage_reached(project.stage, Stage.TRANSLATED):
        if not _stage_reached(project.stage, Stage.TRANSCRIBED):
            raise ValueError(
                f"{json_path} 尚未转录（stage={project.stage.value}），无法翻译；"
                "请先跑 transcribe 阶段"
            )
        strategy = strategy or SlidingWindowStrategy()
        _emit(progress_cb, media_path, "translate", message="开始翻译（摘要→术语表→逐句）")
        try:
            strategy.translate(
                project,
                cfg,
                progress_cb=lambda d, t: _emit(
                    progress_cb, media_path, "translate", d, t, f"翻译 {d}/{t}"
                ),
                auto_confirm=auto_confirm,
            )
        except GlossaryNotConfirmedError:
            # 人工确认检查点：context 产物（摘要+术语表）先落盘，确认后重跑续上
            project.save(json_path)
            _emit(
                progress_cb,
                media_path,
                "translate",
                message="术语表待人工确认，已保存工程文件",
            )
            raise
        project.save(json_path)
        _emit(progress_cb, media_path, "translate", message=f"翻译完成，已保存 {json_path}")

    # ---- 导出 SRT --------------------------------------------------------
    if "export" in stages:
        srt_path = default_srt_path(media_path, from_json=from_json)
        export_srt(project, srt_path, bilingual=bilingual)
        _emit(progress_cb, media_path, "export", message=f"已导出 {srt_path}")

    return project


@dataclass
class BatchResult:
    """批量处理汇总：成功清单与失败清单（含错误信息）。"""

    succeeded: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.succeeded) + len(self.failed)


def run_batch(
    path: Union[str, Path],
    cfg: Config,
    *,
    stages: tuple[str, ...] = PIPELINE_STAGES,
    auto_confirm: bool = False,
    bilingual: bool = True,
    progress_cb: Optional[ProgressCb] = None,
    backend: Optional[AsrBackend] = None,
    strategy: Optional[TranslationStrategy] = None,
) -> BatchResult:
    """遍历 path 下的媒体文件逐个跑流水线；单个失败记录日志继续下一个。

    backend / strategy 传入时在全部文件间复用（避免每文件重建模型/客户端）。
    """
    media_files = find_media_files(path)
    result = BatchResult()
    if not media_files:
        logger.warning("在 %s 下没有找到媒体文件", path)
        return result
    for media_path in media_files:
        try:
            run_pipeline(
                media_path,
                cfg,
                stages=stages,
                auto_confirm=auto_confirm,
                bilingual=bilingual,
                progress_cb=progress_cb,
                backend=backend,
                strategy=strategy,
            )
        except Exception as exc:  # noqa: BLE001 - 批量模式不中断，逐个记录
            logger.error("处理 %s 失败: %s", media_path, exc)
            result.failed.append((media_path, str(exc)))
        else:
            result.succeeded.append(media_path)
    return result
