"""进程内任务注册表 + 单并发 asyncio worker（localhost 单用户，不引入外部队列）。

任务状态机::

    pending ──→ running ──→ done
                  │  └──────→ failed
                  │  └──────→ waiting_confirm ──(resume)──→ pending
                  │  └──────→ cancelled
                  └ 取消标志协作式生效（当前 cue / 块完成后停）

- 同一时刻只跑一个任务（本地单 GPU），其余 pending 排队；
- 同步 pipeline 用 ``asyncio.to_thread`` 执行，pipeline 的 progress_cb 在
  worker 线程里被调用，事件经 ``loop.call_soon_threadsafe`` 推给 SSE 订阅队列；
- 取消是协作式：progress_cb 检查到取消标志后抛
  :class:`~subtitle_translator.pipeline.PipelineCancelledError`，translate
  阶段会把已翻译的 cue 落盘（stage 保持 contexted），resume / 重跑即续上；
- 每次执行任务时重新加载 config.yaml，网页改配置对后续任务生效。
- per-task 日志：任务开始时把 ``<log.dir>/tasks/<task_id>.log`` 的
  FileHandler 挂到 root logger（logsetup.create_task_log_handler，带脱敏
  过滤器和按 contextvars task_id 过滤的 TaskLogFilter），任务边界统一
  记录生命周期日志，失败时 logger.exception 落完整 traceback；任务结束
  （含 waiting_confirm）摘除 handler，文件保留供 ``GET /api/tasks/{id}/log``
  读取。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from ..config import load_config, resolve_api_key, resolve_log_dir
from .. import preflight
from ..logsetup import (
    TASK_LOG_DIR_NAME,
    create_task_log_handler,
    register_secret,
    reset_task_log_context,
    task_log_context,
)
from ..pipeline import (
    PIPELINE_STAGES,
    PipelineCancelledError,
    default_srt_path,
    find_media_files,
    run_batch,
    run_pipeline,
)
from ..translate import GlossaryNotConfirmedError

logger = logging.getLogger(__name__)

TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_DONE = "done"
TASK_FAILED = "failed"
TASK_CANCELLED = "cancelled"
TASK_WAITING_CONFIRM = "waiting_confirm"

TASK_STATES = (
    TASK_PENDING,
    TASK_RUNNING,
    TASK_DONE,
    TASK_FAILED,
    TASK_CANCELLED,
    TASK_WAITING_CONFIRM,
)

# SSE 流在这些状态下关闭（waiting_confirm 不是终态：确认术语后 resume 继续）
TERMINAL_STATES = (TASK_DONE, TASK_FAILED, TASK_CANCELLED)


@dataclass
class Task:
    """一个转录/翻译任务：path（文件或目录）+ 选项 + 事件历史。"""

    id: str
    path: str
    media: list[str]
    options: dict
    status: str = TASK_PENDING
    created_at: float = field(default_factory=time.time)
    events: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    cancel_requested: bool = False
    log_path: Optional[str] = None  # per-task 日志文件（任务开始执行时创建）

    def snapshot(self) -> dict:
        """列表 / 详情接口的序列化视图（progress 为最近一条事件快照）。

        artifacts 为已存在的导出产物（SRT）路径列表，前端据此给下载链接；
        options 为创建任务时的选项快照，前端详情页「运行信息」展示用。
        """
        return {
            "id": self.id,
            "path": self.path,
            "media": list(self.media),
            "options": dict(self.options),
            "status": self.status,
            "created_at": self.created_at,
            "error": self.error,
            "progress": self.events[-1] if self.events else None,
            "artifacts": self._artifacts(),
        }

    def _artifacts(self) -> list[str]:
        result = []
        for media in self.media:
            try:
                srt = default_srt_path(media)
                if srt.is_file():
                    result.append(str(srt))
            except OSError:
                continue
        return result


class TaskManager:
    """任务注册表 + 单并发 worker。由 app lifespan 启动 / 停止。"""

    def __init__(self, config_path: Union[str, Path]) -> None:
        self.config_path = Path(config_path)
        self.tasks: dict[str, Task] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._queue: Optional[asyncio.Queue] = None
        self._worker: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task_handlers: dict[str, logging.Handler] = {}

    # ---------------------------------------------------------- 生命周期

    def start(self) -> None:
        """在运行中的事件循环上启动 worker（app lifespan 调用，幂等）。"""
        if self._worker is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._work_loop())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        for task_id in list(self._task_handlers):
            self._detach_task_log(task_id)

    # ---------------------------------------------------------- 任务操作

    def create_task(self, path: Union[str, Path], options: dict) -> Task:
        """校验路径（文件/目录自动判断）、登记任务并排队。

        路径无效抛 FileNotFoundError / ValueError；目录里没有媒体文件抛
        ValueError。
        """
        media = find_media_files(path)
        if not media:
            raise ValueError(f"在 {path} 下没有找到媒体文件")
        task = Task(
            id=uuid.uuid4().hex[:12],
            path=str(path),
            media=[str(m) for m in media],
            options=dict(options),
        )
        self.tasks[task.id] = task
        self._enqueue(task)
        return task

    def get(self, task_id: str) -> Task:
        try:
            return self.tasks[task_id]
        except KeyError:
            raise KeyError(f"任务不存在: {task_id}") from None

    def list(self) -> list[Task]:
        """按创建时间排序的任务列表。"""
        return sorted(self.tasks.values(), key=lambda t: t.created_at)

    def cancel(self, task_id: str) -> Task:
        """取消任务。running 协作式生效；pending / waiting_confirm 立即取消。

        已终态的任务抛 ValueError。
        """
        task = self.get(task_id)
        if task.status == TASK_RUNNING:
            task.cancel_requested = True
        elif task.status in (TASK_PENDING, TASK_WAITING_CONFIRM):
            task.status = TASK_CANCELLED
            self._emit_status(task, "任务已取消")
        else:
            raise ValueError(f"任务已处于终态（{task.status}），无法取消")
        return task

    def resume(self, task_id: str) -> Task:
        """waiting_confirm 任务重新排队（术语表确认后继续逐句翻译）。"""
        task = self.get(task_id)
        if task.status != TASK_WAITING_CONFIRM:
            raise ValueError(f"只有 waiting_confirm 状态的任务可以 resume（当前 {task.status}）")
        task.error = None
        task.cancel_requested = False
        task.status = TASK_PENDING
        self._emit_status(task, "任务已重新排队")
        self._enqueue(task)
        return task

    # ---------------------------------------------------------- SSE 订阅

    def subscribe(self, task_id: str) -> asyncio.Queue:
        """新建订阅队列。调用方随后应同步快照 task.events 做历史回放
        （两者之间没有 await，不会丢事件）。"""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(task_id, []).append(queue)
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue) -> None:
        subscribers = self._subscribers.get(task_id)
        if subscribers and queue in subscribers:
            subscribers.remove(queue)

    # ---------------------------------------------------------- per-task 日志

    def _attach_task_log(self, task: Task) -> None:
        """为任务创建 ``<log.dir>/tasks/<task_id>.log`` handler（挂到 root logger）。

        handler 上的 TaskLogFilter 只放行该任务上下文（contextvars）里产生的
        记录；脱敏过滤器与全局一致。重复调用（resume 重跑）追加写同一文件。
        """
        log_dir = resolve_log_dir(load_config(self.config_path))
        log_path = log_dir / TASK_LOG_DIR_NAME / f"{task.id}.log"
        handler = create_task_log_handler(log_path, task.id)
        logging.getLogger().addHandler(handler)
        self._task_handlers[task.id] = handler
        task.log_path = str(log_path)

    def _detach_task_log(self, task_id: str) -> None:
        """任务结束后摘除 handler（日志文件保留，供 /log 端点读取）。"""
        handler = self._task_handlers.pop(task_id, None)
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()

    # ---------------------------------------------------------- 内部

    def _enqueue(self, task: Task) -> None:
        if self._queue is None:
            # 未走 lifespan 的场景（测试直接操作 manager）兜底
            self.start()
        self._queue.put_nowait(task.id)

    def _record(self, task: Task, event: dict) -> None:
        """记录事件并推给订阅者。可从 worker 线程或事件循环调用。

        统一补上 ``time``（epoch 秒）：前端详情页据此估算 ETA 与各阶段耗时，
        历史回放也保留真实发生时刻。
        """
        event = {**event, "time": time.time()}
        task.events.append(event)
        subscribers = self._subscribers.get(task.id) or []
        if self._loop is not None:
            for queue in subscribers:
                self._loop.call_soon_threadsafe(queue.put_nowait, event)

    def _emit_status(self, task: Task, message: str) -> None:
        """状态变更事件：沿用 pipeline event 五键 + task_id + status。"""
        self._record(
            task,
            {
                "task_id": task.id,
                "status": task.status,
                "media": task.path,
                "stage": "",
                "done": 0,
                "total": 0,
                "message": message,
            },
        )

    async def _work_loop(self) -> None:
        """单并发 worker：逐个取 pending 任务执行。"""
        while True:
            task_id = await self._queue.get()
            task = self.tasks.get(task_id)
            if task is None or task.status != TASK_PENDING:
                continue  # 排队期间被取消 / 状态已变化
            task.status = TASK_RUNNING
            self._attach_task_log(task)
            # 任务日志上下文：contextvars 随 asyncio.to_thread 传播进 worker
            # 线程，任务内产生的日志记录（含 pipeline/transcribe/translate）
            # 全部进入 per-task 日志文件
            token = task_log_context(task.id)
            try:
                self._emit_status(task, "任务开始")
                logger.info("任务 %s 开始：%s（options=%s）", task.id, task.path, task.options)
                await asyncio.to_thread(self._run_sync, task)
            except PipelineCancelledError:
                task.status = TASK_CANCELLED
                task.error = None
                logger.info("任务 %s 已取消", task.id)
                self._emit_status(task, "任务已取消，进度已保存")
            except GlossaryNotConfirmedError as exc:
                task.status = TASK_WAITING_CONFIRM
                task.error = str(exc)
                logger.info("任务 %s 进入术语表待确认：%s", task.id, exc)
                self._emit_status(task, "术语表待人工确认，等待网页确认后 resume")
            except Exception as exc:  # noqa: BLE001 - 任务边界统一兜底
                task.status = TASK_FAILED
                task.error = f"{type(exc).__name__}: {exc}"
                # 完整 traceback 进任务日志（脱敏过滤器已挂在 handler 上）
                logger.exception("任务 %s 失败", task.id)
                self._emit_status(task, f"任务失败：{task.error}")
            else:
                task.status = TASK_DONE
                logger.info("任务 %s 完成", task.id)
                self._emit_status(task, "任务完成")
            finally:
                reset_task_log_context(token)
                self._detach_task_log(task.id)

    def _run_sync(self, task: Task) -> None:
        """worker 线程里跑同步 pipeline。progress_cb 记录事件并检查取消标志。"""
        cfg = load_config(self.config_path)
        # 本次任务的 api_key 注册进脱敏过滤器（配置可能刚在网页改过）
        register_secret(resolve_api_key(cfg))
        # 轻量复核：模型路径 / ffmpeg 仍存在（防 serve 启动后运行期间文件被删），
        # 只查存在性，不下载不联网
        problems = preflight.recheck_model_paths(cfg)
        if problems:
            raise RuntimeError("任务启动前复核未通过：\n" + "\n".join(problems))
        options = task.options
        if options.get("language"):
            cfg.asr.language = options["language"]
        stages = (
            ("transcribe", "export") if options.get("transcribe_only") else PIPELINE_STAGES
        )

        def progress(event: dict) -> None:
            self._record(task, {**event, "task_id": task.id, "status": task.status})
            if task.cancel_requested:
                raise PipelineCancelledError("任务已被用户取消")

        kwargs = dict(
            stages=stages,
            auto_confirm=bool(options.get("auto_confirm")),
            bilingual=bool(options.get("bilingual", True)),
            progress_cb=progress,
        )
        path = Path(task.path)
        if path.is_dir():
            result = run_batch(path, cfg, **kwargs)
            if result.failed:
                details = "; ".join(f"{m.name}: {err}" for m, err in result.failed)
                raise RuntimeError(
                    f"{len(result.failed)}/{result.total} 个文件失败 — {details}"
                )
        else:
            run_pipeline(path, cfg, **kwargs)
