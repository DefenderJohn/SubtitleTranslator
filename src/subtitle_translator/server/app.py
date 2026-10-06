"""FastAPI 应用工厂：REST + SSE 进度推送 + Range 视频流 + 前端静态托管。

薄壳原则：业务逻辑全部走 pipeline / models / config，本模块只做
HTTP 协议转换与路径 / 参数校验。任务调度见 :mod:`.tasks`。

- SSE 事件直接透传 pipeline 的 event 协议（{media, stage, done, total,
  message}），附加 task_id 与 status 字段；订阅时先回放历史事件，
  后打开的页面能恢复进度；
- 视频流支持 HTTP Range（拖动进度条），单区间 bytes=start-end；
- api_key 不明文返回（masked），PUT 时空字符串 / mask 值表示不修改；
- 前端构建产物 frontend/dist 存在时挂载到 /（SPA 路由回退 index.html），
  否则给占位提示页。
"""

from __future__ import annotations

import json
import mimetypes
from contextlib import asynccontextmanager
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional, Union

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import (
    AsrConfig,
    Config,
    TranslateConfig,
    UiConfig,
    load_config,
    resolve_api_key,
    save_config,
)
from ..models import SubtitleProject
from ..pipeline import (
    PROJECT_SUFFIX,
    default_srt_path,
    find_media_files,
    project_json_path,
)
from ..srt import export_srt
from .tasks import TERMINAL_STATES, TaskManager

# 前端构建产物的默认位置：<repo>/frontend/dist（server/app.py 上四级为仓库根）
DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"

VIDEO_CHUNK_SIZE = 256 * 1024

_PLACEHOLDER_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>SubtitleTranslator</title></head>
<body style="font-family: sans-serif; max-width: 40em; margin: 4em auto;">
<h1>SubtitleTranslator 网页服务已启动</h1>
<p>前端尚未构建。请在仓库根目录执行 <code>cd frontend &amp;&amp; npm run build</code>
产出 <code>frontend/dist</code> 后重启服务。API 文档见 <a href="/docs">/docs</a>。</p>
</body></html>
"""


# ---------------------------------------------------------------------------
# 请求体模型
# ---------------------------------------------------------------------------


class TaskOptions(BaseModel):
    auto_confirm: bool = False
    bilingual: bool = True
    transcribe_only: bool = False
    language: Optional[str] = None


class CreateTaskRequest(BaseModel):
    path: str
    options: TaskOptions = Field(default_factory=TaskOptions)


class CuePatchRequest(BaseModel):
    path: str
    cue_id: int
    text: Optional[str] = None
    translation: Optional[str] = None


class GlossaryUpdate(BaseModel):
    src: str  # 匹配键
    new_src: Optional[str] = None  # 改原文
    dst: Optional[str] = None  # 改译文


class GlossaryPatchRequest(BaseModel):
    path: str
    updates: list[GlossaryUpdate] = Field(default_factory=list)
    confirm: list[str] = Field(default_factory=list)  # 按 src 确认单条
    confirm_all: bool = False


class ExportRequest(BaseModel):
    path: str
    bilingual: bool = True


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _resolve_json_path(path: Union[str, Path]) -> Path:
    """接受 .sub.json 或媒体文件路径，统一成工程 JSON 路径。"""
    p = Path(path)
    if not p.name.endswith(PROJECT_SUFFIX):
        p = project_json_path(p)
    return p


def _load_project_or_404(path: Union[str, Path]) -> tuple[SubtitleProject, Path]:
    json_path = _resolve_json_path(path)
    if not json_path.is_file():
        raise HTTPException(status_code=404, detail=f"工程文件不存在: {json_path}")
    try:
        return SubtitleProject.load(json_path), json_path
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"工程文件无法解析 — {exc}") from exc


def _mask_api_key(key: Optional[str]) -> Optional[str]:
    """api_key 不明文返回：'sk-abcdef' -> 'sk-...***'。"""
    if not key:
        return key
    return key[:3] + "...***" if len(key) > 3 else "***"


def _default_browse_root() -> Path:
    """目录浏览默认起点：用户主目录，不可用（无法解析/不可读）时退回进程工作目录。"""
    try:
        home = Path.home()
        if home.is_dir():
            return home
    except (RuntimeError, OSError):
        pass
    return Path.cwd()


def _is_dir_quiet(path: Path) -> bool:
    """is_dir 的容错版：stat 失败（无权限、悬挂链接等）按非目录处理。"""
    try:
        return path.is_dir()
    except OSError:
        return False


def _config_payload(cfg: Config) -> dict:
    data = {"asr": asdict(cfg.asr), "translate": asdict(cfg.translate), "ui": asdict(cfg.ui)}
    data["translate"]["api_key"] = _mask_api_key(cfg.translate.api_key)
    # 告知前端密钥是否已可用（可能来自 api_key_env 环境变量），不泄露值
    data["translate"]["api_key_resolved"] = resolve_api_key(cfg) is not None
    return data


def _parse_range(header: str, size: int) -> tuple[int, int]:
    """解析单区间 Range 头（bytes=start-end / start- / -suffix），返回闭区间。

    非法或超出范围抛 ValueError（调用方转 416）。
    """
    if not header.startswith("bytes=") or "," in header:
        raise ValueError(f"不支持的 Range 头: {header!r}")
    start_s, _, end_s = header[len("bytes=") :].partition("-")
    try:
        if start_s == "":
            # 后缀区间：最后 N 字节
            suffix = int(end_s)
            if suffix <= 0:
                raise ValueError
            start, end = max(size - suffix, 0), size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
    except ValueError:
        raise ValueError(f"不支持的 Range 头: {header!r}") from None
    if start < 0 or start >= size or end < start:
        raise ValueError(f"Range 超出范围: {header!r}（文件 {size} 字节）")
    return start, min(end, size - 1)


def _iter_file_range(path: Path, start: int, end: int):
    with path.open("rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(VIDEO_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


# ---------------------------------------------------------------------------
# 应用工厂
# ---------------------------------------------------------------------------


def create_app(
    config_path: Union[str, Path] = "config.yaml",
    frontend_dist: Optional[Union[str, Path]] = None,
) -> FastAPI:
    config_path = Path(config_path)
    manager = TaskManager(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.start()
        yield
        await manager.stop()

    app = FastAPI(title="SubtitleTranslator", lifespan=lifespan)
    app.state.manager = manager
    app.state.config_path = config_path

    # 开发时 vite dev server 端口不定：允许 localhost / 127.0.0.1 任意端口
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _get_task(task_id: str):
        try:
            return manager.get(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # ---------------------------------------------------------- 任务

    @app.post("/api/tasks", status_code=201)
    async def create_task(req: CreateTaskRequest):
        # async def：在事件循环线程里入队（asyncio.Queue 非线程安全）
        try:
            task = manager.create_task(req.path, req.options.model_dump())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return task.snapshot()

    @app.get("/api/tasks")
    def list_tasks():
        return [task.snapshot() for task in manager.list()]

    @app.get("/api/tasks/{task_id}")
    def task_detail(task_id: str):
        return _get_task(task_id).snapshot()

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel_task(task_id: str):
        _get_task(task_id)
        try:
            return manager.cancel(task_id).snapshot()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/tasks/{task_id}/resume")
    async def resume_task(task_id: str):
        _get_task(task_id)
        try:
            return manager.resume(task_id).snapshot()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/tasks/{task_id}/events")
    async def task_events(task_id: str):
        """SSE 订阅：先回放历史事件，再实时推送；终态后关闭流。"""
        task = _get_task(task_id)

        async def stream():
            queue = manager.subscribe(task_id)
            # subscribe 与快照之间没有 await，事件不会丢
            history = list(task.events)
            try:
                for event in history:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if task.status in TERMINAL_STATES:
                    return
                while True:
                    event = await queue.get()
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event.get("status") in TERMINAL_STATES:
                        return
            finally:
                manager.unsubscribe(task_id, queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---------------------------------------------------------- 媒体

    @app.get("/api/media")
    def browse_media(path: str = ""):
        p = Path(path) if path else _default_browse_root()
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"路径不存在: {p}")
        try:
            media = find_media_files(p)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        directories = []
        if p.is_dir():
            # 单个子目录无权限（/root、挂载点残留等）跳过；目录本身不可读则返回空清单
            try:
                children = list(p.iterdir())
            except OSError:
                children = []
            directories = sorted(d.name for d in children if _is_dir_quiet(d))
        return {
            "path": str(p),
            "parent": str(p.parent) if p.parent != p else None,
            "directories": directories,
            "media": [str(m) for m in media],
        }

    @app.get("/api/video")
    def stream_video(path: str, range: Optional[str] = Header(None)):
        """视频流，支持单区间 Range（bytes=start-end），供预览拖动进度条。"""
        p = Path(path)
        if not p.is_file():
            raise HTTPException(status_code=404, detail=f"文件不存在: {p}")
        size = p.stat().st_size
        content_type = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        headers = {"Accept-Ranges": "bytes"}
        if range:
            try:
                start, end = _parse_range(range, size)
            except ValueError as exc:
                raise HTTPException(
                    status_code=416,
                    detail=str(exc),
                    headers={"Content-Range": f"bytes */{size}"},
                ) from exc
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            headers["Content-Length"] = str(end - start + 1)
            return StreamingResponse(
                _iter_file_range(p, start, end),
                status_code=206,
                media_type=content_type,
                headers=headers,
            )
        headers["Content-Length"] = str(size)
        return StreamingResponse(
            _iter_file_range(p, 0, size - 1), media_type=content_type, headers=headers
        )

    # ---------------------------------------------------------- 工程数据

    @app.get("/api/project")
    def read_project(path: str):
        project, _ = _load_project_or_404(path)
        return project.to_dict()

    @app.patch("/api/project/cues")
    def patch_cue(req: CuePatchRequest):
        project, json_path = _load_project_or_404(req.path)
        cue = next((c for c in project.cues if c.id == req.cue_id), None)
        if cue is None:
            raise HTTPException(status_code=404, detail=f"cue 不存在: id={req.cue_id}")
        if req.text is not None:
            cue.text = req.text
        if req.translation is not None:
            cue.translation = req.translation
        project.save(json_path)
        return cue.to_dict()

    @app.get("/api/project/glossary")
    def read_glossary(path: str):
        project, _ = _load_project_or_404(path)
        return {"path": str(_resolve_json_path(path)), "glossary": [g.to_dict() for g in project.glossary]}

    @app.patch("/api/project/glossary")
    def patch_glossary(req: GlossaryPatchRequest):
        project, json_path = _load_project_or_404(req.path)
        by_src = {g.src: g for g in project.glossary}
        for update in req.updates:
            entry = by_src.get(update.src)
            if entry is None:
                raise HTTPException(status_code=404, detail=f"术语不存在: {update.src}")
            if update.new_src:
                entry.src = update.new_src
                by_src[update.new_src] = entry
            if update.dst is not None:
                entry.dst = update.dst
        for src in req.confirm:
            entry = by_src.get(src)
            if entry is None:
                raise HTTPException(status_code=404, detail=f"术语不存在: {src}")
            entry.confirmed = True
        if req.confirm_all:
            for entry in project.glossary:
                entry.confirmed = True
        project.save(json_path)
        return {"path": str(json_path), "glossary": [g.to_dict() for g in project.glossary]}

    @app.post("/api/project/export")
    def export_project(req: ExportRequest):
        project, json_path = _load_project_or_404(req.path)
        srt_path = default_srt_path(json_path, from_json=True)
        export_srt(project, srt_path, bilingual=req.bilingual)
        return {"srt_path": str(srt_path)}

    # ---------------------------------------------------------- 配置

    @app.get("/api/config")
    def read_config():
        return _config_payload(load_config(config_path))

    @app.put("/api/config")
    def update_config(payload: dict = Body(...)):
        """局部更新 config.yaml。api_key 传空字符串或 mask 值表示不修改。"""
        section_classes = {"asr": AsrConfig, "translate": TranslateConfig, "ui": UiConfig}
        unknown_sections = set(payload) - set(section_classes)
        if unknown_sections:
            raise HTTPException(status_code=400, detail=f"未知配置节: {sorted(unknown_sections)}")
        cfg = load_config(config_path)
        masked_key = _mask_api_key(cfg.translate.api_key)
        for section, cls in section_classes.items():
            values = payload.get(section)
            if values is None:
                continue
            if not isinstance(values, dict):
                raise HTTPException(status_code=400, detail=f"{section} 节必须是对象")
            known = {f.name for f in fields(cls)}
            unknown_fields = set(values) - known
            if unknown_fields:
                raise HTTPException(
                    status_code=400, detail=f"{section} 节未知字段: {sorted(unknown_fields)}"
                )
            target = getattr(cfg, section)
            for key, value in values.items():
                if section == "translate" and key == "api_key":
                    # 空字符串 / 原样回传的 mask 值 = 不修改
                    if value in ("", None, masked_key):
                        continue
                setattr(target, key, value)
            # dataclass 的 setattr 不校验，重建一次触发 __post_init__ 等检查
            try:
                cls(**asdict(target))
            except (ValueError, TypeError) as exc:
                raise HTTPException(status_code=400, detail=f"{section} 节配置非法: {exc}") from exc
        # 文件里已有 / 新设了明文 key 时保留之，其余情况不落盘明文
        save_config(cfg, config_path, include_api_key=bool(cfg.translate.api_key))
        return _config_payload(cfg)

    # ---------------------------------------------------------- 前端静态托管（最后注册，避免挡住 /api）

    dist = Path(frontend_dist) if frontend_dist else DEFAULT_FRONTEND_DIST
    if dist.is_dir():
        dist_root = dist.resolve()
        index_html = dist_root / "index.html"

        def _serve_spa(full_path: str):
            """托管前端构建产物；前端路由（如 /tasks/xxx）回退到 index.html。"""
            if full_path:
                candidate = (dist_root / full_path).resolve()
                # 防目录穿越：必须落在 dist 内
                if candidate.is_file() and candidate.is_relative_to(dist_root):
                    return FileResponse(candidate)
            return FileResponse(index_html)

        @app.get("/", include_in_schema=False)
        def spa_index():
            return _serve_spa("")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa_fallback(full_path: str):
            return _serve_spa(full_path)
    else:

        @app.get("/", response_class=HTMLResponse)
        def placeholder():
            return _PLACEHOLDER_HTML

    return app
