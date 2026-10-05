"""FastAPI 网页服务薄壳：REST + SSE 进度推送 + Range 视频流 + 前端静态托管。

业务逻辑全部在 core（pipeline / models / config），本包只做协议转换：

- :mod:`.tasks`：进程内任务注册表 + 单并发 asyncio worker（不引入外部队列）；
- :mod:`.app`：``create_app`` 应用工厂与全部路由。

``fastapi`` 在 ``web`` extra，这里延迟导入以保持 core 轻量。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI

__all__ = ["create_app"]


def create_app(
    config_path: Union[str, "Path"] = "config.yaml",
    frontend_dist: Optional[Union[str, "Path"]] = None,
) -> "FastAPI":
    """创建 FastAPI 应用。延迟导入，未安装 web extra 时 core 不受影响。"""
    from .app import create_app as _create_app

    return _create_app(config_path=config_path, frontend_dist=frontend_dist)
