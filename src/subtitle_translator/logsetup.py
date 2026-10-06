"""日志统一配置与脱敏（cli 入口与 server 启动共用本模块）。

- ``setup_logging``：console handler（简洁格式，级别取 ``log.level``）
  + RotatingFileHandler（``log.dir/subtitle-translator.log``，10MB×5，
  DEBUG 起，含时间/模块/行号），幂等（重复调用不叠加 handler）；
- ``SanitizeFilter``：对每条记录的 message 做脱敏——注册的密钥精确替换、
  通用密钥模式（sk-xxx / Bearer xxx / api_key=xxx / URL query 里的
  key/token）打码、用户主目录前缀替换为 ``~``；console 与 file handler
  （含 per-task handler）统一挂载同一个全局实例；
- per-task 日志：任务执行期间用 contextvars 记录当前 task_id
  （``task_log_context``），``create_task_log_handler`` 建的 handler 用
  ``TaskLogFilter`` 只放行本任务的记录（asyncio.to_thread 会把 context
  传播进 worker 线程）。
"""

from __future__ import annotations

import contextvars
import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

from .config import Config, LogConfig, resolve_log_dir

# 主日志文件名 / per-task 日志子目录名
LOG_FILE_NAME = "subtitle-translator.log"
TASK_LOG_DIR_NAME = "tasks"

# RotatingFileHandler：10MB × 5
FILE_MAX_BYTES = 10 * 1024 * 1024
FILE_BACKUP_COUNT = 5

_CONSOLE_FORMAT = "%(levelname)s [%(name)s] %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s"

# 标记本模块安装的 handler，保证 setup_logging 幂等
_HANDLER_MARK = "_subtitle_translator_handler"

# 注册的密钥太短（如占位符 "0"）做精确替换会毁掉正常日志，只收足够长的
_MIN_SECRET_LEN = 8

_SECRET_PATTERNS = [
    # sk- 开头的 API key（OpenAI / DeepSeek 等）
    (re.compile(r"sk-[A-Za-z0-9]{8,}"), "sk-***"),
    # Authorization: Bearer xxx
    (re.compile(r"Bearer\s+\S+", re.IGNORECASE), "Bearer ***"),
    # api_key=xxx / token=xxx 等键值形态（含 URL query），不吞掉 & 之后的参数
    (
        re.compile(
            r"(?i)\b(api_key|apikey|access_token|token|key)=([^\s&]+)"
        ),
        None,  # 用分组替换，见 sanitize()
    ),
]


class SanitizeFilter(logging.Filter):
    """日志脱敏过滤器：清洗 message 中的密钥与用户主目录路径。"""

    def __init__(self) -> None:
        super().__init__()
        self._secrets: set[str] = set()
        try:
            self._home = str(Path.home())
        except RuntimeError:
            self._home = ""

    def register_secret(self, value: Optional[str]) -> None:
        """注册一个精确替换的密钥值（如 resolve_api_key 解析出的 api_key）。"""
        if value and len(value) >= _MIN_SECRET_LEN:
            self._secrets.add(value)

    def sanitize(self, text: str) -> str:
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, "***")
        for pattern, replacement in _SECRET_PATTERNS:
            if replacement is None:
                text = pattern.sub(lambda m: f"{m.group(1)}=***", text)
            else:
                text = pattern.sub(replacement, text)
        if self._home and self._home != "/" and self._home in text:
            text = text.replace(self._home, "~")
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - 格式化失败的记录不清洗也不拦截
            return True
        sanitized = self.sanitize(message)
        if sanitized != message:
            record.msg = sanitized
            record.args = None
        return True


# 全局唯一脱敏过滤器实例：挂在所有 handler 上，register_secret 全局生效
sanitizer = SanitizeFilter()


def register_secret(value: Optional[str]) -> None:
    """把密钥值注册进全局脱敏过滤器（精确替换）。"""
    sanitizer.register_secret(value)


# ---------------------------------------------------------------------------
# per-task 日志上下文（contextvars，随 asyncio.to_thread 传播进 worker 线程）
# ---------------------------------------------------------------------------

_current_task_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "subtitle_translator_task_id", default=None
)


def task_log_context(task_id: str) -> contextvars.Token:
    """进入任务日志上下文：此后本 context 产生的日志记录归属该任务。"""
    return _current_task_id.set(task_id)


def reset_task_log_context(token: contextvars.Token) -> None:
    _current_task_id.reset(token)


class TaskLogFilter(logging.Filter):
    """只放行指定 task_id 上下文里产生的记录。"""

    def __init__(self, task_id: str) -> None:
        super().__init__()
        self.task_id = task_id

    def filter(self, record: logging.LogRecord) -> bool:
        return _current_task_id.get() == self.task_id


def create_task_log_handler(log_path: Union[str, Path], task_id: str) -> logging.Handler:
    """建 per-task FileHandler（追加写、完整格式、挂脱敏 + task 过滤）。"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    handler.addFilter(sanitizer)
    handler.addFilter(TaskLogFilter(task_id))
    setattr(handler, _HANDLER_MARK, True)
    return handler


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------


def setup_logging(cfg: Optional[Union[Config, LogConfig]] = None) -> Path:
    """配置 root logger：console（简洁）+ 滚动文件（完整），幂等。

    返回日志文件路径。重复调用不叠加 handler，只按新配置刷新级别。
    """
    if cfg is None:
        log_cfg = LogConfig()
    elif isinstance(cfg, Config):
        log_cfg = cfg.log
    else:
        log_cfg = cfg
    log_dir = resolve_log_dir(log_cfg)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAME
    console_level = getattr(logging, log_cfg.level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = None
    rotating = None
    for handler in root.handlers:
        if not getattr(handler, _HANDLER_MARK, False):
            continue
        if isinstance(handler, RotatingFileHandler):
            rotating = handler
        elif isinstance(handler, logging.StreamHandler):
            console = handler

    if console is None:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        console.addFilter(sanitizer)
        setattr(console, _HANDLER_MARK, True)
        root.addHandler(console)
    console.setLevel(console_level)

    if rotating is None:
        rotating = RotatingFileHandler(
            log_file,
            maxBytes=FILE_MAX_BYTES,
            backupCount=FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        rotating.setFormatter(logging.Formatter(_FILE_FORMAT))
        rotating.addFilter(sanitizer)
        setattr(rotating, _HANDLER_MARK, True)
        root.addHandler(rotating)
    rotating.setLevel(logging.DEBUG)

    return log_file


def shutdown_logging() -> None:
    """摘除并关闭本模块安装的 root handler（测试隔离 / 重新配置用）。"""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
            handler.close()
