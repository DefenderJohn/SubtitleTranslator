"""OpenAI 兼容端点 HTTP 客户端封装。

基于官方 ``openai`` SDK 的 chat.completions（非 streaming）。本地 vLLM /
Ollama 等 OpenAI 兼容端点走同一接口。对 429 / 5xx / 超时 / 连接错误做
指数退避重试，SDK 自带的重试关闭（``max_retries=0``），由本层统一控制。

结构化输出：``chat_json`` 强制携带 ``response_format={"type": "json_schema",
...}``（OpenAI strict 模式 / vLLM guided decoding 同一形态），不做任何探测
或降级——端点不支持时直接把 400 错误抛清晰，由用户换端点。
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional, Union

from openai import APIConnectionError, APITimeoutError, OpenAI

from ..config import Config, TranslateConfig, resolve_api_key

BACKOFF_BASE_SECONDS = 1.0  # 退避基数：第 n 次重试等待 base * 2^n 秒
BACKOFF_CAP_SECONDS = 30.0  # 单次退避上限

Messages = list[dict[str, str]]


class ChatError(RuntimeError):
    """chat 请求在重试耗尽后仍失败，或遇到不可重试的错误。"""


class InvalidModelJsonError(ValueError):
    """JSON 模式下端点仍返回了无法解析为 JSON 对象的内容。"""


def _is_retryable(exc: BaseException) -> bool:
    """429 / 5xx / 超时 / 连接错误可重试，其余（如 400 鉴权）直接抛。"""
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            status = int(status)
        except (TypeError, ValueError):
            return False
        return status == 429 or status >= 500
    return isinstance(exc, (APITimeoutError, APIConnectionError))


class ChatClient:
    """chat.completions 封装：``chat`` 纯文本 / ``chat_json`` JSON schema 输出。"""

    def __init__(
        self,
        cfg: Union[Config, TranslateConfig],
        *,
        backoff_base: float = BACKOFF_BASE_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        translate = cfg.translate if isinstance(cfg, Config) else cfg
        self.cfg = translate
        self.backoff_base = backoff_base
        self._sleep = sleep
        self._client = OpenAI(
            base_url=translate.base_url,
            api_key=resolve_api_key(translate) or "EMPTY",
            timeout=translate.request_timeout,
            max_retries=0,
        )

    def _create(
        self,
        messages: Messages,
        temperature: float,
        response_format: Optional[dict] = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        if not resp.choices:
            return ""
        content = resp.choices[0].message.content
        return content or ""

    def chat(
        self,
        messages: Messages,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """发送一轮对话并返回文本内容；失败按指数退避重试，耗尽后抛 ChatError。"""
        if temperature is None:
            temperature = self.cfg.temperature
        for attempt in range(self.cfg.max_retries + 1):
            try:
                return self._create(messages, temperature, response_format)
            except Exception as exc:
                if not _is_retryable(exc) or attempt >= self.cfg.max_retries:
                    hint = ""
                    if (
                        response_format is not None
                        and getattr(exc, "status_code", None) == 400
                    ):
                        hint = (
                            "（该端点可能不支持 response_format json_schema "
                            "结构化输出，请确认端点支持该能力或更换端点）"
                        )
                    raise ChatError(
                        f"chat 请求失败（第 {attempt + 1} 次尝试）{hint}: {exc}"
                    ) from exc
                self._sleep(min(self.backoff_base * (2**attempt), BACKOFF_CAP_SECONDS))
        raise ChatError("chat 请求失败：重试耗尽")  # pragma: no cover - 不可达

    def chat_json(
        self,
        messages: Messages,
        *,
        schema: dict,
        name: str,
        temperature: Optional[float] = None,
    ) -> dict:
        """JSON schema 结构化输出：强制 response_format，返回解析后的 dict。

        端点不支持 json_schema（HTTP 400）时抛带提示的 ChatError；返回内容
        不是合法 JSON 对象时抛 InvalidModelJsonError。
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema},
        }
        text = self.chat(messages, temperature=temperature, response_format=response_format)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvalidModelJsonError(
                f"模型输出不是合法 JSON: {text[:200]!r}"
            ) from exc
        if not isinstance(data, dict):
            raise InvalidModelJsonError(f"模型输出不是 JSON 对象: {text[:200]!r}")
        return data
