"""本地 mock OpenAI 兼容端点，用于验证翻译链路的真实 HTTP 路径。

不依赖第三方库（stdlib http.server），实现 POST /v1/chat/completions。
翻译层全部走 JSON schema 结构化输出：按请求里 response_format 的
json_schema.name 回声符合 schema 的 JSON：

- ``summary`` → ``{"summary": 固定摘要}``
- ``glossary`` → ``{"entries": [{src, dst, count}]}``
- ``translation`` → ``{"translation": "【译】<原文>"}``（从 user 抽待译原文）

请求不带 response_format 时退回纯文本形态（摘要文本 / ``原文 | 译文 | 次数``
行 / 译文文本），用于对照旧行为。

``--reject-response-format`` 模拟不支持结构化输出的端点：凡带
response_format 的请求一律 400，用于验证客户端报错提示。

用法：
    python scripts/mock_translate_server.py --port 8399
然后把 config.yaml 的 translate.base_url 指向 http://127.0.0.1:8399/v1 即可。
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MOCK_SUMMARY = "（mock 摘要）一段英文演讲片段，语言口语化，无特殊专有名词。"
MOCK_GLOSSARY_ENTRIES = [
    {"src": "Hello", "dst": "你好", "count": 2},
    {"src": "World", "dst": "世界", "count": 1},
]

REJECT_RESPONSE_FORMAT = False  # --reject-response-format 置真


def _translation_source(messages: list[dict]) -> str:
    user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    # 逐句翻译：抽出「请翻译以下字幕原文：」之后、前瞻块之前的原文
    body = user.split("请翻译以下字幕原文：", 1)[-1]
    return body.split("\n\n以下为后续原文", 1)[0].strip()


def _mock_reply(messages: list[dict], schema_name: str | None) -> str:
    if schema_name == "summary":
        return json.dumps({"summary": MOCK_SUMMARY}, ensure_ascii=False)
    if schema_name == "glossary":
        return json.dumps({"entries": MOCK_GLOSSARY_ENTRIES}, ensure_ascii=False)
    if schema_name == "translation":
        return json.dumps(
            {"translation": f"【译】{_translation_source(messages)}"},
            ensure_ascii=False,
        )
    # 无 response_format：纯文本对照路径
    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    if "影视内容分析助手" in system:
        return MOCK_SUMMARY
    if "术语管理助手" in system:
        return "\n".join(f"{e['src']} | {e['dst']} | {e['count']}" for e in MOCK_GLOSSARY_ENTRIES)
    return f"【译】{_translation_source(messages)}"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib 约定
        if not self.path.rstrip("/").endswith("/v1/chat/completions"):
            self.send_error(404, "only /v1/chat/completions is implemented")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            response_format = payload.get("response_format")
            if REJECT_RESPONSE_FORMAT and response_format is not None:
                self.send_error(
                    400,
                    "unsupported parameter: response_format "
                    "(this mock simulates an endpoint without json_schema support)",
                )
                return
            schema_name = None
            if isinstance(response_format, dict):
                schema_name = (response_format.get("json_schema") or {}).get("name")
            content = _mock_reply(payload.get("messages") or [], schema_name)
        except Exception as exc:  # noqa: BLE001 - mock 端点，统一 400
            self.send_error(400, f"bad request: {exc}")
            return
        resp = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": payload.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib 签名
        pass


def main() -> None:
    global REJECT_RESPONSE_FORMAT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8399)
    parser.add_argument(
        "--reject-response-format",
        action="store_true",
        help="模拟不支持结构化输出的端点：带 response_format 的请求一律 400",
    )
    args = parser.parse_args()
    REJECT_RESPONSE_FORMAT = args.reject_response_format
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"mock OpenAI 端点：http://{args.host}:{args.port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
