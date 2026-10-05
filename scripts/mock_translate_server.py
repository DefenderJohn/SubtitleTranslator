"""本地 mock OpenAI 兼容端点，用于验证翻译链路的真实 HTTP 路径。

不依赖第三方库（stdlib http.server），实现 POST /v1/chat/completions，
按 system prompt 区分三类请求并返回固定响应：

- 摘要（影视内容分析助手）→ 固定摘要文本
- 术语表（术语管理助手）→ 固定 ``原文 | 译文 | 出现次数`` 行
- 逐句翻译（字幕翻译员）→ 把 user 里的待译原文包一层 ``【译】...`` 返回

用法：
    python scripts/mock_translate_server.py --port 8399
然后把 config.yaml 的 translate.base_url 指向 http://127.0.0.1:8399/v1 即可。
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MOCK_SUMMARY = "（mock 摘要）一段英文演讲片段，语言口语化，无特殊专有名词。"
MOCK_GLOSSARY = "Hello | 你好 | 2\nWorld | 世界 | 1"


def _mock_reply(messages: list[dict]) -> str:
    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    if "影视内容分析助手" in system:
        return MOCK_SUMMARY
    if "术语管理助手" in system:
        return MOCK_GLOSSARY
    # 逐句翻译：抽出「请翻译以下字幕原文：」之后、前瞻块之前的原文
    body = user.split("请翻译以下字幕原文：", 1)[-1]
    body = body.split("\n\n以下为后续原文", 1)[0].strip()
    return f"【译】{body}"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib 约定
        if not self.path.rstrip("/").endswith("/v1/chat/completions"):
            self.send_error(404, "only /v1/chat/completions is implemented")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            content = _mock_reply(payload.get("messages") or [])
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8399)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"mock OpenAI 端点：http://{args.host}:{args.port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
