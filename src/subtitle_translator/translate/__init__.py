"""翻译层（占位，后续阶段实现）。

OpenAI 兼容端点（base_url / api_key / model / temperature 四字段；
本地 vLLM / Ollama 也走同一接口）。

三步走：
1. 整片字幕一次调用生成摘要；
2. 提取高频专名生成术语表（src/dst/count，上限约 50 条，解析失败重试）；
3. 滑动窗口逐句翻译（历史对 + 前瞻 + 术语表注入 system prompt）。

术语表有人工确认检查点（glossary.confirmed）；可选术语后校验
（第一版只标记不重翻）。翻译策略为可插拔接口，第一版只实现这一个策略。
"""
