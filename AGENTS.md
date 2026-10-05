# AGENTS.md

## 项目简介

SubtitleTranslator：音视频字幕转录与翻译工具。core 纯 Python 库 + 两个薄壳（CLI、FastAPI + React 网页）。设计定案见 [docs/DESIGN.md](docs/DESIGN.md)，所有开发必须以该文档为依据。

## 项目结构

```
├── pyproject.toml              # 现代打包（src 布局）；重依赖在 optional extras
├── src/subtitle_translator/    # core 库
│   ├── config.py               # 单份 config.yaml 加载与校验（api_key_env 引用）
│   ├── models.py               # 数据结构 / JSON schema（唯一事实来源）
│   ├── transcribe.py           # 转录层：qwen-asr，vLLM / transformers 双 backend，
│   │                           #   ffmpeg silencedetect 切块（ForcedAligner ≤5min）
│   ├── media.py                # ffmpeg 封装：时长探测 / 静音检测 / 音频切块
│   ├── segment.py              # 分段器：词级时间戳 → 字幕行
│   ├── translate/              # 翻译层：OpenAI 兼容端点，三步走，可插拔策略
│   │                           #   client.py(SDK封装+退避) prompts.py context.py
│   │                           #   glossary.py strategy.py(ABC+SlidingWindow)
│   ├── pipeline.py             # 流水线编排 + 断点续传（stage 驱动）+ run_batch 批量；
│   │                           #   进度回调协议 progress_cb({media,stage,done,total,message})
│   ├── srt.py                  # SRT / 双语 SRT 导出
│   ├── cli.py                  # CLI 薄壳（argparse：run / export / glossary / serve / config init）
│   └── server/                 # FastAPI 薄壳（REST + SSE + Range 视频流）
│                               #   tasks.py(任务注册表+单并发worker+状态机) app.py(create_app+路由)
├── frontend/                   # React + Ant Design 5 + TypeScript（Vite 构建，
│                               #   dist 由 server 托管）；src/api.ts 集中 API 封装，
│                               #   pages/(任务/任务详情/设置) + components/(目录浏览/术语表/cue 校对)
├── tests/                      # pytest
├── docs/DESIGN.md              # 设计定案
├── docs/ENVIRONMENT.md         # 环境摸底
└── c/test.ogg                  # 旧版测试音频（保留，供真实冒烟用）
```

## 开发约定

- **git**：每个小进展单独提交；commit message 用中文；不要 push，不要改 git 配置，不要改历史。
- **测试**：改代码后跑 `python -m pytest`。
- **密钥**：绝不提交 api_key；配置中用 `api_key_env` 环境变量引用。
- **文档同步**：代码行为变化时同步更新 docs/DESIGN.md（以及本文件的结构说明）。
- **依赖**：核心保持轻量（pyyaml + openai）；torch / vllm / qwen-asr 等只放 optional extras，版本只钉下界。
- **进度回调协议**：`progress_cb({"media", "stage", "done", "total", "message"})`（pipeline 层统一定义，网页 SSE 直接复用；transcribe / translate 内层的 `(done, total)` 回调由 pipeline 包装成该协议）。
- **协作式取消**：`progress_cb` 抛 `pipeline.PipelineCancelledError` 即取消（当前 cue/块完成后停，translate 阶段会先落盘已翻译 cue）；网页层在回调里检查任务取消标志。
- **server 层**：fastapi/uvicorn/httpx 在 `web`/`dev` extras，延迟导入保持 core 可独立用；业务逻辑不写在 server（薄壳，只协议转换）；SSE 事件 = pipeline 五键 + task_id + status。
