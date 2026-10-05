# AGENTS.md

## 项目简介

SubtitleTranslator 重构版：音视频字幕转录与翻译工具。core 纯 Python 库 + 两个薄壳（CLI、FastAPI + React 网页）。设计定案见 [docs/DESIGN.md](docs/DESIGN.md)，所有开发必须以该文档为依据。

## 项目结构

```
├── pyproject.toml              # 现代打包（src 布局）；重依赖在 optional extras
├── src/subtitle_translator/    # core 库
│   ├── config.py               # 单份 config.yaml 加载与校验（api_key_env 引用）
│   ├── models.py               # 数据结构 / JSON schema（唯一事实来源）
│   ├── transcribe.py           # 转录层：qwen-asr，vLLM / transformers 双 backend，
│   │                           #   ffmpeg silencedetect 切块（ForcedAligner ≤5min）
│   ├── segment.py              # 分段器：词级时间戳 → 字幕行
│   ├── translate/              # 翻译层：OpenAI 兼容端点，三步走，可插拔策略
│   ├── pipeline.py             # 流水线编排 + 断点续传（stage 驱动）
│   ├── srt.py                  # SRT / 双语 SRT 导出
│   ├── cli.py                  # CLI 薄壳
│   └── server/                 # FastAPI 薄壳（REST + SSE + Range 视频流）
├── frontend/                   # React + Ant Design（Vite），阶段 7 占位
├── tests/                      # pytest
├── docs/DESIGN.md              # 设计定案
├── docs/ENVIRONMENT.md         # 环境摸底
└── （legacy，最后阶段才清理：Translator_shell.py / Config.ini / requirements.txt）
```

## 开发约定

- **git**：每个小进展单独提交；commit message 用中文；不要 push，不要改 git 配置，不要改历史。
- **测试**：改代码后跑 `python -m pytest`。
- **密钥**：绝不提交 api_key；配置中用 `api_key_env` 环境变量引用。
- **文档同步**：代码行为变化时同步更新 docs/DESIGN.md（以及本文件的结构说明）。
- **依赖**：核心保持轻量；torch / vllm / qwen-asr 等只放 optional extras，版本只钉下界。
- 占位模块只写 docstring 说明职责，业务实现按阶段逐步填充。
