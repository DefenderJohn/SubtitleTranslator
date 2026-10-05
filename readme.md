# SubtitleTranslator（重构中）

音视频字幕转录与翻译工具。

> **本项目正在整体重构。** 2024 年初的旧版本（openai-whisper 转录 + 本地 ChatGLM3-6B 逐条翻译）已被新架构取代。

## 新架构

core 纯 Python 库 + 两个薄壳（CLI、FastAPI + React 网页）：

- **流水线**：音视频 → 转录 → 分段 →（摘要 + 术语表，人工确认）→ 滑动窗口逐句翻译 → 导出 SRT / 双语 SRT
- **转录**：`qwen-asr`（Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B 词级时间戳），vLLM / transformers 双 backend，ffmpeg 静音切块
- **翻译**：OpenAI 兼容端点（本地 vLLM / Ollama 同一接口），摘要 → 术语表 → 滑动窗口三步走
- **存储**：每个视频一个 JSON（唯一事实来源），SRT 为导出物，断点续传

完整设计定案见 [docs/DESIGN.md](docs/DESIGN.md)，环境摸底见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)，开发约定见 [AGENTS.md](AGENTS.md)。

## 安装与测试（骨架阶段）

```bash
pip install -e .          # 核心依赖（仅 pyyaml）
pip install -e .[dev]     # 含 pytest
python -m pytest
```

重依赖按环境单独安装：`pip install -e .[asr]`（qwen-asr / torch / vllm）、`pip install -e .[web]`（FastAPI / uvicorn）。ffmpeg 为硬依赖，需系统级安装。

## 命令行使用

```bash
# 生成默认配置，按需修改 translate.base_url / model，用 api_key_env 引用密钥
subtitle-translator config init

# 完整流水线：转录 → 摘要+术语表 → 逐句翻译 → 导出双语 SRT
subtitle-translator run movie.mp4 --auto-confirm

# 目录批量处理（单文件失败不中断，最后汇总成功/失败清单）
subtitle-translator run /path/to/videos/ --auto-confirm

# 只转录，导出原文 SRT
subtitle-translator run movie.mp4 --transcribe-only

# 对已有工程文件重跑后续阶段（不接触媒体文件，用于重翻/调术语后重跑）
subtitle-translator run movie.sub.json --from-json --auto-confirm

# 术语表人工确认检查点：查看 → 确认 → 重跑续上
subtitle-translator glossary movie.sub.json --show
subtitle-translator glossary movie.sub.json --confirm-all

# 从工程文件导出 SRT（--bilingual 双语，译文在上）
subtitle-translator export movie.sub.json --bilingual -o out.srt
```

断点续传：每个媒体文件对应 `xxx.sub.json` 工程文件（唯一事实来源），每完成一个阶段立即落盘，重跑自动跳过已完成阶段。CLI 参数（如 `--language en`）覆盖 yaml 配置。

## Legacy 文件

以下文件是 2024 年旧版的遗留，**暂时保留**，将在重构最后阶段清理：

- `Translator_shell.py`（旧命令行主程序）
- `Config.ini`（旧配置）
- `requirements.txt`（旧依赖清单）
