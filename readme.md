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

## Legacy 文件

以下文件是 2024 年旧版的遗留，**暂时保留**，将在重构最后阶段清理：

- `Translator_shell.py`（旧命令行主程序）
- `Config.ini`（旧配置）
- `requirements.txt`（旧依赖清单）
