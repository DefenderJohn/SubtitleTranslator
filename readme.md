# SubtitleTranslator

音视频字幕转录与翻译工具：本地 Qwen3-ASR 转录（词级时间戳）→ OpenAI 兼容 LLM 端点三步走翻译（摘要 → 术语表人工确认 → 滑动窗口逐句）→ 导出 SRT / 双语 SRT。core 纯 Python 库 + CLI / 网页（FastAPI + React）双壳，断点续传。

**详细使用文档见 [docs/USAGE.md](docs/USAGE.md)**（安装、配置、CLI / 网页操作、常见问题）；设计定案见 [docs/DESIGN.md](docs/DESIGN.md)，环境摸底见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)，开发约定见 [AGENTS.md](AGENTS.md)。

## 快速开始

```bash
# 1. 建 Python 3.12 环境并安装（NVIDIA GPU；Turing 老卡用 transformers backend 即可）
conda create -n subtr python=3.12 -y && conda activate subtr
pip install -e ".[asr,web]" && pip install imageio-ffmpeg

# 2. 下载模型（modelscope），并在 config 里填本地路径
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir /path/to/Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir /path/to/Qwen3-ForcedAligner-0.6B

# 3. 生成配置，按需修改：asr.model / aligner_model 填本地路径、
#    translate.base_url / model / api_key_env（backend 默认 transformers，开箱即用）
subtitle-translator config init

# 4a. 命令行：完整流水线（转录 → 翻译 → 双语 SRT）
subtitle-translator run movie.mp4 --auto-confirm

# 4b. 或网页：浏览器打开 http://127.0.0.1:7860
subtitle-translator serve
```

## 开发

```bash
pip install -e .[dev]
python -m pytest
```

前端源码在 `frontend/`（React + Ant Design 5，Vite 构建，产物 `frontend/dist` 由后端托管）；改动后 `npm install && npm run build` 重新构建。API 清单见 [docs/DESIGN.md](docs/DESIGN.md) 第 8 节。
