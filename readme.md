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

## 网页服务

```bash
pip install -e .[web]        # FastAPI / uvicorn
subtitle-translator serve    # 默认 http://127.0.0.1:7860（ui 节配置，--host/--port 覆盖）
```

FastAPI 薄壳：任务管理（创建/列表/取消/术语确认后 resume）、SSE 进度推送（直接透传 pipeline 事件协议，含历史回放）、Range 视频流预览、工程 JSON / 术语表 / cue 文本在线编辑、config.yaml 读写（api_key 不明文返回）。单并发任务队列（本地单 GPU）。API 文档启动后见 `/docs`，完整清单见 [docs/DESIGN.md](docs/DESIGN.md) 第 8 节。

### 网页使用

前端是 `frontend/` 下的 React + Ant Design 5（Vite + TypeScript）应用，构建产物 `frontend/dist` 由后端直接托管（前端路由回退 index.html）：

```bash
cd frontend
npm install
npm run build          # 产出 frontend/dist
subtitle-translator serve   # 浏览器打开 http://127.0.0.1:7860
```

页面功能：

- **任务页**（默认）：路径输入 + 目录浏览弹窗选择媒体文件/目录，选项（自动确认术语 / 双语导出 / 仅转录 / 源语言）创建任务；任务列表含状态标签与实时进度条（SSE 推送），可取消、进详情。
- **任务详情页**：事件日志流（SSE 历史回放 + 实时追加）；`waiting_confirm` 时展示摘要与可编辑术语表，「全部确认并继续」resume 续跑；完成后一键导出 SRT 并显示结果路径。
- **字幕校对**（详情页 tab）：cue 表格（起止时间 / 原文 / 译文可编辑），术语不一致标记（`glossary_miss`）红色提示；右侧视频预览，点击 cue 行跳转播放。
- **设置页**：读写 config.yaml（ASR / 翻译 / 界面三组）；api_key 显示掩码值、留空不修改，密钥已通过环境变量配置时显示绿色提示。

开发模式：`npm run dev`（vite dev server，`/api` 代理到 127.0.0.1:7860，需先启动后端）。
