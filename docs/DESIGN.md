# SubtitleTranslator 重构设计文档

> 本文档是重构的设计定案，是后续所有开发阶段的依据。任何对设计的变更必须先改本文档再改代码。

## 1. 整体架构

core 纯 Python 库 + 两个薄壳（CLI、FastAPI + React 网页）。业务逻辑全部在 core，壳只做交互与转发。

```
┌─────────────────────────────────────────────────────────┐
│                      用户入口（薄壳）                      │
│   ┌──────────────┐        ┌──────────────────────────┐  │
│   │  CLI (cli.py) │        │  Web: FastAPI + React    │  │
│   └──────┬───────┘        │  (server/ + frontend/)   │  │
│          │                └────────────┬─────────────┘  │
│          └────────────────┬────────────┘                │
├───────────────────────────▼─────────────────────────────┤
│                 core 库（subtitle_translator）            │
│                                                          │
│   pipeline.py 流水线编排 + 断点续传                        │
│     │                                                    │
│     ▼                                                    │
│   transcribe.py 转录（qwen-asr，vLLM / transformers）      │
│     │  ffmpeg silencedetect 切块（ForcedAligner ≤5min）    │
│     ▼                                                    │
│   segment.py 分段（标点 / 停顿 / 时长字符上限）              │
│     │                                                    │
│     ▼                                                    │
│   translate/ 翻译（OpenAI 兼容端点，三步走）                 │
│     ① 摘要 → ② 术语表（人工确认检查点）→ ③ 滑动窗口翻译       │
│     │                                                    │
│     ▼                                                    │
│   srt.py 导出 SRT / 双语 SRT                              │
│                                                          │
│   models.py 数据结构与 JSON schema（唯一事实来源）           │
│   config.py  单份 config.yaml                             │
└─────────────────────────────────────────────────────────┘
```

## 2. 流水线

音视频 → 转录 → 分段 →（摘要 + 术语表，人工确认检查点）→ 滑动窗口逐句翻译 → 导出 SRT / 双语 SRT。

## 3. 转录层（transcribe.py + media.py）

- 使用官方 `qwen-asr` Python 包（`Qwen3ASRModel`）：
  - 转录模型：`Qwen/Qwen3-ASR-1.7B`
  - 对齐模型：`Qwen/Qwen3-ForcedAligner-0.6B`，产出词级时间戳（`return_time_stamps=True`）
- 两个 backend：**vLLM 优先（`Qwen3ASRModel.LLM(...)`）、transformers 兜底（`.from_pretrained(...)`）**，做成配置项 `asr.backend`，对上层接口零差异（`AsrBackend` 抽象：`load()` / `transcribe_chunk(audio_path, language)` / `unload()`）。`qwen_asr` 在 `load()` 内延迟 import，未安装时模块仍可 import、`load()` 报清晰错误。
- **硬约束**：ForcedAligner 单次只支持 ≤ 5 分钟音频。因此必须自写切块层：
  - 用 `ffmpeg silencedetect` 找静音边界（刻意不引入 silero-vad 等额外 torch 依赖）；
  - `chunk_plan(duration, silences, max_seconds)` 纯函数切块：优先在目标切点之前最接近目标的静音中点下刀，找不到静音时硬切并记 warning；
  - `transcribe_media(path, cfg, backend=None, progress_cb=None)`：probe → 切块 → 逐块抽 16kHz 单声道 wav 转录 → 词级时间戳加块偏移量拼接 → 直接流水线调用分段器产出 cues（`stage=transcribed`）。
- **ffmpeg 是硬依赖**（将来剪辑功能也要用）。`media.py` 封装：二进制路径解析顺序为 `asr.ffmpeg_path` 配置 → PATH → imageio-ffmpeg 静态二进制，找不到时报错并给出安装指引；没有 ffprobe 时用 `ffmpeg -i` 的 stderr 解析 Duration 兜底。
- **词级时间戳不存顶层 `raw_words` 字段**：分段后每个 cue 的 `words` 已完整承载词级时间戳，展平所有 cue 的 words 即可恢复原始词序列，避免冗余。

## 4. 分段器（segment.py）

独立纯函数模块：`segment_words(words, *, max_chars=42, max_duration=7.0, min_duration=1.0, gap_threshold=0.6)`。

断点优先级：句末标点（`.!?。！？…`）> 词间 gap > `gap_threshold` > 次级标点（`,;:，；：、`）> 超长硬切（记 warning）。贪心延长当前行，触限（字符/时长）时回看行内最佳断点。

约束：单行 ≤ `max_chars`（英文按字符含空格）且 ≤ `max_duration` 秒；短于 `min_duration` 的行与相邻行合并（但不违反 max 约束）；单个超长词保留不拆。text 拼接：英文词间补空格，CJK 不补，标点符号前不补；撇号词 / 连字符词只在词边界下刀，不会被拆坏。cue 的 `words` 字段保留该行词级时间戳（修轴功能预留）。

## 5. 翻译层（translate/）

- 接口：OpenAI 兼容端点，配置仅四字段：`base_url` / `api_key` / `model` / `temperature`。本地 vLLM / Ollama 也走同一接口。
- 三步走：
  1. **摘要**：整片字幕一次调用生成摘要；
  2. **术语表**：提取高频专名生成术语表（`src` / `dst` / `count`，上限约 50 条，解析失败重试）；
  3. **滑动窗口逐句翻译**：历史对 + 前瞻 + 术语表注入 system prompt。
- **人工确认检查点**：术语表翻译前需人工确认（`glossary.confirmed`）。
- 可选术语后校验：第一版只标记不重翻。
- 翻译策略做成**可插拔接口**，但第一版只实现上述这一个策略。

## 6. 存储（models.py）

**每个视频一个 JSON 文件，是唯一事实来源；SRT 只是导出物。**

JSON schema：

```json
{
  "version": 1,
  "source": { "file": "...", "duration": 0.0, "language": "..." },
  "models": { "asr": "...", "aligner": "...", "translator": "..." },
  "summary": "...",
  "glossary": [ { "src": "...", "dst": "...", "count": 0, "confirmed": false } ],
  "cues": [
    {
      "id": 1,
      "start": 0.0,
      "end": 0.0,
      "text": "...",
      "translation": "...",
      "words": [ { "text": "...", "start": 0.0, "end": 0.0 } ]
    }
  ],
  "stage": "transcribed"
}
```

- `stage` 取值：`empty`（初始态，尚未转录）→ `transcribed` → `contexted` → `translated`，驱动断点续传。
- `words` 词级时间戳为未来功能（波形修轴、剪辑）预留。
- 序列化约定：缺字段给默认值；`version` 与当前 SCHEMA_VERSION 不匹配时抛出 `SchemaVersionError`；非法 `stage` 抛 `ValueError`。
- SRT 导出语义：双语导出译文在上、原文在下（与旧版一致）；单语导出优先译文、无译文退化为原文。

## 7. 配置（config.py）

- 单份 `config.yaml` 为唯一配置存储，网页 / CLI / 脚本共用。
- 分三节：`asr` / `translate` / `ui`。
- `translate.api_key` 支持 `api_key_env` 环境变量引用，避免明文密钥入库；`api_key_env` 优先于明文，`save_config` 默认不落盘明文 key。
- 默认值：`asr.backend=vllm`（可选 `transformers`）、`asr.model=Qwen/Qwen3-ASR-1.7B`、`asr.aligner_model=Qwen/Qwen3-ForcedAligner-0.6B`、`asr.chunk_max_seconds=290`（ForcedAligner ≤5min 留余量）、`asr.language=null`（源语言，null=自动检测）、`asr.ffmpeg_path=""`（空=自动探测：PATH → imageio-ffmpeg）；`translate.history_count=10`、`forward_count=1`、`glossary_max_entries=50`、`target_language=简体中文`、`additional_prompt=翻译当前字幕到简体中文`。

## 8. 网页（server/ + frontend/）

- 后端：FastAPI（REST + SSE 进度推送 + Range 请求视频流）。
- 前端：React + Ant Design，Vite 构建。
- 第一版只做：上传/选文件、任务列表+进度、字幕表格文本编辑、术语表确认、下载。
- 波形修轴和剪辑是未来功能；数据格式已预留（words 词级时间戳）。

## 9. 断点续传

每阶段产物落盘 JSON，重跑时跳过已完成阶段（由 `stage` 字段驱动）。

## 10. 依赖策略

- 核心依赖尽量轻（当前只有 pyyaml）。
- `torch` / `vllm` / `qwen-asr` / `transformers` 列为 optional extra（`asr`），FastAPI 等为 `web` extra，按环境单独安装。
- 依赖版本只钉下界，不钉死。
