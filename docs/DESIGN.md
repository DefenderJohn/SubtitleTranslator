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

## 3. 转录层（transcribe.py）

- 使用官方 `qwen-asr` Python 包（`Qwen3ASRModel`）：
  - 转录模型：`Qwen/Qwen3-ASR-1.7B`
  - 对齐模型：`Qwen/Qwen3-ForcedAligner-0.6B`，产出词级时间戳
- 两个 backend：**vLLM 优先、transformers 兜底**，做成配置项，对上层接口零差异。
- **硬约束**：ForcedAligner 单次只支持 ≤ 5 分钟音频。因此必须自写切块层：
  - 用 `ffmpeg silencedetect` 找静音边界切块（刻意不引入 silero-vad 等额外 torch 依赖）；
  - 逐块转录 + 对齐，然后拼接并偏移时间戳。
- **ffmpeg 是硬依赖**（将来剪辑功能也要用）。

## 4. 分段器（segment.py）

独立模块。输入词级时间戳，按以下规则切成字幕行（cue）：

- 标点；
- 停顿（词间 gap）；
- 时长与字符上限。

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
- `translate.api_key` 支持 `api_key_env` 环境变量引用，避免明文密钥入库。

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
