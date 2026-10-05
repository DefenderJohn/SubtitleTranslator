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

### 2.1 编排（pipeline.py）

- 媒体文件发现 `find_media_files(path)`：path 是文件则校验扩展名后直接使用，是目录则递归查找（扩展名集合：flac/m4a/mp3/mp4/mpeg/mpga/oga/ogg/wav/webm + mkv/mov/avi/m4v）。
- 工程文件约定：`xxx.mp4` → `xxx.sub.json`（同名同目录，`project_json_path`）；默认导出 `xxx.srt`（`default_srt_path`）。
- `run_pipeline(media_path, cfg, *, stages=("transcribe","translate","export"), auto_confirm=False, bilingual=True, progress_cb=None, backend=None, strategy=None, from_json=False) -> SubtitleProject`：
  - 已存在工程 JSON 则加载，按 `meta.stage` 跳过已完成阶段（断点续传核心）；**每完成一个阶段立即落盘**；
  - 术语表人工确认检查点失败（`GlossaryNotConfirmedError`）时也先落盘（stage=contexted，摘要+术语表已保存）再抛出，确认后重跑即从逐句翻译续上；
  - `stages=("transcribe", "export")` 即 transcribe-only：只转录并导出原文 SRT，不碰翻译端点；
  - `from_json=True`：media_path 直接是已有 .sub.json，跳过转录、不接触媒体文件（重翻 / 调术语后重跑）；
  - `backend` / `strategy` 可注入（测试、批量复用），为 None 时按 cfg 自建。
- `run_batch(path, cfg, ...)`：遍历媒体文件逐个 `run_pipeline`，单文件失败记日志继续，最后返回 `BatchResult(succeeded, failed)` 汇总；传入的 backend / strategy 在全部文件间复用。
- **进度回调协议**（网页 SSE 直接复用）：`progress_cb(event: dict)`，event 固定五键：
  ```json
  {"media": "/path/movie.mp4", "stage": "translate", "done": 12, "total": 100, "message": "翻译 12/100"}
  ```
  `stage` ∈ `transcribe` / `translate` / `export`；阶段开始与结束时 `done=0, total=0`（信息在 `message`），进行中 `done/total` 为实际进度（转录=块数，翻译=cue 条数）。
- **协作式取消协议**（网页任务取消用）：`progress_cb` 可抛 `PipelineCancelledError`；取消在回调边界生效（当前 cue / 块完成后停）。translate 阶段被取消时先把已翻译的 cue 落盘（stage 保持 `contexted`）再传播，重跑即断点续传；`run_batch` 遇取消停止整个批次（不记为失败）。

### 2.2 CLI（cli.py）

argparse 实现（stdlib，无新依赖），入口 `subtitle-translator`：

```
subtitle-translator run <路径> [--config config.yaml] [--transcribe-only] [--auto-confirm]
                               [--no-bilingual] [--from-json] [--language en]
subtitle-translator export <xxx.sub.json> [--bilingual] [-o out.srt]
subtitle-translator glossary <xxx.sub.json> [--confirm-all] [--show]
subtitle-translator serve [--config config.yaml] [--host] [--port]
subtitle-translator config init [path]
```

- `run` 的路径是目录时自动走 `run_batch`；CLI 参数覆盖 yaml（如 `--language` 覆盖 `asr.language`）。
- 配置体检：跑翻译前校验 `translate.model`（缺失时报错并给出 config init 指引）；`resolve_api_key` 为 None 时 stderr 警告并给出 `api_key_env` 配置指引（本地端点可忽略）。
- 检查点失败时 stderr 打印后续操作指引（glossary --show / --confirm-all / --auto-confirm）。
- 进度在 stderr 简单打印（不引 tqdm，网页才是主交互）。

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

- 接口：OpenAI 兼容端点，端点连接配置为四字段：`base_url` / `api_key`（或 `api_key_env`）/ `model` / `temperature`（其余翻译行为配置见 §7）。本地 vLLM / Ollama 也走同一接口。HTTP 用官方 `openai` SDK 的 chat.completions（非 streaming），SDK 自重试关闭，由 `client.py` 统一做指数退避（429 / 5xx / 超时 / 连接错误，最多 `max_retries` 次，超时 `request_timeout` 默认 120s）。
- 模块拆分：`client.py`（SDK 封装+退避）、`prompts.py`（prompt 模板）、`context.py`（摘要）、`glossary.py`（术语表提取与解析）、`strategy.py`（策略 ABC + 实现）。
- 三步走（`SlidingWindowStrategy.translate(project, cfg, progress_cb=None, auto_confirm=False)` 编排，断点续传时已完成步骤自动跳过）：
  1. **摘要**：整片字幕一次调用生成摘要，存入 `project.meta.summary`；全文超过 `SUMMARY_MAX_CHARS`（常量，100K 字符）时按 cue 边界分块 map-reduce（逐块摘要→合并摘要）；
  2. **术语表**：基于摘要+全文提取高频专名，要求模型按固定格式 `原文 | 译文 | 出现次数` 逐行输出，解析成 GlossaryEntry，上限 `glossary_max_entries`；一条都解析不出时换提示（追加严格格式说明）并降批（条数上限减半）重试，最多 `glossary_max_retries` 次，仍失败抛 `GlossaryExtractError`。**生成后 stage 变为 `contexted`；逐句翻译开始前必须所有条目 `confirmed=true`**（人工确认检查点），否则抛 `GlossaryNotConfirmedError`；`auto_confirm=True`（CLI `--auto-confirm`）自动全部置 true；
  3. **滑动窗口逐句翻译**：每条 cue 的 messages 组装为：system（角色 + 目标语言 + additional_prompt + 摘要 + 已确认术语表）→ 前 `history_count` 条已翻译的「原文→译文」拼成 user/assistant 消息对 → 当前 user（待译原文 + 后 `forward_count` 条原文，明确标注「不要翻译」）。调用间是独立请求，无服务端状态。
- **防御性解析**：译文剥离编号前缀与多余空白；空输出 / 明显复读（译文 > 原文 10 倍或 > 500 字符）触发单条重试（最多 2 次，第二次降 temperature 至一半），仍失败保留原文占位并打 `translation_failed` 标记，不中断整体流程。
- **术语后校验**（第一版只标记不重翻）：原文含某术语 src 而译文不含对应 dst，在该 cue 的 `flags` 上记 `glossary_miss:<src>`。
- 进度回调：`progress_cb(done, total)`，逐条推进。
- 翻译策略为可插拔接口 `TranslationStrategy` ABC（`translate(project, cfg, progress_cb, auto_confirm) -> project`），第一版只实现 `SlidingWindowStrategy`；测试注入 fake client，不打真实 API。

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
      "words": [ { "text": "...", "start": 0.0, "end": 0.0 } ],
      "flags": []
    }
  ],
  "stage": "transcribed"
}
```

- `stage` 取值：`empty`（初始态，尚未转录）→ `transcribed` → `contexted` → `translated`，驱动断点续传。
- `words` 词级时间戳为未来功能（波形修轴、剪辑）预留。
- `cues[].flags` 为翻译层标记列表，当前取值：`translation_failed`（重试后仍失败，保留原文占位）、`glossary_miss:<src>`（术语后校验未命中）。
- 序列化约定：缺字段给默认值；`version` 与当前 SCHEMA_VERSION 不匹配时抛出 `SchemaVersionError`；非法 `stage` 抛 `ValueError`。
- SRT 导出语义：双语导出译文在上、原文在下（与旧版一致）；单语导出优先译文、无译文退化为原文。

## 7. 配置（config.py）

- 单份 `config.yaml` 为唯一配置存储，网页 / CLI / 脚本共用。
- 分三节：`asr` / `translate` / `ui`。
- `translate.api_key` 支持 `api_key_env` 环境变量引用，避免明文密钥入库；`api_key_env` 优先于明文，`save_config` 默认不落盘明文 key。
- 默认值：`asr.backend=vllm`（可选 `transformers`）、`asr.model=Qwen/Qwen3-ASR-1.7B`、`asr.aligner_model=Qwen/Qwen3-ForcedAligner-0.6B`、`asr.chunk_max_seconds=290`（ForcedAligner ≤5min 留余量）、`asr.language=null`（源语言，null=自动检测）、`asr.ffmpeg_path=""`（空=自动探测：PATH → imageio-ffmpeg）；`translate.history_count=10`、`forward_count=1`、`glossary_max_entries=50`、`target_language=简体中文`、`additional_prompt=翻译当前字幕到简体中文`、`request_timeout=120`（秒）、`max_retries=4`（HTTP 退避重试）、`glossary_max_retries=3`（术语表解析重试）。

## 8. 网页（server/ + frontend/）

- 后端：FastAPI（REST + SSE 进度推送 + Range 请求视频流）。
- 前端：React + Ant Design，Vite 构建。
- 第一版只做：上传/选文件、任务列表+进度、字幕表格文本编辑、术语表确认、下载。
- 波形修轴和剪辑是未来功能；数据格式已预留（words 词级时间戳）。

### 8.1 server 架构（localhost 单用户）

- **薄壳**：业务逻辑全部走 core（pipeline / models / config），server 只做协议转换与参数校验。
- **任务调度**：进程内任务注册表 + 单并发 asyncio worker（本地单 GPU，排队即可，不引入 Celery/Redis）。同步 pipeline 用 `asyncio.to_thread` 执行；pipeline 的 progress_cb 在 worker 线程被调用，事件经 `loop.call_soon_threadsafe` 推给 SSE 订阅队列（asyncio.Queue 非线程安全，任务入队端点用 async def 在事件循环线程执行）。
- **SSE**：`GET /api/tasks/{id}/events` 直接透传 pipeline event 五键，附加 `task_id` 与 `status`；订阅时先回放历史事件（后打开的页面可恢复进度），任务进入终态（done/failed/cancelled）后关流。状态变更也产生事件（stage 为空串）。
- **任务状态机**：
  ```
  pending ──→ running ──→ done
        │        ├──────→ failed
        │        ├──────→ waiting_confirm ──(POST resume)──→ pending
        │        └──────→ cancelled
        └──(cancel)──→ cancelled
  ```
  waiting_confirm = 术语表人工确认检查点未通过（GlossaryNotConfirmedError），网页确认术语后 resume 续跑；running 的取消是协作式（当前 cue 完成后停，已翻译进度已落盘）。
- **每次执行任务重新加载 config.yaml**：网页改配置对后续任务生效。
- **CORS**：允许 localhost / 127.0.0.1 任意端口（vite dev server）。
- **静态托管**：`frontend/dist` 存在时挂载到 `/`（前端路由回退 index.html，SPA fallback），不存在时 `/` 返回占位提示页；API 路由优先于静态挂载。
- **路径安全**：本工具是 localhost 单用户工具，API 的路径参数就是本机文件路径（选文件/浏览目录是功能本身），不做沙箱化。

### 8.2 API 清单

任务与媒体：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/tasks` | 创建任务：body `{path, options{auto_confirm, bilingual, transcribe_only, language}}`，path 文件/目录自动判断，返回任务快照（含 id） |
| GET | `/api/tasks` | 任务列表（id、path、media、status、progress 快照、created_at） |
| GET | `/api/tasks/{id}` | 任务详情 |
| GET | `/api/tasks/{id}/events` | SSE 订阅（历史回放 + 实时推送，终态关流） |
| POST | `/api/tasks/{id}/cancel` | 取消（协作式）；终态返回 409 |
| POST | `/api/tasks/{id}/resume` | waiting_confirm 任务重新排队继续；其他状态 409 |
| GET | `/api/media?path=` | 浏览目录：返回子目录名 + 递归媒体文件清单（复用 find_media_files） |
| GET | `/api/video?path=` | 视频流，支持单区间 Range（bytes=start-end / start- / -suffix），206 + Content-Range，非法区间 416 |

工程数据（JSON 是唯一事实来源，直接读写 `.sub.json`；path 参数传 `.sub.json` 或对应媒体文件路径均可）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/project?path=` | 读工程完整 JSON |
| PATCH | `/api/project/cues` | 编辑单条 cue 的 text/translation（body 含 path、cue_id） |
| GET | `/api/project/glossary?path=` | 读术语表 |
| PATCH | `/api/project/glossary` | 改术语（updates 按 src 匹配，可改 dst / new_src）、按 src confirm 单条或 confirm_all |
| POST | `/api/project/export` | 对工程导出 SRT（bilingual 参数），返回 srt_path |

配置：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/config` | 读 config.yaml；**api_key 不明文返回**（masked，如 `sk-...***`），另返回 `api_key_resolved` 表示密钥是否已可用（可能来自 api_key_env） |
| PUT | `/api/config` | 局部更新（按节给 dict，未知节/字段 400，改后重建配置节触发校验）；api_key 传空字符串或 mask 值表示不修改；文件里已有/新设明文 key 时保留，否则不落盘明文 |

启动：`subtitle-translator serve [--config path] [--host] [--port]`（host/port 默认取 ui 节配置）。

## 9. 断点续传

每阶段产物落盘 JSON（`xxx.sub.json`），重跑时跳过已完成阶段（由 `stage` 字段驱动）。检查点中断（术语表待确认）也会先落盘 stage=contexted，确认术语后重跑即从逐句翻译续上；崩溃重启不丢已完成阶段的进度。

## 10. 依赖策略

- 核心依赖尽量轻（当前只有 pyyaml + openai）。
- `torch` / `vllm` / `qwen-asr` / `transformers` 列为 optional extra（`asr`），FastAPI 等为 `web` extra，按环境单独安装。
- 依赖版本只钉下界，不钉死。
