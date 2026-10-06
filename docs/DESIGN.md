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
│     │  ffmpeg silencedetect 切块（对齐器输入上限）          │
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
│   preflight.py 启动预检（进入服务即 ready-to-use，见 §11）    │
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
- **启动预检**：跑 pipeline 前执行 preflight（见 §11），失败退出码 2 并打印报告；`--skip-preflight` 为逃生门（不推荐）。`--transcribe-only` 时不查翻译端点，`--from-json` 时不查 ffmpeg/模型/GPU。
- 检查点失败时 stderr 打印后续操作指引（glossary --show / --confirm-all / --auto-confirm）。
- 进度在 stderr 简单打印（不引 tqdm，网页才是主交互）。

## 3. 转录层（transcribe.py + media.py）

- 使用官方 `qwen-asr` Python 包（`Qwen3ASRModel`）：
  - 转录模型：`Qwen/Qwen3-ASR-1.7B`
  - 对齐模型：`Qwen/Qwen3-ForcedAligner-0.6B`，产出词级时间戳（`return_time_stamps=True`）
  - 结果对象 `ASRTranscription(language, text, time_stamps)`；`time_stamps` 为 `ForcedAlignResult`（可迭代），词元素字段 `text / start_time / end_time`（`_extract_result` 兼容 dict 形态与 start/end 别名）。
  - `language` 参数只接受规范语言名（"English"/"Chinese" 等 30 种）；`_normalize_language` 把 ISO 代码（en/zh/ja...）映射为规范名，其余值透传由 qwen-asr 校验。
  - `from_pretrained` 的 `dtype` 是 torch 对象（`_resolve_dtype` 把配置字符串转过去）；forced aligner 经 `forced_aligner_kwargs` 传 dtype/device_map。
- 两个 backend：**transformers（`.from_pretrained(...)`，默认）与 vLLM（`Qwen3ASRModel.LLM(...)`，可选加速）**，做成配置项 `asr.backend`，对上层接口零差异（`AsrBackend` 抽象：`load()` / `transcribe_chunk(audio_path, language)` / `unload()`）。默认 transformers 的原因：Turing（sm_75）等老卡兼容性 + 依赖更轻（vLLM 单列 `asr-vllm` extra）。`qwen_asr` 在 `load()` 内延迟 import，未安装时模块仍可 import、`load()` 报清晰错误；vllm 缺失时报错指向 `asr-vllm` extra。
- **本地模型离线加载**：`asr.model` 与 `asr.aligner_model` 都是本地存在的目录时（`_is_local_model_path`），`load()` 在 import qwen_asr 之前设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，并向 `from_pretrained` / `forced_aligner_kwargs` 透传 `local_files_only=True`（环境变量是兜底：qwen-asr 内部的 AutoProcessor 加载不透传 kwargs）；任一项是 hub ID 时维持现状（允许联网下载）。修复背景：模型经 modelscope 下载到本地目录时 HF 缓存为空，配 hub ID 会因连不上 HF endpoint 报 OSError。
- **硬约束**：ForcedAligner 单次只支持短音频（qwen-asr 0.0.6 内部按 180s 自行再切块）。因此必须自写切块层：
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
- 查找规则：CLI / serve 默认读**当前工作目录**的 `config.yaml`（`--config` 可指定别的路径），文件不存在时静默用内置默认值——没有其他 fallback（不查 `~/.subtitle_translator/`、不查安装目录），改配置必须改到生效的那份上。
- 分四节：`asr` / `translate` / `ui` / `log`。
- `translate.api_key` 支持 `api_key_env` 环境变量引用，避免明文密钥入库；`api_key_env` 优先于明文，`save_config` 默认不落盘明文 key。
- 默认值：`asr.backend=transformers`（可选 `vllm`，需 `asr-vllm` extra，Turing 等老卡不建议）、`asr.model=Qwen/Qwen3-ASR-1.7B`、`asr.aligner_model=Qwen/Qwen3-ForcedAligner-0.6B`、`asr.chunk_max_seconds=290`（对齐器输入上限留余量；qwen-asr 内部还会按 180s 再切块）、`asr.dtype=float16`（Turing 等不支持 bf16 原生计算的卡用 float16，bf16 机器可自行改回）、`asr.language=null`（源语言，null=自动检测；支持 ISO 代码如 en/zh）、`asr.ffmpeg_path=""`（空=自动探测：PATH → imageio-ffmpeg）；`translate.history_count=10`、`forward_count=1`、`glossary_max_entries=50`、`target_language=简体中文`、`additional_prompt=翻译当前字幕到简体中文`、`request_timeout=120`（秒）、`max_retries=4`（HTTP 退避重试）、`glossary_max_retries=3`（术语表解析重试）；`ui.host=127.0.0.1`、`ui.port=7860`、`ui.upload_dir=""`（空=`~/.subtitle_translator/uploads`，`resolve_upload_dir` 解析，支持 `~` 展开）；`log.level=INFO`（控制台级别，文件日志始终 DEBUG 起）、`log.dir=""`（空=`~/.subtitle_translator/logs`，`resolve_log_dir` 解析，支持 `~` 展开）。

## 8. 网页（server/ + frontend/）

- 后端：FastAPI（REST + SSE 进度推送 + Range 请求视频流）。
- 前端：React + Ant Design，Vite 构建。
- 页面结构：任务页（新建任务草稿放全局 context，切路由不丢；上传批次失效有探测 + 一键清空）、详情页（头部操作组 + Steps 五阶段进度【转录→建档→术语确认→翻译→导出，当前阶段给进度条与 ETA】+ Tabs【字幕校对（含统计/标记过滤）/ 术语表（stage ≥ contexted 随时可看可改）/ 摘要 / 运行信息（选项快照 + 阶段耗时 + 失败 cue + 事件记录）】，project JSON 只是可选增强，404 一律空态不报错）、设置页（翻译「测试连接」带当前表单测、`POST /api/preflight`「系统检查」）。
- 波形修轴和剪辑是未来功能；数据格式已预留（words 词级时间戳）。

### 8.1 server 架构（localhost 单用户）

- **薄壳**：业务逻辑全部走 core（pipeline / models / config），server 只做协议转换与参数校验。
- **任务调度**：进程内任务注册表 + 单并发 asyncio worker（本地单 GPU，排队即可，不引入 Celery/Redis）。同步 pipeline 用 `asyncio.to_thread` 执行；pipeline 的 progress_cb 在 worker 线程被调用，事件经 `loop.call_soon_threadsafe` 推给 SSE 订阅队列（asyncio.Queue 非线程安全，任务入队端点用 async def 在事件循环线程执行）。
- **SSE**：`GET /api/tasks/{id}/events` 直接透传 pipeline event 五键，附加 `task_id`、`status` 与 `time`（epoch 秒，前端 ETA / 阶段耗时估算用，历史回放保留真实时刻）；订阅时先回放历史事件（后打开的页面可恢复进度），任务进入终态（done/failed/cancelled）后关流。状态变更也产生事件（stage 为空串）。任务快照另含 `options`（创建时的选项，详情页「运行信息」展示）。
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
- **路径安全**：本工具是 localhost 单用户工具，API 的路径参数就是本机文件路径（选文件/浏览目录是功能本身），不做沙箱化。唯一收紧的是下载端点（见 §8.2 `/api/download`）：只允许 `.srt` / `.sub.json` 产物或 upload_dir 内的文件，不裸奔任意文件读。
- **浏览器上传**：`POST /api/upload` 接收 multipart 多文件，流式分块写盘（视频可达 GB 级，不整个读进内存）；文件名取 basename 防路径穿越、扩展名复用 MEDIA_EXTENSIONS 白名单校验（整批先校验后写盘）、同批/盘上撞名自动加 `_2` 后缀去重；每次上传建独立批次子目录 `upload_dir/YYYYMMDD-HHMMSS-xxxxxx/`，避免同名文件互相覆盖。返回的服务器侧路径直接拿去走 `POST /api/tasks`，复用同一套任务/pipeline 机制；上传文件的字幕校对视频预览经 `/api/video?path=` 天然可用。
- **上传生命周期（临时工作副本）**：上传批次目录（含其中的媒体与 .sub.json/.srt 产物）是临时的——server 正常退出（lifespan shutdown，uvicorn 的 Ctrl+C 路径）时删除本会话创建的全部批次目录，删除失败只警告不阻塞退出；运行中不删（任务完成后用户可能还要校对/下载 SRT）。崩溃兜底：启动时清理 upload_dir 下所有批次形态（`YYYYMMDD-HHMMSS-xxxxxx` 命名的子目录，符号链接除外）的遗留目录——前提 **localhost 单实例、upload_dir 不与其他实例共享**；非批次形态的条目（用户手放的文件/目录）不动。启动/退出各在 stderr 打印一行清理摘要（批次数 + 释放空间）。服务器路径模式处理的是用户自己的文件，任何情况下不删。

### 8.2 API 清单

任务与媒体：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/tasks` | 创建任务：body `{path, options{auto_confirm, bilingual, transcribe_only, language}}`，path 文件/目录自动判断，返回任务快照（含 id） |
| GET | `/api/tasks` | 任务列表（id、path、media、options、status、progress 快照、artifacts、created_at） |
| GET | `/api/tasks/{id}` | 任务详情（artifacts 为已存在的导出 SRT 路径列表，可经 `/api/download` 下载） |
| GET | `/api/tasks/{id}/events` | SSE 订阅（历史回放 + 实时推送，终态关流） |
| POST | `/api/tasks/{id}/cancel` | 取消（协作式）；终态返回 409 |
| POST | `/api/tasks/{id}/resume` | waiting_confirm 任务重新排队继续；其他状态 409 |
| GET | `/api/tasks/{id}/log` | per-task 日志文本（尾部优先，最多最后 200KB，`X-Log-Truncated` 头标记截断）；日志写入时已脱敏；任务未执行过（无日志文件）404 |
| GET | `/api/media?path=` | 浏览目录：返回子目录名 + 递归媒体文件清单（复用 find_media_files）；path 省略/为空时默认用户主目录；遍历中无权限/不可读的条目跳过，不因权限问题 500 |
| GET | `/api/video?path=` | 视频流，支持单区间 Range（bytes=start-end / start- / -suffix），206 + Content-Range，非法区间 416 |
| POST | `/api/upload` | 浏览器上传：multipart 字段 `files`（可多文件），流式写盘到 `ui.upload_dir` 的批次子目录，返回 `{batch, paths}`；paths 直接用于创建任务 |
| GET | `/api/download?path=` | 下载产物（Content-Disposition attachment）；仅允许 `.srt` / `.sub.json` 文件或 upload_dir 内的文件，其余 403，不存在 404 |

工程数据（JSON 是唯一事实来源，直接读写 `.sub.json`；path 参数传 `.sub.json` 或对应媒体文件路径均可）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/project?path=` | 读工程完整 JSON |
| PATCH | `/api/project/cues` | 编辑单条 cue 的 text/translation（body 含 path、cue_id） |
| GET | `/api/project/glossary?path=` | 读术语表 |
| PATCH | `/api/project/glossary` | 改术语（updates 按 src 匹配，可改 dst / new_src）、按 src confirm 单条或 confirm_all |
| POST | `/api/project/export` | 对工程导出 SRT（bilingual 参数），返回 srt_path 与 download_url |

配置：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/config` | 读 config.yaml；**api_key 不明文返回**（masked，如 `sk-...***`），另返回 `api_key_resolved` 表示密钥是否已可用（可能来自 api_key_env） |
| PUT | `/api/config` | 局部更新（按节给 dict，未知节/字段 400，改后重建配置节触发校验）；api_key 传空字符串或 mask 值表示不修改；文件里已有/新设明文 key 时保留，否则不落盘明文 |
| POST | `/api/config/test` | 翻译端点连通性测试（设置页「测试连接」用）：body 可选携带未保存的表单配置（`{"translate": {...}}`，api_key 空/mask = 沿用磁盘），不带则用磁盘 config；发「你好」最小请求，返回 `{ok, latency_ms, response_preview(前 50 字符), error}` |
| POST | `/api/preflight` | 手动触发完整启动预检（§11），返回 PreflightReport `{ok, checks[]}`（前端「系统检查」入口预留） |

启动：`subtitle-translator serve [--config path] [--host] [--port]`（host/port 默认取 ui 节配置）。

### 8.3 日志与脱敏（logsetup.py）

工具会分发给他人使用，**日志不能泄露隐私**：所有日志 handler 统一挂脱敏过滤器。

- **统一配置**（`logsetup.setup_logging`，cli 入口与 server `create_app` 共用，幂等可重复调用）：console handler（简洁格式 `级别 [模块] 消息`，级别取 `log.level`）+ RotatingFileHandler（`<log.dir>/subtitle-translator.log`，10MB×5 滚动，DEBUG 起，完整格式含时间/模块/行号）。root logger 固定 DEBUG，级别由各 handler 控制。`shutdown_logging()` 摘除本模块安装的全部 handler（测试隔离用）。
- **脱敏过滤器**（`SanitizeFilter`，全局单例挂到所有 handler，清洗每条记录的 message）：
  - 注册的密钥精确替换：`register_secret(resolve_api_key(cfg))` 在 cli 入口、server 启动、每个任务执行时（配置可能刚改过）各注册一次；短于 8 字符的值不注册（占位符如 `"0"` 做精确替换会毁掉正常日志）；
  - 通用模式打码：`sk-[A-Za-z0-9]{8,}` → `sk-***`；`Bearer xxx` → `Bearer ***`；`api_key=` / `apikey=` / `access_token=` / `token=` / `key=` 键值形态（含 URL query，不吞 `&` 后续参数）→ `键=***`；
  - 路径脱敏：用户主目录前缀（`Path.home()`）替换为 `~`。
- **per-task 日志**：任务开始执行时 TaskManager 建 `<log.dir>/tasks/<task_id>.log` 的 FileHandler 挂到 root logger（`create_task_log_handler`）；归属判定用 contextvars——任务执行期间进入 `task_log_context(task_id)`（`asyncio.to_thread` 会把 context 传播进 worker 线程，pipeline/transcribe/translate 的日志都算该任务的），handler 上的 `TaskLogFilter` 只放行本任务记录。任务边界记生命周期日志，失败时 `logger.exception` 落完整 traceback；任务结束（含 waiting_confirm）摘除 handler，文件保留供 `/api/tasks/{id}/log` 读取（resume 重跑追加写同一文件）。
- **约定**：诊断信息一律走 logger 不走 print；CLI 面向用户的输出（进度、结果、错误提示）保留 print。

## 9. 断点续传

每阶段产物落盘 JSON（`xxx.sub.json`），重跑时跳过已完成阶段（由 `stage` 字段驱动）。检查点中断（术语表待确认）也会先落盘 stage=contexted，确认术语后重跑即从逐句翻译续上；崩溃重启不丢已完成阶段的进度。

## 10. 依赖策略

- 核心依赖尽量轻（当前只有 pyyaml + openai）。
- `torch` / `qwen-asr` / `transformers` 列为 optional extra（`asr`），`vllm` 单列 `asr-vllm` extra（依赖 `asr`；新版 vLLM 对 Turing sm_75 等老架构支持不佳，老卡用 transformers backend），FastAPI 等为 `web` extra，按环境单独安装。
- 依赖版本只钉下界，不钉死。

## 11. 启动预检（preflight.py）

**设计原则：运行时炸不如启动时炸——只要能进入服务，就应该是 ready-to-use 的。** ffmpeg 缺失、模型文件不齐、翻译端点连不通这类问题，必须在 `run` / `serve` 启动时以清晰报错暴露，而不是转录/翻译跑到一半才炸。

- `run_preflight(cfg, *, need_translate=True, need_transcribe=True, download_missing=True, progress_cb=None) -> PreflightReport`：聚合全部检查项；`PreflightReport.checks` 每项为 `{name, status(ok/warn/fail), message}`，**有任一 fail 即整体不通过**（warn 不阻塞）。
- 检查项与判定：
  - **ffmpeg**（硬错误）：`media.find_ffmpeg` 探测，失败信息含安装指引；
  - **ASR / 对齐模型**（硬错误）：本地目录要求关键文件齐全（`config.json`、`tokenizer_config.json`、至少一个 `*.safetensors`，以 Qwen3-ASR-1.7B 实际结构为准）；hub ID 先查 HF 缓存快照（`~/.cache/huggingface/hub/models--*`，尊重 HF_HOME/HF_HUB_CACHE），未命中且 `download_missing=True` 时当场 `snapshot_download`（尊重 HF_ENDPOINT）；**HF 下载失败自动走 modelscope 兜底**（`_download_via_modelscope`：Python API `modelscope.snapshot_download(local_dir=...)` 下载到 `~/.subtitle_translator/models/<模型名>`，成功时经 `CheckResult.resolved_path` 把本地路径写回内存中的 `cfg.asr`——不落盘，message 提示用户写进 config.yaml 持久化；未安装 modelscope 或兜底也失败 = 硬错误，含 pip install modelscope / HF_ENDPOINT 镜像 / 手动下载指引）。模型标识分类：已存在目录 → 本地路径；恰好 `org/name` 形态 → hub ID；其余按「缺失的本地路径」硬错；
  - **翻译端点**（`need_translate` 时）：model / base_url 未配置 = 硬错误；api_key 缺失 = 警告（本地端点不需要 key）；连通性测试 `test_translate_endpoint`（发一条最小 chat 请求「你好」，max_tokens=8，超时 15s，不重试）失败 = 硬错误含排查指引。该函数同时被 `POST /api/config/test` 复用；
  - **GPU**（警告）：`torch.cuda.is_available()`，无 GPU 可跑 CPU 只是慢。
- 接线：`cli run` 在 pipeline 前执行（失败退出码 2；`--transcribe-only` 时 `need_translate=False`，`--from-json` 时 `need_transcribe=False`，`--skip-preflight` 逃生门）；`cli serve` 启动时执行但翻译端点检查降级为警告（serve 是长期进程，用户可能只转录或稍后配 key），且只做存在性检查与必要的模型下载，**模型加载仍留在首次任务时**（启动要快）；server TaskManager 每个任务执行前跑 `recheck_model_paths` 轻量复核（只查存在性，不下载不联网），防运行期间模型文件被删。
- 日志走 logging（脱敏过滤器覆盖），CLI 另以 `format_text()` 打印人可读报告。
