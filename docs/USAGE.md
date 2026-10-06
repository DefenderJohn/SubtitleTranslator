# SubtitleTranslator 使用文档

面向使用者。本文所有命令、参数与默认值均已与实际代码核对（`src/subtitle_translator/cli.py`、`src/subtitle_translator/config.py`）。设计与开发信息见 [DESIGN.md](DESIGN.md)、[ENVIRONMENT.md](ENVIRONMENT.md)。

## 1. 简介

SubtitleTranslator 把音视频文件变成翻译字幕：本地 ASR 模型转录出带词级时间戳的字幕，再交给任意 OpenAI 兼容的 LLM 端点（DeepSeek、通义、本地 vLLM/Ollama 均可）翻译，最后导出 SRT 或双语 SRT。

```
音视频文件
   │  转录（Qwen3-ASR + ForcedAligner，本地 GPU）
   ▼
字幕行（原文 + 词级时间戳）
   │  翻译三步走（OpenAI 兼容端点）
   │  ① 生成全文摘要 → ② 提取术语表（人工确认检查点）→ ③ 滑动窗口逐句翻译
   ▼
导出 SRT / 双语 SRT（译文在上、原文在下）
```

特点：

- 每个媒体文件对应一个 `xxx.sub.json` 工程文件（唯一事实来源），每完成一个阶段立即落盘，中断后重跑自动续传。
- 术语表是**人工确认检查点**：翻译开始前可以先过目、修改专名译法，避免全片人名/地名翻错。
- 命令行（CLI）和网页两种用法，共用同一份 `config.yaml`。

## 2. 安装

### 2.1 环境要求

- **NVIDIA GPU**：转录在本地显卡上跑。Turing 代老卡（如 RTX 2080 Ti，sm_75）实测可用，注意两点：
  - `dtype` 用 `float16`（默认配置已是），Turing 不支持 bf16 原生计算；
  - **不要装 flash-attn**（要求 sm_80+），transformers 默认的 sdpa 注意力即可。
- **conda**（或任意 Python 环境管理）：建议单独建 Python 3.12 环境。
- ffmpeg 不用单独装（见 2.3）。
- 网页功能需要 Node.js，但**仅在自己重新构建前端时才需要**；仓库已带构建好的 `frontend/dist`，直接用即可。

### 2.2 安装步骤（实测路径，2026-10 于 RTX 2080 Ti 验证）

```bash
conda create -n subtr python=3.12 -y
conda activate subtr

cd SubtitleTranslator
pip install -e ".[asr,web]"     # core + 转录层（qwen-asr/torch/transformers）+ 网页服务
pip install imageio-ffmpeg      # 提供 ffmpeg 静态二进制
```

说明：

- `pip install -e .` 只装 core（pyyaml + openai），只能导出/查看术语表等，不能转录。
- 转录层在 `asr` extra 里。**默认 backend 是 transformers，装好 `asr` extra 开箱即用**。
- vLLM backend 更快，但需另装 `pip install -e ".[asr-vllm]"`，且新版 vLLM 对 Turing（sm_75）等老卡支持不佳，**老卡不建议**。装好 vLLM 后想用它，把 config 的 `asr.backend` 改成 `vllm`（见 3.2）。

### 2.3 ffmpeg

ffmpeg 是硬依赖，但无需单独安装：二进制解析顺序为 config 的 `asr.ffmpeg_path` → 系统 PATH → imageio-ffmpeg 自带的静态二进制。上面装了 `imageio-ffmpeg` 就已覆盖。

### 2.4 模型下载

需要两个模型（转录 + 词级时间戳对齐）：

- `Qwen/Qwen3-ASR-1.7B`（约 4.4GB）
- `Qwen/Qwen3-ForcedAligner-0.6B`（约 1.8GB）

国内推荐 modelscope：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir /path/to/models/Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir /path/to/models/Qwen3-ForcedAligner-0.6B
```

或 Hugging Face：

```bash
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir /path/to/models/Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir /path/to/models/Qwen3-ForcedAligner-0.6B
```

下载后在 `config.yaml` 里把 `asr.model` / `asr.aligner_model` 填成上面的本地路径（见 3.2）。网络慢时可先用 aria2 等多连接工具下载（如 `aria2c -x 16 <文件URL>`）。

## 3. 配置

### 3.1 生成 config.yaml

```bash
subtitle-translator config init          # 生成 ./config.yaml（已存在则报错，需先删除）
subtitle-translator config init my.yaml  # 指定输出路径
```

所有命令默认读当前目录的 `config.yaml`（`run`/`serve` 可用 `--config` 指定别的路径）；文件不存在时用内置默认值。

### 3.2 asr 节（转录）

```yaml
asr:
  backend: transformers    # transformers（默认，开箱即用）| vllm（可选加速，老卡不要改）
  model: Qwen/Qwen3-ASR-1.7B                 # 填模型 ID 或本地路径
  aligner_model: Qwen/Qwen3-ForcedAligner-0.6B
  device: cuda
  dtype: float16           # Turing 等老卡用 float16；支持 bf16 的卡可改 bfloat16
  chunk_max_seconds: 290.0 # 静音切块长度上限（对齐器输入限制，一般不用改）
  language: null           # 源语言，null=自动检测；可填 en/zh/ja 等 ISO 代码
  ffmpeg_path: ""          # ffmpeg 路径，空=自动探测（PATH → imageio-ffmpeg）
```

### 3.3 translate 节（翻译）

```yaml
translate:
  base_url: http://127.0.0.1:8000/v1   # OpenAI 兼容端点
  api_key: null
  api_key_env: ""          # 环境变量名（推荐，密钥不落盘）
  model: ""                # 必填，不填跑翻译会报错
  temperature: 0.7
  history_count: 10        # 滑动窗口：每条带前 10 条已译上下文
  forward_count: 1         # 滑动窗口：附带后 1 条原文（标注「不要翻译」）
  glossary_max_entries: 50           # 术语表条数上限
  additional_prompt: 翻译当前字幕到简体中文   # 追加进 system prompt 的指令
  target_language: 简体中文
  request_timeout: 120.0   # 单次请求超时（秒）
  max_retries: 4           # 429/5xx/超时的指数退避重试次数
  glossary_max_retries: 3  # 术语表解析失败的重试次数
```

三个常见端点的填法：

```yaml
# DeepSeek
translate:
  base_url: https://api.deepseek.com/v1
  api_key_env: DEEPSEEK_API_KEY
  model: deepseek-chat

# 通义千问（DashScope 兼容模式）
translate:
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key_env: DASHSCOPE_API_KEY
  model: qwen-plus

# 本地 vLLM / Ollama（无鉴权，api_key 可留空；启动时的警告可忽略）
translate:
  base_url: http://127.0.0.1:8000/v1
  model: qwen2.5-7b-instruct    # 填本地服务实际加载的模型名
```

**密钥安全实践**：不要把 api_key 明文写进 config.yaml。设环境变量（如 `export DEEPSEEK_API_KEY=sk-...`，写进 shell rc），config 里只填 `api_key_env: DEEPSEEK_API_KEY`。解析顺序：`api_key_env` 指向的环境变量优先，其次才是 yaml 明文；保存配置时默认也不会把明文 key 落盘。

**滑动窗口参数**（`history_count` / `forward_count`）：逐句翻译时，每条字幕的请求会带上前面 `history_count` 条已翻译的「原文→译文」作为上下文（保证上下文连贯），再附后面 `forward_count` 条原文供参考（模型被明确告知不要翻译它们）。一般不用改；上下文很长的剧集可适当调大 `history_count`，代价是更多 token。

### 3.4 ui 节（网页）

```yaml
ui:
  host: 127.0.0.1
  port: 7860
  upload_dir: ""       # 网页上传文件的存储目录，空=~/.subtitle_translator/uploads
```

`subtitle-translator serve` 的默认监听地址，`--host` / `--port` 可临时覆盖。网页「上传文件」模式把浏览器本地文件上传到 `upload_dir` 下的批次子目录（`YYYYMMDD-HHMMSS-xxxxxx/`），工程文件和 SRT 也产在媒体文件旁边（即该批次目录内）。

**上传的文件是临时工作副本**：server 正常退出（Ctrl+C）时会删除本次会话上传的全部批次目录；若进程被杀（kill -9 / 断电），下次启动时自动清理遗留批次。所以上传模式的 SRT 要在退出服务前用「下载」按钮保存到浏览器本地；想长期保留产物请用「服务器路径」模式（那里的文件是你自己的，永远不会被删）。

## 4. CLI 使用

### 4.1 完整流程

```bash
subtitle-translator run movie.mp4 --auto-confirm
```

转录 → 摘要+术语表 → 逐句翻译 → 导出双语 SRT（`movie.srt`，译文在上）。进度打在 stderr。

`run` 的路径可以是：

- **单个媒体文件**：支持 flac/m4a/mp3/mp4/mpeg/mpga/oga/ogg/wav/webm/mkv/mov/avi/m4v；
- **目录**：递归找全部媒体文件批量处理，单个失败不中断，最后汇总成功/失败清单（有失败时退出码为 1）。

不加 `--auto-confirm` 时，术语表生成后流水线会停在人工确认检查点（见 4.3 术语表一节）。

### 4.2 常用场景

```bash
# 只转录，不翻译（导出原文 SRT，不触碰翻译端点）
subtitle-translator run movie.mp4 --transcribe-only

# 无人值守：自动确认术语表，一口气跑完
subtitle-translator run movie.mp4 --auto-confirm

# 只要原文 SRT（transcribe-only 本身就是单语原文导出）
subtitle-translator run movie.mp4 --transcribe-only

# 指定源语言（自动检测认错时用；支持 en/zh/ja 等 ISO 代码）
subtitle-translator run movie.mp4 --language en

# 导出单语 SRT 而不是双语（默认双语）
subtitle-translator run movie.mp4 --auto-confirm --no-bilingual

# 重新翻译已有结果：不接触媒体文件，直接从工程文件续跑
# （改了术语表 / 换了翻译模型 / 调了 additional_prompt 之后用）
subtitle-translator run movie.sub.json --from-json --auto-confirm

# 用另一份配置文件
subtitle-translator run movie.mp4 --config my.yaml

# 从工程文件再导出一份 SRT（--bilingual 双语、译文在上；-o 指定输出，默认与工程文件同名 .srt）
subtitle-translator export movie.sub.json --bilingual -o out.srt

# 查看 / 确认术语表
subtitle-translator glossary movie.sub.json --show
subtitle-translator glossary movie.sub.json --confirm-all
```

### 4.3 术语表人工确认检查点

不加 `--auto-confirm` 的完整流程，在术语表生成后会停下来报 `GlossaryNotConfirmedError`（此时摘要和术语表已落盘）。流程：

```bash
subtitle-translator glossary movie.sub.json --show          # 过目术语（格式：原文 | 译文 | 出现次数）
# 想改译法：用编辑器直接改 movie.sub.json 里 glossary 的 dst 字段（或在网页里改）
subtitle-translator glossary movie.sub.json --confirm-all   # 全部确认
subtitle-translator run movie.mp4                           # 重跑，从逐句翻译续上
```

### 4.4 断点续传

每个媒体文件对应同目录下的 `xxx.sub.json` 工程文件，内含 `stage` 字段记录进度：

```
empty → transcribed（转录完成）→ contexted（摘要+术语表完成）→ translated（翻译完成）
```

每完成一个阶段立即落盘。中断（Ctrl-C、崩溃、检查点停住）后直接重跑同一条命令即可：已完成的阶段自动跳过。翻译阶段中途取消时，已翻译的字幕也会先落盘。

想强制重跑某一步，删掉对应产物即可：

- 重跑全部：删掉 `xxx.sub.json`；
- 只重跑翻译/导出：用 `--from-json`（不碰媒体文件），或改完术语表后直接重跑 `run`。

## 5. 网页使用

### 5.1 启动

```bash
pip install -e ".[web]"        # 若安装时没带 web extra
subtitle-translator serve      # 默认 http://127.0.0.1:7860（取 ui 节配置）
subtitle-translator serve --port 8000 --host 0.0.0.0   # 临时覆盖
```

前端 `frontend/dist` 已随仓库构建好，由后端直接托管，浏览器打开 `http://127.0.0.1:7860` 即可。API 文档在 `/docs`。

### 5.2 操作流程

1. **新建任务**（任务页）：「媒体来源」二选一——**服务器路径**（输入路径或用目录浏览弹窗选择媒体文件/目录，弹窗默认从用户主目录开始；浏览时遇到无权限的目录/文件会自动跳过）；**上传文件**（点击或拖拽本地文件到上传区，可多选，逐文件显示进度，上传到服务器 `ui.upload_dir` 的批次子目录）。选项有「自动确认术语 / 双语导出 / 仅转录 / 源语言」。点「创建任务」后，路径模式创建一个任务，上传模式每个已上传文件各创建一个任务。
2. **看进度**：任务列表实时进度条，详情页有事件日志流（页面后打开也能回放历史进度）。单并发队列：同时只跑一个任务，其余排队（本地单 GPU）。可随时取消（当前字幕/块完成后停，进度已落盘）。
3. **术语表确认**：未勾选「自动确认术语」时，任务在术语表生成后进入 `waiting_confirm`。详情页展示全文摘要和可编辑的术语表——**这是定稿专名译法的人工检查点**：人名、地名、作品名在全片翻译前统一过目修改一次，比事后逐条改字幕省力得多。确认后点「全部确认并继续」续跑。
4. **字幕校对**（详情页 tab）：cue 表格可编辑原文/译文；术语不一致的行（译文漏了术语表译法）红色标记；右侧视频预览，点击 cue 行跳转到对应时间点播放（上传的文件同样支持预览）。
5. **导出与下载**：任务完成后一键导出 SRT / 双语 SRT，页面显示结果路径；旁边的「下载」按钮直接把 SRT 下载到浏览器本地。**上传模式的产物是临时的**——server 退出（Ctrl+C）时本次上传的批次目录会被自动清理，SRT 记得在退出前下载。

设置页可在线读写 config.yaml 三组配置（ASR / 翻译 / 界面）；api_key 只显示掩码、留空不修改；每次执行任务都会重新加载 config.yaml，改配置对后续任务立即生效。

## 6. 工作目录产物说明

| 文件 | 说明 |
| --- | --- |
| `xxx.sub.json` | 工程文件，**唯一事实来源**：源文件信息、模型、摘要、术语表、全部字幕行（原文/译文/词级时间戳/标记）、`stage` 进度。SRT、网页编辑都读写它。**别手改**，除非知道在干什么（唯一建议手改的场景：改术语表的 `dst`） |
| `xxx.srt` | 导出物。双语时译文在上、原文在下；单语时优先译文、无译文退化为原文。随时可以 `export` 重新生成 |

字幕行的 `flags` 标记含义：`translation_failed`（重试后仍失败，保留原文占位）、`glossary_miss:<术语>`（原文含该术语但译文没用对应译法）。

## 7. 常见问题

**显存不够 / OOM**
换小模型：`asr.model` 改成 `Qwen/Qwen3-ASR-0.6B`（对齐器本来就是 0.6B）。同时确认 `dtype: float16`、没有重复加载模型的其他进程占显存。

**语言识别错了**
`--language en`（或 zh/ja 等 ISO 代码）指定源语言，也可在 config 的 `asr.language` 固定。注意填 ISO 代码或规范语言名（English/Chinese 等），乱填会被 ASR 拒绝。

**术语翻错了怎么补救**
改术语表（网页术语表编辑，或改 `xxx.sub.json` 里 glossary 的 `dst`），确认后 `--from-json` 重跑翻译：

```bash
subtitle-translator glossary movie.sub.json --confirm-all
subtitle-translator run movie.sub.json --from-json --auto-confirm
```

**翻译接口报错排查**
- `translate.model 未配置`：config 里填 `model`。
- 401/鉴权失败：`api_key_env` 指向的环境变量是否真的设置了（`echo $DEEPSEEK_API_KEY`）；变量名不要填成密钥本身。
- 连接 refused：`base_url` 是否带 `/v1` 后缀、端口对不对；本地端点先确认服务起来了。
- 429/5xx/超时：客户端会自动指数退避重试（默认最多 4 次、单次超时 120 秒），偶发不用管；持续失败再调 `request_timeout` / `max_retries` 或换端点。
- 没有真实 API key 也想验证翻译链路：仓库自带 mock 端点 `python scripts/mock_translate_server.py --port 8399`，把 `translate.base_url` 指向 `http://127.0.0.1:8399/v1` 即可。

**想换 Whisper 或其他 ASR**
架构上转录 backend 是可插拔的（`AsrBackend` 抽象），但当前只实现了 Qwen3-ASR；换 Whisper 需要写一个新的 backend 实现，属于开发改动，见 [DESIGN.md](DESIGN.md) 第 3 节。

**ffmpeg 找不到**
`pip install imageio-ffmpeg`（每个 Python 环境都要装一份）；或系统装 ffmpeg 放 PATH；或在 config 里写死 `asr.ffmpeg_path`。
