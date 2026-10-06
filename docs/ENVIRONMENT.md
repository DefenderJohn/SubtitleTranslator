# 环境摸底（2026-10-06，当日完成真实依赖安装与端到端调测）

| 项目 | 结果 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS，x86_64 |
| Python | 运行环境：**conda env `subtr`，Python 3.12.15**（/home/john/miniconda3/envs/subtr）；base 为 3.13.11（仅开发/测试用） |
| 包管理 | pip 25.3、uv 0.11.6 均可用；pypi 走清华镜像 |
| Node.js | v22.22.0、npm 10.9.4（系统 /usr/bin/node，2026-10-06 前端阶段确认可用）；npm registry 官方源（registry.npmjs.org）连通 |
| ffmpeg | **已通过 imageio-ffmpeg 0.6.0 获得**（静态二进制 ffmpeg 7.0.2）；PATH 上无 ffmpeg/ffprobe。代码用 `asr.ffmpeg_path` 配置或自动探测（PATH → imageio-ffmpeg），时长用 `ffmpeg -i` stderr 解析兜底 |
| GPU | NVIDIA GeForce RTX 2080 Ti，22GB 显存（sm_75 Turing） |
| 驱动 / CUDA | 驱动 590.57，CUDA 13.1；torch 轮子为 cu130 |
| 内存 | 62Gi 总量，约 58Gi 可用 |
| 模型 | 本地目录：`/home/john/translator/models/Qwen3-ASR-1.7B`（4.4GB）、`/home/john/translator/models/Qwen3-ForcedAligner-0.6B`（1.8GB），来自 modelscope（aria2c 多连接） |

## subtr 环境关键包版本（2026-10-06 实测）

- torch 2.13.0+cu130（CUDA 可用，2080 Ti 识别正常）、triton 3.7.1
- transformers 4.57.6（qwen-asr 0.0.6 钉死此版本）、accelerate 1.12.0
- qwen-asr 0.0.6（PyPI 实际版本线是 0.0.x，pyproject 钉 `>=0.0.5`）
- openai 3.x、fastapi 0.142、uvicorn 0.54、httpx 0.28、pytest 8.x、imageio-ffmpeg 0.6.0

## backend 结论：**transformers**（vLLM 放弃）

- **vLLM 未安装、未验证**：本机网络单连接仅 ~0.15MB/s，vLLM 路线需额外数 GB 下载，时间成本不可接受；且新版 vLLM 对 Turing（sm_75）兼容性存疑。按时间盒纪律直接转 transformers backend，一次跑通。
- vLLM 支持保留在代码与 `asr-vllm` extra 中（Turing 老卡不建议；sm_80+ 机器可试 `pip install -e ".[asr-vllm]"`）。
- **transformers backend 实测**：模型加载 + 30s 音频首推理约 27s；`dtype=float16`（Turing 不支持 bf16 原生计算，config 默认已改 float16）；attn 用 transformers 默认（sdpa），**不要装 flash-attn**（要求 sm_80+）。

## 真实 API 对齐结论（qwen-asr 0.0.6 源码确认）

- `Qwen3ASRModel.from_pretrained(path, dtype=torch 对象, device_map=..., forced_aligner=..., forced_aligner_kwargs=dict(dtype=..., device_map=...))`；dtype 是 torch 对象不是字符串（代码内 `_resolve_dtype` 转换）。
- `transcribe(audio=..., language=..., return_time_stamps=True)` 返回 `ASRTranscription(language, text, time_stamps)`；`time_stamps` 是 `ForcedAlignResult`（可迭代 `.items`），词元素字段为 **text / start_time / end_time**。
- `language` 只接受规范语言名（"English"/"Chinese"...，共 30 种）；ISO 代码（en/zh）会被拒，代码内 `_normalize_language` 做映射。
- qwen-asr 内部自行切块（return_time_stamps 时上限 180s），本项目的静音切块（默认 290s）仍保留用于在静音边界下刀，两层兼容。

## 踩坑记录

1. **PyPI 上 qwen-asr 版本线是 0.0.x**：pyproject 原钉 `>=0.1` 直接无解，改为 `>=0.0.5`。
2. **HF 直连 hf CLI 下载失败**（HEAD 请求被代里拦）：改用 modelscope + aria2c 16 连接（单连接 ~0.3MB/s，16 连接 ~2.5MB/s）。
3. **pip --find-links 混远程 index 时会去拉 index 上更新的 torch**（2.14.x，~1GB），本地 wheel 白下；解法：安装命令显式钉 `torch==2.13.0` 或换 uv 解析。
4. **机械硬盘 + aria2c 16 连接随机写会把 load 打到 20+**：大下载期间整机卡顿属正常。
5. subtr 环境是新环境，imageio-ffmpeg 也要装一份（ffmpeg 二进制按环境提供）。
6. 测试若断言「未安装 qwen-asr 时报错」，在装了 qwen-asr 的环境里要用条件跳过（两环境互补各跳 1 个）。

## 端到端验证记录（2026-10-06）

- **真实转录**：`subtitle-translator run c/test.ogg --transcribe-only --language en`（transformers backend，本地模型）——30.09s 英文广告（UPC），10 cues / 58 词级时间戳，时间戳单调且在时长内，SRT 可被 parse_srt 读回。
- **翻译链路**：环境无任何 API key（真实端点未验证）；用 scripts/mock_translate_server.py（stdlib mock OpenAI 端点）验证 openai SDK → HTTP → 解析全通：摘要→术语表→逐句→双语 SRT，stage=translated。
- **serve 联调**：curl 验证 创建任务 → SSE 事件 → waiting_confirm → PATCH glossary confirm_all → resume → done → export 全状态机通过。

## 已安装的相关 Python 包（base conda 环境，开发/测试用）

- torch 2.12.0
- openai-whisper 20250625（旧版遗留）
- PyYAML 6.0.3、ruamel.yaml 0.18.16
- setuptools 80.9.0、wheel 0.45.1
- pytest 8.x、imageio-ffmpeg 0.6.0（静态 ffmpeg 7.0.2 二进制，含 silencedetect）
- fastapi 0.142、uvicorn 0.54、httpx 0.28（网页服务，`web` / `dev` extras）
- **base 未安装**：qwen-asr、vllm、transformers（重依赖全在 subtr）
