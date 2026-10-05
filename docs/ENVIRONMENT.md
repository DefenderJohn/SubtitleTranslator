# 环境摸底（2026-10-06）

| 项目 | 结果 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS，x86_64 |
| Python | 3.13.11（miniconda，/home/john/miniconda3） |
| 包管理 | pip 25.3、uv 0.11.6 均可用 |
| ffmpeg | **未安装**（`ffmpeg` / `ffprobe` 均不存在） |
| GPU | NVIDIA GeForce RTX 2080 Ti，22GB 显存（sm_75 Turing） |
| 驱动 / CUDA | 驱动 590.57，CUDA 13.1 |
| 内存 | 62Gi 总量，约 58Gi 可用 |

## 已安装的相关 Python 包（base conda 环境）

- torch 2.12.0
- openai-whisper 20250625（旧版遗留）
- PyYAML 6.0.3、ruamel.yaml 0.18.16
- setuptools 80.9.0、wheel 0.45.1
- **未安装**：pytest、fastapi、uvicorn、qwen-asr、vllm、transformers

## 风险与注意事项

1. **ffmpeg 缺失是最大阻塞项**：转录切块（silencedetect）和将来的剪辑都硬依赖 ffmpeg，开始实现转录层前必须安装（`apt install ffmpeg` 或 conda 安装）。
2. **GPU 当前被占用**：摸底时显存已用约 21GB / 22.5GB、利用率 27%，说明有其他进程在用卡；跑 ASR 前需确认显存空闲。
3. **RTX 2080 Ti 是 Turing（sm_75）**：新版 vLLM 对老架构的支持需要实际验证；transformers 兜底 backend 因此更重要。
4. **Python 3.13**：qwen-asr / vllm 等重依赖对 3.13 的兼容性需验证，必要时为其单独建环境（uv 可用）。
5. 本摸底只做了检查，未安装任何大型依赖。
