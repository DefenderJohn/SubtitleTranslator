"""转录层（占位，后续阶段实现）。

使用官方 ``qwen-asr`` 包（Qwen3ASRModel）：模型 Qwen/Qwen3-ASR-1.7B，
配合 Qwen/Qwen3-ForcedAligner-0.6B 出词级时间戳。

- 两个 backend：vLLM 优先、transformers 兜底，做成配置项，对上层零差异。
- 硬约束：ForcedAligner 单次只支持 ≤5 分钟音频，因此本层包含自写切块：
  用 ffmpeg silencedetect 找静音边界切块（不引入 silero-vad 等额外 torch
  依赖），逐块转录 + 对齐后拼接并偏移时间戳。
- ffmpeg 是硬依赖（将来剪辑功能也要用）。
"""
