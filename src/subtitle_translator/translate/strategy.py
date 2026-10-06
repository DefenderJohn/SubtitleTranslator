"""翻译策略接口与滑动窗口策略实现（翻译三步走的编排）。

三步走：
1. 全片摘要（context.generate_summary），存入 project.meta.summary；
2. 术语表（glossary.extract_glossary）——生成后 stage 变为 CONTEXTED，
   逐句翻译开始前必须所有条目 confirmed=true（人工确认检查点，
   auto_confirm=True 时自动置 true，供 CLI --auto-confirm 使用）；
3. 滑动窗口逐句翻译：system 注入术语表+摘要，历史对拼成
   user/assistant 消息对，前瞻原文标注不翻译。调用间无服务端状态。

三步全部走 JSON schema 结构化输出（client.chat_json），不做格式降级；
摘要取 ``summary`` 字段、术语表取 ``entries``、译文取 ``translation`` 字段。

译文异常检测：空输出 / 明显复读（译文 > 原文 10 倍或 > 500 字符）触发
单条重试（最多 2 次，第二次降 temperature），仍失败保留原文占位并打
"translation_failed" 标记，不中断整体流程。

术语后校验（第一版只标记）：原文含某术语 src 而译文不含对应 dst，
在该 cue 上记 "glossary_miss:<src>"。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

from ..config import Config, TranslateConfig
from ..models import Cue, Stage, SubtitleProject
from . import prompts
from .client import ChatClient
from .context import generate_summary
from .glossary import extract_glossary

ProgressCb = Callable[[int, int], None]

# 单条译文异常判定：超过原文 10 倍长或超过 500 字符视为明显复读
REPEAT_MAX_RATIO = 10
REPEAT_MAX_CHARS = 500
# 单条翻译重试次数（第二次重试降低 temperature）
CUE_MAX_RETRIES = 2
# 第二次重试的 temperature 折扣
RETRY_TEMPERATURE_FACTOR = 0.5


class GlossaryNotConfirmedError(RuntimeError):
    """术语表存在未确认条目，拒绝开始逐句翻译（人工确认检查点）。"""


class TranslationStrategy(ABC):
    """翻译策略接口。``translate`` 原地修改并返回 project。"""

    @abstractmethod
    def translate(
        self,
        project: SubtitleProject,
        cfg: Union[Config, TranslateConfig],
        progress_cb: Optional[ProgressCb] = None,
        auto_confirm: bool = False,
    ) -> SubtitleProject:
        """执行翻译。auto_confirm=True 时自动确认术语表（CLI --auto-confirm）。"""
        raise NotImplementedError


def _is_suspicious(src: str, dst: str) -> bool:
    """空输出或明显复读（译文长度异常）。"""
    if not dst:
        return True
    return len(dst) > REPEAT_MAX_RATIO * max(len(src), 1) or len(dst) > REPEAT_MAX_CHARS


def _glossary_miss_flags(cue: Cue, project: SubtitleProject) -> list[str]:
    """术语后校验：原文含 src 而译文不含 dst 的术语，生成 glossary_miss 标记。"""
    flags: list[str] = []
    translation = cue.translation or ""
    for entry in project.glossary:
        if (
            entry.src
            and entry.src in cue.text
            and entry.dst
            and entry.dst not in translation
        ):
            flag = f"glossary_miss:{entry.src}"
            if flag not in cue.flags and flag not in flags:
                flags.append(flag)
    return flags


class SlidingWindowStrategy(TranslationStrategy):
    """滑动窗口逐句翻译策略（第一版唯一实现）。

    client 可注入（测试用 fake）；为 None 时按 cfg 创建 ChatClient。
    """

    def __init__(self, client: Optional[ChatClient] = None) -> None:
        self._client = client

    # ------------------------------------------------------------ 三步走

    def translate(
        self,
        project: SubtitleProject,
        cfg: Union[Config, TranslateConfig],
        progress_cb: Optional[ProgressCb] = None,
        auto_confirm: bool = False,
    ) -> SubtitleProject:
        translate_cfg = cfg.translate if isinstance(cfg, Config) else cfg
        client = self._client or ChatClient(translate_cfg)

        # ① 摘要：已有（断点续传）则跳过
        if not project.meta.summary:
            project.meta.summary = generate_summary(client.chat_json, project)
        # ② 术语表：已有（断点续传/人工改过）则跳过
        if not project.glossary:
            project.glossary = extract_glossary(client.chat_json, project, translate_cfg)
        project.stage = Stage.CONTEXTED

        # 人工确认检查点
        if auto_confirm:
            for entry in project.glossary:
                entry.confirmed = True
        unconfirmed = [g.src for g in project.glossary if not g.confirmed]
        if unconfirmed:
            raise GlossaryNotConfirmedError(
                f"术语表有 {len(unconfirmed)} 条未确认（如 {unconfirmed[:3]}），"
                "请人工确认（confirmed=true）后再开始逐句翻译，"
                "或使用 auto_confirm / CLI --auto-confirm 自动确认"
            )

        # ③ 滑动窗口逐句翻译
        self._translate_cues(client, project, translate_cfg, progress_cb)
        if translate_cfg.model:
            project.meta.models.translator = translate_cfg.model
        project.stage = Stage.TRANSLATED
        return project

    # ------------------------------------------------------------ 逐句翻译

    def _translate_cues(
        self,
        client: ChatClient,
        project: SubtitleProject,
        cfg: TranslateConfig,
        progress_cb: Optional[ProgressCb],
    ) -> None:
        system = prompts.translation_system(cfg, project.glossary, project.meta.summary)
        cues = project.cues
        total = len(cues)
        for i, cue in enumerate(cues):
            if cue.translation is not None:
                # 断点续传：已翻译的跳过
                if progress_cb:
                    progress_cb(i + 1, total)
                continue
            messages = self._build_messages(system, cues, i, cfg)
            cue.translation = self._translate_one(client, messages, cue, cfg)
            cue.flags.extend(_glossary_miss_flags(cue, project))
            if progress_cb:
                progress_cb(i + 1, total)

    def _build_messages(
        self,
        system: str,
        cues: list[Cue],
        index: int,
        cfg: TranslateConfig,
    ) -> list[dict[str, str]]:
        """组装 messages：system + 历史对（user/assistant）+ 当前 user（含前瞻）。"""
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        # 前 history_count 条已翻译的「原文→译文」历史对
        history = [c for c in cues[:index] if c.translation is not None]
        for prev in history[-cfg.history_count :]:
            messages.append({"role": "user", "content": prev.text})
            messages.append({"role": "assistant", "content": str(prev.translation)})
        # 后 forward_count 条原文作为前瞻参考
        forward = [c.text for c in cues[index + 1 : index + 1 + cfg.forward_count]]
        messages.append(
            {"role": "user", "content": prompts.translation_user(cues[index].text, forward)}
        )
        return messages

    def _translate_one(
        self,
        client: ChatClient,
        messages: list[dict[str, str]],
        cue: Cue,
        cfg: TranslateConfig,
    ) -> str:
        """单条翻译：异常输出重试（最多 2 次，第二次降 temperature），
        仍失败保留原文占位并打 translation_failed 标记。"""
        for attempt in range(CUE_MAX_RETRIES + 1):
            temperature = cfg.temperature
            if attempt == CUE_MAX_RETRIES:
                temperature = max(0.0, cfg.temperature * RETRY_TEMPERATURE_FACTOR)
            data = client.chat_json(
                messages,
                schema=prompts.TRANSLATION_RESPONSE_SCHEMA,
                name="translation",
                temperature=temperature,
            )
            result = str(data.get("translation") or "").strip()
            if not _is_suspicious(cue.text, result):
                return result
        if "translation_failed" not in cue.flags:
            cue.flags.append("translation_failed")
        return cue.text
