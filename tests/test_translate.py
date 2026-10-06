"""翻译层测试：全部 fake client.chat_json，不打真实 API。"""

from __future__ import annotations

import pytest

from subtitle_translator.config import TranslateConfig
from subtitle_translator.models import Cue, GlossaryEntry, Stage, SubtitleProject
from subtitle_translator.translate import (
    ChatClient,
    ChatError,
    GlossaryExtractError,
    GlossaryNotConfirmedError,
    GlossaryParseError,
    InvalidModelJsonError,
    SlidingWindowStrategy,
    parse_glossary_payload,
)
from subtitle_translator.translate import context, prompts

GLOSSARY_PAYLOAD = {
    "entries": [
        {"src": "Erebus", "dst": "厄瑞玻斯", "count": 5},
        {"src": "Nyra", "dst": "妮拉", "count": 3},
    ]
}


class FakeClient:
    """按 schema name 路由的 fake chat_json 客户端，记录全部调用。

    术语表 / 逐句翻译的响应可用 script 队列编排：队列元素为 dict 原样返回，
    为异常实例则抛出（模拟 InvalidModelJsonError 等）。
    """

    def __init__(self, translations=None, glossary_script=None, summary="测试摘要"):
        self.calls: list[tuple[list[dict], str, float | None]] = []
        self.summary = summary
        # 逐句翻译：callable(messages) -> str，或固定字符串
        self.translations = "默认译文" if translations is None else translations
        # 术语表：响应队列（依次消费），缺省返回 GLOSSARY_PAYLOAD
        self.glossary_script = list(glossary_script) if glossary_script else None

    def chat_json(self, messages, *, schema, name, temperature=None):
        self.calls.append(([dict(m) for m in messages], name, temperature))
        if name == "summary":
            return {"summary": self.summary}
        if name == "glossary":
            outcome = (
                self.glossary_script.pop(0)
                if self.glossary_script is not None
                else GLOSSARY_PAYLOAD
            )
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        # 逐句翻译
        if callable(self.translations):
            return {"translation": self.translations(messages)}
        return {"translation": self.translations}

    def glossary_calls(self) -> list:
        return [c for c in self.calls if c[1] == "glossary"]


def make_cfg(**overrides) -> TranslateConfig:
    defaults = dict(
        base_url="http://fake/v1",
        model="test-model",
        target_language="简体中文",
        additional_prompt="",
        history_count=2,
        forward_count=1,
        temperature=0.7,
    )
    defaults.update(overrides)
    return TranslateConfig(**defaults)


def make_project(n_cues=3, with_context=False) -> SubtitleProject:
    project = SubtitleProject()
    project.cues = [
        Cue(id=i + 1, start=float(i), end=float(i) + 1.0, text=f"原文第{i + 1}句 Erebus")
        for i in range(n_cues)
    ]
    project.stage = Stage.TRANSCRIBED
    if with_context:
        project.meta.summary = "已有摘要"
        project.glossary = [
            GlossaryEntry(src="Erebus", dst="厄瑞玻斯", count=5, confirmed=True)
        ]
    return project


# ------------------------------------------------------------------ 正常流


def test_full_flow():
    project = make_project(3)
    client = FakeClient()
    progress: list[tuple[int, int]] = []
    strategy = SlidingWindowStrategy(client=client)

    result = strategy.translate(
        project, make_cfg(), progress_cb=lambda d, t: progress.append((d, t)),
        auto_confirm=True,
    )

    assert result is project
    assert project.stage == Stage.TRANSLATED
    assert project.meta.summary == "测试摘要"
    assert [g.src for g in project.glossary] == ["Erebus", "Nyra"]
    assert project.glossary[0].count == 5
    assert all(g.confirmed for g in project.glossary)
    assert all(c.translation == "默认译文" for c in project.cues)
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert project.meta.models.translator == "test-model"


def test_full_flow_uses_json_schema():
    """三步调用都带对应 JSON schema。"""
    project = make_project(1)
    client = FakeClient()
    SlidingWindowStrategy(client=client).translate(project, make_cfg(), auto_confirm=True)
    names = [name for _, name, _ in client.calls]
    assert names == ["summary", "glossary", "translation"]


def test_resume_skips_summary_and_glossary():
    """已有摘要+术语表（已确认）时不再调用前两步，直接逐句翻译。"""
    project = make_project(2, with_context=True)
    client = FakeClient()
    SlidingWindowStrategy(client=client).translate(project, make_cfg())
    assert [name for _, name, _ in client.calls] == ["translation", "translation"]


def test_resume_skips_translated_cues():
    project = make_project(3, with_context=True)
    project.cues[0].translation = "已翻好的"
    client = FakeClient()
    SlidingWindowStrategy(client=client).translate(project, make_cfg())
    assert len(client.calls) == 2  # 只翻第 2、3 条
    assert project.cues[0].translation == "已翻好的"


# ------------------------------------------------------------------ 术语表


def test_parse_glossary_payload():
    entries = parse_glossary_payload(GLOSSARY_PAYLOAD)
    assert [(e.src, e.dst, e.count) for e in entries] == [
        ("Erebus", "厄瑞玻斯", 5),
        ("Nyra", "妮拉", 3),
    ]


def test_parse_glossary_payload_skips_malformed_items():
    """缺 src/dst 的条目跳过；count 非法时按 0；多余字段忽略。"""
    entries = parse_glossary_payload(
        {
            "entries": [
                {"src": "Erebus", "dst": "厄瑞玻斯", "count": "5", "extra": 1},
                {"src": "", "dst": "缺原文"},
                {"dst": "缺 src"},
                "不是对象",
                {"src": "Nyra", "dst": "妮拉"},
            ]
        }
    )
    assert [(e.src, e.dst, e.count) for e in entries] == [
        ("Erebus", "厄瑞玻斯", 5),
        ("Nyra", "妮拉", 0),
    ]


def test_parse_glossary_payload_no_entries_raises():
    with pytest.raises(GlossaryParseError):
        parse_glossary_payload({"wrong_key": []})
    with pytest.raises(GlossaryParseError):
        parse_glossary_payload({"entries": []})


def test_glossary_invalid_payload_retries_then_success():
    """术语表 payload 无效 → 重试 → 成功。"""
    project = make_project(1)
    client = FakeClient(glossary_script=[{"废话": True}, GLOSSARY_PAYLOAD])
    SlidingWindowStrategy(client=client).translate(
        project, make_cfg(), auto_confirm=True
    )
    assert len(client.glossary_calls()) == 2
    assert [g.src for g in project.glossary] == ["Erebus", "Nyra"]


def test_glossary_invalid_json_retries():
    """端点返回非法 JSON（InvalidModelJsonError）同样计入重试。"""
    project = make_project(1)
    client = FakeClient(
        glossary_script=[InvalidModelJsonError("not json"), GLOSSARY_PAYLOAD]
    )
    SlidingWindowStrategy(client=client).translate(
        project, make_cfg(), auto_confirm=True
    )
    assert len(client.glossary_calls()) == 2


def test_glossary_total_failure_raises():
    """术语表重试耗尽仍无法解析 → 抛 GlossaryExtractError。"""
    project = make_project(1)
    cfg = make_cfg(glossary_max_retries=3)
    client = FakeClient(glossary_script=[{"废话": True}] * 10)
    with pytest.raises(GlossaryExtractError):
        SlidingWindowStrategy(client=client).translate(project, cfg, auto_confirm=True)
    assert len(client.glossary_calls()) == 3


def test_glossary_prompt_mentions_json_schema():
    messages = prompts.glossary_messages("摘要", "正文", max_entries=10, target_language="简体中文")
    assert '"entries"' in messages[1]["content"]
    schema = prompts.GLOSSARY_RESPONSE_SCHEMA
    item_props = schema["properties"]["entries"]["items"]["properties"]
    assert set(item_props) == {"src", "dst", "count"}


def test_unconfirmed_glossary_blocks_translation():
    """存在未确认条目时拒绝逐句翻译，stage 停在 CONTEXTED。"""
    project = make_project(2)
    client = FakeClient()
    with pytest.raises(GlossaryNotConfirmedError):
        SlidingWindowStrategy(client=client).translate(project, make_cfg())
    assert project.stage == Stage.CONTEXTED
    assert project.meta.summary == "测试摘要"  # 前两步已完成
    assert all(c.translation is None for c in project.cues)


def test_auto_confirm_sets_all_confirmed():
    project = make_project(1)
    client = FakeClient()
    SlidingWindowStrategy(client=client).translate(
        project, make_cfg(), auto_confirm=True
    )
    assert all(g.confirmed for g in project.glossary)


# ------------------------------------------------------------------ 逐句翻译异常检测


def test_empty_output_retries_then_placeholder():
    """空输出重试 2 次仍失败 → 保留原文占位 + translation_failed 标记。"""
    project = make_project(1, with_context=True)
    client = FakeClient(translations="")
    SlidingWindowStrategy(client=client).translate(project, make_cfg())
    cue = project.cues[0]
    assert cue.translation == cue.text
    assert "translation_failed" in cue.flags
    assert len(client.calls) == 3  # 1 次首发 + 2 次重试


def test_missing_translation_field_treated_as_empty():
    """payload 缺 translation 字段按空输出处理，走重试/占位路径。"""

    class MissingFieldClient(FakeClient):
        def chat_json(self, messages, *, schema, name, temperature=None):
            self.calls.append(([dict(m) for m in messages], name, temperature))
            return {}

    project = make_project(1, with_context=True)
    client = MissingFieldClient()
    SlidingWindowStrategy(client=client).translate(project, make_cfg())
    cue = project.cues[0]
    assert cue.translation == cue.text
    assert "translation_failed" in cue.flags


def test_second_retry_lowers_temperature():
    project = make_project(1, with_context=True)
    client = FakeClient(translations="")
    SlidingWindowStrategy(client=client).translate(project, make_cfg(temperature=0.8))
    temps = [t for _, _, t in client.calls]
    assert temps[0] == 0.8 and temps[1] == 0.8
    assert temps[2] == pytest.approx(0.4)  # 第二次重试降 temperature


def test_suspicious_length_triggers_retry():
    """明显复读（译文超长）触发重试，第二次给出正常输出则采用。"""
    project = make_project(1, with_context=True)
    outputs = iter(["复" * 600, "正常译文"])
    client = FakeClient(translations=lambda _: next(outputs))
    SlidingWindowStrategy(client=client).translate(project, make_cfg())
    assert project.cues[0].translation == "正常译文"
    assert "translation_failed" not in project.cues[0].flags
    assert len(client.calls) == 2


def test_translation_field_whitespace_stripped():
    project = make_project(1, with_context=True)
    client = FakeClient(translations="  译文本体 \n")
    SlidingWindowStrategy(client=client).translate(project, make_cfg())
    assert project.cues[0].translation == "译文本体"


def test_translation_system_mentions_json_output():
    system = prompts.translation_system(make_cfg(), [], "摘要")
    assert '"translation"' in system
    schema = prompts.TRANSLATION_RESPONSE_SCHEMA
    assert schema["required"] == ["translation"]


# ------------------------------------------------------------------ prompt 组装


def test_messages_assembly():
    """断言 messages：system 注入摘要+术语表；历史对为 user/assistant；前瞻标注不翻译。"""
    project = make_project(4, with_context=True)
    client = FakeClient(translations=lambda msgs: f"译:{msgs[-1]['content'].splitlines()[1]}")
    cfg = make_cfg(history_count=2, forward_count=1)
    SlidingWindowStrategy(client=client).translate(project, cfg)

    last_messages, _, _ = client.calls[-1]  # 第 4 条 cue 的调用
    system = last_messages[0]["content"]
    assert "已有摘要" in system
    assert "Erebus | 厄瑞玻斯" in system
    assert "简体中文" in system
    # system + 2 历史对（4 条消息）+ 当前 user
    assert [m["role"] for m in last_messages] == [
        "system", "user", "assistant", "user", "assistant", "user",
    ]
    # 历史对是第 2、3 条 cue 的 原文→译文
    assert last_messages[1]["content"] == project.cues[1].text
    assert last_messages[2]["content"] == f"译:{project.cues[1].text}"
    assert last_messages[3]["content"] == project.cues[2].text
    assert last_messages[4]["content"] == f"译:{project.cues[2].text}"
    # 当前 user：待译原文（第 4 条已是末条，无前瞻）
    final_user = last_messages[5]["content"]
    assert project.cues[3].text in final_user
    # 前瞻用第 3 条的调用检查：应含第 4 条原文且标注不翻译
    third_messages, _, _ = client.calls[2]
    third_user = third_messages[-1]["content"]
    assert project.cues[3].text in third_user  # 前瞻：第 4 条原文
    assert "不要翻译" in third_user


def test_additional_prompt_in_system():
    project = make_project(1, with_context=True)
    client = FakeClient()
    SlidingWindowStrategy(client=client).translate(
        project, make_cfg(additional_prompt="保留英文人名注音")
    )
    assert "保留英文人名注音" in client.calls[0][0][0]["content"]


# ------------------------------------------------------------------ 术语后校验


def test_glossary_miss_flag():
    """原文含术语 src 而译文不含 dst → 记 glossary_miss 标记。"""
    project = make_project(1, with_context=True)
    client = FakeClient(translations="完全不含术语的译文")
    SlidingWindowStrategy(client=client).translate(project, make_cfg())
    assert "glossary_miss:Erebus" in project.cues[0].flags


def test_glossary_hit_no_flag():
    project = make_project(1, with_context=True)
    client = FakeClient(translations="厄瑞玻斯出现了")
    SlidingWindowStrategy(client=client).translate(project, make_cfg())
    assert project.cues[0].flags == []


# ------------------------------------------------------------------ 摘要


def test_summary_from_json_field():
    project = make_project(2)
    client = FakeClient(summary="一段英文演讲。")
    SlidingWindowStrategy(client=client).translate(project, make_cfg(), auto_confirm=True)
    assert project.meta.summary == "一段英文演讲。"


def test_summary_missing_field_raises():
    """摘要 payload 缺 summary 字段 → InvalidModelJsonError。"""
    project = make_project(1)

    class BadSummaryClient(FakeClient):
        def chat_json(self, messages, *, schema, name, temperature=None):
            self.calls.append(([dict(m) for m in messages], name, temperature))
            return {}

    with pytest.raises(InvalidModelJsonError):
        SlidingWindowStrategy(client=BadSummaryClient()).translate(
            project, make_cfg(), auto_confirm=True
        )


def test_summary_chunked_map_reduce(monkeypatch):
    """超长全文分块摘要后再合并。"""
    monkeypatch.setattr(context, "SUMMARY_MAX_CHARS", 30)
    project = make_project(6, with_context=False)
    project.cues = [
        Cue(id=i + 1, start=0.0, end=1.0, text="x" * 20) for i in range(6)
    ]
    client = FakeClient()
    SlidingWindowStrategy(client=client).translate(
        project, make_cfg(), auto_confirm=True
    )
    chunk_calls = [
        c for c in client.calls if "部分内容" in c[0][0]["content"]
    ]
    merge_calls = [
        c for c in client.calls if "合并为一份全片摘要" in c[0][0]["content"]
    ]
    assert len(chunk_calls) >= 2
    assert len(merge_calls) == 1
    assert "分段摘要 1" in merge_calls[0][0][1]["content"]
    assert all(c[1] == "summary" for c in chunk_calls + merge_calls)


# ------------------------------------------------------------------ HTTP 重试


class _FakeHttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _make_client(sleeps: list[float]) -> ChatClient:
    cfg = make_cfg()
    return ChatClient(cfg, sleep=sleeps.append)


def test_429_backoff_retry():
    sleeps: list[float] = []
    client = _make_client(sleeps)
    attempts = iter([_FakeHttpError(429), _FakeHttpError(500), "成功"])

    def fake_create(messages, temperature, response_format=None):
        outcome = next(attempts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client._create = fake_create
    assert client.chat([{"role": "user", "content": "hi"}]) == "成功"
    assert sleeps == [1.0, 2.0]  # 指数退避


def test_retry_exhaustion_raises_chat_error():
    sleeps: list[float] = []
    client = _make_client(sleeps)
    client.cfg.max_retries = 2

    def always_fail(messages, temperature, response_format=None):
        raise _FakeHttpError(429)

    client._create = always_fail
    with pytest.raises(ChatError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(sleeps) == 2  # 3 次尝试之间退避 2 次


def test_non_retryable_error_raises_immediately():
    sleeps: list[float] = []
    client = _make_client(sleeps)

    def bad_request(messages, temperature, response_format=None):
        raise _FakeHttpError(400)

    client._create = bad_request
    with pytest.raises(ChatError):
        client.chat([{"role": "user", "content": "hi"}])
    assert sleeps == []  # 400 不重试


def test_timeout_is_retryable():
    from openai import APITimeoutError
    import httpx

    sleeps: list[float] = []
    client = _make_client(sleeps)
    request = httpx.Request("POST", "http://fake/v1/chat/completions")
    attempts = iter([APITimeoutError(request), "成功"])

    def flaky(messages, temperature, response_format=None):
        outcome = next(attempts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client._create = flaky
    assert client.chat([{"role": "user", "content": "hi"}]) == "成功"
    assert sleeps == [1.0]


# ------------------------------------------------------------------ chat_json 结构化输出


def test_chat_json_attaches_response_format_and_parses():
    """chat_json 把 json_schema response_format 传给 SDK，返回值解析为 dict。"""
    client = _make_client([])
    captured: list[dict] = []

    def fake_create(messages, temperature, response_format=None):
        captured.append(response_format)
        return '{"translation": "你好"}'

    client._create = fake_create
    data = client.chat_json(
        [{"role": "user", "content": "hi"}],
        schema=prompts.TRANSLATION_RESPONSE_SCHEMA,
        name="translation",
    )
    assert data == {"translation": "你好"}
    assert captured[0] == {
        "type": "json_schema",
        "json_schema": {
            "name": "translation",
            "schema": prompts.TRANSLATION_RESPONSE_SCHEMA,
        },
    }


def test_chat_json_invalid_json_raises():
    client = _make_client([])
    client._create = lambda messages, temperature, response_format=None: "不是 JSON"
    with pytest.raises(InvalidModelJsonError):
        client.chat_json([], schema={}, name="translation")


def test_chat_json_non_object_json_raises():
    client = _make_client([])
    client._create = lambda messages, temperature, response_format=None: '["数组"]'
    with pytest.raises(InvalidModelJsonError):
        client.chat_json([], schema={}, name="translation")


def test_chat_400_with_response_format_hints_json_schema():
    """带 response_format 的 400 错误提示端点需支持 json_schema。"""
    client = _make_client([])

    def bad_request(messages, temperature, response_format=None):
        raise _FakeHttpError(400)

    client._create = bad_request
    with pytest.raises(ChatError, match="response_format json_schema"):
        client.chat([], response_format={"type": "json_schema", "json_schema": {}})


def test_chat_400_without_response_format_has_no_hint():
    client = _make_client([])

    def bad_request(messages, temperature, response_format=None):
        raise _FakeHttpError(400)

    client._create = bad_request
    with pytest.raises(ChatError) as exc_info:
        client.chat([])
    assert "response_format" not in str(exc_info.value)
