"""T06 provider behaviour at evaluation-changing boundaries (B18, B19, B22) via fake transports. No network."""
import json
import urllib.error
from pathlib import Path

import pytest

from labgene.config import CostCaps, Limits, RoleModel, load_profile
from labgene.contracts import ProviderStatus
from labgene.costs import CallContext, CapExceeded, CostGuard
from labgene.providers.anthropic import AnthropicMessagesProvider
from labgene.providers.base import GenerationRequest, ModelChangedError, ToolSpec
from labgene.providers.call import build_embedder, build_llm, call_llm, carried_tokens, estimate_input_tokens
from labgene.providers.fixture import FixtureHashEmbedder, FixtureProvider
from labgene.providers.gemini import GeminiEmbedder, GeminiInteractionsProvider
from labgene.providers.openai import OpenAIResponsesProvider

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gemini-3.1-pro-preview"
KEY = "sk-TEST-SECRET-KEY-123"
TOOL = ToolSpec(name="describe", description="stats", parameters={"type": "object", "properties": {}})


class FakeTransport:
    """Scripted replies: (status, json_or_bytes[, headers]) or an exception instance to raise."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append({"method": method, "url": url, "headers": headers, "body": json.loads(body)})
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, BaseException):
            raise r
        status, data, *hdrs = r
        return status, (hdrs[0] if hdrs else {}), data if isinstance(data, bytes) else json.dumps(data).encode()


def req(**kw):
    base = dict(role="researcher_planner", model=MODEL, system_instruction="SYS", input=[{"role": "user", "text": "hi"}],
                tools=[TOOL], thinking_level="high", max_output_tokens=100)
    return GenerationRequest(**{**base, **kw})


def interaction(status="completed", text="ok", model=MODEL, **extra):
    return {"id": "int-1", "model": model, "status": status,
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}], **extra}


def sink():
    events = []
    return events, CallContext(sink=events.append)


@pytest.fixture
def keys(monkeypatch):
    for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(k, KEY)


# ---------------------------------------------------------------- B18: explicit config, role/continuation separation

def test_b18_gemini_every_request_states_system_tools_generation_config(keys):
    t = FakeTransport((200, interaction()))
    p = GeminiInteractionsProvider(transport=t)
    p.generate(req())
    p.generate(req(tools=[], thinking_level=None, previous_interaction_id="int-0",
                   response_schema={"type": "object", "properties": {"action": {"type": "string"}}}))
    first, second = (c["body"] for c in t.calls)
    for body in (first, second):
        assert {"model", "input", "system_instruction", "tools", "generation_config", "store"} <= set(body)
        assert body["system_instruction"] == "SYS" and body["model"] == MODEL
        assert all(tool["type"] == "function" for tool in body["tools"])   # never google_search / url_context
        assert KEY not in json.dumps(body)
    assert first["generation_config"] == {"thinking_level": "high", "max_output_tokens": 100}
    assert "previous_interaction_id" not in first and first["store"] is False
    assert first["input"] == [{"type": "user_input", "content": "hi"}]
    assert second["tools"] == [] and second["previous_interaction_id"] == "int-0"
    assert second["response_format"] == {"type": "text", "mime_type": "application/json",
                                         "schema": {"type": "object", "properties": {"action": {"type": "string"}}}}
    assert t.calls[0]["url"].endswith("/v1beta/interactions") and KEY not in t.calls[0]["url"]
    assert t.calls[0]["headers"]["x-goog-api-key"] == KEY


def test_b18_gemini_parses_steps_function_calls_usage_ids_and_model(keys):
    data = {"id": "int-9", "model": f"models/{MODEL}", "status": "requires_action", "steps": [
        {"type": "thought", "signature": "opaque"},
        {"type": "model_output", "content": [{"type": "text", "text": "Checking."}]},
        {"type": "function_call", "id": "c1", "name": "describe", "arguments": {"metrics": ["yield"]}}],
        "usage": {"total_input_tokens": 7, "total_output_tokens": 20, "total_thought_tokens": 22, "total_cached_tokens": 0}}
    r = GeminiInteractionsProvider(transport=FakeTransport((200, data, {"X-Request-Id": "rid-1"}))).generate(req())
    assert r.status is ProviderStatus.ok and r.text == "Checking."
    assert [(f.call_id, f.name, f.arguments) for f in r.function_calls] == [("c1", "describe", {"metrics": ["yield"]})]
    assert (r.interaction_id, r.model_returned, r.request_id) == ("int-9", MODEL, "rid-1")
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.reasoning_tokens, r.usage.cached_tokens) == (7, 20, 22, 0)
    assert r.latency_s >= 0 and r.endpoint == "interactions"


def test_b22_gemini_missing_usage_is_none_not_zero(keys):
    r = GeminiInteractionsProvider(transport=FakeTransport((200, interaction()))).generate(req())
    assert r.usage.input_tokens is None and r.usage.output_tokens is None and r.usage.reasoning_tokens is None


def test_b18_gemini_stateless_tool_turns_map_to_function_steps(keys):
    t = FakeTransport((200, interaction()))
    GeminiInteractionsProvider(transport=t).generate(req(input=[
        {"role": "user", "text": "hi"},
        {"role": "model", "text": None, "function_calls": [{"call_id": "c1", "name": "describe", "arguments": {}}]},
        {"role": "tool", "call_id": "c1", "name": "describe", "result": {"n": 1}}]))
    steps = t.calls[0]["body"]["input"]
    assert steps[1] == {"type": "function_call", "id": "c1", "name": "describe", "arguments": {}}
    assert steps[2] == {"type": "function_result", "name": "describe", "call_id": "c1",
                        "result": [{"type": "text", "text": '{"n": 1}'}]}


def test_b18_returned_model_mismatch_raises_after_costing_the_attempt(keys):
    events, ctx = sink()
    p = GeminiInteractionsProvider(transport=FakeTransport((200, interaction(model="gemini-2.5-flash"))))
    with pytest.raises(ModelChangedError):
        call_llm(p, req(), ctx, Limits(), allowed_models=[], sleep=lambda s: None)
    assert len(events) == 1 and events[0].detail["model_returned"] == "gemini-2.5-flash"
    alias = GeminiInteractionsProvider(transport=FakeTransport((200, interaction(model=f"{MODEL}-001"))))
    assert call_llm(alias, req(), ctx, Limits(), allowed_models=[f"{MODEL}-001"]).status is ProviderStatus.ok


def test_b18_finding3_generated_reply_without_a_model_fails_closed(keys):
    events, ctx = sink()
    for body in ({k: v for k, v in interaction().items() if k != "model"}, interaction(model=None),
                 {k: v for k, v in interaction(status="incomplete").items() if k != "model"}):
        with pytest.raises(ModelChangedError, match="unverifiable"):
            call_llm(GeminiInteractionsProvider(transport=FakeTransport((200, body))), req(), ctx, Limits())
    # error replies carry no model and are not model changes
    for reply in ((429, {}), (400, {"error": {"message": "blocked: SAFETY"}})):
        r = call_llm(GeminiInteractionsProvider(transport=FakeTransport(reply)), req(), ctx,
                     Limits(provider_max_attempts=1))
        assert r.status in (ProviderStatus.infra_error, ProviderStatus.refusal) and r.model_returned is None


# ---------------------------------------------------------------- B19: status distinctions, finite retries, no fallback

@pytest.mark.parametrize("reply,status,err", [
    ((429, {"error": {"message": "Resource exhausted"}}), ProviderStatus.infra_error, "http 429"),
    ((503, {"error": {"message": "unavailable"}}), ProviderStatus.infra_error, "http 503"),
    (TimeoutError("timed out"), ProviderStatus.infra_error, "network: TimeoutError"),
    (urllib.error.URLError("dns"), ProviderStatus.infra_error, "network: URLError"),
    ((200, b"<html>gateway</html>"), ProviderStatus.infra_error, "non-JSON"),
    ((200, interaction(status="failed", error={"message": "internal"})), ProviderStatus.infra_error, "failed"),
    ((200, interaction(status="incomplete", text='{"action":"run_exp')), ProviderStatus.incomplete, None),
    ((200, {**interaction(), "steps": [{"type": "model_output", "content": [], "finish_reason": "SAFETY"}]}),
     ProviderStatus.refusal, None),
    ((400, {"error": {"message": "Request blocked: PROHIBITED_CONTENT"}}), ProviderStatus.refusal, None),
])
def test_b19_gemini_distinguishes_incomplete_refusal_429_timeout(keys, reply, status, err):
    r = GeminiInteractionsProvider(transport=FakeTransport(reply)).generate(req())
    assert r.status is status
    if err:
        assert err in r.error
    if status is ProviderStatus.incomplete:
        assert r.text == '{"action":"run_exp'          # truncated text returned as-is, never repaired
    assert KEY not in r.model_dump_json()


def test_b19_infra_retried_finitely_with_every_attempt_costed_and_no_fallback(keys):
    events, ctx = sink()
    sleeps = []
    t = FakeTransport(TimeoutError("slow"))
    r = call_llm(GeminiInteractionsProvider(transport=t), req(), ctx, Limits(provider_max_attempts=3, provider_backoff_s=2.0),
                 sleep=sleeps.append)
    assert r.status is ProviderStatus.infra_error and r.attempt == 3 and r.provider == "gemini"
    assert len(t.calls) == 3 and all(c["body"]["model"] == MODEL for c in t.calls)   # same model, same endpoint
    assert [(e.kind, e.attempt, e.status, e.provider, e.model) for e in events] == \
        [("llm_call", i, "infra_error", "gemini", MODEL) for i in (1, 2, 3)]
    assert sleeps == [2.0, 4.0]


def test_b19_retry_recovers_but_incomplete_and_refusal_are_never_retried():
    events, ctx = sink()
    p = FixtureProvider(script=["infra_error", "ok"])
    r = call_llm(p, req(), ctx, Limits(), sleep=lambda s: None)
    assert r.status is ProviderStatus.ok and r.attempt == 2 and len(events) == 2
    for s in ("incomplete", "refusal"):
        p = FixtureProvider(script=[s, "ok"])
        r = call_llm(p, req(), ctx, Limits(), sleep=lambda s: None)
        assert r.status is ProviderStatus(s) and len(p.requests) == 1


# ---------------------------------------------------------------- B22: cost guard reservation

def test_b22_guard_reserves_worst_case_and_blocks_before_the_call():
    guard = CostGuard(CostCaps(max_calls=1, max_output_tokens=1000))
    events = []
    ctx = CallContext(sink=events.append, guard=guard)
    p = FixtureProvider()
    call_llm(p, req(max_output_tokens=300), ctx, Limits())
    assert guard.calls == 1 and guard.output_tokens == 300     # usage unknown -> reservation kept, never 0
    with pytest.raises(CapExceeded):
        call_llm(p, req(max_output_tokens=300), ctx, Limits())
    assert len(p.requests) == 1 and len(events) == 1
    with pytest.raises(CapExceeded):                            # unbounded output cannot be reserved under a cap
        call_llm(p, req(max_output_tokens=None), CallContext(sink=events.append, guard=CostGuard(CostCaps(max_usd=1.0))),
                 Limits())


def test_b22_finding1_continuation_reserves_the_carried_history_before_the_call(keys):
    t = FakeTransport((200, {**interaction(), "usage": {"total_input_tokens": 3000, "total_output_tokens": 200,
                                                        "total_thought_tokens": 100}}))
    p = GeminiInteractionsProvider(transport=t)
    first_req, cont = req(store=True), req(store=True, input=[{"role": "user", "text": "more"}], previous_interaction_id="int-1")
    first = call_llm(p, first_req, sink()[1], Limits())
    carried = carried_tokens(first_req, first)
    assert carried >= 3000 + 200 + 100                   # the continuation re-processes all of it
    cap = 3000 + estimate_input_tokens(cont) + carried - 1
    for context in (None, carried):                      # unknown history, or history that does not fit
        guard = CostGuard(CostCaps(max_input_tokens=cap))
        guard.input_tokens = 3000                        # the first call, settled
        with pytest.raises(CapExceeded):
            call_llm(p, cont, CallContext(sink=lambda e: None, guard=guard), Limits(), context_tokens=context)
    assert len(t.calls) == 1 and guard.input_tokens == 3000     # blocked BEFORE the continuation was sent


def test_b22_finding1_token_estimate_errs_high_for_digits_and_cjk():
    digits, korean = "1234567890" * 64, "온도와 시간을 바꿔 수율을 측정한다" * 32   # tokenizers: ~1 token per digit / syllable
    assert estimate_input_tokens(req(system_instruction=digits, tools=[], input=[])) >= len(digits)
    assert estimate_input_tokens(req(system_instruction=korean, tools=[], input=[])) >= len(korean.replace(" ", ""))


# ---------------------------------------------------------------- credentials

def test_api_key_read_at_call_time_and_never_logged(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    events, ctx = sink()
    t = FakeTransport((200, interaction()))
    p = GeminiInteractionsProvider(transport=t)
    r = call_llm(p, req(), ctx, Limits(provider_max_attempts=1))
    assert r.status is ProviderStatus.infra_error and "GEMINI_API_KEY is not set" in r.error and not t.calls
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    call_llm(p, req(), ctx, Limits())
    assert t.calls[0]["headers"]["x-goog-api-key"] == KEY
    assert all(KEY not in e.model_dump_json() for e in events)


# ---------------------------------------------------------------- internal-role adapters

def test_openai_responses_request_shape_and_parsing(keys):
    data = {"id": "resp-1", "model": "gpt-6-astra", "status": "completed", "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
        {"type": "function_call", "call_id": "c1", "name": "extract", "arguments": '{"a": 1}'}],
        "usage": {"input_tokens": 10, "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 30},
                  "input_tokens_details": {"cached_tokens": 4}}}
    t = FakeTransport((200, data, {"x-request-id": "req-o"}))
    r = OpenAIResponsesProvider(transport=t).generate(req(role="kg_extractor", model="gpt-6-astra", thinking_level=None,
                                                          reasoning_effort="high", response_schema={"type": "object"},
                                                          previous_interaction_id="resp-0"))
    body = t.calls[0]["body"]
    assert t.calls[0]["url"] == "https://api.openai.com/v1/responses"
    assert t.calls[0]["headers"]["authorization"] == f"Bearer {KEY}"
    assert body["instructions"] == "SYS" and body["input"] == [{"role": "user", "content": "hi"}]
    assert body["reasoning"] == {"effort": "high"} and body["max_output_tokens"] == 100
    assert body["tools"][0]["type"] == "function" and body["previous_response_id"] == "resp-0"
    assert body["text"]["format"]["type"] == "json_schema" and "thinking_level" not in json.dumps(body)
    assert r.status is ProviderStatus.ok and r.text == "done" and r.function_calls[0].arguments == {"a": 1}
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.reasoning_tokens, r.usage.cached_tokens) == (10, 20, 30, 4)
    assert (r.request_id, r.interaction_id, r.model_returned) == ("req-o", "resp-1", "gpt-6-astra")
    for d, s in [({**data, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}, "incomplete"),
                 ({**data, "status": "incomplete", "incomplete_details": {"reason": "content_filter"}}, "refusal"),
                 ({**data, "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}, "refusal")]:
        assert OpenAIResponsesProvider(transport=FakeTransport((200, d))).generate(req()).status is ProviderStatus(s)


def test_anthropic_messages_request_shape_and_parsing(keys):
    data = {"id": "msg_1", "model": "claude-fable-5-1", "stop_reason": "tool_use", "content": [
        {"type": "thinking", "thinking": ""}, {"type": "text", "text": "ok"},
        {"type": "tool_use", "id": "tu1", "name": "check", "input": {"x": 1}}],
        "usage": {"input_tokens": 5, "output_tokens": 9, "cache_read_input_tokens": 2}}
    t = FakeTransport((200, data, {"request-id": "req-a"}))
    r = AnthropicMessagesProvider(transport=t).generate(req(
        role="leakage_gate", model="claude-fable-5-1", thinking_level=None, reasoning_effort="high",
        response_schema={"type": "object"},
        input=[{"role": "user", "text": "q"},
               {"role": "model", "text": None, "function_calls": [{"call_id": "tu0", "name": "check", "arguments": {}}]},
               {"role": "tool", "call_id": "tu0", "name": "check", "result": {"ok": True}}]))
    h, body = t.calls[0]["headers"], t.calls[0]["body"]
    assert h["x-api-key"] == KEY and h["anthropic-version"] == "2023-06-01"
    assert body["system"] == "SYS" and body["max_tokens"] == 100 and "tool_choice" not in body
    assert body["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][2]["content"][0] == {"type": "tool_result", "tool_use_id": "tu0", "content": '{"ok": true}'}
    assert body["tools"][0]["input_schema"] == TOOL.parameters
    assert r.status is ProviderStatus.ok and r.text == "ok" and r.function_calls[0].arguments == {"x": 1}
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.cached_tokens, r.request_id) == (5, 9, 2, "req-a")
    for stop, s in [("max_tokens", "incomplete"), ("refusal", "refusal"), ("end_turn", "ok")]:
        assert AnthropicMessagesProvider(transport=FakeTransport((200, {**data, "stop_reason": stop}))) \
            .generate(req()).status is ProviderStatus(s)
    assert AnthropicMessagesProvider(transport=FakeTransport((529, {}))).generate(req()).status is ProviderStatus.infra_error


# ---------------------------------------------------------------- embeddings

def test_gemini_embedder_request_shape_dimensions_and_no_task_type(keys):
    vec = [0.5] * 3072
    t = FakeTransport((200, {"embeddings": [{"values": vec}, {"values": vec}]}))
    e = GeminiEmbedder(transport=t)
    out = e.embed(["what is x?", "y"], "query")
    assert out.status == "ok" and out.dimensions == 3072 and len(out.vectors) == 2 and e.development_only is False
    call = t.calls[0]
    assert call["url"].endswith("/v1beta/models/gemini-embedding-2:batchEmbedContents")
    r0 = call["body"]["requests"][0]
    assert r0["model"] == "models/gemini-embedding-2" and r0["output_dimensionality"] == 3072
    assert r0["content"]["parts"][0]["text"] == "task: search result | query: what is x?"
    assert "task_type" not in json.dumps(call["body"]) and "taskType" not in json.dumps(call["body"])
    GeminiEmbedder(transport=t).embed(["doc"], "document")
    assert t.calls[1]["body"]["requests"][0]["content"]["parts"][0]["text"] == "title: none | text: doc"
    assert GeminiEmbedder(transport=FakeTransport((429, {}))).embed(["x"], "query").status == "infra_error"


def test_b22_finding4_embedder_is_one_physical_request_and_a_billed_mismatch_is_still_recorded(keys):
    from labgene.knowledge.retrieval import embed
    with pytest.raises(ValueError):                      # the caller batches; nothing is sent
        GeminiEmbedder(transport=FakeTransport((200, {}))).embed(["t"] * 101, "document")
    t = FakeTransport((200, {"embeddings": [{"values": [0.1] * 768}]}, {"x-request-id": "rid-7"}))
    e = GeminiEmbedder(transport=t)
    r = e.embed(["x"], "query")
    assert len(t.calls) == 1 and (r.status, r.dimensions, r.request_id) == ("ok", 768, "rid-7")   # honest, not raised
    events = []
    with pytest.raises(ValueError):                      # never silently mix dimensions into the pinned index...
        embed(e, ["x"], "query", CallContext(sink=events.append))
    assert [(ev.kind, ev.request_id) for ev in events] == [("embedding", "rid-7")]   # ...but the billed call is costed
    mixed = FakeTransport((200, {"embeddings": [{"values": [0.1] * 3072}, {"values": [0.1] * 4}]}, {"x-request-id": "rid-8"}))
    r = GeminiEmbedder(transport=mixed).embed(["a", "b"], "document")
    assert (r.status, r.vectors, r.request_id) == ("infra_error", [], "rid-8") and "malformed" in r.error


def test_fixture_embedder_is_deterministic_and_development_only():
    e = FixtureHashEmbedder(dimensions=64)
    a, b = e.embed(["yield ridge temperature"], "document"), e.embed(["yield ridge temperature"], "query")
    assert e.development_only is True and a.vectors == b.vectors and len(a.vectors[0]) == 64
    assert abs(sum(x * x for x in a.vectors[0]) - 1.0) < 1e-9


# ---------------------------------------------------------------- factory

def test_factory_builds_configured_adapters_and_rejects_mismatched_endpoints():
    offline = load_profile(ROOT / "configs/offline.yaml")
    assert isinstance(build_llm(offline.roles.researcher), FixtureProvider)
    assert build_embedder(offline.roles.embedder).development_only is True
    live = load_profile(ROOT / "configs/live.example.yaml")
    from labgene.providers.claude_cli import ClaudeCodeProvider
    from labgene.providers.codex_cli import CodexExecProvider
    assert isinstance(build_llm(live.roles.researcher), CodexExecProvider)            # U10: researcher + advisors
    assert isinstance(build_llm(live.roles.advisor_product), CodexExecProvider)
    assert isinstance(build_llm(live.roles.leakage_gate), CodexExecProvider)          # U8: gate on Codex
    assert isinstance(build_llm(live.roles.kg_extractor), ClaudeCodeProvider)         # U11: internal roles
    assert isinstance(build_llm(RoleModel(provider="gemini", model=MODEL, endpoint="interactions")),
                      GeminiInteractionsProvider)
    assert isinstance(build_llm(RoleModel(provider="openai", model="gpt-6-astra", endpoint="responses")),
                      OpenAIResponsesProvider)
    assert isinstance(build_llm(RoleModel(provider="anthropic", model="claude-fable-5-1", endpoint="messages")),
                      AnthropicMessagesProvider)
    assert build_embedder(live.roles.embedder).dimensions == 3072
    with pytest.raises(ValueError):
        build_llm(RoleModel(provider="gemini", model=MODEL, endpoint="responses"))
    with pytest.raises(ValueError):
        build_embedder(RoleModel(provider="openai", model="gpt-6-astra", endpoint="responses"))
