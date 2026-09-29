"""Claude Code headless adapter (decision U11): isolation and subscription-only flags, result parsing, real model
check, failure mapping, KG extraction through it, and the login check. A fake runner stands in for `claude -p`."""
import json
import subprocess
from pathlib import Path

import pytest

from labgene.config import Limits, RoleModel
from labgene.contracts import ProviderStatus
from labgene.costs import CallContext
from labgene.knowledge.gate import guarded_generate
from labgene.knowledge.kg import LLMKGExtractor, Ontology
from labgene.providers import claude_cli
from labgene.providers.base import GenerationRequest, ModelChangedError, ToolSpec
from labgene.providers.call import build_llm, call_llm
from labgene.providers.claude_cli import ClaudeCodeProvider, child_env

FIX = Path(__file__).resolve().parents[1] / "fixtures"
REAL_CLI_PROBLEM = claude_cli.cli_problem          # captured before the autouse stub replaces it
SCHEMA = {"type": "object", "required": ["relations"], "properties": {"relations": {"type": "array", "items": {
    "type": "object", "properties": {"subject": {"type": "string"}, "conditions": {"type": "object"}}}}}}
REL = {"subject": "temperature", "predicate": "increases", "object": "rate constant", "conditions": {"regime": "low"}}
OK = {"type": "result", "subtype": "success", "is_error": False, "result": "prose", "session_id": "sess-1",
      "structured_output": {"relations": [REL]}, "total_cost_usd": 0.12,
      "usage": {"input_tokens": 1200, "cache_read_input_tokens": 300, "cache_creation_input_tokens": 50,
                "output_tokens": 900},
      "modelUsage": {"claude-opus-5-5": {"inputTokens": 1200, "outputTokens": 900}}}


def fake(result=OK, rc=0, raise_=None, seen=None, stdout=None):
    def run(args, stdin, cwd, timeout):
        if seen is not None:
            seen.update(args=args, stdin=stdin, cwd_empty=not any(cwd.iterdir()))
        if raise_ is not None:
            raise raise_
        return rc, stdout if stdout is not None else "warning: something\n" + json.dumps(result), ""
    return run


def req(**kw):
    base = dict(role="kg_extractor", model="claude-opus-5-5", system_instruction="KG POLICY v1", reasoning_effort="max",
                input=[{"role": "user", "text": '{"passage": "p"}'}], response_schema=SCHEMA, max_output_tokens=4096)
    return GenerationRequest(**{**base, **kw})


def test_u11_command_is_headless_isolated_and_uses_our_system_prompt():
    seen = {}
    res = ClaudeCodeProvider(runner=fake(seen=seen)).generate(req())
    a = seen["args"]
    assert res.status is ProviderStatus.ok and a[:2] == ["claude", "-p"]
    for flag in ("--safe-mode", "--restricted", "--strict-mcp-config", "--disable-slash-commands",
                 "--no-session-persistence"):
        assert flag in a
    assert "--bare" not in a                                  # bare mode never reads the subscription login
    assert a[a.index("--tools") + 1] == "" and a[a.index("--output-format") + 1] == "json"
    assert a[a.index("--model") + 1] == "claude-opus-5-5" and a[a.index("--effort") + 1] == "max"
    assert a[a.index("--system-prompt") + 1] == "KG POLICY v1"
    assert json.loads(a[a.index("--json-schema") + 1]) == SCHEMA
    assert seen["stdin"] == '{"passage": "p"}' and seen["cwd_empty"]


def test_u11_child_environment_has_no_api_key_and_no_parent_session(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDECODE", "CLAUDE_CODE_SESSION_ID",
              "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_USE_BEDROCK"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "subscription-token")
    env = child_env()
    assert not any(k.upper().startswith(("ANTHROPIC_", "CLAUDECODE", "CLAUDE_CODE_USE", "CLAUDE_CODE_SESSION",
                                         "CLAUDE_CODE_MESSAGING")) for k in env)
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "subscription-token" and "PATH" in {k.upper() for k in env}
    assert env["DISABLE_AUTOUPDATER"] == "1"                 # flags were checked on one version: no self-update


def test_structured_output_usage_ids_and_the_model_that_ran_are_parsed():
    p = ClaudeCodeProvider(runner=fake())
    res = p.generate(req())
    assert json.loads(res.text) == {"relations": [REL]} and res.request_id == "sess-1" and res.interaction_id is None
    assert res.model_returned == "claude-opus-5-5" and p.echoes_model is True and p.billing == "subscription"
    assert (res.usage.input_tokens, res.usage.cached_tokens, res.usage.output_tokens) == (1550, 300, 900)
    assert p.generate(req(response_schema=None)).text == "prose"
    assert res.error is None                                  # only the requested model ran: nothing to note
    helper = {**OK, "modelUsage": {"claude-haiku-4-5": {"outputTokens": 12}, "claude-opus-5-5": {"outputTokens": 900}}}
    res = ClaudeCodeProvider(runner=fake(helper)).generate(req())
    assert res.model_returned == "claude-opus-5-5" and res.status is ProviderStatus.ok
    assert json.loads(res.error.removeprefix("modelUsage ")) == {                  # every model that ran is recorded
        "claude-haiku-4-5": {"inputTokens": None, "outputTokens": 12},
        "claude-opus-5-5": {"inputTokens": None, "outputTokens": 900}}


@pytest.mark.parametrize("model_usage,match", [
    ({"claude-sonnet-5": {}}, "claude-sonnet-5"), ({}, "names no model"),
    ({"claude-sonnet-5": {"outputTokens": 30000}, "claude-opus-5-5": {"outputTokens": 3}}, "claude-sonnet-5")])
def test_u11_another_or_unnamed_model_stops_the_run(model_usage, match):
    p = ClaudeCodeProvider(runner=fake({**OK, "modelUsage": model_usage}))
    with pytest.raises(ModelChangedError, match=match):
        call_llm(p, req(), CallContext(sink=lambda e: None), Limits(provider_backoff_s=0.0))
    with pytest.raises(ModelChangedError):
        guarded_generate(p, req(), CallContext(sink=lambda e: None))


def test_u17_a_safety_fallback_is_a_refusal_and_the_other_models_answer_is_discarded():
    # live canary 2026-09-28: the requested model ran with 0 output tokens, then claude-opus-5 answered
    fb = {**OK, "result": "answer from the fallback model", "structured_output": {"relations": [REL]},
          "modelUsage": {"claude-opus-5-5": {"inputTokens": 2, "outputTokens": 0},
                         "claude-opus-5": {"inputTokens": 678, "outputTokens": 24729}}}
    res = ClaudeCodeProvider(runner=fake(fb)).generate(req())
    assert res.status is ProviderStatus.refusal and res.text is None and res.model_returned == "claude-opus-5-5"
    assert "safety fallback" in res.error and "claude-opus-5" in res.error
    assert call_llm(ClaudeCodeProvider(runner=fake(fb)), req(), CallContext(sink=lambda e: None),
                    Limits(provider_backoff_s=0.0)).status is ProviderStatus.refusal       # not retried, not a stop
    events = []
    ext = LLMKGExtractor(ClaudeCodeProvider(runner=fake(fb)), "claude-opus-5-5")
    assert ext.extract("text", Ontology.load(FIX / "ontology" / "profile.yaml"), CallContext(sink=events.append)) == []
    assert [e.status for e in events] == ["refusal", "refusal"]                  # the call, then the skipped chunk


@pytest.mark.parametrize("runner,status,needle", [
    (fake({**OK, "is_error": True, "subtype": "error_during_execution",
           "result": "Claude AI usage limit reached|1790000000"}, rc=1), ProviderStatus.infra_error, "usage/rate limit"),
    (fake(stdout="not json at all"), ProviderStatus.infra_error, "exit 0"),
    (fake(raise_=subprocess.TimeoutExpired(cmd="claude", timeout=1)), ProviderStatus.infra_error, "timed out"),
    (fake(raise_=FileNotFoundError("claude")), ProviderStatus.infra_error, "could not start"),
    (fake({**OK, "structured_output": None, "result": ""}), ProviderStatus.incomplete, "no structured output"),
    # a schema request answered in prose is never parsed silently (the KG extractor would drop every relation)
    (fake({**OK, "structured_output": None}), ProviderStatus.incomplete, "no structured output"),
])
def test_b19_claude_failures_are_never_output(runner, status, needle):
    res = ClaudeCodeProvider(runner=runner).generate(req())
    assert res.status is status and needle in res.error and res.text is None


def test_claude_code_serves_single_shot_roles_only():
    p = ClaudeCodeProvider(runner=fake())
    with pytest.raises(ValueError, match="tools"):
        p.generate(req(tools=[ToolSpec(name="t", description="d", parameters={"type": "object"})]))
    with pytest.raises(ValueError, match="user turns"):
        p.generate(req(input=[{"role": "model", "text": "hi"}]))


def test_u11_kg_extraction_runs_through_claude_code_with_subscription_billing():
    events = []
    assert isinstance(build_llm(RoleModel(provider="claude_code", model="claude-opus-5-5", endpoint="claude_cli")),
                      ClaudeCodeProvider)
    ext = LLMKGExtractor(ClaudeCodeProvider(runner=fake()), "claude-opus-5-5")
    rels = ext.extract("Raising the temperature increases the rate constant.",
                       Ontology.load(FIX / "ontology" / "profile.yaml"), CallContext(sink=events.append))
    [ev] = events
    assert rels == [REL] and ev.provider == "claude_code" and ev.detail["billing"] == "subscription"
    assert "sdk_version" in ev.detail                         # guarded calls record the CLI version too
    assert ev.detail["model_verified"] is True and ev.detail["model_returned"] == "claude-opus-5-5"


@pytest.mark.parametrize("status,ok", [({"loggedIn": True, "authMethod": "claude.ai"}, True),
                                       ({"loggedIn": True, "authMethod": "api_key"}, False),
                                       ({"loggedIn": False}, False)])
def test_login_check_requires_the_subscription_login_and_never_echoes_the_account(monkeypatch, status, ok):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda b: "claude")
    monkeypatch.setattr(claude_cli, "_run", lambda args, stdin, cwd, t: (
        0, json.dumps({**status, "email": "someone@example.com", "orgName": "Org"}), ""))
    problem = REAL_CLI_PROBLEM()
    assert (problem is None) is ok
    assert "example.com" not in (problem or "") and "Org" not in (problem or "")
