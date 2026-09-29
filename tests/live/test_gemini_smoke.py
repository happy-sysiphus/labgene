"""Live Gemini smoke (T06-live): structured output, function-call round trip, role separation, usage capture.

PAID. Skipped unless LABGENE_LIVE=1 AND LABGENE_LIVE_PROFILE=<approved live profile> whose preflight passes
(finite cost caps + unit prices). Every call goes through call_llm with the profile's CostGuard.
Run: LABGENE_LIVE=1 LABGENE_LIVE_PROFILE=configs/live-gemini-smoke.yaml .venv/Scripts/python -m pytest tests/live -q
"""
import json
import os
from pathlib import Path

import pytest

from labgene.config import load_dotenv, load_profile, load_public_task, preflight_problems
from labgene.contracts import HistoryItem, ProviderStatus, ResearcherView
from labgene.costs import CallContext, CostGuard
from labgene.providers.base import GenerationRequest
from labgene.providers.call import build_llm, call_llm, carried_tokens, estimate_input_tokens
from labgene.researcher.agent import PlannerReviewerResearcher, action_schema
from labgene.researcher.analysis import AnalysisTools

pytestmark = pytest.mark.live
if os.environ.get("LABGENE_LIVE") != "1" or not os.environ.get("LABGENE_LIVE_PROFILE"):
    pytest.skip("live smoke disabled: set LABGENE_LIVE=1 and LABGENE_LIVE_PROFILE", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
TASK = load_public_task(load_profile(ROOT / "configs/offline.yaml"), "fixture_ridge")
OBS = [HistoryItem(kind="observation", action_id=f"a{i}", payload={"observation_id": f"o{i}", "parameters": p, "results": r})
       for i, (p, r) in enumerate([({"temperature": 60.0, "time": 20.0}, {"yield": 40.1}),
                                   ({"temperature": 80.0, "time": 35.0}, {"yield": 91.3})])]
VIEW = ResearcherView(task=TASK, actions_used=2, remaining_actions=48, history=OBS, notes=[])


class Recording:
    def __init__(self, inner):
        self.inner, self.name, self.requests = inner, inner.name, []

    def generate(self, req):
        self.requests.append(req)
        return self.inner.generate(req)


@pytest.fixture(scope="module")
def live():
    profile = load_profile(os.environ["LABGENE_LIVE_PROFILE"])
    if profile.roles.researcher.provider != "gemini":
        pytest.skip("Gemini smoke: the profile's researcher is not on gemini (use configs/live-gemini-smoke.yaml)")
    problems = preflight_problems(profile)
    if problems:
        pytest.fail("profile is not approved for paid calls: " + "; ".join(problems))
    events = []
    guard = CostGuard(profile.cost_caps)
    ctx = CallContext(sink=events.append, guard=guard)
    provider = Recording(build_llm(profile.roles.researcher, timeout_s=profile.limits.provider_timeout_s))
    yield profile, provider, ctx, events
    out = ROOT / "artifacts" / "live-smoke"    # measured evidence (git-ignored): one CostEvent per physical attempt
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in events), encoding="utf-8")
    (out / "guard.json").write_text(json.dumps(guard.summary(), indent=1), encoding="utf-8")


def _req(profile, **kw):
    r = profile.roles.researcher
    return GenerationRequest(model=r.model, thinking_level=r.thinking_level, max_output_tokens=r.max_output_tokens, **kw)


def test_live_structured_output_and_usage(live):
    profile, provider, ctx, events = live
    res = call_llm(provider, _req(profile, role="researcher_finalizer", system_instruction="Reply with one action JSON.",
                                  input=[{"role": "user", "text": "Run an experiment at temperature 83 degC, time 37 min."}],
                                  response_schema=action_schema(TASK)), ctx, profile.limits)
    assert res.status is ProviderStatus.ok, res.error
    assert json.loads(res.text)["action"] in ("consult", "run_experiment")
    assert res.model_returned and res.usage.input_tokens is not None and res.usage.output_tokens is not None


def test_live_function_call_round_trip(live):
    profile, provider, ctx, events = live
    tools = AnalysisTools(VIEW)
    first_req = _req(profile, role="researcher_planner", system_instruction="Use the describe tool first.",
                     input=[{"role": "user", "text": "Call describe once, then summarise the result."}],
                     tools=tools.specs(), store=True)
    first = call_llm(provider, first_req, ctx, profile.limits)
    assert first.status is ProviderStatus.ok and first.function_calls, first.error
    fc = first.function_calls[0]
    out = tools.run(fc.name, fc.arguments).model_dump(mode="json")
    second_req = _req(profile, role="researcher_planner", system_instruction="Use the describe tool first.",
                      input=[{"role": "tool", "call_id": fc.call_id, "name": fc.name, "result": out}],
                      tools=tools.specs(), previous_interaction_id=first.interaction_id, store=True)
    carried = carried_tokens(first_req, first)
    second = call_llm(provider, second_req, ctx, profile.limits, context_tokens=carried)
    assert second.status is ProviderStatus.ok and second.text, second.error
    # history billing is undocumented: the continuation's reservation must cover what it actually billed
    if second.usage.input_tokens is not None:
        assert second.usage.input_tokens <= carried + estimate_input_tokens(second_req)


def test_live_researcher_roles_are_separate(live):
    profile, provider, ctx, events = live
    provider.requests.clear()
    d = PlannerReviewerResearcher(provider, profile.roles.researcher, profile.limits).decide(VIEW, ctx)
    assert d.status == "ok", d.error
    json.loads(d.raw_action_text)
    roles = [r.role for r in provider.requests]
    assert roles[0] == "researcher_planner" and roles[-1] == "researcher_finalizer" and "researcher_reviewer" in roles
    first_of_role = {r.role: r for r in reversed(provider.requests)}
    assert all(r.previous_interaction_id is None for r in first_of_role.values())
    assert all(e.model == profile.roles.researcher.model for e in events)
