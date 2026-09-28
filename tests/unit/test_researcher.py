"""T06 researcher behaviour: role separation (B18), provider failure handling (B19), planner/reviewer powers (B05),
observation-only analysis (B20) and the deterministic offline policy. Fixture results are contract checks only."""
import json
from pathlib import Path

import pytest
import yaml

from labgene.config import FixtureBehaviour, Limits, RoleModel, load_profile, load_public_task
from labgene.contracts import (AnalysisResult, FunctionCall, HistoryItem, Observation, ResearcherView,
                               ValidParameters)
from labgene.costs import CallContext
from labgene.providers.base import ModelChangedError
from labgene.providers.fixture import FixtureProvider
from labgene.providers.gemini import GeminiInteractionsProvider
from labgene.researcher.agent import (FINALIZER_SYSTEM, PLANNER_SYSTEM, PROMPT_HASH, REVIEWER_SYSTEM,
                                      PlannerReviewerResearcher)
from labgene.researcher.analysis import AnalysisError, AnalysisTools
from labgene.researcher.fixture_policy import choose_action, make_fixture_policy
from labgene.simulators.fixture import FUNCTIONS
from labgene.simulators.validation import validate_parameters

ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_profile(ROOT / "configs/offline.yaml")
RIDGE = load_public_task(PROFILE, "fixture_ridge")
CAT = load_public_task(PROFILE, "fixture_catalyst")
ROLE = PROFILE.roles.researcher
ANALYSIS_TOOLS = {"describe", "fit", "predict", "nearest"}
SYSTEMS = {"researcher_planner": PLANNER_SYSTEM, "researcher_reviewer": REVIEWER_SYSTEM,
           "researcher_finalizer": FINALIZER_SYSTEM}


def obs(i, params, results):
    return HistoryItem(kind="observation", action_id=f"a{i}",
                       payload={"observation_id": f"o{i}", "parameters": params, "results": results})


def view(task, history=()):
    history = list(history)
    return ResearcherView(task=task, actions_used=len(history), remaining_actions=50 - len(history),
                          history=history, notes=[])


def ctx(events=None):
    return CallContext(sink=(events.append if events is not None else lambda e: None))


def researcher(provider, limits=None):
    return PlannerReviewerResearcher(provider, ROLE, limits or PROFILE.limits, sleep=lambda s: None)


RIDGE_OBS = [obs(1, {"temperature": 60.0, "time": 20.0}, {"yield": 40.0}),
             obs(2, {"temperature": 80.0, "time": 35.0}, {"yield": 90.0})]


# ---------------------------------------------------------------- B18

def test_b18_roles_use_separate_requests_each_with_explicit_config():
    p = FixtureProvider(make_fixture_policy(FixtureBehaviour()))
    d = researcher(p).decide(view(RIDGE, RIDGE_OBS), ctx())
    roles = [r.role for r in p.requests]
    assert roles == ["researcher_planner", "researcher_planner", "researcher_reviewer", "researcher_finalizer"]
    for r in p.requests:
        assert r.system_instruction == SYSTEMS[r.role]
        assert (r.model, r.thinking_level, r.max_output_tokens) == (ROLE.model, "high", 4096)
    planner1, planner2, reviewer, finalizer = p.requests
    assert planner1.previous_interaction_id is None
    assert planner2.previous_interaction_id == "fixture-int-1"          # only its own tool loop is continued
    assert reviewer.previous_interaction_id is None and finalizer.previous_interaction_id is None
    assert finalizer.tools == [] and finalizer.response_schema is not None and finalizer.store is False
    assert d.status == "ok" and json.loads(d.raw_action_text)["action"] == "run_experiment"
    assert [a.method for a in d.analysis] == ["descriptive_statistics"]


def test_b18_no_continuation_reused_across_decisions_or_episodes():
    p = FixtureProvider(make_fixture_policy(FixtureBehaviour()))
    r = researcher(p)
    decision_of = []
    for i, v in enumerate([view(RIDGE, RIDGE_OBS), view(RIDGE, RIDGE_OBS[:1]), view(CAT)]):   # 2 decisions + new episode
        r.decide(v, ctx())
        decision_of += [i] * (len(p.requests) - len(decision_of))
    chained = 0
    for i, req in enumerate(p.requests):
        if req.previous_interaction_id:
            j = int(req.previous_interaction_id.rsplit("-", 1)[1]) - 1
            assert j < i and decision_of[j] == decision_of[i] and p.requests[j].role == req.role
            chained += 1
    assert chained == 2


def test_b18_stateless_provider_resends_phase_history_without_ids():
    p = FixtureProvider(make_fixture_policy(FixtureBehaviour()), chain_ids=False)
    researcher(p).decide(view(RIDGE, RIDGE_OBS), ctx())
    cont = p.requests[1]
    assert cont.previous_interaction_id is None
    assert [t["role"] for t in cont.input] == ["user", "model", "tool"] and cont.input[2]["result"]["kind"] == "descriptive_statistic"


def test_b18_prompts_are_fixed_and_requests_are_a_pure_function_of_the_view():
    v = view(CAT, [obs(1, {"catalyst": "C", "loading": 1.0, "temperature": 80.0, "residence_time": 300.0},
                       {"yield": 50.0, "ton": 45.0})])
    runs = []
    for _ in range(2):
        p = FixtureProvider(make_fixture_policy(FixtureBehaviour()))
        researcher(p).decide(v, ctx())
        runs.append([r.model_dump() for r in p.requests])
    assert runs[0] == runs[1]
    text = json.dumps(runs[0])
    assert '"condition"' not in text and "set_id" not in text and "baseline" not in text
    assert len(PROMPT_HASH) == 64


def test_b18_returned_model_change_stops_the_researcher(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    reply = json.dumps({"id": "i1", "model": "gemini-2.5-flash", "status": "completed",
                        "steps": [{"type": "model_output", "content": [{"type": "text", "text": "plan"}]}]}).encode()
    p = GeminiInteractionsProvider(transport=lambda *a: (200, {}, reply))
    role = RoleModel(provider="gemini", model="gemini-3.1-pro-preview", endpoint="interactions", thinking_level="high",
                     max_output_tokens=1024)
    with pytest.raises(ModelChangedError):
        PlannerReviewerResearcher(p, role, Limits(), sleep=lambda s: None).decide(view(RIDGE), ctx())


class HistoryBillingGemini:
    """Fake Interactions transport that bills stored history as input on every continuation (the conservative
    reading of the undocumented billing): total_input_tokens = chained history + ~bytes/4 of the new request."""

    def __init__(self):
        self.history, self.calls = {}, 0

    def __call__(self, method, url, headers, body, timeout):
        b = json.loads(body)
        self.calls += 1
        prev = b.get("previous_interaction_id")
        total_in = self.history.get(prev, 0) + len(body) // 4
        iid = f"int-{self.calls}"
        self.history[iid] = total_in + 200
        if b.get("response_format"):
            steps = [{"type": "model_output", "content": [{"type": "text", "text": '{"action":"consult","args":{"question":"q"}}'}]}]
        elif b["tools"] and not prev:
            steps = [{"type": "function_call", "id": "c1", "name": "describe", "arguments": {}}]
        else:
            steps = [{"type": "model_output", "content": [{"type": "text", "text": "plan " * 60}]}]
        return 200, {}, json.dumps({"id": iid, "model": ROLE.model, "status": "completed", "steps": steps,
                                    "usage": {"total_input_tokens": total_in, "total_output_tokens": 200}}).encode()


def test_b22_finding1_tool_loop_continuations_never_push_input_tokens_past_the_cap(monkeypatch):
    from labgene.config import CostCaps
    from labgene.costs import CapExceeded, CostGuard
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    role = RoleModel(provider="gemini", model=ROLE.model, endpoint="interactions", thinking_level="high",
                     max_output_tokens=200)
    v = view(RIDGE, [obs(i, {"temperature": 30.0 + i, "time": 10.0 + i}, {"yield": float(i)}) for i in range(20)])
    stopped_before_continuation = False
    for cap in range(2000, 60001, 2000):
        t, guard = HistoryBillingGemini(), CostGuard(CostCaps(max_input_tokens=cap))
        try:
            PlannerReviewerResearcher(GeminiInteractionsProvider(transport=t), role, Limits(), sleep=lambda s: None) \
                .decide(v, CallContext(sink=lambda e: None, guard=guard))
        except CapExceeded:
            stopped_before_continuation |= t.calls == 1
        assert guard.input_tokens <= cap, f"cap {cap} overrun after the fact: {guard.input_tokens}"
    assert stopped_before_continuation                     # the tool-loop continuation path was actually exercised


# ---------------------------------------------------------------- B19

def test_b19_truncated_finalizer_json_is_returned_as_is_for_the_harness_to_reject():
    v = view(RIDGE)
    full = json.dumps(choose_action(v.model_dump(mode="json"), FixtureBehaviour()), sort_keys=True)
    p = FixtureProvider(make_fixture_policy(FixtureBehaviour()), script=["ok", "ok", "incomplete"])
    d = researcher(p).decide(v, ctx())
    assert d.status == "provider_incomplete" and d.raw_action_text == full[: len(full) // 2] and d.note is None
    with pytest.raises(ValueError):
        json.loads(d.raw_action_text)
    assert len(p.requests) == 3                                     # incomplete is not retried as infra


def test_b19_refusal_and_infra_error_are_distinct_and_retries_are_finite():
    p = FixtureProvider(make_fixture_policy(FixtureBehaviour()), script=["ok", "ok", "refusal"])
    d = researcher(p).decide(view(RIDGE), ctx())
    assert d.status == "provider_refusal" and d.raw_action_text is None

    events = []
    p = FixtureProvider(make_fixture_policy(FixtureBehaviour()), script=["ok", "ok"] + ["infra_error"] * 9)
    d = researcher(p, Limits(provider_max_attempts=3)).decide(view(RIDGE), ctx(events))
    assert d.status == "infra_error" and d.raw_action_text is None
    assert [(e.role, e.attempt, e.status) for e in events][-3:] == \
        [("researcher_finalizer", i, "infra_error") for i in (1, 2, 3)]
    assert len(events) == len(p.requests) == 5 and all(e.provider == "fixture" for e in events)

    p = FixtureProvider(make_fixture_policy(FixtureBehaviour()), script=["infra_error"] * 3)
    d = researcher(p, Limits(provider_max_attempts=3)).decide(view(RIDGE), ctx())
    assert d.status == "infra_error" and {r.role for r in p.requests} == {"researcher_planner"}


# ---------------------------------------------------------------- B05: planner/reviewer cannot act; bounded tool loop

def test_b05_planner_and_reviewer_get_only_bounded_analysis_tools():
    base = make_fixture_policy(FixtureBehaviour())

    def greedy(req):                    # tries to act and to call tools forever
        if req.role == "researcher_finalizer" or not req.tools:
            return base(req)
        return [FunctionCall(call_id="x", name="run_experiment", arguments={"parameters": {"temperature": 83, "time": 37}}),
                FunctionCall(call_id="y", name="describe", arguments={})]

    p = FixtureProvider(greedy)
    d = researcher(p, Limits(researcher_max_analysis_calls=3)).decide(view(RIDGE, RIDGE_OBS), ctx())
    for r in p.requests:
        assert {t.name for t in r.tools} <= ANALYSIS_TOOLS
        for t in r.input:
            if t.get("role") == "tool" and t["name"] == "run_experiment":
                assert "unknown analysis tool" in t["result"]["error"]
    for role in ("researcher_planner", "researcher_reviewer"):
        reqs = [r for r in p.requests if r.role == role]
        executed = sum(1 for r in reqs for t in r.input if t.get("role") == "tool"
                       and "limit" not in t["result"].get("error", ""))
        assert executed <= 3 and len(reqs) <= 3 + 1 and reqs[-1].tools == []
    assert d.status == "ok" and all(isinstance(a, AnalysisResult) for a in d.analysis)


# ---------------------------------------------------------------- B20: observation-only analysis

def _private_markers():
    out = []
    for f in (ROOT / "tests/fixtures/private").glob("*.yaml"):
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        out += data["answer_bundle"]["secret_markers"]
    return out


@pytest.mark.parametrize("name,args", [
    ("read_file", {"path": "tests/fixtures/private/fixture_ridge.yaml"}),
    ("run_experiment", {"parameters": {"temperature": 83, "time": 37}}),
    ("simulate", {"temperature": 83}),
    ("describe", {"path": "../../private/fixture_ridge.yaml"}),
    ("describe", {"metrics": ["file:///C:/labgene/tests/fixtures/private/fixture_ridge.yaml"]}),
    ("fit", {"metric": "https://fixture.example/papers/ridge-answer"}),
    ("describe", {"condition": "product"}),
    ("describe", {"set_id": "smoke", "set_rep": 2}),
    ("fit", {"metric": "yield", "simulator": "fixture.ridge"}),
    ("nearest", {"point": {"../../private": 1.0}}),
    ("predict", {"metric": "yield", "points": [{"temperature": 83.0, "time": 37.0}], "url": "http://x"}),
    ("describe", "not-an-object"),
])
def test_b20_analysis_tools_reject_file_network_simulator_and_other_condition_access(name, args):
    tools = AnalysisTools(view(RIDGE, RIDGE_OBS))
    with pytest.raises(AnalysisError) as e:
        tools.run(name, args)
    assert not any(m in str(e.value) for m in _private_markers())


def test_b20_finding2_overflowing_numbers_become_tool_errors_not_crashes():
    tools = AnalysisTools(view(RIDGE, RIDGE_OBS))
    huge = int("9" * 401)
    for name, args in [("nearest", {"point": {"temperature": 1e308, "time": -1e308}}),
                       ("nearest", {"point": {"temperature": huge}}),
                       ("predict", {"metric": "yield", "features": ["temperature"], "points": [{"temperature": huge}]}),
                       ("predict", {"metric": "yield", "features": ["temperature"], "points": [{"temperature": 1e308}]})]:
        with pytest.raises(AnalysisError):
            tools.run(name, args)
    base = make_fixture_policy(FixtureBehaviour())
    bad = FunctionCall(call_id="z", name="nearest", arguments={"point": {"temperature": 1e308, "time": -1e308}})
    p = FixtureProvider(lambda req: [bad] if req.tools and not req.previous_interaction_id else base(req))
    d = researcher(p).decide(view(RIDGE, RIDGE_OBS), ctx())
    assert d.status == "ok"                               # the model gets the error back; decide() does not crash
    assert "overflow" in p.requests[1].input[0]["result"]["error"]


def test_b20_analysis_reads_only_observations_from_the_view():
    consult = HistoryItem(kind="consult", action_id="c1", payload={
        "question": "q", "response": {"answer": "try 83/37", "candidates": [{"parameters": {"temperature": 83.0, "time": 37.0}}]}})
    tools = AnalysisTools(view(RIDGE, [consult] + RIDGE_OBS))
    r = tools.run("describe", {})
    assert r.result["yield"]["n"] == 2 and r.input_observation_ids == ["o1", "o2"]
    assert not isinstance(r, Observation) and r.kind == "descriptive_statistic"


def test_b20_predictions_are_agent_predictions_with_extrapolation_and_range_flags():
    f = lambda t, m: 10 + 0.1 * t - 0.001 * t * t + 0.2 * m + 0.002 * t * m   # exact quadratic
    grid = [(t, m) for t in (30.0, 60.0, 90.0) for m in (10.0, 30.0, 50.0)]
    tools = AnalysisTools(view(RIDGE, [obs(i, {"temperature": t, "time": m}, {"yield": f(t, m)})
                                       for i, (t, m) in enumerate(grid)]))
    fit = tools.run("fit", {"metric": "yield", "degree": 2})
    assert fit.kind == "analysis" and fit.uncertainty["r2"] == pytest.approx(1.0)
    r = tools.run("predict", {"metric": "yield", "degree": 2,
                              "points": [{"temperature": 45.0, "time": 20.0}, {"temperature": 110.0, "time": 20.0},
                                         {"temperature": 150.0, "time": 20.0}]})
    assert r.kind == "agent_prediction" and "NOT an observation" in r.result["note"]
    inside, extrap, outside = r.result["predictions"]
    assert inside["predicted"] == pytest.approx(f(45.0, 20.0), abs=1e-6)
    assert (inside["extrapolation"], inside["out_of_range"]) == (False, False)
    assert (extrap["extrapolation"], extrap["out_of_range"]) == (True, False)
    assert (outside["extrapolation"], outside["out_of_range"]) == (True, True)
    with pytest.raises(AnalysisError):                       # 6 quadratic terms need >= 6 observations
        AnalysisTools(view(RIDGE, RIDGE_OBS)).run("fit", {"metric": "yield", "degree": 2})


def test_b20_categorical_filter_constraints_and_nearest():
    rows = [obs(i, {"catalyst": c, "loading": 1.0, "temperature": t, "residence_time": rt}, {"yield": y, "ton": y})
            for i, (c, t, rt, y) in enumerate([("C", 70.0, 200.0, 40.0), ("C", 80.0, 300.0, 60.0),
                                               ("C", 90.0, 250.0, 55.0), ("A", 70.0, 300.0, 20.0)])]
    tools = AnalysisTools(view(CAT, rows))
    assert tools.run("describe", {"where": {"catalyst": "C"}}).result["yield"]["n"] == 3
    with pytest.raises(AnalysisError):
        tools.run("describe", {"where": {"catalyst": "Z"}})
    p = tools.run("predict", {"metric": "yield", "features": ["temperature", "residence_time"], "where": {"catalyst": "C"},
                              "points": [{"temperature": 100.0, "residence_time": 600.0},
                                         {"temperature": 80.0, "residence_time": 300.0}]})
    assert [x["violates_constraints"] for x in p.result["predictions"]] == [True, False]
    assert p.input_observation_ids == ["o0", "o1", "o2"]
    near = tools.run("nearest", {"point": {"catalyst": "C", "temperature": 88.0}, "k": 2})
    assert [n["observation_id"] for n in near.result["neighbours"]] == ["o2", "o1"]
    specs = {s.name: s for s in tools.specs()}
    assert set(specs) == ANALYSIS_TOOLS
    assert specs["fit"].parameters["properties"]["where"]["properties"]["catalyst"]["enum"] == CAT.parameters[0].choices


# ---------------------------------------------------------------- offline fixture policy (development_only)

def _loop(task, behaviour, n):
    """Mini harness stand-in: validate + fixture simulator; advisor always suggests one fixed candidate."""
    fn = FUNCTIONS[task.simulator_id.split(".")[1]]
    hist, actions = [], []
    for i in range(n):
        a = choose_action(view(task, hist).model_dump(mode="json"), behaviour)
        actions.append(a)
        if a["action"] == "consult":
            cand = {"temperature": 80.0, "time": 40.0} if task is RIDGE else \
                {"catalyst": "C", "loading": 1.2, "temperature": 90.0, "residence_time": 540.0}
            hist.append(HistoryItem(kind="consult", action_id=f"a{i}",
                                    payload={"question": a["args"]["question"],
                                             "response": {"answer": "x", "candidates": [{"parameters": cand}]}}))
            continue
        v = validate_parameters(task, a["args"]["parameters"])
        if isinstance(v, ValidParameters):
            hist.append(obs(i, v.parameters, fn(v.parameters)))
        else:
            hist.append(HistoryItem(kind="invalid_experiment", action_id=f"a{i}",
                                    payload={"submitted_parameters": a["args"]["parameters"], "reason": v.reason}))
    return actions, hist


def test_fixture_policy_produces_consult_valid_and_invalid_actions_deterministically():
    b = FixtureBehaviour(consult_every=0, invalid_request_rate_every=3)
    actions, hist = _loop(RIDGE, b, 10)
    assert _loop(RIDGE, b, 10)[0] == actions                                   # deterministic
    kinds = [h.kind for h in hist]
    assert kinds[0] == "consult" and kinds.count("consult") == 1
    assert actions[1]["args"]["parameters"] == {"temperature": 80.0, "time": 40.0}  # adopts the advisor candidate
    exp_kinds = [k for k in kinds if k != "consult"]
    assert [i for i, k in enumerate(exp_kinds, 1) if k == "invalid_experiment"] == [3, 6, 9]
    valid = [json.dumps(h.payload["parameters"], sort_keys=True) for h in hist if h.kind == "observation"]
    assert len(valid) == len(set(valid))                                        # never repeats a tried point
    assert all(a["args"]["hypothesis"].startswith("fixture:") for a in actions if a["action"] == "run_experiment")
    consults = [i for i, a in enumerate(_loop(RIDGE, FixtureBehaviour(consult_every=4), 9)[0]) if a["action"] == "consult"]
    assert consults == [0, 4, 8]


def test_fixture_policy_respects_categorical_params_and_linear_constraints():
    actions, hist = _loop(CAT, FixtureBehaviour(invalid_request_rate_every=0), 25)
    assert all(h.kind in ("consult", "observation") for h in hist)
    used = {h.payload["parameters"]["catalyst"] for h in hist if h.kind == "observation"}
    assert len(used) > 1                                                        # categorical moves happen
