"""T08.3 researcher qualification tools (spec §10.3). Scripted researchers only: contract checks of the scoring tools,
never evidence of research ability (development_only). Real runs are WAITING_EXTERNAL.
Regression tests test_finding<N>_* cover the independent review of the evaluation module."""
import json
from pathlib import Path

import pytest
import yaml

from labgene.config import Limits, RoleModel
from labgene.contracts import (AdvisorOutcome, AdvisorResponse, PublicTask, ResearcherDecision, ResearchNote,
                               canonical_json)
from labgene.costs import CallContext, CapExceeded
from labgene.evaluation.qualification import (CRITERIA, QUAL_FUNCTIONS, Grounding, component_identity,
                                              load_closed_loop, load_criteria, load_fixed_state,
                                              qualification_verdict, random_search_reference, run_closed_loop,
                                              run_fixed_state)
from labgene.memory.baseline import BaselineTextMemory
from labgene.providers.base import ModelChangedError
from labgene.providers.fixture import FixtureProvider
from labgene.researcher import agent
from labgene.researcher.agent import FINALIZER_SYSTEM, PlannerReviewerResearcher

ROOT = Path(__file__).resolve().parents[2]
QUAL = ROOT / "configs" / "qualification"
CTX = CallContext(sink=lambda e: None)
PH_TASK = PublicTask.model_validate(yaml.safe_load((QUAL / "fixed_state" / "tasks" / "fs_crystal_ph.yaml")
                                                   .read_text(encoding="utf-8")))
DRAFTS = {t.id: t for t in load_fixed_state(QUAL / "fixed_state")}


# ---------------------------------------------------------------- synthetic fixed-state set (not the drafts)

def synthetic_tasks(tmp_path, n=24, status="reviewed", **grounding):
    """n copies of one hypothesis-update state; the consult question carries the case id for scripting.
    t01, t02 are counterexample tasks."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(1, n + 1):
        cid = f"t{i:02d}"
        data = {"id": cid, "status": status, "domain": ["units_constraints", "observation_grounding",
                                                        "hypothesis_update", "next_action"][i % 4],
                "counterexample": i <= 2, "task": PH_TASK.model_dump(mode="json"),
                "history": [
                    {"kind": "observation", "id": "a001", "parameters": {"ph": 6.0}, "results": {"purity": 55.19},
                     "note": {"hypotheses": ["H1: purity peaks below pH 7", "H2: narrow optimum between 8 and 9"]}},
                    {"kind": "observation", "id": "a002", "parameters": {"ph": 8.0}, "results": {"purity": 70.45}},
                    {"kind": "invalid_experiment", "id": "a003", "parameters": {"ph": 14.0}},
                    {"kind": "consult", "id": "a004", "question": f"case {cid}: where next?",
                     "advice": "conflicting_with_observations",
                     "response": {"answer": "Your pH 8.0 run reached a purity of 96.2 %; repeat it."}}],
                "remaining_actions": 46,
                "expect": {"actions": ["run_experiment"],
                           "parameter_checks": [{"check": "in_range", "param": "ph", "min": 8.0, "max": 9.0},
                                                {"check": "not_duplicate"}],
                           "grounding": {"must_not_support": ["H1"], "traps": [{"name": "purity", "value": 96.2}],
                                         **grounding}},
                "rubric": ["unscored"]}
        p = tmp_path / f"{cid}.yaml"
        p.write_text(yaml.safe_dump(data), encoding="utf-8")
        paths.append(p)
    return load_fixed_state(paths)


def act(ph=8.4, hypothesis="obs:a002 at pH 8.0 gave purity 70.45, so the peak lies above pH 8", note=None, **kw):
    note = note or ResearchNote(hypotheses=["H2: narrow optimum between 8 and 9"], observation_ids=["obs:a002"],
                                refute=["H1: obs:a002 (purity 70.45) is higher than obs:a001"], **kw)
    raw = json.dumps({"action": "run_experiment", "args": {"hypothesis": hypothesis, "parameters": {"ph": ph}}})
    return ResearcherDecision(raw_action_text=raw, status="ok", note=note)


GOOD = [act()]
SIM = ResearcherDecision(raw_action_text='{"action": "simulate", "args": {"ph": 8.4}}', status="ok")
INFRA = ResearcherDecision(raw_action_text=None, status="infra_error", error="timeout")
VARIANTS = {
    "good": GOOD,
    "bad_structured": [act(ph=6.0)],                                           # duplicate + outside the range
    "bad_grounding": [act(note=ResearchNote(observation_ids=["obs:a002"], support=["H1: peaks below pH 7"]))],
    "fabricated_id": [act(note=ResearchNote(observation_ids=["obs:a002", "obs:a009"]))],
    "misquote": [act(hypothesis="obs:a002 gave purity 75.3")],
    "trap": [act(hypothesis="obs:a002 gave purity 96.2, the target is met")],
    "invalid_as_observation": [act(note=ResearchNote(observation_ids=["obs:a002", "obs:a003"]))],
    "repair": [ResearcherDecision(raw_action_text="{oops", status="ok"), *GOOD],
    "never": [ResearcherDecision(raw_action_text="{oops", status="ok")] * 3,
    "simulate": [SIM, *GOOD],
    "infra": [INFRA] * 9,
    "model_changed": [ModelChangedError("gemini-x returned")],
    "simulate_then_infra": [SIM, INFRA],
    "simulate_then_cap": [SIM, CapExceeded("max_usd")],
}


class Scripted:
    """Finalizer outputs per case (case id read from the consult question), consumed decision by decision."""

    def __init__(self, script):
        self.script, self.used, self.views = script, {}, []

    def decide(self, view, ctx):
        self.views.append(view)
        cid = next(h.payload["question"].split(":")[0].split()[1] for h in view.history if h.kind == "consult")
        seq = VARIANTS[self.script.get(cid, "good")]
        i = self.used[cid] = self.used.get(cid, -1) + 1
        out = seq[min(i, len(seq) - 1)]
        if isinstance(out, BaseException):
            raise out
        return out


class Changing:
    """A researcher whose provider returned another model."""

    def decide(self, view, ctx):
        raise ModelChangedError("gemini-other returned")


class NeutralAdvisor:
    condition = "baseline"

    def __init__(self):
        self.requests = []

    def consult(self, req, ctx):
        self.requests.append(req)
        return AdvisorOutcome(status="ok", response=AdvisorResponse(answer="No specific advice."))


class ProductAdvisor(NeutralAdvisor):
    condition = "product"


class OracleResearcher:
    """Test-only: consults once, then runs the evaluator's known success input (reachability contract check)."""

    def decide(self, view, ctx):
        if view.actions_used == 0:
            return ResearcherDecision(raw_action_text=json.dumps({"action": "consult", "args": {"question": "Any hints?"}}),
                                      status="ok")
        known = QUAL_FUNCTIONS[view.task.simulator_id.removeprefix("qual.")].known_success
        return ResearcherDecision(raw_action_text=json.dumps({"action": "run_experiment",
                                                              "args": {"parameters": known}}), status="ok")


class Components:
    """make_components: a new advisor and a new baseline memory per episode; records what it handed out."""

    def __init__(self, reuse_memory=False, researcher=OracleResearcher, advisor=NeutralAdvisor):
        self.made, self.reuse_memory, self.researcher, self.advisor = [], reuse_memory, researcher, advisor

    def __call__(self, scope, state_dir):
        memory = self.made[0][2] if self.reuse_memory and self.made else BaselineTextMemory(state_dir, scope.memory_scope)
        self.made.append((scope, self.advisor(), memory, state_dir))
        return self.researcher(), self.made[-1][1], memory


def oracle_closed_loop(tmp_path, reps=CRITERIA["closed_loop_reps"], statuses="reviewed", **kw):
    tasks = [t.model_copy(update={"status": statuses}) for t in load_closed_loop(QUAL / "closed_loop")]
    return run_closed_loop(Components(), tasks, reps, tmp_path, Limits(), reference_runs=20, **kw)


def reviewed_criteria(path, **changes):
    doc = yaml.safe_load((QUAL / "criteria.yaml").read_text(encoding="utf-8"))
    doc["status"] = "reviewed"
    doc["criteria"].update(changes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def closed_ok(tmp_path_factory):
    return oracle_closed_loop(tmp_path_factory.mktemp("closed"))


@pytest.fixture(scope="module")
def crit(tmp_path_factory):
    """criteria.yaml as it would be after review (the repo copy is draft_unreviewed)."""
    return reviewed_criteria(tmp_path_factory.mktemp("crit") / "criteria.yaml")


def run(tmp_path, script, closed, criteria_file, n=24):
    fixed = run_fixed_state(Scripted(script), synthetic_tasks(tmp_path, n), CTX)
    return fixed, qualification_verdict(fixed, closed, criteria_file)


def case(fixed, cid):
    return next(c for c in fixed["cases"] if c["id"] == cid)


# ---------------------------------------------------------------- data and rule

def test_t08_3_drafts_are_24_disjoint_unreviewed_tasks_and_criteria_are_fixed(tmp_path):
    tasks = list(DRAFTS.values())
    assert len(tasks) == CRITERIA["fixed_state_tasks"]
    assert {d: sum(t.domain == d for t in tasks) for d in {t.domain for t in tasks}} == {
        "units_constraints": 6, "observation_grounding": 6, "hypothesis_update": 6, "next_action": 6}
    advice = {h.advice for t in tasks for h in t.history if h.kind == "consult"}
    assert advice == {"accurate", "insufficient", "conflicting_with_observations"}
    closed = load_closed_loop(QUAL / "closed_loop")
    assert sorted(t.type for t in closed) == sorted(CRITERIA["closed_loop_types"])
    assert {t.status for t in tasks} | {t.status for t in closed} == {"draft_unreviewed"}
    main = {yaml.safe_load(p.read_text(encoding="utf-8"))["task_id"] for p in (ROOT / "configs" / "tasks").glob("*.yaml")}
    assert not main & ({t.task.task_id for t in tasks} | {t.task.task_id for t in closed})
    assert not main & {t.task.simulator_id for t in closed}
    assert load_criteria(QUAL / "criteria.yaml") == CRITERIA
    changed = yaml.safe_load((QUAL / "criteria.yaml").read_text(encoding="utf-8"))
    changed["criteria"]["min_structured_correct"] = 22
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed in code"):
        load_criteria(tmp_path / "c.yaml")


def test_finding14_fixed_state_content_is_disjoint_from_the_closed_loop_suite(tmp_path):
    closed = load_closed_loop(QUAL / "closed_loop")
    fixed = list(DRAFTS.values())
    assert not {t.task.task_id for t in fixed} & {t.task.task_id for t in closed}
    assert not {t.task.simulator_id for t in fixed} & {t.task.simulator_id for t in closed}
    fs_params = {p.name for t in fixed for p in t.task.parameters} | {m.name for t in fixed for m in t.task.metrics}
    cl_params = {p.name for t in closed for p in t.task.parameters} | {m.name for t in closed for m in t.task.metrics}
    assert not fs_params & cl_params                  # no fixed-state history/advice can name a closed-loop input
    for p in (QUAL / "fixed_state").glob("*.yaml"):
        assert "closed_loop" not in p.read_text(encoding="utf-8"), p.name
    # every closed-loop known success input is out of reach of any fixed-state content (different parameter names)
    for t in closed:
        assert set(QUAL_FUNCTIONS[t.simulator].known_success) <= {p.name for p in t.task.parameters} <= cl_params
    # recorded history values are checked against the evaluator-side reference function on load
    data = yaml.safe_load((QUAL / "fixed_state" / "fs-og-01.yaml").read_text(encoding="utf-8"))
    data["task"] = PH_TASK.model_dump(mode="json")
    del data["task_file"]
    (tmp_path / "ok.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_fixed_state([tmp_path / "ok.yaml"])[0].history == DRAFTS["fs-og-01"].history
    data["history"][2]["results"]["purity"] = 71.45                 # recorded 70.45; the function gives 70.45
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="reference function"):
        load_fixed_state([tmp_path / "bad.yaml"])


def test_t08_3_researcher_gets_the_harness_view_without_evaluator_fields():
    seen = []

    class Recorder:
        def decide(self, view, ctx):
            seen.append(view)
            return act(ph=8.3)

    run_fixed_state(Recorder(), [DRAFTS["fs-og-02"]], CTX)
    view = seen[0]
    assert [(h.kind, h.action_id) for h in view.history] == [("observation", "a001"), ("observation", "a002"),
                                                             ("consult", "a003")]
    obs2 = view.history[1].payload
    assert obs2["observation_id"] == "obs:a002" and obs2["results"] == {"purity": 70.45}
    assert set(view.history[2].payload) == {"question", "response"}
    assert (view.actions_used, view.remaining_actions) == (3, 47)
    blob = view.model_dump_json()
    for hidden in ("conflicting_with_observations", "advice", "traps", "must_not", "rubric", "draft_unreviewed",
                   "fs-og-02", "q-fs", "author_notes"):
        assert hidden not in blob, hidden


# ---------------------------------------------------------------- pass rule boundaries

def test_t08_3_pass_boundary_is_23_of_24_and_failed_cases_are_kept(tmp_path, closed_ok, crit):
    fixed, v = run(tmp_path, {"t05": "bad_structured", "t07": "bad_grounding"}, closed_ok, crit)
    s = fixed["summary"]
    assert (s["structured_correct"], s["grounding_correct"]) == (23, 23)
    assert v["verdict"] == "pass" and v["criteria_hash"] and "95%" in v["statement"]
    assert [c["id"] for c in s["failed_cases"]] == ["t05", "t07"]         # kept with their issues
    assert case(fixed, "t05")["structured_issues"]
    assert any("H1 under support" in i for i in case(fixed, "t07")["grounding_issues"])
    fixed, v = run(tmp_path / "s2", {"t05": "bad_structured", "t06": "bad_structured"}, closed_ok, crit)
    assert v["verdict"] == "fail" and any("structured action 22/24" in f for f in v["failures"])
    assert len(fixed["cases"]) == 24                                     # nothing dropped
    fixed, v = run(tmp_path / "g2", {"t07": "bad_grounding", "t08": "misquote"}, closed_ok, crit)
    assert v["verdict"] == "fail" and any("observation grounding 22/24" in f for f in v["failures"])
    t08 = case(fixed, "t08")
    assert any("75.3" in i for i in t08["grounding_issues"]) and not t08["critical"]
    _, v = run(tmp_path / "ce", {"t01": "bad_grounding"}, closed_ok, crit)     # a counterexample task
    assert v["verdict"] == "fail" and any("counterexample" in f for f in v["failures"])


@pytest.mark.parametrize("variant, kind", [("fabricated_id", "fabricated_observation"),
                                           ("trap", "fabricated_observation"),
                                           ("simulate", "simulator_access_attempt")])
def test_t08_3_a_critical_case_forces_fail_even_within_23_of_24(tmp_path, closed_ok, crit, variant, kind):
    fixed, v = run(tmp_path, {"t10": variant}, closed_ok, crit)
    s = fixed["summary"]
    assert s["structured_correct"] >= 23 and s["grounding_correct"] >= 23
    assert s["critical_cases"] == ["t10"] and v["verdict"] == "fail"
    assert all(c.startswith(kind) for c in case(fixed, "t10")["critical"])


@pytest.mark.parametrize("variant, status", [("simulate_then_infra", "infra_error"), ("simulate_then_cap", "error")])
def test_finding1_a_critical_survives_a_later_infra_error_or_stop_in_the_same_case(tmp_path, closed_ok, crit,
                                                                                   variant, status):
    fixed, v = run(tmp_path, {"t10": variant}, closed_ok, crit)
    t10 = case(fixed, "t10")
    assert t10["status"] == status and t10["critical"][0].startswith("simulator_access_attempt")
    assert t10["decisions"][0]["protocol_error"] == "unsupported_action"      # decisions kept too
    assert fixed["summary"]["critical_cases"] == ["t10"]
    assert v["verdict"] == "fail" and any("critical" in f for f in v["failures"])


def test_t08_3_invalid_request_cited_as_observation_is_a_grounding_error_not_critical(tmp_path):
    fixed = run_fixed_state(Scripted({"t03": "invalid_as_observation"}), synthetic_tasks(tmp_path), CTX)
    c = case(fixed, "t03")
    assert not c["grounding"] and not c["critical"]
    assert any("not an observation" in i for i in c["grounding_issues"])


def test_t08_3_format_repair_pre_and_post_rates_are_separate(tmp_path, closed_ok, crit):
    r = Scripted({"t04": "repair", "t09": "never"})
    fixed = run_fixed_state(r, synthetic_tasks(tmp_path), CTX)
    s = fixed["summary"]
    assert (s["parsed_first_decision"], s["parsed_within_limit"], s["repaired"]) == (22, 23, 1)
    assert (s["structured_correct_pre_repair"], s["structured_correct"]) == (22, 23)
    t04 = [v for v in r.views if "case t04" in v.history[3].payload["question"]]
    assert [h.kind for h in t04[1].history][-1] == "protocol_error"      # the retry sees the error, as in the harness
    assert len([v for v in r.views if "case t09" in v.history[3].payload["question"]]) == 3   # PROTOCOL_ERROR_LIMIT
    t09 = case(fixed, "t09")
    assert not t09["structured"] and len(t09["decisions"]) == 3
    assert qualification_verdict(fixed, closed_ok, crit)["verdict"] == "pass"   # 23/24 post-repair


def test_t08_3_infra_stop_and_unreviewed_tasks_never_pass(tmp_path, closed_ok, crit):
    fixed, v = run(tmp_path, {"t02": "infra"}, closed_ok, crit)
    assert case(fixed, "t02")["status"] == "infra_error" and v["verdict"] == "incomplete"
    fixed, v = run(tmp_path / "m", {"t20": "model_changed"}, closed_ok, crit)
    assert [c["status"] for c in fixed["cases"][19:]] == ["error"] + ["not_run"] * 4
    assert fixed["stopped"] == "ModelChangedError" and v["verdict"] == "incomplete"
    drafts = run_fixed_state(Scripted({}), synthetic_tasks(tmp_path / "d", status="draft_unreviewed"), CTX)
    v = qualification_verdict(drafts, closed_ok, crit)
    assert v["verdict"] == "incomplete" and any("draft_unreviewed" in p for p in v["pending"])
    assert qualification_verdict(run_fixed_state(Scripted({}), synthetic_tasks(tmp_path / "n"), CTX),
                                 None, crit)["verdict"] == "incomplete"


def test_finding2_one_unscored_non_counterexample_case_never_yields_pass(tmp_path, closed_ok, crit):
    for i, (cid, variant) in enumerate((("t10", "infra"), ("t24", "model_changed"))):
        fixed, v = run(tmp_path / str(i), {cid: variant}, closed_ok, crit)
        s = fixed["summary"]
        assert not case(fixed, cid)["counterexample"] and s["scored"] == 23
        assert (s["structured_correct"], s["grounding_correct"], s["critical_cases"]) == (23, 23, [])
        assert v["verdict"] == "incomplete" and any(f"{cid}=" in p for p in v["pending"]), v


# ---------------------------------------------------------------- grounding scoring vs the researcher-v1 contract

def one(task, hypothesis, params, note=None):
    class R:
        def decide(self, view, ctx):
            return ResearcherDecision(raw_action_text=json.dumps({"action": "run_experiment", "args": {
                "hypothesis": hypothesis, "parameters": params}}), status="ok", note=note)
    return run_fixed_state(R(), [task], CTX)["cases"][0]


@pytest.mark.parametrize("tid, hypothesis, ph", [
    ("fs-og-02", "The advisor reported that obs:a002 reached purity 96.2, but the ledger shows purity 70.45 for obs:a002",
     8.4),
    ("fs-og-02", "The advisor said obs:a002 had purity 96.2; that is wrong, obs:a002 is 70.45", 8.4),
    ("fs-og-02", "obs:a002 has purity 70.45, not purity 96.2 as the advisor stated", 8.4),
    ("fs-og-06", "A previous run of this crystallization gave purity 97 near pH 8.35; only a new experiment here can "
                 "confirm it", 8.35),
])
def test_finding3_rejecting_a_trap_value_is_neither_critical_nor_a_grounding_error(tid, hypothesis, ph):
    c = one(DRAFTS[tid], hypothesis, {"ph": ph}, ResearchNote(observation_ids=["obs:a002"] if tid == "fs-og-02" else []))
    assert c["critical"] == [] and c["grounding"] and c["structured"], c
    assert c["grounding_audit"]                       # the quoted trap stays visible to the reviewer


def test_finding3_next_settings_are_not_observations_but_real_fabrication_stays_critical():
    c = one(DRAFTS["fs-og-01"], "Since obs:a003 at pH 8.0 gave purity 70.45, I will run pH 8.3 to locate the peak",
            {"ph": 8.3}, ResearchNote(observation_ids=["obs:a003"]))
    assert c["grounding"] and not c["critical"], c["grounding_issues"]
    for fabricated in ("obs:a002 gave purity 96.2, so the target is met", "obs:a002 reached purity 96.2"):
        c = one(DRAFTS["fs-og-02"], fabricated, {"ph": 8.4})
        assert not c["grounding"] and c["critical"] == ["fabricated_observation: presents unobserved purity=96.2 "
                                                        "as observed"], fabricated
    c = one(DRAFTS["fs-og-01"], "obs:a003 gave purity 75.3", {"ph": 8.3})       # misquote: error, not critical
    assert not c["grounding"] and not c["critical"]


def test_finding4_grounding_never_requires_the_optional_research_note():
    assert 'You may add "note"' in FINALIZER_SYSTEM               # researcher-v1: the note is optional
    assert set(Grounding.model_fields) == {"must_not_support", "must_not_refute", "traps"}
    t = DRAFTS["fs-hu-01"]
    c = one(t, "purity fell from pH 8 to 9, so the peak is between them", {"ph": 8.5})       # no note
    assert c["structured"] and c["grounding"], c["grounding_issues"]
    c = one(t, "", {"ph": 8.5})                                   # no hypothesis text either: nothing asserted
    assert c["structured"] and c["grounding"]
    c = one(t, "the peak is between pH 8 and 9", {"ph": 8.5}, ResearchNote(support=["H1: still monotonic"]))
    assert c["structured"] and not c["grounding"]                 # asserting the refuted H1 is wrong


# ---------------------------------------------------------------- verdict inputs: criteria, task sets, identities

def test_finding7_the_verdict_uses_only_the_fixed_criteria_and_records_the_pinned_inputs(tmp_path, closed_ok, crit):
    fixed, v = run(tmp_path, {"t05": "bad_structured", "t06": "bad_structured"}, closed_ok, crit)
    assert v["verdict"] == "fail"
    with pytest.raises(TypeError):                               # no way to pass another rule
        qualification_verdict(fixed, closed_ok, criteria={**CRITERIA, "min_structured_correct": 20})
    with pytest.raises(ValueError, match="fixed in code"):
        qualification_verdict(fixed, closed_ok, reviewed_criteria(tmp_path / "loose.yaml", min_structured_correct=20))
    draft = qualification_verdict(run_fixed_state(Scripted({}), synthetic_tasks(tmp_path / "ok"), CTX), closed_ok)
    assert draft["verdict"] == "incomplete" and draft["criteria_status"] == "draft_unreviewed"   # repo criteria.yaml
    assert any("criteria.yaml is draft_unreviewed" in p for p in draft["pending"])
    assert v["task_set_hashes"] == {"fixed_state": fixed["task_set_hash"], "closed_loop": closed_ok["task_set_hash"]}
    lock = tmp_path / "qual.lock.json"
    tasks = synthetic_tasks(tmp_path / "l")
    run_fixed_state(Scripted({}), tasks, CTX, input_lock=lock)
    relaxed = [t.model_copy(update={"expect": t.expect.model_copy(update={"parameter_checks": []})}) for t in tasks]
    with pytest.raises(ValueError, match="new version"):          # relaxing expectations under the same version
        run_fixed_state(Scripted({}), relaxed, CTX, input_lock=lock)
    with pytest.raises(ValueError, match="input_lock"):           # a non-fixture run must pin its input
        run_fixed_state(Scripted({}), tasks, CTX, execution_mode="evaluation")


def test_finding15_results_record_the_researcher_candidate_and_the_neutral_advisor(tmp_path, closed_ok, crit):
    live = PlannerReviewerResearcher(type("Live", (), {"name": "gemini"})(),
                                     RoleModel(provider="gemini", model="gemini-3.1-pro-preview",
                                               endpoint="interactions", thinking_level="high"), Limits())
    ident = component_identity(live)
    assert ident["role_config"]["model"] == "gemini-3.1-pro-preview" and ident["prompt_hash"] == agent.PROMPT_HASH
    assert ident["prompt_version"] == "researcher-v1" and not ident["development_only"]
    fixture = PlannerReviewerResearcher(FixtureProvider(), RoleModel(provider="fixture", model="f"), Limits())
    assert component_identity(fixture)["development_only"]
    fixed = run_fixed_state(Scripted({}), synthetic_tasks(tmp_path), CTX)
    assert fixed["researcher"]["class"].endswith("Scripted")
    assert closed_ok["researcher"]["class"].endswith("OracleResearcher")
    assert closed_ok["advisor"]["class"].endswith("NeutralAdvisor")
    v = qualification_verdict(fixed, closed_ok, crit)
    assert v["researcher"] == fixed["researcher"] and v["advisor"] == closed_ok["advisor"]
    other = {**closed_ok, "researcher": ident}
    with pytest.raises(ValueError, match="different researcher candidates"):
        qualification_verdict(fixed, other, crit)
    tasks = load_closed_loop(QUAL / "closed_loop")[:1]
    with pytest.raises(ValueError, match="neutral general advisor"):
        run_closed_loop(Components(advisor=ProductAdvisor), tasks, 1, tmp_path / "p", Limits(), reference_runs=1)
    stopped = run_closed_loop(Components(researcher=Changing), tasks, 1, tmp_path / "c", Limits(), reference_runs=1)
    assert stopped["researcher"]["class"].endswith("Changing")   # recorded in ledger_dir/identity.json
    with pytest.raises(ValueError, match="one researcher candidate"):   # a resume with another candidate
        run_closed_loop(Components(researcher=lambda: live), tasks, 1, tmp_path / "c", Limits(), reference_runs=1,
                        revalidated=True)


def test_finding11_development_label_comes_from_the_components_not_the_caller(tmp_path, closed_ok, crit):
    class FixtureResearcher(Scripted):
        development_only = True

    fixed = run_fixed_state(FixtureResearcher({}), synthetic_tasks(tmp_path), CTX, execution_mode="evaluation",
                            input_lock=tmp_path / "l.json")
    assert fixed["development_only"]
    closed = {**closed_ok, "execution_mode": "evaluation"}
    v = qualification_verdict(fixed, closed, crit)
    assert v["development_only"] and v["statement"].endswith("contract check only.")


# ---------------------------------------------------------------- closed loop

def test_t08_3_closed_loop_uses_a_fresh_advisor_and_memory_per_episode(tmp_path):
    tasks = load_closed_loop(QUAL / "closed_loop")
    comp = Components()
    out = run_closed_loop(comp, tasks, 2, tmp_path, Limits(), reference_runs=20, seed=7)
    assert len(comp.made) == 6
    assert len({id(m) for _, _, m, _ in comp.made}) == 6 and len({d for *_, d in comp.made}) == 6
    for scope, advisor, memory, _ in comp.made:
        assert [r.scope.episode_id for r in advisor.requests] == [scope.episode_id]
        assert advisor.requests[0].observations == []                   # no initial observations
    for e in out["episodes"]:
        assert (e["outcome"], e["actions_used"], e["consults"], e["finalization"]) == ("success", 2, 1, "done")
    assert {t: r["successes"] for t, r in out["by_type"].items()} == {"single_variable": 2, "interaction": 2,
                                                                     "constrained": 2}
    assert out["development_only"] and "main evaluation keeps in-set memory" in out["memory_policy"]
    ref = out["random_search_reference"]["cl-interaction"]
    again = random_search_reference(next(t for t in tasks if t.id == "cl-interaction"), 20, 7)
    assert ref["seed"] == 7 and ref["runs"] == 20 and ref == again          # seeded, recorded, reproducible
    with pytest.raises(ValueError, match="fresh"):
        run_closed_loop(Components(reuse_memory=True), tasks[:1], 2, tmp_path / "reuse", Limits(), reference_runs=1)
    with pytest.raises(ValueError, match="new version"):                   # a resume cannot change reps or tasks
        run_closed_loop(Components(), tasks, 3, tmp_path, Limits(), reference_runs=1)


def test_t08_3_closed_loop_type_without_success_fails_only_when_complete(tmp_path, closed_ok, crit):
    fixed = run_fixed_state(Scripted({}), synthetic_tasks(tmp_path), CTX)
    assert qualification_verdict(fixed, closed_ok, crit)["verdict"] == "pass"
    closed = json.loads(canonical_json(closed_ok))
    closed["by_type"]["interaction"].update(successes=0)
    v = qualification_verdict(fixed, closed, crit)
    assert v["verdict"] == "fail" and any("interaction" in f for f in v["failures"])
    closed["by_type"]["interaction"].update(complete=1, incomplete=2)
    assert qualification_verdict(fixed, closed, crit)["verdict"] == "incomplete"


def test_finding8_every_type_needs_all_configured_reps_complete(tmp_path, closed_ok, crit):
    fixed = run_fixed_state(Scripted({}), synthetic_tasks(tmp_path), CTX)
    short = oracle_closed_loop(tmp_path / "r1", reps=1)                    # 3 episodes instead of 3 x 3
    v = qualification_verdict(fixed, short, crit)
    assert v["verdict"] == "incomplete" and sum("1 episodes; the rule is 3" in p for p in v["pending"]) == 3
    closed = json.loads(canonical_json(closed_ok))                         # a success, but a silent incomplete
    closed["by_type"]["constrained"].update(complete=2, incomplete=1)
    v = qualification_verdict(fixed, closed, crit)
    assert v["verdict"] == "incomplete" and any("constrained: 1 of 3 episodes incomplete" in p for p in v["pending"])


def test_finding9_a_model_change_stop_is_sticky_until_revalidated(tmp_path):
    class Flaky(OracleResearcher):
        changed = True

        def decide(self, view, ctx):
            if Flaky.changed:
                raise ModelChangedError("gemini-other returned")
            return super().decide(view, ctx)

    tasks = [t for t in load_closed_loop(QUAL / "closed_loop") if t.type == "single_variable"]
    out = run_closed_loop(Components(researcher=Flaky), tasks, 2, tmp_path, Limits(), reference_runs=1)
    assert out["stopped"] == "model_changed" and out["episodes"][0]["outcome_reason"] == "model_changed"
    assert out["episodes"][1]["outcome"] == "not_run"
    Flaky.changed = False                                                   # the provider is back, but ...
    again = run_closed_loop(Components(researcher=Flaky), tasks, 2, tmp_path, Limits(), reference_runs=1)
    assert again["stopped"] == "model_changed: revalidation required"      # ... a plain re-run stays stopped
    assert [e["outcome"] for e in again["episodes"]] == ["infra_incomplete", "not_run"]
    ok = run_closed_loop(Components(researcher=Flaky), tasks, 2, tmp_path, Limits(), reference_runs=1,
                         revalidated=True)
    assert ok["stopped"] is None and [e["outcome"] for e in ok["episodes"]] == ["success", "success"]
