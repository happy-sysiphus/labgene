"""Episode behaviour at the evaluation-changing boundaries (plan §7). Offline fixtures: contract checks only
(development_only), never evidence of research ability or product value."""
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from labgene import faults
from labgene.config import Limits, load_profile, load_public_task
from labgene.contracts import (AdvisorOutcome, AdvisorResponse, AnalysisResult, CandidateSuggestion,
                               FinalizationStatus, Outcome, ResearcherDecision, ResearchNote, RunScope, Usage,
                               payload_hash)
from labgene.costs import CapExceeded
from labgene.faults import InjectedCrash
from labgene.harness.ledger import Ledger
from labgene.harness.runner import EpisodeRunner
from labgene.memory.base import FinalizeResult, finalize_event_id
from labgene.simulators.base import SimulatorInfraError
from labgene.simulators.fixture import FixtureSimulator

ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_profile(ROOT / "configs/offline.yaml")
RIDGE = load_public_task(PROFILE, "fixture_ridge")
CATALYST = load_public_task(PROFILE, "fixture_catalyst")
LIMITS = Limits(infra_max_action_retries=2)   # 1 try + 2 retries


def exp(**p):
    return json.dumps({"action": "run_experiment", "args": {"hypothesis": "h", "parameters": p}})


def ask(q="Which region of the ranges looks promising?"):
    return json.dumps({"action": "consult", "args": {"question": q}})


MISS = exp(temperature=20, time=1)
HIT = exp(temperature=83, time=37)
HITS = {"fixture_ridge": HIT, "fixture_catalyst": exp(catalyst="C", loading=1.2, temperature=90, residence_time=540)}
POLICY = lambda v: [ask(), MISS, HITS[v.task.task_id]][v.actions_used]   # noqa: E731


class FakeResearcher:
    """Scripted finalizer output as a pure function of the view (so a resume re-asks deterministically)."""

    def __init__(self, policy):
        self.policy, self.views = policy, []

    def decide(self, view, ctx):
        self.views.append(view)
        ctx.emit(kind="llm_call", role="researcher_finalizer")
        out = self.policy(view)
        if isinstance(out, BaseException):
            raise out
        return out if isinstance(out, ResearcherDecision) else ResearcherDecision(raw_action_text=out, status="ok")


class FakeAdvisor:
    """Pops scripted outcomes/exceptions from a (possibly shared) list, then answers normally."""

    def __init__(self, condition="baseline", answer="Try the middle of the ranges.", outcomes=None):
        self.condition, self.answer, self.requests = condition, answer, []
        self.outcomes = outcomes if outcomes is not None else []

    def consult(self, req, ctx):
        self.requests.append(req)
        ctx.emit(kind="llm_call", role=f"advisor_{self.condition}")
        if self.outcomes:
            o = self.outcomes.pop(0)
            if isinstance(o, BaseException):
                raise o
            return o
        return AdvisorOutcome(status="ok", response=AdvisorResponse(answer=self.answer))


class CountingSim(FixtureSimulator):
    def __init__(self, task=RIDGE, fail=0):
        super().__init__(task.simulator_id, task.simulator_version, task.simulator_id.split(".")[1])
        self.calls, self.fail = 0, fail

    def evaluate(self, parameters):
        self.calls += 1
        if self.fail:
            self.fail -= 1
            raise SimulatorInfraError("worker timeout")
        return super().evaluate(parameters)


class FakeMemory:
    """In-memory ConditionMemory, idempotent by event id and natural ids; `fail` = failures (`error`) after a
    partial write. Like a real provider call, it asks the cost guard first (refused before any write)."""

    def __init__(self, scope, fail=0, error=None):
        self.scope, self.fail, self.error = scope, fail, error or RuntimeError("memory store unavailable")
        self.records, self.done, self.calls, self.closed = {}, set(), [], False

    def finalize_episode(self, data, ctx):
        self.calls.append(data)
        if ctx.guard:
            ctx.guard.settle(ctx.guard.reserve("memory-model", 1, 1), Usage())
        ctx.emit(kind="finalize", role="memory")
        before = len(self.records)
        for r in [*data.observations, *data.errors, *data.consults]:
            self.records.setdefault(getattr(r, "observation_id", r.action_id), r)
        if self.fail:
            self.fail -= 1
            raise self.error
        self.done.add(data.event_id)
        return FinalizeResult(event_id=data.event_id, status="done", records_added=len(self.records) - before)

    def is_finalized(self, event_id):
        return event_id in self.done

    def state_hash(self):
        return payload_hash(sorted(self.records))

    def close(self):
        self.closed = True


def scope(order=1, visit=1, task=RIDGE, condition="baseline"):
    return RunScope(run_id="run", condition=condition, set_id="smoke", set_rep=1,
                    episode_id=f"{condition}-r1-e{order:03d}", episode_order=order, task_id=task.task_id,
                    visit_index=visit)


def make(run_dir, policy, *, sc=None, advisor=None, sim=None, memory=None, ledger=None, limits=LIMITS):
    sc = sc or scope()
    return EpisodeRunner(ledger or Ledger(run_dir), sc, RIDGE, FakeResearcher(policy), advisor or FakeAdvisor(),
                         sim or CountingSim(), memory or FakeMemory(sc.memory_scope), limits)


def reopen(r):
    """A new process on the same ledger file with the same (stateless) components."""
    return EpisodeRunner(Ledger(r.ledger.path.parent), r.scope, r.task, r.researcher, r.advisor, r.simulator,
                         r.memory, r.limits)


EID = "baseline-r1-e001"


@pytest.mark.parametrize("last,outcome,sims,consults", [
    (HIT, Outcome.success, 50, 0),
    (MISS, Outcome.budget_exhausted, 50, 0),
    (ask(), Outcome.budget_exhausted, 49, 1),
])
def test_b01_b21_fiftieth_action_decides_and_51st_never_runs(tmp_path, last, outcome, sims, consults):
    r = make(tmp_path, lambda v: MISS if v.actions_used < 49 else last)
    st = r.run()
    assert (st.outcome, st.actions_used, st.consultation_requests) == (outcome, 50, consults)
    assert st.actions_to_success == (50 if outcome is Outcome.success else None)   # B21: never 51, never filled in
    assert (r.simulator.calls, len(r.advisor.requests), len(r.researcher.views)) == (sims, consults, 50)
    if consults:
        assert r.advisor.requests[0].remaining_actions == 0
    assert st.finalization is FinalizationStatus.done and len(r.memory.records) == 50   # every record saved
    again = reopen(r).run()
    assert again == st
    assert (r.simulator.calls, len(r.advisor.requests), len(r.researcher.views)) == (sims, consults, 50)


def test_b02_invalid_experiments_charged_protocol_errors_not(tmp_path):
    script = iter([
        exp(temperature=83),                          # missing parameter -> invalid, 1 action
        exp(temperature=500, time=37),                # out of range -> invalid, 1 action
        "not json", '{"action":"dance"}',             # protocol errors 1, 2
        MISS,                                         # committed action resets the streak
        ResearcherDecision(raw_action_text='{"action":"consult","args":{"quest', status="provider_incomplete"),
        '{"action":"consult","args":{}}',
        ResearcherDecision(raw_action_text=None, status="provider_refusal"),
    ])
    r = make(tmp_path, lambda v: next(script))
    st = r.run()
    assert (st.outcome, st.actions_to_success) == (Outcome.protocol_error, None)
    assert (st.invalid_experiment_requests, st.experiment_evaluations, st.actions_used) == (2, 1, 3)
    assert (st.protocol_errors_total, st.protocol_streak) == (5, 3)
    assert r.simulator.calls == 1 and len(r.ledger.observations(EID)) == 1
    assert [e.submitted_parameters for e in r.ledger.errors(EID)] == [{"temperature": 83},
                                                                      {"temperature": 500, "time": 37}]
    assert [h.kind for h in r.researcher.views[-1].history] == [
        "invalid_experiment", "invalid_experiment", "protocol_error", "protocol_error", "observation",
        "protocol_error", "protocol_error"]
    assert set(r.researcher.views[-1].history[0].payload) == {"submitted_parameters", "reason"}   # no value


def test_b03_new_action_id_same_parameters_is_a_new_experiment(tmp_path):
    r = make(tmp_path, lambda v: MISS if v.actions_used < 2 else HIT)
    st = r.run()
    o1, o2, _ = r.ledger.observations(EID)
    assert o1.action_id != o2.action_id and o1.parameters == o2.parameters and o1.results == o2.results
    assert (st.experiment_evaluations, st.actions_to_success, r.simulator.calls) == (3, 3, 3)


@pytest.mark.parametrize("point,extra", [
    ("before_reserve@2", ("llm_call", "researcher_finalizer")),       # decision lost; researcher asked again
    ("after_reserve@2", None),                                        # pending action runs on resume, once
    ("after_execute_before_commit@1", ("llm_call", "advisor_baseline")),   # consult physically re-run
    ("after_execute_before_commit@2", ("simulator", "simulator")),        # deterministic re-evaluation
    ("after_commit@3", None),                                         # success recovered from committed rows
    ("before_finalize", None),
    ("after_finalize", ("finalize", "memory")),                       # memory saved, ledger not yet told
])
def test_b04_forced_crash_then_resume_restores_exact_state(tmp_path, monkeypatch, point, extra):
    ref = make(tmp_path / "ref", POLICY)
    ref_st = ref.run()
    monkeypatch.setenv("LABGENE_FAULT", point)
    faults.reset()
    r = make(tmp_path / "run", POLICY)
    with pytest.raises(InjectedCrash):
        r.run()
    monkeypatch.delenv("LABGENE_FAULT")
    resumed = reopen(r)
    st = resumed.run()
    assert st == ref_st and st.pending_action_id is None
    assert (st.outcome, st.actions_to_success, st.consultation_requests, st.experiment_evaluations) == \
        (Outcome.success, 3, 1, 2)
    key = lambda o: (o.observation_id, o.parameters, o.results)   # noqa: E731
    assert [key(o) for o in resumed.ledger.observations(EID)] == [key(o) for o in ref.ledger.observations(EID)]
    assert sorted(r.memory.records) == sorted(ref.memory.records)            # no duplicate records
    assert {d.event_id for d in r.memory.calls} == {finalize_event_id(r.scope)}
    costs = lambda L: Counter((e.kind, e.role) for e in L.cost_events(EID))   # noqa: E731
    assert costs(resumed.ledger) - costs(ref.ledger) == (Counter([extra]) if extra else Counter())


@pytest.mark.parametrize("first", [ask(), "not json"])   # the decision becomes an action / a protocol error
def test_b04_crash_while_journaling_a_decision_keeps_its_note(tmp_path, monkeypatch, first):
    """Finding 2: a kill between journaling a decision and its note must not lose the note forever."""
    policy = lambda v: ResearcherDecision(raw_action_text=POLICY(v) if v.history else first,   # noqa: E731
                                          status="ok", note=ResearchNote(hypotheses=[f"h{len(v.history)}"]))
    ref = make(tmp_path / "ref", policy)
    ref_st = ref.run()
    add_note, calls = Ledger.add_note, []

    def crash_on_first_note(self, *a):
        calls.append(a)
        if len(calls) == 1:
            raise InjectedCrash("killed while journaling the first decision")
        return add_note(self, *a)

    monkeypatch.setattr(Ledger, "add_note", crash_on_first_note)
    r = make(tmp_path / "run", policy)
    with pytest.raises(InjectedCrash):
        r.run()
    resumed = reopen(r)
    assert resumed.run() == ref_st and ref_st.outcome is Outcome.success
    assert resumed.ledger.notes(EID) == ref.ledger.notes(EID)
    assert r.researcher.views[-1] == ref.researcher.views[-1]


@pytest.mark.parametrize("advisor_out,sim_fail,reason,pending", [
    ([AdvisorOutcome(status="infra_error", error="503")] * 3, 0, "advisor_infra", "a001"),
    ([CapExceeded("max_usd")], 0, "cost_cap", "a001"),
    ([], 3, "simulator_infra", "a002"),
])
def test_b04_infra_retries_same_action_then_checkpoint_and_resume(tmp_path, advisor_out, sim_fail, reason, pending):
    r = make(tmp_path, POLICY, advisor=FakeAdvisor(outcomes=list(advisor_out)), sim=CountingSim(fail=sim_fail))
    st = r.run()
    assert (st.outcome, r.ledger.outcome_reason(EID), st.pending_action_id) == \
        (Outcome.infra_incomplete, reason, f"{EID}:{pending}")
    assert st.finalization is FinalizationStatus.not_started and not r.memory.calls   # not finalized
    decisions = len(r.researcher.views)
    st = reopen(r).run()
    assert (st.outcome, st.actions_used, st.consultation_requests, st.experiment_evaluations) == \
        (Outcome.success, 3, 1, 2)
    assert len(r.researcher.views) == decisions + 3 - int(pending[-1])   # pending action ran before any new decision
    tried = Counter(e.action_id for e in r.ledger.cost_events(EID) if e.kind in ("llm_call", "simulator")
                    and e.role != "researcher_finalizer")
    assert tried[f"{EID}:{pending}"] == len(advisor_out) + sim_fail + 1      # every physical attempt kept


@pytest.mark.parametrize("policy,delivered", [("retry_as_infra", False), ("deliver_marked", True)])
def test_b04_partial_advisor_response_follows_profile(tmp_path, policy, delivered):
    partial = AdvisorOutcome(status="partial", response=AdvisorResponse(answer="Half an ans"))
    r = make(tmp_path, POLICY, advisor=FakeAdvisor(outcomes=[partial]),
             limits=Limits(infra_max_action_retries=2, partial_advisor_response=policy))
    assert r.run().outcome is Outcome.success
    (c,) = r.ledger.consults(EID)
    assert (c.response.status == "partial") == delivered and len(r.advisor.requests) == (1 if delivered else 2)


def test_b05_consult_streaks_and_experiment_only_runs(tmp_path):
    r = make(tmp_path / "c", lambda v: ask(f"q{v.actions_used}") if v.actions_used < 5 else HIT)
    st = r.run()
    assert (st.outcome, st.consultation_requests, st.actions_to_success) == (Outcome.success, 5, 6)
    assert [len(q.prior_consults) for q in r.advisor.requests] == [0, 1, 2, 3, 4]
    assert [q.remaining_actions for q in r.advisor.requests] == [49, 48, 47, 46, 45]
    e = make(tmp_path / "e", lambda v: MISS if v.actions_used < 3 else HIT)
    st = e.run()
    assert (st.consultation_requests, st.actions_to_success, e.advisor.requests) == (0, 4, [])


def test_b05_only_the_finalizer_action_executes(tmp_path):
    reviewer = ResearchNote(hypotheses=["optimum near 83/37"], next_action_reason=f"reviewer proposes {HIT}")
    first = ResearcherDecision(raw_action_text=ask(), status="ok", note=reviewer)
    r = make(tmp_path, lambda v: [first, MISS, HIT][v.actions_used])
    st = r.run()
    assert [o.parameters for o in r.ledger.observations(EID)] == [{"temperature": 20.0, "time": 1.0},
                                                                  {"temperature": 83.0, "time": 37.0}]
    assert (st.consultation_requests, st.experiment_evaluations, r.simulator.calls) == (1, 2, 2)
    assert r.researcher.views[1].notes == [reviewer]


def test_b06_success_only_from_an_observation_committed_in_this_episode(tmp_path):
    L, mem = Ledger(tmp_path), FakeMemory(scope().memory_scope)
    first = make(tmp_path, lambda v: HIT, ledger=L, memory=mem)
    assert first.run().outcome is Outcome.success                              # same set, earlier episode
    claim = AdvisorResponse(answer="Target met: yield 95 % at 83 degC / 37 min. Success.",
                            candidates=[CandidateSuggestion(parameters={"temperature": 83, "time": 37})])
    pred = AnalysisResult(analysis_id="p1", input_observation_ids=[], method="extrapolation",
                          result={"yield": 99.0}, kind="agent_prediction")
    script = iter([ask("Did we already succeed?"),
                   ResearcherDecision(raw_action_text=MISS, status="ok", analysis=[pred],
                                      note=ResearchNote(hypotheses=["success already achieved"])),
                   '{"action":"declare_success","args":{"yield":99}}', '{"action":"success"}', '{"action":"done"}'])
    second = make(tmp_path, lambda v: next(script), ledger=L, memory=mem, sc=scope(2, visit=2),
                  advisor=FakeAdvisor(outcomes=[AdvisorOutcome(status="ok", response=claim)]))
    st = second.run()
    assert (st.outcome, st.actions_to_success) == (Outcome.protocol_error, None)
    assert [o.meets_success_criteria for o in L.observations("baseline-r1-e002")] == [False]
    for sql in ("UPDATE observations SET meets_success_criteria=1", "DELETE FROM observations"):
        with pytest.raises(sqlite3.IntegrityError):
            L.db.execute(sql)


def test_b08_finalize_failure_keeps_outcome_and_retries_same_event(tmp_path):
    mem = FakeMemory(scope().memory_scope, fail=3)      # partial write, then failure, on every attempt
    r = make(tmp_path, lambda v: MISS if v.actions_used < 1 else HIT, memory=mem)
    st = r.run()
    assert (st.outcome, st.finalization, st.actions_to_success) == (Outcome.success, FinalizationStatus.failed, 2)
    st = reopen(r).run()
    assert (st.outcome, st.finalization, st.actions_to_success) == (Outcome.success, FinalizationStatus.done, 2)
    assert [d.event_id for d in mem.calls] == [finalize_event_id(r.scope)] * 4
    assert len(mem.records) == 2 and st.memory_hash_end == mem.state_hash() != st.memory_hash_start
    assert [e["status"] for e in r.ledger.finalization_events(EID)] == ["failed"] * 3 + ["done"]
    assert r.ledger.state(EID).actions_used == 2                              # saving is not an action


def test_b09_new_episode_starts_with_empty_researcher_state(tmp_path):
    L, mem = Ledger(tmp_path), FakeMemory(scope().memory_scope)
    noted = lambda v: ResearcherDecision(raw_action_text=POLICY(v), status="ok",   # noqa: E731
                                         note=ResearchNote(hypotheses=[f"h{v.actions_used}"]))
    r1 = make(tmp_path, noted, ledger=L, memory=mem)
    r1.run()
    assert mem.records and L.notes(EID)
    r2 = make(tmp_path, noted, ledger=L, memory=mem, sc=scope(2, visit=2))
    r2.run()
    v = r2.researcher.views[0]
    assert (v.actions_used, v.remaining_actions, v.history, v.notes) == (0, 50, [], [])
    assert v == r1.researcher.views[0]
    assert r2.advisor.requests[0].observations == [] and r2.advisor.requests[0].prior_consults == []
