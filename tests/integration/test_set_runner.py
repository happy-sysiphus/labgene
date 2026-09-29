"""Set execution: two conditions, revisits, stop/resume rules. Offline fixtures: contract checks only."""
import pytest

from labgene.config import CostCaps, SetPlan, load_set_plan
from labgene.contracts import AdvisorOutcome, AdvisorResponse, FinalizationStatus, Outcome
from labgene.costs import CostGuard
from labgene.harness.ledger import Ledger
from labgene.harness.runner import ConditionComponents, SetRunner, SetRunStatus
from labgene.memory.base import finalize_event_id
from labgene.providers.base import ModelChangedError
from test_episode import (CATALYST, HITS, LIMITS, MISS, POLICY, RIDGE, ROOT, CountingSim, FakeAdvisor, FakeMemory,
                          FakeResearcher, ask)

PLAN = load_set_plan(ROOT / "configs/set_plans/smoke.yaml")   # reps 2, both conditions, ridge/catalyst/ridge


class CitingAdvisor(FakeAdvisor):
    """Cites the newest observation in its condition's memory by its full harness id, as real memory renders it."""

    def __init__(self, condition, memory):
        super().__init__(condition)
        self.memory = memory

    def consult(self, req, ctx):
        self.requests.append(req)
        ctx.emit(kind="llm_call", role=f"advisor_{self.condition}")
        cited = sorted(k for k in self.memory.records if k.startswith("obs:"))[-1:]
        return AdvisorOutcome(status="ok", response=AdvisorResponse(
            answer=f"An earlier run ({', '.join(cited) or 'none'}) looked promising.", cited_observation_ids=cited))


class World:
    """Factory double: one FakeMemory per (condition, set, rep); fresh=True restores it to the empty initial state."""

    def __init__(self, memory_fail=0, advisor_outcomes=None, memory_error=None, cite=False, policy=None):
        self.memory_fail, self.advisor_outcomes = memory_fail, advisor_outcomes if advisor_outcomes is not None else []
        self.memory_error, self.cite = memory_error, cite
        self.policy = policy or (lambda: POLICY)    # factory: one scripted policy per condition/set, kept on resume
        self.policies = {}
        self.memories, self.fresh_calls, self.researchers = {}, [], {}

    def __call__(self, ms, fresh):
        self.fresh_calls.append((ms.condition, ms.set_rep, fresh))
        if fresh:
            self.memories[ms.key] = FakeMemory(ms, fail=self.memory_fail, error=self.memory_error)
            self.memory_fail = 0
        pol = self.policies.setdefault(ms.key, self.policy())
        mem, res = self.memories[ms.key], FakeResearcher(pol)
        self.researchers.setdefault(ms.key, []).append(res)
        advisor = CitingAdvisor(ms.condition, mem) if self.cite else \
            FakeAdvisor(ms.condition, "Same answer.", self.advisor_outcomes)
        return ConditionComponents(researcher=res, memory=mem, advisor=advisor,
                                   simulators={t.simulator_id: CountingSim(t) for t in (RIDGE, CATALYST)},
                                   tasks={t.task_id: t for t in (RIDGE, CATALYST)})


def ids(rep, condition):
    return [f"{condition}-r{rep}-e{o:03d}" for o in (1, 2, 3)]


def test_set_two_conditions_with_revisit_runs_end_to_end(tmp_path):
    L, w = Ledger(tmp_path), World()
    assert SetRunner(L, "run", PLAN, w, LIMITS).run() == SetRunStatus("complete")
    assert w.fresh_calls == [(c, rep, True) for rep in (1, 2) for c in ("baseline", "product")]
    for rep in (1, 2):
        for c in ("baseline", "product"):
            states = [L.state(i) for i in ids(rep, c)]
            assert all((s.outcome, s.finalization, s.actions_to_success, s.actions_used)
                       == (Outcome.success, FinalizationStatus.done, 3, 3) for s in states)
            assert [(s.scope.task_id, s.scope.visit_index) for s in states] == \
                [("fixture_ridge", 1), ("fixture_catalyst", 1), ("fixture_ridge", 2)]
            # memory accumulates inside the set: each episode starts where the previous one's save ended
            assert [s.memory_hash_start for s in states[1:]] == [s.memory_hash_end for s in states[:2]]
            mem = w.memories[f"run/{c}/smoke/rep{rep}"]
            assert len(mem.records) == 9 and mem.closed          # 3 episodes x (1 consult + 2 observations)
            assert {r.scope.condition for r in mem.records.values() if hasattr(r, "scope")} == {c}
            assert {d.event_id for d in mem.calls} == {finalize_event_id(s.scope) for s in states}
        # I7: researcher inputs are byte-identical across conditions (advisors answer identically here)
        views = {c: [v.model_dump_json() for r in w.researchers[f"run/{c}/smoke/rep{rep}"] for v in r.views]
                 for c in ("baseline", "product")}
        assert views["baseline"] == views["product"] and "baseline" not in views["baseline"][-1]


def test_b08_finalization_failure_blocks_next_episode_until_recovered(tmp_path):
    L, w = Ledger(tmp_path), World(memory_fail=99)
    status = SetRunner(L, "run", PLAN, w, LIMITS).run()
    assert status == SetRunStatus("finalization_failed", "baseline-r1-e001", "memory_error")
    st = L.state("baseline-r1-e001")
    assert (st.outcome, st.finalization, st.actions_to_success) == (Outcome.success, FinalizationStatus.failed, 3)
    assert L.state("baseline-r1-e002") is None and L.state("product-r1-e001") is None
    mem = w.memories["run/baseline/smoke/rep1"]
    mem.fail = 0                                           # store recovered
    assert SetRunner(L, "run", PLAN, w, LIMITS).run() == SetRunStatus("complete")
    assert w.fresh_calls[:2] == [("baseline", 1, True), ("baseline", 1, False)]   # resume reopens, never restores
    e1 = [d.event_id for d in mem.calls if d.scope.episode_id == "baseline-r1-e001"]
    assert set(e1) == {"finalize:baseline:smoke:rep1:baseline-r1-e001"} and len(e1) == 4
    assert L.state("baseline-r1-e001").outcome is Outcome.success and len(mem.records) == 9


@pytest.mark.parametrize("outs,status,reason", [
    ([ModelChangedError("returned gemini-x")], "model_changed", "returned gemini-x"),
    ([AdvisorOutcome(status="infra_error", error="503")] * 3, "infra_incomplete", "advisor_infra"),
])
def test_set_runner_stops_on_infra_or_model_change_and_resumes_pending_action(tmp_path, outs, status, reason):
    L, w = Ledger(tmp_path), World(advisor_outcomes=list(outs))
    assert SetRunner(L, "run", PLAN, w, LIMITS).run() == SetRunStatus(status, "baseline-r1-e001", reason)
    st = L.state("baseline-r1-e001")
    assert (st.outcome, st.pending_action_id, st.actions_used) == \
        (Outcome.infra_incomplete, "baseline-r1-e001:a001", 0)
    assert L.outcome_reason("baseline-r1-e001") == ("model_changed" if status == "model_changed" else reason)
    assert L.state("product-r1-e001") is None
    resume = SetRunner(L, "run", PLAN, w, LIMITS, revalidated=status == "model_changed")
    assert resume.run() == SetRunStatus("complete")
    assert w.fresh_calls[1] == ("baseline", 1, False)
    assert L.consults("baseline-r1-e001")[0].action_id == "baseline-r1-e001:a001"
    assert L.state("baseline-r1-e001").actions_used == 3


@pytest.mark.parametrize("where", ["advisor", "finalization"])
def test_b18_model_change_stop_is_sticky_until_revalidated(tmp_path, where):
    """Finding 5: after a model change a plain resume must not continue the set (spec §10.1)."""
    err, eid = ModelChangedError("returned gemini-x"), "baseline-r1-e001"
    w = World(advisor_outcomes=[err]) if where == "advisor" else World(memory_fail=1, memory_error=err)
    L = Ledger(tmp_path)
    assert SetRunner(L, "run", PLAN, w, LIMITS).run() == SetRunStatus("model_changed", eid, "returned gemini-x")
    mem = w.memories["run/baseline/smoke/rep1"]
    decisions = lambda: sum(len(r.views) for r in w.researchers[mem.scope.key])   # noqa: E731
    before = (L.state(eid), decisions(), len(mem.calls))
    for _ in range(2):   # the provider looks fine again, but nothing runs without revalidation
        assert SetRunner(L, "run", PLAN, w, LIMITS).run() == SetRunStatus("model_changed", eid, "revalidation required")
    assert (L.state(eid), decisions(), len(mem.calls)) == before
    assert SetRunner(L, "run", PLAN, w, LIMITS, revalidated=True).run() == SetRunStatus("complete")
    st = L.state(eid)
    assert (st.outcome, st.finalization, st.actions_used, st.actions_to_success) == \
        (Outcome.success, FinalizationStatus.done, 3, 3)


def test_b08_cost_cap_during_finalization_stops_at_once_with_reason(tmp_path):
    """Finding 3: a cap refusal is not a store failure: no pointless retries, and the CLI can tell them apart."""
    L, w, eid = Ledger(tmp_path), World(), "baseline-r1-e001"
    capped = CostGuard(CostCaps(max_calls=0))   # of these fakes only memory asks the guard
    assert SetRunner(L, "run", PLAN, w, LIMITS, capped).run() == SetRunStatus("finalization_failed", eid, "cost_cap")
    st, mem = L.state(eid), w.memories["run/baseline/smoke/rep1"]
    assert (st.outcome, st.actions_to_success, st.finalization) == (Outcome.success, 3, FinalizationStatus.failed)
    assert len(mem.calls) == 1 and not mem.records                  # refused before the call, not retried
    assert [e["status"] for e in L.finalization_events(eid)] == ["failed"]
    assert L.state("baseline-r1-e002") is None                      # next episode blocked
    assert SetRunner(L, "run", PLAN, w, LIMITS).run() == SetRunStatus("complete")   # cap raised by the operator
    assert L.finalization_reason(eid) is None and L.state(eid).finalization is FinalizationStatus.done


def test_i7_advisor_citing_past_episode_ids_does_not_reveal_condition(tmp_path):
    """Finding 4: harness ids embed the condition; advisor citations of earlier episodes must not leak it."""
    w = World(cite=True)
    assert SetRunner(Ledger(tmp_path), "run", PLAN, w, LIMITS).run() == SetRunStatus("complete")
    for rep in (1, 2):
        views = {c: "".join(v.model_dump_json() for r in w.researchers[f"run/{c}/smoke/rep{rep}"] for v in r.views)
                 for c in ("baseline", "product")}
        assert views["baseline"] == views["product"]
        assert "obs:e002:a003" in views["baseline"]                 # the citation still reaches the researcher
        assert "baseline" not in views["baseline"] and "product" not in views["product"]



# ---------------------------------------------------------------- U26: progression sets under a set-wide budget

def prog(budget, tasks=("fixture_ridge", "fixture_catalyst")):
    return SetPlan(set_id="smoke", reps=1, episodes=list(tasks), mode="progression", action_budget=budget)


def episodes(L, cond="baseline"):
    rows = L.db.execute("SELECT episode_id FROM episodes WHERE condition=? ORDER BY episode_order", (cond,))
    return [L.state(r["episode_id"]) for r in rows]


def failing_first(n_fail):
    """Misses every experiment on the first n_fail attempts at fixture_ridge, then solves every task (ask, miss, hit).
    Attempts are counted from the attempt's first decision."""
    seen = {"ridge": 0}

    def policy(v):
        if v.task.task_id == "fixture_ridge" and v.actions_used == 0:
            seen["ridge"] += 1
        if v.task.task_id == "fixture_ridge" and seen["ridge"] <= n_fail:
            return MISS
        return [ask(), MISS, HITS[v.task.task_id]][v.actions_used]
    return policy


def test_u26_progression_moves_on_after_a_success_and_retries_a_failed_task(tmp_path):
    L, w = Ledger(tmp_path), World(policy=lambda: failing_first(1))
    assert SetRunner(L, "run", prog(300), w, LIMITS).run() == SetRunStatus("complete")
    for c in ("baseline", "product"):
        eps = episodes(L, c)
        assert [(e.scope.task_id, e.scope.visit_index, e.scope.action_budget, e.outcome) for e in eps] == [
            ("fixture_ridge", 1, 50, Outcome.budget_exhausted),     # failed attempt: same task again
            ("fixture_ridge", 2, 50, Outcome.success),              # cleared -> next task
            ("fixture_catalyst", 1, 50, Outcome.success)]           # all cleared -> the set ends
        assert sum(e.actions_used for e in eps) == 56 and all(e.finalization is FinalizationStatus.done for e in eps)
        # the retry's advisor sees the failed attempt's records (memory kept inside the set)
        assert eps[1].memory_hash_start == eps[0].memory_hash_end


def test_u26_the_set_budget_truncates_the_last_attempt_and_ends_the_set(tmp_path):
    L, w = Ledger(tmp_path), World(policy=lambda: failing_first(2))
    assert SetRunner(L, "run", prog(60, ("fixture_ridge", "fixture_catalyst")), w, LIMITS).run() ==         SetRunStatus("complete")
    eps = episodes(L)
    assert [(e.scope.visit_index, e.scope.action_budget, e.actions_used, e.outcome) for e in eps] == [
        (1, 50, 50, Outcome.budget_exhausted), (2, 10, 10, Outcome.budget_exhausted)]   # 60 used: stop
    # the truncated attempt's researcher saw its own, smaller budget
    views = [v for r in w.researchers["run/baseline/smoke/rep1"] for v in r.views]
    assert views[50].remaining_actions == 10 and views[-1].remaining_actions == 1


def test_u26_a_truncated_attempt_can_still_clear_its_task(tmp_path):
    L, w = Ledger(tmp_path), World(policy=lambda: failing_first(1))
    assert SetRunner(L, "run", prog(56), w, LIMITS).run() == SetRunStatus("complete")
    assert [(e.scope.task_id, e.scope.action_budget, e.outcome) for e in episodes(L)] == [
        ("fixture_ridge", 50, Outcome.budget_exhausted), ("fixture_ridge", 6, Outcome.success),
        ("fixture_catalyst", 3, Outcome.success)]                  # 50 + 3 used -> 3 left, enough to clear it


def test_u26_protocol_error_termination_is_a_failed_attempt(tmp_path):
    seen = {"n": 0}

    def policy(v):
        if v.actions_used == 0 and not v.history:
            seen["n"] += 1
        return "not an action" if seen["n"] == 1 else [ask(), MISS, HITS[v.task.task_id]][v.actions_used]
    L, w = Ledger(tmp_path), World(policy=lambda: policy)
    assert SetRunner(L, "run", prog(300, ("fixture_ridge",)), w, LIMITS).run() == SetRunStatus("complete")
    assert [(e.scope.visit_index, e.outcome, e.actions_used) for e in episodes(L)] == [
        (1, Outcome.protocol_error, 0), (2, Outcome.success, 3)]


def test_u26_interrupted_progression_resumes_with_the_same_scopes(tmp_path):
    outs = [AdvisorOutcome(status="ok", response=AdvisorResponse(answer="a"))] * 2 +         [AdvisorOutcome(status="infra_error", error="503")] * 3        # fails at the first consult of episode 2
    L, w = Ledger(tmp_path), World(advisor_outcomes=list(outs), policy=lambda: failing_first(1))
    first = SetRunner(L, "run", prog(300), w, LIMITS).run()
    assert first.status == "infra_incomplete"
    stopped = first.episode_id
    assert SetRunner(L, "run", prog(300), w, LIMITS).run() == SetRunStatus("complete")
    assert L.state(stopped).outcome in (Outcome.success, Outcome.budget_exhausted)
    b = episodes(L, "baseline")
    assert [(e.scope.task_id, e.scope.visit_index, e.scope.action_budget) for e in b] == [
        ("fixture_ridge", 1, 50), ("fixture_ridge", 2, 50), ("fixture_catalyst", 1, 50)]


def test_u26_u40_a_smaller_budget_cannot_resume_a_set_but_a_larger_one_continues_it(tmp_path):
    L, w = Ledger(tmp_path), World(policy=lambda: failing_first(1))
    SetRunner(L, "run", prog(56), w, LIMITS).run()          # e002 was a truncated 6-action attempt
    with pytest.raises(ValueError, match="does not match the set plan"):
        SetRunner(L, "run", prog(55), w, LIMITS).run()       # would recompute e002 with a 5-action budget
    # U40: an extension (larger budget) keeps the finished truncated attempt as it ran and continues the set
    assert SetRunner(L, "run", prog(70), w, LIMITS).run() == SetRunStatus("complete")
    eps = episodes(L)
    assert eps[1].scope.action_budget == 6 and len(eps) > 2


def test_u26_progression_plans_are_validated():
    with pytest.raises(ValueError):
        SetPlan(set_id="s", episodes=["a", "a"], mode="progression", action_budget=300)
    with pytest.raises(ValueError):
        SetPlan(set_id="s", episodes=["a"], mode="progression")
    with pytest.raises(ValueError):
        prog(300).visits()


def test_u26_parallel_conditions_run_at_the_same_time_with_the_sequential_results(tmp_path):
    import threading
    meet = threading.Barrier(2, timeout=20)    # both conditions must be inside an episode at once

    def policy():
        p, first = failing_first(1), [True]

        def step(v):
            if first[0]:
                first[0] = False
                meet.wait()
            return p(v)
        return step

    seq_L, par_L = Ledger(tmp_path / "seq"), Ledger(tmp_path / "par")
    assert SetRunner(seq_L, "run", prog(300), World(policy=lambda: failing_first(1)), LIMITS).run() == \
        SetRunStatus("complete")
    plan = prog(300).model_copy(update={"parallel_conditions": True})
    assert SetRunner(par_L, "run", plan, World(policy=policy), LIMITS, guard=CostGuard(CostCaps())).run() == \
        SetRunStatus("complete")
    def key(e):
        return e.scope, e.outcome, e.actions_used, e.finalization
    for c in ("baseline", "product"):
        assert [key(e) for e in episodes(par_L, c)] == [key(e) for e in episodes(seq_L, c)]


def test_u26_ctrl_c_stops_a_parallel_set_at_the_next_decision_and_it_resumes(tmp_path):
    import _thread
    import threading
    import time

    def slow():
        p = failing_first(5)

        def step(v):
            time.sleep(0.01)
            return p(v)
        return step

    L, w = Ledger(tmp_path), World(policy=slow)
    plan = prog(300).model_copy(update={"parallel_conditions": True})
    threading.Timer(0.3, _thread.interrupt_main).start()
    with pytest.raises(KeyboardInterrupt):
        SetRunner(L, "run", plan, w, LIMITS).run()
    n = L.db.execute("SELECT COUNT(*) FROM actions WHERE status='committed'").fetchone()[0]
    assert 0 < n < 200                                            # both sets together would commit ~512
    assert SetRunner(L, "run", plan, w, LIMITS).run() == SetRunStatus("complete")   # the running episodes resume


def test_u36_attempts_stalled_in_protocol_errors_stop_the_condition_and_resume_continues(tmp_path):
    seen = {"n": 0}

    def policy(v):
        if v.actions_used == 0 and not v.history:
            seen["n"] += 1
        return "not an action" if seen["n"] <= 4 else [ask(), MISS, HITS[v.task.task_id]][v.actions_used]
    L, w = Ledger(tmp_path), World(policy=lambda: policy)
    plan = prog(300, ("fixture_ridge",)).model_copy(update={"conditions": ["baseline"]})
    first = SetRunner(L, "run", plan, w, LIMITS).run()
    assert (first.status, first.episode_id) == ("protocol_stalled", "baseline-r1-e003")
    assert SetRunner(L, "run", plan, w, LIMITS).run() == SetRunStatus("complete")    # the operator resumed
    assert [(e.outcome, e.actions_used) for e in episodes(L)] == [(Outcome.protocol_error, 0)] * 4 + [
        (Outcome.success, 3)]
