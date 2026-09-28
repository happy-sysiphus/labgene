"""Set execution: two conditions, revisits, stop/resume rules. Offline fixtures: contract checks only."""
import pytest

from labgene.config import CostCaps, load_set_plan
from labgene.contracts import AdvisorOutcome, AdvisorResponse, FinalizationStatus, Outcome
from labgene.costs import CostGuard
from labgene.harness.ledger import Ledger
from labgene.harness.runner import ConditionComponents, SetRunner, SetRunStatus
from labgene.memory.base import finalize_event_id
from labgene.providers.base import ModelChangedError
from test_episode import CATALYST, LIMITS, POLICY, RIDGE, ROOT, CountingSim, FakeAdvisor, FakeMemory, FakeResearcher

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

    def __init__(self, memory_fail=0, advisor_outcomes=None, memory_error=None, cite=False):
        self.memory_fail, self.advisor_outcomes = memory_fail, advisor_outcomes if advisor_outcomes is not None else []
        self.memory_error, self.cite = memory_error, cite
        self.memories, self.fresh_calls, self.researchers = {}, [], {}

    def __call__(self, ms, fresh):
        self.fresh_calls.append((ms.condition, ms.set_rep, fresh))
        if fresh:
            self.memories[ms.key] = FakeMemory(ms, fail=self.memory_fail, error=self.memory_error)
            self.memory_fail = 0
        mem, res = self.memories[ms.key], FakeResearcher(POLICY)
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
