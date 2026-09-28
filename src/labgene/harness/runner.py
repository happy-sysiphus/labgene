"""Episode and set execution (T01; spec §4-6, §11.2, §13.5). Only the harness executes external actions;
external actions of one episode run strictly one at a time."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal

from .. import faults
from ..advisors.base import Advisor
from ..config import Limits, SetPlan
from ..contracts import (MAX_ACTIONS, PROTOCOL_ERROR_LIMIT, ActionEnvelope, ActionKind, ConsultExchange,
                         ConsultRequest, EpisodeState, ExperimentError, FinalizationStatus, InvalidParameters,
                         MemoryScope, Observation, Outcome, ProtocolError, PublicTask, ResearcherView, RunScope,
                         payload_hash)
from ..costs import CallContext, CapExceeded, CostGuard
from ..memory.base import ConditionMemory, FinalizeInput, FinalizeResult, finalize_event_id
from ..providers.base import ModelChangedError
from ..researcher.base import Researcher
from ..simulators.base import SimulatorAdapter, SimulatorInfraError
from ..simulators.validation import validate_parameters
from .ledger import Ledger, now
from .parsing import ActionParseError, parse_action

TERMINAL = (Outcome.success, Outcome.budget_exhausted, Outcome.protocol_error)


class _Incomplete(Exception):
    """Finite infra retries exhausted -> checkpoint as infra_incomplete (never converted to a failure)."""


class EpisodeRunner:
    def __init__(self, ledger: Ledger, scope: RunScope, task: PublicTask, researcher: Researcher, advisor: Advisor,
                 simulator: SimulatorAdapter, memory: ConditionMemory, limits: Limits, guard: CostGuard | None = None,
                 state_hash: Callable[[], str] | None = None):
        if task.task_id != scope.task_id:
            raise ValueError(f"task {task.task_id} does not match scope task {scope.task_id}")
        self.ledger, self.scope, self.task, self.limits, self.guard = ledger, scope, task, limits, guard
        self.researcher, self.advisor, self.simulator, self.memory = researcher, advisor, simulator, memory
        self.eid = scope.episode_id
        self._state_hash = state_hash   # whole condition state (memory + knowledge) when wired; else memory only
        self.attempts = range(1, limits.infra_max_action_retries + 2)   # first try + finite retries

    def state_hash(self) -> str:
        return self._state_hash() if self._state_hash is not None else self.memory.state_hash()

    def _ctx(self, phase: str = "runtime", action_id: str | None = None) -> CallContext:
        return CallContext(sink=self.ledger.record_cost, phase=phase, scope_key=self.scope.memory_scope.key,
                           episode_id=self.eid, action_id=action_id, guard=self.guard)

    def run(self) -> EpisodeState:
        """Start, resume or finalize this episode. Raises ModelChangedError after checkpointing (set must stop)."""
        L, eid = self.ledger, self.eid
        L.start_episode(self.scope, self.state_hash())
        st0 = L.state(eid)
        if st0.outcome not in TERMINAL:
            if st0.outcome is Outcome.infra_incomplete:
                L.episode_event(eid, "resumed", st0.outcome_reason)
            reason = None
            try:
                self._act()
            except _Incomplete as e:
                reason = str(e)
            except CapExceeded:
                reason = "cost_cap"
            except ModelChangedError:
                L.set_outcome(eid, Outcome.infra_incomplete, reason="model_changed")
                L.episode_event(eid, "interrupted", "model_changed")
                raise
            if reason is not None:
                L.set_outcome(eid, Outcome.infra_incomplete, reason=reason)
                L.episode_event(eid, "interrupted", reason)
        st = L.state(eid)
        if st.outcome in TERMINAL and st.finalization is not FinalizationStatus.done:
            self._finalize(st)
            st = L.state(eid)
        return st

    def _act(self) -> None:
        L, eid = self.ledger, self.eid
        L.set_outcome(eid, Outcome.running)   # also resumes an infra_incomplete checkpoint
        while True:
            st = L.state(eid)
            # success before exhaustion: a 50th-action success is a success. Only this episode's observations count.
            n = L.success_actions(eid)
            if n is not None:
                return L.set_outcome(eid, Outcome.success, actions_to_success=n)
            if st.actions_used >= MAX_ACTIONS:
                return L.set_outcome(eid, Outcome.budget_exhausted)
            if st.protocol_streak >= PROTOCOL_ERROR_LIMIT:
                return L.set_outcome(eid, Outcome.protocol_error)
            env = L.pending(eid) or self._decide(st)   # a pending action always runs before any new decision
            if env is not None:
                self._execute(env)

    def _view(self, st: EpisodeState) -> ResearcherView:
        v = ResearcherView(task=self.task, actions_used=st.actions_used, remaining_actions=MAX_ACTIONS - st.actions_used,
                           history=self.ledger.history(self.eid), notes=self.ledger.notes(self.eid))
        # I7: harness ids embed the condition. The researcher sees "a003" for this episode and "e002:a003" for an
        # earlier one (advisors cite memory ids); memory is per condition/set/rep, so these stay unique.
        s = self.scope
        return ResearcherView.model_validate_json(
            v.model_dump_json().replace(f"{self.eid}:", "").replace(f"{s.condition}-r{s.set_rep}-", ""))

    def _decide(self, st: EpisodeState) -> ActionEnvelope | None:
        L, eid = self.ledger, self.eid
        seq, di = L.next_indices(eid)
        view = self._view(st)
        for attempt in self.attempts:
            d = self.researcher.decide(view, self._ctx())
            if d.status != "infra_error":
                break
            L.infra_event(eid, None, "researcher", attempt, d.error)
        else:
            raise _Incomplete("researcher_infra")
        reason, detail = None, d.error or ""
        if d.status != "ok":
            reason = d.status   # provider_incomplete | provider_refusal: a truncated action is never executed
        else:
            try:
                kind, args = parse_action(d.raw_action_text)
            except ActionParseError as e:
                reason, detail = e.reason, e.detail
        if reason:   # one transaction each, note included: a crash never keeps a decision without its note
            L.protocol_event(ProtocolError(scope=self.scope, decision_index=di, raw_text=d.raw_action_text,
                                           reason=reason, detail=detail), d.note, d.analysis)
            return None
        faults.crash_point("before_reserve")
        env = ActionEnvelope(action_id=f"{eid}:a{seq:03d}", scope=self.scope, raw_text=d.raw_action_text,
                             kind=kind, args=args, payload_hash=payload_hash({"kind": kind.value, "args": args}))
        L.reserve(env, seq, di, d.note, d.analysis)
        faults.crash_point("after_reserve")
        return env

    def _execute(self, env: ActionEnvelope) -> None:
        record = self._consult(env) if env.kind is ActionKind.consult else self._experiment(env)
        faults.crash_point("after_execute_before_commit")
        self.ledger.commit(env.action_id, record)
        faults.crash_point("after_commit")

    def _consult(self, env: ActionEnvelope) -> ConsultExchange:
        L, eid = self.ledger, self.eid
        req = ConsultRequest(action_id=env.action_id, scope=self.scope, question=env.args["question"], task=self.task,
                             observations=L.observations(eid), experiment_errors=L.errors(eid),
                             prior_consults=L.consults(eid),
                             remaining_actions=MAX_ACTIONS - L.state(eid).actions_used - 1)
        ctx = self._ctx(action_id=env.action_id)
        deliver_partial = self.limits.partial_advisor_response == "deliver_marked"
        for attempt in self.attempts:   # same action_id every time; nothing is counted until commit
            out = self.advisor.consult(req, ctx)
            if out.response is not None and (out.status == "ok" or (out.status == "partial" and deliver_partial)):
                resp = out.response if out.status == "ok" else out.response.model_copy(update={"status": "partial"})
                return ConsultExchange(action_id=env.action_id, question=req.question, response=resp)
            L.infra_event(eid, env.action_id, "advisor", attempt, out.error or out.status)
        raise _Incomplete("advisor_infra")

    def _experiment(self, env: ActionEnvelope) -> Observation | ExperimentError:
        raw = env.args.get("parameters")
        v = validate_parameters(self.task, raw)
        if isinstance(v, InvalidParameters):
            return ExperimentError(action_id=env.action_id, scope=self.scope, submitted_parameters=raw, reason=v.reason)
        ctx = self._ctx(action_id=env.action_id)
        for attempt in self.attempts:   # deterministic simulator: re-evaluating the same input is safe
            t0 = time.perf_counter()
            try:
                out = self.simulator.evaluate(v.parameters)
            except SimulatorInfraError as e:
                ctx.emit(kind="simulator", role="simulator", model=self.task.simulator_id, status="infra_error",
                         latency_s=time.perf_counter() - t0, attempt=attempt)
                self.ledger.infra_event(self.eid, env.action_id, "simulator", attempt, str(e))
                continue
            ctx.emit(kind="simulator", role="simulator", model=out.simulator_id, latency_s=out.elapsed_s,
                     attempt=attempt, detail={"simulator_version": out.simulator_version, "cache_hit": out.cache_hit})
            return Observation(observation_id=f"obs:{env.action_id}", action_id=env.action_id, scope=self.scope,
                               parameters=v.parameters, results=out.results, units=out.units,
                               simulator_id=out.simulator_id, simulator_version=out.simulator_version,
                               meets_success_criteria=self.task.is_success(out.results), created_at=now())
        raise _Incomplete("simulator_infra")

    def _finalize(self, st: EpisodeState) -> None:
        """Auto-save ALL committed records into this condition's memory under one fixed event id.
        Failure leaves the scientific outcome untouched and finalization=failed."""
        L, eid = self.ledger, self.eid
        data = FinalizeInput(event_id=finalize_event_id(self.scope), scope=self.scope, outcome=st.outcome,
                             observations=L.observations(eid), errors=L.errors(eid), consults=L.consults(eid),
                             infra_notes=L.infra_notes(eid))
        ctx = self._ctx(phase="finalization")
        faults.crash_point("before_finalize")
        reason = "memory_error"
        for attempt in self.attempts:
            try:
                res = self.memory.finalize_episode(data, ctx)
            except ModelChangedError:
                L.set_finalization(eid, FinalizationStatus.failed, reason="model_changed")
                raise
            except Exception as e:   # store/provider failure: retried with the same event id
                res = FinalizeResult(event_id=data.event_id, status="failed", error=f"{type(e).__name__}: {e}")
                reason = "cost_cap" if isinstance(e, CapExceeded) else "memory_error"
            faults.crash_point("after_finalize")
            L.record_finalization(eid, attempt, res)
            if res.status == "done":
                return L.set_finalization(eid, FinalizationStatus.done, self.state_hash())
            if reason == "cost_cap":   # the guard refuses before the call: retrying cannot succeed
                break
        L.set_finalization(eid, FinalizationStatus.failed, reason=reason)


@dataclass
class ConditionComponents:
    researcher: Researcher
    advisor: Advisor
    memory: ConditionMemory
    simulators: dict[str, SimulatorAdapter]
    tasks: dict[str, PublicTask]
    close: Callable[[], None] | None = None     # releases simulators/stores at scope end (memory closed separately)
    state_hash: Callable[[], str] | None = None  # whole condition state hash; default memory.state_hash


@dataclass(frozen=True)
class SetRunStatus:
    """What the CLI reports. Anything but 'complete' means the run stopped and can be resumed."""
    status: Literal["complete", "infra_incomplete", "finalization_failed", "model_changed"]
    episode_id: str | None = None
    reason: str | None = None


class SetRunner:
    """revalidated=True: the operator revalidated after a model change (§10.1); without it the run stays stopped."""

    def __init__(self, ledger: Ledger, run_id: str, plan: SetPlan,
                 factory: Callable[[MemoryScope, bool], ConditionComponents], limits: Limits,
                 guard: CostGuard | None = None, revalidated: bool = False):
        self.ledger, self.run_id, self.plan, self.factory, self.limits, self.guard, self.revalidated = \
            ledger, run_id, plan, factory, limits, guard, revalidated

    def run(self) -> SetRunStatus:
        for rep in range(1, self.plan.reps + 1):
            for condition in self.plan.conditions:
                ms = MemoryScope(run_id=self.run_id, condition=condition, set_id=self.plan.set_id, set_rep=rep)
                # restore the initial snapshot only at a true set start; a resumed set reopens its state
                c = self.factory(ms, not self.ledger.has_episodes(ms))
                try:
                    stop = self._run_set(ms, c)
                finally:
                    c.memory.close()
                    if c.close is not None:
                        c.close()
                if stop:
                    return stop
        return SetRunStatus("complete")

    def _run_set(self, ms: MemoryScope, c: ConditionComponents) -> SetRunStatus | None:
        for order, task_id, visit in self.plan.visits():
            scope = RunScope(**ms.model_dump(), episode_id=f"{ms.condition}-r{ms.set_rep}-e{order:03d}",
                             episode_order=order, task_id=task_id, visit_index=visit)
            L, eid = self.ledger, scope.episode_id
            st = L.state(eid)
            if st and st.outcome in TERMINAL and st.finalization is FinalizationStatus.done:
                continue
            if st and not self.revalidated and "model_changed" in (L.outcome_reason(eid), L.finalization_reason(eid)):
                return SetRunStatus("model_changed", eid, "revalidation required")   # sticky until revalidated
            task = c.tasks[task_id]
            runner = EpisodeRunner(L, scope, task, c.researcher, c.advisor, c.simulators[task.simulator_id],
                                   c.memory, self.limits, self.guard, state_hash=c.state_hash)
            try:
                st = runner.run()
            except ModelChangedError as e:   # requires revalidation before anything else runs
                return SetRunStatus("model_changed", eid, str(e))
            if st.outcome is Outcome.infra_incomplete:
                return SetRunStatus("infra_incomplete", eid, L.outcome_reason(eid))
            if st.finalization is not FinalizationStatus.done:   # never start the next episode before the save
                return SetRunStatus("finalization_failed", eid, L.finalization_reason(eid))
        return None
