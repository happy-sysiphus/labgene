"""Episode and set execution (T01; spec §4-6, §11.2, §13.5). Only the harness executes external actions;
external actions of one episode run strictly one at a time."""
from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
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
                 state_hash: Callable[[], str] | None = None, stop: threading.Event | None = None,
                 hidden_success: list | tuple = ()):
        if task.task_id != scope.task_id:
            raise ValueError(f"task {task.task_id} does not match scope task {scope.task_id}")
        missing = {h.metric for h in task.hidden_success} - {c.metric for c in hidden_success}
        if missing:   # U44: a declared hidden criterion without its evaluator threshold could never succeed
            raise ValueError(f"task {task.task_id}: no evaluator threshold for hidden criteria {sorted(missing)}")
        self.hidden_success = tuple(hidden_success)   # evaluator-held; never put in a view, error or record
        self.ledger, self.scope, self.task, self.limits, self.guard = ledger, scope, task, limits, guard
        self.researcher, self.advisor, self.simulator, self.memory = researcher, advisor, simulator, memory
        self.eid = scope.episode_id
        self._state_hash = state_hash   # whole condition state (memory + knowledge) when wired; else memory only
        self.stop = stop                # set by SetRunner on Ctrl+C in a parallel set (checked before each decision)
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
            if self.stop is not None and self.stop.is_set():   # leave the episode running: resumable, as Ctrl+C
                raise KeyboardInterrupt("stop requested")
            st = L.state(eid)
            # success before exhaustion: a 50th-action success is a success. Only this episode's observations count.
            n = L.success_actions(eid)
            if n is not None:
                return L.set_outcome(eid, Outcome.success, actions_to_success=n)
            if st.actions_used >= self.scope.action_budget:   # 50, or less for a truncated last attempt (U26)
                return L.set_outcome(eid, Outcome.budget_exhausted)
            if st.protocol_streak >= PROTOCOL_ERROR_LIMIT:
                return L.set_outcome(eid, Outcome.protocol_error)
            env = L.pending(eid) or self._decide(st)   # a pending action always runs before any new decision
            if env is not None:
                self._execute(env)

    def _view(self, st: EpisodeState) -> ResearcherView:
        history = self.ledger.history(self.eid)
        v = ResearcherView(task=self.task, actions_used=st.actions_used,
                           remaining_actions=self.scope.action_budget - st.actions_used,
                           history=history, notes=self.ledger.notes(self.eid),
                           required_action=required_action(self.scope, history))
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
                if view.required_action and kind.value != view.required_action:
                    raise ActionParseError("action_not_allowed", "every experiment follows one consultation: the "
                                                                 f"next action must be {view.required_action}")
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
                             remaining_actions=self.scope.action_budget - L.state(eid).actions_used - 1)
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
                               meets_success_criteria=self.task.is_success(out.results, self.hidden_success),
                               created_at=now())
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
    hidden_success: dict[str, list] = field(default_factory=dict)   # U44: task_id -> evaluator-held thresholds


@dataclass(frozen=True)
class SetRunStatus:
    """What the CLI reports. Anything but 'complete' means the run stopped and can be resumed."""
    status: Literal["complete", "infra_incomplete", "finalization_failed", "model_changed", "protocol_stalled"]
    episode_id: str | None = None
    reason: str | None = None


STALL_LIMIT = 3   # U36


def required_action(scope: RunScope, history: list) -> str | None:
    """U45: with consult_before_experiment, actions alternate consult -> run_experiment (an invalid experiment
    request takes the experiment's turn). None = the researcher chooses freely."""
    if not scope.consult_before_experiment:
        return None
    acts = [h.kind for h in history if h.kind in ("consult", "observation", "invalid_experiment")]
    return "run_experiment" if acts and acts[-1] == "consult" else "consult"


def same_attempt(stored: RunScope, planned: RunScope) -> bool:
    """U40: a finished attempt equals the scope a (longer) plan recomputes for it, except that an attempt that ran
    under a smaller set budget (a truncated last attempt of a shorter plan) keeps the budget it had."""
    return stored == planned or (stored.action_budget < planned.action_budget and
                                 stored.model_copy(update={"action_budget": planned.action_budget}) == planned)


def progression_replay_problems(L: Ledger, plan: SetPlan, ms: MemoryScope) -> list[str]:
    """U40: the stored attempts of a set, replayed under `plan` the way SetRunner._run_progression will replay them;
    a mismatch would stop `resume`, so an extension is refused before it changes anything."""
    tasks, budget = plan.episodes, plan.action_budget
    used, order, k, visits = 0, 0, 0, {}
    while k < len(tasks) and used < budget:
        order += 1
        t = tasks[k]
        visits[t] = visits.get(t, 0) + 1
        scope = RunScope(**ms.model_dump(), episode_id=f"{ms.condition}-r{ms.set_rep}-e{order:03d}",
                         episode_order=order, task_id=t, visit_index=visits[t],
                         action_budget=min(MAX_ACTIONS, budget - used),
                         consult_before_experiment=plan.consult_before_experiment)
        st = L.state(scope.episode_id)
        if st is None:
            return []                    # the plan continues from here
        if not same_attempt(st.scope, scope):
            return [f"{scope.episode_id}: the stored attempt {st.scope.model_dump()} is not what the plan recomputes "
                    f"({scope.model_dump()})"]
        if st.outcome not in TERMINAL or st.finalization is not FinalizationStatus.done:
            return [f"{scope.episode_id} is not finished and saved"]
        used += st.actions_used
        k += st.outcome is Outcome.success
    return []


class SetRunner:
    """revalidated=True: the operator revalidated after a model change (§10.1); without it the run stays stopped."""

    def __init__(self, ledger: Ledger, run_id: str, plan: SetPlan,
                 factory: Callable[[MemoryScope, bool], ConditionComponents], limits: Limits,
                 guard: CostGuard | None = None, revalidated: bool = False):
        self.ledger, self.run_id, self.plan, self.factory, self.limits, self.guard, self.revalidated = \
            ledger, run_id, plan, factory, limits, guard, revalidated
        self._stop = threading.Event()

    def run(self) -> SetRunStatus:
        for rep in range(1, self.plan.reps + 1):
            scopes = [MemoryScope(run_id=self.run_id, condition=c, set_id=self.plan.set_id, set_rep=rep)
                      for c in self.plan.conditions]
            if self.plan.parallel_conditions:
                # U26: one thread per condition, each with its own ledger connection (SQLite serialises the writes);
                # the cost guard is shared and locked. A stop in one condition lets the other finish its set.
                # Ctrl+C reaches only this thread: it sets self._stop and both workers stop at their next decision.
                with ThreadPoolExecutor(len(scopes), thread_name_prefix="labgene-cond") as ex:
                    fs = [ex.submit(self._run_condition_own_ledger, ms) for ms in scopes]
                    try:
                        while wait(fs, timeout=1).not_done:   # a timeout keeps Ctrl+C deliverable on Windows
                            pass
                    except KeyboardInterrupt:
                        self._stop.set()
                        raise
                    stops = [f.result() for f in fs]
            else:
                stops = []
                for ms in scopes:
                    stops.append(self._run_condition(ms, self.ledger))
                    if stops[-1]:
                        break
            stop = next((s for s in stops if s), None)
            if stop:
                return stop
        return SetRunStatus("complete")

    def _run_condition_own_ledger(self, ms: MemoryScope) -> SetRunStatus | None:
        L = Ledger(self.ledger.path.parent)
        try:
            return self._run_condition(ms, L)
        finally:
            L.close()

    def _run_condition(self, ms: MemoryScope, L: Ledger) -> SetRunStatus | None:
        # restore the initial snapshot only at a true set start; a resumed set reopens its state
        c = self.factory(ms, not L.has_episodes(ms))
        try:
            return self._run_progression(ms, c, L) if self.plan.mode == "progression" else self._run_fixed(ms, c, L)
        finally:
            c.memory.close()
            if c.close is not None:
                c.close()

    def _run_fixed(self, ms: MemoryScope, c: ConditionComponents, L: Ledger) -> SetRunStatus | None:
        for order, task_id, visit in self.plan.visits():
            scope = RunScope(**ms.model_dump(), episode_id=f"{ms.condition}-r{ms.set_rep}-e{order:03d}",
                             episode_order=order, task_id=task_id, visit_index=visit,
                             consult_before_experiment=self.plan.consult_before_experiment)
            stop, _ = self._episode(scope, c, L)
            if stop:
                return stop
        return None

    def _run_progression(self, ms: MemoryScope, c: ConditionComponents, L: Ledger) -> SetRunStatus | None:
        """U26: tasks in plan order. A failed attempt (budget exhausted or protocol error) retries the same task in a
        new episode (advisor memory kept); a success moves to the next task. The set ends when every task succeeded
        or plan.action_budget actions are used; each attempt gets min(MAX_ACTIONS, remaining). Every scope follows
        from the outcomes committed before it, so a resume recomputes exactly the stored scopes.
        U36: attempts that end in protocol_error without an action use no budget; after every STALL_LIMIT of them in a
        row on one task this condition stops (protocol_stalled) for inspection; `resume` continues with the next
        attempt (a replayed stalled attempt never stops the set again)."""
        tasks, budget = self.plan.episodes, self.plan.action_budget
        used, order, k, visits, stalled = 0, 0, 0, {}, 0
        while k < len(tasks) and used < budget:
            order += 1
            t = tasks[k]
            visits[t] = visits.get(t, 0) + 1
            scope = RunScope(**ms.model_dump(), episode_id=f"{ms.condition}-r{ms.set_rep}-e{order:03d}",
                             episode_order=order, task_id=t, visit_index=visits[t],
                             action_budget=min(MAX_ACTIONS, budget - used),
                             consult_before_experiment=self.plan.consult_before_experiment)
            prior = L.state(scope.episode_id)
            replayed = prior is not None and prior.outcome in TERMINAL and prior.finalization is FinalizationStatus.done
            if replayed and prior.scope != scope and same_attempt(prior.scope, scope):
                scope = prior.scope        # U40: a finished attempt keeps the (truncated) budget it ran with
            stop, st = self._episode(scope, c, L)
            if stop:
                return stop
            used += st.actions_used
            k += st.outcome is Outcome.success
            stalled = stalled + 1 if st.outcome is Outcome.protocol_error and st.actions_used == 0 else 0
            if stalled and stalled % STALL_LIMIT == 0 and not replayed:
                return SetRunStatus("protocol_stalled", scope.episode_id,
                                    f"{stalled} attempts in a row at {t} ended in protocol errors without an action")
        return None

    def _episode(self, scope: RunScope, c: ConditionComponents,
                 L: Ledger) -> tuple[SetRunStatus | None, EpisodeState | None]:
        """Run (or skip, if already terminal and saved) one episode: (stop status, final state)."""
        eid = scope.episode_id
        st = L.state(eid)
        if st is not None and st.scope != scope:
            raise ValueError(f"ledger episode {eid} does not match the set plan ({st.scope} != {scope})")
        if st and st.outcome in TERMINAL and st.finalization is FinalizationStatus.done:
            return None, st
        if st and not self.revalidated and "model_changed" in (L.outcome_reason(eid), L.finalization_reason(eid)):
            return SetRunStatus("model_changed", eid, "revalidation required"), None   # sticky until revalidated
        task = c.tasks[scope.task_id]
        runner = EpisodeRunner(L, scope, task, c.researcher, c.advisor, c.simulators[task.simulator_id],
                               c.memory, self.limits, self.guard, state_hash=c.state_hash, stop=self._stop,
                               hidden_success=c.hidden_success.get(task.task_id, ()))
        try:
            st = runner.run()
        except ModelChangedError as e:   # requires revalidation before anything else runs
            return SetRunStatus("model_changed", eid, str(e)), None
        if st.outcome is Outcome.infra_incomplete:
            return SetRunStatus("infra_incomplete", eid, L.outcome_reason(eid)), None
        if st.finalization is not FinalizationStatus.done:   # never start the next episode before the save
            return SetRunStatus("finalization_failed", eid, L.finalization_reason(eid)), None
        return None, st
