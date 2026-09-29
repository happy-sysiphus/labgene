"""Episode ledger (T01; plan §4.3, decisions I5): SQLite at <run_dir>/ledger.sqlite.

Commit boundaries: reserve() and commit() are each one BEGIN IMMEDIATE transaction. An action is
charged exactly when its row turns 'committed' together with its observation/error/consult row.
Counters are never stored: EpisodeState is derived from committed rows.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..contracts import (MAX_ACTIONS, ActionEnvelope, ActionKind, AnalysisResult, ConsultExchange, CostEvent,
                         EpisodeState, ExperimentError, FinalizationStatus, HistoryItem, MemoryScope, Observation,
                         Outcome, ProtocolError, ResearchNote, RunScope, canonical_json, sha256_text)
from ..memory.base import FinalizeResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes(
  episode_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, condition TEXT NOT NULL, set_id TEXT NOT NULL,
  set_rep INTEGER NOT NULL, episode_order INTEGER NOT NULL, task_id TEXT NOT NULL, visit_index INTEGER NOT NULL,
  scope_json TEXT NOT NULL, outcome TEXT NOT NULL DEFAULT 'running', outcome_reason TEXT,
  actions_to_success INTEGER, finalization TEXT NOT NULL DEFAULT 'not_started', finalization_reason TEXT,
  memory_hash_start TEXT, memory_hash_end TEXT, started_at TEXT NOT NULL, ended_at TEXT);
CREATE TABLE IF NOT EXISTS actions(
  action_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
  seq INTEGER NOT NULL, decision_index INTEGER NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('consult','run_experiment')), raw_text TEXT,
  payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('reserved','committed')),
  counted_as TEXT CHECK(counted_as IN ('consultation','evaluation','invalid')),
  result_json TEXT, reserved_at TEXT NOT NULL, committed_at TEXT,
  UNIQUE(episode_id, seq),
  CHECK((status = 'committed') = (counted_as IS NOT NULL AND result_json IS NOT NULL)));
CREATE TABLE IF NOT EXISTS observations(
  observation_id TEXT PRIMARY KEY, action_id TEXT NOT NULL UNIQUE REFERENCES actions(action_id),
  episode_id TEXT NOT NULL REFERENCES episodes(episode_id), meets_success_criteria INTEGER NOT NULL,
  json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiment_errors(
  action_id TEXT PRIMARY KEY REFERENCES actions(action_id),
  episode_id TEXT NOT NULL REFERENCES episodes(episode_id), json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS consult_exchanges(
  action_id TEXT PRIMARY KEY REFERENCES actions(action_id),
  episode_id TEXT NOT NULL REFERENCES episodes(episode_id), json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS protocol_events(
  id INTEGER PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
  decision_index INTEGER NOT NULL, reason TEXT NOT NULL, raw_text TEXT, detail TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notes(
  id INTEGER PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
  decision_index INTEGER NOT NULL, note_json TEXT, analysis_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS action_conflicts(
  id INTEGER PRIMARY KEY, action_id TEXT NOT NULL REFERENCES actions(action_id),
  payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL, raw_text TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS episode_events(
  id INTEGER PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
  kind TEXT NOT NULL CHECK(kind IN ('interrupted','resumed')), detail TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS infra_events(
  id INTEGER PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
  action_id TEXT REFERENCES actions(action_id), stage TEXT NOT NULL, attempt INTEGER NOT NULL,
  error TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cost_events(
  id INTEGER PRIMARY KEY, episode_id TEXT, action_id TEXT, phase TEXT NOT NULL, kind TEXT NOT NULL,
  role TEXT NOT NULL, status TEXT NOT NULL, json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS finalization_events(
  id INTEGER PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(episode_id), event_id TEXT NOT NULL,
  attempt INTEGER NOT NULL, status TEXT NOT NULL, records_added INTEGER, error TEXT, created_at TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS observations_no_update BEFORE UPDATE ON observations
  BEGIN SELECT RAISE(ABORT, 'observations are immutable'); END;
CREATE TRIGGER IF NOT EXISTS observations_no_delete BEFORE DELETE ON observations
  BEGIN SELECT RAISE(ABORT, 'observations are immutable'); END;
CREATE TRIGGER IF NOT EXISTS committed_actions_frozen BEFORE UPDATE ON actions WHEN OLD.status = 'committed'
  BEGIN SELECT RAISE(ABORT, 'committed actions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS scientific_outcome_final BEFORE UPDATE OF outcome ON episodes
  WHEN OLD.outcome IN ('success','budget_exhausted','protocol_error') AND NEW.outcome IS NOT OLD.outcome
  BEGIN SELECT RAISE(ABORT, 'scientific outcome is final'); END;
"""

_COUNTED = {ConsultExchange: ("consultation", "consult_exchanges"), Observation: ("evaluation", "observations"),
            ExperimentError: ("invalid", "experiment_errors")}


class LedgerError(Exception):
    """Budget/pending/commit rule violated. Nothing was charged."""


class ActionConflict(LedgerError):
    """Same action_id, different payload: recorded in action_conflicts, original kept."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, run_dir: str | Path):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "ledger.sqlite"
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=60)   # U26: two condition threads
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        if self._one("PRAGMA foreign_keys") != 1:
            raise LedgerError("SQLite foreign keys are not enforced on this connection")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def _tx(self) -> Iterator[None]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")

    def _one(self, sql: str, *args: Any) -> Any:
        row = self.db.execute(sql, args).fetchone()
        return None if row is None else row[0]

    # ------------------------------------------------------------ episodes

    def start_episode(self, scope: RunScope, memory_hash_start: str | None) -> None:
        """Idempotent: a resumed episode keeps its start hash. Same id with another scope is refused."""
        s = canonical_json(scope)
        with self._tx():
            old = self._one("SELECT scope_json FROM episodes WHERE episode_id=?", scope.episode_id)
            if old is None:
                self.db.execute(
                    "INSERT INTO episodes(episode_id, run_id, condition, set_id, set_rep, episode_order, task_id,"
                    " visit_index, scope_json, memory_hash_start, started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (scope.episode_id, scope.run_id, scope.condition, scope.set_id, scope.set_rep,
                     scope.episode_order, scope.task_id, scope.visit_index, s, memory_hash_start, now()))
            elif old != s:
                raise LedgerError(f"episode {scope.episode_id} already exists with a different scope")

    def has_episodes(self, ms: MemoryScope) -> bool:
        return self._one("SELECT 1 FROM episodes WHERE run_id=? AND condition=? AND set_id=? AND set_rep=? LIMIT 1",
                         ms.run_id, ms.condition, ms.set_id, ms.set_rep) is not None

    def state(self, episode_id: str) -> EpisodeState | None:
        ep = self.db.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        if ep is None:
            return None
        c = dict(self.db.execute("SELECT counted_as, COUNT(*) FROM actions WHERE episode_id=? AND status='committed'"
                                 " GROUP BY counted_as", (episode_id,)).fetchall())
        last = self._one("SELECT COALESCE(MAX(decision_index), 0) FROM actions"
                         " WHERE episode_id=? AND status='committed'", episode_id)
        st = EpisodeState(
            scope=RunScope.model_validate_json(ep["scope_json"]), outcome=ep["outcome"],
            consultation_requests=c.get("consultation", 0), experiment_evaluations=c.get("evaluation", 0),
            invalid_experiment_requests=c.get("invalid", 0),
            protocol_streak=self._one("SELECT COUNT(*) FROM protocol_events WHERE episode_id=? AND decision_index>?",
                                      episode_id, last),
            protocol_errors_total=self._one("SELECT COUNT(*) FROM protocol_events WHERE episode_id=?", episode_id),
            actions_to_success=ep["actions_to_success"],
            pending_action_id=self._one("SELECT action_id FROM actions WHERE episode_id=? AND status='reserved'",
                                        episode_id),
            finalization=ep["finalization"], memory_hash_start=ep["memory_hash_start"],
            memory_hash_end=ep["memory_hash_end"], outcome_reason=ep["outcome_reason"],
            finalization_reason=ep["finalization_reason"])
        if not 0 <= st.actions_used <= MAX_ACTIONS:
            raise LedgerError(f"budget invariant broken for {episode_id}: {st.actions_used}")
        return st

    def outcome_reason(self, episode_id: str) -> str | None:
        return self._one("SELECT outcome_reason FROM episodes WHERE episode_id=?", episode_id)

    def set_outcome(self, episode_id: str, outcome: Outcome, reason: str | None = None,
                    actions_to_success: int | None = None) -> None:
        """A scientific outcome, once set, is never rewritten (DB trigger)."""
        self.db.execute("UPDATE episodes SET outcome=?, outcome_reason=?, actions_to_success=?, ended_at=?"
                        " WHERE episode_id=?", (outcome.value, reason, actions_to_success,
                                                None if outcome is Outcome.running else now(), episode_id))

    def success_actions(self, episode_id: str) -> int | None:
        """actions_used at the first committed successful observation of THIS episode, else None."""
        return self._one(
            "SELECT (SELECT COUNT(*) FROM actions c WHERE c.episode_id=a.episode_id AND c.status='committed'"
            " AND c.seq<=a.seq) FROM observations o JOIN actions a USING(action_id)"
            " WHERE o.episode_id=? AND o.meets_success_criteria=1 ORDER BY a.seq LIMIT 1", episode_id)

    def set_finalization(self, episode_id: str, status: FinalizationStatus, memory_hash_end: str | None = None,
                         reason: str | None = None) -> None:
        self.db.execute("UPDATE episodes SET finalization=?, finalization_reason=?,"
                        " memory_hash_end=COALESCE(?, memory_hash_end) WHERE episode_id=?",
                        (status.value, reason, memory_hash_end, episode_id))

    def finalization_reason(self, episode_id: str) -> str | None:
        """Why the last finalization failed: memory_error | cost_cap | model_changed."""
        return self._one("SELECT finalization_reason FROM episodes WHERE episode_id=?", episode_id)

    def record_finalization(self, episode_id: str, attempt: int, res: FinalizeResult) -> None:
        self.db.execute("INSERT INTO finalization_events(episode_id, event_id, attempt, status, records_added, error,"
                        " created_at) VALUES (?,?,?,?,?,?,?)",
                        (episode_id, res.event_id, attempt, res.status, res.records_added, res.error, now()))

    def finalization_events(self, episode_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM finalization_events WHERE episode_id=? ORDER BY id",
                                                 (episode_id,))]

    # ------------------------------------------------------------ actions

    def next_indices(self, episode_id: str) -> tuple[int, int]:
        """(next action seq, next decision index). Decisions yield one action or one protocol event."""
        n = self._one("SELECT COUNT(*) FROM actions WHERE episode_id=?", episode_id)
        p = self._one("SELECT COUNT(*) FROM protocol_events WHERE episode_id=?", episode_id)
        return n + 1, n + p + 1

    def reserve(self, env: ActionEnvelope, seq: int, decision_index: int, note: ResearchNote | None = None,
                analysis: list[AnalysisResult] = ()) -> str | None:
        """Journal an action, with the note of the decision that chose it, before it executes. Returns None when
        reserved (or already pending with the same payload); the stored result JSON when this exact action was
        already committed (no new charge). Refuses a 51st action and a second pending action in the episode."""
        payload = canonical_json({"kind": env.kind.value, "args": env.args})
        h = sha256_text(payload)
        eid = env.scope.episode_id
        with self._tx():
            old = self.db.execute("SELECT payload_hash, result_json FROM actions WHERE action_id=?",
                                  (env.action_id,)).fetchone()
            if old is not None and old["payload_hash"] == h:
                return old["result_json"]
            if old is None:
                n, pending = self.db.execute("SELECT COUNT(*), SUM(status='reserved')"
                                             " FROM actions WHERE episode_id=?", (eid,)).fetchone()
                if n >= env.scope.action_budget:
                    raise LedgerError(f"{eid}: action budget of {env.scope.action_budget} is used up")
                if pending:
                    raise LedgerError(f"{eid}: another action is still pending")
                self.db.execute("INSERT INTO actions(action_id, episode_id, seq, decision_index, kind, raw_text,"
                                " payload_json, payload_hash, status, reserved_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                (env.action_id, eid, seq, decision_index, env.kind.value, env.raw_text, payload, h,
                                 "reserved", now()))
                self.add_note(eid, decision_index, note, analysis)
                return None
            self.db.execute("INSERT INTO action_conflicts(action_id, payload_json, payload_hash, raw_text, created_at)"
                            " VALUES (?,?,?,?,?)", (env.action_id, payload, h, env.raw_text, now()))
        raise ActionConflict(f"action {env.action_id} is already journaled with a different payload")

    def pending(self, episode_id: str) -> ActionEnvelope | None:
        r = self.db.execute("SELECT a.*, e.scope_json FROM actions a JOIN episodes e USING(episode_id)"
                            " WHERE a.episode_id=? AND a.status='reserved'", (episode_id,)).fetchone()
        if r is None:
            return None
        p = json.loads(r["payload_json"])
        return ActionEnvelope(action_id=r["action_id"], scope=RunScope.model_validate_json(r["scope_json"]),
                              raw_text=r["raw_text"] or "", kind=ActionKind(p["kind"]), args=p["args"],
                              payload_hash=r["payload_hash"])

    def commit(self, action_id: str, record: ConsultExchange | Observation | ExperimentError) -> None:
        """Charge the pending action exactly once, atomically with its result row."""
        counted_as, table = _COUNTED[type(record)]
        body = record.model_dump_json()
        with self._tx():
            row = self.db.execute("SELECT episode_id, kind FROM actions WHERE action_id=? AND status='reserved'",
                                  (action_id,)).fetchone()
            if row is None or record.action_id != action_id \
                    or (row["kind"] == "consult") != isinstance(record, ConsultExchange):
                raise LedgerError(f"cannot commit {action_id}: not pending, or the record does not match the action")
            self.db.execute("UPDATE actions SET status='committed', counted_as=?, result_json=?, committed_at=?"
                            " WHERE action_id=?", (counted_as, body, now(), action_id))
            if isinstance(record, Observation):
                self.db.execute("INSERT INTO observations(observation_id, action_id, episode_id, meets_success_criteria,"
                                " json) VALUES (?,?,?,?,?)", (record.observation_id, action_id, row["episode_id"],
                                                              int(record.meets_success_criteria), body))
            else:
                self.db.execute(f"INSERT INTO {table}(action_id, episode_id, json) VALUES (?,?,?)",
                                (action_id, row["episode_id"], body))

    def _records(self, table: str, model: type, episode_id: str) -> list:
        return [model.model_validate_json(r[0]) for r in self.db.execute(
            f"SELECT t.json FROM {table} t JOIN actions a USING(action_id) WHERE t.episode_id=? ORDER BY a.seq",
            (episode_id,))]

    def observations(self, episode_id: str) -> list[Observation]:
        return self._records("observations", Observation, episode_id)

    def errors(self, episode_id: str) -> list[ExperimentError]:
        return self._records("experiment_errors", ExperimentError, episode_id)

    def consults(self, episode_id: str) -> list[ConsultExchange]:
        return self._records("consult_exchanges", ConsultExchange, episode_id)

    # ------------------------------------------------------------ decisions, events, costs

    def protocol_event(self, err: ProtocolError, note: ResearchNote | None = None,
                       analysis: list[AnalysisResult] = ()) -> None:
        """A protocol error and its decision's note, in one transaction."""
        with self._tx():
            self.db.execute("INSERT INTO protocol_events(episode_id, decision_index, reason, raw_text, detail,"
                            " created_at) VALUES (?,?,?,?,?,?)", (err.scope.episode_id, err.decision_index,
                                                                  err.reason, err.raw_text, err.detail, now()))
            self.add_note(err.scope.episode_id, err.decision_index, note, analysis)

    def add_note(self, episode_id: str, decision_index: int, note: ResearchNote | None,
                 analysis: list[AnalysisResult]) -> None:
        """Researcher notes/analyses (incl. agent_prediction) are kept for audit; never observations.
        Called inside reserve()/protocol_event() so a decision is never journaled without its note."""
        if not (note or analysis):
            return
        self.db.execute("INSERT INTO notes(episode_id, decision_index, note_json, analysis_json, created_at)"
                        " VALUES (?,?,?,?,?)", (episode_id, decision_index, note and note.model_dump_json(),
                                                json.dumps([a.model_dump(mode="json") for a in analysis]), now()))

    def notes(self, episode_id: str) -> list[ResearchNote]:
        return [ResearchNote.model_validate_json(r[0]) for r in self.db.execute(
            "SELECT note_json FROM notes WHERE episode_id=? AND note_json IS NOT NULL ORDER BY id", (episode_id,))]

    def history(self, episode_id: str) -> list[HistoryItem]:
        """Committed actions and protocol errors in decision order, with the exact stored values."""
        items: list[tuple[int, HistoryItem]] = []
        for r in self.db.execute("SELECT decision_index, action_id, counted_as, result_json FROM actions"
                                 " WHERE episode_id=? AND status='committed'", (episode_id,)):
            d = json.loads(r["result_json"])
            if r["counted_as"] == "consultation":
                kind, payload = "consult", {"question": d["question"], "response": d["response"]}
            elif r["counted_as"] == "evaluation":
                kind, payload = "observation", {k: v for k, v in d.items() if k not in ("scope", "created_at")}
            else:
                kind, payload = "invalid_experiment", {"submitted_parameters": d["submitted_parameters"],
                                                       "reason": d["reason"]}
            items.append((r["decision_index"], HistoryItem(kind=kind, action_id=r["action_id"], payload=payload)))
        for r in self.db.execute("SELECT decision_index, reason, detail FROM protocol_events WHERE episode_id=?",
                                 (episode_id,)):
            items.append((r["decision_index"], HistoryItem(kind="protocol_error",
                                                           payload={"reason": r["reason"], "detail": r["detail"]})))
        return [item for _, item in sorted(items, key=lambda x: x[0])]

    def infra_event(self, episode_id: str, action_id: str | None, stage: str, attempt: int, error: str | None) -> None:
        self.db.execute("INSERT INTO infra_events(episode_id, action_id, stage, attempt, error, created_at)"
                        " VALUES (?,?,?,?,?,?)", (episode_id, action_id, stage, attempt, error, now()))

    def episode_event(self, episode_id: str, kind: str, detail: str | None) -> None:
        """Interruption/recovery history (§11.2): kept even after a resumed episode completes."""
        self.db.execute("INSERT INTO episode_events(episode_id, kind, detail, created_at) VALUES (?,?,?,?)",
                        (episode_id, kind, detail, now()))

    def episode_events(self, episode_id: str) -> list[tuple[str, str | None]]:
        return self.db.execute("SELECT kind, detail FROM episode_events WHERE episode_id=? ORDER BY id",
                               (episode_id,)).fetchall()

    def infra_notes(self, episode_id: str) -> list[str]:
        """For memory: which stage failed, never raw error text (worker errors may name private paths)."""
        return [f"{r[0]} infra failure on {r[1] or 'decision'} (attempt {r[2]})" for r in self.db.execute(
            "SELECT stage, action_id, attempt FROM infra_events WHERE episode_id=? ORDER BY id", (episode_id,))]

    def record_cost(self, e: CostEvent) -> None:
        """The CostSink for every CallContext of this run."""
        self.db.execute("INSERT INTO cost_events(episode_id, action_id, phase, kind, role, status, json, created_at)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        (e.episode_id, e.action_id, e.phase, e.kind, e.role, e.status, e.model_dump_json(), now()))

    def cost_events(self, episode_id: str) -> list[CostEvent]:
        return [CostEvent.model_validate_json(r[0]) for r in self.db.execute(
            "SELECT json FROM cost_events WHERE episode_id=? ORDER BY id", (episode_id,))]
