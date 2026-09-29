"""Read-only view of a finished harness run: its consultations, the exact ConsultRequest each advisor received, the
pre-declared sample (spec §3) and whether the run's sets reached their end. Never writes to the run directory."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from labgene.config import SetPlan
from labgene.contracts import ConsultExchange, ConsultRequest, ExperimentError, Observation, PublicTask, RunScope

TERMINAL = ("success", "budget_exhausted", "protocol_error")


def open_ledger(run_dir: str | Path) -> sqlite3.Connection:
    """The run's ledger, read-only (a file: URI, so a non-ASCII path is percent-encoded)."""
    path = (Path(run_dir) / "ledger.sqlite").resolve()
    if not path.exists():
        raise FileNotFoundError(f"no ledger in {run_dir}")
    con = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


@dataclass(frozen=True)
class ConsultRecord:
    action_id: str
    episode_id: str
    condition: str
    task_id: str
    episode_order: int
    seq: int
    exchange: ConsultExchange

    @property
    def ok(self) -> bool:
        """A complete answer; partial answers are only counted (spec §3.1)."""
        return self.exchange.response.status == "complete"


def consult_records(con: sqlite3.Connection) -> list[ConsultRecord]:
    """Every committed consultation, ordered by condition, set rep, episode order and action sequence."""
    rows = con.execute(
        "SELECT c.action_id, c.json, a.seq, e.episode_id, e.condition, e.task_id, e.episode_order "
        "FROM consult_exchanges c JOIN actions a ON a.action_id = c.action_id "
        "JOIN episodes e ON e.episode_id = c.episode_id WHERE a.status = 'committed' "
        "ORDER BY e.condition, e.set_rep, e.episode_order, a.seq").fetchall()
    return [ConsultRecord(r["action_id"], r["episode_id"], r["condition"], r["task_id"], r["episode_order"], r["seq"],
                          ConsultExchange.model_validate_json(r["json"])) for r in rows]


def pick(n: int, m: int) -> list[int]:
    """m of the positions 0..n-1, evenly spread: round(i*(n-1)/(m-1)) with halves rounded up (spec §3.2)."""
    if n <= 0 or m <= 0:
        return []
    if m == 1:
        return [0]
    return [(2 * i * (n - 1) + (m - 1)) // (2 * (m - 1)) for i in range(m)]


def draw_sample(records: list[ConsultRecord], per_task: int) -> list[str]:
    """The judge sample (spec §3.2): per (condition, task) at most `per_task` complete consults, evenly spread over the
    task's attempts in time order. Deterministic: the same ledger always gives the same sample."""
    groups: dict[tuple[str, str], list[ConsultRecord]] = {}
    for r in records:
        if r.ok:
            groups.setdefault((r.condition, r.task_id), []).append(r)
    out: list[str] = []
    for key in sorted(groups):
        g = groups[key]
        out += [g[i].action_id for i in pick(len(g), min(per_task, len(g)))]
    return out


def consult_request(con: sqlite3.Connection, rec: ConsultRecord, task: PublicTask) -> ConsultRequest:
    """What EpisodeRunner._consult built for this consultation: the episode's committed records before it."""
    scope = RunScope.model_validate_json(con.execute("SELECT scope_json FROM episodes WHERE episode_id=?",
                                                     (rec.episode_id,)).fetchone()[0])

    def before(table: str, model: type) -> list:
        return [model.model_validate_json(r[0]) for r in con.execute(
            f"SELECT t.json FROM {table} t JOIN actions a ON a.action_id = t.action_id "
            "WHERE t.episode_id=? AND a.seq<? AND a.status='committed' ORDER BY a.seq", (rec.episode_id, rec.seq))]

    used = con.execute("SELECT COUNT(*) FROM actions WHERE episode_id=? AND status='committed' AND seq<?",
                       (rec.episode_id, rec.seq)).fetchone()[0]
    return ConsultRequest(action_id=rec.action_id, scope=scope, question=rec.exchange.question, task=task,
                          observations=before("observations", Observation),
                          experiment_errors=before("experiment_errors", ExperimentError),
                          prior_consults=before("consult_exchanges", ConsultExchange),
                          remaining_actions=scope.action_budget - used - 1)


def episode_orders(con: sqlite3.Connection, condition: str) -> dict[str, int]:
    return {r[0]: r[1] for r in con.execute("SELECT episode_id, episode_order FROM episodes WHERE condition=?",
                                            (condition,))}


def set_finished(con: sqlite3.Connection, plan: SetPlan) -> bool:
    """True when every started episode is terminal and saved and each condition's set reached its end
    (progression: every task cleared or the action budget used; fixed: every planned episode ran)."""
    if con.execute(f"SELECT 1 FROM episodes WHERE outcome NOT IN ({','.join('?' * len(TERMINAL))}) "
                   "OR finalization != 'done' LIMIT 1", TERMINAL).fetchone():
        return False
    for cond in plan.conditions:
        for rep in range(1, plan.reps + 1):
            eps = con.execute("SELECT task_id, outcome FROM episodes WHERE condition=? AND set_id=? AND set_rep=?",
                              (cond, plan.set_id, rep)).fetchall()
            if plan.mode == "fixed":
                if len(eps) < len(plan.episodes):
                    return False
                continue
            used = con.execute("SELECT COUNT(*) FROM actions a JOIN episodes e ON e.episode_id = a.episode_id "
                               "WHERE e.condition=? AND e.set_id=? AND e.set_rep=? AND a.status='committed'",
                               (cond, plan.set_id, rep)).fetchone()[0]
            cleared = {e["task_id"] for e in eps if e["outcome"] == "success"}
            if not (set(plan.episodes) <= cleared or used >= (plan.action_budget or 0)):
                return False
    return True
