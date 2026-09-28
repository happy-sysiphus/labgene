"""Product condition memory (T03, §11.3, §13.3, §13.5, decisions I3).

Structured store in <state_dir>/memory.sqlite: observation table (lossless, source=experiment),
experiment errors (no values), consult cases (gated derived text), episode cases, and
experiment-derived relations with provenance "experiment" and applicability conditions.
Written only at finalization (same idempotent core as the baseline memory).
"""
from __future__ import annotations

import json
from typing import Any

from ..contracts import ConsultExchange, ExperimentError, Observation, canonical_json
from .base import FinalizeInput
from .baseline import SqliteMemory, episode_summary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations(
  observation_id TEXT PRIMARY KEY, action_id TEXT NOT NULL, episode_id TEXT NOT NULL, episode_order INTEGER NOT NULL,
  task_id TEXT NOT NULL, task_version TEXT, simulator_id TEXT NOT NULL, simulator_version TEXT NOT NULL,
  parameters TEXT NOT NULL, results TEXT NOT NULL, units TEXT NOT NULL, meets_success_criteria INTEGER NOT NULL,
  visit_index INTEGER NOT NULL, created_at TEXT NOT NULL, source TEXT NOT NULL CHECK (source = 'experiment'),
  record TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiment_errors(
  action_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, episode_order INTEGER NOT NULL, task_id TEXT NOT NULL,
  visit_index INTEGER NOT NULL, reason TEXT NOT NULL, record TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS consult_cases(
  action_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, episode_order INTEGER NOT NULL, task_id TEXT NOT NULL,
  visit_index INTEGER NOT NULL, question TEXT NOT NULL, response TEXT NOT NULL, withheld INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS episode_cases(
  episode_id TEXT PRIMARY KEY, episode_order INTEGER NOT NULL, task_id TEXT NOT NULL, task_version TEXT,
  visit_index INTEGER NOT NULL, summary TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS relations(
  relation_id TEXT PRIMARY KEY, subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
  provenance TEXT NOT NULL CHECK (provenance = 'experiment'), observation_ids TEXT NOT NULL,
  conditions TEXT NOT NULL, episode_id TEXT NOT NULL);
"""


class ProductMemory(SqliteMemory):
    condition = "product"
    schema = _SCHEMA

    def _consult_stored(self, action_id: str) -> bool:
        return self._con.execute("SELECT 1 FROM consult_cases WHERE action_id=?", (action_id,)).fetchone() is not None

    def _rows(self, d: FinalizeInput, withheld: set[str]) -> list[tuple[str, tuple]]:
        s, task = d.scope, self.tasks.get(d.scope.task_id)
        tv = task.version if task else None
        ep = (s.episode_id, s.episode_order, s.task_id)
        rows: list[tuple[str, tuple]] = []
        for o in d.observations:
            rows.append(("INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (o.observation_id, o.action_id, *ep, tv, o.simulator_id, o.simulator_version,
                          canonical_json(o.parameters), canonical_json(o.results), canonical_json(o.units),
                          int(o.meets_success_criteria), s.visit_index, o.created_at, o.source, canonical_json(o))))
            cond = {"task_id": s.task_id, "task_version": tv, "simulator_id": o.simulator_id,
                    "simulator_version": o.simulator_version}
            pred = "met_success_criteria_at" if o.meets_success_criteria else "missed_success_criteria_at"
            rows.append(("INSERT OR IGNORE INTO relations VALUES (?,?,?,?,?,?,?,?)",
                         (f"rel:{o.observation_id}", s.task_id, pred, o.observation_id, "experiment",
                          canonical_json([o.observation_id]), canonical_json(cond), s.episode_id)))
        rows += [("INSERT OR IGNORE INTO experiment_errors VALUES (?,?,?,?,?,?,?)",
                  (e.action_id, *ep, s.visit_index, e.reason, canonical_json(e))) for e in d.errors]
        rows += [("INSERT OR IGNORE INTO consult_cases VALUES (?,?,?,?,?,?,?,?)",
                  (c.action_id, *ep, s.visit_index, c.question, canonical_json(c.response),
                   int(c.action_id in withheld))) for c in d.consults]
        rows.append(("INSERT OR IGNORE INTO episode_cases VALUES (?,?,?,?,?,?)",
                     (s.episode_id, s.episode_order, s.task_id, tv, s.visit_index,
                      canonical_json(episode_summary(d, task)))))
        return rows

    # ------------------------------------------------------------ read API (product advisor)
    def _select(self, table: str, cols: str, task_id: str | None, extra: str = "", args: tuple = ()) -> list[tuple]:
        where = " AND ".join(w for w in ["(? IS NULL OR task_id = ?)", extra] if w)
        return self._con.execute(f"SELECT {cols} FROM {table} WHERE {where} ORDER BY episode_order, rowid",
                                 (task_id, task_id, *args)).fetchall()

    def observations(self, task_id: str | None = None, simulator_version: str | None = None) -> list[Observation]:
        rows = self._select("observations", "record", task_id, "(? IS NULL OR simulator_version = ?)",
                            (simulator_version, simulator_version))
        return [Observation.model_validate_json(r[0]) for r in rows]

    def errors(self, task_id: str | None = None) -> list[ExperimentError]:
        return [ExperimentError.model_validate_json(r[0]) for r in self._select("experiment_errors", "record", task_id)]

    def consult_cases(self, task_id: str | None = None) -> list[dict[str, Any]]:
        """Gate-allowed consult exchanges only; withheld ones are never returned."""
        rows = self._select("consult_cases", "episode_id, task_id, visit_index, action_id, question, response",
                            task_id, "withheld = 0")
        return [{"episode_id": ep, "task_id": t, "visit_index": v,
                 "exchange": ConsultExchange(action_id=a, question=q, response=json.loads(r))}
                for ep, t, v, a, q, r in rows]

    def episode_cases(self, task_id: str | None = None) -> list[dict[str, Any]]:
        rows = self._select("episode_cases", "episode_id, episode_order, task_id, task_version, visit_index, summary",
                            task_id)
        return [{"episode_id": ep, "episode_order": o, "task_id": t, "task_version": tv, "visit_index": v,
                 **json.loads(s)} for ep, o, t, tv, v, s in rows]

    def relations(self, task_id: str | None = None) -> list[dict[str, Any]]:
        rows = self._con.execute(
            "SELECT relation_id, subject, predicate, object, provenance, observation_ids, conditions, episode_id "
            "FROM relations WHERE (? IS NULL OR subject = ?) ORDER BY rowid", (task_id, task_id)).fetchall()
        return [{"relation_id": i, "subject": s, "predicate": p, "object": o, "provenance": pv,
                 "observation_ids": json.loads(ids), "conditions": json.loads(c), "episode_id": ep}
                for i, s, p, o, pv, ids, c, ep in rows]
