"""Baseline condition memory (T03, §4.1, §11.3, §13.5, decisions I3).

Raw text records of this condition's own episodes in <state_dir>/memory.sqlite, written only at
finalization, keyed by natural ids (obs:<observation_id>, err:<action_id>, consult:<action_id>,
outcome:<episode_id>). render() gives the full chronological text or, over the limit, a cached
summary; the raw log is never rewritten.

SqliteMemory is the shared finalization core (also used by product.py).
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from ..config import Limits, RoleModel
from ..contracts import (GateStatus, MemoryScope, Observation, Outcome, ProviderStatus,
                         PublicTask, canonical_json, payload_hash, sha256_text)
from ..costs import CallContext
from ..faults import crash_point
from ..providers.base import GenerationRequest, LLMProvider
from ..providers.call import call_llm
from .base import FinalizeInput, FinalizeResult, finalize_event_id
from .snapshot import sqlite_logical_hash

GateDerived = Callable[[str, dict, CallContext], GateStatus]
_TERMINAL = (Outcome.success, Outcome.budget_exhausted, Outcome.protocol_error)   # infra_incomplete is resumable

_COMMON_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS finalization_events(
  event_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, input_hash TEXT NOT NULL,
  records_added INTEGER NOT NULL, finalized_at TEXT NOT NULL);
"""


def best_observation(observations: list[Observation], task: PublicTask | None) -> Observation | None:
    """First observation meeting the public success rule; else (task known) the smallest normalized
    shortfall to the public targets; else None (direction unknown)."""
    for o in observations:
        if o.meets_success_criteria:
            return o
    if task is None or not task.success or not observations:
        return None

    def shortfall(o: Observation) -> float:
        total = 0.0
        for c in task.success:
            v = o.results.get(c.metric)
            if v is None or not math.isfinite(v):
                return math.inf
            gap = (c.target - c.tolerance) - v if c.direction == "maximize" else v - (c.target + c.tolerance)
            total += max(0.0, gap) / max(abs(c.target), 1.0)
        return total

    return min(observations, key=shortfall)


def episode_summary(d: FinalizeInput, task: PublicTask | None) -> dict[str, Any]:
    best = best_observation(d.observations, task)
    return {"outcome": d.outcome.value,
            "consultation_requests": len(d.consults), "experiment_evaluations": len(d.observations),
            "invalid_experiment_requests": len(d.errors),
            "actions_used": len(d.consults) + len(d.observations) + len(d.errors),
            "best_observation_id": best.observation_id if best else None,
            "best_results": best.results if best else None,
            "infra_notes": list(d.infra_notes)}


class SqliteMemory:
    """Finalization core: one transaction per episode, natural-id INSERT OR IGNORE, fixed event id."""
    condition: str
    schema: str

    def __init__(self, state_dir: str | Path, scope: MemoryScope, gate_derived: GateDerived | None = None,
                 tasks: Mapping[str, PublicTask] | None = None, limits: Limits = Limits()):
        if scope.condition != self.condition:
            raise ValueError(f"{type(self).__name__} serves the {self.condition} condition only")
        if gate_derived is None and not limits.development_only:
            raise ValueError("a derived-text gate is required once limits are frozen (no ungated consult answers)")
        self.scope, self.gate_derived, self.tasks, self.limits = scope, gate_derived, dict(tasks or {}), limits
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.state_dir / "memory.sqlite", isolation_level=None)
        self._con.executescript(_COMMON_SCHEMA)
        # bind at open (atomic, first opener wins): a dir serves one MemoryScope even before any finalize
        self._con.execute("INSERT OR IGNORE INTO meta VALUES ('scope', ?)", (scope.key,))
        if self._con.execute("SELECT value FROM meta WHERE key='scope'").fetchone()[0] != scope.key:
            self._con.close()
            raise ValueError("state_dir belongs to another memory scope")
        self._con.executescript(self.schema)

    # ------------------------------------------------------------ subclass hooks
    def _rows(self, d: FinalizeInput, withheld: set[str]) -> list[tuple[str, tuple]]:
        raise NotImplementedError

    def _consult_stored(self, action_id: str) -> bool:
        raise NotImplementedError

    # ------------------------------------------------------------ ConditionMemory
    def is_finalized(self, event_id: str) -> bool:
        return self._con.execute("SELECT 1 FROM finalization_events WHERE event_id=?", (event_id,)).fetchone() is not None

    def state_hash(self) -> str:
        """Memory CONTENT only: the scope-binding row is excluded so equal content hashes equal across set reps
        (B10: every rep starts from the same initial memory). Snapshots still hash every table."""
        return sqlite_logical_hash(self._con, exclude=("meta",))

    def close(self) -> None:
        self._con.close()

    def finalize_episode(self, data: FinalizeInput, ctx: CallContext) -> FinalizeResult:
        self._check(data)
        t0 = time.monotonic()
        fctx = ctx.child(phase="finalization", episode_id=data.scope.episode_id, action_id=None)
        in_hash = payload_hash(data)
        added = skipped = 0
        try:
            row = self._con.execute("SELECT input_hash FROM finalization_events WHERE event_id=?",
                                    (data.event_id,)).fetchone()
            if row and row[0] != in_hash:
                raise ValueError("finalize event conflict: same event_id, different payload (existing kept)")
            if not row:
                withheld = self._gate_consults(data, fctx)
                rows = self._rows(data, withheld)
                self._con.execute("BEGIN IMMEDIATE")
                try:
                    added = sum(self._con.execute(sql, params).rowcount for sql, params in rows)
                    crash_point("during_finalize")
                    self._con.execute("INSERT INTO finalization_events VALUES (?,?,?,?,?)",
                                      (data.event_id, data.scope.episode_id, in_hash, added,
                                       datetime.now(timezone.utc).isoformat()))
                    self._con.execute("COMMIT")
                except BaseException:
                    if self._con.in_transaction:   # sqlite may already have rolled back (BUSY/FULL/IOERR)
                        self._con.execute("ROLLBACK")
                    raise
                skipped = len(rows) - added
            res = FinalizeResult(event_id=data.event_id, status="done", records_added=added,
                                 records_skipped_duplicate=skipped, memory_hash=self.state_hash())
        except Exception as e:   # outcome is the harness's; memory only reports failed
            res = FinalizeResult(event_id=data.event_id, status="failed", error=f"{type(e).__name__}: {e}")
        fctx.emit(kind="finalize", role=f"memory_{self.condition}", phase="finalization", action_id=None,
                  status=res.status, latency_s=time.monotonic() - t0,
                  detail={"event_id": data.event_id, "records_added": res.records_added,
                          "records_skipped_duplicate": res.records_skipped_duplicate})
        return res

    # ------------------------------------------------------------ internals
    def _check(self, d: FinalizeInput) -> None:
        """Trust boundary: only this scope's own episode, terminal outcome, fixed event id."""
        if d.scope.memory_scope != self.scope:
            raise ValueError("finalize input scope does not match this memory scope")
        if d.event_id != finalize_event_id(d.scope):
            raise ValueError("finalize event_id must be finalize_event_id(scope)")
        if d.outcome not in _TERMINAL:
            raise ValueError(f"only a terminal episode is finalized, not {d.outcome.value}")
        if any(x.scope != d.scope for x in [*d.observations, *d.errors]):
            raise ValueError("observation/error from another episode scope")

    def _gate_consults(self, d: FinalizeInput, ctx: CallContext) -> set[str]:
        """Consult answers are derived text: gate each not-yet-stored one. Non-allow -> withheld."""
        withheld = set()
        for c in d.consults:
            if self.gate_derived is None or self._consult_stored(c.action_id):
                continue   # ponytail: no gate configured = stored as delivered; live wiring must pass one
            try:
                status = self.gate_derived(canonical_json(c), {"kind": "consult_answer", "action_id": c.action_id,
                                           "episode_id": d.scope.episode_id, "task_id": d.scope.task_id,
                                           "scope_key": self.scope.key}, ctx)
            except Exception:
                status = GateStatus.error
            if status != GateStatus.allow:
                withheld.add(c.action_id)
        return withheld


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


_CUT = " [...truncated]"


def _clip(text: str, n: int) -> str:
    """Over n chars: cut at the last whitespace (never inside a value) and mark the cut."""
    if len(text) <= n:
        return text
    keep = max(n - len(_CUT), 0)
    cut = max(text.rfind(" ", 0, keep + 1), text.rfind("\n", 0, keep + 1), 0)
    return (text[:cut].rstrip() + _CUT)[:n]


def _action_order(d: FinalizeInput) -> dict[str, int] | None:
    """Commit order from harness-issued ids '<episode_id>:a<seq>' (runner.py); None if any id lacks it.
    ponytail: FinalizeInput has no cross-kind order field (contract change requested); else grouped by kind."""
    pat = re.compile(re.escape(d.scope.episode_id) + r":a(\d+)")
    found = {x.action_id: pat.fullmatch(x.action_id) for x in (*d.consults, *d.observations, *d.errors)}
    return {a: int(m.group(1)) for a, m in found.items()} if all(found.values()) else None


class SummarizerUnavailable(Exception):
    """Summary could not be produced or was withheld by the gate. Never replaced by a fixture."""


SUMMARY_PROMPT_VERSION = "baseline-summary-v1"
SUMMARY_PROMPT = (
    "You condense a research advisor's own past records (experiments, invalid requests, consultations, "
    "episode outcomes). Keep exact numeric values, units, parameter settings and observation IDs for the "
    "best and most informative results of every task; keep misses and invalid-request reasons; never invent "
    "or round values. Plain text, at most {max_chars} characters.")
DIGEST_ID = "deterministic-digest-v1"   # development_only fallback


class BaselineTextMemory(SqliteMemory):
    condition = "baseline"
    schema = """
CREATE TABLE IF NOT EXISTS records(
  seq INTEGER PRIMARY KEY, record_id TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
  episode_id TEXT NOT NULL, episode_order INTEGER NOT NULL, task_id TEXT NOT NULL, visit_index INTEGER NOT NULL,
  withheld INTEGER NOT NULL DEFAULT 0, body TEXT NOT NULL);
"""

    def __init__(self, state_dir: str | Path, scope: MemoryScope, summarizer: LLMProvider | None = None,
                 limits: Limits = Limits(), *, summarizer_role: RoleModel | None = None,
                 gate_derived: GateDerived | None = None, tasks: Mapping[str, PublicTask] | None = None):
        if (summarizer is None) != (summarizer_role is None):
            raise ValueError("summarizer and summarizer_role go together")
        if summarizer is None and not limits.development_only:
            raise ValueError("a summarizer is required once limits are frozen (the digest is development_only)")
        super().__init__(state_dir, scope, gate_derived, tasks, limits)
        self.summarizer, self.summarizer_role = summarizer, summarizer_role

    def _consult_stored(self, action_id: str) -> bool:
        return self._con.execute("SELECT 1 FROM records WHERE record_id=?", (f"consult:{action_id}",)).fetchone() is not None

    def _rows(self, d: FinalizeInput, withheld: set[str]) -> list[tuple[str, tuple]]:
        s = d.scope
        sql = ("INSERT OR IGNORE INTO records(record_id, kind, episode_id, episode_order, task_id, visit_index, "
               "withheld, body) VALUES (?,?,?,?,?,?,?,?)")
        at = (s.episode_id, s.episode_order, s.task_id, s.visit_index)
        recs = [(c.action_id, (f"consult:{c.action_id}", "consult", *at, int(c.action_id in withheld),
                               canonical_json(c))) for c in d.consults]
        recs += [(o.action_id, (f"obs:{o.observation_id}", "observation", *at, 0, canonical_json(o)))
                 for o in d.observations]
        recs += [(e.action_id, (f"err:{e.action_id}", "error", *at, 0, canonical_json(e))) for e in d.errors]
        if order := _action_order(d):   # seq = insertion order = render order
            recs.sort(key=lambda r: order[r[0]])
        rows = [(sql, params) for _, params in recs]
        rows.append((sql, (f"outcome:{s.episode_id}", "outcome", *at, 0,
                           canonical_json(episode_summary(d, self.tasks.get(s.task_id))))))
        return rows

    # ------------------------------------------------------------ rendering
    def observations(self, task_id: str | None = None) -> list[Observation]:
        """Stored (non-withheld) observations of this condition/set, in commit order; exact values."""
        return [o for kind, _e, _o, task, *_r, body in self._visible() if kind == "observation"
                and (task_id is None or task == task_id) for o in [Observation.model_validate_json(body)]]

    def _visible(self) -> list[tuple]:
        return self._con.execute("SELECT kind, episode_id, episode_order, task_id, visit_index, body FROM records "
                                 "WHERE withheld=0 ORDER BY seq").fetchall()

    @staticmethod
    def _line(kind: str, b: dict) -> str:
        j = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True)   # noqa: E731
        if kind == "observation":
            return (f"- experiment {b['action_id']} -> observation {b['observation_id']}: parameters {j(b['parameters'])}"
                    f" results {j(b['results'])} units {j(b['units'])} meets_success_criteria={b['meets_success_criteria']}")
        if kind == "error":
            return (f"- invalid request {b['action_id']} (no measured value): {b['reason']}; "
                    f"submitted {j(b['submitted_parameters'])}")
        if kind == "consult":
            r = b["response"]
            cand = f"\n  candidates: {j(r['candidates'])}" if r.get("candidates") else ""
            return f"- consult {b['action_id']}: Q: {b['question']}\n  A: {r['answer']}{cand}"
        notes = f"; infra notes: {j(b['infra_notes'])}" if b["infra_notes"] else ""
        return (f"- outcome {b['outcome']}: actions used {b['actions_used']} (consults {b['consultation_requests']}, "
                f"experiments {b['experiment_evaluations']}, invalid {b['invalid_experiment_requests']}){notes}")

    def _blocks(self) -> list[tuple[str, str]]:
        """(episode header, record text) per visible record, chronological."""
        return [(f"## episode {order} | task {task} | visit {visit}", self._line(kind, json.loads(body)))
                for kind, _ep, order, task, visit, body in self._visible()]

    @staticmethod
    def _join(blocks: list[tuple[str, str]]) -> str:
        out, last = [], None
        for head, line in blocks:
            if head != last:
                out.append(head)
                last = head
            out.append(line)
        return "\n".join(out)

    def render(self, max_chars: int | None = None, ctx: CallContext | None = None) -> str:
        """Full chronological text if it fits, else a cached summary. Raw records are never modified."""
        max_chars = max_chars or self.limits.baseline_memory_max_chars
        full = self._join(self._blocks())
        if len(full) <= max_chars:
            return full
        if ctx is None:
            raise ValueError("summarization needs a CallContext for cost accounting")
        sid = f"{self.summarizer.name}:{self.summarizer_role.model}" if self.summarizer else DIGEST_ID
        key = payload_hash({"records": sha256_text(full), "summarizer": sid, "prompt": SUMMARY_PROMPT_VERSION,
                            "max_chars": max_chars})
        path = self.state_dir / "cache" / f"baseline_summary_{key}.json"
        if path.exists():
            text = json.loads(path.read_text(encoding="utf-8"))["text"]
        else:
            text = self._summarize(full, max_chars, ctx) if self.summarizer else self._digest(max_chars, ctx)
        if self.summarizer and self.gate_derived:   # generated text: gate on every use (policy may change)
            try:
                status = self.gate_derived(text, {"kind": "memory_summary", "scope_key": self.scope.key}, ctx)
            except Exception:
                status = GateStatus.error
            if status != GateStatus.allow:
                raise SummarizerUnavailable("memory summary unavailable")
        if not path.exists():
            _atomic_write(path, json.dumps({"text": text, "summarizer": sid, "prompt_version": SUMMARY_PROMPT_VERSION,
                                            "development_only": self.summarizer is None}, ensure_ascii=False))
        return text

    def _summarize(self, full: str, max_chars: int, ctx: CallContext) -> str:
        role = self.summarizer_role
        req = GenerationRequest(role="internal_summarizer", model=role.model,
                                system_instruction=SUMMARY_PROMPT.format(max_chars=max_chars),
                                input=[{"role": "user", "text": full}], thinking_level=role.thinking_level,
                                reasoning_effort=role.reasoning_effort,
                                max_output_tokens=role.max_output_tokens or max(256, max_chars // 2))
        # costed + finitely retried; another returned model raises ModelChangedError (the set must stop)
        res = call_llm(self.summarizer, req, ctx, self.limits, role.allowed_returned_models)
        if res.status != ProviderStatus.ok or not res.text:
            raise SummarizerUnavailable(f"summarizer returned {res.status.value}")
        return _clip(res.text, max_chars)

    def _digest(self, max_chars: int, ctx: CallContext) -> str:
        """development_only deterministic fallback: per-task best results + exact most recent records."""
        t0 = time.monotonic()
        by_task: dict[str, list[Observation]] = {}
        for kind, *_, body in self._visible():
            if kind == "observation":
                o = Observation.model_validate_json(body)
                by_task.setdefault(o.scope.task_id, []).append(o)
        best = []
        for task_id, obs in by_task.items():
            b = best_observation(obs, self.tasks.get(task_id))
            best.append(f"- {task_id}: {len(obs)} observations; " + (
                f"best {b.observation_id} results {json.dumps(b.results, sort_keys=True)} "
                f"meets_success_criteria={b.meets_success_criteria}" if b else "no observation met the success rule"))
        blocks = self._blocks()
        head = "\n".join(["[development_only deterministic digest: older records omitted, recent ones verbatim]",
                          "Per-task best results:", *best, "Most recent records (verbatim):"])
        used, k = len(head), 0      # take the longest suffix of records (with episode headers) that fits
        for i in range(len(blocks) - 1, -1, -1):
            h, line = blocks[i]
            cost = len(line) + 1 + (len(h) + 1 if i == len(blocks) - 1 or blocks[i + 1][0] != h else 0)
            if used + cost > max_chars:
                break
            used, k = used + cost, k + 1
        recent = blocks[len(blocks) - k:] if k else []
        ctx.emit(kind="other", role="internal_summarizer", provider="deterministic", model=DIGEST_ID,
                 latency_s=time.monotonic() - t0,
                 detail={"purpose": "baseline_memory_summary", "development_only": True,
                         "records_shown": k, "records_total": len(blocks)})
        return _clip("\n".join([head, self._join(recent)]) if recent else head, max_chars)
