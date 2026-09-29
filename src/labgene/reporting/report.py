"""Public run report (T07 results; spec §8.1-8.3, §11.2; plan T09 default analysis; B21-B23).

build_report reads a private temp COPY of <run_dir>/ledger.sqlite (the run's ledger is never opened, so never
written) plus prebuild_costs.jsonl / sessions.jsonl when present, and returns a public projection built from
whitelisted fields only. A final scan refuses any output whose harness-authored strings contain private content
(answer bundles, secret markers, blocked-document identities, hidden asset paths, the private dir, API keys).
Offline fixture numbers are contract checks, never research or product value.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, get_args

import yaml

from ..config import CREDENTIAL_ENV, Paths, SetPlan, repo_root_for
from ..contracts import CONDITIONS, CostEvent, EpisodeState, Observation, PrivateTaskAssets, PublicTask, payload_hash
from ..harness.ledger import Ledger
from ..harness.runner import TERMINAL as _TERMINAL

REPORT_VERSION = "2"   # 2: model check vs config, analysis_reps, privacy_scan, banner reasons, CSV tag columns
BOOTSTRAP_SEED = 20260928
BOOTSTRAP_B = 10_000
TERMINAL = tuple(o.value for o in _TERMINAL)
PHASES = get_args(CostEvent.model_fields["phase"].annotation)
TOKENS = ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens")
# Reason vocabularies written by harness.runner. Anything else is exported as "other", never as raw text.
INFRA_REASONS = {"model_changed", "cost_cap", "advisor_infra", "simulator_infra", "researcher_infra"}
FINALIZATION_REASONS = {"memory_error", "cost_cap", "model_changed"}
SECRET_ENV = (*CREDENTIAL_ENV.values(), "LABGENE_SEARCH_API_KEY")
MANIFEST_PUBLIC = ("run_id", "created_at", "execution_mode", "contract_version", "spec_sha256", "code_version",
                   "profile_hash", "set_plan_hash", "set_plan", "tasks", "initial_state_hashes", "models", "prompts",
                   "analysis_plan_hash", "frozen", "development_only_fields", "evidence_hashes")

EPISODE_COLS = ("run_id", "condition", "set_id", "set_rep", "episode_order", "episode_id", "task_id", "visit_index",
                "visit", "outcome", "outcome_reason", "success", "actions_used", "actions_to_success",
                "consultation_requests", "experiment_attempts", "experiment_evaluations", "invalid_experiment_requests",
                "protocol_errors_total", "finalization", "finalization_reason", "interruptions", "resumes",
                "interruption_reasons", "memory_hash_start", "memory_hash_end", "action_budget")
ACTION_COLS = ("run_id", "condition", "set_id", "set_rep", "episode_id", "episode_order", "task_id", "visit_index",
               "action_number", "action_id", "kind", "counted_as", "parameters", "results", "invalid_reason",
               "meets_success_criteria", "best_so_far_results", "best_so_far_gap", "best_so_far_action_id")
COST_COLS = ("phase", "condition", "kind", "role", "events", "retries", "statuses", "latency_s",
             *(f"{t}_{k}" for t in TOKENS for k in ("sum", "reported", "unavailable")))

DEFINITIONS = {
    "success_at_50": "successes / terminal episodes (success, budget_exhausted, protocol_error) of one "
                     "(condition, set_rep); infra_incomplete / running / not_started episodes are never counted as "
                     "failures; always shown with n",
    "complete": "a set (condition, set_rep) is complete only if every planned episode is terminal AND finalized "
                "(finalization=done); incomplete sets are listed with their reasons, never dropped",
    "actions_to_success": "actions_used at the first successful observation; empty for every non-success (never 51 "
                          "or max+1); summaries cover successful episodes only and carry the success rate and n",
    "running": "episode interrupted mid-run (resumable checkpoint); not terminal",
    "finalization_failed": "terminal episode whose end-of-episode memory save is not done (failed or not yet run); "
                           "its scientific outcome is kept",
    "best_so_far": "among this episode's valid observations up to and including this action, the one with the "
                   "smallest gap = max over the task's success criteria of shortfall/|threshold| (threshold = "
                   "target - tolerance for maximize, target + tolerance for minimize; shortfall = distance on the "
                   "wrong side of the threshold, else 0; |threshold| 0 -> 1). gap 0 = every criterion met. Ties keep "
                   "the earlier observation. Single-metric tasks: the best metric value until success",
    "paired_unit": "episodes within a set share condition memory and are NOT independent; the unit of analysis is "
                   "the set pair (baseline and product at the same set_rep). Only set reps complete in both "
                   "conditions are paired",
    "physical_attempts": "every CostEvent is one physical operation/attempt; retries = events with attempt > 1",
    "progression_set": "U26: tasks in plan order; an attempt that fails (budget exhausted or protocol error) is "
                       "retried as a new episode of the same task (advisor memory kept), a success moves to the next "
                       "task; the set ends when every task is cleared or the set action budget is used; the last "
                       "attempt may get fewer than 50 actions (truncated, counted in truncated_attempts and in "
                       "success_at_50 with its own budget)",
    "progression_primary": "U27: per (condition, set_rep), more tasks cleared within the set action budget is "
                           "better; tie -> fewer total actions used. With fewer than 2 complete set pairs the "
                           "comparison is descriptive (no interval, no significance claim)",
    "actions_to_clear": "actions used by the failed attempts at a task plus actions_to_success of the attempt that "
                        "cleared it; empty for a task not cleared",
    "tokens": "sum over REPORTED values only; counted over model calls only (llm_call: all fields; embedding: "
              "input_tokens), so simulator/gate/retrieval/search/finalize events are never 'unavailable'; "
              "'unavailable' = model calls whose endpoint did not report the value (never counted as 0); sum is "
              "null when nothing was reported",
    "budget_actions": "only committed researcher actions (runtime) consume the 50-action budget; prebuild, "
                      "finalization and evaluation_support never do",
    "model_check": "per llm_call/embedding event: 'mismatch' if the recorded or provider-returned model is outside "
                   "the role's configured model + allowed_returned_models (manifest.models; researcher_* -> "
                   "researcher; roles without a config: returned vs recorded); 'unverified' if the event carries no "
                   "provider-returned model (e.g. calls through knowledge.gate.guarded_generate record one model "
                   "string only) or the provider cannot report the model it ran (model_verified=false: the Codex "
                   "CLI records its pinned -m); else 'ok'",
    "analysis_reps": "set reps complete in EVERY planned condition; condition means and progression use only these, "
                     "so the conditions are never compared over different rep sets",
    "privacy_scan": "second layer after whitelisted fields: harness-authored strings are scanned for private content "
                    "(the researcher's own parameters/results are not suppressed, spec §7.3); tasks_missing lists "
                    "tasks whose answer bundle was not found, so their bundle strings were not scanned",
}


def _snapshot(run_dir: Path, tmp: Path) -> Ledger | None:
    """Ledger over a private copy of the run's ledger (+ journal). The run's file is never opened, so never written;
    a hot journal left by a hard kill is rolled back in the copy (mode=ro cannot), giving the last committed state."""
    if not (run_dir / "ledger.sqlite").exists():
        return None
    # ponytail: the copy is not atomic against a live writer; report after the run stops (the CLI does)
    for name in ("ledger.sqlite", "ledger.sqlite-journal"):
        if (run_dir / name).exists():
            shutil.copyfile(run_dir / name, tmp / name)
    return Ledger(tmp)


# ---------------------------------------------------------------- public API

def build_report(run_dir: Path, manifest: dict) -> dict:
    """Public projection of one run. Raises ValueError if the ledger and manifest disagree or if any private
    content would be exposed."""
    run_dir = Path(run_dir)
    root = _root(run_dir)
    plan = SetPlan.model_validate(manifest["set_plan"])
    run_id = manifest["run_id"]
    tasks, task_source = _task_defs(manifest, root, plan)
    with tempfile.TemporaryDirectory() as tmp:
        L = _snapshot(run_dir, Path(tmp))
        try:
            episodes = _episode_rows(L, plan, run_id)
            actions = [a for r in episodes if r["episode_id"] for a in _action_rows(L, r, tasks[r["task_id"]])]
            ledger_costs = [CostEvent.model_validate_json(j) for (j,) in
                            L.db.execute("SELECT json FROM cost_events ORDER BY id")] if L else []
        finally:
            if L is not None:
                L.close()
    sets = [_set_metrics([r for r in episodes if (r["condition"], r["set_rep"]) == (c, rep)], plan, (c, rep))
            for rep in range(1, plan.reps + 1) for c in plan.conditions]
    pre = run_dir / "prebuild_costs.jsonl"
    prebuild = [CostEvent.model_validate_json(ln) for ln in pre.read_text(encoding="utf-8").splitlines()
                if ln.strip()] if pre.exists() else None
    forbidden, missing = _forbidden(manifest, root, tasks)
    rep = {
        "report_version": REPORT_VERSION,
        "banner": _banner(manifest, plan, all(s["complete"] for s in sets)),
        "privacy_scan": {"tasks_checked": sorted(set(tasks) - set(missing)), "tasks_missing": missing},
        "definitions": DEFINITIONS,
        "reproducibility": {**{k: manifest[k] for k in MANIFEST_PUBLIC if k in manifest},
                            "profile_id": (manifest.get("profile") or {}).get("profile_id"),
                            "task_defs_source": task_source},
        "sets": sets,
        "conditions": _conditions(sets, plan),
        "paired_comparison": _paired(sets, plan),
        "costs": _costs(ledger_costs, prebuild, episodes, _sessions(run_dir / "sessions.jsonl"),
                        manifest.get("models") or {}),
        "episodes": episodes,
        "actions": actions,
    }
    _assert_public(rep, forbidden)
    return rep


def write_report(run_dir: Path, manifest: dict, out_dir: Path | None = None) -> dict[str, Path]:
    """Write report.json, episodes.csv, actions.csv, costs.csv, report.md (default <run_dir>/report).
    Nothing is written when build_report refuses."""
    rep = build_report(run_dir, manifest)
    tag = {"execution_mode": rep["banner"]["execution_mode"], "report_label": rep["banner"]["label"]}   # on every row
    texts = {"report.json": json.dumps(rep, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
             "episodes.csv": _csv(rep["episodes"], EPISODE_COLS, tag),
             "actions.csv": _csv(rep["actions"], ACTION_COLS, tag),
             "costs.csv": _csv([_flat_cost(g) for g in rep["costs"]["groups"]], COST_COLS, tag),
             "report.md": _markdown(rep)}
    out = Path(out_dir) if out_dir is not None else Path(run_dir) / "report"
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, text in texts.items():
        paths[name] = out / name
        paths[name].write_text(text, encoding="utf-8", newline="")
    return paths


# ---------------------------------------------------------------- inputs

def _root(run_dir: Path) -> Path:
    """Repo root that relative profile paths resolve against (like app/config); else this checkout's root."""
    return next((c for c in [run_dir.resolve(), *run_dir.resolve().parents] if (c / "pyproject.toml").exists()),
                repo_root_for(Path(__file__)))


def _paths(manifest: dict) -> Paths:
    return Paths.model_validate((manifest.get("profile") or {}).get("paths", {}))


def _resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


def _task_defs(manifest: dict, root: Path, plan: SetPlan) -> tuple[dict[str, PublicTask], str]:
    """manifest["task_defs"] (PublicTask dumps) first; else configs via profile.paths.tasks_dir. Either way the
    definitions must match the run's recorded task hashes."""
    defs, source = manifest.get("task_defs"), "manifest.task_defs"
    if not defs:
        d = _resolve(root, _paths(manifest).tasks_dir)
        defs = {t: yaml.safe_load((d / f"{t}.yaml").read_text(encoding="utf-8")) for t in dict.fromkeys(plan.episodes)}
        source = "profile.paths.tasks_dir (manifest has no task_defs)"
    tasks = {t: PublicTask.model_validate(v) for t, v in defs.items()}
    changed = sorted(t for t, h in (manifest.get("tasks") or {}).items() if t in tasks and payload_hash(tasks[t]) != h)
    missing = sorted(set(plan.episodes) - set(tasks))
    if changed or missing:
        raise ValueError(f"task definitions do not match the run (changed: {changed}, missing: {missing})")
    return tasks, source


def _sessions(p: Path) -> dict[str, Any]:
    lines = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] if p.exists() else []
    return {"sessions": len(lines), "wall_s_total": round(sum(x.get("wall_s", 0.0) for x in lines), 3) if lines else None,
            "code_changed_during_run": any(x.get("code_changed") for x in lines)}


# ---------------------------------------------------------------- episodes and actions

def _vocab(value: str | None, allowed: set[str]) -> str | None:
    return None if value is None else value if value in allowed else "other"


def _episode_rows(L: Ledger | None, plan: SetPlan, run_id: str) -> list[dict[str, Any]]:
    """One row per PLANNED episode (not_started included). Ledger episodes outside the plan are an error."""
    if plan.mode == "progression":
        return _progression_rows(L, plan, run_id)
    found = {} if L is None else {
        (r["run_id"], r["set_id"], r["condition"], r["set_rep"], r["episode_order"]): r["episode_id"]
        for r in L.db.execute("SELECT episode_id, run_id, condition, set_id, set_rep, episode_order FROM episodes")}
    rows = []
    for rep in range(1, plan.reps + 1):
        for cond in plan.conditions:
            for order, task_id, visit in plan.visits():
                eid = found.pop((run_id, plan.set_id, cond, rep, order), None)
                st = L.state(eid) if eid else None
                if st is not None and (st.scope.task_id, st.scope.visit_index) != (task_id, visit):
                    raise ValueError(f"ledger episode {eid} does not match the set plan")
                rows.append(_episode_row(run_id, plan.set_id, cond, rep, order, task_id, visit, st,
                                         L.episode_events(eid) if eid else []))
    if found:
        raise ValueError(f"ledger has episodes outside this run's set plan: {sorted(found.values())}")
    return rows


def _progression_rows(L: Ledger | None, plan: SetPlan, run_id: str) -> list[dict[str, Any]]:
    """Progression sets (U26): the episodes the ledger holds, in order. Each must have the scope the progression
    rule derives from the outcomes before it (task, visit, budget), as the runner enforces; a non-terminal episode
    can only be the last one. Anything else (or an episode outside this run's plan) is an error."""
    if L is None:
        return []
    by: dict[tuple[str, int], list[str]] = defaultdict(list)
    for r in L.db.execute("SELECT episode_id, run_id, set_id, condition, set_rep FROM episodes ORDER BY episode_order"):
        if (r["run_id"], r["set_id"]) != (run_id, plan.set_id) or r["condition"] not in plan.conditions \
                or not 1 <= r["set_rep"] <= plan.reps:
            raise ValueError(f"ledger has episodes outside this run's set plan: {r['episode_id']}")
        by[(r["condition"], r["set_rep"])].append(r["episode_id"])
    rows = []
    for rep in range(1, plan.reps + 1):
        for cond in plan.conditions:
            used, k, visits, open_ = 0, 0, Counter(), False
            for n, eid in enumerate(by.pop((cond, rep), []), start=1):
                st = L.state(eid)
                if open_ or k >= len(plan.episodes) or used >= plan.action_budget:
                    raise ValueError(f"ledger episode {eid} does not follow the progression set plan")
                t = plan.episodes[k]
                visits[t] += 1
                s, planned = st.scope, min(50, plan.action_budget - used)
                # U40: a finished attempt of a shorter plan (extended later) keeps the smaller budget it ran with
                budget_ok = s.action_budget == planned or (st.outcome.value in TERMINAL and s.action_budget < planned)
                if (s.episode_order, s.task_id, s.visit_index) != (n, t, visits[t]) or not budget_ok:
                    raise ValueError(f"ledger episode {eid} does not follow the progression set plan")
                rows.append(_episode_row(run_id, plan.set_id, cond, rep, n, t, visits[t], st, L.episode_events(eid)))
                if st.outcome.value in TERMINAL:
                    used += st.actions_used
                    k += st.outcome.value == "success"
                else:
                    open_ = True
    return rows


def _episode_row(run_id: str, set_id: str, cond: str, rep: int, order: int, task_id: str, visit: int,
                 st: EpisodeState | None, events: list[tuple[str, str | None]] = ()) -> dict[str, Any]:
    """events: (interrupted|resumed, reason) history, kept after recovery (spec §11.2: report incompletes and
    whether they were recovered)."""
    get = (lambda a: getattr(st, a)) if st is not None else (lambda a: None)   # noqa: E731
    outcome = st.outcome.value if st is not None else "not_started"
    success = outcome == "success"
    return {"run_id": run_id, "condition": cond, "set_id": set_id, "set_rep": rep, "episode_order": order,
            "episode_id": st.scope.episode_id if st is not None else None, "task_id": task_id, "visit_index": visit,
            "visit": "first" if visit == 1 else "revisit", "outcome": outcome,
            "outcome_reason": _vocab(get("outcome_reason"), INFRA_REASONS),
            "success": success if outcome in TERMINAL else None,   # unknown, never a failure, while not terminal
            "actions_used": get("actions_used"),
            "actions_to_success": get("actions_to_success") if success else None,   # never 51 / max+1 (B21)
            "consultation_requests": get("consultation_requests"), "experiment_attempts": get("experiment_attempts"),
            "experiment_evaluations": get("experiment_evaluations"),
            "invalid_experiment_requests": get("invalid_experiment_requests"),
            "protocol_errors_total": get("protocol_errors_total"),
            "finalization": st.finalization.value if st is not None else None,
            "finalization_reason": _vocab(get("finalization_reason"), FINALIZATION_REASONS),
            "interruptions": sum(k == "interrupted" for k, _ in events), "resumes": sum(k == "resumed" for k, _ in events),
            "interruption_reasons": ";".join(sorted({_vocab(d, INFRA_REASONS) or "other" for k, d in events
                                                     if k == "interrupted"})) or None,
            "memory_hash_start": get("memory_hash_start"), "memory_hash_end": get("memory_hash_end"),
            "action_budget": st.scope.action_budget if st is not None else None}


def _gap(task: PublicTask, results: dict[str, float]) -> float:
    """Worst relative shortfall to the success thresholds (DEFINITIONS['best_so_far'])."""
    worst = 0.0
    for c in task.success:
        v = results.get(c.metric)
        if v is None or not math.isfinite(v):
            return math.inf
        thr = c.target - c.tolerance if c.direction == "maximize" else c.target + c.tolerance
        short = max(0.0, thr - v) if c.direction == "maximize" else max(0.0, v - thr)
        worst = max(worst, short / (abs(thr) or 1.0))
    return worst


def _action_rows(L: Ledger, ep: dict[str, Any], task: PublicTask) -> list[dict[str, Any]]:
    """Committed actions in order. Consult/hypothesis text and advisor answers are not exported."""
    out, best = [], None
    rows = L.db.execute("SELECT action_id, kind, counted_as, payload_json, result_json FROM actions"
                        " WHERE episode_id=? AND status='committed' ORDER BY seq", (ep["episode_id"],))
    for n, r in enumerate(rows, start=1):
        args, res = json.loads(r["payload_json"])["args"], json.loads(r["result_json"])
        a = {k: ep[k] for k in ("run_id", "condition", "set_id", "set_rep", "episode_id", "episode_order", "task_id",
                                "visit_index")}
        a.update(action_number=n, action_id=r["action_id"], kind=r["kind"], counted_as=r["counted_as"],
                 parameters=args.get("parameters") if r["kind"] == "run_experiment" else None,
                 results=None, invalid_reason=None, meets_success_criteria=None)
        if r["counted_as"] == "evaluation":
            obs = Observation.model_validate(res)
            a.update(results=obs.results, meets_success_criteria=obs.meets_success_criteria)
            g = _gap(task, obs.results)
            if best is None or g < best[0]:
                best = (g, obs.action_id, obs.results)
        elif r["counted_as"] == "invalid":
            a["invalid_reason"] = res["reason"]
        if best is not None:
            a.update(best_so_far_results=best[2], best_so_far_gap=None if math.isinf(best[0]) else best[0],
                     best_so_far_action_id=best[1])
        else:
            a.update(best_so_far_results=None, best_so_far_gap=None, best_so_far_action_id=None)
        out.append(a)
    return out


# ---------------------------------------------------------------- set metrics and comparison

def _rate(k: int, n: int) -> dict[str, Any]:
    return {"successes": k, "n": n, "rate": k / n if n else None}


def _set_metrics(rows: list[dict[str, Any]], plan: SetPlan | None = None, key: tuple[str, int] = ("", 0)
                 ) -> dict[str, Any]:
    if plan is not None and plan.mode == "progression":
        out = _fixed_metrics(rows, key) if rows else _empty_metrics(key, plan)
        out["progression_set"] = prog = _progression_metrics(rows, plan)
        if prog["set_end"] not in ("all_cleared", "budget_used"):
            out["complete"] = False
            if not out["incomplete_reasons"]:
                out["incomplete_reasons"] = [f"set ended early ({prog['set_end']}): the next attempt has not run"]
        return out
    return _fixed_metrics(rows, key)


def _empty_metrics(key: tuple[str, int], plan: SetPlan) -> dict[str, Any]:
    return {"condition": key[0], "set_id": plan.set_id, "set_rep": key[1], "planned": 0, "terminal": 0,
            "complete": False, "counts": {**{k: 0 for k in (*TERMINAL, "infra_incomplete", "running", "not_started")},
                                          "finalization_failed": 0},
            "success_at_50": _rate(0, 0), "actions_to_success": {"over": "successful episodes only", "n": 0,
                                                                 "terminal_n": 0, "success_rate": None, "mean": None,
                                                                 "median": None, "min": None, "max": None},
            "first_visit": _rate(0, 0), "revisit": _rate(0, 0), "progression": [],
            "incomplete_reasons": ["not started"]}


def _progression_metrics(rows: list[dict[str, Any]], plan: SetPlan) -> dict[str, Any]:
    term = [r for r in rows if r["outcome"] in TERMINAL and r["finalization"] == "done"]
    total = sum(r["actions_used"] for r in term)
    per_task, cleared = [], 0
    for t in plan.episodes:
        att = [r for r in term if r["task_id"] == t]
        ok = next((i for i, r in enumerate(att) if r["success"]), None)
        cleared += ok is not None
        per_task.append({"task_id": t, "attempts": len(att), "cleared": ok is not None,
                         "first_attempt_success": att[0]["success"] if att else None,
                         "actions_used": sum(r["actions_used"] for r in att),
                         "actions_to_clear": (sum(r["actions_used"] for r in att[:ok]) + att[ok]["actions_to_success"])
                         if ok is not None else None})
    pending = any(r["outcome"] not in TERMINAL or r["finalization"] != "done" for r in rows)
    set_end = ("incomplete" if pending else "all_cleared" if cleared == len(plan.episodes)
               else "budget_used" if total >= plan.action_budget else "incomplete")
    return {"tasks_cleared": cleared, "tasks": len(plan.episodes), "total_actions": total,
            "action_budget": plan.action_budget, "set_end": set_end, "attempts": len(term),
            "truncated_attempts": sum(1 for r in term if (r.get("action_budget") or 50) < 50), "per_task": per_task}


def _fixed_metrics(rows: list[dict[str, Any]], key: tuple[str, int] = ("", 0)) -> dict[str, Any]:
    term = [r for r in rows if r["outcome"] in TERMINAL]
    pending = [r for r in rows if r["outcome"] not in TERMINAL or r["finalization"] != "done"]
    succ = [r for r in term if r["success"]]
    counts = Counter(r["outcome"] for r in rows)
    ats = [r["actions_to_success"] for r in succ]
    visits = {v: [r for r in term if (r["visit_index"] == 1) == (v == "first_visit")] for v in ("first_visit", "revisit")}
    return {
        "condition": rows[0]["condition"], "set_id": rows[0]["set_id"], "set_rep": rows[0]["set_rep"],
        "planned": len(rows), "terminal": len(term), "complete": not pending,
        "counts": {**{k: counts.get(k, 0) for k in (*TERMINAL, "infra_incomplete", "running", "not_started")},
                   "finalization_failed": sum(r["finalization"] != "done" for r in term)},
        "success_at_50": _rate(len(succ), len(term)),
        "actions_to_success": {"over": "successful episodes only", "n": len(ats), "terminal_n": len(term),
                               "success_rate": _rate(len(succ), len(term))["rate"],
                               "mean": statistics.fmean(ats) if ats else None,
                               "median": statistics.median(ats) if ats else None,
                               "min": min(ats, default=None), "max": max(ats, default=None)},
        **{v: _rate(sum(r["success"] for r in rs), len(rs)) for v, rs in visits.items()},
        "progression": [{k: r[k] for k in ("episode_order", "task_id", "visit", "outcome", "success", "actions_used",
                                           "actions_to_success")} for r in rows],
        "incomplete_reasons": [_why(r) for r in pending],
    }


def _why(r: dict[str, Any]) -> str:
    if r["outcome"] in TERMINAL:
        what, reason = f"finalization {r['finalization']}", r["finalization_reason"]
    else:
        what, reason = r["outcome"], r["outcome_reason"]
    return f"episode {r['episode_order']}: {what}" + (f" ({reason})" if reason else "")


def _conditions(sets: list[dict[str, Any]], plan: SetPlan) -> dict[str, Any]:
    """Per condition over the analysis reps (complete in EVERY planned condition, the same reps _paired uses);
    progression by episode order (first visits vs revisits)."""
    by = {(s["condition"], s["set_rep"]): s for s in sets}
    if plan.mode == "progression":
        reps = range(1, plan.reps + 1)
        return {c: {"complete_reps": [r for r in reps if by[(c, r)]["complete"]],
                    "incomplete_reps": [r for r in reps if not by[(c, r)]["complete"]],
                    "by_rep": [{"set_rep": r, **{k: by[(c, r)]["progression_set"][k] for k in (
                        "tasks_cleared", "tasks", "total_actions", "action_budget", "set_end", "attempts",
                        "truncated_attempts")}} for r in reps]} for c in plan.conditions}
    reps = range(1, plan.reps + 1)
    both = [r for r in reps if all(by[(c, r)]["complete"] for c in plan.conditions)]
    out = {}
    for c in plan.conditions:
        done = [by[(c, r)] for r in both]
        out[c] = {"complete_reps": [r for r in reps if by[(c, r)]["complete"]],
                  "incomplete_reps": [r for r in reps if not by[(c, r)]["complete"]],
                  "analysis_reps": both,
                  "mean_success_at_50": statistics.fmean(s["success_at_50"]["rate"] for s in done) if done else None,
                  "by_episode_order": [{"episode_order": o, "task_id": t, "visit": "first" if v == 1 else "revisit",
                                        **_rate(sum(s["progression"][o - 1]["success"] for s in done), len(done))}
                                       for o, t, v in plan.visits()]}
    return out


def _paired(sets: list[dict[str, Any]], plan: SetPlan) -> dict[str, Any]:
    """Plan T09 default: per set_rep product - baseline success@50, complete pairs only, paired bootstrap over reps.
    Progression plans (U27): tasks cleared, then total actions, per complete set pair; descriptive."""
    if plan.mode == "progression":
        return _paired_progression(sets, plan)
    base = {"unit": DEFINITIONS["paired_unit"], "difference": "product - baseline success@50"}
    if set(plan.conditions) != set(CONDITIONS):
        return {**base, "available": False, "reason": f"set plan runs {plan.conditions}, not both conditions"}
    by = {(s["condition"], s["set_rep"]): s for s in sets}
    pairs, excluded = [], []
    for rep in range(1, plan.reps + 1):
        b, p = by[("baseline", rep)], by[("product", rep)]
        bad = [f"{c} incomplete: {'; '.join(s['incomplete_reasons'])}" for c, s in (("baseline", b), ("product", p))
               if not s["complete"]]
        if bad:
            excluded.append({"set_rep": rep, "reason": " | ".join(bad)})
            continue
        br, pr = b["success_at_50"]["rate"], p["success_at_50"]["rate"]
        pairs.append({"set_rep": rep, "baseline": br, "product": pr, "difference": pr - br})
    diffs = [x["difference"] for x in pairs]
    boot: dict[str, Any] = {"method": "paired percentile bootstrap resampling whole set pairs with replacement",
                            "seed": BOOTSTRAP_SEED, "B": BOOTSTRAP_B, "level": 0.95, "interval": None}
    if len(diffs) >= 2:
        rng, n = random.Random(BOOTSTRAP_SEED), len(diffs)
        means = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_B)]
        q = statistics.quantiles(means, n=40, method="inclusive")   # q[0] = 2.5th, q[-1] = 97.5th percentile
        boot["interval"] = [q[0], q[-1]]
    else:
        boot["not_computed"] = "fewer than 2 complete set pairs"
    return {**base, "available": True, "pairs": pairs, "n_pairs": len(pairs),
            "mean_difference": statistics.fmean(diffs) if diffs else None,
            "excluded": excluded, "n_excluded": len(excluded), "bootstrap": boot}


def _paired_progression(sets: list[dict[str, Any]], plan: SetPlan) -> dict[str, Any]:
    base = {"unit": DEFINITIONS["paired_unit"], "primary": DEFINITIONS["progression_primary"], "kind": "progression"}
    if set(plan.conditions) != set(CONDITIONS):
        return {**base, "available": False, "reason": f"set plan runs {plan.conditions}, not both conditions"}
    by = {(s["condition"], s["set_rep"]): s for s in sets}
    pairs, excluded = [], []
    for rep in range(1, plan.reps + 1):
        b, p = by[("baseline", rep)], by[("product", rep)]
        bad = [f"{c} incomplete: {'; '.join(s['incomplete_reasons'])}" for c, s in (("baseline", b), ("product", p))
               if not s["complete"]]
        if bad:
            excluded.append({"set_rep": rep, "reason": " | ".join(bad)})
            continue
        bp, pp = b["progression_set"], p["progression_set"]
        kb, kp = (-bp["tasks_cleared"], bp["total_actions"]), (-pp["tasks_cleared"], pp["total_actions"])
        pairs.append({"set_rep": rep,
                      "baseline": {"tasks_cleared": bp["tasks_cleared"], "total_actions": bp["total_actions"]},
                      "product": {"tasks_cleared": pp["tasks_cleared"], "total_actions": pp["total_actions"]},
                      "better": "product" if kp < kb else "baseline" if kb < kp else "tie"})
    tally = Counter(x["better"] for x in pairs)
    return {**base, "available": True, "pairs": pairs, "n_pairs": len(pairs), "excluded": excluded,
            "n_excluded": len(excluded), "tally": {k: tally.get(k, 0) for k in ("product", "baseline", "tie")},
            "statement": "descriptive: fewer than 2 complete set pairs" if len(pairs) < 2 else
                         "per-pair outcomes; no interval is computed for this ordinal primary metric"}


def _banner(manifest: dict, plan: SetPlan, complete: bool) -> dict[str, Any]:
    """Only a frozen, fixture-free, two-condition, complete evaluation is a main result; the report re-checks this
    itself instead of trusting upstream preflight."""
    mode, frozen = manifest["execution_mode"], manifest.get("frozen") is True
    dev = list(manifest.get("development_only_fields") or [])
    why: list[str] = []
    if mode == "offline_fixture":
        label, text = "contract_check", ("OFFLINE FIXTURE RUN (development_only): contract checks only. These numbers "
                                         "are NOT research performance and NOT product value.")
    elif mode == "live_development":
        label, text = "development_evidence", ("LIVE DEVELOPMENT RUN: development evidence only; not a main "
                                               "evaluation result.")
    elif mode != "evaluation":
        raise ValueError(f"unknown execution_mode {mode!r}")
    else:
        why = [w for bad, w in (
            (not frozen, "manifest not frozen"),
            (bool(dev), f"development-only components {dev}"),
            (set(plan.conditions) != set(CONDITIONS), f"set plan runs {plan.conditions}, not both conditions"),
            (not complete, "not every planned set is complete (incomplete sets are listed, never dropped or "
                           "counted as failures)")) if bad]
        label = "evaluation_not_main" if why else "main_evaluation_result"
        text = (f"EVALUATION MODE, NOT a main evaluation result: {'; '.join(why)}." if why else
                "FROZEN EVALUATION, both conditions, no development-only components, every planned set complete: "
                "main evaluation result.")
    return {"execution_mode": mode, "frozen_manifest": frozen, "all_sets_complete": complete,
            "development_only_fields": dev, "label": label, "is_main_evaluation_result": label == "main_evaluation_result",
            "not_main_because": why, "statement": text}


# ---------------------------------------------------------------- costs (§8.2, B22)

def _agg(es: list[CostEvent]) -> dict[str, Any]:
    d: dict[str, Any] = {"events": len(es), "retries": sum(e.attempt > 1 for e in es),
                         "statuses": dict(sorted(Counter(e.status for e in es).items())),
                         "latency_s": sum((e.latency_s for e in es), 0.0)}
    for t in TOKENS:
        # token fields exist for model calls only (DEFINITIONS['tokens']); a reported value always counts
        calls = [e for e in es if e.kind == "llm_call" or (e.kind == "embedding" and t == "input_tokens")
                 or getattr(e.usage, t) is not None]
        got = [getattr(e.usage, t) for e in calls if getattr(e.usage, t) is not None]
        d[t] = {"sum": sum(got) if got else None, "reported": len(got), "unavailable": len(calls) - len(got)}
    return d


def _model_row(e: CostEvent, models: dict[str, Any]) -> tuple:
    """(role, provider, configured, recorded, returned, check) per DEFINITIONS['model_check']."""
    cfg = models.get("researcher" if e.role.startswith("researcher_") else e.role) or {}
    ret = e.detail.get("model_returned") if isinstance(e.detail.get("model_returned"), str) else None
    accepted = {cfg["model"], *cfg.get("allowed_returned_models", [])} if cfg.get("model") else {e.model}
    check = ("mismatch" if any(m is not None and m not in accepted for m in (e.model, ret))
             else "ok" if ret is not None and e.detail.get("model_verified", True) is not False else "unverified")
    return e.role, e.provider, cfg.get("model"), e.model, ret, check


def _costs(ledger: list[CostEvent], prebuild: list[CostEvent] | None, episodes: list[dict[str, Any]],
           sessions: dict[str, Any], models_cfg: dict[str, Any]) -> dict[str, Any]:
    events = ledger + (prebuild or [])
    cond_of = {r["episode_id"]: r["condition"] for r in episodes if r["episode_id"]}

    def condition(e: CostEvent) -> str | None:
        if e.episode_id in cond_of:
            return cond_of[e.episode_id]
        parts = (e.scope_key or "").split("/")   # MemoryScope.key = run/condition/set/repN
        return parts[1] if len(parts) >= 4 and parts[1] in CONDITIONS else None

    groups: dict[tuple, list[CostEvent]] = defaultdict(list)
    for e in events:
        groups[(e.phase, condition(e), e.kind, e.role)].append(e)
    models = Counter(_model_row(e, models_cfg) for e in events if e.kind in ("llm_call", "embedding"))
    runtime_actions = sum(r["actions_used"] or 0 for r in episodes)
    return {
        "sources": {"ledger.sqlite": len(ledger), "prebuild_costs.jsonl": None if prebuild is None else len(prebuild)},
        "by_phase": {ph: {**_agg([e for e in events if e.phase == ph]),
                          "budget_actions": runtime_actions if ph == "runtime" else 0} for ph in PHASES},
        "groups": [{"phase": k[0], "condition": k[1], "kind": k[2], "role": k[3], **_agg(es)}
                   for k, es in sorted(groups.items(), key=lambda kv: tuple(x or "" for x in kv[0]))],
        "models": [{"role": r, "provider": p, "model_configured": cfg, "model_recorded": m, "model_returned": ret,
                    "check": chk, "events": n}
                   for (r, p, cfg, m, ret, chk), n in sorted(models.items(), key=lambda kv: tuple(x or "" for x in kv[0]))],
        "model_mismatch_events": sum(n for k, n in models.items() if k[5] == "mismatch"),
        "model_unverified_events": sum(n for k, n in models.items() if k[5] == "unverified"),
        "sessions": sessions,
    }


# ---------------------------------------------------------------- public projection guard

OWN_DATA = ("parameters", "results", "best_so_far_results")   # researcher's own submissions / observations


def _norm(s: str) -> str:
    return s.casefold().replace("\\", "/")


def _forbidden(manifest: dict, root: Path, tasks: dict[str, PublicTask]) -> tuple[set[str], list[str]]:
    """(strings that must never appear in a public output, tasks whose answer bundle was not found). Numeric hidden
    values (known success inputs) cannot be string-scanned; they are kept out by construction (whitelisted fields)."""
    raw = _paths(manifest).private_dir
    d = _resolve(root, raw)
    # the relative dir only as a path prefix: a bare "private" is an ordinary word (run ids, task names)
    out = {str(d), raw.rstrip("/\\") + "/", *(os.environ.get(k) for k in SECRET_ENV)}
    missing = []
    for t in tasks:
        p = d / f"{t}.yaml"
        if not p.exists():
            missing.append(t)   # recorded in report["privacy_scan"], never silent
            continue
        a = PrivateTaskAssets.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
        b = a.answer_bundle
        out |= {b.bundle_id, *b.secret_markers, *b.hidden_asset_paths, *a.model_artifacts.values()}
        for doc in b.blocked_documents:
            out |= {doc.doi, doc.arxiv_id, *doc.titles, *doc.urls}
    # ponytail: strings under 6 chars are skipped (false positives on short ids); raise if bundles use short ids
    return {_norm(s) for s in out if s and len(s) >= 6}, sorted(missing)


def _strings(x: Any) -> Iterator[str]:
    """Every string value and dict key. Numbers are skipped: they are harness counts or simulator results."""
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)


def _assert_public(rep: dict[str, Any], forbidden: set[str]) -> None:
    """Scan harness-authored strings. The researcher's own parameters/results are exported even if they equal a
    hidden value (spec §7.3, B12). CSV/MD are rendered from these same strings."""
    scanned = {**rep, "actions": [{k: v for k, v in a.items() if k not in OWN_DATA} for a in rep["actions"]]}
    text = "\0".join(_norm(s) for s in _strings(scanned))
    hits = sum(f in text for f in forbidden)
    if hits:   # never echo the matched content
        raise ValueError(f"public report would expose {hits} private item(s); nothing was written")


# ---------------------------------------------------------------- rendering

def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return str(v)


def _csv(rows: list[dict[str, Any]], cols: tuple[str, ...], tag: dict[str, str]) -> str:
    """tag (execution_mode, report_label) leads every row: a CSV shared alone still says what its numbers are."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow((*tag, *cols))
    w.writerows([*tag.values(), *(_cell(r[c]) for c in cols)] for r in rows)
    return buf.getvalue()


def _flat_cost(g: dict[str, Any]) -> dict[str, Any]:
    return {**{k: v for k, v in g.items() if k not in TOKENS},
            **{f"{t}_{k}": g[t][k] for t in TOKENS for k in ("sum", "reported", "unavailable")}}


def _num(v: Any) -> str:
    return "" if v is None else f"{v:.3f}" if isinstance(v, float) else _cell(v)


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(_num(x).replace("|", "\\|") for x in r) + " |" for r in rows), ""]


def _r(x: dict[str, Any]) -> str:
    return f"{x['successes']}/{x['n']}" + (f" = {x['rate']:.3f}" if x["rate"] is not None else "")


def _tok(d: dict[str, Any]) -> str:
    return f"{_num(d['sum']) or 'n/a'} (unavailable {d['unavailable']})"


def _markdown(rep: dict[str, Any]) -> str:
    b, rp, pc, co = rep["banner"], rep["reproducibility"], rep["paired_comparison"], rep["costs"]
    L = [f"# LabGene run report: {rp.get('run_id')}", "",
         f"> **{b['label']}** (execution_mode={b['execution_mode']}, frozen={_cell(b['frozen_manifest'])}, "
         f"all sets complete={_cell(b['all_sets_complete'])}): {b['statement']}", ""]
    if rep["privacy_scan"]["tasks_missing"]:
        L += [f"Privacy scan PARTIAL: no answer bundle found for tasks {rep['privacy_scan']['tasks_missing']}; "
              "their bundle strings were not scanned (whitelisted fields only).", ""]
    L += ["## Set results (per condition, set_rep)", ""]
    L += _table(["condition", "rep", "complete", "success@50", *TERMINAL, "infra_incomplete", "running",
                 "finalization_failed", "not_started", "first visit", "revisit", "actions to success (successful only)"],
                [[s["condition"], s["set_rep"], s["complete"], _r(s["success_at_50"]), *(s["counts"][k] for k in (
                    *TERMINAL, "infra_incomplete", "running", "finalization_failed", "not_started")),
                  _r(s["first_visit"]), _r(s["revisit"]),
                  f"n={s['actions_to_success']['n']} of {s['actions_to_success']['terminal_n']} terminal "
                  f"(success rate {_num(s['actions_to_success']['success_rate']) or 'n/a'}); "
                  f"median {_num(s['actions_to_success']['median'])}, mean {_num(s['actions_to_success']['mean'])}"]
                 for s in rep["sets"]])
    for s in rep["sets"]:
        if s["incomplete_reasons"]:
            L.append(f"- {s['condition']} rep {s['set_rep']} INCOMPLETE: {'; '.join(s['incomplete_reasons'])}")
    if pc.get("kind") == "progression":
        return "\n".join(L + _markdown_progression(rep) + _markdown_tail(rep))
    L += ["", "## Paired comparison (product - baseline success@50)", "", f"Unit: {pc['unit']}.", ""]
    if not pc["available"]:
        L += [f"Not available: {pc['reason']}", ""]
    else:
        bt = pc["bootstrap"]
        L += _table(["set_rep", "baseline", "product", "difference"],
                    [[x["set_rep"], x["baseline"], x["product"], x["difference"]] for x in pc["pairs"]])
        L += [f"- complete pairs: {pc['n_pairs']}; mean difference: {_num(pc['mean_difference']) or 'n/a'}",
              f"- {bt['method']}, seed={bt['seed']}, B={bt['B']}: " +
              (f"{bt['level']:.0%} interval [{bt['interval'][0]:.3f}, {bt['interval'][1]:.3f}]" if bt["interval"]
               else f"not computed ({bt['not_computed']})"),
              f"- excluded set reps: {pc['n_excluded']}"]
        L += [f"  - rep {x['set_rep']}: {x['reason']}" for x in pc["excluded"]]
        L.append("")
    L += ["## Progression by episode order (reps complete in every condition only)", ""]
    for c, d in rep["conditions"].items():
        L += [f"**{c}**: analysis reps {d['analysis_reps']} (complete {d['complete_reps']}, incomplete "
              f"{d['incomplete_reps']}), mean success@50 over analysis reps {_num(d['mean_success_at_50']) or 'n/a'}",
              ""]
        L += _table(["order", "task", "visit", "success"],
                    [[x["episode_order"], x["task_id"], x["visit"], _r(x)] for x in d["by_episode_order"]])
    return "\n".join(L + _markdown_tail(rep))


def _markdown_progression(rep: dict[str, Any]) -> list[str]:
    pc = rep["paired_comparison"]
    L = ["", "## Progression sets (U26) and primary comparison (U27)", "", f"Primary: {pc['primary']}.", ""]
    L += _table(["condition", "rep", "complete", "set end", "tasks cleared", "total actions", "budget", "attempts",
                 "truncated"],
                [[s["condition"], s["set_rep"], s["complete"], s["progression_set"]["set_end"],
                  f"{s['progression_set']['tasks_cleared']}/{s['progression_set']['tasks']}",
                  s["progression_set"]["total_actions"], s["progression_set"]["action_budget"],
                  s["progression_set"]["attempts"], s["progression_set"]["truncated_attempts"]] for s in rep["sets"]])
    L += _table(["condition", "rep", "task", "attempts", "cleared", "first attempt", "actions used", "actions to clear"],
                [[s["condition"], s["set_rep"], t["task_id"], t["attempts"], t["cleared"], t["first_attempt_success"],
                  t["actions_used"], t["actions_to_clear"]] for s in rep["sets"] for t in s["progression_set"]["per_task"]])
    if not pc["available"]:
        return L + [f"Comparison not available: {pc['reason']}", ""]
    L += _table(["set_rep", "baseline cleared", "baseline actions", "product cleared", "product actions", "better"],
                [[x["set_rep"], x["baseline"]["tasks_cleared"], x["baseline"]["total_actions"],
                  x["product"]["tasks_cleared"], x["product"]["total_actions"], x["better"]] for x in pc["pairs"]])
    L += [f"- {pc['statement']}; tally {pc['tally']}; excluded set reps: {pc['n_excluded']}"]
    L += [f"  - rep {x['set_rep']}: {x['reason']}" for x in pc["excluded"]]
    return L + [""]


def _markdown_tail(rep: dict[str, Any]) -> list[str]:
    rp, co = rep["reproducibility"], rep["costs"]
    L = ["## Costs", "", f"Sources: {_cell(co['sources'])}. Sessions: {_cell(co['sessions'])}.", ""]
    L += _table(["phase", "events", "retries", "budget actions", "latency s", *TOKENS],
                [[ph, d["events"], d["retries"], d["budget_actions"], d["latency_s"], *(_tok(d[t]) for t in TOKENS)]
                 for ph, d in co["by_phase"].items()])
    L += _table(["phase", "condition", "kind", "role", "events", "retries", "statuses", "latency s", *TOKENS],
                [[g["phase"], g["condition"], g["kind"], g["role"], g["events"], g["retries"], g["statuses"],
                  g["latency_s"], *(_tok(g[t]) for t in TOKENS)] for g in co["groups"]])
    L += [f"Model mismatch events: {co['model_mismatch_events']}; unverified (no returned model recorded): "
          f"{co['model_unverified_events']}", ""]
    L += _table(["role", "provider", "configured", "recorded", "returned", "check", "events"],
                [[m["role"], m["provider"], m["model_configured"], m["model_recorded"], m["model_returned"], m["check"],
                  m["events"]] for m in co["models"]])
    L += ["## Reproducibility", ""]
    L += [f"- {k}: `{_cell(v)}`" for k, v in rp.items() if k != "set_plan"]
    L += [f"- set_plan: `{_cell(rp.get('set_plan'))}`", "", "## Definitions", ""]
    L += [f"- **{k}**: {v}" for k, v in rep["definitions"].items()]
    L += ["", "Per-episode and per-action tables: episodes.csv, actions.csv; cost groups: costs.csv.", ""]
    return L
