"""Task validity evidence (T02, plan T08.1, spec §6, §10.4).

validate_task computes what can be computed (determinism across fresh adapters, corner/center validity,
known success inputs actually succeed, error vs data) and takes the rest from PrivateTaskAssets.validity_evidence.
A report is `validated` only when every evidence item exists and no open question remains.
Known success inputs and reference-point evaluations live in `private`, excluded from public().

    python -m labgene.simulators.task_validation <task_id> [--repeats 2]
writes docs/implementation/task-validation/<task_id>.json (public) and private/task-validation/<task_id>.json.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import yaml

from ..config import SimulatorConfig, repo_root_for
from ..contracts import (CategoricalParam, Frozen, PrivateTaskAssets, PublicTask, ValidParameters, canonical_json,
                         payload_hash)
from .base import SimulatorAdapter
from .validation import validate_parameters

PUBLIC_DIR = "docs/implementation/task-validation"
PRIVATE_DIR = "private/task-validation"
SIMULATORS_FILE = f"{PUBLIC_DIR}/simulators.yaml"

Row = dict[str, Any]   # {"parameters": {...}, "observed": {metric: value}, "label": str (optional)}


class PrivateValidation(Frozen):
    """Evaluator-only: inputs that reveal where the task succeeds."""
    known_success: list[dict[str, Any]] = []
    reference_points: list[dict[str, Any]] = []
    near_target_rows: list[dict[str, Any]] = []
    range_extremes: dict[str, Any] = {}


class TaskValidationReport(Frozen):
    task_id: str
    task_version: str
    task_hash: str
    simulator_id: str
    simulator_version: str
    simulator_descriptor: str | None          # upstream commit + backend settings (evaluator-side)
    created_at: str
    status: Literal["validated", "unvalidated"]
    reasons: list[str]
    development_only: bool
    paper: dict[str, Any] = {}
    code: dict[str, Any] = {}
    weights_sha256: dict[str, str] = {}
    license: dict[str, Any] = {}
    inputs: dict[str, Any]
    metrics: list[dict[str, Any]]
    success_rule: dict[str, Any]
    determinism: dict[str, Any]
    range_check: dict[str, Any]
    validation_error: dict[str, Any] | None = None
    near_target_residual: dict[str, Any] | None = None
    known_success_summary: dict[str, int]
    open_questions: list[str] = []
    notes: list[str] = []
    private: PrivateValidation = PrivateValidation()   # empty when loaded from a public report

    def public(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"private"})


def registrable_for_evaluation(report: TaskValidationReport, task: PublicTask | None = None) -> bool:
    """True only for a validated, non-development report; with task, also for that exact task version."""
    ok = report.status == "validated" and not report.reasons and not report.development_only
    if task is not None:
        ok = ok and report.task_hash == payload_hash(task) and report.simulator_version == task.simulator_version
    return ok


def probe_points(task: PublicTask) -> list[dict[str, Any]]:
    """Every corner of the numeric box and its center, for each categorical choice. Under linear constraints an
    infeasible probe is pulled along the segment toward the first feasible probe, onto the constraint boundary."""
    # ponytail: full corner grid (2^n); sample corners if a task gets more than ~10 numeric parameters.
    # ponytail: no feasible corner/center at all (e.g. equality constraints) -> probes stay infeasible, report says so.
    axes, center = [], []
    for p in task.parameters:
        if isinstance(p, CategoricalParam):
            axes.append([(p.name, c) for c in p.choices])
            center.append([(p.name, c) for c in p.choices])
        else:
            axes.append([(p.name, p.min), (p.name, p.max)])
            mid = (p.min + p.max) / 2
            center.append([(p.name, round(mid) if p.kind == "integer" else mid)])
    raw = [dict(c) for c in itertools.product(*axes)] + [dict(c) for c in itertools.product(*center)]
    ok = lambda x: isinstance(validate_parameters(task, x), ValidParameters)
    anchor = next((x for x in raw if ok(x)), None)
    if anchor is None:
        return raw
    ints = {p.name for p in task.parameters if p.kind == "integer"}

    def pull(x: dict[str, Any]) -> dict[str, Any]:
        # constraints are numeric-only, so the anchor's numbers with x's categories are feasible (t = 0)
        at = lambda t: {k: v if isinstance(v, str) else (round if k in ints else float)(anchor[k] + t * (v - anchor[k]))
                        for k, v in x.items()}
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ok(at(mid)) else (lo, mid)
        return at(lo)
    return [x if ok(x) else pull(x) for x in raw]


def _valid(task: PublicTask, xs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [v.parameters for v in (validate_parameters(task, x) for x in xs) if isinstance(v, ValidParameters)]


def _errors(pairs: list[tuple[float, float]]) -> dict[str, float]:
    res = [p - o for p, o in pairs]
    return {"n": len(res), "mean_residual": sum(res) / len(res), "mae": sum(map(abs, res)) / len(res),
            "rmse": math.sqrt(sum(r * r for r in res) / len(res)), "max_abs": max(map(abs, res))}


def validate_task(task: PublicTask, make_adapter: Callable[[], SimulatorAdapter], private: PrivateTaskAssets, *,
                  repeats: int = 2, data: list[Row] | None = None) -> TaskValidationReport:
    """make_adapter must return a FRESH adapter (fresh worker process, empty cache) per call: determinism is
    checked across repeats, not against a cache (a reused adapter or cache hit blocks validation).
    validity_evidence.identities [{metric, numerator, denominator}] declares metric = numerator / denominator
    (outputs or inputs); known success that holds only where the model breaks it blocks validation.
    validity_evidence paper/code/license/notes/open_questions are copied into the PUBLIC report verbatim:
    they must not name success inputs or regions."""
    ev = private.validity_evidence
    candidates = probe_points(task)
    feasible = _valid(task, candidates)
    corners = list({canonical_json(x): x for x in feasible}.values())   # distinct, so points[:len(corners)] are probes
    # Determinism is also checked where it matters most: at the known-success and reference inputs.
    extra = _valid(task, private.known_success_inputs + [r["parameters"] for r in ev.get("reference_points", [])])
    points = list({canonical_json(x): x for x in corners + extra}.values())
    runs, adapters, hits = [], [], 0
    for _ in range(max(repeats, 1)):
        if adapters and hasattr(adapters[-1], "close"):
            adapters[-1].close()
        adapters.append(make_adapter())
        outs = [adapters[-1].evaluate(p) for p in points]   # points are distinct: any hit is a carried-over cache
        runs.append([canonical_json(o.results) for o in outs])
        hits += sum(o.cache_hit for o in outs)
    adapter = adapters[-1]
    first = [json.loads(r) for r in runs[0][:len(corners)]]
    metric_names = [m.name for m in task.metrics]
    finite = all(all(isinstance(r.get(m), (int, float)) and math.isfinite(r[m]) for m in metric_names) for r in first)
    determinism = {"repeats": len(runs), "points": len(points), "cache_hits": hits,
                   "fresh_adapter_per_repeat": hits == 0 and len({id(a) for a in adapters}) == len(adapters),
                   "identical": all(r == runs[0] for r in runs)}
    range_check = {"probes": len(candidates), "feasible": len(feasible), "distinct_feasible": len(corners),
                   "constraints": len(task.constraints), "all_metrics_finite": finite}
    # Output extremes over public probes stay private: boundary probes can sit next to the success region.
    extremes = ({"min": {m: min(r[m] for r in first) for m in metric_names},
                 "max": {m: max(r[m] for r in first) for m in metric_names}} if finite and first else {})
    # ponytail: only ratio identities (metric = numerator / denominator); add forms when a task needs one.
    idents = ev.get("identities", [])

    def run(x: dict[str, Any]) -> dict[str, Any]:
        v = validate_parameters(task, x)
        if not isinstance(v, ValidParameters):
            return {"parameters": x, "valid": False, "reason": v.reason}
        results = adapter.evaluate(v.parameters).results
        out = {"parameters": v.parameters, "valid": True, "results": results, "success": task.is_success(results)}
        if idents:   # success judged with each identity recomputed from the model's own outputs/inputs
            vals = {**v.parameters, **results}
            out["identity_consistent_success"] = task.is_success(
                {**results, **{i["metric"]: vals[i["numerator"]] / vals[i["denominator"]] for i in idents}})
        return out

    known = [run(x) for x in private.known_success_inputs]
    refs = [{**r, **run(r["parameters"])} for r in ev.get("reference_points", [])]
    validation_error = near_target = None
    near_rows: list[dict[str, Any]] = []
    if data:
        # Evaluator-owned rows are model-fidelity probes, evaluated as recorded (some sit just outside the
        # nominal public ranges); they are never experiments.
        used = [{**row, "results": adapter.evaluate(row["parameters"]).results} for row in data]
        outside = sum(not isinstance(validate_parameters(task, row["parameters"]), ValidParameters) for row in data)
        validation_error = {"rows": len(used), "rows_outside_public_rules": outside,
                            "basis": ev.get("data", {}).get("basis", ""),
                            "per_metric": {m: _errors([(e["results"][m], e["observed"][m]) for e in used])
                                           for m in metric_names}}
        near_rows = [e for e in used if task.is_success(e["observed"])]
        near_target = {"definition": "data rows whose OBSERVED values meet the public success rule",
                       "per_metric": ({m: _errors([(e["results"][m], e["observed"][m]) for e in near_rows])
                                       for m in metric_names} if near_rows else None)}
    if refs:
        near_target = near_target or {}
        near_target["reference_points"] = {r.get("label", f"ref{i}"): {m: r["results"][m] - r["observed"][m]
                                                                       for m in metric_names}
                                           for i, r in enumerate(refs) if r["valid"]}
    if hasattr(adapter, "close"):
        adapter.close()

    reasons: list[str] = []
    if ev.get("kind") == "artificial_fixture":
        reasons.append("artificial fixture: offline contract check, not a scientific model")
    if ev.get("development_only"):
        reasons.append(f"development_only: {ev.get('purpose', 'not a main-evaluation task')}")
    for key, need in (("paper", ("doi", "version")), ("code", ("repo", "commit")), ("license", ("code",))):
        miss = [f for f in need if not (ev.get(key) or {}).get(f)]
        if miss:
            reasons.append(f"{key} evidence missing: {', '.join(miss)}")
    if not ev.get("mechanistic") and not private.model_artifacts:
        reasons.append("model weights sha256 not recorded")
    transforms = ev.get("metric_transforms") or {}
    if any(m not in transforms for m in metric_names):
        reasons.append("metric transform not recorded for: " + ", ".join(m for m in metric_names if m not in transforms))
    if determinism["repeats"] < 2:
        reasons.append("determinism needs >= 2 repeats with fresh adapters")
    elif not determinism["fresh_adapter_per_repeat"]:
        reasons.append("determinism not shown: repeats reused an adapter or were served from a cache")
    elif not determinism["identical"]:
        reasons.append("outputs differ across fresh adapters (not deterministic)")
    if len(corners) < 2 or not finite:
        reasons.append("range evidence missing: fewer than 2 feasible probe inputs or non-finite outputs")
    if validation_error is None:
        reasons.append("no observed data: validation error vs data not measured")
    if not near_target or not (near_target.get("per_metric") or near_target.get("reference_points")):
        reasons.append("near-target residual not measured")
    if not ev.get("success_rule_mapping"):
        reasons.append("success rule not mapped to the paper's goal")
    if not known:
        reasons.append("no evaluator-held known success inputs")
    elif not all(k.get("success") for k in known):
        reasons.append(f"{sum(not k.get('success') for k in known)} of {len(known)} known success inputs do not succeed")
    broken = sum(bool(k.get("success")) and not k.get("identity_consistent_success") for k in known) if idents else 0
    if broken:
        reasons.append(f"{broken} of {len(known)} known success inputs succeed only where the model breaks a declared "
                       "identity (" + "; ".join(f"{i['metric']} = {i['numerator']} / {i['denominator']}" for i in idents)
                       + "): success would exploit model error")
    reasons += [f"open question: {q}" for q in ev.get("open_questions", [])]

    return TaskValidationReport(
        task_id=task.task_id, task_version=task.version, task_hash=payload_hash(task),
        simulator_id=task.simulator_id, simulator_version=task.simulator_version,
        simulator_descriptor=getattr(adapter, "descriptor", None),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status="unvalidated" if reasons else "validated", reasons=reasons,
        development_only=bool(ev.get("development_only") or ev.get("kind") == "artificial_fixture"),
        paper=ev.get("paper") or {}, code=ev.get("code") or {}, weights_sha256=private.model_artifacts,
        license=ev.get("license") or {},
        inputs={"parameters": [p.model_dump(mode="json") for p in task.parameters],
                "constraints": [c.model_dump(mode="json") for c in task.constraints]},
        metrics=[{**m.model_dump(mode="json"), "transform": transforms.get(m.name, "unrecorded")} for m in task.metrics],
        success_rule={"criteria": [c.model_dump(mode="json") for c in task.success], "all_required": True,
                      "paper_goal_mapping": ev.get("success_rule_mapping", ""), "identities": idents},
        determinism=determinism, range_check=range_check, validation_error=validation_error,
        near_target_residual=near_target,
        known_success_summary={"checked": len(known), "valid": sum(k["valid"] for k in known),
                               "succeeded": sum(bool(k.get("success")) for k in known),
                               **({"identity_consistent": sum(bool(k.get("identity_consistent_success")) for k in known)}
                                  if idents else {})},
        open_questions=list(ev.get("open_questions", [])), notes=list(ev.get("notes", [])),
        private=PrivateValidation(known_success=known, reference_points=refs, near_target_rows=near_rows,
                                  range_extremes=extremes))


def write_report(report: TaskValidationReport, root: str | Path) -> tuple[Path, Path]:
    root = Path(root)
    out = []
    for folder, body in ((PUBLIC_DIR, report.public()), (PRIVATE_DIR, report.model_dump(mode="json"))):
        p = root / folder / f"{report.task_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        out.append(p)
    return out[0], out[1]


def load_data(spec: dict[str, Any], task: PublicTask, root: Path) -> list[Row]:
    """CSV named by validity_evidence.data: {csv, header_rows, columns: {param: col}, metrics: {metric: col}}."""
    cat = {p.name for p in task.parameters if isinstance(p, CategoricalParam)}
    with open(root / spec["csv"], encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[spec.get("header_rows", 1):]
    col = {name: head.index(c) for name, c in {**spec["columns"], **spec["metrics"]}.items()}
    return [{"parameters": {k: r[col[k]] if k in cat else float(r[col[k]]) for k in spec["columns"]},
             "observed": {m: float(r[col[m]]) for m in spec["metrics"]}} for r in body if r]


def load_private(root: Path, task_id: str) -> PrivateTaskAssets:
    for d in ("private", "tests/fixtures/private"):
        p = root / d / f"{task_id}.yaml"
        if p.exists():
            return PrivateTaskAssets.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
    raise FileNotFoundError(f"no private assets for {task_id} under private/ or tests/fixtures/private/")


def main(argv: list[str] | None = None) -> int:
    from .factory import build_simulator
    ap = argparse.ArgumentParser(prog="python -m labgene.simulators.task_validation")
    ap.add_argument("task_id")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--simulators", default=SIMULATORS_FILE, help="YAML: simulator_id -> SimulatorConfig")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else repo_root_for(Path.cwd() / "x")
    task = PublicTask.model_validate(yaml.safe_load((root / "configs/tasks" / f"{a.task_id}.yaml").read_text(encoding="utf-8")))
    private = load_private(root, a.task_id)
    sims = yaml.safe_load((root / a.simulators).read_text(encoding="utf-8"))
    cfg = SimulatorConfig.model_validate(sims[task.simulator_id])
    spec = private.validity_evidence.get("data")
    report = validate_task(task, lambda: build_simulator(task.simulator_id, cfg, task, root), private,
                           repeats=a.repeats, data=load_data(spec, task, root) if spec else None)
    pub, _ = write_report(report, root)
    print(json.dumps({"task_id": report.task_id, "status": report.status, "reasons": report.reasons,
                      "public_report": str(pub)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
