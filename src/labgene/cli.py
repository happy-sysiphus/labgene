"""`python -m labgene <command>` (T07). Exit codes: 0 ok/complete, 2 blocked by preflight or bad input,
3 run stopped (infra_incomplete / finalization_failed / model_changed / cost cap / knowledge infra; resumable),
4 a validation ran and did not pass (fail / provisional / incomplete), 70 injected crash (fault test)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_profile, load_set_plan


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _run_dir(a) -> Path:
    return Path(a.artifacts_dir) / a.run_id


def _report(run) -> dict:
    from .reporting.report import write_report
    paths = write_report(run.run_dir, json.loads((run.run_dir / "manifest.json").read_text(encoding="utf-8")))
    return {k: str(v) for k, v in paths.items()}


def _execute(run, revalidated: bool) -> int:
    status = run.execute(revalidated=revalidated)
    out = {"run_id": run.manifest.run_id, "execution_mode": run.manifest.execution_mode, "status": status.status,
           "episode_id": status.episode_id, "reason": status.reason, "report": _report(run)}
    if run.manifest.execution_mode == "offline_fixture":
        out["note"] = "offline fixture run: contract check only, not research or product performance"
    _print(out)
    return 0 if status.status == "complete" else 3


def _blocked(problems: list[str]) -> int:
    _print({"ok": False, "problems": problems})
    return 2


def cmd_preflight(a) -> int:
    from .app import run_preflight
    from .config import preflight_problems
    p = load_profile(a.profile)
    probs = run_preflight(p, load_set_plan(a.plan)) if a.plan else preflight_problems(p)
    _print({"ok": not probs, "execution_mode": p.execution_mode, "profile_hash": p.hash, "problems": probs})
    return 0 if not probs else 2


def _new_run(a, evaluation: bool | None):
    """evaluation=None accepts any mode (build-corpus); True/False must match the profile."""
    from .app import Run, run_preflight
    p, plan = load_profile(a.profile), load_set_plan(a.plan)
    if evaluation is not None and evaluation != (p.execution_mode == "evaluation"):
        return None, ["run-evaluation needs an execution_mode=evaluation profile" if evaluation
                      else "evaluation profiles run only through `run-evaluation` (frozen-manifest checks)"]
    probs = run_preflight(p, plan)
    if probs:
        return None, probs
    return Run.create(p, plan, a.run_id), []


def cmd_run_set(a, evaluation: bool = False) -> int:
    run, probs = _new_run(a, evaluation)
    return _blocked(probs) if probs else _execute(run, a.revalidated)


def cmd_build_corpus(a) -> int:
    run, probs = _new_run(a, None)
    if probs:
        return _blocked(probs)
    try:
        hashes = run.build_initial_states()
    finally:
        run.session_line("build-corpus")
    _print({"run_id": a.run_id, "initial_state_hashes": hashes, "next": f"python -m labgene resume --run-id {a.run_id}"})
    return 0


def _run_for_tool(a):
    """Evaluation-support tools reuse the run machinery (gate, initial states, persisted cost guard)."""
    from .app import Run, run_preflight
    run_dir = Path(load_profile(a.profile).resolve(load_profile(a.profile).paths.artifacts_dir)) / a.run_id
    if (run_dir / "manifest.json").exists():
        return Run.load(run_dir), []
    p, plan = load_profile(a.profile), load_set_plan(a.plan)
    probs = [x for x in run_preflight(p, plan) if not x.startswith(("evaluation requires frozen_manifest",
                                                                      "frozen manifest"))]
    return (None, probs) if probs else (Run.create(p, plan, a.run_id), [])


def _tool_ctx(run, name: str):
    from .contracts import CostEvent
    from .costs import CallContext
    path = run.run_dir / f"{name}_costs.jsonl"

    def sink(e: CostEvent) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(e.model_dump_json() + "\n")
    return CallContext(sink=sink, phase="evaluation_support", scope_key=f"{run.manifest.run_id}/{name}", guard=run.guard)


def _write(run, name: str, body: dict) -> Path:
    out = run.run_dir / f"{name}.json"
    out.write_text(json.dumps(body, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def cmd_validate_gate(a) -> int:
    """T08.4: the profile's leakage gate on a pre-fixed case file. Bundles come from the evaluator's private assets."""
    import yaml
    from .app import build_checker
    from .contracts import PrivateTaskAssets
    from .evaluation.gate_validation import load_gate_cases, run_gate_validation
    run, probs = _run_for_tool(a)
    if probs:
        return _blocked(probs)
    cases = load_gate_cases(a.cases)
    d = run.profile.resolve(run.profile.paths.private_dir)
    bundles = {f"{b.bundle_id}@{b.version}": b for f in sorted(d.glob("*.yaml"))
               for b in [PrivateTaskAssets.model_validate(yaml.safe_load(f.read_text(encoding="utf-8"))).answer_bundle]}
    missing = [b for b in cases.answer_bundles if b not in bundles]
    if missing:
        return _blocked([f"answer bundle(s) not in {run.profile.paths.private_dir}: {missing}"])
    try:
        res = run_gate_validation(build_checker(run.profile), [bundles[b] for b in cases.answer_bundles], cases,
                                  _tool_ctx(run, "gate_validation"), input_lock=Path(a.cases).with_suffix(".lock.json"))
    finally:
        run.session_line("validate-gate")
    # evaluator-side result (case texts may quote answer material): kept in the run dir, never in a public report
    out = _write(run, "gate_validation", res)
    _print({"status": res["status"], "passed": res["passed"], "counts": res["counts"], "result": str(out)})
    return 0 if res["passed"] else 4


def cmd_validate_retrieval(a) -> int:
    """T08.5: retrieval configs on a pre-fixed dev set over a restored copy of this run's product initial state."""
    from .app import build_checker, build_embedder, build_ontology, build_reranker
    from .evaluation.retrieval_validation import load_dev_set, run_retrieval_validation
    from .knowledge.store import KnowledgeStore
    from .memory.snapshot import restore_state
    run, probs = _run_for_tool(a)
    if probs:
        return _blocked(probs)
    p = run.profile
    try:
        run.build_initial_states()
        n = 1
        while (state := run.run_dir / "validation" / f"product-{n}").exists():
            n += 1
        restore_state(run.run_dir / "initial" / "product", state)
        store = KnowledgeStore(state, build_checker(p), run.bundles(), run.plan.set_id, embedder=build_embedder(p),
                               ontology=build_ontology(p),
                               retrieval=p.knowledge.retrieval.model_copy(update={"reranker": "none"}),
                               execution_mode=p.execution_mode)
        try:
            res = run_retrieval_validation(store, load_dev_set(a.devset), _tool_ctx(run, "retrieval_validation"),
                                           reranker=build_reranker(p),
                                           input_lock=Path(a.devset).with_suffix(".lock.json"))
        finally:
            store.close()
    finally:
        run.session_line("validate-retrieval")
    out = _write(run, "retrieval_validation", res)
    _print({"adoption": res.get("adoption"), "result": str(out)})
    return 0 if not res.get("adoption", {}).get("provisional", True) else 4


def cmd_qualify_researcher(a) -> int:
    """T08.3 / spec §10.3: 24 fixed-state tasks + 3 closed-loop types x reps with the NEUTRAL general advisor
    (baseline advisor, fresh memory and restored baseline initial state per episode)."""
    from .advisors.baseline import BaselineAdvisor
    from .advisors.fixture_policy import make_fixture_advisor_policy
    from .app import _llm, build_checker, build_search_provider
    from .config import FixtureBehaviour
    from .evaluation.qualification import (load_closed_loop, load_fixed_state, qualification_verdict,
                                           run_closed_loop, run_fixed_state)
    from .knowledge.build import load_baseline_initial
    from .knowledge.gated import GatedSearch
    from .knowledge.store import KnowledgeStore
    from .memory.baseline import BaselineTextMemory
    from .memory.snapshot import restore_state
    from .researcher.agent import PlannerReviewerResearcher
    from .researcher.fixture_policy import make_fixture_policy
    run, probs = _run_for_tool(a)
    if probs:
        return _blocked(probs)
    p, lim, suite = run.profile, run.profile.limits, Path(a.suite)
    rpolicy = make_fixture_policy(p.fixture or FixtureBehaviour()) if p.roles.researcher.provider == "fixture" else None
    researcher = PlannerReviewerResearcher(_llm(p, "researcher", rpolicy), p.roles.researcher, lim)
    role = p.roles.advisor_baseline
    stores = []

    def make_components(scope, state_dir):
        restore_state(run.run_dir / "initial" / "baseline", state_dir)
        store = KnowledgeStore(state_dir, build_checker(p), run.bundles(), run.plan.set_id,
                               retrieval=p.knowledge.retrieval, execution_mode=p.execution_mode)
        stores.append(store)
        summ = None if p.roles.internal_summarizer.provider == "fixture" else _llm(p, "internal_summarizer")
        memory = BaselineTextMemory(state_dir, scope.memory_scope, summ, lim,
                                    summarizer_role=p.roles.internal_summarizer if summ else None,
                                    gate_derived=store.gate, tasks={})
        policy = make_fixture_advisor_policy("baseline") if role.provider == "fixture" else None
        advisor = BaselineAdvisor(_llm(p, "advisor_baseline", policy), role, lim, memory=memory, store=store,
                                  search=GatedSearch(store, build_search_provider(p, run.guard),
                                                     max_results=p.search.max_results,
                                                     max_chars=lim.max_tool_response_chars),
                                  initial_text=load_baseline_initial(state_dir))
        return researcher, advisor, memory

    lock = suite / "input.lock.json"
    try:
        run.build_initial_states()
        fixed = run_fixed_state(researcher, load_fixed_state(suite / "fixed_state"), _tool_ctx(run, "qualification"),
                                lim, execution_mode=p.execution_mode, input_lock=lock)
        closed = run_closed_loop(make_components, load_closed_loop(suite / "closed_loop"), a.reps,
                                 run.run_dir / "qualification", lim, guard=run.guard, execution_mode=p.execution_mode,
                                 revalidated=a.revalidated, input_lock=lock)
    finally:
        for s in stores:
            s.close()
        run.session_line("qualify-researcher", revalidated=a.revalidated)
    verdict = qualification_verdict(fixed, closed)
    out = _write(run, "qualification", {"fixed_state": fixed, "closed_loop": closed, "verdict": verdict})
    _print({"verdict": verdict["verdict"], "failures": verdict["failures"], "pending": verdict["pending"],
            "development_only": verdict["development_only"], "result": str(out)})
    return 0 if verdict["verdict"] == "pass" else 4


def cmd_resume(a) -> int:
    from .app import Run, run_preflight
    run = Run.load(_run_dir(a))
    probs = run_preflight(run.profile, run.plan)
    if probs:
        return _blocked(probs)
    return _execute(run, a.revalidated)


def cmd_report(a) -> int:
    from .app import Run
    _print(_report(Run.load(_run_dir(a))))
    return 0


def cmd_validate_task(a) -> int:
    from .app import load_private
    from .config import load_public_task
    from .simulators.factory import build_simulator
    from .simulators.task_validation import validate_task, write_report
    p = load_profile(a.profile)
    task = load_public_task(p, a.task_id)
    cfg = p.simulators.get(task.simulator_id)
    if cfg is None:
        return _blocked([f"profile has no simulators.{task.simulator_id}"])
    private = load_private(p, [a.task_id])[a.task_id]
    report = validate_task(task, lambda: build_simulator(task.simulator_id, cfg, task, p.root), private, repeats=a.repeats)
    pub, _ = write_report(report, p.root)
    _print({"task_id": report.task_id, "status": report.status, "reasons": report.reasons, "public_report": str(pub)})
    return 0


def cmd_freeze(a) -> int:
    from .app import freeze
    try:
        body = freeze(load_profile(a.profile), load_set_plan(a.plan), Path(a.analysis_plan), Path(a.out))
    except (ValueError, FileExistsError) as e:
        return _blocked([str(e)])
    _print(body)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m labgene")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, *opts):
        s = sub.add_parser(name)
        for o in opts:
            if o == "profile":
                s.add_argument("--profile", required=True)
            elif o == "plan":
                s.add_argument("--plan", required=True)
            elif o == "plan?":
                s.add_argument("--plan")
            elif o == "run":
                s.add_argument("--run-id", required=True)
                s.add_argument("--artifacts-dir", default="artifacts")
            elif o == "revalidated":
                s.add_argument("--revalidated", action="store_true",
                               help="operator revalidated after a provider model change (spec §10.1)")
        s.set_defaults(fn=fn)
        return s

    add("preflight", cmd_preflight, "profile", "plan?")
    add("build-corpus", cmd_build_corpus, "profile", "plan", "run")
    add("run-set", cmd_run_set, "profile", "plan", "run", "revalidated")
    add("run-evaluation", lambda a: cmd_run_set(a, evaluation=True), "profile", "plan", "run", "revalidated")
    add("resume", cmd_resume, "run", "revalidated")
    add("report", cmd_report, "run")
    s = add("validate-task", cmd_validate_task, "profile")
    s.add_argument("task_id")
    s.add_argument("--repeats", type=int, default=2)
    s = add("freeze", cmd_freeze, "profile", "plan")
    s.add_argument("--analysis-plan", required=True)
    s.add_argument("--out", required=True)
    s = add("validate-gate", cmd_validate_gate, "profile", "plan", "run")
    s.add_argument("--cases", required=True)
    s = add("validate-retrieval", cmd_validate_retrieval, "profile", "plan", "run")
    s.add_argument("--devset", required=True)
    s = add("qualify-researcher", cmd_qualify_researcher, "profile", "plan", "run", "revalidated")
    s.add_argument("--suite", default="configs/qualification")
    s.add_argument("--reps", type=int, default=3)

    a = ap.parse_args(argv)
    from .costs import CapExceeded
    from .faults import InjectedCrash
    from .knowledge.gate import KnowledgeInfraError
    from .providers.base import ModelChangedError
    try:
        return a.fn(a)
    except InjectedCrash as e:     # forced interruption: state is whatever was durably committed
        print(f"interrupted: {e}", file=sys.stderr)
        return 70
    except (CapExceeded, KnowledgeInfraError, ModelChangedError) as e:   # stopped; state kept; resumable
        _print({"status": "stopped", "reason": type(e).__name__, "detail": str(e)})
        return 3
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as e:   # bad input / refused configuration
        _print({"ok": False, "problems": [f"{type(e).__name__}: {e}"]})
        return 2

