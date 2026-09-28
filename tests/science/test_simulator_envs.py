"""Real simulator environments (T02-ext). Skip cleanly when the isolated env / pinned clone / private assets
are missing. Rebuild: docs/implementation/task-validation/README.md."""
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from labgene.config import SimulatorConfig
from labgene.contracts import PublicTask
from labgene.simulators.factory import build_simulator
from labgene.simulators.task_validation import SIMULATORS_FILE, load_private, registrable_for_evaluation, validate_task
from labgene.simulators.validation import validate_parameters

pytestmark = pytest.mark.science
ROOT = Path(__file__).resolve().parents[2]
SIMS = yaml.safe_load((ROOT / SIMULATORS_FILE).read_text(encoding="utf-8"))


def setup(task_id: str, env: str):
    if not (ROOT / ".envs" / env / "Scripts" / "python.exe").exists() or not (ROOT / ".envs" / "src" / env / ".git").exists():
        pytest.skip(f".envs/{env} or its pinned clone is missing")
    task = PublicTask.model_validate(yaml.safe_load((ROOT / "configs/tasks" / f"{task_id}.yaml").read_text(encoding="utf-8")))
    cfg = SimulatorConfig.model_validate(SIMS[task.simulator_id])
    return task, (lambda: build_simulator(task.simulator_id, cfg, task, ROOT))


def private_or_skip(task_id: str):
    try:
        return load_private(ROOT, task_id)
    except FileNotFoundError:
        pytest.skip(f"private/{task_id}.yaml missing")


def evaluate_fresh(make, params, n=2):
    outs = []
    for _ in range(n):
        sim = make()
        try:
            outs.append(sim.evaluate(params))
        finally:
            sim.close()
    return outs


@pytest.mark.parametrize("t,gpc", [(0.2, 0.417024), (1.0, 0.975190), (5.0, 1.0)])
def test_aldenv_reproduces_research_smoke_in_fresh_workers(t, gpc):
    # Evaluator-side probe: (5, 5) lies outside the task's public dose budget, so it bypasses task validation.
    _, make = setup("aldenv_fastfast", "aldenv")
    a, b = evaluate_fresh(make, {"t1": t, "t2": t})
    assert a.results == b.results == {"gpc": gpc} and not a.cache_hit and not b.cache_hit


def test_aldenv_connection_check_report_is_development_only():
    task, make = setup("aldenv_fastfast", "aldenv")
    r = validate_task(task, make, private_or_skip("aldenv_fastfast"), repeats=2)
    assert r.determinism["identical"] and r.known_success_summary == {"checked": 1, "valid": 1, "succeeded": 1}
    assert r.simulator_descriptor.startswith("aldenv@90055ef811134f4f3b569a088491d729746538fd;")
    assert r.status == "unvalidated" and r.development_only and not registrable_for_evaluation(r, task)


def test_summit_task_ranges_match_upstream_domain():
    task, _ = setup("summit_reizman_case1", "summit")
    code = ("import sys, json; sys.path.insert(0, '.envs/src/summit')\n"
            "from summit.benchmarks import ReizmanSuzukiEmulator as E\n"
            "d = E.setup_domain(); print(json.dumps({v.name: (list(v.levels) if hasattr(v, 'levels') else"
            " [float(b) for b in v.bounds]) for v in d.input_variables}))")
    out = subprocess.run([str(ROOT / ".envs/summit/Scripts/python.exe"), "-c", code], cwd=ROOT, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    up = json.loads(out.stdout.strip().splitlines()[-1])
    p = {x.name: x for x in task.parameters}
    assert p["catalyst"].choices == up["catalyst"] and len(up["catalyst"]) == 8
    for mine, theirs in [("residence_time", "t_res"), ("temperature", "temperature"), ("catalyst_loading", "catalyst_loading")]:
        assert [p[mine].min, p[mine].max] == up[theirs]


def test_summit_candidate_deterministic_paper_optimum_and_never_validated():
    task, make = setup("summit_reizman_case1", "summit")
    private = private_or_skip("summit_reizman_case1")
    # The paper-optimum conditions and the model's pinned outputs there are evaluator-held (git-ignored).
    ref = private.validity_evidence["reference_points"][0]
    a, b = evaluate_fresh(make, validate_parameters(task, ref["parameters"]).parameters)
    assert a.results == b.results == pytest.approx(ref["model_output_pinned"], abs=1e-3)
    r = validate_task(task, make, private, repeats=2)
    assert r.determinism["identical"] and r.range_check["all_metrics_finite"]
    assert r.known_success_summary["succeeded"] == r.known_success_summary["checked"] >= 1
    assert r.status == "unvalidated" and r.open_questions and not registrable_for_evaluation(r, task)
