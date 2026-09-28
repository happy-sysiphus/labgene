"""T02 adapter / task-validation boundaries. Fixture simulators are contract checks only (development_only)."""
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from labgene.config import SimulatorConfig, load_profile, load_public_task
from labgene.contracts import AnswerBundle, PrivateTaskAssets, canonical_json
from labgene.simulators.base import SimOutput, SimulatorInfraError
from labgene.simulators.factory import build_simulator
from labgene.simulators.task_validation import load_private, registrable_for_evaluation, validate_task
from labgene.simulators.worker import WorkerSimulator, opaque_version

REPO = Path(__file__).resolve().parents[2]
PROFILE = load_profile(REPO / "configs/offline.yaml")

# Fake worker: real labgene serve() loop with a scripted backend. Lives in <tmp>/src so the adapter's
# PYTHONPATH=<root>/src finds it; argv: mode [marker-file].
FAKE = '''
import os, sys, time
sys.path.insert(0, {src!r})
from labgene.simulators.worker import serve
mode, marker = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
if mode == "closedstdin":   # request pipe gone while the worker is still alive
    os.close(0)
    print('{{"ready": true, "version": "fake@1", "units": {{"y": "u"}}}}', flush=True)
    time.sleep(30)
def evaluate(p):
    if mode == "crash" or (mode == "crash_once" and not os.path.exists(marker)):
        open(marker or os.devnull, "w").close()
        os._exit(3)
    if mode == "hang":
        time.sleep(30)
    if mode == "garbage":
        print("this is not json", flush=True)
    if mode == "error":
        raise KeyError("LEAK-RIDGE-7Q2 temperature=83.0")
    if mode == "nan":
        return {{"y": float("nan")}}
    return {{"y": p["x"] * 2.0}}
serve(("fake@2" if mode == "badversion" else "fake@1", {{"y": "u"}}, evaluate), sys.stdin, sys.stdout)
'''


@pytest.fixture
def fake_root(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "fakeworker.py").write_text(FAKE.format(src=str(REPO / "src")), encoding="utf-8")
    return tmp_path


def fake_sim(root, mode, marker="", metrics=("y",), timeout_s=10.0, python=sys.executable):
    return WorkerSimulator("fake.sim", opaque_version("fake@1"), python=python, root=root, metrics=list(metrics),
                           module="fakeworker", args=[mode, marker], timeout_s=timeout_s)


def test_worker_module_imports_without_pydantic():
    """Worker envs have no pydantic/yaml: the entrypoint module must import with them absent."""
    code = ("import sys; sys.modules['pydantic'] = None; sys.modules['yaml'] = None\n"
            "import labgene.simulators.worker as w; print(w.opaque_version('x'))")
    out = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(REPO / "src")},
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == opaque_version("x")


def test_b03_new_request_same_parameters_same_value_cache_hit_still_returned(fake_root):
    sim = fake_sim(fake_root, "ok")
    try:
        a, b, c = sim.evaluate({"x": 1.5}), sim.evaluate({"x": 1.5}), sim.evaluate({"x": 2.0})
    finally:
        sim.close()
    assert (a.cache_hit, b.cache_hit, c.cache_hit) == (False, True, False)
    assert a.results == b.results == {"y": 3.0} and c.results == {"y": 4.0}   # a hit is a full result: harness charges it
    assert a.units == {"y": "u"} and a.simulator_version == opaque_version("fake@1")


@pytest.mark.parametrize("mode,kw", [
    ("crash", {}), ("hang", {"timeout_s": 1.0}), ("garbage", {}), ("error", {}), ("nan", {}),
    ("badversion", {}), ("ok", {"metrics": ("y", "z")}), ("ok", {"python": "C:/no/such/python.exe"}),
])
def test_b04_worker_failures_are_infra_errors_without_detail(fake_root, mode, kw):
    sim = fake_sim(fake_root, mode, **kw)
    with pytest.raises(SimulatorInfraError) as e:
        sim.evaluate({"x": 7.25})
    msg = str(e.value)
    assert "LEAK" not in msg and "83.0" not in msg and "7.25" not in msg
    assert sim._proc is None   # failed worker is torn down, never reused


def test_b04_worker_request_pipe_closed_is_infra_error(fake_root):
    """Finding 3: a failed stdin write must not leak a raw OSError from close()."""
    sim = fake_sim(fake_root, "closedstdin", timeout_s=5.0)
    with pytest.raises(SimulatorInfraError):
        sim.evaluate({"x": 1.0})
    assert sim._proc is None
    sim.close()   # idempotent


FAKE_ALD = "class FastFast:\n    def __init__(self, noise, round_to):\n        pass\n    def __call__(self, t1, t2):\n        return {k} * (t1 + t2)\n"


def test_b03_edited_upstream_checkout_cannot_reuse_simulator_version(tmp_path):
    """Finding 4: same simulator_version must mean same numbers; an edited pinned clone fails the handshake."""
    up = tmp_path / "aldenv"
    (up / ".git").mkdir(parents=True)
    (up / ".git" / "HEAD").write_text("0" * 40 + "\n", encoding="utf-8")
    pkg = up / "src" / "aldenv" / "envs"
    pkg.mkdir(parents=True)
    for f in (up / "src" / "aldenv" / "__init__.py", pkg / "__init__.py"):
        f.write_text("", encoding="utf-8")
    (pkg / "steadystate.py").write_text(FAKE_ALD.format(k="1.0"), encoding="utf-8")
    args = ["--backend", "aldenv", "--source", str(up)]
    hello = subprocess.run([sys.executable, "-m", "labgene.simulators.worker", *args], input="", capture_output=True,
                           text=True, timeout=60, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    version = opaque_version(json.loads(hello.stdout.splitlines()[0])["version"])
    make = lambda: WorkerSimulator("ald", version, python=sys.executable, root=REPO, metrics=["gpc"], args=args)
    sim = make()
    try:
        assert sim.evaluate({"t1": 0.25, "t2": 0.5}).results == {"gpc": 0.75}
    finally:
        sim.close()
    (pkg / "steadystate.py").write_text(FAKE_ALD.format(k="2.25"), encoding="utf-8")
    sim = make()
    with pytest.raises(SimulatorInfraError):
        sim.evaluate({"t1": 0.25, "t2": 0.5})


def test_b04_worker_restarts_after_crash_and_requery_is_identical(fake_root, tmp_path):
    sim = fake_sim(fake_root, "crash_once", marker=str(tmp_path / "crashed"))
    try:
        with pytest.raises(SimulatorInfraError):
            sim.evaluate({"x": 1.25})
        again = sim.evaluate({"x": 1.25})     # recovery re-queries the deterministic simulator
    finally:
        sim.close()
    assert again.results == {"y": 2.5} and not again.cache_hit


def test_b04_fixture_simulators_deterministic_across_instances():
    for sid, task_id, p in [("fixture.ridge", "fixture_ridge", {"temperature": 70.0, "time": 20.0}),
                            ("fixture.catalyst", "fixture_catalyst",
                             {"catalyst": "B", "loading": 1.1, "temperature": 75.0, "residence_time": 300.0})]:
        task = load_public_task(PROFILE, task_id)
        outs = [build_simulator(sid, PROFILE.simulators[sid], task, REPO).evaluate(p) for _ in range(3)]
        assert all(o.results == outs[0].results for o in outs)
        assert outs[0].simulator_version == task.simulator_version


def _fixture_report(task_id, **evidence):
    task = load_public_task(PROFILE, task_id)
    private = load_private(REPO, task_id)
    data = evidence.pop("_data", None)
    if evidence:
        private = PrivateTaskAssets(**{**private.model_dump(), "validity_evidence": evidence})
    make = lambda: build_simulator(task.simulator_id, PROFILE.simulators[task.simulator_id], task, REPO)
    return task, private, validate_task(task, make, private, data=data)


@pytest.mark.parametrize("task_id", ["fixture_ridge", "fixture_catalyst"])
def test_b23_fixture_tasks_never_validated_or_registrable(task_id):
    task, _, report = _fixture_report(task_id)
    assert report.status == "unvalidated" and report.development_only
    assert any("artificial fixture" in r for r in report.reasons)
    assert report.determinism["identical"] and report.known_success_summary["succeeded"] == 1
    assert not registrable_for_evaluation(report, task)


COMPLETE = {"kind": "data_trained_surrogate", "paper": {"doi": "10.5555/x", "version": "v1"},
            "code": {"repo": "r", "commit": "c"}, "license": {"code": "MIT"}, "mechanistic": True,
            "metric_transforms": {"yield": "identity"}, "success_rule_mapping": "paper goal -> yield >= 89"}


def test_b23_validated_only_when_every_evidence_item_exists():
    data = [{"parameters": {"temperature": 83.0, "time": 37.0}, "observed": {"yield": 94.0}},
            {"parameters": {"temperature": 60.0, "time": 20.0}, "observed": {"yield": 30.0}}]
    task, _, full = _fixture_report("fixture_ridge", **COMPLETE, _data=data)
    assert full.status == "validated" and not full.reasons, full.reasons
    assert registrable_for_evaluation(full, task)
    assert not registrable_for_evaluation(full, task.model_copy(update={"version": "2"}))   # stale report
    for drop in ["paper", "code", "license", "metric_transforms", "success_rule_mapping", "mechanistic"]:
        ev = {k: v for k, v in COMPLETE.items() if k != drop}
        _, _, r = _fixture_report("fixture_ridge", **ev, _data=data)
        assert r.status == "unvalidated" and not registrable_for_evaluation(r, task), drop
    for extra in [{"open_questions": ["target unclear"]}, {"development_only": True}]:
        _, _, r = _fixture_report("fixture_ridge", **COMPLETE, **extra, _data=data)
        assert r.status == "unvalidated", extra
    _, _, no_data = _fixture_report("fixture_ridge", **COMPLETE)
    assert no_data.status == "unvalidated" and any("no observed data" in r for r in no_data.reasons)


def test_b13_public_projection_excludes_success_inputs():
    _, private, report = _fixture_report("fixture_ridge")
    public = json.dumps(report.public())
    assert "private" not in report.public()
    assert report.private.known_success and report.private.known_success[0]["success"]
    for x in private.known_success_inputs:
        assert canonical_json(x) not in public and '"temperature": 83.0' not in public
    for marker in private.answer_bundle.secret_markers:
        assert marker not in public and marker not in report.model_dump_json()


class _Stub:
    """Scripted adapter for validation-gate tests: f(params) -> results; optional per-instance cache."""
    def __init__(self, task, f, cache=False):
        self.simulator_id, self.simulator_version, self._f = task.simulator_id, task.simulator_version, f
        self._cache, self.seen = ({} if cache else None), []

    def evaluate(self, p):
        self.seen.append(p)
        k = canonical_json(p)
        hit = self._cache is not None and k in self._cache
        results = self._cache[k] if hit else self._f(p)
        if self._cache is not None:
            self._cache[k] = results
        return SimOutput(results=results, units={m: "u" for m in results}, simulator_id=self.simulator_id,
                         simulator_version=self.simulator_version, cache_hit=hit)


def _assets(task_id, evidence, known):
    return PrivateTaskAssets(task_id=task_id, validity_evidence=evidence, known_success_inputs=known,
                             answer_bundle=AnswerBundle(bundle_id="b", version="1", set_scope="*", blocked_documents=[]))


RIDGE_DATA = [{"parameters": {"temperature": 83.0, "time": 37.0}, "observed": {"yield": 94.0}},
              {"parameters": {"temperature": 60.0, "time": 20.0}, "observed": {"yield": 30.0}}]


def test_b23_reused_cached_adapter_is_not_determinism_evidence():
    """Finding 5: a non-deterministic simulator hidden behind a surviving cache must not validate."""
    task = load_public_task(PROFILE, "fixture_ridge")
    one = _Stub(task, lambda p: {"yield": 95.0 - random.random() * 1e-3}, cache=True)
    r = validate_task(task, lambda: one, _assets(task.task_id, COMPLETE, [{"temperature": 83.0, "time": 37.0}]),
                      data=RIDGE_DATA)
    assert not r.determinism["fresh_adapter_per_repeat"] and r.status == "unvalidated"
    assert not registrable_for_evaluation(r, task)
    fresh = validate_task(task, lambda: _Stub(task, lambda p: {"yield": 95.0}, cache=True),
                          _assets(task.task_id, COMPLETE, [{"temperature": 83.0, "time": 37.0}]), data=RIDGE_DATA)
    assert fresh.status == "validated", fresh.reasons


def test_b23_fixture_backend_cannot_stand_in_for_worker_task():
    """Finding 6: a fixture never serves (or carries the version of) a real worker-simulated task."""
    task = load_public_task(PROFILE, "aldenv_fastfast")
    with pytest.raises(ValueError, match="fixture"):
        build_simulator(task.simulator_id, SimulatorConfig(backend="fixture", fixture_function="ridge"), task, REPO)
    ridge = load_public_task(PROFILE, "fixture_ridge")
    with pytest.raises(ValueError, match="fixture"):   # fixture function without the task's metrics
        build_simulator(ridge.simulator_id, SimulatorConfig(backend="fixture", fixture_function="catalyst"), ridge, REPO)


def test_b23_constrained_task_range_evidence_covers_feasible_region():
    """Finding 7: with a budget constraint most box corners are infeasible; probes are pulled onto the
    constraint boundary instead of resting on one feasible corner. Probe outputs stay private."""
    task = load_public_task(PROFILE, "aldenv_fastfast")   # t1 + t2 <= 1.0; only (min, min) corner feasible
    stub = _Stub(task, lambda p: {"gpc": min(p["t1"], p["t2"])})
    r = validate_task(task, lambda: stub, _assets(task.task_id, {}, []), repeats=1)
    pts = {canonical_json(p) for p in stub.seen}
    assert r.range_check["feasible"] == r.range_check["probes"] == 5 and len(pts) >= 4
    assert all(p["t1"] + p["t2"] <= 1.0 + 1e-9 for p in stub.seen)
    assert sum(abs(p["t1"] + p["t2"] - 1.0) < 1e-6 for p in map(json.loads, pts)) >= 3
    assert "max" not in json.dumps(r.public()["range_check"])
    # extremes cover probes only (duplicate boundary probes must not shift the evaluator-held input into them)
    r2 = validate_task(task, lambda: _Stub(task, lambda p: {"gpc": 5.0 if p["t1"] == 0.45 else 0.1}),
                       _assets(task.task_id, {}, [{"t1": 0.45, "t2": 0.5}]), repeats=1)
    assert r2.private.range_extremes["max"]["gpc"] == 0.1


def test_b23_known_success_through_model_identity_violation_blocks_validation():
    """Finding 1: success that exists only because the model breaks a declared identity (TON = yield/loading)
    is not evidence of a reachable goal."""
    task = load_public_task(PROFILE, "fixture_catalyst")
    ev = {**COMPLETE, "metric_transforms": {"yield": "identity", "ton": "identity"},
          "identities": [{"metric": "ton", "numerator": "yield", "denominator": "loading"}]}
    data = [{"parameters": {"catalyst": "C", "loading": 1.2, "temperature": 90.0, "residence_time": 540.0},
             "observed": {"yield": 98.0, "ton": 81.0}},
            {"parameters": {"catalyst": "A", "loading": 0.5, "temperature": 40.0, "residence_time": 60.0},
             "observed": {"yield": 2.0, "ton": 4.0}}]
    const = lambda: _Stub(task, lambda p: {"yield": 90.0, "ton": 60.0})   # TON independent of loading
    at = lambda loading: [{"catalyst": "C", "loading": loading, "temperature": 90.0, "residence_time": 540.0}]
    bad = validate_task(task, const, _assets(task.task_id, ev, at(2.5)), data=data)   # 90/2.5 = 36 < 57
    assert bad.status == "unvalidated" and any("identity" in x for x in bad.reasons), bad.reasons
    assert bad.known_success_summary["succeeded"] == 1 and bad.known_success_summary["identity_consistent"] == 0
    ok = validate_task(task, const, _assets(task.task_id, ev, at(1.2)), data=data)     # 90/1.2 = 75 >= 57
    assert ok.status == "validated", ok.reasons


def test_b13_real_private_answers_absent_from_publishable_files():
    """Finding 2: evaluator-held inputs (git-ignored private/*.yaml) appear in no committable file."""
    privs = sorted((REPO / "private").glob("*.yaml"))
    if not privs:
        pytest.skip("no real private assets under private/")
    answers = []
    for p in privs:
        a = PrivateTaskAssets.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
        answers += a.known_success_inputs + [r["parameters"] for r in a.validity_evidence.get("reference_points", [])]
    num = re.compile(r"-?\d+(?:\.\d+)?")
    files = [f for d in ("src", "tests", "configs", "docs") for f in (REPO / d).rglob("*")
             if f.suffix in {".py", ".yaml", ".yml", ".json", ".md", ".txt"}]
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for i in range(len(lines)):
            window = "\n".join(lines[i:i + 3])
            nums = {float(x) for x in num.findall(window)}
            for x in answers:
                if all(v in window if isinstance(v, str) else float(v) in nums for v in x.values()):
                    pytest.fail(f"{f.relative_to(REPO)}:{i + 1} holds an evaluator-held input")
