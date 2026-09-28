"""Reporting at the evaluation-changing boundaries (B21, B22, B23 part, public projection).
Ledgers are built with direct Ledger calls. Offline fixture data: contract checks only (development_only)."""
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from labgene.config import SetPlan, load_profile, load_public_task
from labgene.contracts import (ActionEnvelope, ActionKind, AdvisorResponse, ConsultExchange, CostEvent,
                               ExperimentError, FinalizationStatus, Observation, Outcome, ProtocolError, RunManifest,
                               RunScope, Usage, payload_hash)
from labgene.harness.ledger import Ledger
from labgene.memory.base import FinalizeResult
from labgene.reporting import report
from labgene.reporting.report import BOOTSTRAP_B, BOOTSTRAP_SEED, build_report, write_report

ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_profile(ROOT / "configs/offline.yaml")
RIDGE = load_public_task(PROFILE, "fixture_ridge")
CATALYST = load_public_task(PROFILE, "fixture_catalyst")
TASKS = {t.task_id: t for t in (RIDGE, CATALYST)}
PRIVATE_DIR = PROFILE.resolve(PROFILE.paths.private_dir)
G = PROFILE.roles.researcher.model
MISS = {"fixture_ridge": {"yield": 50.0}, "fixture_catalyst": {"yield": 50.0, "ton": 20.0}}
HIT = {"fixture_ridge": {"yield": 95.0}, "fixture_catalyst": {"yield": 81.0, "ton": 58.0}}


def manifest(plan, mode="offline_fixture", **kw):
    m = RunManifest(run_id="run", created_at="2026-09-28T00:00:00+00:00", execution_mode=mode, spec_sha256="spec",
                    code_version={"source_tree_sha256": "tree"}, profile_hash=PROFILE.hash,
                    profile=PROFILE.model_dump(mode="json"), set_plan_hash=plan.hash,
                    set_plan=plan.model_dump(mode="json"), tasks={t: payload_hash(v) for t, v in TASKS.items()},
                    initial_state_hashes={"baseline": "ib", "product": "ip"},
                    models={n: r.model_dump(mode="json") for n, r in PROFILE.roles}).model_dump(mode="json")
    return {**m, "task_defs": {t: v.model_dump(mode="json") for t, v in TASKS.items()}, **kw}


class Ep:
    """Direct ledger writes for one episode (the same calls EpisodeRunner makes)."""

    def __init__(self, L, cond, rep, order, task_id, visit=1):
        self.L, self.task, self.n = L, TASKS[task_id], 0
        self.scope = RunScope(run_id="run", condition=cond, set_id="smoke", set_rep=rep,
                              episode_id=f"{cond}-r{rep}-e{order:03d}", episode_order=order, task_id=task_id,
                              visit_index=visit)
        self.eid = self.scope.episode_id
        L.start_episode(self.scope, f"start-{self.eid}")

    def _act(self, kind, args, make, commit=True):
        self.n += 1
        env = ActionEnvelope(action_id=f"{self.eid}:a{self.n:03d}", scope=self.scope, raw_text="{}", kind=kind,
                             args=args, payload_hash=payload_hash({"kind": kind.value, "args": args}))
        self.L.reserve(env, self.n, self.n)
        if commit:
            self.L.commit(env.action_id, make(env.action_id))
        return self

    def consult(self, answer="Try the middle.", commit=True):
        return self._act(ActionKind.consult, {"question": "Where?"}, lambda a: ConsultExchange(
            action_id=a, question="Where?", response=AdvisorResponse(answer=answer)), commit)

    def run(self, results, params=None, hypothesis="h"):
        p = params or {"x": 1}
        return self._act(ActionKind.run_experiment, {"hypothesis": hypothesis, "parameters": p}, lambda a: Observation(
            observation_id=f"obs:{a}", action_id=a, scope=self.scope, parameters=p, results=results, units={},
            simulator_id=self.task.simulator_id, simulator_version="1", meets_success_criteria=self.task.is_success(results),
            created_at="t"))

    def invalid(self, params, reason):
        return self._act(ActionKind.run_experiment, {"hypothesis": "h", "parameters": params}, lambda a: ExperimentError(
            action_id=a, scope=self.scope, submitted_parameters=params, reason=reason))

    def end(self, outcome, reason=None, fin=FinalizationStatus.done, fin_reason=None):
        n = self.L.success_actions(self.eid) if outcome is Outcome.success else None
        self.L.set_outcome(self.eid, outcome, reason=reason, actions_to_success=n)
        if fin is not FinalizationStatus.not_started:
            self.L.set_finalization(self.eid, fin, f"end-{self.eid}" if fin is FinalizationStatus.done else None, fin_reason)
        return self


def success(L, cond, rep, order, task_id, visit=1, misses=1):
    e = Ep(L, cond, rep, order, task_id, visit).consult()
    for _ in range(misses):
        e.run(MISS[task_id])
    return e.run(HIT[task_id]).end(Outcome.success)


def exhausted(L, cond, rep, order, task_id, visit=1):
    e = Ep(L, cond, rep, order, task_id, visit).consult().invalid({"x": 999}, "x outside the allowed range")
    for _ in range(48):
        e.run(MISS[task_id])
    return e.end(Outcome.budget_exhausted)


def rows(text):
    return list(csv.DictReader(io.StringIO(text)))


PLAN3 = SetPlan(set_id="smoke", reps=3, episodes=["fixture_ridge", "fixture_catalyst", "fixture_ridge"])


def three_rep_ledger(tmp_path):
    """rep1: baseline 2/3, product 3/3. rep2: 2/3 vs 2/3. rep3: baseline complete, product infra_incomplete + not
    started (the SetRunner stop), so rep3 must be excluded from pairing."""
    L = Ledger(tmp_path)
    success(L, "baseline", 1, 1, "fixture_ridge")
    exhausted(L, "baseline", 1, 2, "fixture_catalyst")
    success(L, "baseline", 1, 3, "fixture_ridge", visit=2, misses=0)
    for o, t, v in PLAN3.visits():
        success(L, "product", 1, o, t, v)
    success(L, "baseline", 2, 1, "fixture_ridge")
    success(L, "baseline", 2, 2, "fixture_catalyst")
    exhausted(L, "baseline", 2, 3, "fixture_ridge", visit=2)
    success(L, "product", 2, 1, "fixture_ridge")
    exhausted(L, "product", 2, 2, "fixture_catalyst")
    success(L, "product", 2, 3, "fixture_ridge", visit=2)
    for o, t, v in PLAN3.visits():
        success(L, "baseline", 3, o, t, v)
    success(L, "product", 3, 1, "fixture_ridge")
    Ep(L, "product", 3, 2, "fixture_catalyst").consult(commit=False).end(Outcome.infra_incomplete, "advisor_infra",
                                                                           fin=FinalizationStatus.not_started)
    L.close()


def test_b21_episode_table_set_metrics_revisit_split_and_complete_pairs_only(tmp_path):
    three_rep_ledger(tmp_path)
    ledger_bytes = hashlib.sha256((tmp_path / "ledger.sqlite").read_bytes()).hexdigest()
    paths = write_report(tmp_path, manifest(PLAN3))
    assert hashlib.sha256((tmp_path / "ledger.sqlite").read_bytes()).hexdigest() == ledger_bytes   # read-only
    rep = json.loads(paths["report.json"].read_text(encoding="utf-8"))

    # failed episodes carry no actions_to_success (never 51 / max+1); successes carry the real count
    eps = {r["episode_id"]: r for r in rows(paths["episodes.csv"].read_text(encoding="utf-8"))}
    assert (eps["baseline-r1-e002"]["outcome"], eps["baseline-r1-e002"]["actions_used"],
            eps["baseline-r1-e002"]["actions_to_success"], eps["baseline-r1-e002"]["success"]) == \
        ("budget_exhausted", "50", "", "false")
    assert [eps[f"baseline-r1-e00{i}"]["actions_to_success"] for i in (1, 3)] == ["3", "2"]
    assert eps["baseline-r1-e003"]["visit"] == "revisit" and eps["baseline-r1-e001"]["visit"] == "first"
    assert eps["product-r3-e002"]["outcome_reason"] == "advisor_infra" and eps["product-r3-e002"]["success"] == ""
    assert all(r["actions_to_success"] in ("", "2", "3") for r in eps.values())
    not_started = [r for r in rep["episodes"] if r["outcome"] == "not_started"]
    assert [(r["condition"], r["set_rep"], r["episode_order"], r["episode_id"]) for r in not_started] == \
        [("product", 3, 3, None)]

    sets = {(s["condition"], s["set_rep"]): s for s in rep["sets"]}
    b1 = sets[("baseline", 1)]
    assert b1["complete"] and b1["success_at_50"] == {"successes": 2, "n": 3, "rate": 2 / 3}
    assert (b1["first_visit"], b1["revisit"]) == ({"successes": 1, "n": 2, "rate": 0.5},
                                                  {"successes": 1, "n": 1, "rate": 1.0})
    assert {k: b1["actions_to_success"][k] for k in ("n", "terminal_n", "success_rate", "median", "min", "max")} == \
        {"n": 2, "terminal_n": 3, "success_rate": 2 / 3, "median": 2.5, "min": 2, "max": 3}
    assert [p["outcome"] for p in b1["progression"]] == ["success", "budget_exhausted", "success"]
    p3 = sets[("product", 3)]
    assert not p3["complete"] and p3["success_at_50"] == {"successes": 1, "n": 1, "rate": 1.0}
    assert {k: p3["counts"][k] for k in ("success", "infra_incomplete", "not_started", "budget_exhausted")} == \
        {"success": 1, "infra_incomplete": 1, "not_started": 1, "budget_exhausted": 0}
    assert p3["incomplete_reasons"] == ["episode 2: infra_incomplete (advisor_infra)", "episode 3: not_started"]

    pc = rep["paired_comparison"]
    assert [p["set_rep"] for p in pc["pairs"]] == [1, 2]
    assert [p["difference"] for p in pc["pairs"]] == pytest.approx([1 / 3, 0.0])
    assert pc["mean_difference"] == pytest.approx(1 / 6)
    assert pc["n_excluded"] == 1 and pc["excluded"][0]["set_rep"] == 3
    assert "product incomplete" in pc["excluded"][0]["reason"] and "infra_incomplete" in pc["excluded"][0]["reason"]
    assert "NOT independent" in pc["unit"] and "set pair" in pc["unit"]
    bt = pc["bootstrap"]
    assert (bt["seed"], bt["B"]) == (BOOTSTRAP_SEED, BOOTSTRAP_B)
    assert bt["interval"] == pytest.approx([0.0, 1 / 3])   # resampled means of [1/3, 0] lie in {0, 1/6, 1/3}
    assert build_report(tmp_path, manifest(PLAN3))["paired_comparison"] == pc   # fixed seed: deterministic
    co = rep["conditions"]
    assert co["product"]["complete_reps"] == [1, 2] and co["product"]["incomplete_reps"] == [3]
    # baseline rep 3 is complete but unpaired: condition summaries use only reps complete in BOTH conditions,
    # so the side-by-side means differ by exactly the paired mean difference (never over different rep sets)
    assert co["baseline"]["complete_reps"] == [1, 2, 3] and co["baseline"]["analysis_reps"] == [1, 2]
    assert co["product"]["mean_success_at_50"] - co["baseline"]["mean_success_at_50"] == pytest.approx(1 / 6)
    assert [x["successes"] for x in co["baseline"]["by_episode_order"]] == [2, 1, 1]
    md = paths["report.md"].read_text(encoding="utf-8")
    assert "excluded set reps: 1" in md and "product rep 3 INCOMPLETE" in md


def test_b21_bootstrap_is_reproducible_from_its_recorded_seed(tmp_path, monkeypatch):
    three_rep_ledger(tmp_path)
    monkeypatch.setattr(report, "BOOTSTRAP_B", 9)   # tiny B: an unseeded resampler would differ between calls
    runs = [build_report(tmp_path, manifest(PLAN3))["paired_comparison"]["bootstrap"] for _ in range(5)]
    assert all(r == runs[0] for r in runs) and (runs[0]["seed"], runs[0]["B"]) == (BOOTSTRAP_SEED, 9)

def test_b21_finalization_failure_keeps_success_and_marks_set_incomplete(tmp_path):
    plan = SetPlan(set_id="smoke", reps=1, episodes=["fixture_ridge", "fixture_catalyst"])
    L = Ledger(tmp_path)
    Ep(L, "baseline", 1, 1, "fixture_ridge").consult().run(MISS["fixture_ridge"]).run(HIT["fixture_ridge"]) \
        .end(Outcome.success, fin=FinalizationStatus.failed, fin_reason="memory_error")
    L.close()
    rep = build_report(tmp_path, manifest(plan))
    e = rep["episodes"][0]
    assert (e["outcome"], e["success"], e["actions_to_success"], e["finalization"], e["finalization_reason"]) == \
        ("success", True, 3, "failed", "memory_error")
    b, p = rep["sets"]
    assert not b["complete"] and b["counts"]["finalization_failed"] == 1 and b["counts"]["not_started"] == 1
    assert b["incomplete_reasons"] == ["episode 1: finalization failed (memory_error)", "episode 2: not_started"]
    assert p["counts"]["not_started"] == 2 and p["success_at_50"] == {"successes": 0, "n": 0, "rate": None}
    pc = rep["paired_comparison"]
    assert (pc["n_pairs"], pc["n_excluded"], pc["mean_difference"], pc["bootstrap"]["interval"]) == (0, 1, None, None)


def test_b21_action_table_best_so_far_is_multi_metric_aware(tmp_path):
    plan = SetPlan(set_id="smoke", reps=1, conditions=["baseline"], episodes=["fixture_catalyst", "fixture_ridge"])
    L = Ledger(tmp_path)
    (Ep(L, "baseline", 1, 1, "fixture_catalyst").run({"yield": 85.0, "ton": 40.0}).run({"yield": 70.0, "ton": 65.0})
     .invalid({"catalyst": "Z"}, "catalyst must be one of ['A', 'B', 'C', 'D']").consult()
     .run({"yield": 81.0, "ton": 58.0}).end(Outcome.success))
    (Ep(L, "baseline", 1, 2, "fixture_ridge").run({"yield": 50.0}).run({"yield": 70.0}).run({"yield": 60.0})
     .end(Outcome.budget_exhausted))
    L.close()
    paths = write_report(tmp_path, manifest(plan))
    rep = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    cat = [a for a in rep["actions"] if a["task_id"] == "fixture_catalyst"]
    # thresholds yield 78, ton 57: (85,40) gap 17/57; (70,65) gap 8/78 is closer -> becomes best and is carried
    assert [(a["action_number"], a["kind"], a["counted_as"], a["best_so_far_action_id"].rsplit(":", 1)[1],
             round(a["best_so_far_gap"], 4), a["meets_success_criteria"]) for a in cat] == [
        (1, "run_experiment", "evaluation", "a001", round(17 / 57, 4), False),
        (2, "run_experiment", "evaluation", "a002", round(8 / 78, 4), False),
        (3, "run_experiment", "invalid", "a002", round(8 / 78, 4), None),
        (4, "consult", "consultation", "a002", round(8 / 78, 4), None),
        (5, "run_experiment", "evaluation", "a005", 0.0, True)]
    assert cat[2]["parameters"] == {"catalyst": "Z"} and cat[2]["invalid_reason"].startswith("catalyst must be one of")
    assert cat[2]["results"] is None and cat[3]["parameters"] is None and cat[4]["results"] == {"yield": 81.0, "ton": 58.0}
    ridge = [a["best_so_far_results"]["yield"] for a in rep["actions"] if a["task_id"] == "fixture_ridge"]
    assert ridge == [50.0, 70.0, 70.0]   # single metric: best value so far
    assert rep["episodes"][0]["actions_to_success"] == 5 and rep["episodes"][1]["actions_to_success"] is None
    acsv = rows(paths["actions.csv"].read_text(encoding="utf-8"))
    assert len(acsv) == 8 and json.loads(acsv[2]["parameters"]) == {"catalyst": "Z"} and acsv[3]["results"] == ""
    assert "best_so_far" in rep["definitions"] and rep["paired_comparison"]["available"] is False


def test_b22_costs_unavailable_usage_phases_and_finalization_adds_no_actions(tmp_path):
    plan = SetPlan(set_id="smoke", reps=1, episodes=["fixture_ridge"])
    L = Ledger(tmp_path)
    b = success(L, "baseline", 1, 1, "fixture_ridge", misses=0)
    p = success(L, "product", 1, 1, "fixture_ridge", misses=0)
    rt = dict(phase="runtime", scope_key="run/baseline/smoke/rep1", episode_id=b.eid)
    for e in [
        CostEvent(kind="llm_call", role="researcher_finalizer", model=G, **rt),                       # usage n/a
        CostEvent(kind="llm_call", role="researcher_finalizer", model=G, status="infra_error", attempt=2, **rt),
        CostEvent(kind="llm_call", role="advisor_baseline", provider="gemini", model=G, latency_s=1.5,
                  usage=Usage(input_tokens=100, output_tokens=20), detail={"model_returned": G}, **rt),
        CostEvent(kind="simulator", role="simulator", model="fixture.ridge", latency_s=0.25, **rt),
        *(CostEvent(kind=k, role=k, **rt) for k in ("gate", "retrieval", "search")),   # no tokens by nature
        CostEvent(kind="llm_call", role="advisor_product", provider="gemini", model=G, phase="runtime",
                  episode_id=p.eid, usage=Usage(input_tokens=50), detail={"model_returned": "g-other"}),
        CostEvent(kind="finalize", role="memory_baseline", phase="finalization", episode_id=b.eid, latency_s=0.1),
        CostEvent(kind="llm_call", role="internal_summarizer", phase="finalization", episode_id=b.eid,
                  model="fixture-summarizer", usage=Usage(input_tokens=7, output_tokens=3),
                  detail={"model_returned": "fixture-summarizer"}),
    ]:
        L.record_cost(e)
    L.close()
    pre = [CostEvent(kind="gate", role="leakage_gate", phase="prebuild", status="ok"),
           CostEvent(kind="embedding", role="embedder", phase="prebuild", model="fixture-hash-embedder",
                     usage=Usage(input_tokens=30))]
    (tmp_path / "prebuild_costs.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in pre), encoding="utf-8")
    paths = write_report(tmp_path, manifest(plan))
    rep = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    co = rep["costs"]
    run, fin, preb = co["by_phase"]["runtime"], co["by_phase"]["finalization"], co["by_phase"]["prebuild"]
    # only the 4 runtime model calls count: 2 reported, 2 unavailable (never 0); simulator/gate/retrieval/search
    # events carry no tokens and are not counted as unavailable
    assert run["input_tokens"] == {"sum": 150, "reported": 2, "unavailable": 2}
    assert run["reasoning_tokens"] == {"sum": None, "reported": 0, "unavailable": 4}
    assert (run["events"], run["retries"], run["latency_s"]) == (8, 1, 1.75)
    assert (fin["events"], fin["budget_actions"], fin["input_tokens"]["sum"]) == (2, 0, 7)
    assert run["budget_actions"] == 4 and [e["actions_used"] for e in rep["episodes"]] == [2, 2]   # save: 0 actions
    assert (preb["events"], preb["input_tokens"], co["sources"]["prebuild_costs.jsonl"]) == \
        (2, {"sum": 30, "reported": 1, "unavailable": 0}, 2)
    assert preb["output_tokens"] == {"sum": None, "reported": 0, "unavailable": 0}   # embeddings have no output
    assert co["by_phase"]["evaluation_support"]["events"] == 0
    groups = {(g["phase"], g["condition"], g["role"]): g for g in co["groups"]}
    assert ("prebuild", None, "leakage_gate") in groups and ("runtime", "product", "advisor_product") in groups
    assert groups[("runtime", "baseline", "researcher_finalizer")]["statuses"] == {"infra_error": 1, "ok": 1}
    for k in ("simulator", "gate", "retrieval", "search"):
        assert groups[("runtime", "baseline", k)]["input_tokens"] == {"sum": None, "reported": 0, "unavailable": 0}
    assert co["model_mismatch_events"] == 1
    assert [m["role"] for m in co["models"] if m["check"] == "mismatch"] == ["advisor_product"]
    ccsv = {(r["phase"], r["role"]): r for r in rows(paths["costs.csv"].read_text(encoding="utf-8"))}
    r = ccsv[("runtime", "researcher_finalizer")]
    assert (r["input_tokens_sum"], r["input_tokens_unavailable"], r["events"], r["retries"]) == ("", "2", "2", "1")


def test_b22_model_check_against_the_configured_model_flags_swaps_and_unverified_calls(tmp_path):
    plan = SetPlan(set_id="smoke", reps=1, conditions=["baseline"], episodes=["fixture_ridge"])
    L = Ledger(tmp_path)
    rt = dict(kind="llm_call", phase="runtime", episode_id=success(L, "baseline", 1, 1, "fixture_ridge").eid)
    for e in [
        # knowledge.gate.guarded_generate records model=<returned or requested> and no detail.model_returned
        CostEvent(role="leakage_gate", model="gate-model-SWAPPED", **rt),
        CostEvent(role="leakage_gate", model="fixture-marker-gate", **rt),
        CostEvent(role="researcher_planner", model=G, detail={"model_returned": f"{G}-0925"}, **rt),   # allowed alias
        CostEvent(role="reranker", model="rr", detail={"model_returned": "rr-2"}, **rt),   # role without a config
    ]:
        L.record_cost(e)
    L.close()
    m = manifest(plan)
    m["models"]["researcher"]["allowed_returned_models"] = [f"{G}-0925"]
    co = build_report(tmp_path, m)["costs"]
    assert {(x["role"], x["model_recorded"]): (x["model_configured"], x["check"]) for x in co["models"]} == {
        ("leakage_gate", "gate-model-SWAPPED"): ("fixture-marker-gate", "mismatch"),
        ("leakage_gate", "fixture-marker-gate"): ("fixture-marker-gate", "unverified"),
        ("researcher_planner", G): (G, "ok"),
        ("reranker", "rr"): (None, "mismatch")}
    assert (co["model_mismatch_events"], co["model_unverified_events"]) == (2, 1)


@pytest.mark.parametrize("mode,frozen,complete,extra,label,why", [
    ("offline_fixture", False, True, {}, "contract_check", None),
    ("live_development", False, True, {}, "development_evidence", None),
    ("evaluation", False, True, {}, "evaluation_not_main", "manifest not frozen"),
    ("evaluation", True, False, {}, "evaluation_not_main", "not every planned set is complete"),
    ("evaluation", True, True, {"development_only_fields": ["roles.researcher", "search.fixture"]},
     "evaluation_not_main", "development-only components"),
    ("evaluation", True, True, {"conditions": ["baseline"]}, "evaluation_not_main", "not both conditions"),
    ("evaluation", True, True, {}, "main_evaluation_result", None),
])
def test_b23_banner_only_a_complete_frozen_fixture_free_paired_evaluation_is_a_main_result(
        tmp_path, mode, frozen, complete, extra, label, why):
    plan = SetPlan(set_id="smoke", reps=1, conditions=extra.get("conditions", ["baseline", "product"]),
                   episodes=["fixture_ridge"])
    L = Ledger(tmp_path)
    success(L, "baseline", 1, 1, "fixture_ridge")
    if complete and "product" in plan.conditions:
        success(L, "product", 1, 1, "fixture_ridge")
    L.record_cost(CostEvent(kind="simulator", role="simulator", phase="runtime"))
    L.close()
    m = manifest(plan, mode, frozen=frozen, development_only_fields=extra.get("development_only_fields", []))
    paths = write_report(tmp_path, m)
    rep = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    md = paths["report.md"].read_text(encoding="utf-8")
    b = rep["banner"]
    assert (b["execution_mode"], b["label"], b["is_main_evaluation_result"]) == (mode, label, label == "main_evaluation_result")
    assert md.startswith("# LabGene run report") and f"**{label}**" in md.split("\n## ")[0] and b["statement"] in md
    if label != "main_evaluation_result":
        assert "main_evaluation_result" not in md
    if why:
        assert [w for w in b["not_main_because"] if why in w] and why in b["statement"]
    if mode == "offline_fixture":
        assert "contract checks only" in b["statement"] and "NOT research performance" in md
    for name in ("episodes.csv", "actions.csv", "costs.csv"):   # a CSV shared alone still says what it is
        rs = rows(paths[name].read_text(encoding="utf-8"))
        assert rs and all((r["execution_mode"], r["report_label"]) == (mode, label) for r in rs), name


def planted_ledger(tmp_path, key):
    """Secrets planted where they must never be exported (raw errors, model text, gate internals, reasons)."""
    marker, path = "LEAK-RIDGE-7Q2", f"{PRIVATE_DIR}\\fixture_ridge.yaml"
    L = Ledger(tmp_path)
    e = Ep(L, "baseline", 1, 1, "fixture_ridge").consult(answer=f"hidden full table fixture_ridge {marker}")
    e.run(MISS["fixture_ridge"], hypothesis="fixture-answers-ridge 10.5555/fixture.ridge.answer")
    L.protocol_event(ProtocolError(scope=e.scope, decision_index=9, raw_text=f"{marker} {path}", reason="invalid_json",
                                   detail="Locating the yield ridge of the fixture reaction"))
    L.infra_event(e.eid, None, "simulator", 1, f"FileNotFoundError: {path} key={key}")
    L.record_cost(CostEvent(kind="gate", role="leakage_gate", phase="runtime", episode_id=e.eid,
                            detail={"internal_detail": f"matched {marker} in bundle fixture-answers-ridge",
                                    "error": f"auth failed for {key} at {PRIVATE_DIR.as_posix()}"}))
    L.set_outcome(e.eid, Outcome.infra_incomplete, reason=f"simulator_infra: {path}")
    L.record_finalization(e.eid, 1, FinalizeResult(event_id="f", status="failed", error=f"{path} {marker}"))
    L.close()
    return [marker, "fixture-answers-ridge", "10.5555/fixture.ridge.answer", "hidden full table fixture_ridge",
            "Locating the yield ridge", str(PRIVATE_DIR), PRIVATE_DIR.as_posix(), "tests/fixtures/private", key]


def test_public_projection_never_exports_planted_secrets(tmp_path, monkeypatch):
    key = "AIzaSyPLANTED-test-key-0123456789"
    monkeypatch.setenv("GEMINI_API_KEY", key)
    secrets = planted_ledger(tmp_path, key)
    m = manifest(SetPlan(set_id="smoke", reps=1, episodes=["fixture_ridge"]),
                 private_assets={"bundle_id": "fixture-answers-ridge", "markers": ["LEAK-RIDGE-7Q2"]})
    paths = write_report(tmp_path, m, tmp_path / "out")
    assert set(paths) == {"report.json", "episodes.csv", "actions.csv", "costs.csv", "report.md"}
    for p in paths.values():
        text = p.read_text(encoding="utf-8").casefold()
        found = [s for s in secrets if s.casefold() in text or json.dumps(s)[1:-1].casefold() in text]
        assert not found, (p.name, found)
    rep = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    assert rep["episodes"][0]["outcome_reason"] == "other"   # unknown reason text is never exported raw


@pytest.mark.parametrize("planted", ["LEAK-RIDGE-7Q2", "private_dir", "api_key"])
def test_public_projection_guard_refuses_private_content_in_exported_fields(tmp_path, monkeypatch, planted):
    key = "AIzaSyPLANTED-test-key-0123456789"
    monkeypatch.setenv("GEMINI_API_KEY", key)
    value = {"LEAK-RIDGE-7Q2": "LEAK-RIDGE-7Q2", "private_dir": str(PRIVATE_DIR), "api_key": key}[planted]
    L = Ledger(tmp_path)
    success(L, "baseline", 1, 1, "fixture_ridge")
    L.close()
    m = manifest(SetPlan(set_id="smoke", reps=1, episodes=["fixture_ridge"]),
                 code_version={"source_tree_sha256": "tree", "note": value})
    with pytest.raises(ValueError) as exc:
        write_report(tmp_path, m, tmp_path / "out")
    assert value.casefold() not in str(exc.value).casefold() and not (tmp_path / "out").exists()


def live_layout(tmp_path, marker):
    """Live-style layout: tmp_path is the repo root (pyproject.toml), private_dir is the bare relative word
    'private' (configs/live.example.yaml), and the ridge bundle carries a numeric secret marker."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "private").mkdir()
    for t in TASKS:
        bundle = yaml.safe_load((PRIVATE_DIR / f"{t}.yaml").read_text(encoding="utf-8"))
        bundle["answer_bundle"]["secret_markers"].append(marker)
        (tmp_path / "private" / f"{t}.yaml").write_text(yaml.safe_dump(bundle), encoding="utf-8")
    m = manifest(SetPlan(set_id="smoke", reps=1, conditions=["baseline"], episodes=["fixture_ridge"]),
                 code_version={"source_tree_sha256": "tree", "note": "private-pilot-01"})
    m["profile"]["paths"]["private_dir"] = "private"
    return m


def test_b12_privacy_guard_keeps_own_observations_and_the_bare_word_private(tmp_path):
    marker = "95.0125"   # a hidden value the researcher may legitimately reach by its own experiments
    m = live_layout(tmp_path, marker)
    L = Ledger(tmp_path / "run")
    Ep(L, "baseline", 1, 1, "fixture_ridge").invalid({"x": marker}, "x must be a finite number") \
        .run(MISS["fixture_ridge"], params={"x": 95.0125}).run({"yield": 95.0125}).end(Outcome.success)
    L.close()
    paths = write_report(tmp_path / "run", m, tmp_path / "out")   # run_id-like text 'private-pilot-01' is fine
    rep = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    assert rep["privacy_scan"] == {"tasks_checked": sorted(TASKS), "tasks_missing": []}
    a = rep["actions"]   # own submissions (number or string) and own observations are exported (spec §7.3)
    assert (a[0]["parameters"], a[1]["parameters"], a[2]["results"]) == ({"x": marker}, {"x": 95.0125}, {"yield": 95.0125})
    assert rep["episodes"][0]["success"] is True
    # the same marker, or a private path, in a harness-authored string is still refused
    for leak in (f"marker {marker}", "private/fixture_ridge.yaml", r"private\fixture_ridge.yaml"):
        with pytest.raises(ValueError):
            build_report(tmp_path / "run", {**m, "code_version": {"note": leak}})


def test_privacy_scan_records_tasks_whose_answer_bundle_is_missing(tmp_path):
    L = Ledger(tmp_path)
    success(L, "baseline", 1, 1, "fixture_ridge")
    L.close()
    m = manifest(SetPlan(set_id="smoke", reps=1, conditions=["baseline"], episodes=["fixture_ridge"]))
    m["profile"]["paths"]["private_dir"] = "moved/elsewhere"   # e.g. reported where private/ is not mounted
    paths = write_report(tmp_path, m)
    rep = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    assert rep["privacy_scan"] == {"tasks_checked": [], "tasks_missing": sorted(TASKS)}
    assert "Privacy scan PARTIAL" in paths["report.md"].read_text(encoding="utf-8")


HARD_KILL = """
import os, sqlite3, sys
db = sqlite3.connect(sys.argv[1], isolation_level=None)
db.execute("PRAGMA cache_size=1")   # spill to the db file so the rollback journal is hot
db.execute("BEGIN IMMEDIATE")
db.execute("UPDATE episodes SET finalization='failed', memory_hash_end='torn'")   # overwrites a committed page
for _ in range(50):
    db.execute("INSERT INTO cost_events(phase, kind, role, status, json, created_at)"
               " VALUES ('runtime', 'other', 'x', 'ok', ?, 't')", ("not json " + "x" * 4000,))
os._exit(1)   # hard kill: no rollback
"""


def test_t09_report_after_a_hard_kill_mid_write_shows_the_last_committed_state(tmp_path):
    L = Ledger(tmp_path)
    success(L, "baseline", 1, 1, "fixture_ridge")
    L.close()
    subprocess.run([sys.executable, "-c", HARD_KILL, str(tmp_path / "ledger.sqlite")], check=False, timeout=60)
    files = [tmp_path / "ledger.sqlite", tmp_path / "ledger.sqlite-journal"]
    assert files[1].exists()
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    rep = build_report(tmp_path, manifest(SetPlan(set_id="smoke", reps=1, episodes=["fixture_ridge"])))
    e = rep["episodes"][0]
    assert (e["outcome"], e["finalization"], e["memory_hash_end"]) == ("success", "done", "end-baseline-r1-e001")
    assert rep["sets"][0]["complete"] is True and rep["sets"][1]["complete"] is False
    assert rep["costs"]["sources"]["ledger.sqlite"] == 0   # the uncommitted rows were rolled back (in the copy)
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in files] == before   # run's files untouched
