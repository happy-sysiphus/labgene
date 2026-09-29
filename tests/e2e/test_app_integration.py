"""T07/T10 integration through the real app wiring (offline fixture: contract checks only).
B17 condition parity, B11 model inputs, B22/B04 caps across sessions, §11.2 interruption history,
snapshot recovery, B23 freeze pins knowledge inputs, CLI exit codes."""
import json
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from labgene import cli
from labgene.app import Run, evaluation_problems, freeze_digest, load_tasks
from labgene.config import load_profile, load_set_plan
from labgene.providers import fixture as fx

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.delenv("LABGENE_FAULT", raising=False)


def files(tmp_path, reps=1, episodes=("fixture_ridge", "fixture_catalyst", "fixture_ridge"), **overrides):
    p = yaml.safe_load((REPO / "configs/offline.yaml").read_text(encoding="utf-8"))
    p["paths"] = {**p.get("paths", {}), "artifacts_dir": str(tmp_path / "artifacts")}
    for k, v in overrides.items():
        p[k] = {**p.get(k, {}), **v} if isinstance(v, dict) else v
    prof, plan = tmp_path / "profile.yaml", tmp_path / "plan.yaml"
    prof.write_text(yaml.safe_dump(p), encoding="utf-8")
    plan.write_text(yaml.safe_dump({"set_id": "it", "reps": reps, "conditions": ["baseline", "product"],
                                    "episodes": list(episodes)}), encoding="utf-8")
    return str(prof), str(plan)


def run_cli(*args):
    return cli.main(list(args))


def test_b17_b11_condition_parity_and_model_inputs_through_real_wiring(tmp_path, monkeypatch):
    cap = []
    orig = fx.FixtureProvider.generate
    monkeypatch.setattr(fx.FixtureProvider, "generate", lambda self, req: (cap.append(req), orig(self, req))[1])
    prof, plan = files(tmp_path)
    run = Run.create(load_profile(prof), load_set_plan(plan), "par")
    assert run.execute().status == "complete"
    con = sqlite3.connect(run.run_dir / "ledger.sqlite")
    n_base = con.execute("SELECT COUNT(*) FROM cost_events WHERE role LIKE 'researcher%' AND kind='llm_call'"
                         " AND episode_id LIKE 'baseline-%'").fetchone()[0]
    con.close()
    res = [r for r in cap if r.role.startswith("researcher")]
    split = {"baseline": res[:n_base], "product": res[n_base:]}      # baseline scope runs first
    seen = defaultdict(lambda: defaultdict(set))
    for c, rs in split.items():
        for r in rs:
            seen[c]["system"].add((r.role, r.system_instruction))
            seen[c]["tools"].add((r.role, json.dumps([t.model_dump() for t in r.tools], sort_keys=True)))
            seen[c]["cfg"].add((r.role, r.model, r.thinking_level, r.reasoning_effort, r.max_output_tokens))
    for k in ("system", "tools", "cfg"):
        assert seen["baseline"][k] == seen["product"][k], k           # same researcher in both conditions
    assert not [r for r in res if any(w in json.dumps(r.input) for w in ("baseline", "product-r", "evidence_cards"))]
    adv = {c: {(r.model, r.thinking_level, r.max_output_tokens) for r in cap if r.role == f"advisor_{c}"}
           for c in ("baseline", "product")}
    assert adv["baseline"] == adv["product"] and len(adv["baseline"]) == 1   # same advisor base model config
    tools = {c: {t.name for r in cap if r.role == f"advisor_{c}" for t in r.tools} for c in ("baseline", "product")}
    assert tools["baseline"] == tools["product"] == {"search", "open"}    # no optimizer, no execution tools
    base_inputs = json.dumps([r.input for r in cap if r.role == "advisor_baseline"])
    assert "evidence_cards" not in base_inputs and "kg_path" not in base_inputs   # no product RAG/KG for baseline
    markers = [m for f in (REPO / "tests/fixtures/private").glob("*.yaml")
               for m in yaml.safe_load(f.read_text(encoding="utf-8"))["answer_bundle"]["secret_markers"]]
    model_facing = json.dumps([[r.system_instruction, r.input] for r in cap if r.role != "leakage_gate"])
    assert not [m for m in markers if m in model_facing]


def test_b22_b04_cost_caps_hold_across_sessions_including_prebuild(tmp_path):
    prof, plan = files(tmp_path, cost_caps={"max_calls": 3})
    assert run_cli("build-corpus", "--profile", prof, "--plan", plan, "--run-id", "cap") == 3   # stopped in prebuild
    run_dir = tmp_path / "artifacts" / "cap"
    spent = json.loads((run_dir / "guard.json").read_text(encoding="utf-8"))["calls"]
    billable = lambda: [ln for ln in (run_dir / "prebuild_costs.jsonl").read_text(encoding="utf-8").splitlines()  # noqa: E731
                        if json.loads(ln)["kind"] in ("llm_call", "embedding")]
    n_calls = len(billable())
    assert spent == 3 and n_calls <= 3
    assert run_cli("resume", "--run-id", "cap", "--artifacts-dir", str(tmp_path / "artifacts")) == 3
    assert len(billable()) == n_calls      # the resumed session made no further billable call
    lines = [json.loads(x) for x in (run_dir / "sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["command"] for x in lines] == ["build-corpus", "execute"]


def test_interruption_history_survives_resume_and_cap_stop_is_resumable(tmp_path):
    prof, plan = files(tmp_path, cost_caps={"max_calls": 40})     # enough for prebuild, not for the set
    art = str(tmp_path / "artifacts")
    assert run_cli("run-set", "--profile", prof, "--plan", plan, "--run-id", "hist") == 3
    run_dir = Path(art) / "hist"
    con = sqlite3.connect(run_dir / "ledger.sqlite")
    stopped = con.execute("SELECT episode_id, outcome, outcome_reason FROM episodes WHERE outcome='infra_incomplete'"
                          ).fetchall()
    con.close()
    assert stopped and stopped[0][2] == "cost_cap"
    p = yaml.safe_load((run_dir / "profile.yaml").read_text(encoding="utf-8"))   # operator raises the approved cap
    p["cost_caps"]["max_calls"] = 100000
    (run_dir / "profile.yaml").write_text(yaml.safe_dump(p), encoding="utf-8")
    assert run_cli("resume", "--run-id", "hist", "--artifacts-dir", art) == 0
    con = sqlite3.connect(run_dir / "ledger.sqlite")
    ev = con.execute("SELECT kind, detail FROM episode_events WHERE episode_id=? ORDER BY id", (stopped[0][0],)).fetchall()
    out = con.execute("SELECT outcome FROM episodes WHERE episode_id=?", (stopped[0][0],)).fetchone()[0]
    con.close()
    assert ev == [("interrupted", "cost_cap"), ("resumed", "cost_cap")] and out == "success"
    rows = json.loads((run_dir / "report" / "report.json").read_text(encoding="utf-8"))["episodes"]
    [row] = [r for r in rows if r["episode_id"] == stopped[0][0]]
    assert (row["interruptions"], row["resumes"], row["interruption_reasons"]) == (1, 1, "cost_cap")


def test_interrupted_initial_snapshot_is_kept_aside_and_redone(tmp_path):
    prof, plan = files(tmp_path)
    art = str(tmp_path / "artifacts")
    assert run_cli("build-corpus", "--profile", prof, "--plan", plan, "--run-id", "snap") == 0
    snap = Path(art) / "snap" / "initial" / "product"
    (snap / "manifest.json").unlink()                              # as if killed before the manifest was written
    assert run_cli("resume", "--run-id", "snap", "--artifacts-dir", art) == 0
    assert (snap / "manifest.json").exists() and (snap.parent / "product.partial-1").is_dir()


def test_b23_freeze_digest_pins_knowledge_inputs_and_analysis_plan_content(tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(REPO / "tests/fixtures/corpus", corpus)
    prof, plan = files(tmp_path, knowledge={"corpus_dir": str(corpus)})
    p, pl = load_profile(prof), load_set_plan(plan)
    tasks = load_tasks(p, pl)
    before = freeze_digest(p, pl, tasks)
    doc = next(corpus.glob("*.md"))
    doc.write_text(doc.read_text(encoding="utf-8") + "\nOptimum: temperature 83 C (LEAK-RIDGE-7Q2)\n", encoding="utf-8")
    after = freeze_digest(p, pl, tasks)
    assert before["knowledge_inputs"]["corpus"] != after["knowledge_inputs"]["corpus"]
    assert {k: v for k, v in before.items() if k != "knowledge_inputs"} == \
           {k: v for k, v in after.items() if k != "knowledge_inputs"}
    analysis = tmp_path / "analysis.md"
    analysis.write_text("paired set-level difference\n", encoding="utf-8")
    frozen = {**before, "analysis_plan_sha256": "0" * 64, "analysis_plan_path": str(analysis)}
    fm = tmp_path / "frozen.json"
    fm.write_text(json.dumps(frozen), encoding="utf-8")
    probs = "\n".join(evaluation_problems(p.model_copy(update={"frozen_manifest": str(fm)}), pl, tasks))
    assert "knowledge_inputs" in probs and "analysis plan content changed" in probs


def test_cli_exit_codes_for_refused_inputs(tmp_path):
    prof, plan = files(tmp_path)
    assert run_cli("run-set", "--profile", prof, "--plan", plan, "--run-id", "dup") == 0
    assert run_cli("run-set", "--profile", prof, "--plan", plan, "--run-id", "dup") == 2   # existing run: use resume
    assert run_cli("resume", "--run-id", "nope", "--artifacts-dir", str(tmp_path / "artifacts")) == 2


def test_u39_regate_then_import_runs_on_the_same_pinned_initial_state(tmp_path):
    prof, plan = files(tmp_path)                                    # main-like plan: ridge + catalyst bundles, set "it"
    pilot = tmp_path / "pilot.yaml"
    pilot.write_text(yaml.safe_dump({"set_id": "pilot", "conditions": ["baseline", "product"],
                                     "episodes": ["fixture_ridge"]}), encoding="utf-8")
    art = tmp_path / "artifacts"
    assert run_cli("build-corpus", "--profile", prof, "--plan", str(pilot), "--run-id", "src") == 0
    assert run_cli("regate-state", "--profile", prof, "--plan", plan, "--run-id", "state", "--from-run", "src") == 0
    rep = json.loads((art / "state" / "build" / "product.regate.json").read_text(encoding="utf-8"))
    assert rep["set_scope"] == "it" and len(rep["answer_bundle_versions"]) == 2 and rep["source_run"] == "src"
    pinned = {c: json.loads((art / "state" / "initial" / c / "manifest.json").read_text(encoding="utf-8"))
              ["combined_sha256"] for c in ("baseline", "product")}
    doc = yaml.safe_load(Path(prof).read_text(encoding="utf-8"))
    doc["knowledge"]["initial_state_from"] = str(art / "state")
    imp = tmp_path / "import.yaml"
    imp.write_text(yaml.safe_dump(doc), encoding="utf-8")
    p, pl = load_profile(imp), load_set_plan(plan)
    assert freeze_digest(p, pl, load_tasks(p, pl))["initial_states"] == pinned
    run = Run.create(p, pl, "imp")
    assert run.build_initial_states() == pinned                     # copied, not rebuilt
    assert not (run.run_dir / "prebuild_costs.jsonl").exists()      # no gate, KG or embedding call
    assert run.execute().status == "complete"                      # the stores open under this run's identity
    # a plan with other answer bundles refuses the imported state
    other = Run.create(p, load_set_plan(pilot), "other")
    with pytest.raises(ValueError, match="re-gate it for this plan"):
        other.build_initial_states()
    # tampering with the source snapshot is seen by preflight (and by freeze's pins)
    db = art / "state" / "initial" / "product" / "state" / "knowledge.sqlite"
    db.write_bytes(db.read_bytes() + b"\0")
    from labgene.app import run_preflight
    assert any("initial_state_from" in x for x in run_preflight(p, pl))


def test_u40_extend_plan_continues_a_finished_pilot_set_after_an_in_place_regate(tmp_path):
    prof, _ = files(tmp_path, fixture={"consult_every": 3})           # consult answers land in both memories
    art = tmp_path / "artifacts"
    common = {"conditions": ["baseline", "product"], "mode": "progression", "parallel_conditions": True}
    pilot, main = tmp_path / "pilot.yaml", tmp_path / "main.yaml"
    pilot.write_text(yaml.safe_dump({"set_id": "pilot", "episodes": ["fixture_ridge"], "action_budget": 50,
                                     **common}), encoding="utf-8")
    main.write_text(yaml.safe_dump({"set_id": "main", "episodes": ["fixture_ridge", "fixture_catalyst"],
                                    "action_budget": 150, **common}), encoding="utf-8")
    assert run_cli("run-set", "--profile", prof, "--plan", str(pilot), "--run-id", "p") == 0
    con = sqlite3.connect(art / "p" / "ledger.sqlite")
    before = con.execute("SELECT episode_id, scope_json, outcome FROM episodes ORDER BY episode_id").fetchall()
    con.close()
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"set_id": "main", "episodes": ["fixture_catalyst"], "action_budget": 150,
                                   **common}), encoding="utf-8")
    assert run_cli("extend-plan", "--run-id", "p", "--artifacts-dir", str(art), "--plan", str(bad)) == 2
    assert run_cli("regate-state", "--profile", prof, "--plan", str(main), "--run-id", "seed", "--from-run", "p") == 0
    # the fixture gate holds consult answers (no source in their context); pretend they had been allowed, so the
    # re-gate has stored answers to re-decide (it withholds them again)
    rep1 = art / "p" / "state"
    for cond, sql in (("product", "UPDATE consult_cases SET withheld=0"),
                      ("baseline", "UPDATE records SET withheld=0 WHERE kind='consult'")):
        with sqlite3.connect(rep1 / cond / "pilot" / "rep1" / "memory.sqlite") as m:
            assert m.execute(sql).rowcount > 0
    assert run_cli("extend-plan", "--run-id", "p", "--artifacts-dir", str(art), "--plan", str(main),
                   "--seed-run", "seed") == 0
    run = Run.load(art / "p")
    assert run.plan.set_id == "pilot" and run.plan.episodes == ["fixture_ridge", "fixture_catalyst"]
    ext = json.loads(next((art / "p").glob("extend-*.json")).read_text(encoding="utf-8"))
    prod = ext["states"]["p/product/pilot/rep1"]
    assert prod["knowledge"]["seeded_verdicts_last_pass"] > 0
    for key in ("p/product/pilot/rep1", "p/baseline/pilot/rep1"):
        c = ext["states"][key]["consults"]
        assert c["rechecked"] > 0 and c["withheld"] and len(c["withheld"]) == c["rechecked"]
    assert (art / "p" / "extend-backup").is_dir()
    assert run_cli("resume", "--run-id", "p", "--artifacts-dir", str(art)) == 0
    con = sqlite3.connect(art / "p" / "ledger.sqlite")
    after = con.execute("SELECT episode_id, scope_json, outcome FROM episodes ORDER BY episode_id").fetchall()
    tasks = {r[0] for r in con.execute("SELECT DISTINCT task_id FROM episodes")}
    con.close()
    assert [r for r in after if r[0] in {b[0] for b in before}] == before     # the pilot episodes are untouched
    assert tasks == {"fixture_ridge", "fixture_catalyst"}
    # a second extension with the same plan finds everything re-gated already (no new gate verdicts needed)
    assert run_cli("extend-plan", "--run-id", "p", "--artifacts-dir", str(art), "--plan", str(main)) == 0


def test_review_an_interrupted_import_is_redone_and_a_regate_keeps_its_source(tmp_path, monkeypatch):
    import labgene.knowledge.regate as rg
    prof, plan = files(tmp_path)
    pilot = tmp_path / "pilot.yaml"
    pilot.write_text(yaml.safe_dump({"set_id": "pilot", "conditions": ["baseline", "product"],
                                     "episodes": ["fixture_ridge"]}), encoding="utf-8")
    art = tmp_path / "artifacts"
    for src in ("a", "b"):
        assert run_cli("build-corpus", "--profile", prof, "--plan", str(pilot), "--run-id", src) == 0
    p, pl = load_profile(prof), load_set_plan(plan)
    run = Run.create(p, pl, "state")
    orig = rg.regate_state
    monkeypatch.setattr(rg, "regate_state", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt("gate down")))
    with pytest.raises(KeyboardInterrupt):
        run.regate_initial_states(art / "a")
    monkeypatch.setattr(rg, "regate_state", orig)
    with pytest.raises(ValueError, match="another source"):
        run.regate_initial_states(art / "b")
    assert run.regate_initial_states(art / "a")                 # finishing with its own source works
    # an import killed after copying state/ but before the manifest is redone on the next call
    doc = yaml.safe_load(Path(prof).read_text(encoding="utf-8"))
    doc["knowledge"]["initial_state_from"] = str(art / "state")
    imp = tmp_path / "import.yaml"
    imp.write_text(yaml.safe_dump(doc), encoding="utf-8")
    r2 = Run.create(load_profile(imp), pl, "imp")
    partial = r2.run_dir / "initial" / "product"
    shutil.copytree(art / "state" / "initial" / "product" / "state", partial / "state")
    assert r2.build_initial_states()["product"] == run.manifest.initial_state_hashes["product"]
    assert (partial.parent / "product.partial-1").is_dir()
