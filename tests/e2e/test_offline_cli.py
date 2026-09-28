"""T07 end-to-end through the real CLI in subprocesses (offline fixture profile: contract checks only).
B04 crash/resume, B07/B09 memory flow, B10 set restore + isolation, B11 no secrets exported, B21/B22 report,
B23 refusals."""
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SECRETS = ["LEAK-RIDGE-7Q2", "LEAK-CAT-9X4", "hidden full table fixture_ridge", "fixture.ridge.answer",
           "Locating the yield ridge of the fixture reaction", "fixture.example/papers/ridge-answer",
           "Optimal catalyst loading for the fixture coupling", "fixture-answers-ridge"]


def cli(*args, env_extra=None, check=None):
    env = {k: v for k, v in os.environ.items() if k != "LABGENE_FAULT"}
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, "-m", "labgene", *args], cwd=REPO, env=env, capture_output=True,
                       text=True, encoding="utf-8", timeout=600)
    if check is not None:
        assert r.returncode == check, (r.returncode, r.stdout[-2000:], r.stderr[-3000:])
    return r


def profile(tmp_path, name="offline", **fixture):
    p = yaml.safe_load((REPO / "configs/offline.yaml").read_text(encoding="utf-8"))
    p.setdefault("paths", {})["artifacts_dir"] = str(tmp_path / "artifacts")
    if fixture:
        p["fixture"] = {**p["fixture"], **fixture}
    f = tmp_path / f"{name}.yaml"
    f.write_text(yaml.safe_dump(p), encoding="utf-8")
    return f


def plan(tmp_path, name="plan", **kw):
    p = {"set_id": "e2e", "reps": 1, "conditions": ["baseline", "product"],
         "episodes": ["fixture_ridge", "fixture_catalyst", "fixture_ridge"], **kw}
    f = tmp_path / f"{name}.yaml"
    f.write_text(yaml.safe_dump(p), encoding="utf-8")
    return f


def ledger_summary(run_dir: Path):
    """Evaluation-relevant ledger content: outcomes, counters, exact observations, memory hashes."""
    con = sqlite3.connect(run_dir / "ledger.sqlite")
    eps = con.execute("SELECT episode_id, outcome, actions_to_success, finalization, memory_hash_start, memory_hash_end"
                      " FROM episodes ORDER BY episode_id").fetchall()
    acts = con.execute("SELECT episode_id, seq, kind, counted_as, payload_hash, result_json FROM actions"
                       " WHERE status='committed' ORDER BY episode_id, seq").fetchall()
    strip = lambda j: {k: v for k, v in json.loads(j).items() if k not in ("created_at", "scope")}   # noqa: E731
    con.close()
    return eps, [(e, s, k, c, h, strip(r)) for e, s, k, c, h, r in acts]


def memory_rows(run_dir: Path):
    """Row counts of every memory table per condition state: a duplicate save would change them."""
    out = {}
    for db in sorted((run_dir / "state").rglob("memory.sqlite")):
        con = sqlite3.connect(db)
        tables = [t for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'") if t != "meta"]
        out[db.relative_to(run_dir).as_posix()] = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                                                  for t in tables}
        con.close()
    return out


def test_t07_offline_two_condition_set_report_b21_b22(tmp_path):
    prof, pl = profile(tmp_path), plan(tmp_path, reps=2)
    out = json.loads(cli("run-set", "--profile", str(prof), "--plan", str(pl), "--run-id", "r1", check=0).stdout)
    assert out["status"] == "complete" and out["execution_mode"] == "offline_fixture"
    run = tmp_path / "artifacts" / "r1"
    eps, acts = ledger_summary(run)
    assert len(eps) == 12 and all(o == "success" and f == "done" for _, o, _, f, _, _ in eps)
    rep = json.loads((run / "report" / "report.json").read_text(encoding="utf-8"))
    blob = json.dumps(rep)
    assert "offline_fixture" in blob and "contract" in blob.lower()
    assert "offline_fixture" in (run / "report" / "episodes.csv").read_text(encoding="utf-8")
    # B22: finalization is costed and adds no actions
    con = sqlite3.connect(run / "ledger.sqlite")
    fin = con.execute("SELECT COUNT(*) FROM cost_events WHERE phase='finalization'").fetchone()[0]
    per_ep = con.execute("SELECT episode_id, COUNT(*) FROM actions WHERE status='committed' GROUP BY episode_id").fetchall()
    con.close()
    assert fin > 0 and all(n <= 50 for _, n in per_ep)
    # resume of a complete run executes nothing new
    cli("resume", "--run-id", "r1", "--artifacts-dir", str(tmp_path / "artifacts"), check=0)
    assert ledger_summary(run)[1] == acts


def test_b01_b21_budget_exhausted_at_exactly_50_without_51st_action(tmp_path):
    prof = profile(tmp_path, "always_consult", consult_every=1)          # researcher never experiments
    pl = plan(tmp_path, episodes=["fixture_ridge"])
    cli("run-set", "--profile", str(prof), "--plan", str(pl), "--run-id", "ex", check=0)
    run = tmp_path / "artifacts" / "ex"
    con = sqlite3.connect(run / "ledger.sqlite")
    rows = con.execute("SELECT e.episode_id, e.outcome, e.actions_to_success, COUNT(a.action_id) FROM episodes e"
                       " JOIN actions a ON a.episode_id=e.episode_id GROUP BY e.episode_id").fetchall()
    con.close()
    assert rows and all(o == "budget_exhausted" and ats is None and n == 50 for _, o, ats, n in rows)
    csv = (run / "report" / "episodes.csv").read_text(encoding="utf-8")
    assert ",51," not in csv and "budget_exhausted" in csv


@pytest.mark.parametrize("fault", ["after_reserve@9", "after_execute_before_commit@7", "after_commit@20",
                                   "before_finalize@2", "during_finalize@3", "after_finalize@4"])
def test_b04_forced_interruption_then_resume_equals_clean_run(tmp_path, fault):
    prof, pl = profile(tmp_path), plan(tmp_path)
    cli("run-set", "--profile", str(prof), "--plan", str(pl), "--run-id", "clean", check=0)
    r = cli("run-set", "--profile", str(prof), "--plan", str(pl), "--run-id", "crash",
            env_extra={"LABGENE_FAULT": fault}, check=70)
    assert "injected crash" in r.stderr
    cli("resume", "--run-id", "crash", "--artifacts-dir", str(tmp_path / "artifacts"), check=0)
    clean, crash = (ledger_summary(tmp_path / "artifacts" / n) for n in ("clean", "crash"))
    no_hash = lambda eps: [e[:4] for e in eps]   # noqa: E731  (hashes cover run ids and timestamps)
    assert no_hash(crash[0]) == no_hash(clean[0])   # outcomes, actions_to_success, finalization identical
    assert crash[1] == clean[1]            # same committed actions, payloads and exact results; no double charge
    assert memory_rows(tmp_path / "artifacts" / "crash") == memory_rows(tmp_path / "artifacts" / "clean")
    con = sqlite3.connect(tmp_path / "artifacts" / "crash" / "ledger.sqlite")
    n_crash = con.execute("SELECT COUNT(*) FROM cost_events").fetchone()[0]
    con.close()
    con = sqlite3.connect(tmp_path / "artifacts" / "clean" / "ledger.sqlite")
    n_clean = con.execute("SELECT COUNT(*) FROM cost_events").fetchone()[0]
    con.close()
    assert n_crash >= n_clean              # physical re-execution cost is kept, never erased


def test_b07_b09_b10_memory_flow_restore_and_isolation(tmp_path):
    prof, pl = profile(tmp_path), plan(tmp_path, reps=2)
    cli("run-set", "--profile", str(prof), "--plan", str(pl), "--run-id", "m", check=0)
    run = tmp_path / "artifacts" / "m"
    eps = {e: (o, s, end) for e, o, _, _, s, end in ledger_summary(run)[0]}
    for c in ("baseline", "product"):
        # B10: every set rep starts from the same initial state; B07: memory grows within a set
        assert eps[f"{c}-r1-e001"][1] == eps[f"{c}-r2-e001"][1]
        assert eps[f"{c}-r1-e003"][1] != eps[f"{c}-r1-e001"][1]
    # B10: condition stores never contain the other condition's (or the other rep's) records
    for c, other in (("baseline", "product"), ("product", "baseline")):
        for rep, other_rep in ((1, 2), (2, 1)):
            db = run / "state" / c / "e2e" / f"rep{rep}" / "memory.sqlite"
            dump = "\n".join(sqlite3.connect(db).iterdump())
            assert f"{other}-r" not in dump and f"{c}-r{other_rep}-" not in dump and f"{c}-r{rep}-e001" in dump
    # B09: the revisit starts with an empty researcher view and reaches past results only via a consultation
    con = sqlite3.connect(run / "ledger.sqlite")
    first = con.execute("SELECT kind FROM actions WHERE episode_id='baseline-r1-e003' ORDER BY seq").fetchall()
    con.close()
    assert first[0][0] == "consult"


def test_b11_no_hidden_answer_in_any_exported_or_ledger_text(tmp_path):
    prof, pl = profile(tmp_path), plan(tmp_path)
    cli("run-set", "--profile", str(prof), "--plan", str(pl), "--run-id", "s", check=0)
    run = tmp_path / "artifacts" / "s"
    texts = [p.read_text(encoding="utf-8") for p in (run / "report").iterdir()]
    texts += [(run / "manifest.json").read_text(encoding="utf-8")]
    con = sqlite3.connect(run / "ledger.sqlite")
    texts += ["\n".join(con.iterdump())]
    con.close()
    blob = "\n".join(texts)
    assert not [s for s in SECRETS if s in blob]


def test_b23_refusals_for_evaluation_and_unapproved_live(tmp_path):
    pl = plan(tmp_path)
    r = cli("run-evaluation", "--profile", "configs/offline.yaml", "--plan", str(pl), "--run-id", "x", check=2)
    assert "execution_mode=evaluation" in r.stdout
    r = cli("preflight", "--profile", "configs/live.example.yaml", check=2)
    assert "cost_caps.max_usd" in r.stdout
    ev = yaml.safe_load((REPO / "configs/live.example.yaml").read_text(encoding="utf-8"))
    ev.update(execution_mode="evaluation", frozen_manifest=str(tmp_path / "missing.json"))
    ev["paths"] = {"artifacts_dir": str(tmp_path / "artifacts"), "tasks_dir": "configs/tasks",
                   "private_dir": "tests/fixtures/private"}
    ev["simulators"] = {"fixture.ridge": {"backend": "fixture", "fixture_function": "ridge"},
                        "fixture.catalyst": {"backend": "fixture", "fixture_function": "catalyst"}}
    f = tmp_path / "eval.yaml"
    f.write_text(yaml.safe_dump(ev), encoding="utf-8")
    fake_env = {"GEMINI_API_KEY": "x", "OPENAI_API_KEY": "x", "ANTHROPIC_API_KEY": "x"}
    r = cli("run-set", "--profile", str(f), "--plan", str(pl), "--run-id", "y", env_extra=fake_env, check=2)
    assert "run-evaluation" in r.stdout
    r = cli("run-evaluation", "--profile", str(f), "--plan", str(pl), "--run-id", "y", env_extra=fake_env, check=2)
    probs = "\n".join(json.loads(r.stdout)["problems"])
    assert "frozen_manifest" in probs and "not validated" in probs and "development_only" in probs
    assert not (tmp_path / "artifacts" / "y").exists()      # nothing ran, nothing was created
    r = cli("freeze", "--profile", "configs/offline.yaml", "--plan", str(pl), "--analysis-plan", str(pl),
            "--out", str(tmp_path / "frozen.json"), check=2)
    assert not (tmp_path / "frozen.json").exists()
