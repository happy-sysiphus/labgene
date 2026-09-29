import json
import shutil
import sqlite3
from pathlib import Path

import yaml

from rag_eval.cli import main


def args(run: Path, out: Path, cfg: Path, *extra: str) -> list[str]:
    return ["run", "--run-dir", str(run), "--out", str(out), "--config", str(cfg), "--workers", "2", *extra]


def judge_calls(out: Path) -> int:
    lines = (out / "costs.jsonl").read_text(encoding="utf-8").splitlines()
    return sum(json.loads(line)["role"].startswith("rag_judge_") for line in lines)


def test_end_to_end_offline_resume_and_the_target_run_untouched(offline_run, eval_config, tmp_path, capsys, digest):
    before = digest(offline_run)
    out = tmp_path / "eval"
    assert main(args(offline_run, out, eval_config)) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["status"] == "done" and res["target_complete"] is True
    s = json.loads((out / "report.json").read_text(encoding="utf-8"))["summary"]
    assert all(s["conditions"][c]["judged"] >= 1 for c in ("baseline", "product"))
    assert s["differences"]["faithfulness"]["baseline"] is not None
    items = [json.loads(line) for line in (out / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert items and all("results" not in i and "best_prior" not in i for i in items)
    private = Path(yaml.safe_load(eval_config.read_text(encoding="utf-8"))["private_dir"]) / "eval"
    assert (private / "sim_raw.jsonl").exists()
    n = judge_calls(out)
    assert n > 0
    assert main(args(offline_run, out, eval_config)) == 0          # resume: every item is done, no new judge call
    capsys.readouterr()
    assert judge_calls(out) == n
    assert digest(offline_run) == before


def test_a_judge_outage_stops_resumably_instead_of_recording_unavailable(offline_run, eval_config, tmp_path, capsys,
                                                                         monkeypatch):
    import rag_eval.cli as cli
    from labgene.providers.fixture import FixtureProvider
    from rag_eval.judge import fixture_judge
    out = tmp_path / "eval"
    with monkeypatch.context() as m:        # e.g. a subscription usage-limit window: every judge call fails
        m.setattr(cli, "build_llm", lambda *a, **k: FixtureProvider(policy=fixture_judge,
                                                                     script=["infra_error"] * 100))
        assert main(args(offline_run, out, eval_config)) == 3
    assert cli.read_lines(out / "judged.jsonl") == []                # nothing recorded as "unavailable"
    assert main(args(offline_run, out, eval_config)) == 0            # the same command resumes once it is back
    capsys.readouterr()
    s = json.loads((out / "report.json").read_text(encoding="utf-8"))["summary"]
    assert all(s["conditions"][c]["judge_unavailable"] == 0 and s["conditions"][c]["judged"] >= 1
               for c in ("baseline", "product"))


def test_refusals(offline_run, eval_config, tmp_path, capsys):
    assert main(args(offline_run, offline_run / "eval", eval_config)) == 2     # --out inside the target run
    out = tmp_path / "eval"
    assert main(args(offline_run, out, eval_config)) == 0
    cfg = yaml.safe_load(eval_config.read_text(encoding="utf-8"))
    cfg["sample_per_task"] = 1
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert main(args(offline_run, out, other)) == 2                 # the same --out with another config
    capsys.readouterr()


def test_an_unfinished_target_needs_the_explicit_flag(offline_run, eval_config, tmp_path, capsys):
    copy = tmp_path / "run"
    shutil.copytree(offline_run, copy)
    db = sqlite3.connect(copy / "ledger.sqlite")
    db.execute("UPDATE episodes SET finalization='not_started' WHERE episode_id=(SELECT MAX(episode_id) FROM episodes)")
    db.commit()
    db.close()
    assert main(args(copy, tmp_path / "e1", eval_config)) == 2
    assert main(args(copy, tmp_path / "e2", eval_config, "--allow-incomplete")) == 0
    capsys.readouterr()
    assert json.loads((tmp_path / "e2" / "manifest.json").read_text(encoding="utf-8"))["target_complete"] is False
    assert "NO (partial run)" in (tmp_path / "e2" / "report.md").read_text(encoding="utf-8")
