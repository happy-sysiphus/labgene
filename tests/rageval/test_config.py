import sqlite3
from pathlib import Path

import pytest

from rag_eval.config import load_config

REPO = Path(__file__).resolve().parents[2]


def test_repo_config_matches_the_spec():
    cfg = load_config(REPO / "configs/rag_eval.yaml")
    assert (cfg.judge.provider, cfg.judge.model, cfg.judge.endpoint, cfg.judge.reasoning_effort) == \
        ("claude_code", "claude-opus-5-5", "claude_cli", "medium")
    assert (cfg.sample_per_task, cfg.bootstrap_draws) == (10, 10000)
    assert (cfg.judge_caps.max_calls, cfg.support_caps.max_calls, cfg.support_caps.max_usd) == (400, 170, 0.5)


def test_unknown_keys_are_refused(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("judge: {provider: fixture, model: m}\njudge_caps: {}\nsupport_caps: {}\ntypo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_offline_run_has_consults_in_both_conditions(offline_run):
    con = sqlite3.connect(offline_run / "ledger.sqlite")
    try:
        n = dict(con.execute("SELECT e.condition, COUNT(*) FROM consult_exchanges c JOIN episodes e "
                             "ON e.episode_id = c.episode_id GROUP BY e.condition").fetchall())
    finally:
        con.close()
    assert n["baseline"] >= 2 and n["product"] >= 2
