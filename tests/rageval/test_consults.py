import shutil
import sqlite3

import pytest

from labgene.app import Run
from rag_eval.consults import consult_records, consult_request, draw_sample, open_ledger, pick, set_finished


def test_pick_spreads_evenly_and_rounds_halves_up():
    assert pick(0, 3) == [] and pick(5, 1) == [0]
    assert pick(10, 10) == list(range(10))
    assert pick(4, 3) == [0, 2, 3]                       # 1.5 -> 2
    assert pick(25, 10) == [0, 3, 5, 8, 11, 13, 16, 19, 21, 24]


def test_sample_is_deterministic_capped_and_complete_answers_only(offline_run):
    con = open_ledger(offline_run)
    try:
        recs = consult_records(con)
    finally:
        con.close()
    sample = draw_sample(recs, 2)
    assert sample == draw_sample(recs, 2)
    by_id = {r.action_id: r for r in recs}
    for cond in ("baseline", "product"):                 # one task: at most 2 per condition
        assert 1 <= sum(by_id[a].condition == cond for a in sample) <= 2
    assert all(by_id[a].ok for a in sample)


def test_request_is_the_episode_state_before_the_consult(offline_run):
    run = Run.load(offline_run)
    con = open_ledger(offline_run)
    try:
        for rec in consult_records(con):
            req = consult_request(con, rec, run.tasks[rec.task_id])
            assert req.question == rec.exchange.question and req.scope.episode_id == rec.episode_id
            earlier = [*req.observations, *req.experiment_errors, *req.prior_consults]
            assert all(int(x.action_id.rsplit(":a", 1)[1]) < rec.seq for x in earlier)
            assert len(earlier) == rec.seq - 1                    # every earlier action is one of the three kinds
            assert req.remaining_actions == req.scope.action_budget - len(earlier) - 1
    finally:
        con.close()


def test_finished_run_and_an_unsaved_episode(offline_run, tmp_path):
    plan = Run.load(offline_run).plan
    con = open_ledger(offline_run)
    try:
        assert set_finished(con, plan)
    finally:
        con.close()
    copy = tmp_path / "run"
    shutil.copytree(offline_run, copy)
    db = sqlite3.connect(copy / "ledger.sqlite")
    db.execute("UPDATE episodes SET finalization='not_started' WHERE episode_id=(SELECT MAX(episode_id) FROM episodes)")
    db.commit()
    db.close()
    con = open_ledger(copy)
    try:
        assert not set_finished(con, plan)
    finally:
        con.close()


def test_ledger_is_opened_read_only(offline_run):
    con = open_ledger(offline_run)          # the tmp path holds non-ASCII characters on this machine: URI encoding
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute("CREATE TABLE x(y)")
    finally:
        con.close()
