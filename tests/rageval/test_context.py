import re
import sqlite3
from pathlib import Path

from labgene.app import Run
from labgene.costs import null_context
from rag_eval.consults import consult_records, consult_request, episode_orders, open_ledger
from rag_eval.context import Blinder, Rebuilder, memory_copy, passages, retrieved_cards, split_text

OBS = re.compile(r"obs:[\w-]+(?::[\w-]+)*")


def episode_of(observation_id: str) -> str:
    return observation_id.removeprefix("obs:").rsplit(":", 1)[0]


def rows(db: Path, table: str) -> int:
    con = sqlite3.connect(db)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def test_split_text_keeps_sections_and_the_size_limit():
    text = "# A\n\npara one\n\npara two\n\n## B\n\n" + "x" * 50 + "\n" + "y" * 50
    parts = split_text(text, 60)
    assert parts == ["# A\n\npara one\n\npara two", "## B\n\n" + "x" * 50, "y" * 50]


def test_blinder_neutralizes_ids_and_condition_prefixes():
    b = Blinder({"obs:product-r1-e001:a002", "card:doc:x#3"})
    assert b("see obs:product-r1-e001:a002 and card:doc:x#3; product-r1-e002") == "see S2 and S1; e002"
    assert Blinder(set())("baseline-r1-e003:a001") == "e003:a001"


def test_passages_follow_the_input_and_carry_citation_ids():
    product = {"task": {"task_id": "t"}, "remaining_actions": 3, "current_episode": {
        "observations": [{"observation_id": "obs:e1:a002", "results": {"yield": 1.5}}], "invalid_requests": [],
        "prior_consults": [{"action_id": "e1:a001", "question": "q?", "response": {"answer": "a", "reasoning": "r"}}]},
        "evidence_cards": [
            {"card_id": "card:doc:x#1", "kind": "literature", "source_ids": ["doc:x#1"], "content_hash": "h",
             "excerpt": "text"},
            {"card_id": "card:path:p1", "kind": "agent_inference", "source_ids": ["rel:1"], "content_hash": "h",
             "summary": "A -> B"},
            {"card_id": "card:obs:obs:e1:a002", "kind": "current_observation", "observation_ids": ["obs:e1:a002"],
             "source_ids": [], "content_hash": "h"}]}
    ps = passages(product, 500)
    assert [set(p.cites) for p in ps] == [set(), {"obs:e1:a002"}, set(), {"card:doc:x#1", "doc:x#1"},
                                          {"card:path:p1", "rel:1"}, {"card:obs:obs:e1:a002", "obs:e1:a002"}]
    assert all("content_hash" not in p.text for p in ps)
    assert [c["card_id"] for c in retrieved_cards(product)] == ["card:doc:x#1", "card:path:p1"]
    baseline = {k: v for k, v in product.items() if k != "evidence_cards"}
    baseline["initial_text"] = {"source_id": "initial:text", "text": "# A\n\nfact one\n\n# B\n\nfact two"}
    baseline["own_memory"] = ("## episode 1 | task t | visit 1\n"
                              "- experiment e1:a002 -> observation obs:e1:a002: parameters {} results {}")
    bs = passages(baseline, 500)
    assert [set(p.cites) for p in bs[3:]] == [{"initial:text"}, {"initial:text"}, {"obs:e1:a002"}]
    assert retrieved_cards(baseline) == []


def test_memory_copy_keeps_only_earlier_episodes(offline_run, tmp_path):
    con = open_ledger(offline_run)
    try:
        for cond, table in (("product", "observations"), ("baseline", "records")):
            orders = episode_orders(con, cond)
            first, second = sorted(orders, key=orders.get)
            state = offline_run / "state" / cond / "rag" / "rep1"
            assert rows(memory_copy(state, orders, first, tmp_path / cond / "1") / "memory.sqlite", table) == 0
            assert rows(memory_copy(state, orders, second, tmp_path / cond / "2") / "memory.sqlite", table) > 0
    finally:
        con.close()


def test_rebuild_reproduces_the_advisor_input_from_copies_only(offline_run, tmp_path, digest):
    before = digest(offline_run)
    run = Run.load(offline_run)
    con = open_ledger(offline_run)
    try:
        recs = consult_records(con)
        saw_past = {}
        for cond in ("product", "baseline"):
            orders = episode_orders(con, cond)
            rb = Rebuilder(run, cond, orders, tmp_path)
            try:
                for rec in (r for r in recs if r.condition == cond):
                    payload, delivered = rb.rebuild(consult_request(con, rec, run.tasks[rec.task_id]), null_context())
                    assert payload["question"] == rec.exchange.question
                    if cond == "product":
                        cited = {c for c in rec.exchange.response.cited_source_ids if c.startswith("card:")}
                        assert cited <= set(delivered)                   # the same retrieval as in the run
                        past = [o for c in payload["evidence_cards"] if c["kind"] == "past_observation"
                                for o in c["observation_ids"]]
                    else:
                        assert delivered == ["initial:text"] and payload["initial_text"]["text"]
                        past = OBS.findall(payload["own_memory"])
                    assert all(orders[episode_of(o)] < rec.episode_order for o in past)
                    saw_past[cond] = saw_past.get(cond, False) or bool(past)
            finally:
                rb.close()
        assert saw_past == {"product": True, "baseline": True}
    finally:
        con.close()
    assert digest(offline_run) == before                                  # the target run is never written
