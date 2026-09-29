"""U39: a built initial state is re-gated under the main plan's bundles and set scope, never loosened, with the build
rules (a blocked chunk blocks its document, a held chunk leaves the indexes, descendants of withdrawn items fall)."""
import json
from pathlib import Path

import pytest
import yaml

from labgene.contracts import GateStatus, PrivateTaskAssets
from labgene.costs import CallContext
from labgene.knowledge.base import GateDecision
from labgene.knowledge.build import BASELINE_INITIAL_ID, build_initial_state, load_baseline_initial
from labgene.knowledge.gate import FixtureMarkerChecker, KnowledgeInfraError
from labgene.knowledge.kg import FixtureKGExtractor, Ontology
from labgene.knowledge.regate import regate_state
from labgene.knowledge.retrieval import HashEmbedder
from labgene.knowledge.store import KnowledgeStore, exposure

FIX = Path(__file__).resolve().parents[1] / "fixtures"
RIDGE, CAT = [PrivateTaskAssets.model_validate(yaml.safe_load((FIX / "private" / f"fixture_{n}.yaml").read_text(
    encoding="utf-8"))).answer_bundle for n in ("ridge", "catalyst")]
ONTO = Ontology.load(FIX / "ontology" / "profile.yaml")
CTX = CallContext(sink=lambda e: None)
BASELINE_SOURCE = "corpus:general_background.md"


class Scripted(FixtureMarkerChecker):
    """The fixture gate, except that under the catalyst bundle the scripted exposure texts get the scripted verdict."""

    def __init__(self, verdicts=None):
        super().__init__()
        self.verdicts, self.calls = dict(verdicts or {}), 0

    def check(self, text, bundle, context, ctx):
        self.calls += 1
        if bundle.bundle_id == CAT.bundle_id and text in self.verdicts:
            return GateDecision(status=self.verdicts[text], policy_version=self.policy_version, checker=self.checker_id)
        return super().check(text, bundle, context, ctx)


def product(state, checker=None):
    build_initial_state(state, "product", FIX / "corpus", checker or FixtureMarkerChecker(), HashEmbedder(), [RIDGE],
                        CTX, set_scope="pilot", ontology=ONTO, extractor=FixtureKGExtractor())


def open_store(state, bundles, scope, checker=None):
    return KnowledgeStore(state, checker or FixtureMarkerChecker(), bundles, scope, embedder=HashEmbedder(), ontology=ONTO)


def rows(s, sql, *a):
    return [tuple(r) for r in s.db.execute(sql, a)]


def test_u39_regate_withdraws_with_the_build_rules_and_never_loosens(tmp_path):
    state = tmp_path / "product"
    product(state)
    s = open_store(state, [RIDGE], "pilot")
    visible = {r[0] for r in rows(s, "SELECT id FROM artifacts WHERE valid=1 AND gate_status='allow'")}
    hidden = {r[0] for r in rows(s, "SELECT id FROM artifacts WHERE NOT (valid=1 AND gate_status='allow')")}
    rels = rows(s, "SELECT r.id, r.source_ids FROM relations r JOIN artifacts a ON a.id=r.id "
                   "WHERE a.valid=1 AND a.gate_status='allow' ORDER BY r.id")
    by_doc = {}
    for (cid,) in rows(s, "SELECT id FROM artifacts WHERE kind='chunk' AND valid=1 AND gate_status='allow' ORDER BY id"):
        by_doc.setdefault(cid.split("#")[0], []).append(cid)
    rel_chunks = {c for _, src in rels for c in json.loads(src)}
    docs = [d for d, cs in by_doc.items() if len(cs) >= 2 and set(cs) & rel_chunks]
    assert len(docs) >= 2, by_doc
    blocked_doc, held_doc = docs[0], docs[1]
    held = next(c for c in by_doc[held_doc] if c in rel_chunks)
    target_rel = next(r for r, src in rels if not set(json.loads(src)) & {*by_doc[blocked_doc], held})

    def exposed(aid):
        row = s.get(aid)
        return exposure(row["text"], s.meta(row))
    verdicts = {exposed(by_doc[blocked_doc][-1]): GateStatus.block, exposed(held): GateStatus.hold,
                exposed(target_rel): GateStatus.block}
    s.close()

    checker = Scripted(verdicts)
    report = regate_state(state, checker, [RIDGE, CAT], "main", CTX, baseline_source=BASELINE_SOURCE, corpus_dir=FIX / "corpus",
                          embedder=HashEmbedder(), ontology=ONTO)
    assert {(w["id"], w["status"]) for w in report["withdrawn"]} == {
        (by_doc[blocked_doc][-1], "block"), (held, "hold"), (target_rel, "block")}
    with pytest.raises(ValueError, match="gated under other"):          # the pilot identity no longer opens it
        open_store(state, [RIDGE], "pilot")
    t = open_store(state, [RIDGE, CAT], "main", checker)
    now = {r[0] for r in rows(t, "SELECT id FROM artifacts WHERE valid=1 AND gate_status='allow'")}
    assert now <= visible and not (now & hidden)                         # never loosened
    gone = visible - now
    assert set(by_doc[blocked_doc]) <= gone and held in gone and target_rel in gone
    assert not [r for r, src in rels if r in now and set(json.loads(src)) & {*by_doc[blocked_doc], held}]
    assert set(by_doc[held_doc]) - {held} <= now                        # a hold does not block the document
    assert rows(t, "SELECT gate_status, valid FROM artifacts WHERE id=?", held) == [("hold", 1)]
    indexed = set(t.bm25.docs) | set(t.vectors.vectors)
    assert not indexed & ({*by_doc[blocked_doc], held}) and indexed <= now
    assert {v.source_id for v in t.retrieve("ridge temperature catalyst", CTX, top_k=50)} <= now
    t.close()
    n = checker.calls
    assert regate_state(state, checker, [RIDGE, CAT], "main", CTX, baseline_source=BASELINE_SOURCE, corpus_dir=FIX / "corpus",
                        embedder=HashEmbedder(), ontology=ONTO) == report and checker.calls == n   # done: no calls


def test_u39_a_gate_error_leaves_the_state_in_progress_and_a_rerun_finishes_it(tmp_path):
    state = tmp_path / "product"
    product(state)
    s = open_store(state, [RIDGE], "pilot")
    first = next(r[0] for r in rows(s, "SELECT id FROM artifacts WHERE kind='chunk' AND gate_status='allow' ORDER BY id"))
    text = exposure(s.get(first)["text"], s.meta(s.get(first)))
    s.close()
    with pytest.raises(KnowledgeInfraError, match="in_progress"):
        regate_state(state, Scripted({text: GateStatus.error}), [RIDGE, CAT], "main", CTX,
                     baseline_source=BASELINE_SOURCE, corpus_dir=FIX / "corpus", embedder=HashEmbedder(), ontology=ONTO)
    with pytest.raises(ValueError, match="being re-gated"):              # nothing may use a half-checked state
        open_store(state, [RIDGE, CAT], "main")
    report = regate_state(state, Scripted(), [RIDGE, CAT], "main", CTX, baseline_source=BASELINE_SOURCE, corpus_dir=FIX / "corpus",
                          embedder=HashEmbedder(), ontology=ONTO)
    assert report["withdrawn"] == [] and report["answer_bundle_versions"] == sorted(
        f"{b.bundle_id}@{b.version}" for b in (RIDGE, CAT))


def test_u39_baseline_initial_text_is_regated_as_the_build_gated_it(tmp_path):
    state = tmp_path / "baseline"
    build_initial_state(state, "baseline", FIX / "corpus", FixtureMarkerChecker(), None, [RIDGE], CTX,
                        set_scope="pilot", baseline_initial=FIX / "corpus" / "general_background.md")
    text = load_baseline_initial(state)
    assert text
    report = regate_state(state, Scripted({text: GateStatus.hold}), [RIDGE, CAT], "main", CTX,
                          baseline_source=BASELINE_SOURCE, corpus_dir=FIX / "corpus")
    assert [(w["id"], w["status"]) for w in report["withdrawn"]] == [(BASELINE_INITIAL_ID, "hold")]
    assert load_baseline_initial(state) is None


def test_u39_a_state_built_under_the_target_identity_needs_no_regate(tmp_path):
    state = tmp_path / "product"
    product(state)
    with pytest.raises(ValueError, match="nothing to re-gate"):
        regate_state(state, FixtureMarkerChecker(), [RIDGE], "pilot", CTX, baseline_source=BASELINE_SOURCE, corpus_dir=FIX / "corpus",
                     embedder=HashEmbedder(), ontology=ONTO)


def _targets(state):
    s = open_store(state, [RIDGE], "pilot")
    held = next(r[0] for r in rows(s, "SELECT l.parent FROM relations r JOIN lineage l ON l.child = r.id "
                                      "JOIN artifacts a ON a.id = l.parent WHERE a.kind='chunk' ORDER BY l.parent"))
    row = s.get(held)
    text = exposure(row["text"], s.meta(row))
    s.close()
    return held, {text: GateStatus.hold}


def _visible(state, bundles, scope):
    t = open_store(state, bundles, scope)
    out = {r[0] for r in rows(t, "SELECT id FROM artifacts WHERE valid=1 AND gate_status='allow'")}
    idx = set(t.bm25.docs) | set(t.vectors.vectors)
    t.close()
    return out, idx


def test_u39_concurrent_prefetch_decides_exactly_as_the_serial_pass(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    product(a)
    product(b)
    held, verdicts = _targets(a)
    kw = dict(baseline_source=BASELINE_SOURCE, corpus_dir=FIX / "corpus", embedder=HashEmbedder(), ontology=ONTO)
    ra = regate_state(a, Scripted(verdicts), [RIDGE, CAT], "main", CTX, **kw)
    rb = regate_state(b, Scripted(verdicts), [RIDGE, CAT], "main", CTX, workers=4, **kw)
    assert [(w["id"], w["status"]) for w in ra["withdrawn"]] == [(w["id"], w["status"]) for w in rb["withdrawn"]] \
        == [(held, "hold")]
    assert _visible(a, [RIDGE, CAT], "main") == _visible(b, [RIDGE, CAT], "main")


def test_u39_a_crash_inside_a_withdrawal_is_finished_by_the_rerun(tmp_path, monkeypatch):
    from labgene.knowledge.retrieval import BM25Index
    state = tmp_path / "product"
    product(state)
    held, verdicts = _targets(state)
    kw = dict(baseline_source=BASELINE_SOURCE, corpus_dir=FIX / "corpus", embedder=HashEmbedder(), ontology=ONTO)
    orig = BM25Index.remove

    def crash(self, ids):
        raise KeyboardInterrupt("killed mid-withdrawal")
    monkeypatch.setattr(BM25Index, "remove", crash)
    with pytest.raises(KeyboardInterrupt):
        regate_state(state, Scripted(verdicts), [RIDGE, CAT], "main", CTX, **kw)
    monkeypatch.setattr(BM25Index, "remove", orig)
    with pytest.raises(ValueError, match="being re-gated"):
        open_store(state, [RIDGE, CAT], "main")
    t = KnowledgeStore(state, FixtureMarkerChecker(), [RIDGE, CAT], "main", embedder=HashEmbedder(), ontology=ONTO,
                       regating=True)
    assert t.visible(held) is not None           # its descendants fell first; the item itself is still 'allow'
    t.close()
    report = regate_state(state, Scripted(verdicts), [RIDGE, CAT], "main", CTX, **kw)
    assert [(w["id"], w["status"]) for w in report["withdrawn"]] == [(held, "hold")]
    now, idx = _visible(state, [RIDGE, CAT], "main")
    assert held not in now and held not in idx and idx <= now


def test_u39_u40_seeded_verdicts_are_the_same_requests_under_another_set_scope(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    product(a)
    product(b)
    held, verdicts = _targets(a)
    kw = dict(baseline_source=BASELINE_SOURCE, corpus_dir=FIX / "corpus", embedder=HashEmbedder(), ontology=ONTO)
    first = regate_state(a, Scripted(verdicts), [RIDGE, CAT], "main", CTX, **kw)
    counting = Scripted(verdicts)
    second = regate_state(b, counting, [RIDGE, CAT], "pilot", CTX, seed=(a / "knowledge.sqlite", "main"), **kw)
    assert counting.calls == 0 and second["seeded_verdicts_last_pass"] > 0      # every verdict came from the seed
    assert [(w["id"], w["status"]) for w in second["withdrawn"]] == [(w["id"], w["status"]) for w in first["withdrawn"]]
    assert _visible(a, [RIDGE, CAT], "main") == _visible(b, [RIDGE, CAT], "pilot")
    # a re-gated state can be re-gated again for another plan (never loosened)
    third = regate_state(b, Scripted(verdicts), [RIDGE], "pilot2", CTX, **kw)
    assert third["withdrawn"] == [] and _visible(b, [RIDGE], "pilot2")[0] <= _visible(a, [RIDGE, CAT], "main")[0]


def test_review_held_chunk_blocked_under_a_new_bundle_blocks_its_document(tmp_path):
    state = tmp_path / "product"
    product(state)
    s = open_store(state, [RIDGE], "pilot")
    held = next(r[0] for r in rows(s, "SELECT a.id FROM artifacts a WHERE a.kind='chunk' AND a.gate_status='allow' "
                                      "AND NOT EXISTS (SELECT 1 FROM relations r WHERE r.source_ids LIKE '%' || a.id || '%') "
                                      "AND (SELECT count(*) FROM artifacts b WHERE b.id LIKE substr(a.id, 1, instr(a.id, '#')) || '%' "
                                      "AND b.kind='chunk') >= 2 ORDER BY a.id"))
    doc = held.split("#")[0]
    with s.db:                                                # as if the build had held it (never indexed)
        s.db.execute("UPDATE artifacts SET gate_status='hold' WHERE id=?", (held,))
    s.bm25.remove([held])
    s.vectors.remove([held])
    text = exposure(s.get(held)["text"], s.meta(s.get(held)))
    siblings = [r[0] for r in rows(s, "SELECT id FROM artifacts WHERE kind='chunk' AND id LIKE ? AND id != ? "
                                      "AND gate_status='allow'", doc + "#%", held)]
    s.close()
    report = regate_state(state, Scripted({text: GateStatus.block}), [RIDGE, CAT], "main", CTX,
                          baseline_source=BASELINE_SOURCE, corpus_dir=FIX / "corpus", embedder=HashEmbedder(),
                          ontology=ONTO)
    assert [(w["id"], w.get("document_blocked")) for w in report["withdrawn"]] == [(held, doc)]
    now, idx = _visible(state, [RIDGE, CAT], "main")
    assert siblings and not set(siblings) & now and not set(siblings) & idx


def test_review_a_kill_inside_a_document_block_leaves_no_withdrawn_id_indexed(tmp_path, monkeypatch):
    from labgene.knowledge.retrieval import BM25Index
    state = tmp_path / "product"
    product(state)
    s = open_store(state, [RIDGE], "pilot")
    chunk = next(r[0] for r in rows(s, "SELECT id FROM artifacts WHERE kind='chunk' AND gate_status='allow' ORDER BY id"))
    text = exposure(s.get(chunk)["text"], s.meta(s.get(chunk)))
    s.close()
    kw = dict(baseline_source=BASELINE_SOURCE, corpus_dir=FIX / "corpus", embedder=HashEmbedder(), ontology=ONTO)
    orig = BM25Index.remove

    def crash(self, ids):
        raise KeyboardInterrupt("killed inside invalidate")
    monkeypatch.setattr(BM25Index, "remove", crash)
    with pytest.raises(KeyboardInterrupt):
        regate_state(state, Scripted({text: GateStatus.block}), [RIDGE, CAT], "main", CTX, **kw)
    monkeypatch.setattr(BM25Index, "remove", orig)
    report = regate_state(state, Scripted({text: GateStatus.block}), [RIDGE, CAT], "main", CTX, **kw)
    now, idx = _visible(state, [RIDGE, CAT], "main")
    assert chunk not in now and idx <= now
    assert not [w for w in report["withdrawn"] if w.get("applying")]


def test_u42_adopting_a_new_checker_keeps_every_approval_and_decides_new_items_with_it(tmp_path):
    from labgene.knowledge.regate import adopt_checker

    class Other(FixtureMarkerChecker):
        checker_id = "fixture-marker-v2"
    state = tmp_path / "product"
    product(state)
    before, idx = _visible(state, [RIDGE], "pilot")
    entry = adopt_checker(state, [RIDGE], "pilot", Other(), note="user: gate on another model")
    assert entry["checker"] == "fixture-marker-v2"
    with pytest.raises(ValueError, match="gated under other"):
        open_store(state, [RIDGE], "pilot")                           # the old checker no longer decides here
    t = KnowledgeStore(state, Other(), [RIDGE], "pilot", embedder=HashEmbedder(), ontology=ONTO)
    assert {r[0] for r in rows(t, "SELECT id FROM artifacts WHERE valid=1 AND gate_status='allow'")} == before
    assert json.loads(t.get_state("checker_history"))[-1]["note"] == "user: gate on another model"
    t.close()
    assert adopt_checker(state, [RIDGE], "pilot", Other(), note="again")["unchanged"]


def test_u43_previous_records_stand_but_every_known_verdict_is_applied(tmp_path):
    from types import SimpleNamespace
    from labgene.knowledge.regate import accept_previous_records

    class Opus(FixtureMarkerChecker):
        checker_id = "llm:claude_code:opus:gate"

        def check(self, text, bundle, context, ctx):
            raise AssertionError("no model check may run")
    known_state = tmp_path / "known"
    product(known_state)
    held, verdicts = _targets(known_state)
    regate_state(known_state, Scripted(verdicts), [RIDGE, CAT], "main", CTX, baseline_source=BASELINE_SOURCE,
                 corpus_dir=FIX / "corpus", embedder=HashEmbedder(), ontology=ONTO)     # a hold is known for `held`
    state = tmp_path / "product"
    product(state)
    before, _ = _visible(state, [RIDGE], "pilot")
    astra_like = SimpleNamespace(checker_id=FixtureMarkerChecker.checker_id, policy_version="fixture-policy-v1")
    rep = accept_previous_records(state, [RIDGE, CAT], "pilot", Opus(),
                                  [(known_state / "knowledge.sqlite", "main", astra_like)], "U43 test",
                                  baseline_source=BASELINE_SOURCE, corpus_dir=FIX / "corpus",
                                  embedder=HashEmbedder(), ontology=ONTO)
    assert [(w["id"], w["status"]) for w in rep["withdrawn"]] == [(held, "hold")]    # the known hold is applied
    t = KnowledgeStore(state, Opus(), [RIDGE, CAT], "pilot", embedder=HashEmbedder(), ontology=ONTO)
    now = {r[0] for r in rows(t, "SELECT id FROM artifacts WHERE valid=1 AND gate_status='allow'")}
    assert t.get_state("consults_regated") == t.get_state("gate_identity")
    t.close()
    derived = {r[0] for r in rows(open_store_ro(state), "SELECT child FROM lineage WHERE parent=?", held)}
    assert held not in now and now <= before and not derived & now      # the hold and its relations only
    assert before - now == {held} | (derived & before) and rep["kept"] > 0


def open_store_ro(state):
    import sqlite3
    con = sqlite3.connect(f"file:{(state / 'knowledge.sqlite').as_posix()}?mode=ro", uri=True)
    return type("S", (), {"db": con})()
