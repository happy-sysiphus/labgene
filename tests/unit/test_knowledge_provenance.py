"""T04 provenance boundary B14: blocking a source invalidates every derivative transitively; new derived text is
gated on its own; answer-bundle / policy / scope changes never reuse an old allow."""
from pathlib import Path

import pytest
import yaml

from labgene.contracts import GateStatus, MemoryScope, PrivateTaskAssets
from labgene.costs import CallContext
from labgene.knowledge.base import GateDecision
from labgene.knowledge.build import build_initial_state
from labgene.knowledge.cards import literature_card
from labgene.knowledge.gate import FixtureMarkerChecker
from labgene.knowledge.gated import GatedSearch
from labgene.knowledge.kg import FixtureKGExtractor, Ontology, kg_paths
from labgene.knowledge.retrieval import HashEmbedder
from labgene.knowledge.search import FixtureSearchProvider
from labgene.knowledge.store import KnowledgeStore

FIX = Path(__file__).resolve().parents[1] / "fixtures"
BUNDLES = [PrivateTaskAssets.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))).answer_bundle
           for p in sorted((FIX / "private").glob("*.yaml"))]
ONTO = Ontology.load(FIX / "ontology" / "profile.yaml")
SCOPE = MemoryScope(run_id="t", condition="product", set_id="smoke", set_rep=1)
CTX = CallContext(sink=lambda e: None)
NOTES = "doc:ridge_kinetics_notes"


def product_store(state_dir) -> KnowledgeStore:
    build_initial_state(state_dir, "product", FIX / "corpus", FixtureMarkerChecker(), HashEmbedder(), BUNDLES, CTX,
                        set_scope="smoke", ontology=ONTO, extractor=FixtureKGExtractor())
    return KnowledgeStore(state_dir, FixtureMarkerChecker(), BUNDLES, "smoke", embedder=HashEmbedder(), ontology=ONTO)


class CountingChecker(FixtureMarkerChecker):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    def check(self, *a, **kw) -> GateDecision:
        self.calls += 1
        return super().check(*a, **kw)


def test_b14_blocked_source_invalidates_chunks_summaries_relations_vectors_cards_transitively(tmp_path):
    store = product_store(tmp_path)
    chunks = [r["id"] for r in store.db.execute("SELECT id FROM artifacts WHERE id LIKE ?", (NOTES + "#%",))]
    summary, st = store.register_derived("summary", "Temperature raises the rate constant of substrate R.",
                                         [chunks[0]], "internal_summarizer:test", CTX)
    card = literature_card(store, summary, store.access(SCOPE))
    rels = [r.id for r in store.relations() if any(s.startswith(NOTES) for s in r.source_ids)]
    assert st == GateStatus.allow and rels and summary in store.bm25.docs and summary in store.vectors.vectors
    affected = set(store.invalidate(NOTES))
    expected = {NOTES, *chunks, summary, card.card_id, *rels, *(f"vec:{c}" for c in chunks), f"vec:{summary}"}
    assert expected <= affected
    reopened = KnowledgeStore(tmp_path, FixtureMarkerChecker(), BUNDLES, "smoke", embedder=HashEmbedder(), ontology=ONTO)
    for s in (store, reopened):                       # in memory AND persisted index files
        assert not affected & set(s.bm25.docs) and not affected & set(s.vectors.vectors)
        assert not affected & {v.source_id for v in s.retrieve("temperature rate constant substrate R yield", CTX, 50)}
        assert not [p for p in kg_paths(s, "temperature") if affected & {r.id for r in p}]
        assert literature_card(s, chunks[0], s.access(SCOPE)) is None
        assert GatedSearch(s, FixtureSearchProvider(FIX / "web_corpus")).open(summary, CTX).status == "unavailable"
    # other documents are untouched
    assert store.visible("doc:catalyst_screening_manual#0") is not None


def test_b14_blocked_web_source_drops_cached_search_results(tmp_path):
    class Counting(FixtureSearchProvider):
        calls = 0

        def search(self, q, n):
            Counting.calls += 1
            return super().search(q, n)

    tool = GatedSearch(KnowledgeStore(tmp_path, FixtureMarkerChecker(), BUNDLES, "smoke"), Counting(FIX / "web_corpus"))
    q = "Arrhenius temperature dependence"
    src = next(v.source_id for v in tool.search(q, CTX) if v.title.startswith("Arrhenius"))
    tool.search(q, CTX)
    assert Counting.calls == 1                                         # served from the cache artifact
    affected = tool.store.invalidate(src)
    assert any(a.startswith("cache:search:") for a in affected)
    again = tool.search(q, CTX)
    assert Counting.calls == 2 and src not in {v.source_id for v in again}
    assert tool.open(src, CTX).status == "unavailable"


def test_b14_new_derived_text_is_gated_itself_even_from_an_allowed_parent(tmp_path):
    store = product_store(tmp_path)
    parent = f"{NOTES}#0"
    sid, st = store.register_derived("summary", "Summary: the optimum sits in LEAK-RIDGE-7Q2 territory.", [parent],
                                     "internal_summarizer:test", CTX)
    assert st == GateStatus.block and store.visible(sid) is None and sid not in store.bm25.docs
    assert sid not in {v.source_id for v in store.retrieve("summary optimum territory", CTX, 50)}
    rid, rst = store.add_relation("time", "increases", "yield", [parent], "kg:test", CTX,
                                  conditions={"fixed": {"note": "LEAK-RIDGE-7Q2"}})
    assert rst == GateStatus.block and rid not in {r.id for r in store.relations()}
    with pytest.raises(ValueError):                                    # nothing derives from unapproved material
        store.register_derived("summary", "Summary of a blocked copy.", ["doc:ridge_answer_copy"], "x", CTX)


def test_b14_multi_parent_relation_falls_with_any_parent_and_regenerates_only_from_approved(tmp_path):
    store = product_store(tmp_path)
    a, b = f"{NOTES}#0", "doc:reaction_engineering_review#0"
    rid, st = store.add_relation("temperature", "increases", "yield", [a, b], "kg:test", CTX)
    assert st == GateStatus.allow and rid in {r.id for r in store.relations()}
    store.invalidate(NOTES)
    assert rid not in {r.id for r in store.relations()}
    with pytest.raises(ValueError):
        store.add_relation("temperature", "increases", "yield", [a, b], "kg:test", CTX)
    rid2, st2 = store.add_relation("temperature", "increases", "yield", [b], "kg:test", CTX)
    assert st2 == GateStatus.allow and rid2 != rid and rid2 in {r.id for r in store.relations()}
    assert store.ancestors(rid2) == sorted([b, "doc:reaction_engineering_review"])


def test_b14_stored_approvals_web_sources_and_search_cache_are_not_reused_under_new_bundle_scope_or_policy(tmp_path):
    store = product_store(tmp_path)
    q = "Arrhenius temperature dependence"
    assert GatedSearch(store, FixtureSearchProvider(FIX / "web_corpus")).search(q, CTX)   # sticky source + cache
    store.close()
    cat, ridge = BUNDLES
    v2 = ridge.model_copy(update={"version": "2",
                                  "secret_markers": [*ridge.secret_markers, "substrate R", "activation energy"]})
    for bundles, scope, chk in [([cat, v2], "smoke", FixtureMarkerChecker()), (BUNDLES, "main", FixtureMarkerChecker()),
                                (BUNDLES, "smoke", FixtureMarkerChecker(policy_version="fixture-policy-v2"))]:
        with pytest.raises(ValueError, match="fresh state dir"):
            KnowledgeStore(tmp_path, chk, bundles, scope, embedder=HashEmbedder(), ontology=ONTO)
    with pytest.raises(ValueError, match="fresh state dir"):
        build_initial_state(tmp_path, "product", FIX / "corpus", FixtureMarkerChecker(), HashEmbedder(), [cat, v2], CTX,
                            set_scope="smoke")
    # what v1 approved is judged afresh under v2 (not vacuous: v1 served both)
    build_initial_state(tmp_path / "v2", "product", FIX / "corpus", FixtureMarkerChecker(), HashEmbedder(), [cat, v2],
                        CTX, set_scope="smoke")
    s2 = KnowledgeStore(tmp_path / "v2", FixtureMarkerChecker(), [cat, v2], "smoke", embedder=HashEmbedder())
    assert not [v for v in s2.retrieve("substrate R temperature rate constant", CTX, 50) if "substrate R" in v.text]
    assert not [v for v in GatedSearch(s2, FixtureSearchProvider(FIX / "web_corpus")).search(q, CTX)
                if "activation energy" in f"{v.title} {v.text}".casefold()]
    # same identity: the persisted gate cache is reused; changed content misses it
    chk = CountingChecker()
    same = KnowledgeStore(tmp_path, chk, BUNDLES, "smoke", embedder=HashEmbedder(), ontology=ONTO)
    text, src = "Notebook entry about thermostat ramps and jacket limits.", {"source": "corpus:x"}
    assert same.gate(text, src, CTX) == same.gate(text, src, CTX) == GateStatus.allow and chk.calls == len(BUNDLES)
    assert same.gate(text + " Revised.", src, CTX) == GateStatus.allow and chk.calls == 2 * len(BUNDLES)


class Flaky(FixtureMarkerChecker):
    """Same checker id/policy as the fixture checker, but errors while `failing` is set."""
    failing = True

    def check(self, text, bundle, context, ctx) -> GateDecision:
        if self.failing:
            return GateDecision(status=GateStatus.error, policy_version=self.policy_version, checker=self.checker_id,
                                cache_key="")
        return super().check(text, bundle, context, ctx)


def test_b13_derived_text_gate_error_is_not_frozen_and_is_gated_again_on_retry(tmp_path):
    product_store(tmp_path).close()
    flaky = Flaky()
    store = KnowledgeStore(tmp_path, flaky, BUNDLES, "smoke", embedder=HashEmbedder(), ontology=ONTO)
    parent, text = f"{NOTES}#0", "A careful summary of thermostat ramp behaviour in the batch reactor."
    sid, st = store.register_derived("summary", text, [parent], "internal_summarizer:test", CTX)
    rid, rst = store.add_relation("time", "increases", "yield", [parent], "kg:test", CTX)
    assert (st, rst) == (GateStatus.error, GateStatus.error) and store.get(sid) is None and store.get(rid) is None
    assert rid not in {r.id for r in store.relations()}
    flaky.failing = False
    assert store.register_derived("summary", text, [parent], "internal_summarizer:test", CTX) == (sid, GateStatus.allow)
    assert store.add_relation("time", "increases", "yield", [parent], "kg:test", CTX) == (rid, GateStatus.allow)
    assert store.visible(sid) is not None and sid in store.bm25.docs and rid in {r.id for r in store.relations()}


def test_b14_invalidation_survives_a_second_store_instance_writing_the_indexes(tmp_path):
    a = product_store(tmp_path)
    b = KnowledgeStore(tmp_path, FixtureMarkerChecker(), BUNDLES, "smoke", embedder=HashEmbedder(), ontology=ONTO)
    affected = set(a.invalidate(NOTES))
    summary, _ = b.register_derived("summary", "Catalyst loading raises conversion in the screening rig.",
                                    ["doc:catalyst_screening_manual#0"], "internal_summarizer:test", CTX)
    fresh = KnowledgeStore(tmp_path, FixtureMarkerChecker(), BUNDLES, "smoke", embedder=HashEmbedder(), ontology=ONTO)
    assert affected & set(b.vectors.vectors) == set() and not affected & set(fresh.bm25.docs)
    assert not affected & set(fresh.vectors.vectors)
    assert summary in fresh.bm25.docs and summary in fresh.vectors.vectors
