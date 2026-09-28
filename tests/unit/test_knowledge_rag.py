"""T05 retrieval boundaries: B15 (structure/units/OCR provenance), hybrid retrieval + embedder pinning,
conditional KG paths, B17 part (no product optimizer; baseline gets no product RAG), prebuild cost phase."""
from pathlib import Path

import pytest
import yaml

import labgene.knowledge as knowledge_pkg
from labgene.config import Limits, RetrievalConfig
from labgene.contracts import MemoryScope, PrivateTaskAssets
from labgene.costs import CallContext
from labgene.knowledge.build import build_initial_state, load_baseline_initial
from labgene.knowledge.cards import literature_card
from labgene.knowledge.gate import FixtureMarkerChecker, KnowledgeInfraError
from labgene.knowledge.kg import FixtureKGExtractor, Ontology, kg_paths
from labgene.knowledge.parse import OCR_UNCERTAINTY, chunk_document, parse_document
from labgene.knowledge.retrieval import HashEmbedder, tokenize
from labgene.knowledge.store import KnowledgeStore

FIX = Path(__file__).resolve().parents[1] / "fixtures"
BUNDLES = [PrivateTaskAssets.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))).answer_bundle
           for p in sorted((FIX / "private").glob("*.yaml"))]
ONTO = Ontology.load(FIX / "ontology" / "profile.yaml")
SCOPE = MemoryScope(run_id="t", condition="product", set_id="smoke", set_rep=1)
CTX = CallContext(sink=lambda e: None)


def parsed(name):
    p = FIX / "corpus" / f"{name}.md"
    doc = parse_document(p.read_text(encoding="utf-8"), p.stem)
    return doc, chunk_document(doc)


def product_store(state_dir, events=None, **kw) -> KnowledgeStore:
    ctx = CallContext(sink=events.append) if events is not None else CTX
    build_initial_state(state_dir, "product", FIX / "corpus", FixtureMarkerChecker(), HashEmbedder(), BUNDLES, ctx,
                        set_scope="smoke", ontology=ONTO, extractor=FixtureKGExtractor())
    return KnowledgeStore(state_dir, FixtureMarkerChecker(), BUNDLES, "smoke", embedder=HashEmbedder(), ontology=ONTO, **kw)


def test_b15_split_table_repeats_header_and_units_and_keeps_parent_locator():
    doc, chunks = parsed("catalyst_screening_manual")
    parts = [c for c in chunks if c.kind == "table"]
    assert len(parts) == 3
    rows = []
    for c in parts:
        lines = c.text.split("\n")
        assert lines[0] == "| setpoint | measured | deviation |" and lines[2] == "| [degC] | [degC] | [degC] |"
        assert c.locator["table"]["units"] == lines[2] and c.locator["page"] == 2
        assert c.locator["section"] == ["Catalyst screening manual", "Thermostat calibration table"]
        s, e = c.locator["span"]
        ps, pe = c.locator["table"]["parent_span"]
        assert doc.text[s:e] == "\n".join(lines[3:]) and ps <= s < e <= pe     # rows verbatim at their own span
        assert c.header.startswith("Thermostat and dosing manual for catalyst screening rigs | ")
        rows += lines[3:]
    assert rows == doc.text[ps:pe].split("\n")[3:]                            # nothing lost or duplicated


def test_b15_units_ranges_inequalities_formulas_stay_verbatim_and_linked_to_source():
    doc, chunks = parsed("ridge_kinetics_notes")
    for c in chunks:
        s, e = c.locator["span"]
        assert doc.text[s:e] == c.text and c.locator["doc_id"] == "ridge_kinetics_notes"
    joined = "\n".join(c.text for c in chunks)
    for s in ["T_max ≤ 120 °C", "t_hold is 1–60 min", "80–95 %", "40 degC < T < 100 degC", "< 5 %",
              r"\exp\left(-\frac{E_a}{R T}\right)", "| [1] | [degC] | [min] | [%] |"]:
        assert s in joined, s
    kinds = {c.kind for c in chunks}
    assert {"formula", "procedure", "table", "paragraph"} <= kinds
    assert tokenize("Al2O3 at 1.2 mol% and 90 °C vs 90 degC, residence_time") == \
        ["Al2O3", "at", "1.2", "mol%", "and", "90", "degc", "vs", "90", "degc", "residence_time"]


def test_b15_ocr_suspect_values_are_flagged_uncertain_never_verified(tmp_path):
    _, chunks = parsed("catalyst_screening_manual")
    scanned = [c for c in chunks if "11O degC" in c.text]
    assert scanned and all(c.uncertain and c.uncertainty == [OCR_UNCERTAINTY] for c in scanned)
    assert not any(c.uncertain for c in chunks if c.locator["page"] != 3)
    inline = chunk_document(parse_document("---\ntitle: t\n---\n# S\nPump dead volume 2.4 mL [ocr?] measured.\n", "x"))
    assert inline[0].uncertain
    store = product_store(tmp_path)
    hit = next(v for v in store.retrieve("maximum jacket temperature service note", CTX, 10) if "11O" in v.text)
    card = literature_card(store, hit.source_id, store.access(SCOPE))
    assert card.excerpt == hit.text and OCR_UNCERTAINTY in card.uncertainty
    assert "verified" not in card.model_projection()["locator"]


def test_hybrid_retrieval_fuses_bm25_and_dense_and_only_returns_approved_chunks(tmp_path):
    events = []
    store = product_store(tmp_path, events)
    views = store.retrieve("Al2O3 loading conversion mol%", CallContext(sink=events.append), 5)
    assert views[0].source_id == "doc:catalyst_screening_manual#0"
    assert all(store.visible(v.source_id) is not None for v in views)
    assert all(v.locator and v.locator["doc_id"] for v in views)
    kinds = [e.kind for e in events if e.phase == "runtime"]
    assert "retrieval" in kinds and "embedding" in kinds
    expanded = product_store(tmp_path, retrieval=RetrievalConfig(query_expansion=True))
    assert expanded.retrieve("alumina conversion", CTX, 5)[0].source_id == "doc:catalyst_screening_manual#0"


def test_vector_index_refuses_mixing_embedder_model_or_dimensions_or_silently_dropping_it(tmp_path):
    product_store(tmp_path).close()
    for other in (HashEmbedder(dimensions=128), HashEmbedder(model="other-embedder")):
        with pytest.raises(ValueError, match="refusing to mix"):
            KnowledgeStore(tmp_path, FixtureMarkerChecker(), BUNDLES, "smoke", embedder=other)
    with pytest.raises(ValueError, match="pinned to an embedder"):              # no silent BM25-only fallback
        KnowledgeStore(tmp_path, FixtureMarkerChecker(), BUNDLES, "smoke")
    with pytest.raises(ValueError, match="needs an embedder"):
        build_initial_state(tmp_path / "p", "product", FIX / "corpus", FixtureMarkerChecker(), None, BUNDLES, CTX,
                            set_scope="smoke")
    with pytest.raises(ValueError, match="development_only"):
        KnowledgeStore(tmp_path / "live", FixtureMarkerChecker(), BUNDLES, "smoke", execution_mode="evaluation")


def test_b10_prebuild_interrupted_after_chunks_are_committed_is_completed_on_rerun(tmp_path):
    class FailsOnce(HashEmbedder):
        calls = 0

        def embed(self, texts, kind):
            FailsOnce.calls += 1
            res = super().embed(texts, kind)
            return res.model_copy(update={"status": "infra_error"}) if FailsOnce.calls == 1 else res

    args = (tmp_path, "product", FIX / "corpus", FixtureMarkerChecker())
    kw = {"set_scope": "smoke", "ontology": ONTO, "extractor": FixtureKGExtractor()}
    with pytest.raises(KnowledgeInfraError):
        build_initial_state(*args, FailsOnce(), BUNDLES, CTX, **kw)
    man = build_initial_state(*args, HashEmbedder(), BUNDLES, CTX, **kw)
    assert man["documents"][0]["status"] == "ok"                              # the interrupted doc was finished
    store = KnowledgeStore(tmp_path, FixtureMarkerChecker(), BUNDLES, "smoke", embedder=HashEmbedder(), ontology=ONTO)
    clean = product_store(tmp_path / "clean")                                  # same inputs, never interrupted
    assert set(store.bm25.docs) == set(clean.bm25.docs) == set(store.vectors.vectors) != set()
    assert {r.id for r in store.relations()} == {r.id for r in clean.relations()} != set()


def test_kg_links_registered_ids_and_never_joins_incompatible_conditions(tmp_path):
    store = product_store(tmp_path)
    assert ONTO.link("Al2O3") == "CHEBI:30187" and ONTO.link("unobtainium") == "unresolved"
    two_hop = [p for p in kg_paths(store, "temperature") if len(p) == 2]
    assert [(p[0].object_label, p[1].object_label) for p in two_hop] == [("rate constant", "yield")]
    assert two_hop[0][0].conditions["range"] == {"degC": [40.0, 100.0]} and two_hop[0][0].subject == "quantitykind:Temperature"
    parent = "doc:catalyst_screening_manual#0"
    for obj, cond in [("selectivity", {"material": "catalyst Z on silica"}),                   # other material
                      ("TON", {"material": "substrate R", "range": {"degC": [110.0, 120.0]}}),  # disjoint range
                      ("conversion", {})]:                                                   # general statement
        store.add_relation("rate constant", "increases", obj, [parent], "kg:test", CTX, conditions=cond)
    store.add_relation("unobtainium", "increases", "yield", [parent], "kg:test", CTX)
    paths = kg_paths(store, "temperature")
    ends = {p[1].object_label for p in paths if len(p) == 2}
    assert ends == {"yield", "conversion"}                            # only condition-compatible joins
    assert all(len(p) <= 2 for p in paths)
    assert not kg_paths(store, "unobtainium")                          # unresolved entities are never a path start


def test_b17_product_knowledge_tooling_has_no_bo_optimizer_and_baseline_gets_no_product_rag(tmp_path):
    forbidden = ("acquisition", "expected_improvement", "gaussianprocess", "gaussian_process", "upper_confidence",
                 "bayesian_opt", "botorch", "skopt", "gpytorch", "sklearn")
    for p in Path(knowledge_pkg.__file__).parent.glob("*.py"):
        src = p.read_text(encoding="utf-8").casefold()
        assert not [t for t in forbidden if t in src], p.name
    events = []
    man = build_initial_state(tmp_path, "baseline", FIX / "corpus", FixtureMarkerChecker(), HashEmbedder(), BUNDLES,
                              CallContext(sink=events.append), set_scope="smoke",
                              baseline_initial=FIX / "corpus" / "general_background.md",
                              limits=Limits(baseline_initial_text_max_chars=300))
    text = load_baseline_initial(tmp_path)
    assert man["baseline_initial"] == {"chars": 300, "truncated": True, "available": True} and len(text) == 300
    assert not (tmp_path / "index" / "vectors.jsonl").exists() and not (tmp_path / "index" / "bm25.json").exists()
    assert "documents" not in man and {e.phase for e in events} == {"prebuild"}


def test_b22_prebuild_costs_are_tagged_prebuild(tmp_path):
    events = []
    product_store(tmp_path, events)
    assert {e.kind for e in events} >= {"gate", "embedding"} and {e.phase for e in events} == {"prebuild"}
