"""T04 gate boundaries: B11 (blocked material never returned), B12 (legitimate material allowed),
B13 (hold/error/context expansion expose nothing). Offline fixtures = contract checks only."""
import json
from pathlib import Path

import pytest
import yaml

from labgene.contracts import (EvidenceKind, GateStatus, MemoryScope, Observation, PrivateTaskAssets, ProviderResult,
                               ProviderStatus, RunScope, canonical_json)
from labgene.costs import CallContext
from labgene.knowledge.base import UNAVAILABLE_MESSAGE, GateDecision, SourceView
from labgene.knowledge.build import build_initial_state, ingest_document, load_baseline_initial
from labgene.knowledge.cards import kg_path_card, literature_card, observation_card
from labgene.knowledge.gate import FixtureMarkerChecker, KnowledgeInfraError, LLMLeakageChecker, identity_blocked
from labgene.knowledge.gated import GatedSearch
from labgene.knowledge.kg import FixtureKGExtractor, Ontology, kg_paths
from labgene.knowledge.retrieval import HashEmbedder
from labgene.knowledge.search import FixtureSearchProvider, GoogleCSESearchProvider
from labgene.knowledge.store import KnowledgeStore

FIX = Path(__file__).resolve().parents[1] / "fixtures"
BUNDLES = [PrivateTaskAssets.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))).answer_bundle
           for p in sorted((FIX / "private").glob("*.yaml"))]
ONTO = Ontology.load(FIX / "ontology" / "profile.yaml")
SCOPE = MemoryScope(run_id="t", condition="product", set_id="smoke", set_rep=1)
SECRETS = ["LEAK-RIDGE-7Q2", "LEAK-CAT-9X4", "hidden full table fixture_ridge"]
ANSWER_TEXT = ["ridge apex", "apex conditions", "preprint version", "we map the correlated", "accepted manuscript",
               "jointly maximize yield and ton", "def ridge", "source listing", "ridge grid dump", "cheat sheet",
               "recipe from the ridge paper", "answer sheet"]
BLOCKED_URLS = ["papers/ridge-answer", "2601.00001", "papers/catalyst-answer", "repository.fixture.example",
                "code.fixture.example", "data.fixture.example", "blog.fixture.example"]
QUERIES = ["fixture ridge yield temperature time", "optimal catalyst loading fixture coupling TON",
           "hidden full table grid", "ridge recipe apex preprint", "simulator source mirror", "forum reactor settings",
           "review ridges catalysts", "task statement yield of 90 %", "Arrhenius thermostat manual"]


class ScriptedChecker:
    """Test double: status decided by a rule over the exposed text; records what it was asked."""
    development_only = True
    checker_id = "scripted"

    def __init__(self, rule, policy_version="scripted-v1"):
        self.rule, self.policy_version, self.calls = rule, policy_version, []

    def check(self, text, bundle, context, ctx):
        self.calls.append(text)
        st, detail = self.rule(text)
        return GateDecision(status=st, policy_version=self.policy_version, checker=self.checker_id, cache_key="",
                            internal_detail=detail)


class FakeLLM:
    name = "fake"

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def generate(self, req):
        self.requests.append(req)
        status, text = self.replies.pop(0)
        return ProviderResult(role=req.role, provider="fake", endpoint="fixture", status=status, text=text,
                              model_requested=req.model, model_returned=req.model)


def web_tool(state_dir, checker, events):
    store = KnowledgeStore(state_dir, checker, BUNDLES, "smoke")
    return GatedSearch(store, FixtureSearchProvider(FIX / "web_corpus"), max_results=50), CallContext(sink=events.append)


def blob(items) -> str:
    return canonical_json([i.model_dump(mode="json") if hasattr(i, "model_dump") else i for i in items]).casefold()


def test_b11_answer_paper_versions_derived_leaks_and_code_never_reach_search_or_open(tmp_path):
    events = []
    tool, ctx = web_tool(tmp_path, FixtureMarkerChecker(), events)
    returned = []
    for q in QUERIES:
        for v in tool.search(q, ctx):
            returned += [v, tool.open(v.source_id, ctx)]
    text = blob(returned)
    for s in SECRETS + ANSWER_TEXT + BLOCKED_URLS:
        assert s.casefold() not in text, s
    titles = {v.title for v in returned}
    assert "Locating the yield ridge of the fixture reaction" not in titles
    # not vacuous: legitimate pages did come back through the same path
    assert {"A practical review of two-variable yield ridges", "Fixture ridge task statement",
            "Arrhenius temperature dependence of reaction rates", "TS-200 thermostat operating manual"} <= titles
    # the forum snippet was harmless, its fetched content leaked -> unavailable now and dropped from later searches
    forum = next(v for v in returned if v.title.startswith("Forum thread"))
    assert tool.open(forum.source_id, ctx).status == "unavailable"
    assert all(not v.title.startswith("Forum thread") for v in tool.search("forum thread batch reactor settings", ctx))
    # cost events are public-log material: no markers, no gate details
    assert not any(s.casefold() in blob(events) for s in SECRETS)
    assert "secret_marker" not in blob(events)


def test_b11_prebuild_blocks_answer_copy_and_leaky_doc_including_their_kg(tmp_path):
    ctx = CallContext(sink=lambda e: None)
    build_initial_state(tmp_path, "product", FIX / "corpus", FixtureMarkerChecker(), HashEmbedder(), BUNDLES, ctx,
                        set_scope="smoke", ontology=ONTO, extractor=FixtureKGExtractor())
    store = KnowledgeStore(tmp_path, FixtureMarkerChecker(), BUNDLES, "smoke", embedder=HashEmbedder(), ontology=ONTO)
    access = store.access(SCOPE)
    captured = []
    for q in ["ridge apex yield 95", "copied answer sheet optimum", "Locating the yield ridge",
              "temperature increases yield", "notebook optimum settings time increases yield"]:
        for v in store.retrieve(q, ctx, 20):
            captured += [v, literature_card(store, v.source_id, access).model_projection()]
    for term in ["temperature", "time", "rate constant", "loading"]:
        captured += [kg_path_card(store, p, access).model_projection() for p in kg_paths(store, term)]
    text = blob(captured)
    assert captured and "ridge apex" not in text and "answer sheet" not in text
    assert not any(s.casefold() in text for s in SECRETS)
    blocked_docs = ("doc:ridge_answer_copy", "doc:ridge_leaky_lab_notes")
    assert store.relations() and not [r for r in store.relations() for s in r.source_ids if s.startswith(blocked_docs)]


def test_b11_identity_blocks_publisher_url_variants_of_the_answer_doi():
    doi = "10.5555/fixture.ridge.answer"
    for url in [f"https://www.frontiersin.org/articles/{doi}/full", f"https://link.springer.com/content/pdf/{doi}.pdf",
                f"https://pubs.example.org/doi/{doi}/abstract", "https://doi.org/10.5555%2Ffixture.ridge.answer"]:
        assert identity_blocked(url, ["Some page"], {}, BUNDLES), url
    for url in [f"https://pubs.example.org/doi/{doi}2", f"https://pubs.example.org/doi/{doi}-comment/full",
                "https://review.fixture.example/yield-ridges"]:                  # other DOIs / a citing page
        assert not identity_blocked(url, ["Some page"], {}, BUNDLES), url


def test_b11_corpus_front_matter_is_gated_with_every_chunk_that_exposes_it(tmp_path):
    (tmp_path / "c").mkdir()
    body = "# Notes\nA long paragraph about jacket temperatures and stirring in the batch reactor.\n"
    leaky = tmp_path / "c" / "vendor.md"
    leaky.write_text("---\ntitle: Vendor notes\nurl: https://vendor.example/notes/LEAK-RIDGE-7Q2\n"
                     "material: substrate R (LEAK-RIDGE-7Q2 run at 83 degC / 37 min)\n---\n" + body, encoding="utf-8")
    clean = tmp_path / "c" / "clean.md"
    clean.write_text("---\ntitle: Clean notes\nurl: https://vendor.example/notes/clean\nmaterial: substrate R\n---\n"
                     + body, encoding="utf-8")
    store = KnowledgeStore(tmp_path / "s", FixtureMarkerChecker(), BUNDLES, "smoke")
    ctx = CallContext(sink=lambda e: None)
    assert ingest_document(store, leaky, ctx)["status"] == "block" and ingest_document(store, clean, ctx)["status"] == "ok"
    views = store.retrieve("jacket temperatures stirring batch reactor", ctx, 10)
    cards = [literature_card(store, v.source_id, store.access(SCOPE)).model_projection() for v in views]
    assert [v.source_id for v in views] == ["doc:clean#0"] and cards[0]["applicability"] == {"material": "substrate R"}
    assert "leak-ridge-7q2" not in blob(views + cards)


class FlakyMarker(FixtureMarkerChecker):
    """The fixture checker (same identity) erroring on texts where `fails` is true."""

    def __init__(self, fails):
        super().__init__()
        self.fails = fails

    def check(self, text, bundle, context, ctx):
        if self.fails(text):
            return GateDecision(status=GateStatus.error, policy_version=self.policy_version, checker=self.checker_id,
                                cache_key="", internal_detail="timeout")
        return super().check(text, bundle, context, ctx)


def test_b13_prebuild_gate_error_stores_nothing_and_a_rebuild_recovers(tmp_path):
    ctx = CallContext(sink=lambda e: None)
    # the checker fails exactly on the leaking chunk: its clean sibling chunks must not be approved meanwhile
    flaky = FlakyMarker(lambda t: "LEAK" in t)
    args = ("product", FIX / "corpus")
    with pytest.raises(KnowledgeInfraError):
        build_initial_state(tmp_path / "p", *args, flaky, HashEmbedder(), BUNDLES, ctx, set_scope="smoke")
    store = KnowledgeStore(tmp_path / "p", FixtureMarkerChecker(), BUNDLES, "smoke", embedder=HashEmbedder())
    assert store.get("doc:ridge_leaky_lab_notes") is None and store.get("doc:ridge_leaky_lab_notes#0") is None
    assert store.visible("doc:catalyst_screening_manual#0") is not None      # the other documents were finished
    store.close()
    man = build_initial_state(tmp_path / "p", *args, FixtureMarkerChecker(), HashEmbedder(), BUNDLES, ctx,
                              set_scope="smoke")
    status = {d["doc"]: d["status"] for d in man["documents"]}
    assert status["ridge_leaky_lab_notes"] == "block" and status["catalyst_screening_manual"] == "exists"
    base = ("baseline", FIX / "corpus")
    kw = {"set_scope": "smoke", "baseline_initial": FIX / "corpus" / "general_background.md"}
    flaky.fails = lambda t: True
    with pytest.raises(KnowledgeInfraError):
        build_initial_state(tmp_path / "b", *base, flaky, None, BUNDLES, ctx, **kw)
    assert load_baseline_initial(tmp_path / "b") is None
    man = build_initial_state(tmp_path / "b", *base, FixtureMarkerChecker(), None, BUNDLES, ctx, **kw)
    assert man["baseline_initial"]["available"] is True and load_baseline_initial(tmp_path / "b")


def test_b12_citing_review_public_target_and_self_observation_inference_stay_allowed(tmp_path):
    tool, ctx = web_tool(tmp_path, FixtureMarkerChecker(), [])
    review = next(v for v in tool.search("practical review two-variable yield ridges", ctx)
                  if v.title == "A practical review of two-variable yield ridges")
    assert "10.5555/fixture.ridge.answer" in review.text           # citing the answer paper is not a block reason
    opened = tool.open(review.source_id, ctx)
    assert opened.status == "available" and "Locating the yield ridge of the fixture reaction" in opened.text
    target = next(v for v in tool.search("fixture ridge task statement yield of 90 %", ctx)
                  if v.title == "Fixture ridge task statement")
    assert "yield of 90 %" in target.text
    # an inference from own observations that equals the hidden optimum is NOT deleted by string equality
    claim = "From our own runs (obs-12: temperature 83.0 degC, time 37.0 min, yield 95.0 %) the ridge top is near 83 degC."
    assert tool.store.gate(claim, {"source": "experiment:obs-12", "origin": "observation"}, ctx) == GateStatus.allow
    scope = RunScope(run_id="t", condition="product", set_id="smoke", set_rep=1, episode_id="e1", episode_order=1,
                     task_id="fixture_ridge", visit_index=1)
    obs = Observation(observation_id="obs-12", action_id="a12", scope=scope, parameters={"temperature": 83.0, "time": 37.0},
                      results={"yield": 95.0}, units={"yield": "%"}, simulator_id="fixture.ridge", simulator_version="1",
                      meets_success_criteria=True, created_at="2026-09-28T00:00:00Z")
    assert '"temperature":83.0' in observation_card(obs, EvidenceKind.current_observation, tool.store.access(SCOPE)).excerpt


def test_b13_hold_and_checker_error_return_only_the_generic_message(tmp_path):
    events = []
    tool, ctx = web_tool(tmp_path / "a", FixtureMarkerChecker(), events)
    assert all("unknown.fixture.example" not in (v.url or "") for v in tool.search("ridge", ctx))   # ambiguous -> hold
    long_text = "A long enough paragraph about thermostat ramps and jacket limits."
    assert tool.store.gate(long_text, {}, ctx) == GateStatus.hold                                  # unknown source
    assert tool.store.gate(long_text, {"source": "corpus:x.md"}, ctx) == GateStatus.allow
    leak_detail = "timeout while matching LEAK-RIDGE-7Q2 near 83 degC"
    failing = ScriptedChecker(lambda t: (GateStatus.error, leak_detail))
    tool2, ctx2 = web_tool(tmp_path / "b", failing, events)
    assert tool2.search("Arrhenius temperature dependence", ctx2) == []
    assert tool2.open("web:unknown", ctx2) == SourceView(source_id="web:unknown", title="", text=UNAVAILABLE_MESSAGE,
                                                         status="unavailable")
    failing.rule = lambda t: (GateStatus.allow, "")          # errors were not cached: a recovered checker works
    assert tool2.search("Arrhenius temperature dependence", ctx2)
    assert "timeout while matching" not in blob(events) and "leak-ridge-7q2" not in blob(events)


def test_b13_llm_gate_failures_are_unavailable_and_its_reasons_stay_internal(tmp_path):
    ctx = CallContext(sink=lambda e: None)
    b = BUNDLES[1]
    cases = [((ProviderStatus.infra_error, None), GateStatus.error), ((ProviderStatus.ok, "not json"), GateStatus.error),
             ((ProviderStatus.ok, '{"verdict": "maybe"}'), GateStatus.error),
             ((ProviderStatus.refusal, None), GateStatus.error),
             ((ProviderStatus.ok, '{"verdict": "hold"}'), GateStatus.hold),
             ((ProviderStatus.ok, '{"verdict": "allow", "reason": "background"}'), GateStatus.allow)]
    llm = FakeLLM(*[r for r, _ in cases])
    checker = LLMLeakageChecker(llm, "gate-model")
    assert [checker.check("some text", b, {"source": "x"}, ctx).status for _ in cases] == [s for _, s in cases]
    assert {r.role for r in llm.requests} == {"leakage_gate"}
    reason = "reveals the hidden optimum 83 degC / 37 min"
    events = []
    llm2 = FakeLLM(*[(ProviderStatus.ok, json.dumps({"verdict": "block", "reason": reason}))] * 200)
    tool, ctx2 = web_tool(tmp_path, LLMLeakageChecker(llm2, "gate-model"), events)
    returned = tool.search("Arrhenius temperature dependence", ctx2)
    assert returned == [] and "hidden optimum" not in blob(events)


def test_b13_context_expansion_is_gated_again_as_a_new_exposure(tmp_path):
    (tmp_path / "c").mkdir()
    doc = tmp_path / "c" / "probe.md"
    doc.write_text("---\ntitle: Expansion probe\n---\n# Probe\n## Clean\n"
                   "First clean paragraph about thermostat ramps and jacket limits.\n\n"
                   "Second clean paragraph about calibration logs and probes.\n\n## Mixed\n"
                   "Allowed paragraph about stirring speed in the batch reactor.\n\n"
                   "A HELD-SENTENCE paragraph that the gate keeps on hold.\n", encoding="utf-8")
    checker = ScriptedChecker(lambda t: (GateStatus.hold, "x") if "HELD-SENTENCE" in t else (GateStatus.allow, ""))
    store = KnowledgeStore(tmp_path / "s", checker, BUNDLES, "smoke")
    ctx = CallContext(sink=lambda e: None)
    ingest_document(store, doc, ctx, max_chunk_chars=60)
    assert store.visible("doc:probe#3") is None                        # the held chunk itself is never indexed
    tool = GatedSearch(store, FixtureSearchProvider(FIX / "web_corpus"))
    clean = tool.open("doc:probe#0", ctx, expand="section")
    assert clean.status == "available" and "Second clean paragraph" in clean.text and clean.locator["expanded"] == "section"
    for cid, level in [("doc:probe#2", "section"), ("doc:probe#0", "document"), ("doc:probe#2", "document")]:
        v = tool.open(cid, ctx, expand=level)
        assert (v.status, v.title, v.text) == ("unavailable", "", UNAVAILABLE_MESSAGE)
    assert tool.open("doc:probe#2", ctx).status == "available"         # the unexpanded chunk stays usable


def test_google_cse_adapter_contract_with_fake_transport_only(tmp_path):
    calls = []

    def transport(url, timeout):
        calls.append(url)
        if "customsearch" in url:
            return json.dumps({"items": [{"link": "https://x.example/a", "title": "Locating the yield ridge",
                                          "snippet": "s", "pagemap": {"metatags": [
                                              {"citation_doi": "https://doi.org/10.5555/FIXTURE.RIDGE.ANSWER"}]}},
                                         {"link": "https://y.example/b", "title": "Background", "snippet":
                                          "General thermostat background text for reactors."}]}).encode()
        return (b"<html><head><title>T</title><meta name='citation_doi' content='10.1/x'></head>"
                b"<body><script>bad()</script><p>Hello body</p></body></html>")

    env = {"LABGENE_SEARCH_API_KEY": "k-secret", "LABGENE_SEARCH_ENGINE_ID": "cx1"}
    p = GoogleCSESearchProvider(transport=transport, env=env)
    hits = p.search("ridge", 3)
    assert "q=ridge" in calls[0] and "num=3" in calls[0] and "cx=cx1" in calls[0]
    assert hits[0].metadata["doi"].endswith("FIXTURE.RIDGE.ANSWER") and hits[1].url == "https://y.example/b"
    page = p.fetch("https://y.example/b")
    assert (page.title, page.content, page.metadata["doi"]) == ("T", "Hello body", "10.1/x")
    # the gated proxy blocks the answer paper by the provider-reported DOI before any content check
    tool = GatedSearch(KnowledgeStore(tmp_path, FixtureMarkerChecker(), BUNDLES, "smoke"), p)
    assert [v.url for v in tool.search("ridge", CallContext(sink=lambda e: None))] == ["https://y.example/b"]
    with pytest.raises(ValueError):
        GoogleCSESearchProvider(transport=transport, env={})

    def broken(url, timeout):
        raise OSError(f"connection reset for {url}")

    with pytest.raises(KnowledgeInfraError) as e:
        GoogleCSESearchProvider(transport=broken, env=env).search("ridge", 3)
    assert "k-secret" not in str(e.value) and e.value.__cause__ is None
