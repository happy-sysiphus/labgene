"""T08.5 retrieval / answer development validation tools over the fixture corpus (development_only: fixture embedder,
fixture checker, token-overlap judge). B15 (transcriptions), B16 (mechanical vs semantic citation checks).
Regression tests test_finding<N>_* cover the independent review of the evaluation module."""
import json
from pathlib import Path

import pytest
import yaml

from labgene.contracts import (AdvisorResponse, Observation, PrivateTaskAssets, ProviderResult, ProviderStatus,
                               RunScope, payload_hash)
from labgene.config import load_profile, load_public_task
from labgene.costs import CallContext
from labgene.evaluation.retrieval_validation import (AdoptionRule, DevSet, Transcription, adopt, check_transcription,
                                                     evaluate_answer, load_dev_set, run_retrieval_validation,
                                                     score_query)
from labgene.knowledge.build import build_initial_state
from labgene.knowledge.cards import SupportVerdict, TokenOverlapJudge
from labgene.knowledge.gate import FixtureMarkerChecker
from labgene.knowledge.retrieval import HashEmbedder, LLMReranker
from labgene.knowledge.store import KnowledgeStore
from labgene.providers.fixture import FixtureProvider

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures"
BUNDLES = [PrivateTaskAssets.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))).answer_bundle
           for p in sorted((FIX / "private").glob("*.yaml"))]
DEV = load_dev_set(ROOT / "configs" / "retrieval_dev" / "fixture_dev.yaml")
CTX = CallContext(sink=lambda e: None)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    d = tmp_path_factory.mktemp("product")
    build_initial_state(d, "product", FIX / "corpus", FixtureMarkerChecker(), HashEmbedder(), BUNDLES, CTX,
                        set_scope="smoke")
    s = KnowledgeStore(d, FixtureMarkerChecker(), BUNDLES, "smoke", embedder=HashEmbedder())
    yield s
    s.close()


class ReversingLLM:
    """Fake reranker provider: returns the candidates in reverse order."""
    name = "fake"

    def generate(self, req):
        ids = [c["id"] for c in json.loads(req.input[0]["text"])["candidates"]]
        return ProviderResult(role=req.role, provider="fake", endpoint="fixture", status=ProviderStatus.ok,
                              text=json.dumps({"order": ids[::-1]}), model_requested=req.model,
                              model_returned=req.model)


def test_t08_5_fixture_dev_set_metrics_misses_and_provisional_adoption(store, tmp_path):
    lock = tmp_path / "input.lock.json"
    r = run_retrieval_validation(store, DEV, CTX, input_lock=lock)
    assert r["development_only"] and r["embedder"]["development_only"] and r["unresolved_gold"] == {}
    for name in ("bm25_only", "dense_only", "rrf"):
        c = r["configs"][name]
        assert len(c["queries"]) == len(DEV.queries)
        assert c["recall_at_k"] == round(sum(q["recall"] for q in c["queries"]) / len(DEV.queries), 4)
        assert c["mrr"] == round(sum(q["rr"] for q in c["queries"]) / len(DEV.queries), 4)
        for q in c["queries"]:
            assert len(q["retrieved"]) <= DEV.k and bool(q["misses"]) == (q["recall"] < 1)
    assert r["configs"]["rrf_reranker"] == {"not_run": "no reranker supplied"}
    assert r["adoption"] == {"chosen": adopt({k: v for k, v in r["configs"].items() if "not_run" not in v},
                                             DEV.adoption_rule)["chosen"],
                             "complete": False, "not_run": ["rrf_reranker"], "provisional": True}
    changed = DEV.model_copy(update={"adoption_rule": DEV.adoption_rule.model_copy(update={"primary": "mrr"})})
    with pytest.raises(ValueError, match="new version"):
        run_retrieval_validation(store, changed, CTX, input_lock=lock)


def test_finding5_the_whole_dev_set_is_pinned_and_recorded(store, tmp_path):
    lock = tmp_path / "input.lock.json"
    r = run_retrieval_validation(store, DEV, CTX, input_lock=lock)
    assert r["input_hash"] == payload_hash(DEV.model_dump(mode="json", exclude={"version", "status"}))
    post_hoc = [{"k": 8}, {"configs": ["dense_only", "rrf"]},
                {"queries": [q for q in DEV.queries if q.query_id not in ("q01", "q03")]},
                {"transcriptions": DEV.transcriptions[:1]}]
    for update in post_hoc:                                   # same version + lock: every input change is refused
        with pytest.raises(ValueError, match="new version"):
            run_retrieval_validation(store, DEV.model_copy(update=update), CTX, input_lock=lock)
    same = run_retrieval_validation(store, DEV.model_copy(update={"status": "reviewed"}), CTX, input_lock=lock)
    assert same["input_hash"] == r["input_hash"]              # a review-status change is not an input change
    v2 = DEV.model_copy(update={"k": 8, "version": "retrieval-dev-fixture-v2"})
    assert run_retrieval_validation(store, v2, CTX, input_lock=lock)["input_hash"] != r["input_hash"]


class LiveEmbedder(HashEmbedder):
    development_only = False


class LiveChecker(FixtureMarkerChecker):
    development_only = False


def test_finding11_and_12_development_label_and_provisional_adoption_come_from_the_components(tmp_path):
    reviewed = DEV.model_copy(update={"status": "reviewed", "configs": ["bm25_only", "dense_only", "rrf"]})

    def open_store(d, checker):
        build_initial_state(d, "product", FIX / "corpus", checker, LiveEmbedder(), BUNDLES, CTX, set_scope="smoke")
        return KnowledgeStore(d, checker, BUNDLES, "smoke", embedder=LiveEmbedder())

    s = open_store(tmp_path / "fixture_gate", FixtureMarkerChecker())    # live embedder, fixture gate
    try:
        r = run_retrieval_validation(s, reviewed, CTX)
        assert r["development_only"] and r["checker"] == {"id": "fixture-marker-v1", "policy_version":
                                                          "fixture-policy-v1", "development_only": True}
        assert r["adoption"]["complete"] and r["adoption"]["provisional"]
    finally:
        s.close()
    s = open_store(tmp_path / "live", LiveChecker())
    try:
        with pytest.raises(ValueError, match="input_lock"):   # a non-development run must pin its input
            run_retrieval_validation(s, reviewed, CTX)
        r = run_retrieval_validation(s, reviewed, CTX, input_lock=tmp_path / "l.json")
        assert not r["development_only"] and not r["adoption"]["provisional"]
        draft = run_retrieval_validation(s, reviewed.model_copy(update={"status": "draft_unreviewed",
                                                                        "version": "d1"}), CTX,
                                         input_lock=tmp_path / "l.json")
        assert draft["adoption"]["provisional"]               # a draft set never yields a final adoption
        fixture_rr = LLMReranker(FixtureProvider(), "fixture-reranker")
        with_rr = run_retrieval_validation(s, reviewed.model_copy(update={"configs": ["rrf", "rrf_reranker"],
                                                                          "version": "rr1"}), CTX,
                                           reranker=fixture_rr, input_lock=tmp_path / "l.json")
        assert with_rr["development_only"] and with_rr["reranker"]["development_only"]
    finally:
        s.close()


def test_t08_5_unresolvable_gold_counts_as_a_miss_never_dropped(store):
    dev = DevSet.model_validate({**DEV.model_dump(), "configs": ["bm25_only"], "transcriptions": [], "queries": [
        {"query_id": "x1", "query": "ridge apex yield",
         "relevant": [{"doc": "ridge_answer_copy"},
                      {"doc": "reaction_engineering_review", "section": "Ridge-shaped responses"}]}]})
    r = run_retrieval_validation(store, dev, CTX)
    q = r["configs"]["bm25_only"]["queries"][0]
    assert q["recall"] <= 0.5 and {"doc": "ridge_answer_copy"} in q["misses"]
    assert r["unresolved_gold"] == {"x1": [{"doc": "ridge_answer_copy"}]}     # blocked doc: no approved chunk


def test_t08_5_recall_mrr_and_the_predeclared_adoption_rule():
    s = score_query(["c", "a", "x"], [{"a"}, {"b", "z"}])
    assert (s["recall"], s["rr"], s["missed"]) == (0.5, 0.5, [1])
    assert score_query([], [{"a"}]) == {"recall": 0.0, "rr": 0.0, "missed": [0]}
    rule = AdoptionRule(primary="recall_at_k", tie_break=["mrr", "cost_rank"],
                        cost_rank={"bm25_only": 0, "dense_only": 1, "rrf": 2, "rrf_reranker": 3})
    tie = {"rrf": {"recall_at_k": 0.9, "mrr": 0.8}, "bm25_only": {"recall_at_k": 0.9, "mrr": 0.8},
           "dense_only": {"recall_at_k": 0.9, "mrr": 0.7}}
    assert adopt(tie, rule) == {"chosen": "bm25_only", "complete": True, "not_run": []}   # tie -> cheaper
    tie["rrf"]["recall_at_k"] = 0.95
    assert adopt(tie, rule)["chosen"] == "rrf"                                           # primary first


def test_t08_5_rrf_reranker_runs_only_with_an_explicit_reranker(store):
    dev = DevSet.model_validate({**DEV.model_dump(), "configs": ["rrf", "rrf_reranker"], "transcriptions": []})
    r = run_retrieval_validation(store, dev, CTX, reranker=LLMReranker(ReversingLLM(), "fake-reranker"))
    plain, reranked = r["configs"]["rrf"]["queries"], r["configs"]["rrf_reranker"]["queries"]
    assert r["adoption"]["complete"] and store.reranker is None
    assert any(a["retrieved"] != b["retrieved"] for a, b in zip(plain, reranked))
    store.reranker = LLMReranker(ReversingLLM(), "fake-reranker")
    try:
        with pytest.raises(ValueError, match="without a reranker"):
            run_retrieval_validation(store, dev, CTX)
    finally:
        store.reranker = None


def test_b15_transcription_checks_keep_numbers_units_ranges_tables_and_ocr_flags(store):
    r = run_retrieval_validation(store, DevSet.model_validate({**DEV.model_dump(), "configs": ["bm25_only"]}), CTX)
    assert [t["item_id"] for t in r["transcriptions"] if not t["ok"]] == []
    assert len(next(t for t in r["transcriptions"] if t["item_id"] == "t02")["chunks"]) == 3   # split table
    ref = {"doc": "catalyst_screening_manual", "section": "Thermostat calibration table"}
    bad = check_transcription(store, Transcription(item_id="n1", ref=ref, numbers=["121.6"], units=["K"],
                                                   lines_in_every_chunk=["| setpoint | measured |"]))
    assert not bad["ok"] and len(bad["issues"]) == 2 + 3                # number, unit, and the line in each chunk
    ocr = check_transcription(store, Transcription(item_id="n2", ref={"doc": "catalyst_screening_manual",
                                                                      "section": "Scanned service note"}, uncertain=False))
    assert ocr["issues"] == ["transcription flagged uncertain"]
    assert not check_transcription(store, Transcription(item_id="n3", ref={"doc": "ridge_answer_copy"}))["ok"]


class UnavailableJudge:
    development_only = True

    def judge(self, claim, cited_text, ctx):
        return SupportVerdict(supported=None, judge="down", development_only=True, detail="unavailable")


def test_b16_answer_dev_reports_mechanical_and_semantic_separately():
    task = load_public_task(load_profile(ROOT / "configs" / "offline.yaml"), "fixture_ridge")
    scope = RunScope(run_id="d", condition="product", set_id="dev", set_rep=1, episode_id="e1", episode_order=1,
                     task_id=task.task_id, visit_index=1)
    obs = [Observation(observation_id="obs:e1:a001", action_id="e1:a001", scope=scope,
                       parameters={"temperature": 80.0, "time": 35.0}, results={"yield": 93.9}, units={"yield": "%"},
                       simulator_id=task.simulator_id, simulator_version=task.simulator_version,
                       meets_success_criteria=True, created_at="t")]
    delivered = {"s1": "Two-variable yield surfaces often form a correlated ridge. Sequential one-factor searches "
                       "tend to stall on such ridges."}
    resp = AdvisorResponse(answer="Move both variables together.", cited_source_ids=["s1"],
                           cited_observation_ids=["obs:e1:a001"], reasoning="obs:e1:a001 gave yield 93.9.",
                           limitations="Fixture text only.")
    claims = [("Sequential one-factor searches tend to stall on ridges", "s1"),
              ("Catalyst C at 1.2 mol% maximizes TON", "s1")]
    r = evaluate_answer(resp, delivered, obs, task, TokenOverlapJudge(), claims=claims)
    assert r["mechanical"] == {"ok": True, "issues": []}               # ids and numbers fine ...
    assert r["semantic"]["ok"] is False and r["semantic"]["development_only"]   # ... but a claim is unsupported
    assert [j["supported"] for j in r["semantic"]["judgements"]] == [True, False]
    wrong = resp.model_copy(update={"reasoning": "obs:e1:a001 gave yield 91.0.", "cited_source_ids": ["s9"]})
    r = evaluate_answer(wrong, delivered, obs, task, TokenOverlapJudge())
    assert not r["mechanical"]["ok"] and len(r["mechanical"]["issues"]) == 2   # undelivered source + misquote
    assert r["semantic"]["ok"] is None                                  # the cited source was never delivered
    assert evaluate_answer(resp, delivered, obs, task, UnavailableJudge())["semantic"]["ok"] is None
