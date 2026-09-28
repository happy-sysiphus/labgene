"""T05 evidence cards and response checks: B16 (mechanical validity is not semantic support), descriptive
calculations under the multi-metric success rule, study-level independence, access never projected."""
from pathlib import Path

import pytest
import yaml

from labgene.config import load_profile, load_public_task
from labgene.contracts import (AdvisorResponse, CandidateSuggestion, CardAccess, EvidenceKind, GateStatus, MemoryScope,
                               Observation, PrivateTaskAssets, ProviderResult, ProviderStatus, RunScope, canonical_json)
from labgene.costs import CallContext
from labgene.knowledge.build import build_initial_state
from labgene.knowledge.cards import (LLMSupportJudge, TokenOverlapJudge, calculation_card, check_citation_support,
                                     independent_count, kg_path_card, literature_card, observation_cards,
                                     validate_advisor_response)
from labgene.knowledge.gate import FixtureMarkerChecker, study_key
from labgene.knowledge.kg import FixtureKGExtractor, Ontology
from labgene.knowledge.retrieval import HashEmbedder
from labgene.knowledge.store import KnowledgeStore
from labgene.providers.base import ModelChangedError

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures"
PROFILE = load_profile(ROOT / "configs" / "offline.yaml")
RIDGE, CAT = load_public_task(PROFILE, "fixture_ridge"), load_public_task(PROFILE, "fixture_catalyst")
BUNDLES = [PrivateTaskAssets.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))).answer_bundle
           for p in sorted((FIX / "private").glob("*.yaml"))]
SCOPE = MemoryScope(run_id="t", condition="product", set_id="smoke", set_rep=1)
CTX = CallContext(sink=lambda e: None)
ONTO = Ontology.load(FIX / "ontology" / "profile.yaml")


def obs(oid, task, params, results, episode="e1"):
    scope = RunScope(run_id="t", condition="product", set_id="smoke", set_rep=1, episode_id=episode, episode_order=1,
                     task_id=task.task_id, visit_index=1)
    return Observation(observation_id=oid, action_id=f"a-{oid}", scope=scope, parameters=params, results=results,
                       units={m.name: m.unit for m in task.metrics}, simulator_id=task.simulator_id,
                       simulator_version=task.simulator_version, meets_success_criteria=task.is_success(results),
                       created_at="2026-09-28T00:00:00Z")


def product_store(state_dir):
    build_initial_state(state_dir, "product", FIX / "corpus", FixtureMarkerChecker(), HashEmbedder(), BUNDLES, CTX,
                        set_scope="smoke")
    return KnowledgeStore(state_dir, FixtureMarkerChecker(), BUNDLES, "smoke", embedder=HashEmbedder())


def test_b16_valid_but_non_supporting_citation_passes_mechanics_and_fails_semantic_check(tmp_path):
    store = product_store(tmp_path)
    delivered = [v.source_id for v in store.retrieve("thermostat calibration table setpoint deviation", CTX, 5)]
    cited = delivered[0]
    o1 = obs("obs-1", RIDGE, {"temperature": 80.0, "time": 30.0}, {"yield": 89.64})
    resp = AdvisorResponse(answer="In obs-1 the yield was 89.6 %. Raising temperature from 60 to 80 degC should "
                                  "increase the yield further.", cited_source_ids=[cited],
                           cited_observation_ids=["obs-1"], reasoning="Rate rises with temperature.",
                           limitations="Single observation; the cited table is about calibration.",
                           candidates=[CandidateSuggestion(parameters={"temperature": 85.0, "time": 35.0})])
    checked, issues = validate_advisor_response(resp, delivered, [o1], RIDGE, store=store)
    assert issues == [] and checked.validation_issues == [] and checked.candidates[0].within_public_constraints is True
    claim = "Raising temperature from 60 to 80 degC should increase the yield further."
    verdict = check_citation_support(claim, store.view(cited).text, TokenOverlapJudge())
    assert verdict.supported is False and verdict.development_only
    # the same judge accepts a passage that does state the claim (the check is not vacuous)
    good = store.view("doc:ridge_kinetics_notes#2").text
    assert check_citation_support("A higher rate constant increases yield at fixed reaction time.", good,
                                  TokenOverlapJudge()).supported is True
    # a delivered source blocked later in the same consultation is no longer a valid citation
    store.invalidate(store.ancestors(cited)[0])
    assert validate_advisor_response(resp, delivered, [o1], RIDGE, store=store)[1] == \
        [f"cited source {cited} is no longer approved"]


def test_b16_mechanical_checks_catch_undelivered_ids_missing_observations_wrong_numbers_and_ranges():
    o1 = obs("obs-1", RIDGE, {"temperature": 80.0, "time": 30.0}, {"yield": 89.64})
    resp = AdvisorResponse(answer="In obs-1 the yield was 91.0 % at temperature 80.", cited_source_ids=["doc:x#9"],
                           cited_observation_ids=["obs-404"],
                           candidates=[CandidateSuggestion(parameters={"temperature": 500, "time": 30}),
                                       CandidateSuggestion(parameters={"temperature": 70})])
    checked, issues = validate_advisor_response(resp, ["doc:y#1"], [o1], RIDGE)
    text = "\n".join(issues)
    assert len(issues) == 7 and "doc:x#9" in text and "obs-404" in text and "yield=91.0" in text
    assert "outside the allowed range" in text and "missing required parameter" in text
    assert "reasoning is empty" in issues and "limitations is empty" in issues        # §13.4 required fields
    assert [c.within_public_constraints for c in checked.candidates] == [False, False]
    assert checked.validation_issues == issues
    assert validate_advisor_response(AdvisorResponse(answer=" "), [], [], RIDGE)[1] == \
        ["answer is empty", "reasoning is empty", "limitations is empty"]
    # a target value is not attributed to the observation named in the sentence; a wrong observed value still is
    rl = {"reasoning": "r", "limitations": "l"}
    for answer in ["obs-1 fell short of the target yield of 90.0 %.",                    # value named as target
                   "obs-1 ran at temperature 80, whereas a yield of 95 % is expected nearer 85 degC."]:  # other clause
        assert validate_advisor_response(AdvisorResponse(answer=answer, **rl), [], [o1], RIDGE)[1] == [], answer
    wrong = AdvisorResponse(answer="In obs-1, below the target yield of 90 %, the yield was 88.0 %.", **rl)
    assert [i for i in validate_advisor_response(wrong, [], [o1], RIDGE)[1] if "yield=" in i] == \
        ["yield=88.0 attributed to observation obs-1 does not match the recorded value 89.64"]


def test_b16_llm_support_judge_verdicts_and_unavailability_are_never_support():
    class FakeLLM:
        name = "fake"

        def __init__(self, *replies):
            self.replies = list(replies)

        def generate(self, req):
            st, text = self.replies.pop(0)
            return ProviderResult(role=req.role, provider="fake", endpoint="fixture", status=st, text=text,
                                  model_requested=req.model, model_returned=self.model or req.model)

        model = None

    judge = LLMSupportJudge(FakeLLM((ProviderStatus.ok, '{"verdict": "not_supported"}'),
                                    (ProviderStatus.ok, '{"verdict": "supported"}'),
                                    (ProviderStatus.infra_error, None), (ProviderStatus.ok, "garbage")), "judge-model")
    assert [check_citation_support("c", "t", judge).supported for _ in range(4)] == [False, True, None, None]
    swapped = FakeLLM((ProviderStatus.ok, '{"verdict": "supported"}'))
    swapped.model = "another-model"            # an internal role's model change stops the run (spec §10.1)
    with pytest.raises(ModelChangedError):
        check_citation_support("c", "t", LLMSupportJudge(swapped, "judge-model"))


def test_calc_card_best_so_far_follows_the_multi_metric_success_rule():
    p = {"catalyst": "C", "loading": 1.0, "temperature": 80.0, "residence_time": 300.0}
    o1 = obs("o1", CAT, p, {"yield": 85.0, "ton": 40.0})                               # 1 met, ton short 0.283
    o2 = obs("o2", CAT, {**p, "loading": 1.2}, {"yield": 70.0, "ton": 70.0})           # 1 met, yield short 0.1
    o3 = obs("o3", CAT, {**p, "catalyst": "D"}, {"yield": 60.0, "ton": 50.0})          # 0 met
    other = obs("r1", RIDGE, {"temperature": 80.0, "time": 30.0}, {"yield": 99.0})     # other task: ignored
    access = CardAccess(condition="product", set_id="smoke", set_rep=1, gate_status=GateStatus.allow,
                        gate_policy_version="fixture-policy-v1", answer_bundle_version="fixture-answers-catalyst@1")
    card = calculation_card(CAT, [o1, o2, o3, other], access)
    r = card.derivation["result"]
    assert card.kind == EvidenceKind.code_calculation and card.derivation["method"] == "descriptive_statistics"
    assert set(r) == {"count", "best_so_far", "distance_to_targets", "recent", "similar_conditions"}
    assert r["count"] == 3 and r["best_so_far"]["observation_id"] == "o2" and r["best_so_far"]["criteria_satisfied"] == 1
    assert r["distance_to_targets"]["yield"]["satisfied"] is False and r["distance_to_targets"]["ton"]["satisfied"] is True
    assert abs(r["distance_to_targets"]["yield"]["normalized_shortfall"] - 0.1) < 1e-9
    assert [s["observation_id"] for s in r["similar_conditions"]] == ["o1", "o3"]     # nearest to the best, best excluded
    assert [x["observation_id"] for x in r["recent"]] == ["o1", "o2", "o3"]
    assert "access" not in card.model_projection() and "fixture-answers" not in canonical_json(card.model_projection())


def test_observation_cards_keep_exact_values_and_duplicate_study_copies_count_once(tmp_path):
    store = product_store(tmp_path)
    access = store.access(SCOPE)
    cur = obs("obs-1", RIDGE, {"temperature": 80.0, "time": 30.0}, {"yield": 89.64})
    past = obs("obs-0", RIDGE, {"temperature": 60.0, "time": 20.0}, {"yield": 41.2}, episode="e0")
    cards = observation_cards([cur], [past], access)
    assert [c.kind for c in cards] == [EvidenceKind.current_observation, EvidenceKind.past_observation]
    assert '"yield":89.64' in cards[0].excerpt and cards[1].access.episode_id == "e0"
    notes = [literature_card(store, f"doc:ridge_kinetics_notes#{i}", access) for i in (0, 3)]
    summary, _ = store.register_derived("summary", "Rate constant rises with temperature for substrate R.",
                                        ["doc:ridge_kinetics_notes#0"], "internal_summarizer:test", CTX)
    s_card = literature_card(store, summary, access)
    assert s_card.excerpt is None and s_card.summary and s_card.derivation["method"] == "generated_summary"
    review = literature_card(store, "doc:reaction_engineering_review#0", access)
    assert independent_count(notes + [s_card]) == 1 and independent_count(notes + [s_card, review]) == 2


def test_b16_relation_from_a_chunk_and_alternate_versions_of_one_study_are_not_independent(tmp_path):
    build_initial_state(tmp_path, "product", FIX / "corpus", FixtureMarkerChecker(), HashEmbedder(), BUNDLES, CTX,
                        set_scope="smoke", ontology=ONTO, extractor=FixtureKGExtractor())
    store = KnowledgeStore(tmp_path, FixtureMarkerChecker(), BUNDLES, "smoke", embedder=HashEmbedder(), ontology=ONTO)
    access = store.access(SCOPE)
    rel = next(r for r in store.relations() if len(r.source_ids) == 1)
    lit = literature_card(store, rel.source_ids[0], access)
    review = literature_card(store, "doc:reaction_engineering_review#0", access)
    assert independent_count([lit, kg_path_card(store, [rel], access)]) == 1
    assert independent_count([lit, kg_path_card(store, [rel], access), review]) == 2

    def card(meta):
        return lit.model_copy(update={"locator": {"study": study_key(meta, "x")}, "content_hash": str(meta)})

    published = card({"doi": "10.5555/some.study", "arxiv_id": "2601.12345", "title": "Ridge study"})
    preprint = card({"arxiv_id": "2601.12345v2", "title": "Ridge study (preprint)"})
    renamed = card({"url": "https://arxiv.org/abs/2601.12345", "title": "An early draft"})
    other = card({"doi": "10.5555/other.study", "title": "Another study"})
    assert independent_count([published, preprint, renamed]) == 1 and independent_count([preprint, other]) == 2
