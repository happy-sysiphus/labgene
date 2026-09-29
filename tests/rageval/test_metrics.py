from rag_eval.metrics import (answer_relevancy, citation_metrics, claim_metrics, context_precision, cosine,
                              issue_counts)

CLAIMS = [{"text": "a", "type": "fact"}, {"text": "b", "type": "fact"}, {"text": "c", "type": "inference"}]


def test_claim_metrics_score_facts_only_and_never_default():
    v = [{"verdict": "supported", "support": [0]}, {"verdict": "contradicted", "support": []}]
    m = claim_metrics(CLAIMS, v)
    assert (m["n_facts"], m["n_supported"], m["faithfulness"], m["contradiction"]) == (2, 1, 0.5, 0.5)
    assert abs(m["fact_ratio"] - 2 / 3) < 1e-12
    only_inference = claim_metrics([{"text": "c", "type": "inference"}], None)
    assert only_inference["fact_ratio"] == 0.0 and only_inference["faithfulness"] is None
    assert claim_metrics(None, None)["n_claims"] is None
    assert claim_metrics(CLAIMS, None)["faithfulness"] is None           # judge unavailable


def test_citation_metrics_use_answer_level_citations():
    cites = [frozenset({"a"}), frozenset({"b", "c"}), frozenset()]
    v = [{"verdict": "supported", "support": [0]}, {"verdict": "supported", "support": [1]},
         {"verdict": "not_found", "support": []}]
    assert citation_metrics(v, cites, {"c", "z"}) == {"citation_recall": 1 / 3, "citation_precision": 0.5}
    assert citation_metrics(v, cites, set())["citation_precision"] is None      # nothing cited
    assert citation_metrics(None, cites, {"c"}) == {"citation_recall": None, "citation_precision": None}


def test_context_precision_is_rank_weighted():
    m = context_precision([True, False, True])
    assert abs(m["context_precision"] - (1 + 2 / 3) / 2) < 1e-12
    assert abs(m["context_precision_plain"] - 2 / 3) < 1e-12
    assert context_precision([False, False]) == {"context_precision": 0.0, "context_precision_plain": 0.0}
    assert context_precision(None)["context_precision"] is None and context_precision([])["context_precision"] is None


def test_answer_relevancy_and_cosine():
    assert answer_relevancy([1, 0], [[1, 0], [0, 1]], False) == 0.5
    assert answer_relevancy([1, 0], [[1, 0]], True) == 0.0
    assert cosine([0, 0], [1, 0]) == 0.0


def test_issue_counts_follow_the_validator_messages():
    issues = ["answer is empty", "cited source doc:x#1 was not delivered in this consultation",
              "cited source doc:y#2 is no longer approved", "cited observation obs:e:a9 does not exist",
              "yield=80 attributed to observation obs:e:a1 does not match the recorded value 79.0",
              "candidate 0: temperature out of range", "something new"]
    assert issue_counts(issues) == {"empty_field": 1, "uncited_delivery": 1, "no_longer_approved": 1,
                                    "missing_observation": 1, "value_mismatch": 1, "invalid_candidate": 1, "other": 1}
