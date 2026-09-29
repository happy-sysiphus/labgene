import json

from rag_eval.report import JUDGED, cluster_ci, split_sim, summarize, write_reports


def item(**kw):
    base = {"action_id": "x", "condition": "baseline", "task_id": "t1", "episode_id": "b1", "ok": True,
            "sampled": True, "judged": True, "issues": {"value_mismatch": 0, "other": 0}, "has_issue": 0.0,
            "sim_status": "ok", "sim_success": 0.0, "sim_score": 0.5, "sim_improved": 1.0,
            **dict.fromkeys(JUDGED), "n_claims": 3, "n_facts": 2, "n_supported": 1, "n_contradicted": 0,
            "n_cited": 1, "reconstruction_match": True, "unavailable": []}
    return {**base, **kw}


NOT_JUDGED = dict(judged=False, sampled=False, n_claims=None, n_facts=None, n_supported=None, n_contradicted=None,
                  n_cited=None, reconstruction_match=None, unavailable=None)


def test_split_sim_keeps_raw_values_private():
    public, raw = split_sim("a1", {"status": "ok", "success": True, "score": 1.02, "improved": True,
                                   "results": {"ton": 123.4}, "best_prior": 0.5})
    assert public == {"sim_status": "ok", "sim_success": 1.0, "sim_score": 1.02, "sim_improved": 1.0}
    assert raw == {"action_id": "a1", "results": {"ton": 123.4}, "best_prior": 0.5, "score": 1.02}
    assert split_sim("a2", None) == ({"sim_status": None, "sim_success": None, "sim_score": None,
                                      "sim_improved": None}, None)


def test_cluster_ci_is_reproducible_and_needs_both_conditions():
    a = {"b1": [0.1, 0.3], "b2": [0.2], "b3": [0.4]}
    b = {"p1": [0.9], "p2": [0.7, 0.8], "p3": [0.6]}
    ci = cluster_ci(a, b, 400, 7)
    assert ci == cluster_ci(a, b, 400, 7) and 0 < ci[0] <= ci[1] < 1
    assert cluster_ci({}, b, 400, 7) is None and cluster_ci({"b1": []}, b, 400, 7) is None


def test_summary_counts_means_and_primary():
    items = [item(action_id="1", faithfulness=0.5),
             item(action_id="2", episode_id="b2", n_claims=None, n_facts=None, n_supported=None, n_cited=None,
                  reconstruction_match=None, unavailable=["claims"], sim_score=1.0),
             item(action_id="3", **NOT_JUDGED, sim_score=0.0),
             item(action_id="4", ok=False, **NOT_JUDGED, sim_status=None, sim_success=None, sim_score=None,
                  sim_improved=None, has_issue=None),
             item(action_id="5", condition="product", episode_id="p1", faithfulness=1.0, n_supported=2, n_cited=0,
                  reconstruction_match=False, sim_score=1.0)]
    s = summarize(items, 200, 1)
    b, p = s["conditions"]["baseline"], s["conditions"]["product"]
    assert (b["consults"], b["partial"], b["judged"], b["judge_unavailable"]) == (4, 1, 2, 1)
    assert b["faithfulness_micro"] == 0.5 and p["faithfulness_micro"] == 1.0
    assert (p["no_citation"], p["reconstruction_mismatch"]) == (1, 1)
    assert s["primary"] == "faithfulness" and s["differences"]["faithfulness"]["diff"] == 0.5
    assert s["differences"]["sim_score"]["baseline"] == 0.5      # items 1-3 (the partial item 4 is left out)
    assert s["per_task"]["t1"]["product"]["judged"] == 1


def test_reports_carry_no_private_fields(tmp_path):
    items = [item(action_id="1", faithfulness=0.5), item(action_id="2", condition="product", episode_id="p1",
                                                         faithfulness=1.0)]
    manifest = {"target_run": "r", "target_complete": False, "prompt_version": "rag-judge-v1", "sample_size": 2,
                "sample_per_task": 10, "bootstrap_draws": 200,
                "judge": {"provider": "fixture", "model": "m", "reasoning_effort": None}}
    paths = write_reports(tmp_path, manifest, items, summarize(items, 200, 1), {"judge": {}, "support": {}})
    rows = [json.loads(line) for line in (tmp_path / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("results" not in r and "best_prior" not in r for r in rows)
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "faithfulness (primary)" in md and "NO (partial run)" in md and "FIXTURE JUDGE" in md
    assert "## Limitations" in md and set(paths) == {"report", "report_json", "items"}
