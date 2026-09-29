from labgene.app import Run
from labgene.contracts import (CandidateSuggestion, HiddenCriterion, Observation, PublicTask, RunScope,
                               SuccessCriterion)
from rag_eval.simscore import SimScorer, score

TASK = PublicTask(
    task_id="t", version="1", title="t", problem="p",
    parameters=[{"kind": "continuous", "name": "x", "unit": "", "min": 0, "max": 10}],
    metrics=[{"name": "yield", "unit": "%", "direction": "maximize"}, {"name": "ton", "unit": "1", "direction": "maximize"}],
    success=[SuccessCriterion(metric="yield", direction="maximize", target=80, tolerance=0)],
    hidden_success=[HiddenCriterion(metric="ton", direction="maximize", rule="TON near the best")],
    simulator_id="s", simulator_version="1")
HIDDEN = [SuccessCriterion(metric="ton", direction="maximize", target=100, tolerance=2)]       # threshold 98


def test_score_is_ton_over_the_hidden_threshold_once_the_yield_holds():
    assert score(TASK, HIDDEN, {"yield": 79.9, "ton": 500}) == 0.0
    assert score(TASK, HIDDEN, {"yield": 80, "ton": 49}) == 0.5
    assert score(TASK, HIDDEN, {"yield": 90, "ton": 98}) == 1.0
    assert TASK.is_success({"yield": 90, "ton": 98}, HIDDEN) and not TASK.is_success({"yield": 90, "ton": 97.9}, HIDDEN)
    assert score(TASK, [], {"yield": 90, "ton": 98}) == 0.0          # declared hidden criterion without its threshold


def test_task_without_hidden_criteria_scores_its_success():
    t = TASK.model_copy(update={"hidden_success": []})
    assert score(t, [], {"yield": 85}) == 1.0 and score(t, [], {"yield": 70}) == 0.0


def test_first_candidate_statuses_with_the_offline_simulator(offline_run):
    run = Run.load(offline_run)
    task = run.tasks["fixture_catalyst"]
    s = SimScorer(run)
    try:
        assert s.top1(task, [], [], [])["status"] == "no_candidate"
        bad = CandidateSuggestion(parameters={"catalyst": "Z", "loading": 1, "temperature": 50, "residence_time": 100})
        r = s.top1(task, [], [bad], [])
        assert (r["status"], r["score"], r["success"]) == ("invalid_candidate", 0.0, False)
        good = CandidateSuggestion(parameters={"catalyst": "C", "loading": 1.2, "temperature": 90.0,
                                               "residence_time": 540.0})
        r = s.top1(task, [], [good], [])
        assert r["status"] == "ok" and r["success"] and r["score"] == 1.0 and r["improved"]
        scope = RunScope(run_id="r", condition="product", set_id="s", set_rep=1, episode_id="e", episode_order=1,
                         task_id="fixture_catalyst", visit_index=1)
        prior = Observation(observation_id="obs:e:a001", action_id="e:a001", scope=scope, parameters={},
                            results={"yield": 99.0, "ton": 99.0}, units={}, simulator_id="fixture.catalyst",
                            simulator_version="1", meets_success_criteria=True, created_at="t")
        r = s.top1(task, [], [good], [prior])
        assert r["best_prior"] == 1.0 and r["improved"] is False
    finally:
        s.close()
