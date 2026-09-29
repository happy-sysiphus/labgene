"""Recommendation correctness on the task's own simulator (spec §4). Evaluator-side: the hidden thresholds and the raw
outputs here never reach a model or a run state, and report.split_sim keeps them out of the public files."""
from __future__ import annotations

import math
from typing import Any

from labgene.app import Run
from labgene.contracts import CandidateSuggestion, Observation, PublicTask, SuccessCriterion, ValidParameters
from labgene.simulators.factory import build_simulator
from labgene.simulators.validation import validate_parameters


def _ratio(c: SuccessCriterion, v: float | None) -> float:
    """value / threshold (maximize) or threshold / value (minimize); a non-positive threshold or value falls back to
    1.0 when satisfied, else 0.0."""
    if v is None or not math.isfinite(v):
        return 0.0
    thr = c.target - c.tolerance if c.direction == "maximize" else c.target + c.tolerance
    if thr <= 0 or v <= 0:
        return 1.0 if c.satisfied(v) else 0.0
    return v / thr if c.direction == "maximize" else thr / v


def score(task: PublicTask, hidden: list[SuccessCriterion], results: dict[str, float]) -> float:
    """Normalized score (spec §4): 0 when a public criterion fails, else the smallest ratio over the task's hidden
    criteria (v3: TON / the evaluator's TON threshold); 1.0 for a task without hidden criteria. It reaches 1 exactly
    when the task's success rule holds (a declared hidden criterion without a threshold never passes)."""
    declared = {h.metric for h in task.hidden_success}
    thresholds = [c for c in hidden if c.metric in declared]
    if declared - {c.metric for c in thresholds}:
        return 0.0
    if not task.success or not all(c.satisfied(results.get(c.metric)) for c in task.success):
        return 0.0
    return min((_ratio(c, results.get(c.metric)) for c in thresholds), default=1.0)


class SimScorer:
    """One adapter per simulator id of the target run's profile (worker processes start once)."""

    def __init__(self, run: Run):
        self.run, self.sims = run, {}

    def close(self) -> None:
        for s in self.sims.values():
            s.close()

    def _evaluate(self, task: PublicTask, parameters: dict[str, Any]) -> dict[str, float]:
        if task.simulator_id not in self.sims:
            p = self.run.profile
            self.sims[task.simulator_id] = build_simulator(task.simulator_id, p.simulators[task.simulator_id], task,
                                                           p.root)
        out = self.sims[task.simulator_id].evaluate(parameters)
        if out.simulator_version != task.simulator_version:
            raise ValueError(f"simulator {task.simulator_id} is {out.simulator_version}; the task needs "
                             f"{task.simulator_version}")
        return out.results

    def top1(self, task: PublicTask, hidden: list[SuccessCriterion], candidates: list[CandidateSuggestion],
             prior: list[Observation]) -> dict[str, Any]:
        """The advisor's first candidate on the simulator: status ok | no_candidate | invalid_candidate, success,
        score, improved (above the best score observed earlier in the episode), raw results, best_prior."""
        best = max((score(task, hidden, o.results) for o in prior), default=0.0)
        base = {"success": False, "score": 0.0, "improved": False, "results": None, "best_prior": best}
        if not candidates:
            return {**base, "status": "no_candidate"}
        v = validate_parameters(task, candidates[0].parameters)
        if not isinstance(v, ValidParameters):
            return {**base, "status": "invalid_candidate"}
        results = self._evaluate(task, v.parameters)
        s = score(task, hidden, results)
        return {"status": "ok", "success": task.is_success(results, hidden), "score": s, "improved": s > best,
                "results": results, "best_prior": best}
