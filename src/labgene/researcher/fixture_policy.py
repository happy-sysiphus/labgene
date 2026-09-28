"""Deterministic offline researcher policy for FixtureProvider (development_only).

Exercises the real researcher and harness paths (consult, advisor-candidate adoption, valid experiments,
deliberately invalid experiments, analysis-tool round trip). A contract check, NOT a research strategy and never
evidence of research ability or product value. No randomness: the reply is a pure function of the request.

Reads HistoryItem payloads as: observation -> {"observation_id", "parameters", "results"};
consult -> {"response": {"candidates": [{"parameters": ...}]}} (or candidates at top level).
"""
from __future__ import annotations

import json
from typing import Any, Iterator

from ..config import FixtureBehaviour
from ..contracts import CategoricalParam, FunctionCall, IntegerParam, PublicTask, ValidParameters, canonical_json
from ..providers.base import GenerationRequest
from ..providers.fixture import FixturePolicy
from ..simulators.validation import validate_parameters

STEPS = (0.25, 0.125, 0.0625, 0.03125, 0.015625)
development_only = True


def _view(req: GenerationRequest) -> dict[str, Any]:
    return json.loads(req.input[0]["text"])["researcher_view"]


def _key(params: Any) -> str:
    if not isinstance(params, dict):
        return canonical_json(params)
    return canonical_json({k: round(v, 9) if isinstance(v, float) else v for k, v in params.items()})


def _center(task: PublicTask) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in task.parameters:
        if isinstance(p, CategoricalParam):
            out[p.name] = p.choices[0]
        elif isinstance(p, IntegerParam):
            out[p.name] = (p.min + p.max) // 2
        else:
            out[p.name] = (p.min + p.max) / 2.0
    return out


def _out_of_range(task: PublicTask) -> dict[str, Any]:
    params = _center(task)
    for p in task.parameters:
        if not isinstance(p, CategoricalParam):
            params[p.name] = p.max + (p.max - p.min or 1)
            return params
    params[task.parameters[0].name] = "__not_a_choice__"
    return params


def _score(task: PublicTask, results: dict[str, Any]) -> tuple[int, float]:
    met, margin = 0, 0.0
    for c in task.success:
        v = results.get(c.metric)
        if not isinstance(v, (int, float)):
            return (-1, float("-inf"))
        met += c.satisfied(v)
        margin += (v - c.target if c.direction == "maximize" else c.target - v) / max(abs(c.target), 1.0)
    return met, margin


def _candidates(history: list[dict[str, Any]]) -> Iterator[Any]:
    for h in reversed(history):
        if h["kind"] == "consult":
            resp = h["payload"].get("response", h["payload"])
            for c in resp.get("candidates", []) if isinstance(resp, dict) else []:
                yield c.get("parameters") if isinstance(c, dict) else None


def _coordinate(task: PublicTask, obs: list[dict[str, Any]], tried: set[str]) -> tuple[dict[str, Any], str]:
    base = max(obs, key=lambda o: _score(task, o["results"]))["parameters"] if obs else _center(task)
    first = [(dict(base), "start at the centre of the public ranges")] if not obs else []
    moves = []
    for s in STEPS:
        for p in task.parameters:
            if isinstance(p, CategoricalParam):
                moves += [({**base, p.name: c}, f"switch {p.name} to {c}") for c in p.choices if c != base[p.name]]
                continue
            for sign in (1, -1):
                x = min(p.max, max(p.min, base[p.name] + sign * s * (p.max - p.min)))
                x = round(x) if isinstance(p, IntegerParam) else x
                moves.append(({**base, p.name: x}, f"step {p.name} by {sign * s:+g} of its range from the best observation"))
    for cand, why in first + moves:
        v = validate_parameters(task, cand)
        if isinstance(v, ValidParameters) and _key(v.parameters) not in tried:
            return v.parameters, why
    # ponytail: space around the best point exhausted at the finest step -> repeat it (a valid, charged action)
    return dict(base), "repeat the best observation (local neighbourhood exhausted)"


def choose_action(view: dict[str, Any], behaviour: FixtureBehaviour) -> dict[str, Any]:
    task = PublicTask.model_validate(view["task"])
    history, used = view["history"], view["actions_used"]
    if used == 0 or (behaviour.consult_every > 0 and used % behaviour.consult_every == 0):
        q = (f"For task {task.task_id}, which parameter settings are most likely to meet all success criteria? "
             "Please suggest candidates within the public ranges and constraints.")
        return {"action": "consult", "args": {"question": q}, "note": {"next_action_reason": "fixture: scheduled consult"}}
    obs = [h["payload"] for h in history if h["kind"] == "observation"]
    n_exp = sum(h["kind"] in ("observation", "invalid_experiment") for h in history)
    every = behaviour.invalid_request_rate_every
    if every > 0 and (n_exp + 1) % every == 0:
        params, why = _out_of_range(task), "fixture: deliberately out-of-range request (invalid-path contract check)"
    else:
        tried = {_key(o.get("parameters")) for o in obs}
        adopted = next((v.parameters for c in _candidates(history)
                        for v in [validate_parameters(task, c)]
                        if isinstance(v, ValidParameters) and _key(v.parameters) not in tried), None)
        params, why = (adopted, "adopt the advisor's in-range untried candidate") if adopted is not None \
            else _coordinate(task, obs, tried)
    return {"action": "run_experiment", "args": {"hypothesis": f"fixture: {why}", "parameters": params},
            "note": {"next_action_reason": why}}


def make_fixture_policy(behaviour: FixtureBehaviour) -> FixturePolicy:
    """Policy for FixtureProvider serving the researcher roles. The planner calls `describe` once when there
    are observations (tool round trip); planner/reviewer otherwise reply with short text."""
    def policy(req: GenerationRequest) -> str | list[FunctionCall]:
        if req.role == "researcher_finalizer":
            return json.dumps(choose_action(_view(req), behaviour), sort_keys=True)
        if req.role in ("researcher_planner", "researcher_reviewer"):
            if any(t.get("role") == "tool" for t in req.input):
                return f"{req.role}: analysis read; proceed with the best-supported candidate."
            if req.role == "researcher_planner" and req.tools \
                    and any(h["kind"] == "observation" for h in _view(req)["history"]):
                return [FunctionCall(call_id="fixture-describe", name="describe", arguments={})]
            return f"{req.role}: no objections beyond range and duplicate checks."
        return f"[fixture {req.role}]"
    return policy
