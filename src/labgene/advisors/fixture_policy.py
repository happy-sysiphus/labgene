"""Deterministic offline advisor policy for FixtureProvider (development_only). A contract check that exercises the
real advisor path (gated search tool round trip, JSON answer, validation, revisit marking); never research advice,
retrieval quality or product value. No randomness: replies are a pure function of the consultation's requests.

First model call of a consultation (tools offered): one search(question). Then ONE JSON answer citing only ids it
saw (tool results, initial:text, literature/KG card ids, observation ids) and one candidate: the best successful
past observation of this task in its knowledge (baseline: parsed from `own_memory`; product: past_observation
cards), else an in-range point toward the unexplored side of the current best (constraints respected).
A stateful continuation carries only the new turns, so the policy keeps the consultation's first payload.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..contracts import (CategoricalParam, Condition, FunctionCall, IntegerParam, PublicTask, ValidParameters,
                         canonical_json)
from ..knowledge.build import BASELINE_INITIAL_ID
from ..providers.base import GenerationRequest
from ..providers.fixture import FixturePolicy
from ..simulators.validation import validate_parameters

development_only = True
_HEAD = re.compile(r"^## episode \d+ \| task (\S+) \| visit \d+$")
_OBS = re.compile(r"^- experiment \S+ -> observation (\S+): parameters (\{.*?\}) results (\{.*?\}) units \{.*?\} "
                  r"meets_success_criteria=(True|False)$")


def _json(text: Any) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _score(task: PublicTask, results: dict[str, Any]) -> tuple[int, float]:
    """(criteria met, -total normalized shortfall)."""
    met, gap = 0, 0.0
    for c in task.success:
        v = results.get(c.metric)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return -1, float("-inf")
        met += c.satisfied(v)
        gap += max(0.0, c.target - v if c.direction == "maximize" else v - c.target) / max(abs(c.target), 1.0)
    return met, -gap


def _memory_observations(text: str, task_id: str) -> list[dict[str, Any]]:
    """Observation lines of the baseline memory render under this task's episode headers."""
    out, task = [], None
    for line in text.splitlines():
        if m := _HEAD.match(line):
            task = m[1]
        elif (m := _OBS.match(line)) and task == task_id:
            out.append({"observation_id": m[1], "parameters": json.loads(m[2]), "results": json.loads(m[3]),
                        "meets_success_criteria": m[4] == "True"})
    return out


def _card_observations(cards: list[dict[str, Any]], task_id: str) -> list[dict[str, Any]]:
    return [{"observation_id": c["observation_ids"][0], "parameters": rec["parameters"], "results": rec["results"],
             "meets_success_criteria": c["applicability"]["meets_success_criteria"]}
            for c in cards if c["kind"] == "past_observation" and c.get("excerpt")
            and c["applicability"].get("task_id") == task_id for rec in [json.loads(c["excerpt"])]]


def _proposal(task: PublicTask, current: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    center = {p.name: p.choices[0] if isinstance(p, CategoricalParam) else
              (p.min + p.max) // 2 if isinstance(p, IntegerParam) else (p.min + p.max) / 2.0 for p in task.parameters}
    steps = [(center, "fixture: centre of the public ranges")]
    if current:
        best = max(current, key=lambda o: _score(task, o["results"]))["parameters"]
        steps = []
        for p in task.parameters:
            if isinstance(p, CategoricalParam):
                steps += [({**best, p.name: c}, f"fixture: try {p.name}={c} at the best current point")
                          for c in p.choices if c != best[p.name]]
                continue
            far = p.max if p.max - best[p.name] >= best[p.name] - p.min else p.min
            x = (best[p.name] + far) / 2
            steps.append(({**best, p.name: round(x) if isinstance(p, IntegerParam) else x},
                          f"fixture: move {p.name} halfway toward its unexplored bound from the best current point"))
        steps += [(center, "fixture: centre of the public ranges"), (best, "fixture: repeat the best current point")]
    tried = {canonical_json(o["parameters"]) for o in current}
    for params, why in steps:
        v = validate_parameters(task, params)
        if isinstance(v, ValidParameters) and canonical_json(v.parameters) not in tried:
            return v.parameters, why
    return steps[-1]   # ponytail: everything nearby tried -> repeat (a valid, charged suggestion)


def _values(o: dict[str, Any]) -> str:
    return ", ".join(f"{k} = {json.dumps(v)}" for k, v in {**o["parameters"], **o["results"]}.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool))


def _answer(condition: str, root: dict[str, Any], tool_ids: list[str]) -> dict[str, Any]:
    task = PublicTask.model_validate(root["task"])
    current = root["current_episode"]["observations"]
    if condition == "baseline":
        past = _memory_observations(root.get("own_memory") or "", task.task_id)
        sources = [BASELINE_INITIAL_ID] if root.get("initial_text") else []
    else:
        cards = root.get("evidence_cards", [])
        past = _card_observations(cards, task.task_id)
        sources = [c["card_id"] for c in cards if c["kind"] == "literature"]
    lines, cited = [], []
    if current:
        b = max(current, key=lambda o: _score(task, o["results"]))
        lines.append(f"Best current observation {b['observation_id']}: {_values(b)}.")
        cited.append(b["observation_id"])
    wins = [o for o in past if o["meets_success_criteria"]]
    if wins:
        w = max(wins, key=lambda o: _score(task, o["results"]))
        lines.append(f"Past observation {w['observation_id']} met the success criteria: {_values(w)}.")
        cited.append(w["observation_id"])
        params, why = w["parameters"], f"fixture: repeat past successful observation {w['observation_id']}"
    else:
        params, why = _proposal(task, current)
    lines.append("One candidate is suggested; running it is the researcher's choice.")
    return {"answer": " ".join(lines), "cited_source_ids": list(dict.fromkeys(sources[:2] + tool_ids[:2])),
            "cited_observation_ids": cited, "reasoning": "fixture: deterministic contract-check answer (development_only).",
            "limitations": "Offline fixture answer: not research advice; candidates are unverified suggestions.",
            "candidates": [{"parameters": params, "rationale": why}]}


def make_fixture_advisor_policy(condition: Condition) -> FixturePolicy:
    """Policy for the FixtureProvider serving role advisor_<condition>."""
    role, state = f"advisor_{condition}", {}

    def policy(req: GenerationRequest) -> str | list[FunctionCall]:
        if req.role != role:
            return f"[fixture {req.role}]"
        first = req.input[0] if req.input else {}
        payload = _json(first.get("text")) if first.get("role") == "user" else None
        if isinstance(payload, dict) and "question" in payload:
            state["root"] = payload        # this consultation's first request (also resent in stateless mode)
            if len(req.input) == 1 and req.tools:
                return [FunctionCall(call_id="fixture-search", name="search", arguments={"query": payload["question"]})]
        if "root" not in state:
            return "fixture: no consultation context"
        tool_ids = [r["source_id"] for t in req.input if t.get("role") == "tool"
                    for r in (t["result"].get("results") or []) if r.get("status") == "available"]
        return json.dumps(_answer(condition, state["root"], tool_ids), sort_keys=True)
    return policy
