"""Planner -> reviewer -> finalizer researcher (spec §3.1.1, §3.1.2, §11.1, §11.5; decisions I2, I7).

- 3 logical calls per decision, one per role, each a separate fresh request (roles never share a continuation).
- Planner/reviewer get ONLY the observation-only analysis tools (bounded by limits.researcher_max_analysis_calls
  per phase); they cannot consult or run experiments. The finalizer gets no tools and a response schema for ONE
  action JSON (+ optional research note). Transport retries happen inside call_llm and are all costed.
- previous_interaction_id is used only to continue a phase's own tool loop; nothing is carried across phases,
  decisions or episodes. Prompts are fixed constants, identical in both conditions; the view is sent as JSON and
  carries no condition/set identity.
- The researcher never executes anything and never repairs output: the finalizer's raw text goes to the harness.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from pydantic import ValidationError

from ..config import Limits, RoleModel
from ..contracts import (AnalysisResult, CategoricalParam, IntegerParam, ProviderResult, ProviderStatus, PublicTask, ResearcherDecision,
                         ResearcherView, ResearchNote, canonical_json, sha256_text)
from ..costs import CallContext
from ..providers.base import GenerationRequest, LLMProvider, ToolSpec
from ..providers.call import call_llm, carried_tokens
from .analysis import AnalysisError, AnalysisTools

PROMPT_VERSION = "researcher-v1"

_CONTEXT = """You are part of a research agent solving a virtual-experiment task. The user message is JSON.
`researcher_view` holds the public task (parameters with units and allowed ranges, linear constraints, metrics,
success criteria that must ALL hold), actions used and remaining out of a fixed budget, this episode's history
(consultation answers, real experiment observations, invalid requests, protocol errors, notices) and your earlier
research notes.
Facts: only observations in the history are measurements. Consultation answers are advice; analysis-tool outputs
are your own calculations or predictions. Neither is a measurement, and only a real experiment can meet the
success criteria.
Actions (chosen later by the finalizer, exactly one per decision): `consult` asks the consultation system one
question (1 action); `run_experiment` runs one parameter combination (1 action; an invalid request also costs 1).
Consult when you expect the answer to be worth an action; experimenting without consulting is equally valid."""

PLANNER_SYSTEM = _CONTEXT + """

Your step: PLANNER. You cannot consult or run experiments. You may call the analysis tools (describe, fit, predict,
nearest); they see only this episode's observations and the public task.
Reply in concise plain text: (1) what the observations show, citing observation ids; (2) competing hypotheses and
what would distinguish them; (3) one to three candidate next actions with full parameter values in public units,
and why; (4) whether a consultation is worth an action now."""

REVIEWER_SYSTEM = _CONTEXT + """

Your step: REVIEWER of the planner's proposal (`planner_proposal`). You cannot consult or run experiments and have
no access to answers or extra material. You may call the analysis tools.
Check: units and allowed ranges; linear constraints; every parameter present; duplicates of already-tried
combinations; evidence against the proposal; claims not supported by observations; remaining budget.
Reply in concise plain text: the problems found and the corrected recommendation."""

FINALIZER_SYSTEM = _CONTEXT + """

Your step: FINALIZER. Using `planner_proposal` and `reviewer_critique`, choose exactly ONE action and reply with
ONE JSON object only:
{"action":"consult","args":{"question":"..."}}
or {"action":"run_experiment","args":{"hypothesis":"...","parameters":{"<name>":<value>, ...}}}
`parameters` must give every task parameter exactly once, in public units, within ranges and constraints.
You may add "note": {"hypotheses":[...],"observation_ids":[...],"support":[...],"refute":[...],
"uncertainty":"...","next_action_reason":"..."} (short). Do not claim success; the harness judges it."""

PROMPT_HASH = sha256_text(canonical_json([PROMPT_VERSION, PLANNER_SYSTEM, REVIEWER_SYSTEM, FINALIZER_SYSTEM]))


def action_schema(task: PublicTask) -> dict[str, Any]:
    """Structured-output schema (Gemini JSON Schema subset): types only. Ranges/units are checked by the harness."""
    params = {p.name: {"type": "string", "enum": p.choices} if isinstance(p, CategoricalParam)
              else {"type": "integer" if isinstance(p, IntegerParam) else "number"} for p in task.parameters}
    strs = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "required": ["action", "args"], "properties": {
        "action": {"type": "string", "enum": ["consult", "run_experiment"]},
        "args": {"type": "object", "properties": {"question": {"type": "string"}, "hypothesis": {"type": "string"},
                                                   "parameters": {"type": "object", "properties": params}}},
        "note": {"type": "object", "properties": {"hypotheses": strs, "observation_ids": strs, "support": strs,
                                                  "refute": strs, "uncertainty": {"type": "string"},
                                                  "next_action_reason": {"type": "string"}}}}}


def _as_input(res: ProviderResult) -> str:
    if res.status is ProviderStatus.refusal:
        return "[no output: the model declined this step]"
    text = res.text or ""
    return text + "\n[output truncated at the token limit]" if res.status is ProviderStatus.incomplete else text


def _note(text: str | None) -> ResearchNote | None:
    try:
        obj = json.loads(text or "")
        return ResearchNote.model_validate(obj["note"]) if isinstance(obj, dict) and isinstance(obj.get("note"), dict) else None
    except (ValueError, ValidationError):
        return None


_DECISION_STATUS = {ProviderStatus.ok: "ok", ProviderStatus.incomplete: "provider_incomplete",
                    ProviderStatus.refusal: "provider_refusal", ProviderStatus.infra_error: "infra_error"}


class PlannerReviewerResearcher:
    """researcher.base.Researcher. role_cfg is profile.roles.researcher (shared by all three steps)."""

    def __init__(self, provider: LLMProvider, role_cfg: RoleModel, limits: Limits,
                 sleep: Callable[[float], None] = time.sleep):
        self.provider, self.cfg, self.limits, self.sleep = provider, role_cfg, limits, sleep
        self.allowed_models = [role_cfg.model, *role_cfg.allowed_returned_models]

    def _request(self, role: str, system: str, turns: list[dict[str, Any]], tools: list[ToolSpec],
                 schema: dict[str, Any] | None = None, prev: str | None = None, store: bool = False) -> GenerationRequest:
        c = self.cfg
        return GenerationRequest(role=role, model=c.model, system_instruction=system, input=turns, tools=tools,
                                 response_schema=schema, thinking_level=c.thinking_level,
                                 reasoning_effort=c.reasoning_effort, max_output_tokens=c.max_output_tokens,
                                 previous_interaction_id=prev, store=store)

    def _call(self, req: GenerationRequest, ctx: CallContext, context_tokens: int = 0) -> ProviderResult:
        return call_llm(self.provider, req, ctx, self.limits, self.allowed_models, self.sleep, context_tokens)

    def _phase(self, role: str, system: str, payload: dict[str, Any], tools: AnalysisTools, ctx: CallContext,
               analysis: list[AnalysisResult]) -> ProviderResult:
        """One logical step with a bounded analysis-tool loop. store=True only so the loop can continue the
        phase's OWN interaction via previous_interaction_id (interaction-scoped config is re-sent each call);
        `carried` is that interaction's history, reserved against the cost caps on every continuation."""
        specs, limit, used, carried = tools.specs(), self.limits.researcher_max_analysis_calls, 0, 0
        history: list[dict[str, Any]] = [{"role": "user", "text": canonical_json(payload)}]
        pending, prev = history, None
        while True:
            req = self._request(role, system, pending, specs if used < limit else [], prev=prev, store=True)
            res = self._call(req, ctx, carried)
            if res.status is not ProviderStatus.ok or not res.function_calls or used >= limit:
                return res
            outs = []
            for fc in res.function_calls:
                if used >= limit:
                    out: dict[str, Any] = {"error": "analysis call limit for this step reached"}
                else:
                    used += 1
                    try:
                        r = tools.run(fc.name, fc.arguments)
                        analysis.append(r)
                        out = r.model_dump(mode="json")
                    except AnalysisError as e:
                        out = {"error": str(e)}
                outs.append({"role": "tool", "call_id": fc.call_id, "name": fc.name, "result": out})
            if res.interaction_id:                 # stateful: continue this phase's own interaction
                prev, pending, carried = res.interaction_id, outs, carried_tokens(req, res, carried)
            else:                                  # stateless provider: resend the phase history
                history = history + [{"role": "model", "text": res.text,
                                      "function_calls": [fc.model_dump() for fc in res.function_calls]}] + outs
                prev, pending = None, history

    def decide(self, view: ResearcherView, ctx: CallContext) -> ResearcherDecision:
        tools = AnalysisTools(view)
        analysis: list[AnalysisResult] = []
        v = view.model_dump(mode="json")
        plan = self._phase("researcher_planner", PLANNER_SYSTEM, {"researcher_view": v}, tools, ctx, analysis)
        if plan.status is ProviderStatus.infra_error:
            return ResearcherDecision(raw_action_text=None, status="infra_error", analysis=analysis,
                                      error=f"researcher_planner: {plan.error}")
        review = self._phase("researcher_reviewer", REVIEWER_SYSTEM,
                             {"researcher_view": v, "planner_proposal": _as_input(plan)}, tools, ctx, analysis)
        if review.status is ProviderStatus.infra_error:
            return ResearcherDecision(raw_action_text=None, status="infra_error", analysis=analysis,
                                      error=f"researcher_reviewer: {review.error}")
        final_input = {"researcher_view": v, "planner_proposal": _as_input(plan), "reviewer_critique": _as_input(review)}
        final = self._call(self._request("researcher_finalizer", FINALIZER_SYSTEM,
                                         [{"role": "user", "text": canonical_json(final_input)}], [],
                                         schema=action_schema(view.task)), ctx)
        status = _DECISION_STATUS[final.status]
        return ResearcherDecision(raw_action_text=final.text, status=status,
                                  note=_note(final.text) if status == "ok" else None, analysis=analysis,
                                  error=f"researcher_finalizer: {final.error}" if final.error else None)
