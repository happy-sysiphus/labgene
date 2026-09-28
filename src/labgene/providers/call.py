"""Costed, finitely-retried provider calls + adapter factory (T06, spec §8.2, §11.5).

call_llm is the ONLY way core code should call an LLMProvider: every physical attempt is reserved against the
CostGuard, settled and emitted as a CostEvent; only infra_error is retried (same model, same request);
incomplete/refusal are returned; a returned model outside the accepted set raises ModelChangedError.
Nothing here ever substitutes another model or a fixture.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Iterable

from ..config import Limits, RoleModel
from ..contracts import ProviderResult, ProviderStatus
from ..costs import CallContext, CapExceeded
from .anthropic import AnthropicMessagesProvider
from .base import Embedder, GenerationRequest, LLMProvider, ModelChangedError
from .fixture import FixtureHashEmbedder, FixturePolicy, FixtureProvider
from .gemini import GeminiEmbedder, GeminiInteractionsProvider
from .http import Transport
from .openai import OpenAIResponsesProvider


def estimate_input_tokens(req: GenerationRequest) -> int:
    """Input tokens of the NEW content in req, erring high: tokenizers split digits and CJK/non-ASCII text
    finely, so those count 1 each and other characters 1 per 3.
    ponytail: heuristic, not a tokenizer; settle() replaces it with reported usage."""
    text = (req.system_instruction + json.dumps(req.input, ensure_ascii=False, default=str)
            + json.dumps([t.model_dump() for t in req.tools], ensure_ascii=False)
            + json.dumps(req.response_schema or {}, ensure_ascii=False))
    heavy = sum(1 for ch in text if ch.isdigit() or ord(ch) > 127)
    return heavy + (len(text) - heavy) // 3 + 1


def carried_tokens(req: GenerationRequest, res: ProviderResult, context_tokens: int = 0) -> int:
    """Tokens a continuation of res.interaction_id re-processes: this call's whole input (carried context + new
    content) plus its output and reasoning. max(reported, estimated): the Interactions docs do not say how stored
    history is billed, so the reservation errs high."""
    u = res.usage
    out = (u.output_tokens + (u.reasoning_tokens or 0)) if u.output_tokens is not None else (req.max_output_tokens or 0)
    return max(u.input_tokens or 0, context_tokens + estimate_input_tokens(req)) + out


def call_llm(provider: LLMProvider, req: GenerationRequest, ctx: CallContext, limits: Limits,
             allowed_models: Iterable[str] = (), sleep: Callable[[float], None] = time.sleep,
             context_tokens: int | None = None) -> ProviderResult:
    """context_tokens: stored history the provider re-processes because of req.previous_interaction_id
    (track it with carried_tokens). Required for a continuation under an input or USD cap."""
    accepted = {req.model, *allowed_models}
    guard = ctx.guard
    for attempt in range(1, limits.provider_max_attempts + 1):
        reservation = None
        if guard is not None:
            c = guard.caps
            if req.max_output_tokens is None and (c.max_output_tokens is not None or c.max_usd is not None):
                raise CapExceeded(f"{req.role}: max_output_tokens unset, worst-case output cannot be reserved under a cap")
            if req.previous_interaction_id and context_tokens is None and \
                    (c.max_input_tokens is not None or c.max_usd is not None):
                raise CapExceeded(f"{req.role}: continuation history size unknown, it cannot be reserved under a cap")
            # An exception out of generate() leaves this reservation open: conservative (counts toward caps).
            carried = (context_tokens or 0) if req.previous_interaction_id else 0
            reservation = guard.reserve(req.model, estimate_input_tokens(req) + carried, req.max_output_tokens or 0)
        res = provider.generate(req).model_copy(update={"attempt": attempt})
        if reservation is not None:
            guard.settle(reservation, res.usage)
        ctx.emit(kind="llm_call", role=req.role, provider=res.provider, model=req.model, status=res.status.value,
                 request_id=res.request_id, usage=res.usage, latency_s=res.latency_s, attempt=attempt,
                 detail={"endpoint": res.endpoint, "interaction_id": res.interaction_id,
                         "previous_interaction_id": req.previous_interaction_id, "store": req.store,
                         "model_returned": res.model_returned, "sdk_version": res.sdk_version, "error": res.error})
        # A generated reply must name its model: without it a change cannot be detected (§10.1), so fail closed.
        # Error replies (infra_error, HTTP-level refusal) carry no model and are exempt.
        if res.model_returned is None and res.status in (ProviderStatus.ok, ProviderStatus.incomplete):
            raise ModelChangedError(f"{req.role}: provider reply names no model; runtime model {req.model!r} unverifiable")
        if res.model_returned is not None and res.model_returned not in accepted:
            raise ModelChangedError(f"{req.role}: requested {req.model!r}, provider returned {res.model_returned!r}")
        if res.status is not ProviderStatus.infra_error or attempt == limits.provider_max_attempts:
            return res
        sleep(limits.provider_backoff_s * 2 ** (attempt - 1))
    raise AssertionError("unreachable: provider_max_attempts must be >= 1")


ENDPOINTS = {"fixture": "fixture", "gemini": "interactions", "openai": "responses", "anthropic": "messages"}


def build_llm(role_cfg: RoleModel, fixture_policy: FixturePolicy | None = None, transport: Transport | None = None,
              timeout_s: float = 120.0) -> LLMProvider:
    if role_cfg.endpoint != ENDPOINTS[role_cfg.provider]:
        raise ValueError(f"{role_cfg.provider} generation requires endpoint {ENDPOINTS[role_cfg.provider]!r}, "
                         f"got {role_cfg.endpoint!r}")
    if role_cfg.provider == "fixture":
        return FixtureProvider(policy=fixture_policy)
    cls = {"gemini": GeminiInteractionsProvider, "openai": OpenAIResponsesProvider,
           "anthropic": AnthropicMessagesProvider}[role_cfg.provider]
    return cls(transport=transport, timeout_s=timeout_s)


def build_embedder(role_cfg: RoleModel, transport: Transport | None = None, timeout_s: float = 60.0) -> Embedder:
    if role_cfg.provider == "fixture":
        return FixtureHashEmbedder(dimensions=role_cfg.dimensions or 256, model=role_cfg.model)
    if role_cfg.provider == "gemini" and role_cfg.endpoint == "embed_content":
        return GeminiEmbedder(model=role_cfg.model, dimensions=role_cfg.dimensions or 3072, transport=transport,
                              timeout_s=timeout_s)
    raise ValueError(f"no embedding adapter for {role_cfg.provider}/{role_cfg.endpoint}")
