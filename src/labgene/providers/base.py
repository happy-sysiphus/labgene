"""Provider adapter contract (T06, §11.5). Core code only sees these types."""
from __future__ import annotations

from typing import Any, Literal, Protocol

from ..contracts import Frozen, ProviderResult, Usage


class ModelChangedError(Exception):
    """The provider returned a model other than the configured one (or its allowed aliases).
    The running set must stop and be revalidated (§10.1); never silently continue."""


class ToolSpec(Frozen):
    name: str
    description: str
    parameters: dict[str, Any]          # JSON Schema (provider-supported subset)


class GenerationRequest(Frozen):
    role: str                           # researcher_planner | researcher_reviewer | researcher_finalizer |
                                        # advisor_baseline | advisor_product | internal_summarizer | kg_extractor | leakage_gate
    model: str
    system_instruction: str
    # provider-neutral turns: {"role":"user","text"} | {"role":"model","text"?,"function_calls":[{"call_id","name","arguments"}]}
    #                        | {"role":"tool","call_id","name","result"}
    input: list[dict[str, Any]]
    tools: list[ToolSpec] = []
    response_schema: dict[str, Any] | None = None
    thinking_level: str | None = None
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    previous_interaction_id: str | None = None   # same role, same decision/consult only; the provider re-processes the
                                                 # stored history -> pass call_llm(context_tokens=carried_tokens(...))
    store: bool = False


class LLMProvider(Protocol):
    name: str

    def generate(self, req: GenerationRequest) -> ProviderResult:
        """ONE physical attempt. Must not raise for API/network errors: return status=infra_error.
        Never substitutes another model or a fixture on failure."""
        ...


class EmbeddingResult(Frozen):
    vectors: list[list[float]]
    model: str
    dimensions: int
    usage: Usage = Usage()
    request_id: str | None = None
    latency_s: float = 0.0
    status: Literal["ok", "infra_error"] = "ok"
    error: str | None = None


class Embedder(Protocol):
    name: str
    model: str
    dimensions: int
    development_only: bool              # True for fixture embedders; never reported as real embedding quality

    def embed(self, texts: list[str], kind: Literal["query", "document"]) -> EmbeddingResult:
        """Exactly ONE physical request (callers batch and cost each call). Once sent, returns rather than raises;
        EmbeddingResult.dimensions is the returned dimensionality, checked by the caller against its pinned index."""
        ...
