"""Knowledge contracts (T04/T05, §7, §13).

Pipeline: search/fetch proxy -> quarantine -> identity/alt-version check -> content gate
-> approved store. Both advisors use the SAME gated tool; nothing un-gated (title, snippet,
table, context expansion, generated summary) reaches a model.
"""
from __future__ import annotations

from typing import Any, Callable, Literal, Protocol

from ..contracts import AnswerBundle, Frozen, GateStatus
from ..costs import CallContext

UNAVAILABLE_MESSAGE = "This material is unavailable."   # the only thing a model learns on block/hold/error


class RawHit(Frozen):
    """Quarantined search/fetch result. Never exposed to a model."""
    url: str
    title: str = ""
    snippet: str = ""
    content: str | None = None
    provider: str
    retrieved_at: str
    query: str | None = None
    metadata: dict[str, Any] = {}          # doi, arxiv_id, authors... as reported by the provider


class GateDecision(Frozen):
    status: GateStatus
    policy_version: str
    checker: str                           # checker model/prompt version id
    cache_key: str = ""                    # set by the store that keys the gate cache
    internal_detail: str = ""              # evaluator-only; never in model input, errors or public logs


class LeakageChecker(Protocol):
    policy_version: str
    checker_id: str
    development_only: bool                 # fixture checkers can never serve as the live gate

    def check(self, text: str, bundle: AnswerBundle, context: dict[str, Any], ctx: CallContext) -> GateDecision: ...


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[RawHit]: ...

    def fetch(self, url: str) -> RawHit | None: ...


class SourceView(Frozen):
    """Model-facing projection of an approved source or chunk."""
    source_id: str
    title: str
    text: str
    url: str | None = None
    locator: dict[str, Any] | None = None
    status: Literal["available", "unavailable"] = "available"


class GatedSearchTool(Protocol):
    """Shared by both advisors. Returns only gate-approved content, scoped to one MemoryScope."""

    def search(self, query: str, ctx: CallContext) -> list[SourceView]: ...

    def open(self, source_id: str, ctx: CallContext,
             expand: Literal["section", "document"] | None = None) -> SourceView:
        """Context expansion is gated again before exposure."""
        ...


# Gate for NEW derived text (consult answers, summaries, relations) before memory/index storage.
# block/hold/error -> withheld. Observations (source=experiment) are exempt (§7.3).
DerivedTextGate = Callable[[str, dict[str, Any], CallContext], GateStatus]
