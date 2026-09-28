"""Condition memory contract (T03, §11.3, §13.5).

Each (condition, set_id, set_rep) owns one state directory, restored from that condition's
initial snapshot at set start. Layout (shared with knowledge/):
    <state_dir>/memory.sqlite      memory module (baseline text records | product observation/case/relation store)
    <state_dir>/knowledge.sqlite   knowledge module (approved sources, chunks, provenance, KG, gate cache, cards)
    <state_dir>/index/             knowledge module (bm25.json, vectors.jsonl)
    <state_dir>/cache/             summaries / generated caches (any module, file-based)
Snapshots cover the WHOLE directory (memory/snapshot.py), never a single DB file.
"""
from __future__ import annotations

from typing import Literal, Protocol

from ..contracts import ConsultExchange, ExperimentError, Frozen, MemoryScope, Observation, Outcome, RunScope
from ..costs import CallContext


def finalize_event_id(scope: RunScope) -> str:
    """Fixed per episode: identical on every retry (§13.5). Never a provider request id."""
    return f"finalize:{scope.condition}:{scope.set_id}:rep{scope.set_rep}:{scope.episode_id}"


class FinalizeInput(Frozen):
    event_id: str
    scope: RunScope
    outcome: Outcome
    observations: list[Observation]          # ALL committed observations of the episode (success + miss)
    errors: list[ExperimentError]            # invalid requests: stored as errors, no values
    consults: list[ConsultExchange]          # all committed consult exchanges of the episode
    infra_notes: list[str] = []              # infra failures recorded as notes, never as values


class FinalizeResult(Frozen):
    event_id: str
    status: Literal["done", "failed"]
    records_added: int = 0
    records_skipped_duplicate: int = 0
    memory_hash: str | None = None
    error: str | None = None


class ConditionMemory(Protocol):
    scope: MemoryScope

    def finalize_episode(self, data: FinalizeInput, ctx: CallContext) -> FinalizeResult:
        """Idempotent by event_id AND natural ids (observation_id / action_id): a retry after a
        partial failure adds only what is missing. Records processing cost via ctx; never counts
        as an action."""
        ...

    def is_finalized(self, event_id: str) -> bool: ...

    def state_hash(self) -> str:
        """Logical content hash (not file bytes) of this condition's memory."""
        ...

    def close(self) -> None: ...
