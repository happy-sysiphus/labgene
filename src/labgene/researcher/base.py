"""Researcher contract (§3.1). The harness parses raw_action_text; the researcher never executes."""
from __future__ import annotations

from typing import Protocol

from ..contracts import ResearcherDecision, ResearcherView
from ..costs import CallContext


class Researcher(Protocol):
    def decide(self, view: ResearcherView, ctx: CallContext) -> ResearcherDecision:
        """planner -> reviewer -> finalizer (<= 3 logical calls). Returns the finalizer's raw action text.
        Must be a pure function of `view` (+ its own fixed config): no hidden cross-episode state."""
        ...
