"""Advisor contract (§3.2, §11.1, §13). Advisors never run experiments."""
from __future__ import annotations

from typing import Protocol

from ..contracts import AdvisorOutcome, Condition, ConsultRequest
from ..costs import CallContext


class Advisor(Protocol):
    condition: Condition

    def consult(self, req: ConsultRequest, ctx: CallContext) -> AdvisorOutcome:
        """One consultation exchange. Internal search/reasoning/memory reads are internal costs.
        infra_error => harness retries the same action_id; nothing is counted until commit."""
        ...
