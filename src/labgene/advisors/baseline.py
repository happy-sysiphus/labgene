"""General-LLM advisor (spec §3.2 baseline, §11.1, §13.1). Knowledge: the gated initial text, its own condition's
text memory (render within limits.baseline_memory_max_chars) and the shared gated search/open.
No ontology, KG, product RAG or evidence cards."""
from __future__ import annotations

import time
from typing import Any, Callable

from ..config import Limits, RoleModel
from ..contracts import ConsultRequest, Observation
from ..costs import CallContext
from ..knowledge.base import GatedSearchTool
from ..knowledge.build import BASELINE_INITIAL_ID
from ..knowledge.gate import KnowledgeInfraError
from ..knowledge.store import KnowledgeStore
from ..memory.baseline import BaselineTextMemory, SummarizerUnavailable
from ..providers.base import LLMProvider
from .common import COMMON_SYSTEM, PROMPT_VERSION, ConsultController, mentions, prompt_hash

__all__ = ["BaselineAdvisor", "PROMPT_VERSION", "PROMPT_HASH", "SYSTEM"]

SYSTEM = COMMON_SYSTEM + """
Your knowledge: `initial_text` (source id initial:text, curated background) and `own_memory`, your own records
of earlier episodes in this evaluation set (past experiments with exact values, invalid requests, earlier
consultations, outcomes). Past results show what was observed before; only a new experiment in this episode can
meet the success criteria. Cite past observations by their observation ids."""
PROMPT_HASH = prompt_hash(SYSTEM)


class BaselineAdvisor(ConsultController):
    condition = "baseline"
    system = SYSTEM

    def __init__(self, provider: LLMProvider, role: RoleModel, limits: Limits, *, memory: BaselineTextMemory,
                 store: KnowledgeStore, search: GatedSearchTool, initial_text: str | None,
                 sleep: Callable[[float], None] = time.sleep):
        super().__init__(provider, role, limits, store=store, search=search, sleep=sleep)
        self.memory, self.initial_text = memory, initial_text

    def knowledge(self, req: ConsultRequest, ctx: CallContext) -> tuple[dict[str, Any], list[str], list[Observation]]:
        try:
            text = self.memory.render(self.limits.baseline_memory_max_chars, ctx)
        except SummarizerUnavailable:
            raise KnowledgeInfraError("baseline memory unavailable") from None
        # delivered past observations = those the rendered text (full or summary) names; exact values for checks.
        past = [o for o in self.memory.observations() if mentions(text, o.observation_id)]
        initial = {"source_id": BASELINE_INITIAL_ID, "text": self.initial_text} if self.initial_text else None
        return {"initial_text": initial, "own_memory": text}, [BASELINE_INITIAL_ID] if initial else [], past
