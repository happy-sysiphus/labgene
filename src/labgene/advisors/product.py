"""LabGene product advisor (spec §3.2 product, §13.1-13.4). Knowledge: hybrid retrieval (BM25 + dense + RRF; the
original question is kept, ontology expansion only if the profile enables it) as literature cards, conditional KG
paths for ontology entities of the question/task, current + own past observation cards, a descriptive calculation
card, past consultation / episode cases, plus the shared gated search/open. No BO/GP/acquisition optimizer (U4).
Cards reach the model only as model_projection() (never the access block)."""
from __future__ import annotations

import time
from typing import Any, Callable

from ..config import Limits, RoleModel
from ..contracts import (CardAccess, ConsultRequest, EvidenceCard, EvidenceKind, Observation, canonical_json,
                         payload_hash)
from ..costs import CallContext
from ..knowledge.base import GatedSearchTool
from ..knowledge.cards import (calculation_card, independent_count, kg_path_card, literature_card,
                               observation_cards)
from ..knowledge.kg import Ontology, kg_paths
from ..knowledge.store import KnowledgeStore
from ..memory.product import ProductMemory
from ..providers.base import LLMProvider
from .common import COMMON_SYSTEM, PROMPT_VERSION, ConsultController, prompt_hash

__all__ = ["ProductAdvisor", "PROMPT_VERSION", "PROMPT_HASH", "SYSTEM"]

SYSTEM = COMMON_SYSTEM + """
Your knowledge: `evidence_cards` retrieved by the LabGene product. kind literature = approved source text
(`excerpt` is verbatim, `summary` is generated); KG paths state a reported relation or a composed inference with
its conditions; current_observation / past_observation = real experiments with exact values (past ones are from
earlier episodes of this evaluation set: only a new experiment in this episode can meet the success criteria);
code_calculation = descriptive statistics (best so far, distance to targets, similar conditions), not predictions;
agent_inference = earlier consultation answers, not measurements. Cards sharing a `locator.study` come from one
study: count them as one piece of evidence (`independent_literature_studies`). Respect `applicability` (material,
equipment, ranges) and `uncertainty`. Cite literature and KG cards by card_id or their source_ids; cite
observations by observation id."""
PROMPT_HASH = prompt_hash(SYSTEM)


def case_cards(memory: ProductMemory, task_id: str, access: CardAccess) -> list[EvidenceCard]:
    """Past consultation answers (agent_inference) and episode cases (past_observation context) of this task."""
    out = []
    for c in memory.consult_cases(task_id):
        ex = c["exchange"]
        body = {"question": ex.question,
                **ex.response.model_dump(mode="json", include={"answer", "reasoning", "limitations", "candidates"})}
        out.append(EvidenceCard(
            card_id=f"card:consult:{ex.action_id}", kind=EvidenceKind.agent_inference, content_hash=payload_hash(body),
            locator={"episode_id": c["episode_id"], "action_id": ex.action_id, "task_id": task_id},
            summary=canonical_json(body), derivation={"method": "past_consultation_answer"},
            access=access.model_copy(update={"episode_id": c["episode_id"]}),
            uncertainty=["earlier advisor answer: an inference, not a measurement"]))
    for e in memory.episode_cases(task_id):
        best = e.get("best_observation_id")
        out.append(EvidenceCard(
            card_id=f"card:episode:{e['episode_id']}", kind=EvidenceKind.past_observation,
            observation_ids=[best] if best else [], content_hash=payload_hash(e),
            locator={"episode_id": e["episode_id"], "task_id": task_id}, summary=canonical_json(e),
            derivation={"method": "episode_summary"}, access=access.model_copy(update={"episode_id": e["episode_id"]})))
    return out


class ProductAdvisor(ConsultController):
    condition = "product"
    system = SYSTEM

    def __init__(self, provider: LLMProvider, role: RoleModel, limits: Limits, *, memory: ProductMemory,
                 store: KnowledgeStore, search: GatedSearchTool, ontology: Ontology | None,
                 sleep: Callable[[float], None] = time.sleep):
        super().__init__(provider, role, limits, store=store, search=search, sleep=sleep)
        self.memory, self.ontology = memory, ontology

    def knowledge(self, req: ConsultRequest, ctx: CallContext) -> tuple[dict[str, Any], list[str], list[Observation]]:
        s, task, k = self.store, req.task, self.limits.retrieval_top_k
        access = s.access(req.scope.memory_scope, req.scope.episode_id)
        lit = [c for v in s.retrieve(req.question, ctx, top_k=k, max_expansions=self.limits.max_query_expansions) if (c := literature_card(s, v.source_id, access))]
        kg: list[EvidenceCard] = []
        if self.ontology is not None:
            text = " ".join([req.question, task.title, task.problem, *(p.name for p in task.parameters),
                             *(m.name for m in task.metrics)])
            terms = dict.fromkeys(i for _, _, i in self.ontology.mentions(text))
            # ponytail: first top_k paths in term order; rank paths by question relevance if the KG grows
            paths = [p for t in terms for p in kg_paths(s, t)][:k]
            kg = list({c.card_id: c for c in (kg_path_card(s, p, access) for p in paths)}.values())
        past = self.memory.observations(task_id=task.task_id)   # own condition/set only: the memory is scoped
        cards = [*lit, *kg, *observation_cards(req.observations, past, access),
                 calculation_card(task, [*req.observations, *past], access), *case_cards(self.memory, task.task_id, access)]
        delivered = [i for c in [*lit, *kg] for i in (c.card_id, *c.source_ids)]
        return ({"evidence_cards": [c.model_projection() for c in cards],
                 "independent_literature_studies": independent_count([*lit, *kg])}, delivered, past)
