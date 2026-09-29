"""Rebuild what an advisor received in one consultation and turn it into judge passages (spec §3.3, §5).

Only copies under the evaluation's out dir are opened: the target run's state is read once through the sqlite backup
API (labgene.memory.snapshot) and never written. The advisor model is never called (spec R2)."""
from __future__ import annotations

import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labgene.advisors.baseline import BaselineAdvisor
from labgene.advisors.common import base_payload
from labgene.advisors.product import ProductAdvisor
from labgene.app import Run, build_checker, build_embedder, build_ontology, build_reranker
from labgene.contracts import ConsultRequest, MemoryScope, canonical_json
from labgene.costs import CallContext
from labgene.knowledge.build import load_baseline_initial
from labgene.knowledge.store import KnowledgeStore
from labgene.memory.baseline import BaselineTextMemory
from labgene.memory.product import ProductMemory
from labgene.memory.snapshot import snapshot_state
from labgene.providers.call import build_llm

_HEADING = re.compile(r"^#{1,6} ", re.M)
_OBS_ID = re.compile(r"obs:[\w-]+(?::[\w-]+)*")
_CONDITION = re.compile(r"\b(?:baseline|product)-r\d+-")


class _NoAdvisorModel:
    """Stands in for the advisor provider: knowledge() never generates, so a call here is a bug."""
    name = "none"

    def generate(self, req):
        raise RuntimeError("rag_eval never calls the advisor model")


def copy_state(run: Run, condition: str, out: Path) -> Path:
    """<out>/state/<condition>/state: a backup-API copy of the condition's set state, made once. A copy without its
    manifest (written last) was interrupted and is redone."""
    dest = out / "state" / condition
    if not (dest / "manifest.json").exists():
        if dest.exists():
            shutil.rmtree(dest)
        snapshot_state(run.run_dir / "state" / condition / run.plan.set_id / "rep1", dest)
    return dest / "state"


def memory_copy(state: Path, orders: dict[str, int], episode_id: str, dest: Path) -> Path:
    """dest/memory.sqlite (+ cache/): the condition memory while `episode_id` ran. Memory is written at each episode's
    end, so only earlier episodes' rows stay (tables with episode_order, else episode_id)."""
    if (dest / "memory.sqlite").exists():
        return dest
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shutil.copy2(state / "memory.sqlite", tmp / "memory.sqlite")
    if (state / "cache").is_dir():
        shutil.copytree(state / "cache", tmp / "cache")
    cutoff = orders[episode_id]
    later = [(e,) for e, o in orders.items() if o >= cutoff]
    con = sqlite3.connect(tmp / "memory.sqlite")
    try:
        with con:
            tables = [t for (t,) in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
            for t in tables:
                cols = {r[1] for r in con.execute(f'PRAGMA table_info("{t}")')}
                if "episode_order" in cols:
                    con.execute(f'DELETE FROM "{t}" WHERE episode_order >= ?', (cutoff,))
                elif "episode_id" in cols:
                    con.executemany(f'DELETE FROM "{t}" WHERE episode_id = ?', later)
    finally:
        con.close()
    tmp.rename(dest)
    return dest


class Rebuilder:
    """One condition's copied state, opened once. rebuild() gives the advisor's exact user JSON and the source ids it
    delivered: retrieval may embed the question, memory may use the summary cache (or the summarizer on a miss)."""

    def __init__(self, run: Run, condition: str, orders: dict[str, int], out: Path):
        p = run.profile
        self.run, self.condition, self.orders, self.out = run, condition, orders, out
        self.state = copy_state(run, condition, out)
        product = condition == "product"
        self.store = KnowledgeStore(self.state, build_checker(p), run.bundles(), run.plan.set_id,
                                    embedder=build_embedder(p) if product else None,
                                    ontology=build_ontology(p) if product else None, retrieval=p.knowledge.retrieval,
                                    reranker=build_reranker(p) if product else None, execution_mode=p.execution_mode)
        self.scope = MemoryScope(run_id=run.manifest.run_id, condition=condition, set_id=run.plan.set_id, set_rep=1)

    def close(self) -> None:
        self.store.close()

    def rebuild(self, req: ConsultRequest, ctx: CallContext) -> tuple[dict[str, Any], list[str]]:
        p, lim = self.run.profile, self.run.profile.limits
        mem = memory_copy(self.state, self.orders, req.scope.episode_id,
                          self.out / "memory" / self.condition / req.scope.episode_id)
        if self.condition == "product":
            memory = ProductMemory(mem, self.scope, gate_derived=self.store.gate, tasks=self.run.tasks, limits=lim)
            advisor = ProductAdvisor(_NoAdvisorModel(), p.roles.advisor_product, lim, memory=memory, store=self.store,
                                     search=None, ontology=self.store.ontology)
        else:
            r = p.roles.internal_summarizer
            summ = None if r.provider == "fixture" else build_llm(r, timeout_s=lim.provider_timeout_s)
            memory = BaselineTextMemory(mem, self.scope, summ, lim, summarizer_role=r if summ else None,
                                        gate_derived=self.store.gate, tasks=self.run.tasks)
            advisor = BaselineAdvisor(_NoAdvisorModel(), p.roles.advisor_baseline, lim, memory=memory,
                                      store=self.store, search=None, initial_text=load_baseline_initial(self.state))
        try:
            extra, delivered, _ = advisor.knowledge(req, ctx)
        finally:
            memory.close()
        return {**base_payload(req), **extra}, delivered


@dataclass(frozen=True)
class Passage:
    text: str
    cites: frozenset[str] = frozenset()     # answer citation ids that count as citing this passage


def split_text(text: str, max_chars: int) -> list[str]:
    """Markdown heading sections, each packed from its paragraphs (lines when a paragraph is too long) into pieces of
    at most max_chars; a single longer line stays whole."""
    starts = sorted({0, *(m.start() for m in _HEADING.finditer(text))})
    out: list[str] = []
    for a, b in zip(starts, [*starts[1:], len(text)]):
        units = [u for u in re.split(r"\n\s*\n", text[a:b].strip()) if u.strip()]
        units = [x for u in units for x in ([u] if len(u) <= max_chars else u.split("\n")) if x.strip()]
        buf = ""
        for u in units:
            if buf and len(buf) + 2 + len(u) > max_chars:
                out.append(buf)
                buf = ""
            buf = f"{buf}\n\n{u}" if buf else u
        if buf:
            out.append(buf)
    return out


def passages(payload: dict[str, Any], max_chars: int) -> list[Passage]:
    """The advisor's whole input as passages in input order (spec §5): task, this episode's observations, invalid
    requests and earlier consults, then the condition's knowledge (product: one card each; baseline: initial-text
    sections, then own-memory pieces)."""
    ep = payload["current_episode"]
    out = [Passage(canonical_json({"task": payload["task"], "remaining_actions": payload["remaining_actions"]}))]
    out += [Passage(canonical_json(o), frozenset({o["observation_id"]})) for o in ep["observations"]]
    out += [Passage(canonical_json(e)) for e in ep["invalid_requests"]]
    out += [Passage(canonical_json({"question": c["question"], "answer": c["response"].get("answer", ""),
                                    "reasoning": c["response"].get("reasoning", "")})) for c in ep["prior_consults"]]
    for c in payload.get("evidence_cards") or []:
        ids = c.get("observation_ids", []) if c["kind"] in ("current_observation", "past_observation") \
            else c.get("source_ids", [])
        out.append(Passage(canonical_json({k: v for k, v in c.items() if k != "content_hash"}),
                           frozenset({c["card_id"], *ids})))
    if init := payload.get("initial_text"):
        out += [Passage(t, frozenset({init["source_id"]})) for t in split_text(init["text"], max_chars)]
    if mem := payload.get("own_memory"):
        out += [Passage(t, frozenset(_OBS_ID.findall(t))) for t in split_text(mem, max_chars)]
    return out


def retrieved_cards(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The product's retrieval output in delivered order: literature cards, then KG path cards (spec §4)."""
    return [c for c in payload.get("evidence_cards") or []
            if c["kind"] == "literature" or c["card_id"].startswith("card:path:")]


class Blinder:
    """Judge-facing text for one consultation (spec §5): every harness id in `ids` becomes a neutral token S1, S2, ...
    (numbered in sorted id order, longest match first); any remaining condition-bearing id prefix is dropped."""

    def __init__(self, ids: set[str]):
        ids = {i for i in ids if i}
        self.tokens = {i: f"S{n}" for n, i in enumerate(sorted(ids), start=1)}
        alts = "|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True))
        self._rx = re.compile(rf"(?<![\w:-])(?:{alts})(?![\w-])") if ids else None

    def __call__(self, text: str) -> str:
        if self._rx is not None:
            text = self._rx.sub(lambda m: self.tokens[m.group(0)], text)
        return _CONDITION.sub("", text)
