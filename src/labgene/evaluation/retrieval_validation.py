"""Retrieval and answer development validation (T08.5, spec §13.6). Evaluator-side.

Compares retrieval configs on a dev set with a PRE-DECLARED adoption rule. The WHOLE dev set (queries, gold, k,
configs, adoption rule, transcriptions) is pinned by hash per dev-set version before scoring and recorded; a changed
input under the same version and lock is refused. The adoption is provisional unless every declared config ran on a
reviewed set with no development_only component (embedder, gate checker, reranker provider).
bm25_only / dense_only straight from the store's BM25Index / VectorIndex, rrf and rrf_reranker through
KnowledgeStore.retrieve. Metrics: recall@k and MRR@k (macro over queries), per-query misses. A gold item that no
approved chunk matches stays a miss (reported as unresolved), never dropped. Transcription checks compare approved
chunks with gold numbers / units / ranges / table lines / uncertainty flags. The answer-dev helper reports mechanical
checks (cards.validate_advisor_response) and semantic support (cards.check_citation_support) separately.
Retrieval scores alone never establish product value; fixture embedders/judges are development_only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from ..config import Strict, _load_yaml
from ..contracts import AdvisorResponse, Observation, PublicTask, payload_hash
from ..costs import CallContext
from ..knowledge.cards import check_citation_support, validate_advisor_response
from ..knowledge.retrieval import LLMReranker
from ..knowledge.store import KnowledgeStore
from . import pin_hash

ConfigName = Literal["bm25_only", "dense_only", "rrf", "rrf_reranker"]
Metric = Literal["recall_at_k", "mrr"]


class GoldRef(Strict):
    """An artifact id, or a document (+ optional section heading / page) resolved to approved chunks."""
    id: str | None = None
    doc: str | None = None
    section: str | None = None
    page: int | None = None

    @model_validator(mode="after")
    def _one(self) -> "GoldRef":
        if (self.id is None) == (self.doc is None):
            raise ValueError("give exactly one of id / doc")
        return self


class DevQuery(Strict):
    query_id: str
    query: str
    relevant: list[GoldRef]


class Transcription(Strict):
    item_id: str
    ref: GoldRef
    numbers: list[str] = []
    units: list[str] = []
    ranges: list[str] = []               # verbatim after whitespace collapse (dashes, inequalities kept)
    lines_in_every_chunk: list[str] = [] # e.g. table header and unit rows repeated in every split chunk
    uncertain: bool | None = None        # True: every matching chunk must carry an uncertainty flag


class AdoptionRule(Strict):
    primary: Metric
    tie_break: list[Literal["recall_at_k", "mrr", "cost_rank"]]
    cost_rank: dict[str, int]            # lower = cheaper


class DevSet(Strict):
    version: str
    status: Literal["development_only", "draft_unreviewed", "reviewed"]
    corpus: str
    k: int
    configs: list[ConfigName]
    adoption_rule: AdoptionRule
    queries: list[DevQuery]
    transcriptions: list[Transcription] = []

    @model_validator(mode="after")
    def _costs(self) -> "DevSet":
        if missing := set(self.configs) - set(self.adoption_rule.cost_rank):
            raise ValueError(f"adoption_rule.cost_rank lacks {sorted(missing)}")
        return self


def load_dev_set(path: str | Path) -> DevSet:
    return DevSet.model_validate(_load_yaml(path))


def _chunks(store: KnowledgeStore, ref: GoldRef) -> list[tuple[str, str, dict]]:
    """Approved, valid chunks matching the reference: (id, text, meta)."""
    if ref.id is not None:
        r = store.visible(ref.id)
        return [(r["id"], r["text"], store.meta(r))] if r is not None else []
    out = []
    for r in store.db.execute("SELECT id, text, meta FROM artifacts WHERE kind='chunk' AND valid=1 AND gate_status='allow'"
                              " ORDER BY id"):
        m = json.loads(r["meta"])
        loc = m.get("locator") or {}
        if loc.get("doc_id") != ref.doc or (ref.page is not None and loc.get("page") != ref.page):
            continue
        if ref.section is not None and ref.section.casefold() not in [s.casefold() for s in loc.get("section", [])]:
            continue
        out.append((r["id"], r["text"], m))
    return out


def score_query(retrieved: list[str], relevant: list[set[str]]) -> dict[str, Any]:
    """recall@k = gold items with any chunk in `retrieved` / gold items; reciprocal rank of the first relevant hit."""
    hits = [bool(ids & set(retrieved)) for ids in relevant]
    union = set().union(*relevant) if relevant else set()
    rank = next((i for i, x in enumerate(retrieved, start=1) if x in union), None)
    return {"recall": sum(hits) / len(relevant) if relevant else 0.0, "rr": 1.0 / rank if rank else 0.0,
            "missed": [i for i, h in enumerate(hits) if not h]}


def _ranking(store: KnowledgeStore, config: str, query: str, k: int, ctx: CallContext,
             reranker: LLMReranker | None) -> list[str]:
    n = store.retrieval.candidates_per_retriever
    if config == "bm25_only":
        return [i for i in store.bm25.search(query, n) if store.visible(i) is not None][:k]
    if config == "dense_only":
        return [i for i in store.vectors.search(query, n, ctx) if store.visible(i) is not None][:k]
    store.reranker = reranker if config == "rrf_reranker" else None
    try:
        return [v.source_id for v in store.retrieve(query, ctx, top_k=k)]
    finally:
        store.reranker = None


def check_transcription(store: KnowledgeStore, t: Transcription) -> dict[str, Any]:
    chunks = _chunks(store, t.ref)
    if not chunks:
        return {"item_id": t.item_id, "ok": False, "chunks": [], "issues": ["no approved chunk matches the reference"]}
    joined = "\n".join(text for _, text, _ in chunks)
    flat = " ".join(joined.split())
    found = set(re.findall(r"\d+(?:\.\d+)?", joined))
    issues = [f"number {x} missing" for x in t.numbers if x not in found]
    issues += [f"unit {u!r} missing" for u in t.units if u not in joined]
    issues += [f"range {r!r} not preserved" for r in t.ranges if " ".join(r.split()) not in flat]
    issues += [f"{cid} lacks line {line!r}" for cid, text, _ in chunks for line in t.lines_in_every_chunk
               if line not in text.split("\n")]
    flagged = [bool(m.get("uncertainty")) for _, _, m in chunks]
    if t.uncertain is True and not all(flagged):
        issues.append("uncertain transcription not flagged")
    if t.uncertain is False and any(flagged):
        issues.append("transcription flagged uncertain")
    return {"item_id": t.item_id, "ok": not issues, "chunks": [c[0] for c in chunks], "issues": issues}


def adopt(results: dict[str, dict[str, Any]], rule: AdoptionRule) -> dict[str, Any]:
    """Maximize rule.primary; ties by rule.tie_break (metrics higher, cost_rank lower). complete=False if a
    declared config did not run: the choice is then provisional."""
    ran = {n: r for n, r in results.items() if "not_run" not in r}

    def key(n: str) -> tuple:
        return (-ran[n][rule.primary], *[rule.cost_rank[n] if t == "cost_rank" else -ran[n][t] for t in rule.tie_break])

    return {"chosen": min(ran, key=key) if ran else None, "complete": len(ran) == len(results),
            "not_run": sorted(set(results) - set(ran))}


def run_retrieval_validation(store: KnowledgeStore, dev: DevSet, ctx: CallContext,
                             reranker: LLMReranker | None = None, input_lock: str | Path | None = None) -> dict[str, Any]:
    """store: an opened product KnowledgeStore without a reranker (rrf_reranker gets `reranker` explicitly).
    A config that cannot run here is reported as not_run, never substituted. input_lock pins the dev-set hash (all of
    it but version/status) per version before scoring; required for a non-development run."""
    if store.reranker is not None:
        raise ValueError("open the store without a reranker; pass the reranker to compare rrf_reranker")
    embedder = store.vectors.embedder if store.vectors is not None else None
    rr_provider = reranker.provider if reranker is not None else None
    development_only = dev.status == "development_only" or any(
        bool(getattr(c, "development_only", False)) for c in (embedder, store.checker, rr_provider))
    rule_hash = payload_hash(dev.adoption_rule)
    input_hash = payload_hash(dev.model_dump(mode="json", exclude={"version", "status"}))
    if input_lock is not None:
        pin_hash(input_lock, f"retrieval_input:{dev.version}", input_hash)
    elif not development_only:
        raise ValueError("a non-development retrieval validation must pin its input (input_lock)")
    gold = {q.query_id: [{c[0] for c in _chunks(store, g)} for g in q.relevant] for q in dev.queries}
    results: dict[str, dict[str, Any]] = {}
    for config in dev.configs:
        if config == "rrf_reranker" and reranker is None:
            results[config] = {"not_run": "no reranker supplied"}
            continue
        if config == "dense_only" and store.vectors is None:
            results[config] = {"not_run": "the store has no vector index"}
            continue
        rows = []
        for q in dev.queries:
            got = _ranking(store, config, q.query, dev.k, ctx, reranker)
            s = score_query(got, gold[q.query_id])
            rows.append({"query_id": q.query_id, "retrieved": got, "recall": s["recall"], "rr": s["rr"],
                         "misses": [q.relevant[i].model_dump(exclude_none=True) for i in s["missed"]]})
        results[config] = {"recall_at_k": round(sum(r["recall"] for r in rows) / len(rows), 4),
                           "mrr": round(sum(r["rr"] for r in rows) / len(rows), 4), "queries": rows}
    unresolved = {q.query_id: [g.model_dump(exclude_none=True) for g, ids in zip(q.relevant, gold[q.query_id]) if not ids]
                  for q in dev.queries}
    adoption = adopt(results, dev.adoption_rule)
    adoption["provisional"] = not adoption["complete"] or dev.status != "reviewed" or development_only
    return {"dev_set_version": dev.version, "status": dev.status, "k": dev.k, "input_hash": input_hash,
            "development_only": development_only,
            "embedder": None if embedder is None else {"model": embedder.model,
                                                       "development_only": embedder.development_only},
            "checker": {"id": store.checker.checker_id, "policy_version": store.checker.policy_version,
                        "development_only": bool(store.checker.development_only)},
            "reranker": None if reranker is None else {
                "model": reranker.model, "provider": getattr(rr_provider, "name", None),
                "development_only": bool(getattr(rr_provider, "development_only", False))},
            "retrieval_config": store.retrieval.model_dump(mode="json"),
            "adoption_rule": dev.adoption_rule.model_dump(), "adoption_rule_hash": rule_hash,
            "unresolved_gold": {k: v for k, v in unresolved.items() if v},
            "configs": results, "adoption": adoption,
            "transcriptions": [check_transcription(store, t) for t in dev.transcriptions]}


def evaluate_answer(response: AdvisorResponse, delivered: dict[str, str], observations: list[Observation],
                    task: PublicTask, judge: Any, ctx: CallContext | None = None,
                    claims: list[tuple[str, str]] | None = None, store: KnowledgeStore | None = None) -> dict[str, Any]:
    """Answer-dev case (§13.6). delivered: source id -> text given to the advisor in this consultation.
    claims: (claim, cited source id); default = the whole answer against each cited source. Mechanical and semantic
    results are reported separately; a mechanical pass says nothing about support. semantic.ok: True only if every
    claim is supported, False if any is not, None if any verdict is unavailable or nothing was cited."""
    _, mech = validate_advisor_response(response, list(delivered), observations, task, store)
    pairs = claims if claims is not None else [(response.answer, s) for s in response.cited_source_ids]
    sem = []
    for claim, sid in pairs:
        if sid not in delivered:
            sem.append({"claim": claim, "source_id": sid, "supported": None, "detail": "cited source not delivered"})
            continue
        v = check_citation_support(claim, delivered[sid], judge, ctx)
        sem.append({"claim": claim, "source_id": sid, **v.model_dump()})
    verdicts = [s["supported"] for s in sem]
    ok = False if False in verdicts else None if (None in verdicts or not verdicts) else True
    return {"mechanical": {"ok": not mech, "issues": mech},
            "semantic": {"ok": ok, "judgements": sem, "development_only": bool(getattr(judge, "development_only", True))}}
