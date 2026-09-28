"""Initial-state construction per condition (T04/T05). All costs are emitted with phase=prebuild (B22).

product : corpus -> parse -> identity check -> chunk -> gate each chunk -> index (BM25 + dense) -> KG relations
baseline: the curated initial text, capped at Limits.baseline_initial_text_max_chars, gated as exposed.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ..config import Limits, RetrievalConfig
from ..contracts import AnswerBundle, Condition, GateStatus, sha256_text
from ..costs import CallContext
from .base import LeakageChecker
from .gate import PREPROCESSING_VERSION, KnowledgeInfraError, identity_blocked, study_key
from .kg import Ontology, extract_relations
from .parse import PARSER_VERSION, chunk_document, parse_document
from .store import KnowledgeStore, exposure

BASELINE_INITIAL_ID = "initial:text"   # condition-neutral: advisors cite it and the researcher sees citations (I7)


def ingest_document(store: KnowledgeStore, path: Path, ctx: CallContext, extractor: Any = None,
                    max_chunk_chars: int = 1200) -> dict[str, Any]:
    """Idempotent per doc_id: a doc counts as done only after its chunks, index entries and KG relations are all
    committed, so a re-run after a crash finishes it. Each chunk is gated together with the metadata its views and
    cards expose. A doc whose identity is blocked, or with ANY chunk blocked, is blocked as a whole (nothing
    indexed, no KG). A gate error on any chunk stores nothing (status 'error'; the next build retries it).
    Held chunks are kept out of the index; they still count for context expansion."""
    text = path.read_text(encoding="utf-8")
    doc = parse_document(text, doc_id=path.stem)
    meta, sid, source_hash = doc.meta, f"doc:{path.stem}", sha256_text(text)
    done = store.get_state(f"ingested:{sid}")
    if done is not None:
        if done != source_hash:
            raise ValueError(f"{path.name} changed after it was ingested; build a fresh state dir")
        return {"doc": path.stem, "status": "exists"}
    base = {"title": meta.get("title", ""), "url": meta.get("url"), "body": doc.body, "file": path.name,
            "source_hash": source_hash, "study": study_key(meta, sid), "parser": PARSER_VERSION}
    blocked = identity_blocked(meta.get("url"), [meta.get("title", "")], meta, store.bundles)
    chunks = [] if blocked else chunk_document(doc, max_chars=max_chunk_chars)
    applicability = {k: meta[k] for k in ("material", "equipment") if meta.get(k)}
    metas = [{"header": c.header, "locator": c.locator, "uncertainty": c.uncertainty, "study": base["study"],
              "applicability": applicability, "url": meta.get("url")} for c in chunks]
    context = {"source": meta.get("url") or f"corpus:{path.name}", "origin": "corpus"}
    statuses = [store.gate(exposure(c.text, m), context, ctx) for c, m in zip(chunks, metas)]
    if blocked or GateStatus.block in statuses:    # blocked text is not copied into the product store
        store.add("source", "", [], "corpus", GateStatus.block, id=sid, meta=base, valid=False)
        store.set_state(f"ingested:{sid}", source_hash)
        return {"doc": path.stem, "status": "block"}
    if GateStatus.error in statuses:               # no verdict for some chunk: the doc-level block rule can't run
        return {"doc": path.stem, "status": "error"}
    store.add("source", text, [], "corpus", None, id=sid, meta=base)   # whole doc never gated -> never exposed
    allowed = []
    for n, (c, m, st) in enumerate(zip(chunks, metas, statuses)):
        cid = store.add("chunk", c.text, [sid], f"parser:{PARSER_VERSION}", st, id=f"{sid}#{n}", meta=m)
        if st == GateStatus.allow:
            allowed.append(cid)
    store.index(allowed, ctx)
    relations = [r for cid in allowed for r in extract_relations(store, cid, extractor, ctx, applicability)] \
        if extractor is not None and store.ontology is not None else []
    store.set_state(f"ingested:{sid}", source_hash)
    return {"doc": path.stem, "status": "ok", "chunks": len(chunks), "allowed": len(allowed),
            "not_allowed": len(chunks) - len(allowed), "relations": len(relations)}


def build_initial_state(state_dir: str | Path, condition: Condition, corpus_dir: str | Path, checker: LeakageChecker,
                        embedder: Any, bundles: list[AnswerBundle], ctx: CallContext, *, set_scope: str,
                        baseline_initial: str | Path | None = None, ontology: Ontology | None = None,
                        extractor: Any = None, limits: Limits = Limits(),
                        retrieval: RetrievalConfig = RetrievalConfig(),
                        execution_mode: str = "offline_fixture") -> dict[str, Any]:
    """Build ONE condition's initial knowledge state. Returns a manifest (no hidden values, no gate details).
    A gate error stores nothing for the affected document/text and raises KnowledgeInfraError after the other
    documents are done; re-running the build finishes the rest."""
    ctx = ctx.child(phase="prebuild")
    if execution_mode != "offline_fixture" and extractor is not None and extractor.development_only:
        raise ValueError(f"{type(extractor).__name__} is development_only; not allowed in {execution_mode}")
    state_dir = Path(state_dir)
    product = condition == "product"
    if product and embedder is None:
        raise ValueError("the product condition needs an embedder (hybrid retrieval; no BM25-only fallback)")
    store = KnowledgeStore(state_dir, checker, bundles, set_scope, embedder=embedder if product else None,
                           ontology=ontology if product else None, retrieval=retrieval, execution_mode=execution_mode)
    try:
        man: dict[str, Any] = {
            "condition": condition, "execution_mode": execution_mode, "set_scope": set_scope,
            "gate": {"checker": checker.checker_id, "policy_version": checker.policy_version,
                     "preprocessing": PREPROCESSING_VERSION, "development_only": checker.development_only},
            "answer_bundle_versions": sorted(f"{b.bundle_id}@{b.version}" for b in bundles)}
        if product:
            man["documents"] = [ingest_document(store, p, ctx, extractor) for p in sorted(Path(corpus_dir).glob("*.md"))]
            man["parser"] = PARSER_VERSION
            man["embedder"] = {"name": embedder.name, "model": embedder.model, "dimensions": embedder.dimensions,
                               "development_only": embedder.development_only}
            man["ontology"] = None if ontology is None else {"profile_id": ontology.profile_id,
                                                             "development_only": ontology.development_only}
            man["kg_extractor"] = getattr(extractor, "extractor_id", None)
            failed = sum(d["status"] == "error" for d in man["documents"])
            if failed:
                raise KnowledgeInfraError(f"leakage gate unavailable for {failed} corpus document(s); nothing from "
                                          "them was stored; re-run the build")
        elif baseline_initial is not None:
            doc = parse_document(Path(baseline_initial).read_text(encoding="utf-8"), "baseline_initial")
            body = doc.text[doc.body:]
            text = body[:limits.baseline_initial_text_max_chars]
            st = store.gate(text, {"source": f"corpus:{Path(baseline_initial).name}", "origin": "baseline_initial"}, ctx)
            if st == GateStatus.error:
                raise KnowledgeInfraError("leakage gate unavailable for the baseline initial text; nothing was "
                                          "stored; re-run the build")
            store.add("source", text, [], "baseline_initial", st, id=BASELINE_INITIAL_ID,
                      meta={"title": doc.meta.get("title", ""), "truncated": len(body) > len(text)})
            if store.get(BASELINE_INITIAL_ID)["content_hash"] != sha256_text(text):
                raise ValueError("the baseline initial text changed after this state was built; build a fresh state dir")
            man["baseline_initial"] = {"chars": len(text), "truncated": len(body) > len(text),
                                       "available": store.visible(BASELINE_INITIAL_ID) is not None}
    finally:
        store.close()
    return man


def load_baseline_initial(state_dir: str | Path) -> str | None:
    """The gated baseline initial text, or None if absent/not allowed."""
    p = Path(state_dir) / "knowledge.sqlite"
    if not p.exists():
        return None
    con = sqlite3.connect(p)
    try:
        r = con.execute("SELECT text FROM artifacts WHERE id=? AND valid=1 AND gate_status='allow'",
                        (BASELINE_INITIAL_ID,)).fetchone()
    finally:
        con.close()
    return r[0] if r else None
