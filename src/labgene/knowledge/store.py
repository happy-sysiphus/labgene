"""Approved-knowledge store for ONE (condition, set_id, set_rep) state dir (T04/T05, spec §7, §13.2-13.5).

<state_dir>/knowledge.sqlite: artifacts (raw/source/chunk/summary/relation/card/vector/cache) with lineage,
content hash, gate status and created_by; gate cache; KG relations. <state_dir>/index/: bm25.json, vectors.jsonl,
manifest.json. An artifact reaches a model only if valid=1 AND gate_status='allow'. Raw hits and corpus sources
(gate_status NULL: never gated as a whole) are never exposed; only their approved children are.
Every approval belongs to the gate identity (answer bundles, set scope, checker, policy, preprocessing) pinned in
the state on first open; opening it under another identity is refused, so an old allow is never reused (B14).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import RetrievalConfig
from ..contracts import AnswerBundle, CardAccess, GateStatus, MemoryScope, canonical_json, payload_hash, sha256_text
from ..costs import CallContext
from .base import UNAVAILABLE_MESSAGE, LeakageChecker, RawHit, SourceView
from .gate import PREPROCESSING_VERSION, cache_key, combine
from .kg import CLAIM_STATUSES, UNRESOLVED, Ontology, Relation, clean_conditions
from .retrieval import BM25Index, LLMReranker, VectorIndex, rrf

INDEXED_KINDS = ("chunk", "summary")
EXPOSED_META = ("header", "title", "url", "study", "applicability", "locator", "uncertainty")  # shown by views/cards
_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL, content_hash TEXT NOT NULL, gate_status TEXT,
  valid INTEGER NOT NULL DEFAULT 1, created_by TEXT NOT NULL, created_at TEXT NOT NULL, meta TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS lineage(
  child TEXT NOT NULL REFERENCES artifacts(id), parent TEXT NOT NULL REFERENCES artifacts(id),
  PRIMARY KEY(child, parent));
CREATE INDEX IF NOT EXISTS lineage_parent ON lineage(parent);
CREATE TABLE IF NOT EXISTS gate_cache(
  cache_key TEXT PRIMARY KEY, status TEXT NOT NULL, policy_version TEXT NOT NULL, checker TEXT NOT NULL,
  bundle TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS relations(
  id TEXT PRIMARY KEY REFERENCES artifacts(id), subject TEXT NOT NULL, subject_label TEXT NOT NULL,
  predicate TEXT NOT NULL, object TEXT NOT NULL, object_label TEXT NOT NULL, source_ids TEXT NOT NULL,
  conditions TEXT NOT NULL, claim_status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
_DESCENDANTS = """WITH RECURSIVE d(id) AS (SELECT ? UNION SELECT l.child FROM lineage l JOIN d ON l.parent = d.id)
SELECT d.id FROM d JOIN artifacts a ON a.id = d.id"""
_ANCESTORS = """WITH RECURSIVE u(id) AS (SELECT ? UNION SELECT l.parent FROM lineage l JOIN u ON l.child = u.id)
SELECT u.id FROM u WHERE u.id != ?"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _aid(kind: str, text: str, parents: list[str]) -> str:
    return f"{kind}:{sha256_text(canonical_json([kind, text, sorted(parents)]))[:16]}"


def exposure(text: str, meta: dict[str, Any]) -> str:
    """Exactly what a view or card of an artifact can show: its text plus the exposed metadata. Gate this."""
    return "\n".join([*(f"{k}: {meta[k] if isinstance(meta[k], str) else canonical_json(meta[k])}"
                        for k in EXPOSED_META if meta.get(k)), text])


def unavailable(source_id: str) -> SourceView:
    """The only thing a model learns on block/hold/error/missing: no title, no detail."""
    return SourceView(source_id=source_id, title="", text=UNAVAILABLE_MESSAGE, status="unavailable")


class KnowledgeStore:
    def __init__(self, state_dir: str | Path, checker: LeakageChecker, bundles: list[AnswerBundle], set_scope: str,
                 *, embedder: Any = None, ontology: Ontology | None = None,
                 retrieval: RetrievalConfig = RetrievalConfig(), reranker: LLMReranker | None = None,
                 execution_mode: str = "offline_fixture"):
        if not bundles:
            raise ValueError("no answer bundles: the leakage gate cannot run")
        if execution_mode != "offline_fixture":
            for c in (checker, embedder):
                if c is not None and c.development_only:
                    raise ValueError(f"{type(c).__name__} is development_only; not allowed in {execution_mode}")
        if retrieval.reranker == "llm" and reranker is None:
            raise ValueError("retrieval.reranker=llm requires an LLMReranker")
        self.dir = Path(state_dir)
        (self.dir / "index").mkdir(parents=True, exist_ok=True)
        self.checker, self.bundles, self.set_scope = checker, list(bundles), set_scope
        self.ontology, self.retrieval = ontology, retrieval
        self.reranker = reranker if retrieval.reranker == "llm" else None
        self.db = sqlite3.connect(self.dir / "knowledge.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(_SCHEMA)
        # ponytail: a changed identity refuses the state; in-place re-gating of every approval if rebuilds cost too much
        ident = payload_hash([sorted(payload_hash(b) for b in bundles), set_scope, checker.checker_id,
                              checker.policy_version, PREPROCESSING_VERSION])
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO state VALUES ('gate_identity', ?)", (ident,))
        if self.get_state("gate_identity") != ident:
            self.db.close()
            raise ValueError("this knowledge state was gated under other answer bundles, set scope, checker or "
                             "policy; its approvals cannot be reused: build a fresh state dir")
        if embedder is None and (self.dir / "index" / "manifest.json").exists():
            self.db.close()
            raise ValueError("this state's vector index is pinned to an embedder; pass it (no BM25-only fallback)")
        self.bm25 = BM25Index(self.dir / "index" / "bm25.json", retrieval.bm25_k1, retrieval.bm25_b)
        self.vectors = VectorIndex(self.dir / "index", embedder) if embedder is not None else None

    def close(self) -> None:
        self.db.close()

    def get_state(self, key: str) -> str | None:
        r = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def set_state(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, value))

    # ------------------------------------------------------------ gate

    def gate(self, text: str, context: dict[str, Any], ctx: CallContext) -> GateStatus:
        """Content gate against EVERY answer bundle (most restrictive wins). Errors are never cached.
        Returns only the status: internal_detail is dropped here and never persisted."""
        out = []
        for b in self.bundles:
            key = cache_key(text, b, self.set_scope, self.checker, context)
            row = self.db.execute("SELECT status FROM gate_cache WHERE cache_key=?", (key,)).fetchone()
            if row:
                st = GateStatus(row["status"])
            else:
                d = self.checker.check(text, b, context, ctx)
                st = d.status
                if st != GateStatus.error:
                    with self.db:
                        self.db.execute("INSERT OR REPLACE INTO gate_cache VALUES (?,?,?,?,?,?)",
                                        (key, st.value, d.policy_version, d.checker, f"{b.bundle_id}@{b.version}", _now()))
            ctx.emit(kind="gate", role="leakage_gate", status="error" if st == GateStatus.error else "ok",
                     detail={"verdict": st.value, "cache_hit": row is not None, "checker": self.checker.checker_id,
                             "policy_version": self.checker.policy_version})
            out.append(st)
        return combine(out)

    def access(self, scope: MemoryScope, episode_id: str | None = None) -> CardAccess:
        return CardAccess(condition=scope.condition, set_id=scope.set_id, set_rep=scope.set_rep, episode_id=episode_id,
                          gate_status=GateStatus.allow, gate_policy_version=self.checker.policy_version,
                          answer_bundle_version=",".join(sorted(f"{b.bundle_id}@{b.version}" for b in self.bundles)))

    # ------------------------------------------------------------ artifacts / lineage

    def add(self, kind: str, text: str, parents: list[str], created_by: str, status: GateStatus | None, *,
            id: str | None = None, meta: dict | None = None, valid: bool = True) -> str:
        """Idempotent insert: an existing id (possibly invalidated) is never overwritten or revived."""
        aid = id or _aid(kind, text, parents)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?,?,?,?,?,?)",
                            (aid, kind, text, sha256_text(text), status.value if status else None, int(valid),
                             created_by, _now(), canonical_json(meta or {})))
            self.db.executemany("INSERT OR IGNORE INTO lineage VALUES (?,?)", [(aid, p) for p in parents])
        return aid

    def quarantine(self, hit: RawHit) -> str:
        """Raw search/fetch result: stored, never exposed."""
        return self.add("raw", canonical_json(hit), [], f"search:{hit.provider}", None,
                        id="raw:" + sha256_text(canonical_json(hit.model_dump(exclude={"retrieved_at", "query"})))[:16],
                        meta={"url": hit.url})

    def get(self, artifact_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()

    def visible(self, artifact_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM artifacts WHERE id=? AND valid=1 AND gate_status='allow'",
                               (artifact_id,)).fetchone()

    def meta(self, row: sqlite3.Row) -> dict[str, Any]:
        return json.loads(row["meta"])

    def ancestors(self, artifact_id: str) -> list[str]:
        return sorted(r[0] for r in self.db.execute(_ANCESTORS, (artifact_id, artifact_id)))

    def register_derived(self, kind: str, text: str, parents: list[str], created_by: str, ctx: CallContext,
                         meta: dict[str, Any] | None = None) -> tuple[str, GateStatus]:
        """Gate a NEW generated summary/relation/caption on its own text (the parents' allow is not inherited)
        before it becomes visible or indexed. Every parent must be currently approved, so a regeneration after
        an invalidation can only use the remaining approved parents. A gate error stores nothing (a retry is gated
        afresh). Returns (id, status); the id exists in the store only if the status is not error."""
        bad = [p for p in parents if self.visible(p) is None]
        if bad:
            raise ValueError(f"cannot derive from unapproved or invalidated artifacts: {bad}")
        studies = sorted({self.meta(self.get(p)).get("study") for p in parents} - {None})
        meta = {"study": "|".join(studies) or None, **(meta or {})}
        st = self.gate(exposure(text, meta), {"source": "derived:" + ",".join(parents), "origin": "derived",
                                              "kind": kind}, ctx)
        if st == GateStatus.error:
            return _aid(kind, text, parents), st
        existed = self.get(_aid(kind, text, parents)) is not None
        aid = self.add(kind, text, parents, created_by, st, meta=meta)
        row = self.get(aid)
        if kind in INDEXED_KINDS and not existed:
            self.index([aid], ctx)
        return aid, GateStatus(row["gate_status"]) if row["valid"] else GateStatus.block

    def invalidate(self, artifact_id: str) -> list[str]:
        """Transitive: every descendant (chunks, summaries, relations, cards, vectors, caches) becomes invalid and
        leaves the BM25/vector indexes. A multi-parent artifact falls if ANY parent falls. Returns affected ids."""
        ids = [r[0] for r in self.db.execute(_DESCENDANTS, (artifact_id,))]
        with self.db:
            self.db.executemany("UPDATE artifacts SET valid=0 WHERE id=?", [(i,) for i in ids])
        self.bm25.remove(ids)
        if self.vectors is not None:
            self.vectors.remove(ids)
        return sorted(ids)

    # ------------------------------------------------------------ retrieval

    def index(self, ids: list[str], ctx: CallContext) -> None:
        """Only approved chunk/summary artifacts enter BM25 and the vector index."""
        rows = [r for r in map(self.visible, ids) if r is not None and r["kind"] in INDEXED_KINDS]
        items = {r["id"]: f"{self.meta(r).get('header', '')}\n{r['text']}" for r in rows}
        if not items:
            return
        self.bm25.add(items)
        if self.vectors is not None:
            e = self.vectors.embedder
            for i, v in self.vectors.add(items, ctx).items():
                self.add("vector", "", [i], f"embedder:{e.name}:{e.model}", None, id=f"vec:{i}",
                         meta={"model": e.model, "dimensions": e.dimensions, "vector_hash": payload_hash(v)})

    def retrieve(self, query: str, ctx: CallContext, top_k: int = 8, max_expansions: int | None = None) -> list[SourceView]:
        """Hybrid BM25 + exact-cosine dense, fused by RRF. The original query is always one ranking; the optional
        ontology expansion adds a ranking, never replaces it. Results are re-checked for approval."""
        n = self.retrieval.candidates_per_retriever
        rankings = [self.bm25.search(query, n)]
        expanded = self.ontology.expand(query) if self.retrieval.query_expansion and self.ontology             and max_expansions != 0 else None
        if expanded and expanded != query:
            rankings.append(self.bm25.search(expanded, n))
        if self.vectors is not None:
            rankings.append(self.vectors.search(query, n, ctx))
        ids = [i for i, _ in rrf(rankings, self.retrieval.rrf_k) if self.visible(i) is not None]
        if self.reranker is not None and ids:
            ids = self.reranker.rerank(query, [(i, self.visible(i)["text"]) for i in ids], ctx)
        views = [self.view(i) for i in ids[:top_k]]
        ctx.emit(kind="retrieval", role="retriever", detail={"rankings": len(rankings), "candidates": len(ids),
                                                             "returned": len(views), "expanded": bool(expanded)})
        return views

    def view(self, artifact_id: str) -> SourceView:
        r = self.visible(artifact_id)
        if r is None:
            return unavailable(artifact_id)
        m = self.meta(r)
        if m.get("origin") == "web":     # search view: the gated title + snippet + url
            return SourceView(source_id=r["id"], title=m["title"], text=m["snippet"], url=m["url"])
        return SourceView(source_id=r["id"], title=m.get("header") or m.get("title", ""), text=r["text"],
                          url=m.get("url"), locator=m.get("locator"))

    def expansion(self, chunk_id: str, level: str) -> tuple[str, dict] | None:
        """UNGATED parent-section or whole-document text around an approved corpus chunk. Callers must gate it."""
        r = self.visible(chunk_id)
        loc = (self.meta(r).get("locator") or {}) if r is not None and r["kind"] == "chunk" else {}
        doc = self.get(f"doc:{loc.get('doc_id')}")
        if not loc or doc is None or not doc["valid"] or level not in ("section", "document"):
            return None
        body = self.meta(doc).get("body", 0)
        if level == "document":
            s, e = body, len(doc["text"])
        else:
            sec = loc["section"]
            spans = [m["locator"]["span"] for m in (json.loads(x[0]) for x in self.db.execute(
                "SELECT a.meta FROM artifacts a JOIN lineage l ON l.child = a.id WHERE l.parent=? AND a.kind='chunk'",
                (doc["id"],))) if m["locator"]["section"][:len(sec)] == sec]
            s, e = min(x[0] for x in spans), max(x[1] for x in spans)
        return doc["text"][s:e], {**loc, "span": [s, e], "expanded": level}

    # ------------------------------------------------------------ KG

    def add_relation(self, subject: str, predicate: str, obj: str, parents: list[str], created_by: str,
                     ctx: CallContext, conditions: dict[str, Any] | None = None,
                     claim_status: str = "reported") -> tuple[str, GateStatus]:
        if claim_status not in CLAIM_STATUSES:
            raise ValueError(f"claim_status must be one of {CLAIM_STATUSES}")
        link = self.ontology.link if self.ontology else (lambda _: UNRESOLVED)
        s, p, o = link(subject), link(predicate), link(obj)
        conditions = clean_conditions(conditions)
        text = (f"{subject} [{s}] {predicate} [{p}] {obj} [{o}]; conditions: {canonical_json(conditions)}; "
                f"claim: {claim_status}")
        rid, st = self.register_derived("relation", text, parents, created_by, ctx)
        if st == GateStatus.error:
            return rid, st
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO relations VALUES (?,?,?,?,?,?,?,?,?)",
                            (rid, s, subject, p, o, obj, canonical_json(parents), canonical_json(conditions), claim_status))
        return rid, st

    def relations(self) -> list[Relation]:
        """Approved, valid relations only."""
        rows = self.db.execute("SELECT r.* FROM relations r JOIN artifacts a ON a.id = r.id "
                               "WHERE a.valid=1 AND a.gate_status='allow' ORDER BY r.id")
        return [Relation(id=r["id"], subject=r["subject"], subject_label=r["subject_label"], predicate=r["predicate"],
                         object=r["object"], object_label=r["object_label"], source_ids=json.loads(r["source_ids"]),
                         conditions=json.loads(r["conditions"]), claim_status=r["claim_status"]) for r in rows]
