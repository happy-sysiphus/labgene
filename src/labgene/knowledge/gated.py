"""GatedSearchTool shared by both advisors (T04, spec §7.4, §13.2.8).

search/fetch proxy -> quarantine -> identity/alt-version check -> content gate on exactly what is exposed ->
approved store. search() omits anything not allowed; open() returns UNAVAILABLE_MESSAGE on block/hold/error.
"""
from __future__ import annotations

import json
import time
from typing import Literal

from ..contracts import GateStatus, canonical_json, sha256_text
from ..costs import CallContext, CapExceeded
from .base import RawHit, SearchProvider, SourceView
from .gate import KnowledgeInfraError, identity_blocked, norm_url, study_key
from .store import KnowledgeStore, exposure, unavailable


class GatedSearch:
    """For ONE MemoryScope/state dir. A source that was blocked or invalidated stays unavailable in this state."""

    def __init__(self, store: KnowledgeStore, provider: SearchProvider, *, max_results: int = 5,
                 max_chars: int = 8000):
        self.store, self.provider = store, provider
        self.max_results, self.max_chars = max_results, max_chars

    def _titles(self, hit: RawHit) -> list[str]:
        return [hit.title, hit.metadata.get("title", ""), hit.metadata.get("citation_title", "")]

    def search(self, query: str, ctx: CallContext) -> list[SourceView]:
        qkey = f"cache:search:{sha256_text(canonical_json([self.provider.name, self.max_results, query]))[:16]}"
        cached = self.store.db.execute("SELECT text FROM artifacts WHERE id LIKE ? AND valid=1 AND gate_status='allow'",
                                       (qkey + ":%",)).fetchone()
        if cached:
            ctx.emit(kind="search", role="gated_search", provider=self.provider.name, detail={"cache_hit": True})
            return [SourceView(**v) for v in json.loads(cached[0])]
        t0 = time.perf_counter()
        try:
            hits = self.provider.search(query, self.max_results)
        except CapExceeded:   # an approved cap stops the run; it is never downgraded to "search unavailable"
            raise
        except Exception as e:
            ctx.emit(kind="search", role="gated_search", provider=self.provider.name, status="error",
                     latency_s=time.perf_counter() - t0, detail={"op": "search", "error": type(e).__name__})
            raise KnowledgeInfraError("search provider unavailable") from None
        ctx.emit(kind="search", role="gated_search", provider=self.provider.name, latency_s=time.perf_counter() - t0,
                 detail={"op": "search", "hits": len(hits)})
        admitted = [self._admit(h, ctx) for h in hits]
        views = [self.store.view(sid) for sid, _ in admitted if sid is not None]
        if GateStatus.error not in [st for _, st in admitted]:   # transient gate errors are not frozen into the cache
            ids = [v.source_id for v in views]
            self.store.add("cache", canonical_json([v.model_dump(mode="json") for v in views]), ids, "gated_search",
                           GateStatus.allow, id=f"{qkey}:{sha256_text(canonical_json(ids))[:8]}")
        return views

    def _admit(self, hit: RawHit, ctx: CallContext) -> tuple[str | None, GateStatus]:
        """(approved source id | None, status). The status is internal; it is never exposed."""
        raw = self.store.quarantine(hit)
        sid = "web:" + sha256_text(norm_url(hit.url) or hit.url)[:16]
        old = self.store.get(sid)
        if old is not None:
            ok = old["valid"] and old["gate_status"] == "allow"
            return (sid, GateStatus.allow) if ok else (None, GateStatus.block)
        if identity_blocked(hit.url, self._titles(hit), hit.metadata, self.store.bundles):
            self.store.add("source", "", [raw], f"search:{hit.provider}", GateStatus.block, id=sid, valid=False)
            return None, GateStatus.block
        # title + snippet + url are exposed together, so they are gated together
        st = self.store.gate(f"{hit.title}\n{hit.snippet}\n{hit.url}", {"source": hit.url, "origin": "web_search"}, ctx)
        if st == GateStatus.block:
            self.store.add("source", "", [raw], f"search:{hit.provider}", st, id=sid, valid=False)
        if st != GateStatus.allow:
            return None, st
        meta = {"origin": "web", "url": hit.url, "title": hit.title, "snippet": hit.snippet,
                "study": study_key({**hit.metadata, "url": hit.url, "title": hit.title}, sid)}
        return self.store.add("source", f"{hit.title}\n{hit.snippet}\n{hit.url}", [raw], f"search:{hit.provider}",
                              st, id=sid, meta=meta), st

    def open(self, source_id: str, ctx: CallContext,
             expand: Literal["section", "document"] | None = None) -> SourceView:
        """Web source: fetch + identity + gate the content actually returned (truncated to max_chars).
        Corpus chunk with expand: the parent section / whole document is gated again as a new exposure."""
        row = self.store.visible(source_id)
        if row is None:
            return unavailable(source_id)
        m = self.store.meta(row)
        if m.get("origin") == "web":
            return self._open_web(source_id, m, ctx)
        if expand is None:
            return self.store.view(source_id)
        exp = self.store.expansion(source_id, expand)
        if exp is None:
            return unavailable(source_id)
        text, loc = exp[0][:self.max_chars], {**exp[1], "truncated": len(exp[0]) > self.max_chars}
        title = m.get("header", "")
        st = self.store.gate(exposure(text, {"header": title, "locator": loc}),
                             {"source": f"expansion:{source_id}", "origin": "context_expansion"}, ctx)
        if st != GateStatus.allow:
            return unavailable(source_id)
        return SourceView(source_id=source_id, title=title, text=text, locator=loc)

    def _open_web(self, sid: str, m: dict, ctx: CallContext) -> SourceView:
        cid = f"{sid}#content"
        if self.store.visible(cid) is not None:
            return self.store.view(cid)
        t0 = time.perf_counter()
        try:
            hit = self.provider.fetch(m["url"])
        except CapExceeded:
            raise
        except Exception as e:   # a dead page is unavailable, not a harness failure
            ctx.emit(kind="search", role="gated_search", provider=self.provider.name, status="error",
                     latency_s=time.perf_counter() - t0, detail={"op": "fetch", "error": type(e).__name__})
            return unavailable(sid)
        ctx.emit(kind="search", role="gated_search", provider=self.provider.name, latency_s=time.perf_counter() - t0,
                 detail={"op": "fetch", "found": hit is not None})
        if hit is None or not hit.content:
            return unavailable(sid)
        raw = self.store.quarantine(hit)
        if identity_blocked(hit.url, self._titles(hit), hit.metadata, self.store.bundles):
            self.store.invalidate(sid)
            return unavailable(sid)
        text = hit.content[:self.max_chars]
        meta = {"header": m["title"], "url": hit.url, "study": m.get("study"),
                "locator": {"url": hit.url, "truncated": len(hit.content) > self.max_chars}}
        st = self.store.gate(exposure(text, meta), {"source": hit.url, "origin": "web_fetch"}, ctx)
        if st == GateStatus.block:
            self.store.invalidate(sid)      # the whole source is blocked from now on (with its derivatives)
        if st != GateStatus.allow:
            return unavailable(sid)
        self.store.add("chunk", text, [sid, raw], f"fetch:{hit.provider}", st, id=cid, meta=meta)
        return self.store.view(cid)
