"""Search providers behind the gated proxy (T04). Results land in quarantine; never shown un-gated."""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode, urlsplit

import yaml

from .base import RawHit
from .gate import KnowledgeInfraError, norm_url
from .retrieval import tokenize

Transport = Callable[[str, float], bytes]      # (url, timeout_s) -> response body
MAX_BODY_BYTES = 5_000_000


class FixtureSearchProvider:
    """development_only: token-overlap search over <dir>/*.yaml pages (url/title/doi/arxiv_id/snippet/content).
    Offline contract checks only."""
    name = "fixture"
    development_only = True

    def __init__(self, corpus_dir: str | Path):
        self.pages = []
        for p in sorted(Path(corpus_dir).glob("*.yaml")):
            with open(p, encoding="utf-8") as f:
                self.pages.append(yaml.safe_load(f))

    def _hit(self, p: dict, query: str | None, content: bool) -> RawHit:
        return RawHit(url=p["url"], title=p.get("title") or "", snippet=p.get("snippet") or "",
                      content=(p.get("content") or None) if content else None, provider=self.name,
                      retrieved_at="fixture", query=query,
                      metadata={k: str(p[k]) for k in ("doi", "arxiv_id") if p.get(k)})

    def search(self, query: str, max_results: int) -> list[RawHit]:
        q = set(tokenize(query))
        scored = [(len(q & set(tokenize(f"{p.get('title', '')} {p.get('snippet', '')} {p.get('content', '')}"))),
                   p["url"], p) for p in self.pages]
        return [self._hit(p, query, False) for s, _, p in sorted(scored, key=lambda x: (-x[0], x[1])) if s][:max_results]

    def fetch(self, url: str) -> RawHit | None:
        n = norm_url(url)
        return next((self._hit(p, None, True) for p in self.pages if norm_url(p["url"]) == n), None)


def _urllib_transport(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "labgene-harness/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(MAX_BODY_BYTES)


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title, self.parts, self.meta, self._skip, self._in_title = "", [], {}, 0, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style", "noscript"):
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta" and a.get("content"):
            key = {"citation_doi": "doi", "dc.identifier": "doi", "citation_arxiv_id": "arxiv_id",
                   "citation_title": "citation_title"}.get((a.get("name") or "").casefold())
            if key:
                self.meta.setdefault(key, a["content"])

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip and data.strip():
            self.parts.append(data.strip())


class GoogleCSESearchProvider:
    """Google Programmable Search (Custom Search JSON API). Credentials from env LABGENE_SEARCH_API_KEY /
    LABGENE_SEARCH_ENGINE_ID. Errors never carry the request URL (it contains the key)."""
    name = "google_cse"
    development_only = False
    ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, transport: Transport | None = None, timeout_s: float = 20.0,
                 env: dict[str, str] | None = None):
        env = dict(os.environ) if env is None else env
        self._key, self._cx = env.get("LABGENE_SEARCH_API_KEY"), env.get("LABGENE_SEARCH_ENGINE_ID")
        if not self._key or not self._cx:
            raise ValueError("LABGENE_SEARCH_API_KEY and LABGENE_SEARCH_ENGINE_ID must be set for google_cse search")
        self._transport, self.timeout_s = transport or _urllib_transport, timeout_s

    def _get(self, url: str) -> bytes:
        try:
            return self._transport(url, self.timeout_s)
        except Exception as e:
            raise KnowledgeInfraError(f"search transport failed ({type(e).__name__})") from None

    def search(self, query: str, max_results: int) -> list[RawHit]:
        body = self._get(self.ENDPOINT + "?" + urlencode(
            {"key": self._key, "cx": self._cx, "q": query, "num": max(1, min(10, max_results))}))
        try:
            items = json.loads(body).get("items", [])
        except (ValueError, AttributeError):
            raise KnowledgeInfraError("search response was not valid JSON") from None
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out = []
        for it in items[:max_results]:
            tags = ((it.get("pagemap") or {}).get("metatags") or [{}])[0]
            meta = {k: v for k, v in {"doi": tags.get("citation_doi") or tags.get("dc.identifier"),
                                      "arxiv_id": tags.get("citation_arxiv_id"),
                                      "citation_title": tags.get("citation_title")}.items() if v}
            out.append(RawHit(url=it["link"], title=it.get("title", ""), snippet=it.get("snippet", ""),
                              provider=self.name, retrieved_at=now, query=query, metadata=meta))
        return out

    def fetch(self, url: str) -> RawHit | None:
        if urlsplit(url).scheme not in ("http", "https"):
            return None
        body = self._get(url)
        if body.startswith(b"%PDF"):
            return None     # ponytail: PDF fetch needs the T08 parser; unavailable until then
        page = _Page()
        page.feed(body.decode("utf-8", errors="replace"))
        return RawHit(url=url, title=page.title.strip(), content=" ".join(page.parts) or None, provider=self.name,
                      retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), metadata=page.meta)
