"""Offline fixture provider + hash embedder (development_only). Contract checks only:
never evidence of research ability, retrieval quality or product value."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Callable, Iterable, Literal

from ..contracts import FunctionCall, ProviderResult, ProviderStatus, Usage
from .base import EmbeddingResult, GenerationRequest

FixturePolicy = Callable[[GenerationRequest], "str | list[FunctionCall]"]


def _default_policy(req: GenerationRequest) -> str:
    return f"[fixture {req.role}]"


class FixtureProvider:
    """Replies via `policy(req)` (text, or a list of FunctionCall). Captures every request in `.requests`
    (leakage / role-isolation tests). `script` injects statuses per call, in order (then ok):
    infra_error -> no content; incomplete -> the policy text cut in half (as a max-tokens truncation would);
    refusal -> no content. `chain_ids=False` mimics a stateless provider (no interaction_id)."""
    development_only = True

    def __init__(self, policy: FixturePolicy | None = None, script: Iterable[str | ProviderStatus] = (),
                 name: str = "fixture", chain_ids: bool = True):
        self.name = name
        self.policy = policy or _default_policy
        self.script = [ProviderStatus(s) for s in script]
        self.chain_ids = chain_ids
        self.requests: list[GenerationRequest] = []

    def generate(self, req: GenerationRequest) -> ProviderResult:
        self.requests.append(req)
        n = len(self.requests)
        status = self.script.pop(0) if self.script else ProviderStatus.ok
        text, calls = None, []
        if status in (ProviderStatus.ok, ProviderStatus.incomplete):
            out = self.policy(req)
            text, calls = (out, []) if isinstance(out, str) else (None, list(out))
            if status is ProviderStatus.incomplete:
                text, calls = (text or "")[: len(text or "") // 2], []
        return ProviderResult(role=req.role, provider="fixture", endpoint="fixture", status=status, text=text,
                              function_calls=calls, usage=Usage(), request_id=f"fixture-req-{n}",
                              interaction_id=f"fixture-int-{n}" if self.chain_ids else None,
                              model_requested=req.model, model_returned=req.model,
                              error="fixture: scripted infra_error" if status is ProviderStatus.infra_error else None)


class FixtureHashEmbedder:
    """Deterministic signed feature hashing of word tokens, L2-normalised. development_only=True."""
    name = "fixture"
    development_only = True

    def __init__(self, dimensions: int = 256, model: str = "fixture-hash-embedder"):
        self.model, self.dimensions = model, dimensions

    def embed(self, texts: list[str], kind: Literal["query", "document"]) -> EmbeddingResult:
        vectors = []
        for t in texts:
            v = [0.0] * self.dimensions
            for tok in re.findall(r"\w+", t.lower()):
                h = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:8], "big")
                v[h % self.dimensions] += 1.0 if (h >> 63) else -1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return EmbeddingResult(vectors=vectors, model=self.model, dimensions=self.dimensions)
