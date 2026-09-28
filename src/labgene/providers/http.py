"""Shared REST plumbing for provider adapters (T06). stdlib urllib; transport is injectable for tests.

Transport contract: transport(method, url, headers, body_bytes, timeout_s) -> (status, headers, body_bytes).
HTTP error statuses are returned, not raised; network/timeout failures raise OSError/HTTPException.
API keys travel only in request headers, are read from the environment at call time and never logged.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from http.client import HTTPException
from typing import Any, Callable

from ..config import CREDENTIAL_ENV
from ..contracts import ProviderResult, ProviderStatus
from .base import GenerationRequest

Transport = Callable[[str, str, dict[str, str], bytes, float], tuple[int, dict[str, str], bytes]]

REQUEST_ID_HEADERS = ("x-request-id", "request-id", "x-goog-request-id")


def urllib_transport(method: str, url: str, headers: dict[str, str], body: bytes, timeout: float
                     ) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()


def api_key(provider: str) -> str | None:
    return os.environ.get(CREDENTIAL_ENV[provider]) or None


class HttpReply:
    """Outcome of one POST: either `error` (network/timeout) or status + parsed JSON (None if unparsable)."""

    def __init__(self, status: int | None, headers: dict[str, str], data: Any, latency_s: float, error: str | None):
        self.status, self.headers, self.data, self.latency_s, self.error = status, headers, data, latency_s, error

    @property
    def request_id(self) -> str | None:
        return next((self.headers[h] for h in REQUEST_ID_HEADERS if h in self.headers), None)

    def failure(self) -> str | None:
        """Non-None when the reply is an infra failure (network, timeout, any non-200, non-JSON body)."""
        if self.error:
            return self.error
        if self.status != 200:
            msg = self.data.get("error") if isinstance(self.data, dict) else None
            msg = msg.get("message", msg) if isinstance(msg, dict) else msg
            return f"http {self.status}: {str(msg or '')[:300]}"
        if not isinstance(self.data, dict):
            return "http 200 with non-JSON body"
        return None


def post_json(transport: Transport, url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float) -> HttpReply:
    t0 = time.perf_counter()
    try:
        status, hdrs, raw = transport("POST", url, {**headers, "content-type": "application/json"},
                                      json.dumps(body, ensure_ascii=False).encode("utf-8"), timeout_s)
    except (OSError, HTTPException) as e:          # URLError, TimeoutError, ConnectionReset, IncompleteRead
        return HttpReply(None, {}, None, time.perf_counter() - t0, f"network: {type(e).__name__}: {str(e)[:200]}")
    latency = time.perf_counter() - t0
    try:
        data = json.loads(raw.decode("utf-8")) if raw else None
    except ValueError:
        data = None
    return HttpReply(status, {k.lower(): v for k, v in hdrs.items()}, data, latency, None)


def infra_result(req: GenerationRequest, provider: str, endpoint: str, error: str, reply: HttpReply | None = None,
                 sdk_version: str | None = None) -> ProviderResult:
    return ProviderResult(role=req.role, provider=provider, endpoint=endpoint, status=ProviderStatus.infra_error,
                          model_requested=req.model, error=error, sdk_version=sdk_version,
                          request_id=reply.request_id if reply else None,
                          latency_s=reply.latency_s if reply else 0.0)


def strip_models_prefix(model: Any) -> str | None:
    return model.removeprefix("models/") if isinstance(model, str) and model else None
