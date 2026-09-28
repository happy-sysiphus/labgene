"""Gemini Interactions adapter + gemini-embedding-2 embedder (T06, spec §10.1, §11.5). REST via stdlib.

API shape verified 2026-09-28 against:
  https://ai.google.dev/gemini-api/docs/interactions-overview   (POST v1beta/interactions, x-goog-api-key,
      store defaults true, previous_interaction_id carries history only: system_instruction/tools/
      generation_config are interaction-scoped and must be re-sent every call)
  https://ai.google.dev/api/interactions-api                    (status: in_progress|requires_action|completed|
      failed|cancelled|incomplete; response `steps` [model_output{content[text]}, function_call{id,name,arguments},
      thought]; usage total_input_tokens/total_output_tokens/total_thought_tokens/total_cached_tokens; `model`)
  https://ai.google.dev/gemini-api/docs/function-calling        (tools [{type:function,name,description,parameters}],
      input steps user_input / function_call / function_result{name,call_id,result[{type:text,text}]})
  https://ai.google.dev/gemini-api/docs/structured-output       (response_format {type:text, mime_type:application/json, schema})
  https://ai.google.dev/gemini-api/docs/thinking                (generation_config.thinking_level minimal|low|medium|high)
  https://ai.google.dev/gemini-api/docs/embeddings              (models/{m}:batchEmbedContents; gemini-embedding-2 has no
      task_type: task goes in the text, "task: search result | query: ..." / "title: ... | text: ...")
  https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2 (128-3072 dims, 8192 input tokens)
Only `function` tools are ever sent: no Google Search grounding, URL context, code execution or managed agents.
"""
from __future__ import annotations

import json
from typing import Any, Literal

from ..contracts import FunctionCall, ProviderResult, ProviderStatus, Usage
from .base import EmbeddingResult, GenerationRequest
from .http import Transport, api_key, infra_result, post_json, strip_models_prefix, urllib_transport

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
SDK = "rest-v1beta/urllib"
# ponytail: refusal is detected by block/safety markers in status detail fields; the exact Interactions
# safety-block shape is unverified until the live smoke (tests/live/test_gemini_smoke.py).
_REFUSAL_MARKERS = ("SAFETY", "PROHIBITED", "BLOCKLIST", "SPII", "RECITATION", "REFUS", "BLOCKED")


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _steps(turn: dict[str, Any]) -> list[dict[str, Any]]:
    role = turn.get("role")
    if role == "user":
        return [{"type": "user_input", "content": turn["text"]}]
    if role == "tool":
        return [{"type": "function_result", "name": turn["name"], "call_id": turn["call_id"],
                 "result": [{"type": "text", "text": json.dumps(turn["result"], ensure_ascii=False, sort_keys=True)}]}]
    if role == "model":
        out = [{"type": "model_output", "content": [{"type": "text", "text": turn["text"]}]}] if turn.get("text") else []
        return out + [{"type": "function_call", "id": c.get("call_id"), "name": c["name"], "arguments": c["arguments"]}
                      for c in turn.get("function_calls", [])]
    raise ValueError(f"unsupported turn role: {role!r}")


def _refused(data: dict[str, Any], steps: list[dict[str, Any]]) -> bool:
    fields = [data.get(k) for k in ("incomplete_details", "error", "prompt_feedback", "finish_reason", "block_reason")]
    blob = json.dumps(fields + [s.get("finish_reason") for s in steps], default=str).upper()
    return any(m in blob for m in _REFUSAL_MARKERS)


class GeminiInteractionsProvider:
    """LLMProvider for POST v1beta/interactions. One physical attempt per generate()."""
    name = "gemini"
    endpoint = "interactions"

    def __init__(self, transport: Transport | None = None, timeout_s: float = 120.0, base_url: str = BASE_URL):
        self.transport = transport or urllib_transport
        self.timeout_s = timeout_s
        self.url = f"{base_url}/interactions"

    @staticmethod
    def build_body(req: GenerationRequest) -> dict[str, Any]:
        gen = {"thinking_level": req.thinking_level, "max_output_tokens": req.max_output_tokens}
        body: dict[str, Any] = {
            "model": req.model,
            "input": [s for t in req.input for s in _steps(t)],
            "system_instruction": req.system_instruction,
            "tools": [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                      for t in req.tools],
            "generation_config": {k: v for k, v in gen.items() if v is not None},
            "store": req.store,
        }
        if req.previous_interaction_id:
            body["previous_interaction_id"] = req.previous_interaction_id
        if req.response_schema is not None:
            body["response_format"] = {"type": "text", "mime_type": "application/json", "schema": req.response_schema}
        return body

    def generate(self, req: GenerationRequest) -> ProviderResult:
        key = api_key("gemini")
        if not key:
            return infra_result(req, self.name, self.endpoint, "credential GEMINI_API_KEY is not set", sdk_version=SDK)
        reply = post_json(self.transport, self.url, {"x-goog-api-key": key}, self.build_body(req), self.timeout_s)
        data = reply.data if isinstance(reply.data, dict) else {}
        fail = reply.failure()
        if fail:
            if reply.status == 400 and _refused(data, []):
                return ProviderResult(role=req.role, provider=self.name, endpoint=self.endpoint,
                                      status=ProviderStatus.refusal, model_requested=req.model, error=fail,
                                      request_id=reply.request_id, latency_s=reply.latency_s, sdk_version=SDK)
            return infra_result(req, self.name, self.endpoint, fail, reply, SDK)
        # ponytail: the guide calls the list `outputs`, the API reference `steps`; accept both until live smoke pins it.
        steps = [s for s in (data.get("steps") or data.get("outputs") or []) if isinstance(s, dict)]
        texts: list[str] = []
        calls: list[FunctionCall] = []
        for s in steps:
            t = s.get("type")
            if t == "model_output":
                texts += [c.get("text", "") for c in s.get("content") or [] if isinstance(c, dict) and c.get("type") == "text"]
            elif t == "text":
                texts.append(s.get("text", ""))
            elif t == "function_call":
                args = s.get("arguments")
                calls.append(FunctionCall(call_id=s.get("id"), name=str(s.get("name")),
                                          arguments=args if isinstance(args, dict) else {"_unparsed": args}))
        text = "".join(texts) if texts else data.get("output_text")
        st = data.get("status")
        error = None
        if _refused(data, steps):
            status = ProviderStatus.refusal
        elif st in ("completed", "requires_action"):
            status = ProviderStatus.ok
        elif st == "incomplete":
            status = ProviderStatus.incomplete
        else:                                              # failed / cancelled / in_progress / unknown
            status, error = ProviderStatus.infra_error, f"interaction status {st!r}"
        u = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        usage = Usage(input_tokens=_int(u.get("total_input_tokens")), output_tokens=_int(u.get("total_output_tokens")),
                      reasoning_tokens=_int(u.get("total_thought_tokens")), cached_tokens=_int(u.get("total_cached_tokens")))
        return ProviderResult(role=req.role, provider=self.name, endpoint=self.endpoint, status=status,
                              text=text if isinstance(text, str) else None, function_calls=calls, usage=usage,
                              request_id=reply.request_id, interaction_id=data.get("id"), model_requested=req.model,
                              model_returned=strip_models_prefix(data.get("model")), sdk_version=SDK,
                              latency_s=reply.latency_s, error=error)


class GeminiEmbedder:
    """Embedder for models/{model}:batchEmbedContents. Task is expressed in the text (no task_type).
    One embed() = ONE physical request (callers cost each call and batch themselves). Once a request is made
    the result is returned, never raised: `dimensions` is what came back, and callers reject a mismatch with
    their pinned index (knowledge.retrieval.embed does) after recording the call."""
    name = "gemini"
    development_only = False
    BATCH = 100   # ponytail: max texts per request; confirm the batchEmbedContents limit in the live smoke

    def __init__(self, model: str = "gemini-embedding-2", dimensions: int = 3072, transport: Transport | None = None,
                 timeout_s: float = 60.0, base_url: str = BASE_URL):
        self.model, self.dimensions = model, dimensions
        self.transport = transport or urllib_transport
        self.timeout_s = timeout_s
        self.url = f"{base_url}/models/{model}:batchEmbedContents"

    @staticmethod
    def format_text(text: str, kind: Literal["query", "document"]) -> str:
        if kind == "query":
            return f"task: search result | query: {text}"
        if kind == "document":
            return f"title: none | text: {text}"
        raise ValueError(f"kind must be 'query' or 'document', got {kind!r}")

    def embed(self, texts: list[str], kind: Literal["query", "document"]) -> EmbeddingResult:
        if len(texts) > self.BATCH:
            raise ValueError(f"embed() takes at most {self.BATCH} texts (one physical request); got {len(texts)}")
        if not texts:
            return EmbeddingResult(vectors=[], model=self.model, dimensions=self.dimensions)
        formatted = [self.format_text(t, kind) for t in texts]
        key = api_key("gemini")
        if not key:
            return EmbeddingResult(vectors=[], model=self.model, dimensions=self.dimensions, status="infra_error",
                                   error="credential GEMINI_API_KEY is not set")
        body = {"requests": [{"model": f"models/{self.model}", "content": {"parts": [{"text": t}]},
                              "output_dimensionality": self.dimensions} for t in formatted]}
        reply = post_json(self.transport, self.url, {"x-goog-api-key": key}, body, self.timeout_s)
        fail = reply.failure()
        embs = None if fail else reply.data.get("embeddings")
        vectors = [e.get("values") if isinstance(e, dict) else None for e in embs] if isinstance(embs, list) else []
        ok = not fail and len(vectors) == len(texts) and all(
            isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
            for v in vectors)
        dims = {len(v) for v in vectors} if ok else set()
        if len(dims) != 1:
            return EmbeddingResult(vectors=[], model=self.model, dimensions=self.dimensions, status="infra_error",
                                   error=fail or "malformed batchEmbedContents response",
                                   request_id=reply.request_id, latency_s=reply.latency_s)
        return EmbeddingResult(vectors=[[float(x) for x in v] for v in vectors], model=self.model,
                               dimensions=dims.pop(), request_id=reply.request_id, latency_s=reply.latency_s)
