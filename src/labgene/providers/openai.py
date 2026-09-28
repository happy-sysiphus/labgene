"""OpenAI Responses adapter for INTERNAL roles only (kg_extractor / internal_summarizer), spec §10.1.

Verified 2026-09-28: https://developers.openai.com/api/reference/resources/responses/methods/create and
https://developers.openai.com/api/docs/models/gpt-6-astra (Responses supported; effort low..max; 128k output).
POST /v1/responses {model, instructions, input, tools[{type:function,...}], reasoning{effort}, max_output_tokens,
text{format{type:json_schema,name,schema,strict}}, store, previous_response_id}. Response: id, status
(completed|incomplete|failed|in_progress), incomplete_details.reason, model, output[message{content[output_text|refusal]},
function_call{call_id,name,arguments(JSON string)}], usage{input_tokens, output_tokens,
output_tokens_details.reasoning_tokens, input_tokens_details.cached_tokens}.
Usage.output_tokens here EXCLUDES reasoning tokens (costs.CostGuard.settle adds reasoning_tokens back).
"""
from __future__ import annotations

import json
from typing import Any

from ..contracts import FunctionCall, ProviderResult, ProviderStatus, Usage
from .base import GenerationRequest
from .http import Transport, api_key, infra_result, post_json, urllib_transport

URL = "https://api.openai.com/v1/responses"
SDK = "rest-v1/urllib"


def _items(turn: dict[str, Any]) -> list[dict[str, Any]]:
    role = turn.get("role")
    if role == "user":
        return [{"role": "user", "content": turn["text"]}]
    if role == "tool":
        return [{"type": "function_call_output", "call_id": turn["call_id"],
                 "output": json.dumps(turn["result"], ensure_ascii=False, sort_keys=True)}]
    if role == "model":
        out = [{"role": "assistant", "content": turn["text"]}] if turn.get("text") else []
        return out + [{"type": "function_call", "call_id": c.get("call_id"), "name": c["name"],
                       "arguments": json.dumps(c["arguments"], ensure_ascii=False)} for c in turn.get("function_calls", [])]
    raise ValueError(f"unsupported turn role: {role!r}")


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


class OpenAIResponsesProvider:
    name = "openai"
    endpoint = "responses"

    def __init__(self, transport: Transport | None = None, timeout_s: float = 120.0, url: str = URL):
        self.transport = transport or urllib_transport
        self.timeout_s = timeout_s
        self.url = url

    @staticmethod
    def build_body(req: GenerationRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": req.model,
            "instructions": req.system_instruction,
            "input": [i for t in req.input for i in _items(t)],
            # ponytail: strict=False (our schemas are not all strict-mode compatible); callers re-validate output
            "tools": [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters,
                       "strict": False} for t in req.tools],
            "store": req.store,
        }
        if req.reasoning_effort:
            body["reasoning"] = {"effort": req.reasoning_effort}
        if req.max_output_tokens is not None:
            body["max_output_tokens"] = req.max_output_tokens
        if req.previous_interaction_id:
            body["previous_response_id"] = req.previous_interaction_id
        if req.response_schema is not None:
            body["text"] = {"format": {"type": "json_schema", "name": "output", "schema": req.response_schema, "strict": False}}
        return body

    def generate(self, req: GenerationRequest) -> ProviderResult:
        key = api_key("openai")
        if not key:
            return infra_result(req, self.name, self.endpoint, "credential OPENAI_API_KEY is not set", sdk_version=SDK)
        reply = post_json(self.transport, self.url, {"authorization": f"Bearer {key}"}, self.build_body(req), self.timeout_s)
        fail = reply.failure()
        if fail:
            return infra_result(req, self.name, self.endpoint, fail, reply, SDK)
        data = reply.data
        texts: list[str] = []
        calls: list[FunctionCall] = []
        refused = False
        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for c in item.get("content") or []:
                    if c.get("type") == "output_text":
                        texts.append(c.get("text", ""))
                    elif c.get("type") == "refusal":
                        refused = True
            elif item.get("type") == "function_call":
                try:
                    args = json.loads(item.get("arguments") or "{}")
                except ValueError:
                    args = None
                calls.append(FunctionCall(call_id=item.get("call_id"), name=str(item.get("name")),
                                          arguments=args if isinstance(args, dict) else {"_unparsed": item.get("arguments")}))
        st = data.get("status")
        reason = (data.get("incomplete_details") or {}).get("reason")
        error = None
        if refused or reason == "content_filter":
            status = ProviderStatus.refusal
        elif st == "completed":
            status = ProviderStatus.ok
        elif st == "incomplete":
            status = ProviderStatus.incomplete
        else:
            status, error = ProviderStatus.infra_error, f"response status {st!r}"
        u = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        out, rt = _int(u.get("output_tokens")), _int((u.get("output_tokens_details") or {}).get("reasoning_tokens"))
        usage = Usage(input_tokens=_int(u.get("input_tokens")),
                      output_tokens=out - rt if out is not None and rt is not None else out,
                      reasoning_tokens=rt, cached_tokens=_int((u.get("input_tokens_details") or {}).get("cached_tokens")))
        return ProviderResult(role=req.role, provider=self.name, endpoint=self.endpoint, status=status,
                              text="".join(texts) if texts else None, function_calls=calls, usage=usage,
                              request_id=reply.request_id, interaction_id=data.get("id"), model_requested=req.model,
                              model_returned=data.get("model") if isinstance(data.get("model"), str) else None,
                              sdk_version=SDK, latency_s=reply.latency_s, error=error)
