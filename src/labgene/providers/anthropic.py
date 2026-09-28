"""Anthropic Messages adapter for INTERNAL roles only (leakage_gate etc.), spec §10.1.

Verified 2026-09-28: https://platform.claude.com/docs/en/models/fable-5-1/overview (claude-fable-5-1, 128k output,
adaptive thinking always on, forced tool use returns an error -> we never send tool_choice),
https://platform.claude.com/docs/en/build-with-claude/effort (output_config.effort low..max, anthropic-version 2023-06-01),
https://platform.claude.com/docs/en/build-with-claude/structured-outputs (output_config.format{type:json_schema,schema},
no beta header; stop_reason refusal / max_tokens).
Stateless: no continuation id; callers resend history. Usage.output_tokens includes thinking (not reported separately).
"""
from __future__ import annotations

import json
from typing import Any

from ..contracts import FunctionCall, ProviderResult, ProviderStatus, Usage
from .base import GenerationRequest
from .http import Transport, api_key, infra_result, post_json, urllib_transport

URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
SDK = f"rest-{API_VERSION}/urllib"


def _blocks(turn: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    role = turn.get("role")
    if role == "user":
        return "user", [{"type": "text", "text": turn["text"]}]
    if role == "tool":
        return "user", [{"type": "tool_result", "tool_use_id": turn["call_id"],
                         "content": json.dumps(turn["result"], ensure_ascii=False, sort_keys=True)}]
    if role == "model":
        out = [{"type": "text", "text": turn["text"]}] if turn.get("text") else []
        return "assistant", out + [{"type": "tool_use", "id": c.get("call_id"), "name": c["name"], "input": c["arguments"]}
                                   for c in turn.get("function_calls", [])]
    raise ValueError(f"unsupported turn role: {role!r}")


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


class AnthropicMessagesProvider:
    name = "anthropic"
    endpoint = "messages"

    def __init__(self, transport: Transport | None = None, timeout_s: float = 120.0, url: str = URL):
        self.transport = transport or urllib_transport
        self.timeout_s = timeout_s
        self.url = url

    @staticmethod
    def build_body(req: GenerationRequest) -> dict[str, Any]:
        if req.max_output_tokens is None:
            raise ValueError("Anthropic Messages requires max_output_tokens (max_tokens)")
        if req.previous_interaction_id:
            raise ValueError("Anthropic Messages is stateless: resend history instead of previous_interaction_id")
        messages: list[dict[str, Any]] = []
        for t in req.input:
            role, blocks = _blocks(t)
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += blocks
            else:
                messages.append({"role": role, "content": blocks})
        body: dict[str, Any] = {"model": req.model, "max_tokens": req.max_output_tokens,
                                "system": req.system_instruction, "messages": messages}
        if req.tools:
            body["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in req.tools]
        oc: dict[str, Any] = {}
        if req.reasoning_effort:
            oc["effort"] = req.reasoning_effort
        if req.response_schema is not None:
            oc["format"] = {"type": "json_schema", "schema": req.response_schema}
        if oc:
            body["output_config"] = oc
        return body

    def generate(self, req: GenerationRequest) -> ProviderResult:
        key = api_key("anthropic")
        if not key:
            return infra_result(req, self.name, self.endpoint, "credential ANTHROPIC_API_KEY is not set", sdk_version=SDK)
        reply = post_json(self.transport, self.url, {"x-api-key": key, "anthropic-version": API_VERSION},
                          self.build_body(req), self.timeout_s)
        fail = reply.failure()
        if fail:
            return infra_result(req, self.name, self.endpoint, fail, reply, SDK)
        data = reply.data
        texts: list[str] = []
        calls: list[FunctionCall] = []
        for b in data.get("content") or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                texts.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                inp = b.get("input")
                calls.append(FunctionCall(call_id=b.get("id"), name=str(b.get("name")),
                                          arguments=inp if isinstance(inp, dict) else {"_unparsed": inp}))
        stop = data.get("stop_reason")
        error = None
        if stop == "refusal":
            status = ProviderStatus.refusal
        elif stop in ("max_tokens", "model_context_window_exceeded"):
            status = ProviderStatus.incomplete
        elif stop in ("end_turn", "tool_use", "stop_sequence"):
            status = ProviderStatus.ok
        else:
            status, error = ProviderStatus.infra_error, f"stop_reason {stop!r}"
        u = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        usage = Usage(input_tokens=_int(u.get("input_tokens")), output_tokens=_int(u.get("output_tokens")),
                      cached_tokens=_int(u.get("cache_read_input_tokens")))
        return ProviderResult(role=req.role, provider=self.name, endpoint=self.endpoint, status=status,
                              text="".join(texts) if texts else None, function_calls=calls, usage=usage,
                              request_id=reply.request_id, model_requested=req.model,
                              model_returned=data.get("model") if isinstance(data.get("model"), str) else None,
                              sdk_version=SDK, latency_s=reply.latency_s, error=error)
