"""Claude Code headless adapter (Claude Max subscription) for single-shot INTERNAL roles: kg_extractor and
internal_summarizer (user decision U11, 2026-09-28: claude-opus-5-5 at effort max). The subscription is used only
through Claude Code itself, never through the API or the Agent SDK.

Checked 2026-09-28 with Claude Code 2.1.283 (`claude --help`, `claude auth status`; no model call yet):
- `claude -p --model <m> --effort <e> --system-prompt <text> --tools "" --safe-mode --restricted
   --strict-mcp-config --disable-slash-commands --no-session-persistence --output-format json [--json-schema <s>]`,
   user turn on stdin, cwd = an empty temporary directory.
- NOT --bare: bare mode never reads the OAuth login. --safe-mode turns off CLAUDE.md, skills, plugins, hooks and MCP
  servers; --tools "" removes every built-in tool; --restricted ignores the user/project/local settings files.
- `claude auth status` prints JSON; "loggedIn": true with "authMethod": "claude.ai" is the subscription login.
- The subprocess gets no ANTHROPIC_* variable (an API key there would bill per token) and none of the calling
  session's CLAUDECODE / CLAUDE_CODE_* variables (a nested session refuses to start or attaches to its parent).
Result JSON per the headless docs (confirmed by the canary run before any live use): result, structured_output
(with --json-schema), is_error, session_id, usage{input_tokens, cache_read_input_tokens,
cache_creation_input_tokens, output_tokens}, modelUsage{<model id>: {outputTokens, ...}}. The model that ran is the
modelUsage entry with the most output (helper models never mask a swap; echoes_model=True). Claude Code's safety
fallback (live canary 2026-09-28: the requested model ran with 0 output tokens, then claude-opus-5 answered) is a
refusal of the requested model (user decision U17): the other model's output is discarded, never returned. A schema request without
structured_output is incomplete (never silently parsed from prose). Usage.input_tokens counts cached input too;
output_tokens includes thinking. The child runs with DISABLE_AUTOUPDATER=1 and each call records the CLI version.
Claude Code has no per-request output cap: GenerationRequest.max_output_tokens only sizes the cost reservation.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from ..contracts import ProviderResult, ProviderStatus, Usage
from .base import GenerationRequest

_LIMIT_WORDS = ("usage limit", "rate limit", "429", "too many requests", "quota", "limit reached")
_KEEP_ENV = ("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR")   # where the subscription login lives

Runner = Callable[[list[str], str, Path, float], "tuple[int, str, str]"]


def child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.upper() in _KEEP_ENV
           or not k.upper().startswith(("ANTHROPIC_", "CLAUDECODE", "CLAUDE_CODE_"))}
    return {**env, "DISABLE_AUTOUPDATER": "1"}   # no self-update mid-run: flags were checked on one version


def _run(args: list[str], stdin: str, cwd: Path, timeout_s: float) -> tuple[int, str, str]:
    exe = os.environ.get("LABGENE_CLAUDE_BIN") or shutil.which(args[0]) or args[0]
    p = subprocess.run([exe, *args[1:]], input=stdin, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout_s, env=child_env())
    return p.returncode, p.stdout, p.stderr


def cli_problem(claude_bin: str = "claude") -> str | None:
    """Preflight (local command, no model call): None when Claude Code is logged in with a Claude subscription.
    The account e-mail is never read. The version is not pinned (Claude Code updates itself); each call records it."""
    if shutil.which(claude_bin) is None and not os.environ.get("LABGENE_CLAUDE_BIN"):
        return f"{claude_bin} CLI not found on PATH"
    try:
        rc, out, _ = _run([claude_bin, "auth", "status"], "", Path.cwd(), 30)
        d = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return f"claude auth status failed ({type(e).__name__})"
    ok = isinstance(d, dict) and d.get("loggedIn") is True and d.get("authMethod") == "claude.ai"
    return None if ok else "claude is not logged in with a Claude subscription (`claude auth login`)"


class ClaudeCodeProvider:
    name = "claude_code"
    endpoint = "claude_cli"
    echoes_model = True
    billing = "subscription"

    def __init__(self, claude_bin: str = "claude", timeout_s: float = 600.0, runner: Runner | None = None):
        self.claude_bin, self.timeout_s, self.runner = claude_bin, timeout_s, runner or _run

    def version(self) -> str | None:
        """Per call: Claude Code updates itself outside our runs."""
        if self.runner is not _run:
            return None
        try:
            return _run([self.claude_bin, "--version"], "", Path.cwd(), 30)[1].strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def command(self, req: GenerationRequest) -> list[str]:
        if req.tools:
            raise ValueError("the claude_code provider serves single-shot roles only: harness tools are not supported")
        if any(t.get("role") != "user" for t in req.input):
            raise ValueError("the claude_code provider accepts user turns only (no model/tool turns)")
        args = [self.claude_bin, "-p", "--model", req.model, "--system-prompt", req.system_instruction,
                "--tools", "", "--safe-mode", "--restricted", "--strict-mcp-config", "--disable-slash-commands",
                "--no-session-persistence", "--output-format", "json"]
        if req.reasoning_effort:
            args += ["--effort", req.reasoning_effort]
        if req.response_schema is not None:
            args += ["--json-schema", json.dumps(req.response_schema)]
        return args

    def generate(self, req: GenerationRequest) -> ProviderResult:
        args = self.command(req)
        base = dict(role=req.role, provider=self.name, endpoint=self.endpoint, model_requested=req.model,
                    sdk_version=self.version())
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="labgene-claude-") as d:
            try:
                rc, stdout, stderr = self.runner(args, "\n\n".join(str(t["text"]) for t in req.input), Path(d),
                                                 self.timeout_s)
            except subprocess.TimeoutExpired:
                return ProviderResult(**base, status=ProviderStatus.infra_error, latency_s=time.perf_counter() - t0,
                                      error=f"claude -p timed out after {self.timeout_s}s")
            except OSError as e:
                return ProviderResult(**base, status=ProviderStatus.infra_error, latency_s=time.perf_counter() - t0,
                                      error=f"claude -p could not start ({type(e).__name__})")
        latency = time.perf_counter() - t0
        d = _result_json(stdout)
        if d is None:
            msg = (stderr.strip().splitlines() or [f"exit {rc}, no JSON result"])[-1]
            return ProviderResult(**base, status=ProviderStatus.infra_error, latency_s=latency, error=_error(msg))
        u = d.get("usage") or {}
        cache_read, created = u.get("cache_read_input_tokens") or 0, u.get("cache_creation_input_tokens") or 0
        usage = Usage(input_tokens=None if u.get("input_tokens") is None else u["input_tokens"] + cache_read + created,
                      cached_tokens=cache_read, output_tokens=u.get("output_tokens"))
        ids = dict(usage=usage, request_id=d.get("session_id"), latency_s=latency)
        if rc != 0 or d.get("is_error"):
            return ProviderResult(**base, status=ProviderStatus.infra_error, **ids,
                                  error=_error(str(d.get("result") or d.get("subtype") or f"exit {rc}")))
        mu = d.get("modelUsage") if isinstance(d.get("modelUsage"), dict) else {}
        ran = max(mu, key=lambda m: (mu[m] or {}).get("outputTokens") or 0, default=None)   # the model that answered
        # every model that ran is kept on the record whenever the answer is not purely the requested model
        seen = None if list(mu) == [req.model] else "modelUsage " + json.dumps(
            {m: {k: (v or {}).get(k) for k in ("inputTokens", "outputTokens")} for m, v in mu.items()})
        if ran not in (None, req.model) and req.model in mu and not (mu[req.model] or {}).get("outputTokens"):
            return ProviderResult(**base, status=ProviderStatus.refusal, model_returned=req.model, **ids,
                                  error=f"safety fallback: {req.model} produced no output and {ran} answered; "
                                        f"that answer was discarded; {seen}")
        if req.response_schema is not None:
            so = d.get("structured_output")
            text, why = (json.dumps(so, ensure_ascii=False), None) if so is not None else \
                (None, "claude -p returned no structured output")
        else:
            text = d.get("result") or None
            why = None if text else "claude -p returned no result"
        note = "; ".join(x for x in (why, seen) if x) or None
        return ProviderResult(**base, status=ProviderStatus.ok if text else ProviderStatus.incomplete, text=text,
                              model_returned=ran, error=note, **ids)


def _result_json(stdout: str) -> dict | None:
    """The single result object: the whole stdout, or its last line when warnings precede it."""
    for cand in (stdout, (stdout.strip().splitlines() or [""])[-1]):
        try:
            d = json.loads(cand)
        except ValueError:
            continue
        if isinstance(d, dict):
            return d
    return None


def _error(msg: str) -> str:
    return ("usage/rate limit: " if any(w in msg.lower() for w in _LIMIT_WORDS) else "") + msg[:300]
