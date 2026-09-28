"""Stage-1 action parsing (decisions I1). Stage 2 (experiment parameters) is simulators.validation:
an interpretable run_experiment with missing/bad parameters passes here and is charged as invalid."""
from __future__ import annotations

import json
import re
from typing import Any

from ..contracts import ActionKind

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)
_FENCE_MARK = re.compile(r"```(?:json)?", re.I)


class ActionParseError(Exception):
    """Protocol error: not an action; counts toward the consecutive-violation streak."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """json keeps only the last of repeated keys; a repeated action/args means more than one action."""
    keys = [k for k, _ in pairs]
    if keys.count("action") > 1 or keys.count("args") > 1:
        raise ActionParseError("multiple_actions", "submit exactly one action object")
    return dict(pairs)


_DECODER = json.JSONDecoder(object_pairs_hook=_object)


def parse_action(raw_text: str | None) -> tuple[ActionKind, dict[str, Any]]:
    """Exactly one {"action": ..., "args": {...}} object, bare or in one fenced block.
    Truncated JSON is never repaired."""
    text = (raw_text or "").strip()
    if m := _FENCE.fullmatch(text):
        text = m.group(1)
    try:
        obj, end = _DECODER.raw_decode(text)
    except json.JSONDecodeError as e:
        raise ActionParseError("invalid_json", f"not valid JSON ({e.msg})") from None
    if rest := text[end:].strip():
        try:   # a second (possibly fenced) value after the first one
            _DECODER.raw_decode(_FENCE_MARK.sub("", rest).strip())
        except json.JSONDecodeError:
            raise ActionParseError("invalid_json", "unexpected text after the action object") from None
        raise ActionParseError("multiple_actions", "submit exactly one action object")
    if isinstance(obj, list) or (isinstance(obj, dict) and (isinstance(obj.get("action"), list) or "actions" in obj)):
        raise ActionParseError("multiple_actions", "submit exactly one action object")
    if not isinstance(obj, dict):
        raise ActionParseError("invalid_json", "the action must be a JSON object")
    action = obj.get("action")
    if action not in ("consult", "run_experiment"):
        raise ActionParseError("unsupported_action",
                               f"unsupported action {str(action)[:40]!r}; use consult or run_experiment")
    args = obj["args"] if isinstance(obj.get("args"), dict) else {}
    if action == "consult":
        q = args.get("question")
        if not isinstance(q, str) or not q.strip():
            raise ActionParseError("missing_question", "consult needs a non-empty string args.question")
        return ActionKind.consult, {"question": q}
    return ActionKind.run_experiment, {"hypothesis": args.get("hypothesis"), "parameters": args.get("parameters")}
