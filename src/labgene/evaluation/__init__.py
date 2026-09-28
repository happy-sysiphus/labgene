"""Evaluation support (T08): researcher qualification, leakage-gate validation, retrieval/answer dev validation.
Evaluator-side only. Offline fixture runs of these tools are contract checks (development_only)."""
from __future__ import annotations

import json
import os
from pathlib import Path


def pin_hash(lock_path: str | Path, key: str, digest: str) -> None:
    """Pre-fixed evaluation input (criteria, cases/queries/tasks, rules): the first run pins `digest` under `key`
    BEFORE anything is scored; a later run with another input under the same key is refused, so nothing is changed
    after results (publish a new version)."""
    p = Path(lock_path)
    pins = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if key in pins:
        if pins[key] != digest:
            raise ValueError(f"{key}: the evaluation input differs from the one pinned before results "
                             f"({pins[key][:12]} vs {digest[:12]}); publish a new version instead of changing it")
        return
    pins[key] = digest
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(pins, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)
