"""Fault injection for forced-interruption tests (B04, B08).

LABGENE_FAULT="point" crashes the first time `point` is reached in this process;
"point@N" crashes on the N-th hit. Multiple points: comma-separated.
InjectedCrash derives from BaseException so `except Exception` handlers cannot swallow it,
which mimics a hard kill at that boundary.
"""
from __future__ import annotations

import os

_hits: dict[str, int] = {}


class InjectedCrash(BaseException):
    pass


def crash_point(name: str) -> None:
    spec = os.environ.get("LABGENE_FAULT")
    if not spec:
        return
    for item in spec.split(","):
        point, _, nth = item.strip().partition("@")
        if point != name:
            continue
        _hits[name] = _hits.get(name, 0) + 1
        if _hits[name] == int(nth or 1):
            raise InjectedCrash(f"injected crash at {name} (hit {_hits[name]})")


def reset() -> None:
    _hits.clear()
