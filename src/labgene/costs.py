"""Cost accounting and caps (§8.2, plan T07). Every physical operation is a CostEvent."""
from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .config import CostCaps
from .contracts import CostEvent, CostSink, Usage


def _locked(fn):
    @functools.wraps(fn)
    def w(self, *a, **k):
        with self._lock:
            return fn(self, *a, **k)
    return w


class CapExceeded(Exception):
    """Raised before a call that could exceed an approved cap. Never after the fact."""


@dataclass
class Reservation:
    model: str
    input_tokens: int
    output_tokens: int
    usd: float | None


@dataclass
class CostGuard:
    """Reserve worst-case usage before each call, settle with actual usage after.
    Unknown usage keeps the reservation (conservative). Unknown prices -> usd_guaranteed=False.
    on_change(totals) persists cumulative totals after every reserve/settle; open reservations are counted as spent
    there, so a hard kill mid-call can only over-count on resume (never under-count). Thread-safe: the conditions of a
    set may run in parallel (U26) against one guard."""
    caps: CostCaps
    enforce: bool = True
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    search_calls: int = 0
    prior_wall_s: float = 0.0
    usd_guaranteed: bool = True
    on_change: Callable[[dict[str, Any]], None] | None = None
    _started: float = field(default_factory=time.monotonic)
    _open: list[Reservation] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def _usd(self, model: str, inp: int, out: int) -> float | None:
        p = self.caps.prices.get(model)
        return None if p is None else (inp * p.input_usd_per_mtok + out * p.output_usd_per_mtok) / 1e6

    def wall_s(self) -> float:
        return self.prior_wall_s + time.monotonic() - self._started

    @_locked
    def reserve(self, model: str, est_input_tokens: int, max_output_tokens: int) -> Reservation:
        usd = self._usd(model, est_input_tokens, max_output_tokens)
        if usd is None:
            self.usd_guaranteed = False
        r = Reservation(model, est_input_tokens, max_output_tokens, usd)
        if self.enforce:
            pend_in = sum(x.input_tokens for x in self._open)
            pend_out = sum(x.output_tokens for x in self._open)
            pend_usd = sum(x.usd or 0.0 for x in self._open)
            c = self.caps
            checks = [
                ("max_calls", c.max_calls, self.calls + len(self._open) + 1),
                ("max_input_tokens", c.max_input_tokens, self.input_tokens + pend_in + est_input_tokens),
                ("max_output_tokens", c.max_output_tokens, self.output_tokens + pend_out + max_output_tokens),
                ("max_usd", c.max_usd, self.usd + pend_usd + (usd or 0.0)),
                ("max_wall_s", c.max_wall_s, self.wall_s()),
            ]
            for name, cap, value in checks:
                if cap is not None and value > cap:
                    raise CapExceeded(f"{name}: {value} would exceed approved cap {cap}")
        self._open.append(r)
        self._persist()
        return r

    @_locked
    def reserve_search(self) -> None:
        """One paid search/fetch request (no token usage). Checked before the request."""
        if self.enforce and self.caps.max_search_calls is not None and self.search_calls + 1 > self.caps.max_search_calls:
            raise CapExceeded(f"max_search_calls: {self.search_calls + 1} would exceed approved cap "
                              f"{self.caps.max_search_calls}")
        self.search_calls += 1
        self._persist()

    @_locked
    def totals(self) -> dict[str, Any]:
        """Cumulative spend with open reservations counted as spent (what a resume must assume)."""
        return {"calls": self.calls + len(self._open),
                "input_tokens": self.input_tokens + sum(x.input_tokens for x in self._open),
                "output_tokens": self.output_tokens + sum(x.output_tokens for x in self._open),
                "usd": self.usd + sum(x.usd or 0.0 for x in self._open), "search_calls": self.search_calls,
                "wall_s": self.wall_s(), "usd_guaranteed": self.usd_guaranteed}

    @_locked
    def restore(self, t: dict[str, Any]) -> None:
        self.calls, self.input_tokens, self.output_tokens = t["calls"], t["input_tokens"], t["output_tokens"]
        self.usd, self.search_calls, self.prior_wall_s = t["usd"], t.get("search_calls", 0), t["wall_s"]
        self.usd_guaranteed = self.usd_guaranteed and t.get("usd_guaranteed", True)
        self._started = time.monotonic()

    def _persist(self) -> None:
        if self.on_change is not None:
            self.on_change(self.totals())

    @_locked
    def settle(self, r: Reservation, usage: Usage) -> None:
        self._open.remove(r)
        inp = usage.input_tokens if usage.input_tokens is not None else r.input_tokens
        out_reported = None if usage.output_tokens is None else usage.output_tokens + (usage.reasoning_tokens or 0)
        out = out_reported if out_reported is not None else r.output_tokens
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        usd = self._usd(r.model, inp, out)
        self.usd += usd or 0.0
        self._persist()

    @_locked
    def summary(self) -> dict[str, Any]:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "usd": round(self.usd, 6), "usd_guaranteed": self.usd_guaranteed, "search_calls": self.search_calls,
                "wall_s": round(self.wall_s(), 3), "caps": self.caps.model_dump(mode="json", exclude={"prices"})}


@dataclass(frozen=True)
class CallContext:
    """Passed into every component call so costs land in the ledger with the right scope/phase."""
    sink: CostSink
    phase: str = "runtime"
    scope_key: str | None = None
    episode_id: str | None = None
    action_id: str | None = None
    guard: CostGuard | None = None

    def child(self, **kw: Any) -> "CallContext":
        return replace(self, **kw)

    def emit(self, **kw: Any) -> None:
        kw.setdefault("phase", self.phase)
        kw.setdefault("scope_key", self.scope_key)
        kw.setdefault("episode_id", self.episode_id)
        kw.setdefault("action_id", self.action_id)
        self.sink(CostEvent(**kw))


def null_context(phase: str = "runtime") -> CallContext:
    """For tests/tools that do not persist costs."""
    return CallContext(sink=lambda e: None, phase=phase)
