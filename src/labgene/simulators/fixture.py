"""Deterministic synthetic simulators for offline contract checks.
Artificial functions: NOT scientific models, NOT evidence of research ability or product value."""
from __future__ import annotations

import math
import time
from typing import Any, Callable

from .base import SimOutput


def ridge(p: dict[str, Any]) -> dict[str, float]:
    """Correlated 2-D ridge, max 95 % at temperature=83 degC, time=37 min."""
    a = (p["temperature"] - 83.0) / 20.0
    b = (p["time"] - 37.0) / 15.0
    return {"yield": 95.0 * math.exp(-(a * a + b * b - 0.6 * a * b))}


_CAT_ACTIVITY = {"A": 0.55, "B": 0.8, "C": 1.0, "D": 0.4}


def catalyst(p: dict[str, Any]) -> dict[str, float]:
    """Categorical catalyst + loading/temperature/residence time; yield % and TON (both maximize)."""
    k = 0.004 * math.exp(0.035 * (p["temperature"] - 70.0)) * p["loading"]
    conversion = 1.0 - math.exp(-k * p["residence_time"])
    decomposition = math.exp(-0.5 * ((p["temperature"] - 85.0) / 30.0) ** 2)
    y = 100.0 * _CAT_ACTIVITY[p["catalyst"]] * conversion * decomposition
    return {"yield": y, "ton": 0.9 * y / p["loading"]}


FUNCTIONS: dict[str, Callable[[dict[str, Any]], dict[str, float]]] = {"ridge": ridge, "catalyst": catalyst}
UNITS = {"ridge": {"yield": "%"}, "catalyst": {"yield": "%", "ton": "1"}}


class FixtureSimulator:
    def __init__(self, simulator_id: str, simulator_version: str, function: str):
        self.simulator_id = simulator_id
        self.simulator_version = simulator_version
        self._fn = FUNCTIONS[function]
        self._units = UNITS[function]

    def close(self) -> None:
        pass

    def evaluate(self, parameters: dict[str, Any]) -> SimOutput:
        t0 = time.perf_counter()
        results = {k: round(v, 6) for k, v in self._fn(parameters).items()}
        return SimOutput(results=results, units=self._units, simulator_id=self.simulator_id,
                         simulator_version=self.simulator_version, elapsed_s=time.perf_counter() - t0)
