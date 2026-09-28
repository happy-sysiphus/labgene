"""Simulator adapter contract (T02). Implementations: fixture.py, worker.py (isolated env)."""
from __future__ import annotations

from typing import Any, Protocol

from ..contracts import Frozen


class SimulatorInfraError(Exception):
    """Worker crash/timeout/protocol failure. Distinct from invalid input (never charged as invalid)."""


class SimOutput(Frozen):
    results: dict[str, float]
    units: dict[str, str]
    simulator_id: str
    simulator_version: str
    elapsed_s: float = 0.0
    cache_hit: bool = False


class SimulatorAdapter(Protocol):
    simulator_id: str
    simulator_version: str

    def evaluate(self, parameters: dict[str, Any]) -> SimOutput:
        """parameters are already normalized by validate_parameters. Deterministic for same
        (simulator_version, parameters). Raises SimulatorInfraError on worker failure."""
        ...

    def close(self) -> None:
        """Release worker processes. One adapter per MemoryScope; closed at scope end."""
        ...
