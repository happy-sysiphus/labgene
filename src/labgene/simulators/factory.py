"""Build the simulator adapter a profile's SimulatorConfig names for a task (T02)."""
from __future__ import annotations

from pathlib import Path

from ..config import SimulatorConfig
from ..contracts import PublicTask
from .base import SimulatorAdapter
from .fixture import UNITS, FixtureSimulator
from .worker import DEFAULT_MODULE, WorkerSimulator


def build_simulator(simulator_id: str, cfg: SimulatorConfig, task: PublicTask, root: str | Path) -> SimulatorAdapter:
    """Relative interpreter paths resolve against root; the worker runs with cwd=root, PYTHONPATH=root/src."""
    if simulator_id != task.simulator_id:
        raise ValueError(f"task {task.task_id} uses simulator {task.simulator_id}, not {simulator_id}")
    if cfg.backend == "fixture":
        # Never a stand-in for a real simulator: worker tasks carry opaque "w-" versions, and the fixture
        # function must produce exactly the task's metrics.
        if task.simulator_version.startswith("w-") or set(UNITS.get(cfg.fixture_function or "", {})) != {
                m.name for m in task.metrics}:
            raise ValueError(f"fixture backend {cfg.fixture_function!r} cannot serve task {task.task_id} "
                             f"(simulator {simulator_id}): fixtures never substitute for a real simulator")
        return FixtureSimulator(simulator_id, task.simulator_version, cfg.fixture_function)
    if not cfg.python:
        raise ValueError(f"simulators.{simulator_id}.python must name the isolated env interpreter")
    python = Path(cfg.python) if Path(cfg.python).is_absolute() else Path(root) / cfg.python
    return WorkerSimulator(simulator_id, task.simulator_version, python=python, root=root,
                           metrics=[m.name for m in task.metrics], args=cfg.worker_args,
                           module=cfg.worker_module or DEFAULT_MODULE, timeout_s=cfg.timeout_s)
