"""Isolated simulator worker and its harness-side adapter (T02, spec §3.3).

Worker side runs in a separate env with NO pydantic, so this module imports only stdlib at import time:
    python -m labgene.simulators.worker --backend aldenv --source .envs/src/aldenv --process FastFast
Protocol (JSON lines on stdin/stdout): the worker first prints
    {"ready": true, "version": <descriptor>, "units": {metric: unit}}
then answers each request line (a validated parameter object) with {"results": {...}} or {"error": <type>}.
The descriptor (upstream commit + source-tree hash + env hash + backend settings) stays evaluator-side; model-facing DTOs carry only
opaque_version(descriptor), so the simulator_version never names the hidden model or its code.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

DEFAULT_MODULE = "labgene.simulators.worker"


def opaque_version(descriptor: str) -> str:
    """Public simulator_version for a worker descriptor (hides upstream identity, keeps exactness)."""
    return "w-" + hashlib.sha256(descriptor.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- worker side (stdlib only)

def git_head(repo: Path) -> str:
    """Commit of a git checkout, read from .git without calling git."""
    git = repo / ".git"
    head = (git / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    if (git / ref).exists():
        return (git / ref).read_text(encoding="utf-8").strip()
    for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
        if line.endswith(" " + ref):
            return line.split()[0]
    raise RuntimeError(f"cannot resolve {ref} in {repo}")


def files_sha256(folder: Path) -> str:
    """One hash over (relative path, sha256) of every file under folder, sorted; __pycache__ skipped.
    Catches an edited working tree that git HEAD alone would not."""
    # ponytail: hashes bytes as checked out (CRLF vs LF across machines -> version mismatch, never silent drift).
    files = {f.relative_to(folder).as_posix(): f for f in folder.rglob("*")
             if f.is_file() and "__pycache__" not in f.relative_to(folder).parts}
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}:{hashlib.sha256(files[rel].read_bytes()).hexdigest()}\n".encode("utf-8"))
    return h.hexdigest()


def env_sha256() -> str:
    """Interpreter build + every installed distribution (name==version) this worker can import."""
    from importlib import metadata
    dists = sorted({f"{(d.metadata['Name'] or '').lower()}=={d.version}" for d in metadata.distributions()})
    return hashlib.sha256("\n".join([sys.version, *dists]).encode("utf-8")).hexdigest()


def _pinned_import(src: Path, module_file: str) -> None:
    if not Path(module_file).resolve().is_relative_to(src):
        raise RuntimeError(f"upstream module imported from outside the pinned checkout {src}")


Backend = tuple[str, dict[str, str], Callable[[dict[str, Any]], dict[str, float]]]


def _aldenv(a: argparse.Namespace) -> Backend:
    """aldenv steady-state ALD process (mechanistic, BSD-3-Clause). Fresh process object per call,
    noise=None, fixed round_to, as in the research smoke run."""
    src = Path(a.source).resolve()
    sys.path.insert(0, str(src / "src"))
    from aldenv.envs import steadystate
    _pinned_import(src, steadystate.__file__)
    cls = getattr(steadystate, a.process)
    descriptor = (f"aldenv@{git_head(src)};src_sha256={files_sha256(src / 'src' / 'aldenv')};env_sha256={env_sha256()};"
                  f"process={a.process};noise=None;round_to={a.round_to}")

    def evaluate(p: dict[str, Any]) -> dict[str, float]:
        return {"gpc": float(cls(noise=None, round_to=a.round_to)(float(p["t1"]), float(p["t2"])))}
    return descriptor, {"gpc": "angstrom/cycle"}, evaluate


def _summit(a: argparse.Namespace) -> Backend:
    """Summit ReizmanSuzukiEmulator (pretrained 5-ANN ensemble mean, MIT). Upstream code unmodified."""
    src = Path(a.source).resolve()
    sys.path.insert(0, str(src))
    import pandas as pd
    from summit.benchmarks import experimental_emulator as ee
    from summit.utils.dataset import DataSet
    _pinned_import(src, ee.__file__)
    name = f"reizman_suzuki_case_{a.case}"
    descriptor = (f"summit@{git_head(src)};src_sha256={files_sha256(src / 'summit')};env_sha256={env_sha256()};"
                  f"model={name};model_files_sha256={files_sha256(src / 'summit' / 'benchmarks' / 'models' / name)};"
                  "clip=True;ensemble=mean")
    emu = ee.get_pretrained_reizman_suzuki_emulator(case=a.case)

    def evaluate(p: dict[str, Any]) -> dict[str, float]:
        cond = DataSet.from_df(pd.DataFrame({"catalyst": [p["catalyst"]], "t_res": [float(p["residence_time"])],
                                             "temperature": [float(p["temperature"])],
                                             "catalyst_loading": [float(p["catalyst_loading"])]}))
        r = emu.run_experiments(cond)
        return {"yield": float(r[("yld", "DATA")].iloc[0]), "ton": float(r[("ton", "DATA")].iloc[0])}
    return descriptor, {"yield": "%", "ton": "1"}, evaluate


BACKENDS: dict[str, Callable[[argparse.Namespace], Backend]] = {"aldenv": _aldenv, "summit": _summit}


def serve(backend: Backend, inp, out) -> None:
    descriptor, units, evaluate = backend
    out.write(json.dumps({"ready": True, "version": descriptor, "units": units}) + "\n")
    out.flush()
    for line in inp:
        try:
            reply: dict[str, Any] = {"results": evaluate(json.loads(line))}
        except Exception as e:  # inputs arrive validated: any failure here is infrastructure
            traceback.print_exc(file=sys.stderr)
            reply = {"error": type(e).__name__}
        out.write(json.dumps(reply) + "\n")
        out.flush()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m " + DEFAULT_MODULE)
    ap.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--source", required=True, help="pinned upstream git checkout")
    ap.add_argument("--process", default="FastFast", help="aldenv: steady-state process class")
    ap.add_argument("--round-to", type=int, default=6, help="aldenv: decimal places")
    ap.add_argument("--case", type=int, default=1, help="summit: ReizmanSuzuki case")
    a = ap.parse_args(argv)
    proto, sys.stdout = sys.stdout, sys.stderr   # upstream prints must not corrupt the protocol stream
    serve(BACKENDS[a.backend](a), sys.stdin, proto)


# ---------------------------------------------------------------- harness side

class _WorkerFailure(Exception):
    pass


def _pump(stream, q: queue.Queue) -> None:
    with stream:
        for line in stream:
            q.put(line)
    q.put(None)


class WorkerSimulator:
    """SimulatorAdapter over a persistent isolated worker process; restarted after any failure.
    Timeout / crash / malformed output -> SimulatorInfraError with a generic message (worker stderr is
    inherited, never copied into the error). Deterministic in-memory cache keyed by
    (simulator_id, simulator_version, canonical params); a hit is still a charged action (harness charges)."""

    def __init__(self, simulator_id: str, simulator_version: str, *, python: str | Path, root: str | Path,
                 metrics: list[str], args: list[str] | None = None, module: str = DEFAULT_MODULE,
                 timeout_s: float = 60.0):
        self.simulator_id = simulator_id
        self.simulator_version = simulator_version
        self.descriptor: str | None = None   # evaluator-side only (manifests, validation reports)
        self._cmd = [str(python), "-m", module, *(args or [])]
        self._root = Path(root)
        self._metrics = list(metrics)
        self._timeout = timeout_s
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._units: dict[str, str] = {}
        # ponytail: unbounded per-adapter cache, not thread-safe; build one adapter per set scope (B10).
        self._cache: dict[tuple[str, str, str], dict[str, float]] = {}

    def evaluate(self, parameters: dict[str, Any]):
        from ..contracts import canonical_json          # lazy: the worker env has no pydantic
        from .base import SimOutput, SimulatorInfraError
        t0 = time.perf_counter()
        key = (self.simulator_id, self.simulator_version, canonical_json(parameters))
        hit = key in self._cache
        if not hit:
            try:
                self._cache[key] = self._request(parameters)
            except _WorkerFailure as e:
                self.close()
                raise SimulatorInfraError(f"simulator {self.simulator_id}: {e}") from e
        return SimOutput(results=dict(self._cache[key]), units=dict(self._units), simulator_id=self.simulator_id,
                         simulator_version=self.simulator_version, elapsed_s=time.perf_counter() - t0, cache_hit=hit)

    def close(self) -> None:
        p, self._proc = self._proc, None
        if p is not None:
            p.kill()
            p.wait()
            with contextlib.suppress(OSError):   # unflushed request into a dead pipe: nothing left to deliver
                p.stdin.close()

    def _start(self) -> None:
        # Untrusted upstream code gets no credentials; PYTHONPATH is exactly <root>/src.
        env = {k: v for k, v in os.environ.items() if not k.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET"))}
        env.update(PYTHONPATH=str(self._root / "src"), PYTHONIOENCODING="utf-8")
        try:
            self._proc = subprocess.Popen(self._cmd, cwd=self._root, env=env, stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE, text=True, encoding="utf-8")
        except OSError as e:
            raise _WorkerFailure(f"cannot start worker ({type(e).__name__})") from e
        self._lines = queue.Queue()
        threading.Thread(target=_pump, args=(self._proc.stdout, self._lines), daemon=True).start()
        hello = self._read()
        descriptor = hello.get("version")
        if hello.get("ready") is not True or not isinstance(descriptor, str):
            raise _WorkerFailure("worker handshake malformed")
        if opaque_version(descriptor) != self.simulator_version:
            raise _WorkerFailure(f"worker model version {opaque_version(descriptor)} != task simulator_version "
                                 f"{self.simulator_version}")
        units = hello.get("units")
        if not isinstance(units, dict) or any(m not in units for m in self._metrics):
            raise _WorkerFailure("worker does not report every task metric")
        self.descriptor = descriptor
        self._units = {m: str(units[m]) for m in self._metrics}

    def _read(self) -> dict[str, Any]:
        try:
            line = self._lines.get(timeout=self._timeout)
        except queue.Empty:
            raise _WorkerFailure(f"worker timed out after {self._timeout}s") from None
        if line is None:
            raise _WorkerFailure(f"worker exited (code {self._proc.poll()})")
        try:
            msg = json.loads(line)
        except ValueError:
            raise _WorkerFailure("malformed worker output") from None
        if not isinstance(msg, dict):
            raise _WorkerFailure("malformed worker output")
        return msg

    def _request(self, parameters: dict[str, Any]) -> dict[str, float]:
        if self._proc is None or self._proc.poll() is not None:
            self._start()
        try:
            self._proc.stdin.write(json.dumps(parameters) + "\n")
            self._proc.stdin.flush()
        except OSError as e:
            raise _WorkerFailure(f"worker pipe closed ({type(e).__name__})") from e
        reply = self._read()
        if "error" in reply:   # detail stays in worker stderr; never echoed into harness-visible errors
            raise _WorkerFailure("worker failed to evaluate the request")
        results = reply.get("results")
        if not isinstance(results, dict):
            raise _WorkerFailure("malformed worker output")
        out = {}
        for m in self._metrics:
            v = results.get(m)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise _WorkerFailure(f"worker returned no finite value for {m}")
            out[m] = float(v)
        return out


if __name__ == "__main__":
    main()
