"""Whole state-dir snapshot / restore (T03, §11.3, §13.5, decisions I4).

Every *.sqlite is copied with the sqlite3 backup API (sees committed WAL pages; never a raw copy
of an open WAL db); every other file (index/*, cache/*, ...) is copied as is. The manifest records
per-file sha256 (integrity of the snapshot bytes), a LOGICAL hash for sqlite files (sorted dump of
user tables) and a combined logical hash of the whole directory.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from ..contracts import payload_hash

MANIFEST = "manifest.json"
_COMPANIONS = (".sqlite-wal", ".sqlite-shm", ".sqlite-journal")   # captured through the backup API
BUSY_TIMEOUT_S = 5.0   # a writer holding a sqlite lock makes snapshot fail after this, never hang


def sqlite_logical_hash(con: sqlite3.Connection, exclude: tuple[str, ...] = ()) -> str:
    """Content hash of all user tables (schema + sorted rows), independent of page layout/WAL."""
    h = hashlib.sha256()
    tables = con.execute("SELECT name, sql FROM sqlite_master WHERE type='table' "
                         "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    for name, sql in tables:
        if name in exclude:
            continue
        q = name.replace('"', '""')
        rows = sorted(json.dumps(list(r), default=bytes.hex, ensure_ascii=False)
                      for r in con.execute(f'SELECT * FROM "{q}"'))
        h.update(json.dumps([name, sql, rows], ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def _db_hash(path: Path) -> str:
    with closing(sqlite3.connect(path)) as con:
        return sqlite_logical_hash(con)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files(root: Path) -> list[Path]:
    if not root.is_dir():   # a mistyped path must not snapshot/hash as an empty state
        raise FileNotFoundError(f"state dir not found: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and not p.name.endswith(_COMPANIONS))


def _combined(logical: dict[str, str]) -> str:
    return payload_hash(logical)


def state_dir_hash(state_dir: str | Path, exclude: tuple[str, ...] = ()) -> str:
    """Combined logical hash of a live state dir (same definition as the manifest's); `exclude` = relative paths."""
    root = Path(state_dir)
    return _combined({rel: _db_hash(p) if p.suffix == ".sqlite" else _sha256(p) for p in _files(root)
                      if (rel := p.relative_to(root).as_posix()) not in exclude})


def snapshot_state(state_dir: str | Path, dest_dir: str | Path) -> dict[str, Any]:
    """Copy the whole state dir into <dest_dir>/state and write <dest_dir>/manifest.json.
    Writers must be quiesced between files (each sqlite file is transactionally consistent on its own)."""
    src, dest = Path(state_dir), Path(dest_dir)
    paths = _files(src)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"snapshot destination is not empty: {dest}")
    out = dest / "state"
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, str]] = {}
    for p in paths:
        rel = p.relative_to(src).as_posix()
        q = out / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".sqlite":
            with closing(sqlite3.connect(p, timeout=BUSY_TIMEOUT_S, isolation_level=None)) as s, \
                    closing(sqlite3.connect(q)) as d:
                # take the read lock under the busy timeout first: Connection.backup retries BUSY forever
                s.execute("BEGIN")
                s.execute("SELECT count(*) FROM sqlite_master")
                s.backup(d)
                d.execute("PRAGMA journal_mode=DELETE")   # self-contained snapshot file, no -wal
            files[rel] = {"kind": "sqlite", "logical_sha256": _db_hash(q)}
        else:
            shutil.copy2(p, q)
            files[rel] = {"kind": "file"}
        files[rel]["sha256"] = _sha256(q)
    manifest = {"format": 1, "files": files,
                "combined_sha256": _combined({k: v.get("logical_sha256", v["sha256"]) for k, v in files.items()})}
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


def restore_state(snapshot_dir: str | Path, state_dir: str | Path) -> dict[str, Any]:
    """Restore a snapshot into state_dir. A non-empty target is moved aside to
    <state_dir>.superseded-<n> (never deleted). Raises if the restored logical hash differs."""
    snap, target = Path(snapshot_dir), Path(state_dir)
    manifest = json.loads((snap / MANIFEST).read_text(encoding="utf-8"))
    src = snap / "state"
    actual = {p.relative_to(src).as_posix(): _sha256(p) for p in src.rglob("*") if p.is_file()}
    if actual != {k: v["sha256"] for k, v in manifest["files"].items()}:
        raise ValueError(f"snapshot files do not match manifest: {snap}")
    if target.exists() and any(target.iterdir()):
        n = 1
        while (aside := target.parent / f"{target.name}.superseded-{n}").exists():
            n += 1
        target.rename(aside)
    shutil.copytree(src, target, dirs_exist_ok=True)
    got = state_dir_hash(target)
    if got != manifest["combined_sha256"]:
        raise RuntimeError(f"restored state hash {got} != manifest {manifest['combined_sha256']}")
    return manifest
