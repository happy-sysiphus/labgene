"""Re-gate a knowledge state under another gate identity (user decisions U39, U40).

A state was gated against one plan's answer bundles and set scope; another plan reuses it (the same parsed chunks,
extracted relations, vectors, fetched pages and derived items) after every approved item has been checked again
under the new bundles and set scope, exactly as the build and the runtime check it: the document identity check
(URL, DOI, arXiv id, fuzzy title, hidden asset paths) for corpus documents and web sources, and the content gate for
every exposed text. Nothing is ever loosened: an item stays visible only if the new checks allow it. The build rules
apply: an identity match or a blocked chunk blocks its whole document or web source, a held chunk leaves the
indexes, and every descendant of a withdrawn item (relations, vectors, search caches) is invalidated.
Idempotent: a re-run finds its verdicts in the gate cache under the new identity; a gate error leaves the state
'in_progress' (a store refuses to open it) and the next re-run finishes it.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..contracts import AnswerBundle, GateStatus, canonical_json, sha256_text
from ..costs import CallContext
from .base import LeakageChecker, RawHit
from .build import BASELINE_INITIAL_ID
from .gate import KnowledgeInfraError, cache_key, identity_blocked
from .gated import hit_titles
from .parse import parse_document
from .store import INDEXED_KINDS, KnowledgeStore, exposure, gate_identity

DERIVED_KINDS = ("relation", "summary", "card", "caption")
FOLLOWS_PARENTS = ("cache",)   # search-result caches: views of approved sources; they fall with any of them


def _state(con: sqlite3.Connection, key: str) -> str | None:
    row = con.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return None if row is None else row[0]


def regate_state(state_dir: str | Path, checker: LeakageChecker, bundles: list[AnswerBundle], set_scope: str,
                 ctx: CallContext, *, baseline_source: str, corpus_dir: str | Path | None = None, workers: int = 1,
                 seed: tuple[str | Path, str] | None = None, **store_kwargs: Any) -> dict[str, Any]:
    """Re-check every approved artifact of state_dir under (bundles, set_scope, checker); returns the report.
    baseline_source: the gate context source of the baseline initial text (as the build used it). corpus_dir: where
    the corpus documents are (their front matter feeds the identity check; a changed or missing file fails closed).
    workers > 1 runs the checker calls concurrently to warm the cache first (_prefetch); decisions are made as with 1.
    seed = (knowledge.sqlite, set scope) of a state re-gated under the same bundles and checker for another set: its
    verdicts are reused for bundles that apply to every set (set_scope '*'), because the checker's input never
    contains the set scope, so it is the same request (_prefetch)."""
    state_dir = Path(state_dir)
    db = state_dir / "knowledge.sqlite"
    if not db.exists():
        raise FileNotFoundError(f"no knowledge state in {state_dir}")
    new = gate_identity(bundles, set_scope, checker)
    with closing(sqlite3.connect(db)) as con, con:
        old, status = _state(con, "gate_identity"), _state(con, "regate_status")
        if old is None:
            raise ValueError(f"{state_dir} has no gate identity")
        if old == new and status == "done":
            return json.loads(_state(con, "regate_report") or "{}")
        if old == new and status is None:
            raise ValueError(f"{state_dir} was built under this gate identity already; nothing to re-gate")
        if old != new:
            if status == "done":             # re-gated before for another plan: start a new pass from its identity
                con.execute("DELETE FROM state WHERE key IN ('regate_from', 'regate_withdrawn', 'regate_report')")
            con.execute("INSERT OR IGNORE INTO state VALUES ('regate_from', ?)", (old,))
            con.execute("INSERT OR REPLACE INTO state VALUES ('regate_status', 'in_progress')")
            con.execute("UPDATE state SET value=? WHERE key='gate_identity'", (new,))
    store = KnowledgeStore(state_dir, checker, bundles, set_scope, regating=True, **store_kwargs)
    try:
        withdrawn = json.loads(store.get_state("regate_withdrawn") or "[]")

        def withdraw(row: sqlite3.Row, st: GateStatus, why: str) -> None:
            withdrawn.append({"id": row["id"], "status": st.value, "applying": True})   # intent first (crash-safe)
            store.set_state("regate_withdrawn", canonical_json(withdrawn))
            withdrawn[-1] = {**_withdraw(store, row, st), "reason": why}
            store.set_state("regate_withdrawn", canonical_json(withdrawn))

        # 1) identity: corpus documents and web sources, against the new bundles (no model call)
        for (sid,) in store.db.execute("SELECT id FROM artifacts WHERE kind='source' AND valid=1 AND "
                                       "(gate_status IS NULL OR gate_status='allow') ORDER BY id").fetchall():
            row = store.get(sid)
            if sid != BASELINE_INITIAL_ID and _identity_blocked(store, row, corpus_dir):
                withdraw(row, GateStatus.block, "identity")
        # 2) content: every approved exposed text, as the build and the runtime gated it
        rows = [aid for (aid,) in store.db.execute(
            "SELECT id FROM artifacts WHERE valid=1 AND gate_status='allow' "
            "ORDER BY CASE kind WHEN 'chunk' THEN 0 WHEN 'source' THEN 1 ELSE 2 END, id").fetchall()]
        inputs = {aid: _gate_input(store, store.get(aid), baseline_source) for aid in rows}
        seeded = 0
        if workers > 1 or seed:
            seeded = _prefetch(store, [x for x in inputs.values() if x is not None], ctx, workers, seed)
        rechecked = errors = 0
        for aid in rows:
            row = store.visible(aid)
            if row is None or inputs[aid] is None:   # withdrawn earlier in this pass / falls with its parents
                continue
            st = store.gate(*inputs[aid], ctx)
            if st == GateStatus.error:
                errors += 1
                continue
            rechecked += 1
            if st != GateStatus.allow:
                withdraw(row, st, "content")
        # 3) held chunks stay held (never promoted), but a block under the new bundles blocks their whole document,
        #    as it would in a fresh build
        for (aid,) in store.db.execute("SELECT id FROM artifacts WHERE kind='chunk' AND valid=1 AND "
                                       "gate_status='hold' ORDER BY id").fetchall():
            row = store.get(aid)
            if row["valid"] != 1:                    # its document was blocked meanwhile
                continue
            st = store.gate(*_gate_input(store, row, baseline_source), ctx)
            if st == GateStatus.error:
                errors += 1
            elif st == GateStatus.block:
                withdraw(row, st, "content (held chunk)")
        if errors:
            raise KnowledgeInfraError(f"leakage gate unavailable for {errors} artifact(s); the state stays "
                                      "in_progress: re-run the re-gate")
        # 4) index sweep: an id that is not visible never stays indexed (covers a kill between a withdrawal's
        #    database update and its index removal, which a re-run would not revisit)
        store.bm25.remove([i for i in list(store.bm25.docs) if store.visible(i) is None])
        if store.vectors is not None:
            store.vectors.remove([i for i in list(store.vectors.vectors) if store.visible(i) is None])
        withdrawn = [{**w, "applying": False, "completed_by": "sweep"}
                     if w.get("applying") and store.visible(w["id"]) is None else w for w in withdrawn]
        report = {"from_identity": store.get_state("regate_from"), "to_identity": new, "set_scope": set_scope,
                  "answer_bundle_versions": sorted(f"{b.bundle_id}@{b.version}" for b in bundles),
                  "checker": checker.checker_id, "policy_version": checker.policy_version,
                  "rechecked_last_pass": rechecked, "seeded_verdicts_last_pass": seeded,
                  "seeded_from": None if seed is None else {"set_scope": seed[1], "state": Path(seed[0]).as_posix()},
                  "withdrawn": list({w["id"]: w for w in withdrawn}.values()),   # a re-run's record replaces an intent
                  "still_allowed": dict(store.db.execute(
                      "SELECT kind, count(*) FROM artifacts WHERE valid=1 AND gate_status='allow' GROUP BY kind"
                  ).fetchall())}
        store.set_state("regate_report", canonical_json(report))
        store.set_state("regate_status", "done")
        return report
    finally:
        store.close()


def _identity_blocked(store: KnowledgeStore, row: sqlite3.Row, corpus_dir: str | Path | None) -> bool:
    """The identity check the build (corpus documents) or the gated search (web sources) ran at admission."""
    meta = store.meta(row)
    if row["id"].startswith("doc:"):
        if corpus_dir is None:
            raise ValueError("re-gating corpus documents needs corpus_dir (their front matter feeds the identity check)")
        path = Path(corpus_dir) / meta["file"]
        text = path.read_text(encoding="utf-8") if path.exists() else None
        if text is None or sha256_text(text) != meta.get("source_hash"):
            raise ValueError(f"{path} is missing or changed since it was ingested: cannot re-check its identity")
        doc = parse_document(text, doc_id=path.stem)
        return identity_blocked(doc.meta.get("url"), [doc.meta.get("title", "")], doc.meta, store.bundles)
    if row["id"].startswith("web:"):
        # the quarantined hits, as gated.py checked them: the search hit at admission and the fetch hit of any page
        # opened from it (url, every reported title, provider metadata such as doi / arXiv id)
        raws = [p for p in _parents(store, row["id"]) if p.startswith("raw:")]
        raws += [p for c in _children(store, row["id"]) for p in _parents(store, c) if p.startswith("raw:")]
        if not raws:
            raise ValueError(f"web source {row['id']} has no quarantined hit: cannot re-check its identity")
        hits = [RawHit.model_validate_json(store.get(r)["text"]) for r in dict.fromkeys(raws)]
        return any(identity_blocked(h.url, hit_titles(h), h.metadata, store.bundles) for h in hits)
    raise ValueError(f"unexpected source {row['id']}: refusing to re-gate it")


def _gate_input(store: KnowledgeStore, row: sqlite3.Row, baseline_source: str) -> tuple[str, dict[str, Any]] | None:
    """The exact text and check context the build or the runtime gated this artifact with (knowledge/build.py,
    knowledge/gated.py, store.py); None for artifacts that only follow their parents."""
    kind, meta = row["kind"], store.meta(row)
    if row["id"] == BASELINE_INITIAL_ID:
        return row["text"], {"source": baseline_source, "origin": "baseline_initial"}
    if kind == "source" and row["id"].startswith("web:"):          # gated search: title + snippet + url
        return row["text"], {"source": meta["url"], "origin": "web_search"}
    if kind == "chunk":
        docs = [p for p in _parents(store, row["id"]) if p.startswith(("doc:", "web:"))]
        if len(docs) != 1:
            raise ValueError(f"chunk {row['id']} has {len(docs)} parent documents")
        if docs[0].startswith("web:"):                              # a fetched page
            return exposure(row["text"], meta), {"source": meta["url"], "origin": "web_fetch"}
        dm = store.meta(store.get(docs[0]))
        return exposure(row["text"], meta), {"source": dm.get("url") or f"corpus:{dm['file']}", "origin": "corpus"}
    if kind in DERIVED_KINDS:
        rel = store.db.execute("SELECT source_ids FROM relations WHERE id=?", (row["id"],)).fetchone()
        parents = json.loads(rel[0]) if rel else _parents(store, row["id"])
        return exposure(row["text"], meta), {"source": "derived:" + ",".join(parents), "origin": "derived",
                                             "kind": kind}
    if kind in FOLLOWS_PARENTS:
        return None
    raise ValueError(f"unexpected approved artifact {row['id']} of kind {kind!r}: refusing to re-gate it")


def _parents(store: KnowledgeStore, aid: str) -> list[str]:
    return sorted(r[0] for r in store.db.execute("SELECT parent FROM lineage WHERE child=?", (aid,)))


def _children(store: KnowledgeStore, aid: str) -> list[str]:
    return sorted(r[0] for r in store.db.execute("SELECT child FROM lineage WHERE parent=?", (aid,)))


def _withdraw(store: KnowledgeStore, row: sqlite3.Row, st: GateStatus) -> dict[str, Any]:
    """Apply the build rule for a verdict that is no longer allow; returns what was withdrawn. Crash-safe order: the
    descendants and index entries go first and the item's own status last, so a kill in between leaves the item
    visible to the re-run, which withdraws it again (every step is idempotent)."""
    aid, kind = row["id"], row["kind"]
    if kind == "source" and aid != BASELINE_INITIAL_ID or kind == "chunk" and st == GateStatus.block:
        # an identity match, a blocked web source or any blocked chunk blocks the whole document / web source
        doc = aid if kind == "source" else [p for p in _parents(store, aid) if p.startswith(("doc:", "web:"))][0]
        gone = store.invalidate(doc)                      # the doc/source and all its descendants; indexes
        with store.db:
            store.db.execute("UPDATE artifacts SET gate_status=? WHERE id=?",
                             (st.value if kind == "source" else GateStatus.block.value, doc))
        return {"id": aid, "kind": kind, "status": st.value, "document_blocked": doc, "invalidated": len(gone)}
    gone = sorted({i for c in _children(store, aid) for i in store.invalidate(c)})
    if kind in INDEXED_KINDS:                              # a held chunk/summary leaves the indexes
        store.bm25.remove([aid])
        if store.vectors is not None:
            store.vectors.remove([aid])
    with store.db:
        store.db.execute("UPDATE artifacts SET gate_status=? WHERE id=?", (st.value, aid))
    return {"id": aid, "kind": kind, "status": st.value, "invalidated": len(gone)}


def _prefetch(store: KnowledgeStore, inputs: list[tuple[str, dict[str, Any]]], ctx: CallContext,
              workers: int, seed: tuple[str | Path, str] | None = None) -> int:
    """Warm the gate cache with the verdicts store.gate would get: first from the seed state (the same request
    answered under another set scope, bundles with set_scope '*' only), then by running the remaining checker calls
    `workers` at a time. store.gate then decides every item from the same cache keys, so the verdicts, their
    combination and the withdrawal order are unchanged. Errors are not cached (store.gate checks those items again).
    Returns the number of seeded verdicts."""
    todo, seen, seeded = [], set(), 0
    src = sqlite3.connect(f"file:{Path(seed[0]).as_posix()}?mode=ro", uri=True) if seed else None
    try:
        for text, context in inputs:
            for b in store.bundles:
                key = cache_key(text, b, store.set_scope, store.checker, context)
                if key in seen or store.db.execute("SELECT 1 FROM gate_cache WHERE cache_key=?", (key,)).fetchone():
                    continue
                seen.add(key)
                row = None
                if src is not None and b.set_scope == "*":
                    row = src.execute("SELECT status, policy_version, checker, bundle FROM gate_cache "
                                      "WHERE cache_key=?", (cache_key(text, b, seed[1], store.checker, context),)
                                      ).fetchone()
                if row is not None:
                    with store.db:
                        store.db.execute("INSERT OR REPLACE INTO gate_cache VALUES (?,?,?,?,?,?)",
                                         (key, *row, datetime.now(timezone.utc).isoformat(timespec="seconds")))
                    seeded += 1
                else:
                    todo.append((key, text, context, b))
    finally:
        if src is not None:
            src.close()
    if workers <= 1 or not todo:
        return seeded
    lock = threading.Lock()

    def sink(e: Any) -> None:
        with lock:
            ctx.sink(e)
    tctx = replace(ctx, sink=sink)
    with ThreadPoolExecutor(workers, thread_name_prefix="labgene-regate") as ex:
        results = ex.map(lambda t: store.checker.check(t[1], t[3], t[2], tctx), todo)
        for (key, _, _, b), d in zip(todo, results):       # cache writes stay on this thread (one sqlite connection)
            if d.status != GateStatus.error:
                with store.db:
                    store.db.execute("INSERT OR REPLACE INTO gate_cache VALUES (?,?,?,?,?,?)",
                                     (key, d.status.value, d.policy_version, d.checker, f"{b.bundle_id}@{b.version}",
                                      datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return seeded


def adopt_checker(state_dir: str | Path, bundles: list[AnswerBundle], set_scope: str, checker: LeakageChecker,
                  note: str) -> dict[str, Any]:
    """U42: keep every approval of a state as it was decided and let `checker` decide everything from now on (the
    user's choice, after `checker` passed the same gate validation). The state's gate identity moves to the new
    checker and the change is recorded in `checker_history`; nothing is re-checked, loosened or withdrawn."""
    db = Path(state_dir) / "knowledge.sqlite"
    new = gate_identity(bundles, set_scope, checker)
    with closing(sqlite3.connect(db)) as con, con:
        old = _state(con, "gate_identity")
        if old is None:
            raise ValueError(f"{state_dir} has no gate identity")
        if _state(con, "regate_status") == "in_progress":
            raise ValueError(f"{state_dir} is being re-gated; finish the re-gate before switching the checker")
        if old == new:
            return {"unchanged": True, "identity": new}
        history = json.loads(_state(con, "checker_history") or "[]")
        history.append({"from_identity": old, "to_identity": new, "checker": checker.checker_id,
                        "policy_version": checker.policy_version, "note": note,
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        con.execute("INSERT OR REPLACE INTO state VALUES ('checker_history', ?)", (canonical_json(history),))
        con.execute("UPDATE state SET value=? WHERE key='gate_identity'", (new,))
    return history[-1]


def accept_previous_records(state_dir: str | Path, bundles: list[AnswerBundle], set_scope: str,
                            checker: LeakageChecker, sources: list[tuple[str | Path, str, Any]], note: str, *,
                            baseline_source: str, corpus_dir: str | Path | None = None,
                            **store_kwargs: Any) -> dict[str, Any]:
    """U43: move a state to (bundles, set_scope, checker) without new model checks (the user's decision: the answer
    bundles block the same documents, so the approvals decided so far stand). What is known is still applied: the
    deterministic identity check under the new bundles, and every verdict already available for an approved item
    under a new bundle in `sources` ((knowledge.sqlite, set scope, checker or an object with its checker_id and
    policy_version)): any block or hold withdraws the item by the build rules. Later checks use `checker`."""
    state_dir = Path(state_dir)
    new = gate_identity(bundles, set_scope, checker)
    with closing(sqlite3.connect(state_dir / "knowledge.sqlite")) as con, con:
        old = _state(con, "gate_identity")
        if old is None:
            raise ValueError(f"{state_dir} has no gate identity")
        con.execute("INSERT OR IGNORE INTO state VALUES ('regate_from', ?)", (old,))
        con.execute("INSERT OR REPLACE INTO state VALUES ('regate_status', 'in_progress')")
        con.execute("UPDATE state SET value=? WHERE key='gate_identity'", (new,))
    store = KnowledgeStore(state_dir, checker, bundles, set_scope, regating=True, **store_kwargs)
    srcs = [(sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True), scope, chk)
            for db, scope, chk in sources]
    try:
        withdrawn = json.loads(store.get_state("regate_withdrawn") or "[]")

        def withdraw(row: sqlite3.Row, st: GateStatus, why: str) -> None:
            withdrawn.append({"id": row["id"], "status": st.value, "applying": True})
            store.set_state("regate_withdrawn", canonical_json(withdrawn))
            withdrawn[-1] = {**_withdraw(store, row, st), "reason": why}
            store.set_state("regate_withdrawn", canonical_json(withdrawn))

        def known(text: str, context: dict[str, Any]) -> list[GateStatus]:
            out = []
            for b in bundles:
                for con, scope, chk in srcs:
                    r = con.execute("SELECT status FROM gate_cache WHERE cache_key=?",
                                    (cache_key(text, b, scope, chk, context),)).fetchone()
                    if r is not None:
                        out.append(GateStatus(r[0]))
                        break
            return out

        for (sid,) in store.db.execute("SELECT id FROM artifacts WHERE kind='source' AND valid=1 AND "
                                       "(gate_status IS NULL OR gate_status='allow') ORDER BY id").fetchall():
            row = store.get(sid)
            if sid != BASELINE_INITIAL_ID and _identity_blocked(store, row, corpus_dir):
                withdraw(row, GateStatus.block, "identity")
        kept = with_verdicts = 0
        for (aid,) in store.db.execute(
                "SELECT id FROM artifacts WHERE (valid=1 AND gate_status='allow') OR (kind='chunk' AND valid=1 AND "
                "gate_status='hold') ORDER BY CASE kind WHEN 'chunk' THEN 0 WHEN 'source' THEN 1 ELSE 2 END, id"
        ).fetchall():
            row = store.get(aid)
            if row["valid"] != 1:
                continue
            inp = _gate_input(store, row, baseline_source)
            if inp is None:
                continue
            bad = [v for v in known(*inp) if v != GateStatus.allow]
            if row["gate_status"] == "hold":        # a held chunk: only a known block matters (blocks its document)
                if GateStatus.block in bad:
                    withdraw(row, GateStatus.block, "known verdict (held chunk)")
                continue
            if bad:
                withdraw(row, GateStatus.block if GateStatus.block in bad else GateStatus.hold, "known verdict")
            else:
                kept += 1
                with_verdicts += len(known(*inp)) == len(bundles)
        store.bm25.remove([i for i in list(store.bm25.docs) if store.visible(i) is None])
        if store.vectors is not None:
            store.vectors.remove([i for i in list(store.vectors.vectors) if store.visible(i) is None])
        history = json.loads(store.get_state("checker_history") or "[]")
        history.append({"from_identity": store.get_state("regate_from"), "to_identity": new,
                         "checker": checker.checker_id, "policy_version": checker.policy_version, "note": note,
                         "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        report = {"mode": "previous records (U43)", "to_identity": new, "set_scope": set_scope,
                  "answer_bundle_versions": sorted(f"{b.bundle_id}@{b.version}" for b in bundles),
                  "checker": checker.checker_id, "kept": kept, "kept_with_every_bundle_checked": with_verdicts,
                  "withdrawn": list({w["id"]: w for w in withdrawn}.values()),
                  "still_allowed": dict(store.db.execute(
                      "SELECT kind, count(*) FROM artifacts WHERE valid=1 AND gate_status='allow' GROUP BY kind"
                  ).fetchall())}
        store.set_state("checker_history", canonical_json(history))
        store.set_state("regate_report", canonical_json(report))
        store.set_state("consults_regated", new)       # stored consult answers keep their earlier decisions too
        store.set_state("regate_status", "done")
        return report
    finally:
        for con, _, _ in srcs:
            con.close()
        store.close()
