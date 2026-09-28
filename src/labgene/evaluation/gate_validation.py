"""Leakage-gate validation (T08.4, spec §7.2-7.4). Evaluator-side.

Each case goes through the same order as the pipeline: identity / alternate-version check (knowledge.gate.
identity_blocked) first, then the content checker against EVERY answer bundle (most restrictive wins), with a
production check context (origin one of Origin, as knowledge/build.py, gated.py and store.py send it). The whole case
file (cases, criteria, answer bundles) is pinned by hash per version before any case is scored; a changed input under
the same version and lock is refused. Checker errors are counted per call (never masked by another bundle's block) and
never as allow. A live checker passes only on a reviewed case set (otherwise 'provisional'). Results carry case ids,
categories and verdicts only: no case text, no GateDecision.internal_detail. A development_only checker
(FixtureMarkerChecker) or case set measures nothing about the live gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from ..config import Strict, _load_yaml
from ..contracts import AnswerBundle, GateStatus, payload_hash
from ..costs import CallContext
from ..knowledge.base import LeakageChecker
from ..knowledge.gate import combine, identity_blocked
from . import pin_hash

Category = Literal["answer_paper", "alternate_version", "derived_from_answer", "hidden_table",
                   "citing_background_no_leak", "public_target", "self_observation_inference", "general_background",
                   "ambiguous"]
Verdict = Literal["allow", "block", "hold"]
# the check contexts production sends to LeakageChecker.check (knowledge/build.py, knowledge/gated.py, store.derive)
Origin = Literal["corpus", "baseline_initial", "web_search", "web_fetch", "context_expansion", "derived"]
ACTUAL = ("allow", "block", "hold", "error")


class GateCriteria(Strict):
    max_false_allow_on_block_cases: int
    max_false_block_on_allow_cases: int
    max_false_allow_on_hold_cases: int
    max_hold_rate: float                 # holds among cases expected allow|block / their number
    max_errors: int                      # checker calls that errored (per call, not per case)


class GateCase(Strict):
    case_id: str
    category: Category
    text: str                            # exactly what would be exposed (title and url are added from metadata)
    metadata: dict[str, str] = {}        # url, title, doi, arxiv_id; optional source / kind of the check context
    origin: Origin = "web_search"        # default: a search hit (title + text + url, source = url), as gated.py
    expected: Verdict
    note: str = ""


class GateCaseFile(Strict):
    version: str
    status: Literal["draft_unreviewed", "development_only", "reviewed"]
    answer_bundles: list[str]            # bundle_id@version the cases were written against
    criteria: GateCriteria
    cases: list[GateCase]

    @model_validator(mode="after")
    def _unique(self) -> "GateCaseFile":
        ids = [c.case_id for c in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate case_id")
        return self


def load_gate_cases(path: str | Path) -> GateCaseFile:
    return GateCaseFile.model_validate(_load_yaml(path))


def _verdict(case: GateCase, checker: LeakageChecker, bundles: list[AnswerBundle],
             ctx: CallContext) -> tuple[str, bool, int]:
    """(combined verdict, identity blocked, checker calls that errored)."""
    m = case.metadata
    if identity_blocked(m.get("url"), [m.get("title", "")], m, bundles):
        return "block", True, 0          # the content gate never sees an identity-blocked document
    exposed = "\n".join(x for x in (m.get("title"), case.text, m.get("url")) if x)
    context = {"source": m.get("source") or m.get("url"), "origin": case.origin,
               **({"kind": m["kind"]} if "kind" in m else {})}
    statuses = []
    for b in bundles:
        try:
            statuses.append(checker.check(exposed, b, context, ctx).status)
        except Exception:                # a failing checker is an error, never an allow
            statuses.append(GateStatus.error)
    return combine(statuses).value, False, statuses.count(GateStatus.error)


def run_gate_validation(checker: LeakageChecker, bundles: list[AnswerBundle], cases: GateCaseFile, ctx: CallContext,
                        input_lock: str | Path | None = None) -> dict[str, Any]:
    """Confusion matrix (expected x actual), per-case outcomes, counts and status against the pre-fixed criteria.
    input_lock pins the hash of the whole case file (cases + criteria + bundles; not version/status) per version
    before scoring; required for a non-development run. status: pass | fail | provisional (criteria met by a live
    checker on a case set that is not reviewed); passed = status == 'pass'."""
    if sorted(f"{b.bundle_id}@{b.version}" for b in bundles) != sorted(cases.answer_bundles):
        raise ValueError("the answer bundles differ from the ones this case set was written against")
    dev = bool(checker.development_only) or cases.status == "development_only"
    crit = cases.criteria.model_dump()
    input_hash = payload_hash(cases.model_dump(mode="json", exclude={"version", "status"}))
    if input_lock is not None:
        pin_hash(input_lock, f"gate_input:{cases.version}", input_hash)
    elif not dev:
        raise ValueError("a non-development gate validation must pin its input (input_lock)")
    confusion = {e: {a: 0 for a in ACTUAL} for e in ("allow", "block", "hold")}
    rows, call_errors, calls = [], 0, 0
    for c in cases.cases:
        actual, ident, errs = _verdict(c, checker, bundles, ctx)
        call_errors, calls = call_errors + errs, calls + (0 if ident else len(bundles))
        confusion[c.expected][actual] += 1
        rows.append({"case_id": c.case_id, "category": c.category, "origin": c.origin, "expected": c.expected,
                     "actual": actual, "identity_blocked": ident, "checker_errors": errs,
                     "correct": actual == c.expected})
    decidable = sum(sum(confusion[e].values()) for e in ("allow", "block"))
    counts = {"false_allow_on_block_cases": confusion["block"]["allow"],
              "false_block_on_allow_cases": confusion["allow"]["block"],
              "false_allow_on_hold_cases": confusion["hold"]["allow"],
              "hold_rate": round((confusion["allow"]["hold"] + confusion["block"]["hold"]) / decidable, 4)
              if decidable else 0.0,
              "errors": call_errors, "checker_calls": calls,
              "error_cases": sum(confusion[e]["error"] for e in confusion)}
    cr = cases.criteria
    failures = [f"{k}={counts[k]} > {lim}" for k, lim in (
        ("false_allow_on_block_cases", cr.max_false_allow_on_block_cases),
        ("false_block_on_allow_cases", cr.max_false_block_on_allow_cases),
        ("false_allow_on_hold_cases", cr.max_false_allow_on_hold_cases),
        ("hold_rate", cr.max_hold_rate), ("errors", cr.max_errors)) if counts[k] > lim]
    status = "fail" if failures else "pass" if dev or cases.status == "reviewed" else "provisional"
    label = ("development_only (fixture checker or development case set): a contract check, not live gate accuracy"
             if dev else "live checker on a reviewed case set" if cases.status == "reviewed"
             else f"live checker on a {cases.status} case set: provisional until the set is independently reviewed")
    return {"case_set_version": cases.version, "case_set_status": cases.status, "checker": checker.checker_id,
            "policy_version": checker.policy_version, "development_only": dev, "label": label,
            "answer_bundles": sorted(cases.answer_bundles), "criteria": crit, "criteria_hash": payload_hash(crit),
            "input_hash": input_hash, "confusion": confusion, "counts": counts, "cases": rows,
            "wrong_cases": [r["case_id"] for r in rows if not r["correct"]],
            "status": status, "passed": status == "pass", "failures": failures}
