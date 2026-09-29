"""Per-answer metric arithmetic (spec §4). Inputs come from the judge (None = unavailable) or from stored records;
None in gives None out, never a default."""
from __future__ import annotations

import math
from typing import Any


def claim_metrics(claims: list[dict[str, str]] | None, verdicts: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Counts, fact ratio, faithfulness = supported / fact claims, contradiction = contradicted / fact claims."""
    if claims is None:
        return {"n_claims": None, "n_facts": None, "n_supported": None, "n_contradicted": None,
                "fact_ratio": None, "faithfulness": None, "contradiction": None}
    n_facts = sum(c["type"] == "fact" for c in claims)
    out = {"n_claims": len(claims), "n_facts": n_facts, "fact_ratio": n_facts / len(claims) if claims else None}
    if not n_facts or verdicts is None:
        return {**out, "n_supported": None, "n_contradicted": None, "faithfulness": None, "contradiction": None}
    sup = sum(v["verdict"] == "supported" for v in verdicts)
    con = sum(v["verdict"] == "contradicted" for v in verdicts)
    return {**out, "n_supported": sup, "n_contradicted": con, "faithfulness": sup / n_facts,
            "contradiction": con / n_facts}


def citation_metrics(verdicts: list[dict[str, Any]] | None, cites: list[frozenset[str]],
                     cited: set[str]) -> dict[str, Any]:
    """ALCE-style on answer-level citations (spec §4): recall = fact claims with a supporting passage the answer
    cites / fact claims; precision = cited ids that support at least one claim / cited ids."""
    if verdicts is None:
        return {"citation_recall": None, "citation_precision": None}
    backing = [set().union(*(cites[i] for i in v["support"])) for v in verdicts]
    used = set().union(*backing)
    return {"citation_recall": sum(bool(b & cited) for b in backing) / len(verdicts) if verdicts else None,
            "citation_precision": len(cited & used) / len(cited) if cited else None}


def context_precision(useful: list[bool] | None) -> dict[str, Any]:
    """RAGAS rank-weighted precision sum_k(precision@k * v_k) / sum_k v_k (0 when nothing is useful) and the plain share
    of useful cards. None when nothing was judged or the judge was unavailable."""
    if not useful:
        return {"context_precision": None, "context_precision_plain": None}
    hits, total = 0, 0.0
    for k, u in enumerate(useful, start=1):
        if u:
            hits += 1
            total += hits / k
    return {"context_precision": total / hits if hits else 0.0, "context_precision_plain": hits / len(useful)}


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def answer_relevancy(question: list[float], generated: list[list[float]], noncommittal: bool) -> float:
    """RAGAS answer relevancy: mean cosine of the original question with the questions generated from the answer;
    0 for a noncommittal answer."""
    if noncommittal or not generated:
        return 0.0
    return sum(cosine(question, g) for g in generated) / len(generated)


ISSUE_KINDS = ("empty_field", "uncited_delivery", "no_longer_approved", "missing_observation", "value_mismatch",
               "invalid_candidate", "other")
_MARKERS = (("uncited_delivery", "was not delivered"), ("no_longer_approved", "is no longer approved"),
            ("missing_observation", "does not exist"), ("value_mismatch", "does not match the recorded value"))


def issue_kind(issue: str) -> str:
    """Category of one validate_advisor_response() message (labgene/knowledge/cards.py)."""
    if issue.endswith(" is empty"):
        return "empty_field"
    if issue.startswith("candidate "):
        return "invalid_candidate"
    return next((kind for kind, marker in _MARKERS if marker in issue), "other")


def issue_counts(issues: list[str]) -> dict[str, int]:
    out = dict.fromkeys(ISSUE_KINDS, 0)
    for i in issues:
        out[issue_kind(i)] += 1
    return out
