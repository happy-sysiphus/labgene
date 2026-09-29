"""Aggregation and report files (spec §6). Public rows carry only allowed fields: a hidden threshold or a raw simulator
output never reaches items.jsonl, report.json or report.md (split_sim sends raw values to the private dir)."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

CONDITIONS = ("baseline", "product")
PRIMARY = "faithfulness"
JUDGED = ("faithfulness", "contradiction", "fact_ratio", "context_precision", "context_precision_plain",
          "answer_relevancy", "citation_recall", "citation_precision")
CODED = ("sim_success", "sim_score", "sim_improved", "has_issue")
COUNTS = ("n_claims", "n_facts", "n_supported", "n_contradicted", "n_cited", "reconstruction_match", "unavailable")
LIMITATIONS = (
    "One judge pass per item by one LLM; the product's KG relations were extracted by a model of the same family "
    "(Claude Opus), which may favour KG cards.",
    "The conditions followed different trajectories and questions: an unpaired comparison within one set rep; with "
    "few episodes the intervals are rough.",
    "The judge decides what is a fact or an inference and which cards are useful.",
    "The baseline cites its whole initial text as one source (initial:text), so the citation metrics differ in "
    "structure; compare the conditions on faithfulness.",
    "Context precision exists for the product only (the baseline has no retrieval).",
    "Material opened with tools during a consultation is not rebuilt (counted as a reconstruction mismatch).",
    "Recommendation correctness is measured inside the emulator (U24) and does not say whether the researcher "
    "followed the recommendation.",
)


def split_sim(action_id: str, sim: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """(public fields, private record) of one first-candidate simulation (spec §6)."""
    if sim is None:
        return {"sim_status": None, "sim_success": None, "sim_score": None, "sim_improved": None}, None
    return ({"sim_status": sim["status"], "sim_success": float(sim["success"]), "sim_score": sim["score"],
             "sim_improved": float(sim["improved"])},
            {"action_id": action_id, "results": sim["results"], "best_prior": sim["best_prior"], "score": sim["score"]})


def build_items(records: list[Any], code: dict[str, dict[str, Any]], judged: dict[str, dict[str, Any]],
                sample: set[str]) -> list[dict[str, Any]]:
    """One public row per consult (records: consults.ConsultRecord): identity, coded fields of every consult, and the
    judged metrics of the judged sample (None elsewhere)."""
    out = []
    for r in records:
        c, j = code[r.action_id], judged.get(r.action_id, {})
        out.append({"action_id": r.action_id, "condition": r.condition, "task_id": r.task_id,
                    "episode_id": r.episode_id, "ok": r.ok, "sampled": r.action_id in sample,
                    "judged": r.action_id in judged, "issues": c["issues"],
                    "has_issue": float(any(c["issues"].values())) if r.ok else None,
                    **{k: c[k] for k in ("sim_status", "sim_success", "sim_score", "sim_improved")},
                    **{k: j.get(k) for k in (*JUDGED, *COUNTS)}})
    return out


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def by_episode(items: list[dict[str, Any]], condition: str, metric: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for i in items:
        if i["condition"] == condition and i["ok"] and i.get(metric) is not None:
            out.setdefault(i["episode_id"], []).append(float(i[metric]))
    return out


def cluster_ci(a: dict[str, list[float]], b: dict[str, list[float]], draws: int, seed: int) -> list[float] | None:
    """95 % percentile interval of mean(b) - mean(a): each draw resamples each condition's episodes (clusters) with
    replacement and pools their values. None when a condition has no value."""
    ka, kb = [k for k, v in a.items() if v], [k for k, v in b.items() if v]
    if not ka or not kb:
        return None
    rng, diffs = random.Random(seed), []
    for _ in range(draws):
        va = [x for k in rng.choices(ka, k=len(ka)) for x in a[k]]
        vb = [x for k in rng.choices(kb, k=len(kb)) for x in b[k]]
        diffs.append(sum(vb) / len(vb) - sum(va) / len(va))
    diffs.sort()
    return [diffs[math.floor(0.025 * (draws - 1))], diffs[math.ceil(0.975 * (draws - 1))]]


def summarize(items: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    """Condition means and product - baseline differences with episode-cluster intervals (spec §6). Judged metrics come
    from the judged sample, coded ones from every complete consult."""
    s: dict[str, Any] = {"primary": PRIMARY, "conditions": {}, "differences": {}, "per_task": {}}
    for c in CONDITIONS:
        rows = [i for i in items if i["condition"] == c]
        judged = [i for i in rows if i["judged"]]
        scored = [i for i in judged if i["n_supported"] is not None]
        facts = sum(i["n_facts"] for i in scored)
        s["conditions"][c] = {
            "consults": len(rows), "partial": sum(not i["ok"] for i in rows), "judged": len(judged),
            "judge_unavailable": sum(bool(i["unavailable"]) for i in judged),
            "no_fact_claims": sum(i["n_facts"] == 0 for i in judged),
            "no_citation": sum(i["n_cited"] == 0 for i in judged),
            "reconstruction_mismatch": sum(i["reconstruction_match"] is False for i in judged),
            "faithfulness_micro": sum(i["n_supported"] for i in scored) / facts if facts else None,
            "issues": {k: sum(i["issues"][k] for i in rows if i["ok"]) for k in (rows[0]["issues"] if rows else {})},
            "sim_status": {k: sum(i["sim_status"] == k for i in rows) for k in ("ok", "no_candidate",
                                                                               "invalid_candidate")}}
    for m in (*JUDGED, *CODED):
        a, b = by_episode(items, "baseline", m), by_episode(items, "product", m)
        ma, mb = mean([x for v in a.values() for x in v]), mean([x for v in b.values() for x in v])
        s["differences"][m] = {"baseline": ma, "product": mb, "diff": None if ma is None or mb is None else mb - ma,
                               "ci95": cluster_ci(a, b, draws, seed)}
    for t in sorted({i["task_id"] for i in items}):
        rows = [i for i in items if i["task_id"] == t and i["ok"]]
        s["per_task"][t] = {c: {
            "consults": sum(i["condition"] == c for i in rows),
            "judged": sum(i["condition"] == c and i["judged"] for i in rows),
            "faithfulness": mean([i["faithfulness"] for i in rows if i["condition"] == c and i["faithfulness"] is not None]),
            "sim_score": mean([i["sim_score"] for i in rows if i["condition"] == c and i["sim_score"] is not None])}
            for c in CONDITIONS}
    return s


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render_md(manifest: dict[str, Any], s: dict[str, Any], costs: dict[str, Any]) -> str:
    j = manifest["judge"]
    lines = [f"# Consultation RAG evaluation of run {manifest['target_run']}", "",
             f"- target run finished: {'yes' if manifest['target_complete'] else 'NO (partial run)'}",
             f"- judge: {j['provider']}/{j['model']}, effort {j.get('reasoning_effort')}, prompts "
             f"{manifest['prompt_version']}",
             f"- sample: {manifest['sample_size']} consults, at most {manifest['sample_per_task']} per condition and "
             f"task; intervals: {manifest['bootstrap_draws']} bootstrap draws over episodes"]
    if j["provider"] == "fixture":
        lines.append("- FIXTURE JUDGE: an offline contract check, not a measurement")
    lines += ["", "## Metrics (product - baseline)", "",
              "| metric | baseline | product | difference | 95 % interval |", "|---|---|---|---|---|"]
    for m, d in s["differences"].items():
        ci = "n/a" if d["ci95"] is None else f"[{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
        name = f"{m} (primary)" if m == s["primary"] else m
        lines.append(f"| {name} | {_f(d['baseline'])} | {_f(d['product'])} | {_f(d['diff'])} | {ci} |")
    b, p = s["conditions"]["baseline"], s["conditions"]["product"]
    lines += ["", "## Counts", "", "| | baseline | product |", "|---|---|---|"]
    for k in ("consults", "partial", "judged", "judge_unavailable", "no_fact_claims", "no_citation",
              "reconstruction_mismatch"):
        lines.append(f"| {k} | {b[k]} | {p[k]} |")
    lines.append(f"| faithfulness over pooled claims | {_f(b['faithfulness_micro'])} | {_f(p['faithfulness_micro'])} |")
    for k in ("ok", "no_candidate", "invalid_candidate"):
        lines.append(f"| first candidate {k} | {b['sim_status'][k]} | {p['sim_status'][k]} |")
    for k in b["issues"]:
        lines.append(f"| validation issue {k} | {b['issues'][k]} | {p['issues'].get(k, 0)} |")
    lines += ["", "## Per task (descriptive)", "",
              "| task | condition | consults | judged | faithfulness | sim score |", "|---|---|---|---|---|---|"]
    for t, by in s["per_task"].items():
        for c, v in by.items():
            lines.append(f"| {t} | {c} | {v['consults']} | {v['judged']} | {_f(v['faithfulness'])} | "
                         f"{_f(v['sim_score'])} |")
    lines += ["", "## Costs", "", f"- judge: {json.dumps(costs['judge'])}", f"- support: {json.dumps(costs['support'])}",
              "", "## Limitations (spec §8)", "", *[f"- {x}" for x in LIMITATIONS]]
    return "\n".join(lines) + "\n"


def write_reports(out: Path, manifest: dict[str, Any], items: list[dict[str, Any]], summary: dict[str, Any],
                  costs: dict[str, Any]) -> dict[str, str]:
    (out / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
                                     encoding="utf-8")
    (out / "report.json").write_text(json.dumps({"manifest": manifest, "summary": summary, "costs": costs}, indent=1,
                                                ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(render_md(manifest, summary, costs), encoding="utf-8")
    return {"report": str(out / "report.md"), "report_json": str(out / "report.json"), "items": str(out / "items.jsonl")}
