"""Pretraining-recall probe (user decision U23, pilot step U28): does the researcher model already know the answer
paper's reported optimum for a main-evaluation task?

One costed call per task through the researcher role config (the same model and effort as the experiment). The
prompt names the paper and the substrate pair and asks for the reported optimum as JSON (or "unknown"). The answer
is compared evaluator-side with the private reference point (validity_evidence.reference_points[0]):
recalled = the ligand matches AND at least 2 of 3 continuous conditions are within tolerance
(temperature +-10 degC, residence time +-2 min, loading +-0.3 mol%). Fixed before any probe runs (analysis plan v1).
Any recalled task -> recall_positive -> full anonymisation before freeze (U23). Answers stay in the run dir.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..contracts import ProviderStatus, PublicTask
from ..costs import CallContext
from ..providers.base import GenerationRequest, LLMProvider
from ..providers.call import call_llm

PROBE_VERSION = "recall-probe-v1"
TOLERANCE = {"temperature_c": 10.0, "residence_time_min": 2.0, "loading_mol_percent": 0.3}
LIGANDS = ("xantphos", "xphos", "sphos", "ruphos", "pcy3", "pph3", "ptbu3")
SYSTEM = ("Answer from your own knowledge of the chemistry literature, without guessing. If you do not know the "
          "reported values, say so by setting known=false.")
SCHEMA = {"type": "object", "required": ["known"], "properties": {
    "known": {"type": "boolean"}, "precatalyst_or_ligand": {"type": "string"},
    "residence_time_min": {"type": "number"}, "temperature_c": {"type": "number"},
    "loading_mol_percent": {"type": "number"}, "yield_percent": {"type": "number"}, "ton": {"type": "number"}}}


def question(halide: str, boron: str) -> str:
    return (f"Reizman, Wang, Buchwald and Jensen (Reaction Chemistry & Engineering, 2016) optimized Suzuki-Miyaura "
            f"cross-couplings in an automated droplet-flow system over the palladacycle precatalyst/ligand, the "
            f"residence time (1-10 min), the temperature (30-110 degC) and the catalyst loading (0.5-2.5 mol%). For the "
            f"coupling of {halide} with {boron}, which optimal conditions did they report? Reply as JSON with known, "
            f"precatalyst_or_ligand, residence_time_min, temperature_c, loading_mol_percent, yield_percent and ton.")


def substrates(task: PublicTask) -> tuple[str, str]:
    m = re.search(r"coupling of (.+?) with (.+?) \(", task.problem)
    if not m:
        raise ValueError(f"{task.task_id}: substrates not found in the public problem text")
    return m.group(1), m.group(2)


def _ligand(text: str) -> str | None:
    t = re.sub(r"[^a-z0-9]", "", (text or "").casefold().replace("t-bu", "tbu").replace("tert-butyl", "tbu"))
    return next((lig for lig in LIGANDS if lig in t), None)   # "xantphos" is checked before "xphos"


def judge(answer: dict[str, Any] | None, ref: dict[str, Any]) -> dict[str, Any]:
    """ref: reference parameters with the public catalyst name, residence_time in s, temperature, loading."""
    if not answer or not answer.get("known"):
        return {"recalled": False, "ligand_match": False, "continuous_matches": 0}
    lig = _ligand(answer.get("precatalyst_or_ligand", "")) == _ligand(ref["catalyst"])
    want = {"temperature_c": ref["temperature"], "residence_time_min": ref["residence_time"] / 60.0,
            "loading_mol_percent": ref["catalyst_loading"]}
    close = sum(isinstance(answer.get(k), (int, float)) and abs(float(answer[k]) - v) <= TOLERANCE[k]
                for k, v in want.items())
    return {"recalled": bool(lig and close >= 2), "ligand_match": lig, "continuous_matches": close}


def run_recall_probe(provider: LLMProvider, role_cfg, limits, tasks: list[PublicTask],
                     references: dict[str, dict[str, Any]], ctx: CallContext) -> dict[str, Any]:
    rows = []
    for t in tasks:
        halide, boron = substrates(t)
        req = GenerationRequest(role="researcher_recall_probe", model=role_cfg.model, system_instruction=SYSTEM,
                                input=[{"role": "user", "text": question(halide, boron)}], response_schema=SCHEMA,
                                reasoning_effort=role_cfg.reasoning_effort, thinking_level=role_cfg.thinking_level,
                                max_output_tokens=role_cfg.max_output_tokens)
        res = call_llm(provider, req, ctx, limits, role_cfg.allowed_returned_models)
        try:
            answer = json.loads(res.text) if res.status is ProviderStatus.ok and res.text else None
        except ValueError:
            answer = None
        answer = answer if isinstance(answer, dict) else None   # "unknown", true, a list: no answer
        rows.append({"task_id": t.task_id, "status": res.status.value, "answer": answer,
                     **judge(answer, references[t.task_id])})
    done = [r for r in rows if r["status"] == "ok"]
    verdict = ("recall_positive" if any(r["recalled"] for r in rows) else
               "no_recall" if len(done) == len(rows) else "incomplete")
    return {"probe_version": PROBE_VERSION, "tolerance": TOLERANCE, "verdict": verdict, "tasks": rows}
