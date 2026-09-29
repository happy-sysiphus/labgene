import json
from pathlib import Path

import yaml

from labgene.config import Limits, RoleModel
from labgene.contracts import ProviderResult, ProviderStatus, PublicTask
from labgene.costs import CallContext
from labgene.evaluation.recall_probe import judge, run_recall_probe, substrates

ROOT = Path(__file__).resolve().parents[2]
TASK = PublicTask.model_validate(yaml.safe_load((ROOT / "configs/tasks/suzuki_flow_01.yaml").read_text(encoding="utf-8")))
REF = {"catalyst": "Xantphos Pd G3", "residence_time": 240.0, "temperature": 70.0, "catalyst_loading": 2.0}   # synthetic


def test_judge_needs_ligand_and_two_of_three_continuous():
    hit = {"known": True, "precatalyst_or_ligand": "Xantphos", "residence_time_min": 4, "temperature_c": 65,
           "loading_mol_percent": 1.0}
    assert judge(hit, REF) == {"recalled": True, "ligand_match": True, "continuous_matches": 2}
    assert not judge({**hit, "precatalyst_or_ligand": "XPhos Pd G3"}, REF)["recalled"]   # xantphos != xphos
    assert not judge({**hit, "temperature_c": 90}, REF)["recalled"]
    assert not judge({**hit, "known": False}, REF)["recalled"] and not judge(None, REF)["recalled"]


def test_probe_asks_from_public_text_and_never_sends_the_reference():
    assert substrates(TASK) == ("3-bromoquinoline", "3,5-dimethylisoxazole-4-boronic acid pinacol ester")
    sent = []

    class LLM:
        name = "fake"

        def generate(self, req):
            sent.append(req)
            return ProviderResult(role=req.role, provider="fake", endpoint="fixture", status=ProviderStatus.ok,
                                  text=json.dumps({"known": True, "precatalyst_or_ligand": "Xantphos Pd G3",
                                                   "residence_time_min": 3.5, "temperature_c": 70}),
                                  model_requested=req.model, model_returned=req.model)

    cfg = RoleModel(provider="fixture", model="m", reasoning_effort="max")
    r = run_recall_probe(LLM(), cfg, Limits(), [TASK], {TASK.task_id: REF}, CallContext(sink=lambda e: None))
    assert r["verdict"] == "recall_positive" and r["tasks"][0]["recalled"]
    text = json.dumps([q.model_dump(mode="json") for q in sent])
    assert sent[0].role.startswith("researcher") and sent[0].reasoning_effort == "max"
    assert "240" not in text and "Xantphos" not in text   # no reference value leaks into the question


def test_a_reply_that_is_not_a_json_object_is_no_answer():
    class LLM:
        name = "fake"

        def __init__(self, text):
            self.text = text

        def generate(self, req):
            return ProviderResult(role=req.role, provider="fake", endpoint="fixture", status=ProviderStatus.ok,
                                  text=self.text, model_requested=req.model, model_returned=req.model)

    cfg = RoleModel(provider="fixture", model="m", reasoning_effort="max")
    for text in ('"unknown"', "true", "[1, 2]"):
        r = run_recall_probe(LLM(text), cfg, Limits(), [TASK], {TASK.task_id: REF}, CallContext(sink=lambda e: None))
        assert r["verdict"] == "no_recall" and r["tasks"][0]["answer"] is None
