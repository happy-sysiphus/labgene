import json

import pytest

from labgene.config import Limits, RoleModel
from labgene.costs import null_context
from labgene.providers.base import ModelChangedError
from labgene.providers.fixture import FixtureProvider
from rag_eval.judge import Judge, fixture_judge

ROLE = RoleModel(provider="fixture", model="judge-m", endpoint="fixture", max_output_tokens=512)
LIM = Limits(provider_backoff_s=0.0)
REPLY = {"answer": "a", "reasoning": "", "candidate_rationales": []}


def make(reply: str) -> Judge:
    return Judge(FixtureProvider(policy=lambda req: reply), ROLE, LIM)


def test_claims_are_parsed_and_malformed_replies_are_unavailable():
    ok = make('{"claims": [{"text": "Yield was 80 %.", "type": "fact"}, {"text": "Try 100 degC.", "type": "inference"}]}')
    assert ok.claims("q", REPLY, null_context()) == [{"text": "Yield was 80 %.", "type": "fact"},
                                                      {"text": "Try 100 degC.", "type": "inference"}]
    assert make('{"claims": [{"text": "x", "type": "opinion"}]}').claims("q", REPLY, null_context()) is None
    assert make("not json").claims("q", REPLY, null_context()) is None


def test_verify_aligns_claims_drops_unknown_ids_and_needs_every_claim():
    reply = json.dumps({"verdicts": [{"claim": 1, "verdict": "contradicted", "support": ["P01"]},
                                     {"claim": 0, "verdict": "supported", "support": ["P02", "P99"]}]})
    assert make(reply).verify(["a", "b"], ["c0", "c1"], null_context()) == [
        {"verdict": "supported", "support": [1]}, {"verdict": "contradicted", "support": []}]
    missing = json.dumps({"verdicts": [{"claim": 0, "verdict": "supported", "support": []}]})
    assert make(missing).verify(["a"], ["c0", "c1"], null_context()) is None
    dup = json.dumps({"verdicts": [{"claim": 0, "verdict": "supported", "support": []}] * 2})
    assert make(dup).verify(["a"], ["c0", "c1"], null_context()) is None
    bad = json.dumps({"verdicts": [{"claim": 0, "verdict": "maybe", "support": []}]})
    assert make(bad).verify(["a"], ["c0"], null_context()) is None


def test_useful_and_questions():
    reply = '{"verdicts": [{"card": "C02", "useful": false}, {"card": "C01", "useful": true}]}'
    assert make(reply).useful("q", {}, ["x", "y"], null_context()) == [True, False]
    assert make('{"verdicts": [{"card": "C01", "useful": "yes"}]}').useful("q", {}, ["x"], null_context()) is None
    qs = make('{"questions": ["a?", "b?", "c?"], "noncommittal": false}').questions("ans", null_context())
    assert qs == (["a?", "b?", "c?"], False)
    assert make('{"questions": [], "noncommittal": false}').questions("ans", null_context()) is None


def test_infra_errors_after_retries_are_unavailable_not_defaults():
    p = FixtureProvider(policy=lambda req: "{}", script=["infra_error"] * 3)
    assert Judge(p, ROLE, LIM).questions("ans", null_context()) is None
    assert len(p.requests) == LIM.provider_max_attempts


def test_a_different_returned_model_stops_the_run():
    class Swapped(FixtureProvider):
        def generate(self, req):
            return super().generate(req).model_copy(update={"model_returned": "other-model"})
    with pytest.raises(ModelChangedError):
        Judge(Swapped(policy=lambda req: "{}"), ROLE, LIM).questions("ans", null_context())


def test_requests_carry_the_role_model_and_a_schema():
    p = FixtureProvider(policy=fixture_judge)
    Judge(p, ROLE, LIM).questions("The yield was 80 %.", null_context())
    req = p.requests[0]
    assert (req.model, req.role) == ("judge-m", "rag_judge_questions") and req.response_schema is not None


def test_fixture_judge_round_trip():
    j = Judge(FixtureProvider(policy=fixture_judge), ROLE, LIM)
    claims = j.claims("q", {**REPLY, "answer": "Yield 80 was seen. Try more."}, null_context())
    assert [c["type"] for c in claims] == ["fact", "inference"]
    assert j.verify(["yield 80 %", "other"], ["Yield 80 was seen."], null_context()) == [
        {"verdict": "supported", "support": [0]}]
