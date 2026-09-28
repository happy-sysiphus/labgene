"""T08.4 leakage-gate validation tool. The fixture checker run is a development_only contract check, never live gate
accuracy. B12 (legitimate material allowed, not removed by answer-string match), B13 (no internal detail in outputs).
Regression tests test_finding<N>_* cover the independent review of the evaluation module."""
import re
from collections import Counter
from pathlib import Path
from typing import get_args

import pytest
import yaml

from labgene.contracts import GateStatus, PrivateTaskAssets, canonical_json, payload_hash
from labgene.costs import CallContext
from labgene.evaluation.gate_validation import Origin, load_gate_cases, run_gate_validation
from labgene.knowledge.base import GateDecision
from labgene.knowledge.gate import FixtureMarkerChecker

ROOT = Path(__file__).resolve().parents[2]
ASSETS = [PrivateTaskAssets.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
          for p in sorted((ROOT / "tests" / "fixtures" / "private").glob("*.yaml"))]
BUNDLES = [a.answer_bundle for a in ASSETS]
CASES = load_gate_cases(ROOT / "configs" / "gate_cases" / "dev_fixture.yaml")
CTX = CallContext(sink=lambda e: None)
N_IDENTITY = 5                              # g01-g04 (blocked documents) + g08 (hidden asset path)


class Scripted:
    """Test double: status (or exception) per call, by a rule over (text, bundle, context); records the contexts."""
    checker_id = "scripted"
    policy_version = "scripted-v1"

    def __init__(self, rule, development_only=True):
        self.rule, self.calls, self.contexts, self.development_only = rule, 0, [], development_only

    def check(self, text, bundle, context, ctx):
        self.calls += 1
        self.contexts.append(context)
        out = self.rule(text, bundle, context)
        if isinstance(out, BaseException):
            raise out
        return GateDecision(status=out, policy_version=self.policy_version, checker=self.checker_id,
                            internal_detail="INTERNAL-DETAIL-must-not-leak")


def _numbers(text):
    return {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)}


def answer_string_matcher(text, bundle, context):
    """What B12 forbids: a gate that is right on every other case but also blocks any text containing every value
    of a known success input."""
    for a in ASSETS:
        for inp in a.known_success_inputs:
            vals = [v for v in inp.values()]
            if all((float(v) in _numbers(text)) if isinstance(v, (int, float)) else re.search(rf"\b{v}\b", text)
                   for v in vals):
                return GateStatus.block
    return FixtureMarkerChecker().check(text, bundle, context, CTX).status


def test_b12_dev_gate_cases_allow_citing_background_public_target_and_self_inference():
    r = run_gate_validation(FixtureMarkerChecker(), BUNDLES, CASES, CTX)
    assert len(CASES.cases) >= 12 and r["passed"] and r["wrong_cases"] == []
    assert r["development_only"] and "not live gate accuracy" in r["label"]
    allowed = {c["category"] for c in r["cases"] if c["actual"] == "allow"}
    assert {"citing_background_no_leak", "public_target", "self_observation_inference",
            "general_background"} <= allowed
    assert {c["category"] for c in r["cases"] if c["actual"] == "block"} == {
        "answer_paper", "alternate_version", "derived_from_answer", "hidden_table"}
    assert all(c["identity_blocked"] for c in r["cases"] if c["category"] in ("answer_paper", "alternate_version"))


def test_finding16_b12_case_set_catches_a_pure_answer_string_matcher():
    # the self-inference cases really contain a known success input (else the string-match clause is untested) ...
    inference = [c for c in CASES.cases if c.category == "self_observation_inference"]
    assert {c.case_id for c in inference} == {"g13", "g18"}
    assert all(answer_string_matcher(c.text, BUNDLES[0], {"source": "x"}) is GateStatus.block for c in inference)
    # ... so a gate that deletes by answer-string match fails validation on exactly those cases ...
    r = run_gate_validation(Scripted(answer_string_matcher), BUNDLES, CASES, CTX)
    assert not r["passed"] and set(r["wrong_cases"]) == {"g13", "g18"}
    assert r["counts"]["false_block_on_allow_cases"] == 2
    # ... while the checker under test allows them: it is not a pure answer-string matcher
    ok = {c["case_id"]: c["actual"] for c in run_gate_validation(FixtureMarkerChecker(), BUNDLES, CASES, CTX)["cases"]}
    assert ok["g13"] == ok["g18"] == "allow"


def test_t08_4_confusion_matrix_separates_false_allow_false_block_and_hold():
    allow_all = run_gate_validation(Scripted(lambda t, b, c: GateStatus.allow), BUNDLES, CASES, CTX)
    content_blocks = [c for c in allow_all["cases"] if c["expected"] == "block" and not c["identity_blocked"]]
    assert allow_all["counts"]["false_allow_on_block_cases"] == len(content_blocks) == 3
    assert allow_all["confusion"]["hold"]["allow"] == allow_all["counts"]["false_allow_on_hold_cases"] == 2
    assert not allow_all["passed"] and len(allow_all["failures"]) == 2
    block_all = run_gate_validation(Scripted(lambda t, b, c: GateStatus.block), BUNDLES, CASES, CTX)
    n_allow = sum(c.expected == "allow" for c in CASES.cases)
    assert block_all["counts"]["false_block_on_allow_cases"] == block_all["confusion"]["allow"]["block"] == n_allow
    hold_all = run_gate_validation(Scripted(lambda t, b, c: GateStatus.hold), BUNDLES, CASES, CTX)
    decidable = sum(c.expected != "hold" for c in CASES.cases)
    identity = sum(c["identity_blocked"] for c in hold_all["cases"])
    assert identity == N_IDENTITY and hold_all["counts"]["hold_rate"] == round((decidable - identity) / decidable, 4)
    assert hold_all["counts"]["false_allow_on_block_cases"] == 0 and not hold_all["passed"]


def test_t08_4_checker_errors_are_counted_separately_never_as_allow_and_detail_stays_internal():
    def rule(text, bundle, context):
        if "review" in text.casefold():
            return RuntimeError("checker crashed on https://secret.example/LEAK")
        return GateStatus.error if "task statement" in text.casefold() else GateStatus.allow
    r = run_gate_validation(Scripted(rule), BUNDLES, CASES, CTX)
    errors = [c for c in r["cases"] if c["actual"] == "error"]
    assert {c["case_id"] for c in errors} == {"g09", "g10", "g11", "g12"}
    assert r["counts"]["error_cases"] == 4 and r["counts"]["errors"] == 4 * len(BUNDLES)   # per checker call
    assert not r["passed"] and any(f.startswith("errors=") for f in r["failures"])
    assert r["counts"]["false_allow_on_block_cases"] == 3       # only the content-only blocks the stub allowed
    blob = canonical_json(r)
    for hidden in ("INTERNAL-DETAIL", "LEAK-", "secret_marker", "crashed", "hidden full table"):
        assert hidden not in blob, hidden


def test_finding10_checker_errors_are_counted_per_call_not_masked_by_another_bundles_block():
    def half_broken(text, bundle, context):          # crashes on every call for one bundle, correct for the other
        if bundle.bundle_id == "fixture-answers-ridge":
            return RuntimeError("provider 500")
        return FixtureMarkerChecker().check(text, bundle, context, CTX).status
    only = CASES.model_copy(update={"cases": [c for c in CASES.cases if c.case_id == "g06"]})   # catalyst leak
    r = run_gate_validation(Scripted(half_broken), BUNDLES, only, CTX)
    assert r["cases"][0]["actual"] == "block"                   # the block still wins the combined verdict ...
    assert r["counts"]["errors"] == r["cases"][0]["checker_errors"] == 1    # ... but the crash is counted
    assert r["counts"]["checker_calls"] == 2 and not r["passed"] and r["failures"] == ["errors=1 > 0"]


def test_finding6_the_whole_case_file_is_pinned_and_recorded(tmp_path):
    lock = tmp_path / "gate_input.lock.json"
    r = run_gate_validation(FixtureMarkerChecker(), BUNDLES, CASES, CTX, input_lock=lock)
    assert r["criteria_hash"] == payload_hash(CASES.criteria.model_dump())
    assert r["input_hash"] == payload_hash(CASES.model_dump(mode="json", exclude={"version", "status"}))
    checker = Scripted(lambda t, b, c: GateStatus.allow)
    looser = CASES.model_copy(update={"criteria": CASES.criteria.model_copy(update={"max_false_allow_on_block_cases": 2})})
    dropped = CASES.model_copy(update={"cases": [c for c in CASES.cases if c.case_id not in ("g05", "g06", "g07")]})
    relabeled = CASES.model_copy(update={"cases": [c.model_copy(update={"expected": "hold"}) if c.case_id == "g05"
                                                   else c for c in CASES.cases]})
    for changed in (looser, dropped, relabeled):                # same version + lock: refused before scoring
        with pytest.raises(ValueError, match="new version"):
            run_gate_validation(checker, BUNDLES, changed, CTX, input_lock=lock)
    assert checker.calls == 0
    reviewed = CASES.model_copy(update={"status": "reviewed"})  # a review-status change is not an input change
    assert run_gate_validation(FixtureMarkerChecker(), BUNDLES, reviewed, CTX, input_lock=lock)["input_hash"] == \
        r["input_hash"]
    renamed = dropped.model_copy(update={"version": "gate-cases-dev-fixture-v2"})
    assert run_gate_validation(checker, BUNDLES, renamed, CTX, input_lock=lock)["case_set_version"].endswith("v2")
    with pytest.raises(ValueError, match="answer bundles"):
        run_gate_validation(FixtureMarkerChecker(), BUNDLES[:1], CASES, CTX)
    live = Scripted(lambda t, b, c: GateStatus.allow, development_only=False)
    with pytest.raises(ValueError, match="input_lock"):         # a live validation must pin its input
        run_gate_validation(live, BUNDLES, CASES.model_copy(update={"status": "reviewed"}), CTX)


def test_finding12_a_live_checker_on_a_draft_case_set_is_provisional_never_pass(tmp_path):
    marker = FixtureMarkerChecker()
    live = Scripted(lambda t, b, c: marker.check(t, b, c, CTX).status, development_only=False)
    draft = CASES.model_copy(update={"status": "draft_unreviewed"})
    r = run_gate_validation(live, BUNDLES, draft, CTX, input_lock=tmp_path / "l.json")
    assert r["failures"] == [] and r["status"] == "provisional" and not r["passed"]
    assert "provisional" in r["label"] and not r["development_only"]
    reviewed = run_gate_validation(live, BUNDLES, CASES.model_copy(update={"status": "reviewed"}), CTX,
                                   input_lock=tmp_path / "l.json")
    assert reviewed["status"] == "pass" and reviewed["passed"] and not reviewed["development_only"]
    # the development case set never yields a live claim, even with a live checker
    dev_set = run_gate_validation(live, BUNDLES, CASES, CTX)
    assert dev_set["development_only"] and "not live gate accuracy" in dev_set["label"]


def test_finding13_the_checker_gets_production_check_contexts():
    rec = Scripted(lambda t, b, c: GateStatus.allow)
    run_gate_validation(rec, BUNDLES, CASES, CTX)
    origins = Counter(c["origin"] for c in rec.contexts)
    assert set(origins) <= set(get_args(Origin)) and "gate_validation" not in origins
    assert origins["derived"] == 2 * len(BUNDLES)               # g13, g18 as store.derive sends them
    derived = [c for c in rec.contexts if c["origin"] == "derived"]
    assert all(c["source"].startswith("derived:") and c["kind"] == "summary" for c in derived)
    raw = yaml.safe_load((ROOT / "configs" / "gate_cases" / "dev_fixture.yaml").read_text(encoding="utf-8"))
    raw["cases"][0]["origin"] = "gate_validation"
    with pytest.raises(ValueError):                             # a non-production origin is refused on load
        type(CASES).model_validate(raw)
