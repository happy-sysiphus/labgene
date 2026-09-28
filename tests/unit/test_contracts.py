import pytest

from labgene.config import CostCaps, Prices, load_profile, load_public_task, preflight_problems
from labgene.contracts import InvalidParameters, Usage, ValidParameters
from labgene.costs import CapExceeded, CostGuard
from labgene.simulators.validation import validate_parameters

PROFILE = load_profile("configs/offline.yaml")


def test_parameter_validation_stage2():
    t = load_public_task(PROFILE, "fixture_catalyst")
    ok = validate_parameters(t, {"catalyst": "C", "loading": 1, "temperature": 90, "residence_time": 540})
    assert isinstance(ok, ValidParameters) and ok.parameters["loading"] == 1.0
    for bad in [
        {"catalyst": "C", "loading": 1, "temperature": 90},                                  # missing
        {"catalyst": "Z", "loading": 1, "temperature": 90, "residence_time": 540},           # bad category
        {"catalyst": "C", "loading": 9, "temperature": 90, "residence_time": 540},           # range
        {"catalyst": "C", "loading": 1, "temperature": 110, "residence_time": 600},          # constraint
        {"catalyst": "C", "loading": True, "temperature": 90, "residence_time": 540},        # bool
        {"catalyst": "C", "loading": 1, "temperature": 90, "residence_time": 540, "x": 1},   # unknown
        "not a dict",
    ]:
        assert isinstance(validate_parameters(t, bad), InvalidParameters)


def test_success_rule_multi_metric_and_direction():
    t = load_public_task(PROFILE, "fixture_catalyst")
    assert t.is_success({"yield": 78.0, "ton": 57.0})          # boundary = target - tolerance
    assert not t.is_success({"yield": 99.0, "ton": 56.9})       # both required
    assert not t.is_success({"yield": 99.0})


def test_cost_guard_reserves_before_call_and_keeps_unknown_usage():
    g = CostGuard(CostCaps(max_calls=2, max_output_tokens=1000, prices={"m": Prices(input_usd_per_mtok=1, output_usd_per_mtok=1)}))
    r = g.reserve("m", 10, 600)
    with pytest.raises(CapExceeded):
        g.reserve("m", 10, 600)       # concurrent worst case would exceed output cap
    g.settle(r, Usage())             # usage unavailable -> reservation kept
    assert g.output_tokens == 600 and g.calls == 1


def test_preflight_offline_clean_and_live_template_blocked():
    assert preflight_problems(PROFILE, env={}) == []
    live = load_profile("configs/live.example.yaml")
    probs = preflight_problems(live, env={"GEMINI_API_KEY": "x", "OPENAI_API_KEY": "x", "ANTHROPIC_API_KEY": "x"})
    assert any("cost_caps.max_usd" in p for p in probs)
