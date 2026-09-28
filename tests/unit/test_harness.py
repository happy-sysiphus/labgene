import json
import sqlite3

import pytest

from labgene.contracts import ActionEnvelope, ActionKind, Observation, Outcome, RunScope, payload_hash
from labgene.harness.ledger import ActionConflict, Ledger, LedgerError
from labgene.harness.parsing import ActionParseError, parse_action

SCOPE = RunScope(run_id="run", condition="baseline", set_id="smoke", set_rep=1, episode_id="baseline-r1-e001",
                 episode_order=1, task_id="fixture_ridge", visit_index=1)


@pytest.mark.parametrize("raw,reason", [
    ('{"action":"run_experiment","args":{"parameters":{"temperature":83', "invalid_json"),   # truncated: never repaired
    ("Let us heat it to 83 degC.", "invalid_json"),
    ("", "invalid_json"),
    ('"consult"', "invalid_json"),
    ('{"action":"declare_success","args":{}}', "unsupported_action"),
    ('{"args":{"question":"q"}}', "unsupported_action"),
    ('{"action":"consult","args":{}}', "missing_question"),
    ('{"action":"consult","args":{"question":"   "}}', "missing_question"),
    ('{"action":"consult","args":{"question":7}}', "missing_question"),
    ('[{"action":"consult","args":{"question":"q"}}]', "multiple_actions"),
    ('{"action":"consult","args":{"question":"q"}}\n{"action":"consult","args":{"question":"r"}}', "multiple_actions"),
    ('{"action":["consult","run_experiment"],"args":{"question":"q"}}', "multiple_actions"),
])
def test_b02_stage1_protocol_errors(raw, reason):
    with pytest.raises(ActionParseError) as e:
        parse_action(raw)
    assert e.value.reason == reason


@pytest.mark.parametrize("raw", [
    '{"action":"consult","action":"run_experiment","args":{"parameters":{"temperature":83,"time":37}}}',
    '{"action":"run_experiment","args":{"parameters":{"temperature":20,"time":1}},'
    '"args":{"parameters":{"temperature":83,"time":37}}}',
    '```json\n{"action":"consult","args":{"question":"a"}}\n```\n```json\n{"action":"consult","args":{"question":"b"}}\n```',
])
def test_b02_repeated_action_key_or_second_fenced_block_is_multiple_actions(raw):
    """Finding 1: JSON silently keeps the last repeated key, which would charge and run one of two actions."""
    with pytest.raises(ActionParseError) as e:
        parse_action(raw)
    assert e.value.reason == "multiple_actions"


def test_b02_interpretable_experiment_with_bad_parameters_passes_stage1():
    for raw in ['{"action":"run_experiment","args":{"hypothesis":"h"}}',
                '{"action":"run_experiment","args":{"parameters":"hot"}}',
                '{"action":"run_experiment"}']:
        assert parse_action(raw)[0] is ActionKind.run_experiment
    fenced = '```json\n{"action":"run_experiment","args":{"hypothesis":"h","parameters":{"temperature":83,"time":37}}}\n```'
    assert parse_action(fenced) == (ActionKind.run_experiment,
                                    {"hypothesis": "h", "parameters": {"temperature": 83, "time": 37}})


def env(action_id, params, scope=SCOPE):
    args = {"hypothesis": None, "parameters": params}
    return ActionEnvelope(action_id=action_id, scope=scope, raw_text=json.dumps(args), kind=ActionKind.run_experiment,
                          args=args, payload_hash=payload_hash({"kind": "run_experiment", "args": args}))


def obs(action_id, y):
    return Observation(observation_id=f"obs:{action_id}", action_id=action_id, scope=SCOPE, parameters={},
                       results={"yield": y}, units={"yield": "%"}, simulator_id="fixture.ridge",
                       simulator_version="1", meets_success_criteria=False, created_at="t")


@pytest.fixture
def ledger(tmp_path):
    L = Ledger(tmp_path)
    L.start_episode(SCOPE, "h0")
    yield L
    L.close()


def test_b04_foreign_keys_enforced_on_every_connection(tmp_path, ledger):
    for L in (ledger, Ledger(tmp_path)):
        assert L.db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):   # action for an episode that does not exist
            L.reserve(env("ghost:a001", {}, SCOPE.model_copy(update={"episode_id": "ghost"})), 1, 1)


def test_b03_same_action_id_replay_not_recharged(ledger):
    e = env("baseline-r1-e001:a001", {"temperature": 20, "time": 1})
    assert ledger.reserve(e, 1, 1) is None
    assert ledger.reserve(e, 1, 1) is None                  # resend while pending: same reservation
    ledger.commit(e.action_id, obs(e.action_id, 0.1))
    stored = ledger.reserve(e, 1, 1)                        # resend after commit: stored result, no charge
    assert json.loads(stored)["results"] == {"yield": 0.1}
    st = ledger.state(SCOPE.episode_id)
    assert (st.experiment_evaluations, st.actions_used, len(ledger.observations(SCOPE.episode_id))) == (1, 1, 1)


def test_b03_payload_conflict_keeps_original(ledger):
    a = env("baseline-r1-e001:a001", {"temperature": 20, "time": 1})
    ledger.reserve(a, 1, 1)
    ledger.commit(a.action_id, obs(a.action_id, 0.1))
    with pytest.raises(ActionConflict):
        ledger.reserve(env(a.action_id, {"temperature": 83, "time": 37}), 1, 2)
    conflicts = ledger.db.execute("SELECT action_id, payload_json FROM action_conflicts").fetchall()
    assert [(r[0], json.loads(r[1])["args"]["parameters"]) for r in conflicts] == \
        [(a.action_id, {"temperature": 83, "time": 37})]
    assert json.loads(ledger.db.execute("SELECT payload_json FROM actions").fetchone()[0])["args"]["parameters"] \
        == {"temperature": 20, "time": 1}
    assert [o.results for o in ledger.observations(SCOPE.episode_id)] == [{"yield": 0.1}]
    assert ledger.state(SCOPE.episode_id).actions_used == 1


def test_b03_new_action_id_same_parameters_is_a_new_charge(ledger):
    for i in (1, 2):
        e = env(f"baseline-r1-e001:a00{i}", {"temperature": 20, "time": 1})
        ledger.reserve(e, i, i)
        ledger.commit(e.action_id, obs(e.action_id, 0.1))
    assert ledger.state(SCOPE.episode_id).experiment_evaluations == 2


def test_b01_ledger_refuses_second_pending_and_51st_action(ledger):
    ledger.reserve(env("baseline-r1-e001:a001", {}), 1, 1)
    with pytest.raises(LedgerError):
        ledger.reserve(env("baseline-r1-e001:a002", {}), 2, 2)
    ledger.commit("baseline-r1-e001:a001", obs("baseline-r1-e001:a001", 0.0))
    for i in range(2, 51):
        e = env(f"baseline-r1-e001:a{i:03d}", {"i": i})
        ledger.reserve(e, i, i)
        ledger.commit(e.action_id, obs(e.action_id, 0.0))
    with pytest.raises(LedgerError):
        ledger.reserve(env("baseline-r1-e001:a051", {}), 51, 51)
    st = ledger.state(SCOPE.episode_id)
    assert (st.actions_used, st.pending_action_id) == (50, None)


def test_b06_observations_and_committed_actions_are_immutable(ledger):
    e = env("baseline-r1-e001:a001", {})
    ledger.reserve(e, 1, 1)
    ledger.commit(e.action_id, obs(e.action_id, 0.1))
    for sql in ("UPDATE observations SET meets_success_criteria=1", "DELETE FROM observations",
                "UPDATE actions SET result_json='{}'", "DELETE FROM actions"):
        with pytest.raises(sqlite3.IntegrityError):
            ledger.db.execute(sql)
    with pytest.raises(LedgerError):                        # commit happens exactly once
        ledger.commit(e.action_id, obs(e.action_id, 99.0))
    assert [o.results for o in ledger.observations(SCOPE.episode_id)] == [{"yield": 0.1}]


def test_b08_scientific_outcome_is_never_rewritten(ledger):
    ledger.set_outcome(SCOPE.episode_id, Outcome.success, actions_to_success=7)
    with pytest.raises(sqlite3.IntegrityError):
        ledger.set_outcome(SCOPE.episode_id, Outcome.budget_exhausted)
    assert ledger.state(SCOPE.episode_id).outcome is Outcome.success
