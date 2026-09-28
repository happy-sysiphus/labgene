"""T03 behaviour checks: condition memory, end-of-episode finalization, whole-state restore.
Offline/fake providers only: contract checks, never research or product performance."""
import re
import shutil
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest

from labgene import faults
from labgene.config import Limits, RoleModel, load_profile, load_public_task
from labgene.contracts import (AdvisorResponse, ConsultExchange, ExperimentError, GateStatus, MemoryScope,
                               Observation, Outcome, ProviderResult, ProviderStatus, RunScope, Usage)
from labgene.costs import CallContext
from labgene.memory import snapshot
from labgene.memory.base import FinalizeInput, finalize_event_id
from labgene.memory.baseline import BaselineTextMemory, SummarizerUnavailable
from labgene.memory.product import ProductMemory
from labgene.memory.snapshot import restore_state, snapshot_state, sqlite_logical_hash, state_dir_hash
from labgene.providers.base import ModelChangedError

ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_profile(ROOT / "configs" / "offline.yaml")
TASKS = {"fixture_ridge": load_public_task(PROFILE, "fixture_ridge")}
MEMS = {"baseline": BaselineTextMemory, "product": ProductMemory}
CONDS = pytest.mark.parametrize("cond", ["baseline", "product"])


class Sink(list):
    def ctx(self, **kw):
        return CallContext(sink=self.append, **kw)


def mscope(cond, rep=1):
    return MemoryScope(run_id="r1", condition=cond, set_id="s1", set_rep=rep)


def open_mem(cond, d, rep=1, **kw):
    return MEMS[cond](d, mscope(cond, rep), tasks=TASKS, **kw)


def rs(cond, order=1, rep=1, visit=1):
    return RunScope(run_id="r1", condition=cond, set_id="s1", set_rep=rep, episode_id=f"{cond}-rep{rep}-ep{order}",
                    episode_order=order, task_id="fixture_ridge", visit_index=visit)


def obs(s, i, y, aid=None):
    return Observation(observation_id=f"{s.episode_id}-o{i}", action_id=aid or f"{s.episode_id}-a{i}", scope=s,
                       parameters={"temperature": 50.0 + i, "time": 10.0}, results={"yield": y}, units={"yield": "%"},
                       simulator_id="fixture.ridge", simulator_version="1",
                       meets_success_criteria=TASKS["fixture_ridge"].is_success({"yield": y}),
                       created_at=f"2026-09-28T00:00:{i:02d}Z")


def consult(aid, answer="raise the temperature"):
    return ConsultExchange(action_id=aid, question="what next?", response=AdvisorResponse(answer=answer))


def invalid(s, aid):
    return ExperimentError(action_id=aid, scope=s, submitted_parameters={"temperature": 999},
                           reason="temperature out of range")


def fin(s, observations=(), errors=(), consults=(), outcome=Outcome.budget_exhausted):
    return FinalizeInput(event_id=finalize_event_id(s), scope=s, outcome=outcome, observations=list(observations),
                         errors=list(errors), consults=list(consults))


def saved_obs_ids(m):
    if isinstance(m, ProductMemory):
        return [o.observation_id for o in m.observations()]
    return re.findall(r"observation (\S+):", m.render(10**7))


def saved_outcomes(m):
    if isinstance(m, ProductMemory):
        return [(c["outcome"], c["actions_used"]) for c in m.episode_cases()]
    return [(o, int(n)) for o, n in re.findall(r"- outcome (\w+): actions used (\d+)", m.render(10**7))]


@CONDS
def test_b07_unconsulted_and_post_consult_observations_saved_exactly_once(tmp_path, cond):
    m = open_mem(cond, tmp_path)
    s1 = rs(cond, order=1)                       # never consulted: a miss then a success
    e1 = fin(s1, [obs(s1, 1, 40.0), obs(s1, 2, 95.0)], outcome=Outcome.success)
    s2 = rs(cond, order=2, visit=2)              # consult first, then misses after the last consult + one invalid
    e2 = fin(s2, [obs(s2, 2, 50.0), obs(s2, 3, 60.0)], errors=[invalid(s2, "bad-1")], consults=[consult("c-1")])
    ctx = Sink().ctx()
    assert [m.finalize_episode(e, ctx).status for e in (e1, e2)] == ["done", "done"]
    assert m.finalize_episode(e2, ctx).records_added == 0

    assert saved_obs_ids(m) == [o.observation_id for e in (e1, e2) for o in e.observations]
    assert saved_outcomes(m) == [("success", 2), ("budget_exhausted", 4)]
    if isinstance(m, ProductMemory):
        assert [(e.action_id, e.reason) for e in m.errors()] == [("bad-1", "temperature out of range")]
        rels = m.relations("fixture_ridge")
        assert [r["predicate"] for r in rels] == ["missed_success_criteria_at", "met_success_criteria_at",
                                                  "missed_success_criteria_at", "missed_success_criteria_at"]
        assert all(r["provenance"] == "experiment" and r["conditions"]["simulator_version"] == "1"
                   and r["conditions"]["task_version"] == "1" for r in rels)
        assert m.episode_cases()[0]["best_observation_id"] == e1.observations[1].observation_id
    else:
        assert "invalid request bad-1 (no measured value): temperature out of range" in m.render(10**7)


@CONDS
def test_b08_crash_mid_finalize_then_retry_same_event_no_duplicates(tmp_path, cond, monkeypatch):
    s = rs(cond)
    data = fin(s, [obs(s, 1, 40.0), obs(s, 2, 95.0)], consults=[consult("c-1")], outcome=Outcome.success)
    m = open_mem(cond, tmp_path)
    empty = m.state_hash()
    faults.reset()
    monkeypatch.setenv("LABGENE_FAULT", "during_finalize")
    with pytest.raises(faults.InjectedCrash):
        m.finalize_episode(data, Sink().ctx())
    m.close()
    monkeypatch.delenv("LABGENE_FAULT")
    faults.reset()

    m = open_mem(cond, tmp_path)                  # process restart
    assert not m.is_finalized(data.event_id) and m.state_hash() == empty   # all-or-nothing
    r1 = m.finalize_episode(data, Sink().ctx())
    r2 = m.finalize_episode(data, Sink().ctx())   # e.g. crash after commit, before the harness saw "done"
    assert (r1.status, r2.status, r2.records_added) == ("done", "done", 0) and r1.memory_hash == r2.memory_hash
    assert m.is_finalized(data.event_id)
    assert saved_obs_ids(m) == [o.observation_id for o in data.observations]

    conflict = fin(s, [obs(s, 1, 41.0)], outcome=Outcome.budget_exhausted)   # same event id, other payload
    r3 = m.finalize_episode(conflict, Sink().ctx())
    assert r3.status == "failed" and m.state_hash() == r1.memory_hash
    assert saved_outcomes(m) == [("success", 3)]  # the scientific outcome is never altered by memory


def test_b10_whole_state_dir_snapshot_restore_including_open_wal(tmp_path):
    state, snap = tmp_path / "state", tmp_path / "snap"
    m = open_mem("product", state)
    s1 = rs("product")
    m.finalize_episode(fin(s1, [obs(s1, 1, 40.0)]), Sink().ctx())
    k = sqlite3.connect(state / "knowledge.sqlite", isolation_level=None)
    k.execute("PRAGMA journal_mode=WAL")
    k.execute("PRAGMA wal_autocheckpoint=0")
    k.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, text TEXT)")
    k.executemany("INSERT INTO chunks VALUES (?,?)", [(f"c{i}", "x" * 100) for i in range(50)])
    (state / "index").mkdir()
    (state / "index" / "bm25.json").write_text('{"df": {"a": 1}}', encoding="utf-8")
    (state / "cache").mkdir()
    (state / "cache" / "summary.txt").write_text("cached summary", encoding="utf-8")
    # committed rows live only in the WAL: a raw copy of the main file would lose them
    assert (state / "knowledge.sqlite-wal").stat().st_size > 0
    shutil.copyfile(state / "knowledge.sqlite", tmp_path / "raw.sqlite")
    with closing(sqlite3.connect(tmp_path / "raw.sqlite")) as c:
        assert c.execute("SELECT count(*) FROM sqlite_master WHERE name='chunks'").fetchone()[0] == 0

    manifest = snapshot_state(state, snap)
    assert set(manifest["files"]) == {"memory.sqlite", "knowledge.sqlite", "index/bm25.json", "cache/summary.txt"}
    assert manifest["combined_sha256"] == state_dir_hash(state)
    with closing(sqlite3.connect(state / "memory.sqlite")) as c:   # snapshot hashes every table incl. the binding
        assert manifest["files"]["memory.sqlite"]["logical_sha256"] == sqlite_logical_hash(c)
        assert m.state_hash() == sqlite_logical_hash(c, exclude=("meta",))   # state_hash = content only

    k.execute("INSERT INTO chunks VALUES ('late', 'y')")          # mutate every part
    s2 = rs("product", order=2)
    m.finalize_episode(fin(s2, [obs(s2, 1, 50.0)]), Sink().ctx())
    (state / "index" / "bm25.json").write_text('{"df": {}}', encoding="utf-8")
    (state / "cache" / "later.txt").write_text("later", encoding="utf-8")
    assert state_dir_hash(state) != manifest["combined_sha256"]

    restore_state(snap, tmp_path / "fresh")
    assert state_dir_hash(tmp_path / "fresh") == manifest["combined_sha256"]
    with closing(sqlite3.connect(tmp_path / "fresh" / "knowledge.sqlite")) as c:
        assert c.execute("SELECT count(*) FROM chunks").fetchone()[0] == 50

    k.close()
    m.close()
    restore_state(snap, state)                                     # over a live dir: moved aside, never deleted
    assert (tmp_path / "state.superseded-1" / "cache" / "later.txt").read_text(encoding="utf-8") == "later"
    assert state_dir_hash(state) == manifest["combined_sha256"]
    restore_state(snap, state)
    assert (tmp_path / "state.superseded-2").is_dir()
    m = open_mem("product", state)
    assert saved_obs_ids(m) == [f"{s1.episode_id}-o1"]
    m.close()


def test_b10_memory_scopes_and_conditions_never_share_state(tmp_path):
    ctx = Sink().ctx()
    s = rs("baseline", rep=1)
    a = open_mem("baseline", tmp_path / "a", rep=1)
    a.finalize_episode(fin(s, [obs(s, 1, 40.0)]), ctx)
    a.close()
    with pytest.raises(ValueError):
        open_mem("baseline", tmp_path / "a", rep=2)                # a dir bound to rep 1 cannot serve rep 2
    b = open_mem("baseline", tmp_path / "b", rep=2)
    with pytest.raises(ValueError):
        b.finalize_episode(fin(s, [obs(s, 1, 40.0)]), ctx)         # rep-1 episode into rep-2 memory
    p = open_mem("product", tmp_path / "p")
    with pytest.raises(ValueError):
        p.finalize_episode(fin(s, [obs(s, 1, 40.0)]), ctx)         # baseline episode into product memory
    with pytest.raises(ValueError):
        ProductMemory(tmp_path / "q", mscope("baseline"))
    assert b.render(10**6) == "" and p.observations() == []


@CONDS
def test_b22_finalization_logs_cost_and_adds_zero_actions(tmp_path, cond):
    sink = Sink()
    m = open_mem(cond, tmp_path)
    s = rs(cond)
    data = fin(s, [obs(s, 1, 40.0)], errors=[invalid(s, "bad-1")], consults=[consult("c-1")])
    m.finalize_episode(data, sink.ctx(phase="runtime", action_id="last-action", episode_id=s.episode_id))
    [ev] = sink
    assert (ev.kind, ev.phase, ev.action_id, ev.episode_id, ev.status) == \
        ("finalize", "finalization", None, s.episode_id, "done")
    assert ev.latency_s >= 0
    assert saved_outcomes(m) == [("budget_exhausted", 3)]           # 1 consult + 1 valid + 1 invalid, nothing added


@CONDS
def test_withheld_consult_answer_never_rendered(tmp_path, cond):
    calls = []

    def gate(text, context, ctx):
        calls.append(context["kind"])
        if "BOOM" in text:
            raise RuntimeError("checker down")
        return GateStatus.block if "BLOCK" in text else GateStatus.hold if "HOLD" in text else GateStatus.allow

    m = open_mem(cond, tmp_path, gate_derived=gate)
    s = rs(cond)
    data = fin(s, [obs(s, 1, 40.0)], consults=[consult("c-ok"), consult("c-b", "optimum BLOCK 88"),
                                               consult("c-h", "maybe HOLD"), consult("c-e", "BOOM")])
    assert m.finalize_episode(data, Sink().ctx()).status == "done"
    visible = m.render(10**6) if cond == "baseline" else repr(m.consult_cases())
    assert "raise the temperature" in visible
    assert not any(w in visible for w in ("BLOCK", "HOLD", "BOOM"))
    assert calls == ["consult_answer"] * 4                             # observations are never gated
    assert saved_obs_ids(m) == [f"{s.episode_id}-o1"]


def test_render_summary_keeps_raw_records_and_is_cached(tmp_path):
    m = open_mem("baseline", tmp_path)
    ctx = Sink().ctx()
    for order in range(1, 5):
        s = rs("baseline", order=order, visit=order)
        m.finalize_episode(fin(s, [obs(s, i, 30.0 + i + (60 if order == 2 and i == 5 else 0)) for i in range(1, 11)]), ctx)
    full, h = m.render(10**7), m.state_hash()
    sink = Sink()
    short = m.render(1500, sink.ctx())
    assert len(short) <= 1500 < len(full) and "development_only" in short
    assert "best baseline-rep1-ep2-o5" in short and "observation baseline-rep1-ep4-o10:" in short
    assert m.state_hash() == h and m.render(10**7) == full         # raw log intact
    assert [e.role for e in sink] == ["internal_summarizer"]
    assert m.render(1500, sink.ctx()) == short and len(sink) == 1  # cached by record-set hash
    assert len(list((tmp_path / "cache").glob("baseline_summary_*.json"))) == 1


class FakeSummarizer:
    name = "fake"

    def __init__(self, status=ProviderStatus.ok, text="SUMMARY: best o5 yield 95", returned=None):
        self.status, self.text, self.returned, self.requests = status, text, returned, []

    def generate(self, req):
        self.requests.append(req)
        ok = self.status == ProviderStatus.ok
        return ProviderResult(role=req.role, provider="fake", endpoint="fixture", status=self.status,
                              text=self.text if ok else None, usage=Usage(input_tokens=10),
                              model_requested=req.model, model_returned=(self.returned or req.model) if ok else None)


ROLE = RoleModel(provider="fixture", model="fake-sum")
NO_WAIT = Limits(provider_backoff_s=0.0)
ALLOW = lambda t, c, x: GateStatus.allow   # noqa: E731


def summarizing_mem(tmp_path, llm, gate=ALLOW):
    m = open_mem("baseline", tmp_path, summarizer=llm, gate_derived=gate, summarizer_role=ROLE, limits=NO_WAIT)
    s = rs("baseline")
    m.finalize_episode(fin(s, [obs(s, i, 30.0 + i) for i in range(1, 30)]), Sink().ctx())
    return m


@pytest.mark.parametrize("status,verdict", [(ProviderStatus.ok, GateStatus.allow), (ProviderStatus.ok, GateStatus.block),
                                            (ProviderStatus.infra_error, GateStatus.allow)])
def test_b19_llm_summary_costed_gated_cached_finitely_retried_never_substituted(tmp_path, status, verdict):
    llm = FakeSummarizer(status)
    m = summarizing_mem(tmp_path, llm, lambda t, c, x: verdict)
    sink = Sink()
    if status != ProviderStatus.ok or verdict != GateStatus.allow:
        with pytest.raises(SummarizerUnavailable):
            m.render(500, sink.ctx())
        assert not list((tmp_path / "cache").glob("*.json"))       # nothing withheld/failed is cached
    else:
        assert m.render(500, sink.ctx()) == "SUMMARY: best o5 yield 95"
        assert m.render(500, sink.ctx()) == "SUMMARY: best o5 yield 95" and len(llm.requests) == 1
    tries = NO_WAIT.provider_max_attempts if status == ProviderStatus.infra_error else 1
    assert [(r.role, r.model) for r in llm.requests] == [("internal_summarizer", "fake-sum")] * tries
    assert [(e.kind, e.role, e.usage.input_tokens) for e in sink] == [("llm_call", "internal_summarizer", 10)] * tries


def test_b19_summarizer_returning_another_model_stops_nothing_cached(tmp_path):
    m = summarizing_mem(tmp_path, FakeSummarizer(returned="some-other-flash-model"))
    with pytest.raises(ModelChangedError):
        m.render(500, Sink().ctx())
    assert not list((tmp_path / "cache").glob("*.json"))


@pytest.mark.parametrize("n", [27, 45, 50])
def test_llm_summary_over_limit_is_cut_between_values_and_marked(tmp_path, n):
    text = "Best observed yield was 95.4 % at temperature 123.5"
    out = summarizing_mem(tmp_path, FakeSummarizer(text=text)).render(n, Sink().ctx())
    assert len(out) <= n and out.endswith("[...truncated]")
    assert text.startswith(out[:-len("[...truncated]")].rstrip())
    assert set(re.findall(r"\d+(?:\.\d+)?", out)) <= {"95.4", "123.5"}   # never '123.' or '95.'


def test_b23_development_fallbacks_refused_when_limits_are_frozen(tmp_path):
    frozen = Limits(development_only=False)
    with pytest.raises(ValueError):   # no summarizer -> would serve the development_only digest
        BaselineTextMemory(tmp_path / "a", mscope("baseline"), limits=frozen, gate_derived=ALLOW)
    with pytest.raises(ValueError):   # no derived-text gate -> consult answers/summaries stored ungated
        BaselineTextMemory(tmp_path / "b", mscope("baseline"), FakeSummarizer(), frozen, summarizer_role=ROLE)
    with pytest.raises(ValueError):
        ProductMemory(tmp_path / "c", mscope("product"), limits=frozen)
    BaselineTextMemory(tmp_path / "d", mscope("baseline"), FakeSummarizer(), frozen, summarizer_role=ROLE,
                       gate_derived=ALLOW).close()
    ProductMemory(tmp_path / "e", mscope("product"), gate_derived=ALLOW, limits=frozen).close()


def test_b10_second_scope_on_same_dir_rejected_before_any_finalize(tmp_path):
    a = open_mem("baseline", tmp_path, rep=1)                      # opened, nothing finalized yet
    h = a.state_hash()
    for cond, rep in (("baseline", 2), ("product", 1)):
        with pytest.raises(ValueError):
            open_mem(cond, tmp_path, rep=rep)
    assert a.state_hash() == h                                     # the rejected open wrote nothing
    s = rs("baseline", rep=1)
    assert a.finalize_episode(fin(s, [obs(s, 1, 40.0)]), Sink().ctx()).status == "done"
    assert saved_obs_ids(a) == [f"{s.episode_id}-o1"]
    a.close()


@CONDS
def test_b08_resumable_checkpoint_cannot_consume_the_finalize_event(tmp_path, cond):
    m, s = open_mem(cond, tmp_path), rs(cond)
    with pytest.raises(ValueError):
        m.finalize_episode(fin(s, [obs(s, 1, 40.0)], outcome=Outcome.infra_incomplete), Sink().ctx())
    assert not m.is_finalized(finalize_event_id(s))
    done = fin(s, [obs(s, 1, 40.0), obs(s, 2, 95.0)], outcome=Outcome.success)   # resumed, then terminal
    assert m.finalize_episode(done, Sink().ctx()).status == "done"
    assert saved_obs_ids(m) == [o.observation_id for o in done.observations]
    assert saved_outcomes(m) == [("success", 2)]


def test_b10_snapshot_fails_bounded_while_a_writer_holds_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "BUSY_TIMEOUT_S", 0.2)
    state = tmp_path / "state"
    state.mkdir()
    w = sqlite3.connect(state / "knowledge.sqlite", isolation_level=None)
    w.execute("CREATE TABLE t(x)")
    w.execute("BEGIN EXCLUSIVE")
    w.execute("INSERT INTO t VALUES (1)")
    err = []
    th = threading.Thread(target=lambda: err.append(pytest.raises(sqlite3.OperationalError, snapshot_state,
                                                                  state, tmp_path / "snap")), daemon=True)
    th.start()
    th.join(20)
    alive = th.is_alive()
    w.execute("COMMIT")                                            # lets a hung backup finish
    th.join(20)
    w.close()
    assert not alive and err


def test_b10_snapshot_or_hash_of_missing_state_dir_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        snapshot_state(tmp_path / "does-not-exist", tmp_path / "snap")
    with pytest.raises(FileNotFoundError):
        state_dir_hash(tmp_path / "does-not-exist")
    assert not (tmp_path / "snap").exists()


def test_baseline_render_is_chronological_across_kinds(tmp_path):
    m, s = open_mem("baseline", tmp_path), rs("baseline")
    a = [f"{s.episode_id}:a{n:03d}" for n in range(1, 6)]           # harness-issued ids (runner.py)
    data = fin(s, [obs(s, 1, 40.0, a[0]), obs(s, 2, 45.0, a[1]), obs(s, 4, 50.0, a[3])],
               errors=[invalid(s, a[4])], consults=[consult(a[2])])
    m.finalize_episode(data, Sink().ctx())
    full = m.render(10**6)
    pos = [full.index(x) for x in a]
    assert pos == sorted(pos)
