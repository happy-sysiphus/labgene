"""T06 advisors: B17 (baseline has no product KG/RAG, both share the gated search/open, no optimizer tool),
B11 at advisor level (captured model requests never carry leaked material or gate details), B09/B07 (a finalized
past episode reaches the next consultation of its own condition only), exact current observations in both
conditions, B16 (validation issues surface after bounded repair; mechanical validity is not semantic support),
tool-call / repair bounds, status mapping, revisit marking, costs.
FixtureProvider / fixture gate / hash embedders only: contract checks, never research or product performance."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from labgene.advisors import baseline as baseline_mod, product as product_mod
from labgene.advisors.baseline import BaselineAdvisor
from labgene.advisors.common import PARTIAL_ANSWER, fit
from labgene.advisors.fixture_policy import make_fixture_advisor_policy
from labgene.advisors.product import ProductAdvisor
from labgene.config import CostCaps, Limits, RetrievalConfig, load_profile, load_public_task
from labgene.contracts import (AdvisorResponse, ConsultExchange, ConsultRequest, FunctionCall, MemoryScope,
                               Observation, Outcome, PrivateTaskAssets, ResearcherView, RunScope, canonical_json)
from labgene.costs import CallContext, CapExceeded, CostGuard
from labgene.knowledge.base import RawHit
from labgene.knowledge.build import build_initial_state, load_baseline_initial
from labgene.knowledge.cards import TokenOverlapJudge, check_citation_support
from labgene.knowledge.gate import FixtureMarkerChecker
from labgene.knowledge.gated import GatedSearch
from labgene.knowledge.kg import FixtureKGExtractor, Ontology
from labgene.knowledge.retrieval import HashEmbedder
from labgene.knowledge.search import FixtureSearchProvider
from labgene.knowledge.store import KnowledgeStore
from labgene.memory.base import FinalizeInput, finalize_event_id
from labgene.memory.baseline import BaselineTextMemory
from labgene.memory.product import ProductMemory
from labgene.providers.base import EmbeddingResult, ModelChangedError
from labgene.providers.fixture import FixtureProvider
from labgene.researcher.analysis import AnalysisTools

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures"
PROFILE = load_profile(ROOT / "configs" / "offline.yaml")
RIDGE = load_public_task(PROFILE, "fixture_ridge")
TASKS = {"fixture_ridge": RIDGE}
BUNDLES = [PrivateTaskAssets.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))).answer_bundle
           for p in sorted((FIX / "private").glob("*.yaml"))]
ONTO = Ontology.load(FIX / "ontology" / "profile.yaml")
LIM = Limits(provider_backoff_s=0.0)
Q = "How do temperature and time affect the yield and the rate constant for fixture_ridge?"
NO_SLEEP = lambda s: None  # noqa: E731
# B11 lists (same planted leaks as test_knowledge_gate): markers, blocked-document text, blocked urls, gate details
SECRETS = ["LEAK-RIDGE-7Q2", "LEAK-CAT-9X4", "hidden full table fixture_ridge"]
ANSWER_TEXT = ["ridge apex", "apex conditions", "preprint version", "we map the correlated", "accepted manuscript",
               "jointly maximize yield and ton", "def ridge", "source listing", "ridge grid dump", "cheat sheet",
               "recipe from the ridge paper", "answer sheet"]
BLOCKED_URLS = ["papers/ridge-answer", "2601.00001", "papers/catalyst-answer", "repository.fixture.example",
                "code.fixture.example", "data.fixture.example", "blog.fixture.example"]
GATE_DETAILS = ["secret_marker", "too_short_or_unknown_source"]
QUERIES = ["fixture ridge yield temperature time", "optimal catalyst loading fixture coupling TON",
           "hidden full table grid", "ridge recipe apex preprint", "simulator source mirror", "forum reactor settings",
           "review ridges catalysts", "task statement yield of 90 %", "Arrhenius thermostat manual"]


class FailingEmbedder(HashEmbedder):
    def embed(self, texts, kind):
        return EmbeddingResult(vectors=[], model=self.model, dimensions=self.dimensions, status="infra_error")


class RaisingSearch(FixtureSearchProvider):
    def search(self, query, max_results):
        raise OSError("connection reset")


class OtherModel(FixtureProvider):
    def generate(self, req):
        return super().generate(req).model_copy(update={"model_returned": "some-other-model"})


def make(tmp_path, cond, policy=None, *, limits=LIM, provider=None, web=None, embedder=None):
    """One condition's state dir (knowledge + memory), advisor and captured provider."""
    d = tmp_path / cond
    quiet = CallContext(sink=lambda e: None)
    if cond == "product":
        build_initial_state(d, "product", FIX / "corpus", FixtureMarkerChecker(), HashEmbedder(), BUNDLES, quiet,
                            set_scope="smoke", ontology=ONTO, extractor=FixtureKGExtractor())
        store = KnowledgeStore(d, FixtureMarkerChecker(), BUNDLES, "smoke", embedder=embedder or HashEmbedder(),
                               ontology=ONTO)
    else:
        build_initial_state(d, "baseline", FIX / "corpus", FixtureMarkerChecker(), None, BUNDLES, quiet,
                            set_scope="smoke", baseline_initial=FIX / "corpus" / "general_background.md")
        store = KnowledgeStore(d, FixtureMarkerChecker(), BUNDLES, "smoke")
    search = GatedSearch(store, web or FixtureSearchProvider(FIX / "web_corpus"), max_results=50,
                         max_chars=limits.max_tool_response_chars)
    provider = provider or FixtureProvider(policy=policy or make_fixture_advisor_policy(cond))
    ms = MemoryScope(run_id="r", condition=cond, set_id="smoke", set_rep=1)
    role = PROFILE.roles.advisor_product if cond == "product" else PROFILE.roles.advisor_baseline
    if cond == "product":
        memory = ProductMemory(d, ms, tasks=TASKS, limits=limits)
        advisor = ProductAdvisor(provider, role, limits, memory=memory, store=store, search=search, ontology=ONTO,
                                 sleep=NO_SLEEP)
    else:
        memory = BaselineTextMemory(d, ms, limits=limits, tasks=TASKS)
        advisor = BaselineAdvisor(provider, role, limits, memory=memory, store=store, search=search,
                                  initial_text=load_baseline_initial(d), sleep=NO_SLEEP)
    return SimpleNamespace(advisor=advisor, provider=provider, memory=memory, store=store, events=[])


def scope(cond, order=2):
    return RunScope(run_id="r", condition=cond, set_id="smoke", set_rep=1, episode_id=f"{cond}-r1-e{order:03d}",
                    episode_order=order, task_id="fixture_ridge", visit_index=order)


def obs(s, seq, t, tm, y):
    aid = f"{s.episode_id}:a{seq:03d}"
    return Observation(observation_id=f"obs:{aid}", action_id=aid, scope=s, parameters={"temperature": t, "time": tm},
                       results={"yield": y}, units={"yield": "%"}, simulator_id="fixture.ridge", simulator_version="1",
                       meets_success_criteria=RIDGE.is_success({"yield": y}), created_at="2026-09-28T00:00:00Z")


def request(s, observations=(), question=Q):
    return ConsultRequest(action_id=f"{s.episode_id}:a{len(observations) + 1:03d}", scope=s, question=question,
                          task=RIDGE, observations=list(observations), experiment_errors=[], prior_consults=[],
                          remaining_actions=49 - len(observations))


def consult(env, req, guard=None):
    ctx = CallContext(sink=env.events.append, scope_key=req.scope.memory_scope.key, episode_id=req.scope.episode_id,
                      action_id=req.action_id, guard=guard)
    return env.advisor.consult(req, ctx)


def root_payload(env):
    return json.loads(env.provider.requests[0].input[0]["text"])


def blob(x) -> str:
    return canonical_json([i.model_dump(mode="json") if hasattr(i, "model_dump") else i for i in x]).casefold()


def test_b17_baseline_has_no_product_knowledge_both_share_gated_search_and_no_optimizer(tmp_path):
    envs = {c: make(tmp_path, c) for c in ("baseline", "product")}
    before = {c: e.memory.state_hash() for c, e in envs.items()}
    outs = {c: consult(e, request(scope(c), [obs(scope(c), 1, 61.25, 20.0, 71.5)])) for c, e in envs.items()}
    for c, e in envs.items():
        out, reqs = outs[c], e.provider.requests
        assert out.status == "ok" and out.response.validation_issues == [], (c, out)
        assert {r.role for r in reqs} == {f"advisor_{c}"} and len(reqs) == 2          # search round trip + answer
        assert [t.name for t in reqs[0].tools] == ["search", "open"]                   # identical, gated, both
        assert reqs[0].response_schema is None and reqs[1].previous_interaction_id == "fixture-int-1"
        assert any(i.startswith("web:") for i in out.response.cited_source_ids)        # a gated tool result was cited
        assert out.response.candidates[0].within_public_constraints is True
        assert e.memory.state_hash() == before[c]                                      # I3: memory is read-only here
        assert all(ev.action_id == f"{c}-r1-e002:a002" for ev in e.events)
    b, p = root_payload(envs["baseline"]), root_payload(envs["product"])
    assert "evidence_cards" not in b and "card:" not in blob(envs["baseline"].provider.requests)
    assert b["initial_text"]["source_id"] == "initial:text"
    kinds = {c["kind"] for c in p["evidence_cards"]}
    assert {"literature", "current_observation", "code_calculation"} <= kinds
    assert any((c.get("derivation") or {}).get("method") == "kg_path" for c in p["evidence_cards"])
    assert all("access" not in c for c in p["evidence_cards"])                         # model_projection only
    ev = {c: {x.kind for x in e.events} for c, e in envs.items()}
    assert {"llm_call", "search"} <= ev["baseline"] and "retrieval" not in ev["baseline"]
    assert {"llm_call", "search", "retrieval", "embedding"} <= ev["product"]
    # U4: no optimizer tool anywhere; the researcher's own analysis tools are unchanged and condition-free
    names = {t.name for e in envs.values() for r in e.provider.requests for t in r.tools}
    assert names == {"search", "open"}
    view = ResearcherView(task=RIDGE, actions_used=0, remaining_actions=50, history=[], notes=[])
    assert {t.name for t in AnalysisTools(view).specs()} == {"describe", "fit", "predict", "nearest"}
    assert baseline_mod.PROMPT_HASH != product_mod.PROMPT_HASH and baseline_mod.PROMPT_VERSION == product_mod.PROMPT_VERSION


def test_current_observations_reach_both_advisors_with_exact_values(tmp_path):
    for c in ("baseline", "product"):
        e = make(tmp_path, c)
        s = scope(c)
        o1, o2 = obs(s, 1, 61.25, 20.0, 87.123456789012), obs(s, 2, 100.0, 5.5, 12.000001)
        out = consult(e, request(s, [o1, o2]))
        cur = root_payload(e)["current_episode"]["observations"]
        assert [(x["observation_id"], x["parameters"], x["results"]) for x in cur] == \
            [(o.observation_id, o.parameters, o.results) for o in (o1, o2)]
        text = e.provider.requests[0].input[0]["text"]
        assert "87.123456789012" in text and "12.000001" in text
        # the fixture answer quotes the best current observation exactly; the numeric check accepts it
        assert out.response.cited_observation_ids[0] == o1.observation_id and out.response.validation_issues == []


def greedy_leak_policy(req):
    """Stateless test policy: every leaky query, then open everything returned (+ expand retrieved chunks)."""
    tools = [t for t in req.input if t.get("role") == "tool"]
    if not req.tools or any(t["call_id"].startswith("o") for t in tools):
        return json.dumps({"answer": "done", "reasoning": "r", "limitations": "l"})
    if not tools:
        cards = json.loads(req.input[0]["text"]).get("evidence_cards", [])
        chunks = [c["card_id"].removeprefix("card:") for c in cards if c["card_id"].startswith("card:doc:")]
        return [FunctionCall(call_id=f"s{i}", name="search", arguments={"query": q}) for i, q in enumerate(QUERIES)] + \
            [FunctionCall(call_id=f"x{i}", name="open", arguments={"source_id": c, "expand": "document"})
             for i, c in enumerate(chunks)]
    ids = sorted({r["source_id"] for t in tools for r in t["result"].get("results", []) if r["status"] == "available"})
    return [FunctionCall(call_id=f"o{i}", name="open", arguments={"source_id": x}) for i, x in enumerate(ids)]


def test_b11_leaky_web_corpus_never_reaches_either_advisor_model(tmp_path):
    lim = Limits(provider_backoff_s=0.0, advisor_max_tool_calls=80)
    q = "Where do yield, temperature and time peak for the fixture ridge? Any recipe, preprint or full table?"
    for c in ("baseline", "product"):
        e = make(tmp_path, c, limits=lim, provider=FixtureProvider(policy=greedy_leak_policy, chain_ids=False))
        s = scope(c)
        assert consult(e, request(s, [obs(s, 1, 61.25, 20.0, 71.5)], question=q)).status == "ok"
        text = blob(e.provider.requests)
        for bad in SECRETS + ANSWER_TEXT + BLOCKED_URLS + GATE_DETAILS:
            assert bad.casefold() not in text, (c, bad)
        results = [r for req in e.provider.requests for t in req.input if t.get("role") == "tool"
                   for r in t["result"].get("results", [])]
        titles = {r["title"] for r in results}
        assert "Locating the yield ridge of the fixture reaction" not in titles
        # not vacuous: allowed pages came back and were opened through the same gated path
        assert {"A practical review of two-variable yield ridges", "Fixture ridge task statement"} <= titles
        assert any(r["status"] == "unavailable" for r in results)            # the leaky forum page on open
        events = blob(e.events)
        assert not any(x.casefold() in events for x in SECRETS + GATE_DETAILS)


def finalize_past(env, cond, t, tm, y):
    s1 = scope(cond, order=1)
    o = obs(s1, 2, t, tm, y)
    c = ConsultExchange(action_id=f"{s1.episode_id}:a001", question="what first?",
                        response=AdvisorResponse(answer="Start near the centre.", reasoning="r", limitations="l"))
    res = env.memory.finalize_episode(FinalizeInput(event_id=finalize_event_id(s1), scope=s1, outcome=Outcome.success,
                                                    observations=[o], errors=[], consults=[c]),
                                      CallContext(sink=lambda e: None))
    assert res.status == "done"
    return o


def test_b09_b07_past_episode_reaches_the_next_consult_of_its_own_condition_only(tmp_path):
    envs = {c: make(tmp_path, c) for c in ("baseline", "product")}
    past = {"baseline": finalize_past(envs["baseline"], "baseline", 83.0, 37.0, 95.25),
            "product": finalize_past(envs["product"], "product", 70.0, 45.0, 91.5)}
    for c, e in envs.items():
        other = past["product" if c == "baseline" else "baseline"]
        out = consult(e, request(scope(c)))                        # new episode: the researcher has no observations
        cand = out.response.candidates[0]
        assert cand.parameters == past[c].parameters and cand.revisit == "past_episode_repeat"
        assert past[c].observation_id in out.response.cited_observation_ids and out.response.validation_issues == []
        text = blob(e.provider.requests)
        assert past[c].observation_id.casefold() in text
        assert other.observation_id.casefold() not in text and f"{other.scope.condition}-r1" not in text
    cards = root_payload(envs["product"])["evidence_cards"]
    assert any(k["kind"] == "past_observation" and k["observation_ids"] == [past["product"].observation_id]
               for k in cards)
    assert any(k["kind"] == "agent_inference" and "Start near the centre." in k["summary"] for k in cards)
    assert "Start near the centre." in root_payload(envs["baseline"])["own_memory"]


def test_b16_validation_issues_surface_after_one_bounded_repair(tmp_path):
    s = scope("baseline")
    o1 = obs(s, 1, 61.25, 20.0, 87.123)
    claim = "Raising the jacket temperature to 90 degC will push the yield above 95 %."
    tool_id = []

    def bad(req):
        if req.tools and not any(t.get("role") == "tool" for t in req.input):
            return [FunctionCall(call_id="s", name="search", arguments={"query": "Arrhenius rate constant temperature"})]
        return json.dumps({"answer": f"In {o1.observation_id} the yield was 50.0 %. {claim}",
                           "cited_source_ids": ["web:not-delivered", tool_id[0]],
                           "cited_observation_ids": [o1.observation_id], "reasoning": "r", "limitations": "l"})

    e = make(tmp_path, "baseline", bad)
    arrhenius = next(v for v in e.advisor.search.search("Arrhenius rate constant temperature", CallContext(
        sink=lambda x: None)) if v.title.startswith("Arrhenius"))
    tool_id.append(arrhenius.source_id)
    out = consult(e, request(s, [o1]))
    issues = out.response.validation_issues
    assert out.status == "ok" and len(e.provider.requests) == 3                  # tool round trip + answer + 1 repair
    assert any("web:not-delivered was not delivered" in i for i in issues)
    assert any("yield=50.0 attributed to observation" in i for i in issues)
    repair = e.provider.requests[2]
    assert repair.tools == [] and repair.response_schema is not None
    assert "web:not-delivered" in json.loads(repair.input[0]["text"])["repair"]["issues"][0]
    # the delivered Arrhenius page is a mechanically valid citation, yet it does not support the claim
    assert not any(arrhenius.source_id in i for i in issues)
    assert check_citation_support(claim, arrhenius.text, TokenOverlapJudge()).supported is False


def test_tool_calls_and_repairs_are_bounded(tmp_path):
    calls = []

    def always_search(req):
        calls.append(1)
        if req.tools and len(calls) < 10:          # self-stop so a missing bound fails instead of hanging
            return [FunctionCall(call_id=f"c{n}", name="search", arguments={"query": "yield temperature"})
                    for n in range(2)]
        return json.dumps({"answer": "a", "reasoning": "r", "limitations": "l"})

    e = make(tmp_path, "baseline", always_search, limits=Limits(provider_backoff_s=0.0, advisor_max_tool_calls=3))
    assert consult(e, request(scope("baseline"))).status == "ok"
    results = [t["result"] for r in e.provider.requests for t in r.input if t.get("role") == "tool"]
    assert sum("results" in x for x in results) == 3 and results[-1] == {"error": "tool call limit for this consultation reached"}
    assert [bool(r.tools) for r in e.provider.requests] == [True, True, False]
    assert e.provider.requests[-1].response_schema is not None

    for repairs in (0, 1):
        e = make(tmp_path / f"r{repairs}", "baseline", lambda req: "not json",
                 limits=Limits(provider_backoff_s=0.0, advisor_max_repairs=repairs))
        out = consult(e, request(scope("baseline")))
        assert out.status == "partial" and out.response.status == "partial" and out.response.answer == PARTIAL_ANSWER
        assert len(e.provider.requests) == 1 + repairs and out.response.candidates == []


def test_status_mapping_infra_partial_search_unavailable_and_propagation(tmp_path):
    fx = make_fixture_advisor_policy("baseline")
    e = make(tmp_path / "infra", "baseline", provider=FixtureProvider(policy=fx, script=["infra_error"] * 3),
             limits=Limits(provider_backoff_s=0.0, provider_max_attempts=3))
    out = consult(e, request(scope("baseline")))
    assert out.status == "infra_error" and out.response is None and "fixture" not in out.error
    assert [x.status for x in e.events if x.kind == "llm_call"] == ["infra_error"] * 3     # finite, all costed
    for script in (["incomplete"], ["ok", "refusal"]):
        e = make(tmp_path / script[-1], "baseline", provider=FixtureProvider(policy=fx, script=script))
        out = consult(e, request(scope("baseline")))
        assert (out.status, out.response.status, out.response.answer) == ("partial", "partial", PARTIAL_ANSWER)
    e = make(tmp_path / "search", "baseline", web=RaisingSearch(FIX / "web_corpus"))
    out = consult(e, request(scope("baseline")))
    assert out.status == "ok" and e.provider.requests[1].input[0]["result"] == {"error": "search unavailable"}
    e = make(tmp_path / "embed", "product", embedder=FailingEmbedder())
    out = consult(e, request(scope("product")))
    assert out.status == "infra_error" and e.provider.requests == []                       # no silent RAG-less answer
    with pytest.raises(ModelChangedError):
        consult(make(tmp_path / "model", "baseline", provider=OtherModel(policy=fx)), request(scope("baseline")))
    with pytest.raises(CapExceeded):
        consult(make(tmp_path / "cap", "baseline"), request(scope("baseline")), guard=CostGuard(CostCaps(max_calls=0)))


def test_candidate_revisit_marking_current_past_new(tmp_path):
    e = make(tmp_path, "baseline", lambda req: json.dumps({
        "answer": "a", "reasoning": "r", "limitations": "l",
        "candidates": [{"parameters": {"temperature": 61, "time": 20}}, {"parameters": {"temperature": 83, "time": 37}},
                       {"parameters": {"temperature": 90.5, "time": 12.0}},
                       {"parameters": {"temperature": 500, "time": 12.0}}]}),
        limits=Limits(provider_backoff_s=0.0, advisor_max_tool_calls=0))
    finalize_past(e, "baseline", 83.0, 37.0, 95.25)
    s = scope("baseline")
    out = consult(e, request(s, [obs(s, 1, 61.0, 20.0, 71.5)]))
    c = out.response.candidates
    assert [x.revisit for x in c] == ["current_episode_repeat", "past_episode_repeat", "new", "new"]
    assert [x.within_public_constraints for x in c] == [True, True, True, False]
    assert any("candidate 3" in i for i in out.response.validation_issues)


LONG_URL = "https://www.example-lab-notes.org/articles/2026/reaction-kinetics/temperature-time-yield-optimisation-guide"
LONG_TITLE = "Temperature and time in batch reactions: a practical guide to yield optimisation"
LONG_BODY = " ".join(f'Paragraph {i}: raising the "jacket" temperature (온도)\tspeeds the reaction but may lower '
                     f'selectivity.' for i in range(120))                      # ~12k chars, JSON escapes, non-ASCII


class LongPage:
    name = "longpage"

    def search(self, query, max_results):
        return [RawHit(url=LONG_URL, title=LONG_TITLE, snippet="How temperature and time shape yield in batch reactions.",
                       provider=self.name, retrieved_at="x", query=query)]

    def fetch(self, url):
        return RawHit(url=LONG_URL, title=LONG_TITLE, content=LONG_BODY, provider=self.name, retrieved_at="x")


def test_b17_a_long_opened_page_is_truncated_to_the_tool_limit_never_dropped(tmp_path):
    def policy(req):                                   # stateful chain: each continuation carries only new turns
        tools = [t for t in req.input if t.get("role") == "tool"]
        if req.tools and not tools:
            return [FunctionCall(call_id="s", name="search", arguments={"query": "temperature time yield"})]
        sid = tools[-1]["result"]["results"][0]["source_id"]
        if req.tools and tools[-1]["call_id"] == "s":
            return [FunctionCall(call_id="o", name="open", arguments={"source_id": sid})]
        return json.dumps({"answer": "a", "cited_source_ids": [sid], "reasoning": "r", "limitations": "l"})

    for c in ("baseline", "product"):
        e = make(tmp_path, c, policy, web=LongPage())
        out = consult(e, request(scope(c)))
        opened = next(t["result"] for r in e.provider.requests for t in r.input if t.get("call_id") == "o")
        [v] = opened["results"]
        assert len(canonical_json(opened)) <= LIM.max_tool_response_chars and "omitted" not in opened, c
        assert v["status"] == "available" and v["truncated"] is True and len(v["text"]) > 6000
        assert LONG_BODY.startswith(v["text"])                                  # an exact prefix of the gated text
        assert out.status == "ok" and out.response.validation_issues == []      # the opened page counts as delivered
    # results that cannot fit even without text are counted, never dropped silently
    many = [{"source_id": f"web:{i:016d}", "title": "t" * 100, "text": "x" * 50, "url": "https://e.org/" + "p" * 200,
             "status": "available"} for i in range(50)]
    r = fit(many, 8000)
    assert len(canonical_json(r)) <= 8000 and 0 < len(r["results"]) < 50 and r["omitted"] == 50 - len(r["results"])


def test_b16_b19_a_failed_repair_keeps_the_complete_checked_answer(tmp_path):
    ans = json.dumps({"answer": "Try temperature 80 and time 30.", "cited_source_ids": ["web:never-delivered"],
                      "reasoning": "r", "limitations": "l", "candidates": [{"parameters": {"temperature": 80.0, "time": 30.0}}]})
    replies = iter([ans, "Sorry, corrected: temperature 80, time 30"])
    cases = {"unparseable": FixtureProvider(policy=lambda r: next(replies)),
             **{s: FixtureProvider(policy=lambda r: ans, script=["ok", s]) for s in ("refusal", "incomplete", "infra_error")}}
    lim = Limits(provider_backoff_s=0.0, advisor_max_tool_calls=0, advisor_max_repairs=1, provider_max_attempts=1)
    for name, prov in cases.items():
        e = make(tmp_path / name, "baseline", provider=prov, limits=lim)
        out = consult(e, request(scope("baseline")))
        assert (out.status, len(prov.requests)) == ("ok", 2), name                 # one repair was attempted
        assert out.response.answer == "Try temperature 80 and time 30." and out.response.candidates[0].revisit == "new"
        assert out.response.validation_issues == ["cited source web:never-delivered was not delivered in this consultation"]


def test_b16_researcher_short_observation_ids_are_checked_against_the_ledger(tmp_path):
    s = scope("baseline")
    o1 = obs(s, 1, 61.25, 20.0, 71.5)                # researcher sees obs:a001
    e = make(tmp_path, "baseline", lambda req: json.dumps({
        "answer": "In obs:a001 the yield was 50.0. In obs:e001:a002 the yield was 95.25.",
        "cited_observation_ids": ["obs:a001", "obs:e001:a002"], "reasoning": "r", "limitations": "l"}),
        limits=Limits(provider_backoff_s=0.0, advisor_max_tool_calls=0, advisor_max_repairs=0))
    past = finalize_past(e, "baseline", 83.0, 37.0, 95.25)   # obs:baseline-r1-e001:a002 -> researcher sees obs:e001:a002
    out = consult(e, request(s, [o1], question="obs:a001 gave yield 71.5. Where next?"))
    assert out.response.cited_observation_ids == [o1.observation_id, past.observation_id]
    assert out.response.validation_issues == [
        f"yield=50.0 attributed to observation {o1.observation_id} does not match the recorded value 71.5"]


def test_b17_product_query_expansion_is_bounded_by_max_query_expansions(tmp_path):
    e = make(tmp_path, "product", lambda req: json.dumps({"answer": "a", "reasoning": "r", "limitations": "l"}))
    e.store.retrieval = RetrievalConfig(query_expansion=True)          # profile knowledge.retrieval.query_expansion
    kw = dict(memory=e.memory, store=e.store, search=e.advisor.search, ontology=ONTO, sleep=NO_SLEEP)
    role = PROFILE.roles.advisor_product
    q = "How does temperature affect yield and the rate constant?"
    for n, rankings in ((0, 2), (1, 3)):     # 0 -> original BM25 + dense only; 1 -> + ONE expanded BM25
        e.events.clear()
        e.advisor = ProductAdvisor(e.provider, role, Limits(provider_backoff_s=0.0, advisor_max_tool_calls=0,
                                                            max_query_expansions=n), **kw)
        consult(e, request(scope("product"), question=q))
        [d] = [ev.detail for ev in e.events if ev.kind == "retrieval"]
        assert d["expanded"] is (n > 0) and d["rankings"] == rankings
