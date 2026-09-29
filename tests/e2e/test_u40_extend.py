"""U40 `extend-plan` edge cases from the independent review (offline fixture: contract checks only)."""
import json
import sqlite3
from pathlib import Path

import pytest
import yaml

import labgene.researcher.fixture_policy as fp
from labgene import cli
from labgene.app import Run
from labgene.contracts import PrivateTaskAssets
from labgene.costs import CallContext
from labgene.knowledge.gate import FixtureMarkerChecker
from labgene.knowledge.gated import GatedSearch
from labgene.knowledge.regate import regate_state
from labgene.knowledge.search import FixtureSearchProvider
from labgene.knowledge.store import KnowledgeStore

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.delenv("LABGENE_FAULT", raising=False)


def setup(tmp_path, conditions=("baseline", "product"), main_episodes=("fixture_ridge", "fixture_catalyst"),
          main_budget=150):
    p = yaml.safe_load((REPO / "configs/offline.yaml").read_text(encoding="utf-8"))
    p["paths"] = {"artifacts_dir": str(tmp_path / "artifacts")}
    prof, pilot, main = tmp_path / "profile.yaml", tmp_path / "pilot.yaml", tmp_path / "main.yaml"
    prof.write_text(yaml.safe_dump(p), encoding="utf-8")
    common = {"conditions": list(conditions), "mode": "progression", "parallel_conditions": len(conditions) > 1}
    pilot.write_text(yaml.safe_dump({"set_id": "pilot", "episodes": ["fixture_ridge"], "action_budget": 50,
                                     **common}), encoding="utf-8")
    main.write_text(yaml.safe_dump({"set_id": "main", "episodes": list(main_episodes), "action_budget": main_budget,
                                    **common}), encoding="utf-8")
    return str(prof), str(pilot), str(main), str(tmp_path / "artifacts")


def test_u40_a_truncated_pilot_attempt_keeps_its_budget_after_the_extension(tmp_path, monkeypatch):
    prof, pilot, main, art = setup(tmp_path, conditions=("product",))
    orig, bad = fp.choose_action, {"n": 0}

    def flaky(view, behaviour):     # the first attempt ends in a protocol error after 3 observations
        if bad["n"] < 3 and sum(h["kind"] == "observation" for h in view["history"]) >= 3:
            bad["n"] += 1
            return "not an action"
        return orig(view, behaviour)
    monkeypatch.setattr(fp, "choose_action", flaky)
    assert cli.main(["run-set", "--profile", prof, "--plan", pilot, "--run-id", "p"]) == 0
    monkeypatch.setattr(fp, "choose_action", orig)
    with sqlite3.connect(Path(art) / "p" / "ledger.sqlite") as con:
        budgets = [json.loads(s)["action_budget"] for (s,) in con.execute("SELECT scope_json FROM episodes "
                                                                          "ORDER BY episode_order")]
    assert budgets[0] == 50 and budgets[1] < 50                     # the second pilot attempt was truncated
    assert cli.main(["extend-plan", "--run-id", "p", "--artifacts-dir", art, "--plan", main]) == 0
    assert cli.main(["resume", "--run-id", "p", "--artifacts-dir", art]) == 0


def test_u40_a_kill_between_manifest_and_plan_fails_closed_and_a_rerun_finishes(tmp_path, monkeypatch):
    prof, pilot, main, art = setup(tmp_path)
    assert cli.main(["run-set", "--profile", prof, "--plan", pilot, "--run-id", "p"]) == 0
    orig = Run.write_manifest

    def killed(self):
        orig(self)
        if "fixture_catalyst" in self.manifest.tasks:
            raise KeyboardInterrupt("killed after the manifest, before the plan")
    monkeypatch.setattr(Run, "write_manifest", killed)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["extend-plan", "--run-id", "p", "--artifacts-dir", art, "--plan", main])
    monkeypatch.setattr(Run, "write_manifest", orig)
    assert Run.load(Path(art) / "p").plan.episodes == ["fixture_ridge"]     # still loads, under the old plan
    assert cli.main(["resume", "--run-id", "p", "--artifacts-dir", art]) != 0   # the re-gated states refuse it
    assert cli.main(["extend-plan", "--run-id", "p", "--artifacts-dir", art, "--plan", main]) == 0
    assert cli.main(["resume", "--run-id", "p", "--artifacts-dir", art]) == 0


def test_u40_a_budget_only_extension_needs_no_regate(tmp_path):
    prof, pilot, _, art = setup(tmp_path)
    more = tmp_path / "more.yaml"
    more.write_text(yaml.safe_dump({"set_id": "pilot", "episodes": ["fixture_ridge"], "action_budget": 100,
                                    "conditions": ["baseline", "product"], "mode": "progression"}), encoding="utf-8")
    assert cli.main(["run-set", "--profile", prof, "--plan", pilot, "--run-id", "p"]) == 0
    assert cli.main(["extend-plan", "--run-id", "p", "--artifacts-dir", art, "--plan", str(more)]) == 0
    ext = json.loads(next((Path(art) / "p").glob("extend-*.json")).read_text(encoding="utf-8"))
    assert all(v["knowledge"] == "unchanged gate identity" for v in ext["states"].values())


def test_u40_web_sources_are_rechecked_with_the_identity_admission_used(tmp_path):
    ridge, cat = [PrivateTaskAssets.model_validate(yaml.safe_load((FIX / "private" / f"fixture_{n}.yaml").read_text(
        encoding="utf-8"))).answer_bundle for n in ("ridge", "catalyst")]
    ctx = CallContext(sink=lambda e: None)
    web = tmp_path / "web"
    import shutil
    shutil.copytree(FIX / "web_corpus", web)
    (web / "catalyst_doi_mirror.yaml").write_text(yaml.safe_dump({   # blocked for catalyst only via its DOI
        "url": "https://mirror.fixture.example/items/coupling-notes", "title": "Coupling notes mirror",
        "doi": "10.5555/fixture.catalyst.answer",
        "snippet": "Mirror of the coupling notes on catalyst loading and residence time for the screen.",
        "content": "Catalyst C at 1.2 mol% loading, 90 degC and 540 s residence time meets both targets."}),
        encoding="utf-8")

    def mirror_visible(bundles):
        s = KnowledgeStore(tmp_path / "state", FixtureMarkerChecker(), bundles, "pilot")
        g = GatedSearch(s, FixtureSearchProvider(web))
        hit = [v for v in g.search("coupling notes mirror", ctx) if "mirror.fixture.example" in (v.url or "")]
        opened = g.open(hit[0].source_id, ctx) if hit else None
        s.close()
        return bool(hit), opened is not None and "540 s" in (opened.text or "")
    assert mirror_visible([ridge]) == (True, True)                   # admitted and fetched under the pilot bundle
    rep = regate_state(tmp_path / "state", FixtureMarkerChecker(), [ridge, cat], "pilot", ctx,
                       baseline_source="corpus:x.md", corpus_dir=FIX / "corpus")
    assert any(w.get("reason") == "identity" for w in rep["withdrawn"])
    assert mirror_visible([ridge, cat]) == (False, False)
