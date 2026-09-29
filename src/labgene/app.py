"""Wiring (T07): profile + set plan -> run directory, initial condition states, per-scope components.

Run directory layout (artifacts/<run_id>/):
    manifest.json         RunManifest (+ task_defs, prompt versions) — public, no private assets
    profile.yaml, set_plan.yaml   exact copies used by resume
    ledger.sqlite         harness ledger (actions, observations, costs, finalization)
    prebuild_costs.jsonl  CostEvents of initial-state construction (phase=prebuild)
    sessions.jsonl        one line per CLI session (code version, wall time, cumulative guard)
    guard.json            cumulative cost-cap totals (open reservations counted), primed on every resume
    root.txt              repo root the relative profile paths resolve against
    build/<condition>/    initial-state build dirs (kept)
    initial/<condition>/  snapshots restored at every set start (decisions I4)
    state/<condition>/<set_id>/rep<k>/   live condition states
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import FixtureBehaviour, Profile, SetPlan, load_profile, load_public_task, load_set_plan, preflight_problems
from .contracts import CostEvent, MemoryScope, PrivateTaskAssets, PublicTask, RunManifest, payload_hash, sha256_text
from .costs import CallContext, CostGuard
from .harness.ledger import Ledger
from .harness.runner import ConditionComponents, SetRunner, SetRunStatus

SPEC_PATH = "docs/superpowers/specs/2026-09-28-labgene-harness-design.md"
TASK_VALIDATION_DIR = "docs/implementation/task-validation"


# ---------------------------------------------------------------- inputs

def load_tasks(profile: Profile, plan: SetPlan) -> dict[str, PublicTask]:
    return {t: load_public_task(profile, t) for t in dict.fromkeys(plan.episodes)}


def load_private(profile: Profile, task_ids: list[str]) -> dict[str, PrivateTaskAssets]:
    """Evaluator/gate side only. Never passed to a model-facing component."""
    d = profile.resolve(profile.paths.private_dir)
    out = {}
    for t in task_ids:
        p = d / f"{t}.yaml"
        if not p.exists():
            raise FileNotFoundError(f"no private assets for task {t} in {profile.paths.private_dir}")
        out[t] = PrivateTaskAssets.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
    return out


def code_version(root: Path) -> dict[str, Any]:
    files = sorted((root / "src" / "labgene").rglob("*.py"))
    tree = sha256_text("".join(f"{p.relative_to(root).as_posix()}:{sha256_text(p.read_text(encoding='utf-8'))}\n"
                               for p in files))

    def git(*a: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, timeout=20,
                                  check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "source_tree_sha256": tree}


def spec_sha256(root: Path) -> str | None:
    p = root / SPEC_PATH
    return sha256_text(p.read_text(encoding="utf-8")) if p.exists() else None


def prompt_versions() -> dict[str, Any]:
    from .researcher import agent
    out: dict[str, Any] = {"researcher": {"version": agent.PROMPT_VERSION, "sha256": agent.PROMPT_HASH}}
    from .advisors import baseline, product
    for name, mod in (("advisor_baseline", baseline), ("advisor_product", product)):
        out[name] = {"version": getattr(mod, "PROMPT_VERSION", None), "sha256": getattr(mod, "PROMPT_HASH", None)}
    return out


# ---------------------------------------------------------------- components

class _NoSearch:
    name = "none"
    development_only = False

    def search(self, query: str, max_results: int) -> list:
        return []

    def fetch(self, url: str) -> None:
        return None


def _llm(profile: Profile, role: str, fixture_policy: Any = None):
    from .providers.call import build_llm
    cfg = getattr(profile.roles, role)
    return build_llm(cfg, fixture_policy=fixture_policy, timeout_s=profile.limits.provider_timeout_s)


def _configured(obj: Any, r) -> Any:
    """Internal LLM components get the role's reasoning settings and accepted returned-model aliases (§11.5)."""
    obj.reasoning_effort, obj.thinking_level = r.reasoning_effort, r.thinking_level
    obj.allowed_models = tuple(r.allowed_returned_models)
    return obj


def build_checker(profile: Profile):
    from .knowledge.gate import FixtureMarkerChecker, LLMLeakageChecker
    r = profile.roles.leakage_gate
    if r.provider == "fixture":
        return FixtureMarkerChecker()
    return _configured(LLMLeakageChecker(_llm(profile, "leakage_gate"), r.model,
                                         max_output_tokens=r.max_output_tokens or 512), r)


def build_reranker(profile: Profile):
    """retrieval.reranker=llm uses the internal_summarizer role config (adoption is a T08 decision)."""
    from .knowledge.retrieval import LLMReranker
    if profile.knowledge.retrieval.reranker != "llm":
        return None
    r = profile.roles.internal_summarizer
    return _configured(LLMReranker(_llm(profile, "internal_summarizer"), r.model,
                                   max_output_tokens=r.max_output_tokens or 512), r)


def build_embedder(profile: Profile):
    from .providers.call import build_embedder as _be
    return _be(profile.roles.embedder, timeout_s=profile.limits.provider_timeout_s)


class _CappedSearch:
    """Paid search/fetch requests are checked against cost_caps.max_search_calls BEFORE each request."""

    def __init__(self, inner: Any, guard: CostGuard):
        self.inner, self.guard, self.name = inner, guard, inner.name
        self.development_only = getattr(inner, "development_only", False)

    def search(self, query: str, max_results: int) -> list:
        self.guard.reserve_search()
        return self.inner.search(query, max_results)

    def fetch(self, url: str):
        self.guard.reserve_search()
        return self.inner.fetch(url)


def build_search_provider(profile: Profile, guard: CostGuard | None = None):
    from .knowledge.search import FixtureSearchProvider, GoogleCSESearchProvider
    s = profile.search
    if s.provider == "fixture":
        return FixtureSearchProvider(profile.resolve(s.fixture_corpus))
    if s.provider == "google_cse":
        inner = GoogleCSESearchProvider(timeout_s=profile.limits.provider_timeout_s)
        return _CappedSearch(inner, guard) if guard is not None else inner
    return _NoSearch()


def build_ontology(profile: Profile):
    from .knowledge.kg import Ontology
    p = profile.resolve(profile.knowledge.ontology_profile)
    return Ontology.load(p) if p and p.exists() else None


def build_extractor(profile: Profile):
    from .knowledge.kg import FixtureKGExtractor, LLMKGExtractor
    r = profile.roles.kg_extractor
    if r.provider == "fixture":
        return FixtureKGExtractor()
    return _configured(LLMKGExtractor(_llm(profile, "kg_extractor"), r.model,
                                      max_output_tokens=r.max_output_tokens or 2048), r)


def development_only_components(profile: Profile) -> list[str]:
    """Everything in this profile that is a development-only stand-in (reported in the manifest; refused by freeze)."""
    out = [f"roles.{n}" for n, r in profile.roles if r.provider == "fixture"]
    if profile.search.provider == "fixture":
        out.append("search.fixture")
    if profile.limits.development_only:
        out.append("limits.development_only")
    onto = build_ontology(profile)
    if onto is None:
        out.append("knowledge.ontology_profile(missing)")
    elif onto.development_only:
        out.append(f"ontology:{onto.profile_id}")
    return out


# ---------------------------------------------------------------- run

@dataclass
class Run:
    run_dir: Path
    profile: Profile
    plan: SetPlan
    tasks: dict[str, PublicTask]
    private: dict[str, PrivateTaskAssets]
    manifest: RunManifest
    guard: CostGuard = field(init=False)

    def __post_init__(self) -> None:
        """The guard is primed from guard.json (cumulative, open reservations counted) BEFORE any call of this
        session - prebuild included - and persists after every reserve/settle, so caps hold across resumes and
        hard kills (decisions I13, I19)."""
        self._t0 = time.monotonic()
        self.guard = CostGuard(self.profile.cost_caps, enforce=True,   # unset caps are simply not checked
                               on_change=self._persist_guard)
        gj = self.run_dir / "guard.json"
        if gj.exists():
            self.guard.restore(json.loads(gj.read_text(encoding="utf-8")))

    def _persist_guard(self, totals: dict[str, Any]) -> None:
        tmp = self.run_dir / "guard.json.tmp"
        tmp.write_text(json.dumps(totals), encoding="utf-8")
        tmp.replace(self.run_dir / "guard.json")

    # ------------------------------------------------ creation / loading
    @classmethod
    def create(cls, profile: Profile, plan: SetPlan, run_id: str) -> "Run":
        root = Path(profile.root)
        run_dir = profile.resolve(profile.paths.artifacts_dir) / run_id
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run {run_id} already exists at {run_dir}; use `resume --run-id {run_id}`")
        tasks = load_tasks(profile, plan)
        private = load_private(profile, list(tasks))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "root.txt").write_text(str(Path(profile.root).resolve()), encoding="utf-8")
        prof = profile.model_dump(mode="json")
        (run_dir / "profile.yaml").write_text(yaml.safe_dump(prof, sort_keys=False, allow_unicode=True), encoding="utf-8")
        (run_dir / "set_plan.yaml").write_text(yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False),
                                               encoding="utf-8")
        m = RunManifest(
            run_id=run_id, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            execution_mode=profile.execution_mode, spec_sha256=spec_sha256(root), code_version=code_version(root),
            profile_hash=profile.hash, profile=prof, set_plan_hash=plan.hash, set_plan=plan.model_dump(mode="json"),
            tasks={t: payload_hash(v) for t, v in tasks.items()}, initial_state_hashes={},
            models={n: r.model_dump(mode="json") for n, r in profile.roles},
            frozen=profile.execution_mode == "evaluation",   # run-evaluation only proceeds after the frozen-digest check
            development_only_fields=development_only_components(profile))
        if profile.execution_mode == "evaluation":
            fm = profile.resolve(profile.frozen_manifest)
            m.analysis_plan_hash = json.loads(fm.read_text(encoding="utf-8"))["analysis_plan_sha256"]
            m.evidence_hashes["frozen_manifest"] = sha256_text(fm.read_text(encoding="utf-8"))
        run = cls(run_dir, profile, plan, tasks, private, m)
        run.write_manifest()
        return run

    @classmethod
    def load(cls, run_dir: Path) -> "Run":
        rt = run_dir / "root.txt"
        root = Path(rt.read_text(encoding="utf-8").strip()) if rt.exists() else _repo_root(run_dir)
        profile = load_profile(run_dir / "profile.yaml", root=root)
        plan = load_set_plan(run_dir / "set_plan.yaml")
        m = RunManifest.model_validate_json((run_dir / "manifest.json").read_text(encoding="utf-8"))
        tasks = load_tasks(profile, plan)
        changed = [t for t, v in tasks.items() if payload_hash(v) != m.tasks.get(t)]
        if changed:
            raise ValueError(f"task definitions changed since the run started: {changed}; start a new run")
        return cls(run_dir, profile, plan, tasks, load_private(profile, list(tasks)), m)

    def write_manifest(self) -> None:
        self.manifest.task_defs = {t: v.model_dump(mode="json") for t, v in self.tasks.items()}
        self.manifest.prompts = prompt_versions()
        (self.run_dir / "manifest.json").write_text(self.manifest.model_dump_json(indent=2), encoding="utf-8")

    # ------------------------------------------------ prebuild
    def _prebuild_sink(self, e: CostEvent) -> None:
        with open(self.run_dir / "prebuild_costs.jsonl", "a", encoding="utf-8") as f:
            f.write(e.model_dump_json() + "\n")

    def bundles(self):
        return [a.answer_bundle for a in self.private.values()]

    def _pending_snapshot(self, cond: str) -> Path | None:
        """This condition's initial snapshot dir if it still has to be made (None if done). An interrupted snapshot
        (the manifest is written last) is kept aside and redone."""
        snap = self.run_dir / "initial" / cond
        if (snap / "manifest.json").exists():
            return None
        if snap.exists():
            n = 1
            while (aside := snap.parent / f"{cond}.partial-{n}").exists():
                n += 1
            snap.rename(aside)
        return snap

    def _record_initial_states(self) -> dict[str, str]:
        hashes = {c: json.loads((self.run_dir / "initial" / c / "manifest.json").read_text(encoding="utf-8"))
                  ["combined_sha256"] for c in self.plan.conditions}
        if self.manifest.initial_state_hashes and self.manifest.initial_state_hashes != hashes:
            raise RuntimeError("initial state snapshots differ from the run manifest")
        self.manifest.initial_state_hashes = hashes
        self.write_manifest()
        return hashes

    def build_initial_states(self) -> dict[str, str]:
        """Build + snapshot each condition's initial state once per run (idempotent: existing snapshots are kept).
        With knowledge.initial_state_from the snapshots are copied from that run instead (U39)."""
        from .knowledge.build import build_initial_state
        from .memory.snapshot import snapshot_state
        p, k = self.profile, self.profile.knowledge
        if k.initial_state_from:
            return self._import_initial_states(p.resolve(k.initial_state_from))
        checker, embedder, onto, extractor = build_checker(p), build_embedder(p), build_ontology(p), build_extractor(p)
        for cond in self.plan.conditions:
            snap = self._pending_snapshot(cond)
            if snap is None:
                continue
            ctx = CallContext(sink=self._prebuild_sink, phase="prebuild", guard=self.guard,
                              scope_key=f"{self.manifest.run_id}/{cond}/{self.plan.set_id}/initial")
            build = self.run_dir / "build" / cond
            info = build_initial_state(
                build, cond, p.resolve(k.corpus_dir), checker, embedder, self.bundles(), ctx, set_scope=self.plan.set_id,
                baseline_initial=p.resolve(k.baseline_initial_text), ontology=onto, extractor=extractor, limits=p.limits,
                retrieval=k.retrieval.model_copy(update={"reranker": "none"}),   # prebuild never ranks
                execution_mode=p.execution_mode)
            (build.parent / f"{cond}.build.json").write_text(json.dumps(info, indent=1, ensure_ascii=False),
                                                              encoding="utf-8")
            snapshot_state(build, snap)
        return self._record_initial_states()

    def _import_initial_states(self, src: Path) -> dict[str, str]:
        """U39: copy another run's initial snapshots (built or re-gated under THIS run's gate identity: answer bundles,
        set scope, checker). A frozen evaluation also checks them against the hashes pinned at freeze."""
        from .knowledge.store import gate_identity
        from .memory.snapshot import copy_snapshot
        want = gate_identity(self.bundles(), self.plan.set_id, build_checker(self.profile))
        pinned = None
        if self.manifest.frozen and self.profile.frozen_manifest:
            pinned = json.loads(self.profile.resolve(self.profile.frozen_manifest).read_text(encoding="utf-8"))
            pinned = pinned.get("initial_states")
        for cond in self.plan.conditions:
            snap = self._pending_snapshot(cond)
            if snap is not None:
                copy_snapshot(src / "initial" / cond, snap)
            state = _state_table(self.run_dir / "initial" / cond / "state" / "knowledge.sqlite")
            if state.get("gate_identity") != want or state.get("regate_status") not in (None, "done"):
                raise ValueError(f"the {cond} initial state of {src} is not gated under this run's answer bundles and "
                                 f"set scope '{self.plan.set_id}': re-gate it for this plan (`regate-state`)")
        hashes = self._record_initial_states()
        if pinned is not None and pinned != hashes:
            raise RuntimeError("imported initial states differ from the hashes pinned at freeze")
        self.manifest.evidence_hashes.update({f"initial_state_from:{c}": h for c, h in hashes.items()})
        self.write_manifest()
        return hashes

    def extend_plan(self, plan: SetPlan, workers: int = 1, seed: Path | None = None) -> dict[str, Any]:
        """U40: continue this run's sets under a longer progression plan (the same conditions and reps, the old tasks
        as a prefix, a budget at least as large; the set name is kept because every stored scope uses it). Each
        started set's state is re-gated in place under the new plan's answer bundles first (knowledge artifacts and
        stored consult answers, never loosened); the plan is switched last, so a crash never leaves a half-checked
        state usable (the old plan's identity no longer opens it, the new plan is not written yet). seed: a run
        whose initial states were re-gated for these bundles (`regate-state`), whose verdicts are reused."""
        from .harness.runner import TERMINAL, progression_replay_problems
        from .knowledge.regate import regate_state
        from .knowledge.store import KnowledgeStore, gate_identity
        from .memory.snapshot import snapshot_state
        old, plan = self.plan, plan.model_copy(update={"set_id": self.plan.set_id})
        probs = [m for bad, m in [
            (old.mode != "progression" or plan.mode != "progression", "only progression plans are extended"),
            (plan.conditions != old.conditions or plan.reps != old.reps, "conditions and reps must stay the same"),
            (plan.episodes[:len(old.episodes)] != old.episodes, "the old tasks must be a prefix of the new ones"),
            ((plan.action_budget or 0) < (old.action_budget or 0), "the set action budget cannot shrink")] if bad]
        if probs:
            raise ValueError("cannot extend the plan: " + "; ".join(probs))
        tasks = load_tasks(self.profile, plan)
        private = load_private(self.profile, list(tasks))
        bundles = [a.answer_bundle for a in private.values()]
        p, k = self.profile, self.profile.knowledge
        checker, embedder, onto = build_checker(p), build_embedder(p), build_ontology(p)
        seeds = {}
        if seed is not None:
            sid = load_set_plan(seed / "set_plan.yaml").set_id
            for cond in plan.conditions:
                # a finished re-gate's snapshot, or a stopped one's build dir: every cached verdict is one complete
                # check of this checker for these bundles, so an unfinished seed simply seeds fewer items
                db = next((d for d in (seed / "initial" / cond / "state" / "knowledge.sqlite",
                                       seed / "build" / cond / "knowledge.sqlite") if d.exists()), None)
                if db is None or _state_table(db).get("gate_identity") != gate_identity(bundles, sid, checker):
                    raise ValueError(f"the seed {seed} was not re-gated for these answer bundles and checker")
                seeds[cond] = (db, sid)
        L = Ledger(self.run_dir)
        try:
            eps = L.db.execute("SELECT episode_id, condition, set_rep, task_id, outcome, finalization FROM episodes "
                               "WHERE set_id=?", (plan.set_id,)).fetchall()
            open_eps = [e["episode_id"] for e in eps if e["outcome"] not in {o.value for o in TERMINAL}
                        or e["finalization"] != "done"]
            if open_eps:
                raise ValueError(f"episodes still open or unsaved: {open_eps}; resume the run to finish them first")
            scopes = [MemoryScope(run_id=self.manifest.run_id, condition=cond, set_id=plan.set_id, set_rep=rep)
                      for rep in range(1, plan.reps + 1) for cond in plan.conditions]
            for ms in scopes:               # refuse before anything changes
                if not (self.run_dir / "state" / ms.condition / plan.set_id / f"rep{ms.set_rep}" /
                        "knowledge.sqlite").exists():
                    raise ValueError(f"set {ms.key} has not started; only started sets are extended")
                replay = progression_replay_problems(L, plan, ms)
                if replay:
                    raise ValueError(f"set {ms.key} would not resume under the new plan: {replay[0]}")
            want = gate_identity(bundles, plan.set_id, checker)
            reports = {}
            for ms in scopes:
                rep, cond = ms.set_rep, ms.condition
                state = self.run_dir / "state" / cond / plan.set_id / f"rep{rep}"
                current = _state_table(state / "knowledge.sqlite")
                # nothing to re-decide: built under this identity (consults were gated at finalization under it), or
                # re-gated for it with the stored consult answers re-checked as well
                if current.get("gate_identity") == want and (current.get("regate_status") is None or (
                        current.get("regate_status") == "done" and current.get("consults_regated") == want)):
                    reports[ms.key] = {"knowledge": "unchanged gate identity", "consults": {"rechecked": 0,
                                                                                           "withheld": []}}
                    continue
                backup = self.run_dir / "extend-backup" / f"{cond}-rep{rep}-{old.hash[:12]}"
                if not (backup / "manifest.json").exists():   # the state as it was before the first extension
                    snapshot_state(state, backup)
                ctx = CallContext(sink=self._prebuild_sink, phase="prebuild", guard=self.guard,
                                  scope_key=f"{ms.key}/extend")
                product = cond == "product"
                kw = dict(embedder=embedder if product else None, ontology=onto if product else None,
                          retrieval=k.retrieval.model_copy(update={"reranker": "none"}),
                          execution_mode=p.execution_mode)
                knowledge = regate_state(state, checker, bundles, plan.set_id, ctx,
                                         baseline_source=f"corpus:{Path(k.baseline_initial_text).name}",
                                         corpus_dir=p.resolve(k.corpus_dir), workers=workers, seed=seeds.get(cond),
                                         **kw)
                consults = _regate_consults(state, ms, [e for e in eps if e["condition"] == cond
                                                        and e["set_rep"] == rep], L,
                                            KnowledgeStore(state, checker, bundles, plan.set_id, **kw), ctx)
                _set_state(state / "knowledge.sqlite", "consults_regated", want)
                reports[ms.key] = {"knowledge": knowledge, "consults": consults}
            out = {"from_plan": old.model_dump(mode="json"), "to_plan": plan.model_dump(mode="json"),
                   "states": reports}
            (self.run_dir / f"extend-{plan.hash[:12]}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                                     encoding="utf-8")
            # the switch, last: the manifest first (extra task hashes still load the old plan, which cannot open the
            # re-gated states, so a kill in between fails closed and `extend-plan` is simply re-run), then the plan
            self.manifest.evidence_hashes[f"set_plan_before_extend:{plan.hash[:12]}"] = old.hash
            self.manifest.set_plan, self.manifest.set_plan_hash = plan.model_dump(mode="json"), plan.hash
            self.manifest.tasks = {**self.manifest.tasks, **{t: payload_hash(v) for t, v in tasks.items()}}
            self.tasks = tasks
            self.write_manifest()
            (self.run_dir / "set_plan.yaml").write_text(yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False,
                                                                       allow_unicode=True), encoding="utf-8")
            self.plan, self.private = plan, private
            return out
        finally:
            L.close()

    def regate_initial_states(self, source: Path, workers: int = 1) -> dict[str, Any]:
        """U39: take another run's built initial states (same knowledge inputs, KG extractor and embedder) and
        re-gate every approved item under this run's answer bundles and set scope, then snapshot them as this
        run's initial states. Idempotent; a gate error leaves the condition to finish on the next call."""
        from .knowledge.regate import regate_state
        from .memory.snapshot import restore_state, snapshot_state, state_dir_hash, verify_snapshot
        src = Run.load(source)
        probs = same_knowledge(src, self.profile)
        if probs:
            raise ValueError("cannot re-gate another run's initial states:\n- " + "\n- ".join(probs))
        p, k = self.profile, self.profile.knowledge
        checker, embedder, onto = build_checker(p), build_embedder(p), build_ontology(p)
        reports = {}
        for cond in self.plan.conditions:
            build = self.run_dir / "build" / cond
            out = build.parent / f"{cond}.regate.json"
            snap = self._pending_snapshot(cond)
            if snap is None:
                reports[cond] = json.loads(out.read_text(encoding="utf-8"))
                continue
            origin = {"run_id": src.manifest.run_id, "initial_sha256": src.manifest.initial_state_hashes.get(cond)}
            marker = build.parent / f"{cond}.source.json"
            if not build.exists():                    # restore aside, then rename: a kill never leaves half a copy
                tmp = build.parent / f"{cond}.restoring"
                restore_state(src.run_dir / "initial" / cond, tmp)   # an older partial tmp is moved aside
                tmp.rename(build)
            if not marker.exists():                   # restored but not yet re-gated: it must be the source's state
                if state_dir_hash(build) != verify_snapshot(src.run_dir / "initial" / cond)["combined_sha256"]:
                    raise ValueError(f"{build} is not an untouched copy of {src.manifest.run_id}'s {cond} state; "
                                     "use a new run id")
                marker.write_text(json.dumps(origin), encoding="utf-8")
            elif json.loads(marker.read_text(encoding="utf-8")) != origin:
                raise ValueError(f"the {cond} state being re-gated in this run came from another source than "
                                 f"{src.manifest.run_id}; finish it with that source or use a new run id")
            ctx = CallContext(sink=self._prebuild_sink, phase="prebuild", guard=self.guard,
                              scope_key=f"{self.manifest.run_id}/{cond}/{self.plan.set_id}/regate")
            product = cond == "product"
            rep = regate_state(build, checker, self.bundles(), self.plan.set_id, ctx,
                               baseline_source=f"corpus:{Path(k.baseline_initial_text).name}", corpus_dir=p.resolve(k.corpus_dir),
                               embedder=embedder if product else None, ontology=onto if product else None,
                               retrieval=k.retrieval.model_copy(update={"reranker": "none"}),
                               execution_mode=p.execution_mode, workers=workers)
            reports[cond] = {**rep, "source_run": src.manifest.run_id,
                             "source_initial_sha256": src.manifest.initial_state_hashes.get(cond)}
            out.write_text(json.dumps(reports[cond], indent=1, ensure_ascii=False), encoding="utf-8")
            snapshot_state(build, snap)
        self.manifest.evidence_hashes.update({f"regated_from:{src.manifest.run_id}:{c}": h
                                              for c, h in src.manifest.initial_state_hashes.items()})
        self._record_initial_states()
        return reports

    # ------------------------------------------------ per-scope components
    def factory(self, ms: MemoryScope, fresh: bool) -> ConditionComponents:
        from .advisors.baseline import BaselineAdvisor
        from .advisors.fixture_policy import make_fixture_advisor_policy
        from .advisors.product import ProductAdvisor
        from .knowledge.build import load_baseline_initial
        from .knowledge.gated import GatedSearch
        from .knowledge.store import KnowledgeStore
        from .memory.baseline import BaselineTextMemory
        from .memory.product import ProductMemory
        from .memory.snapshot import restore_state, state_dir_hash
        from .researcher.agent import PlannerReviewerResearcher
        from .researcher.fixture_policy import make_fixture_policy
        from .simulators.factory import build_simulator

        p, lim, cond = self.profile, self.profile.limits, ms.condition
        state = self.run_dir / "state" / cond / ms.set_id / f"rep{ms.set_rep}"
        if fresh:
            restore_state(self.run_dir / "initial" / cond, state)   # superseded dirs are moved aside, never deleted
        elif not state.exists():
            raise FileNotFoundError(f"resumed scope {ms.key} has no state dir {state}")
        product = cond == "product"
        store = KnowledgeStore(state, build_checker(p), self.bundles(), self.plan.set_id,
                               embedder=build_embedder(p) if product else None,
                               ontology=build_ontology(p) if product else None, retrieval=p.knowledge.retrieval,
                               reranker=build_reranker(p) if product else None, execution_mode=p.execution_mode)
        search = GatedSearch(store, build_search_provider(p, self.guard), max_results=p.search.max_results,
                             max_chars=lim.max_tool_response_chars)
        gate = store.gate   # derived consult answers are gated before they are stored (§7.4)
        role = p.roles.advisor_product if product else p.roles.advisor_baseline
        policy = make_fixture_advisor_policy(cond) if role.provider == "fixture" else None
        if product:
            memory = ProductMemory(state, ms, gate_derived=gate, tasks=self.tasks, limits=lim)
            advisor = ProductAdvisor(_llm(p, "advisor_product", policy), role, lim, memory=memory, store=store,
                                     search=search, ontology=store.ontology)
        else:
            summ = None if p.roles.internal_summarizer.provider == "fixture" else _llm(p, "internal_summarizer")
            memory = BaselineTextMemory(state, ms, summ, lim, summarizer_role=p.roles.internal_summarizer if summ else None,
                                        gate_derived=gate, tasks=self.tasks)
            advisor = BaselineAdvisor(_llm(p, "advisor_baseline", policy), role, lim, memory=memory, store=store,
                                      search=search, initial_text=load_baseline_initial(state))
        rpolicy = make_fixture_policy(p.fixture or FixtureBehaviour()) if p.roles.researcher.provider == "fixture" else None
        researcher = PlannerReviewerResearcher(_llm(p, "researcher", rpolicy), p.roles.researcher, lim)
        sims = {}
        for t in self.tasks.values():
            if t.simulator_id not in sims:
                cfg = p.simulators.get(t.simulator_id)
                if cfg is None:
                    raise KeyError(f"profile has no simulators.{t.simulator_id} entry (task {t.task_id})")
                sims[t.simulator_id] = build_simulator(t.simulator_id, cfg, t, p.root)   # one adapter per scope

        def close() -> None:
            for s in sims.values():
                s.close()
            store.close()
        def whole_state_hash() -> str:
            """Memory content + every other state file (knowledge db, indexes, caches): spec 8.1 start/end hash."""
            return sha256_text(memory.state_hash() + state_dir_hash(state, exclude=("memory.sqlite",)))
        return ConditionComponents(researcher=researcher, advisor=advisor, memory=memory, simulators=sims,
                                   tasks=self.tasks, close=close, state_hash=whole_state_hash,
                                   hidden_success={t: a.hidden_success_thresholds for t, a in self.private.items()})

    # ------------------------------------------------ execution
    def execute(self, revalidated: bool = False) -> SetRunStatus:
        ledger = None
        try:
            self.build_initial_states()
            ledger = Ledger(self.run_dir)
            return SetRunner(ledger, self.manifest.run_id, self.plan, self.factory, self.profile.limits,
                             guard=self.guard, revalidated=revalidated).run()
        finally:
            if ledger is not None:
                ledger.close()
            self.session_line("execute", revalidated=revalidated)

    def session_line(self, command: str, **extra: Any) -> None:
        """One line per CLI session (prebuild-only sessions too): code version, wall time, cumulative guard."""
        cv = code_version(Path(self.profile.root))
        line = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "command": command, **extra,
                "wall_s": round(time.monotonic() - self._t0, 3), "code_version": cv,
                "code_changed": cv["source_tree_sha256"] != self.manifest.code_version["source_tree_sha256"],
                "guard": self.guard.summary()}
        with open(self.run_dir / "sessions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")


def _repo_root(start: Path) -> Path:
    for c in [start.resolve(), *start.resolve().parents]:
        if (c / "pyproject.toml").exists():
            return c
    return Path.cwd()


# ---------------------------------------------------------------- preflight for runs (B23)

def run_preflight(profile: Profile, plan: SetPlan) -> list[str]:
    """Config problems + run-level checks: tasks, simulators, private assets, evaluation-only gates."""
    out = preflight_problems(profile)
    try:
        tasks = load_tasks(profile, plan)
    except (OSError, ValueError) as e:
        return out + [f"task definitions: {e}"]
    for t in tasks.values():
        if t.simulator_id not in profile.simulators:
            out.append(f"simulators.{t.simulator_id} missing (task {t.task_id})")
    try:
        private = load_private(profile, list(tasks))
        for t in tasks.values():   # U44: every declared hidden criterion needs its evaluator threshold
            missing = {h.metric for h in t.hidden_success} - \
                {c.metric for c in private[t.task_id].hidden_success_thresholds}
            if missing:
                out.append(f"task {t.task_id}: hidden criteria {sorted(missing)} have no evaluator threshold")
    except (OSError, ValueError) as e:
        out.append(f"private assets: {e}")
    for f, what in ((profile.knowledge.corpus_dir, "knowledge.corpus_dir"),
                    (profile.knowledge.ontology_profile, "knowledge.ontology_profile"),
                    (profile.knowledge.baseline_initial_text, "knowledge.baseline_initial_text")):
        if f is not None and not profile.resolve(f).exists():
            out.append(f"{what} not found: {f}")
    out += [f"knowledge.initial_state_from: the {c} snapshot is {h}"
            for c, h in (initial_state_pins(profile, plan) or {}).items() if h.startswith("unavailable")]
    if profile.execution_mode != "offline_fixture":
        dev = [d for d in development_only_components(profile) if d != "limits.development_only"]
        if profile.execution_mode == "live_development":   # stand-in models/search never; dev DATA is recorded
            dev = [d for d in dev if d.startswith(("roles.", "search."))]
        out += [f"development-only component not allowed in {profile.execution_mode}: {d}" for d in dev]
    if profile.execution_mode == "evaluation":
        out += evaluation_problems(profile, plan, tasks)
    return out


def evaluation_problems(profile: Profile, plan: SetPlan, tasks: dict[str, PublicTask]) -> list[str]:
    from .simulators.task_validation import TaskValidationReport, registrable_for_evaluation
    out = []
    root = Path(profile.root)
    for t in tasks.values():
        rp = root / TASK_VALIDATION_DIR / f"{t.task_id}.json"
        if not rp.exists():
            out.append(f"task {t.task_id}: no TaskValidationReport ({rp.relative_to(root).as_posix()})")
        elif not registrable_for_evaluation(TaskValidationReport.model_validate_json(rp.read_text(encoding="utf-8")), t):
            out.append(f"task {t.task_id}: not validated for evaluation")
    fm = profile.resolve(profile.frozen_manifest) if profile.frozen_manifest else None
    if fm and fm.exists():
        frozen = json.loads(fm.read_text(encoding="utf-8"))
        expect = freeze_digest(profile, plan, tasks)
        for k, v in expect.items():
            if frozen.get(k) != v:
                out.append(f"frozen manifest mismatch: {k} (config or code changed after freeze; create a new evaluation version)")
        ap = frozen.get("analysis_plan_path")
        if not frozen.get("analysis_plan_sha256") or not ap:
            out.append("frozen manifest has no analysis plan")
        elif not (profile.resolve(ap)).exists() or \
                sha256_text(profile.resolve(ap).read_text(encoding="utf-8")) != frozen["analysis_plan_sha256"]:
            out.append("frozen manifest mismatch: analysis plan content changed after freeze")
    return out


def _tree_sha256(path: Path | None) -> str | None:
    """Content hash of a file or of every file under a directory (relative posix paths + bytes)."""
    if path is None or not path.exists():
        return None
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    base = path.parent if path.is_file() else path
    return sha256_text("".join(f"{p.relative_to(base).as_posix()}:{sha256_text(p.read_bytes().hex())}\n"
                               for p in files))


def _state_table(db: Path) -> dict[str, str]:
    """The key/value `state` table of a knowledge.sqlite (read only; {} if absent)."""
    import sqlite3
    from contextlib import closing
    if not db.exists():
        return {}
    with closing(sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)) as con:
        return dict(con.execute("SELECT key, value FROM state").fetchall())


def _regate_consults(state: Path, ms: MemoryScope, eps: list, ledger: Ledger, store: Any,
                     ctx: CallContext) -> dict[str, Any]:
    """U40: stored consult answers that are not withheld are gated again with the text and context finalization gated
    them with (memory.baseline._gate_consults), under the store's current identity; a non-allow verdict withholds
    them from now on. Idempotent (verdicts are cached; withholding is one row update)."""
    import sqlite3
    from contextlib import closing
    from .contracts import GateStatus, canonical_json
    from .knowledge.gate import KnowledgeInfraError
    product, out = ms.condition == "product", {"rechecked": 0, "withheld": []}
    try:
        with closing(sqlite3.connect(state / "memory.sqlite")) as con:
            for e in eps:
                for c in ledger.consults(e["episode_id"]):
                    q = ("SELECT withheld FROM consult_cases WHERE action_id=?", c.action_id) if product else \
                        ("SELECT withheld FROM records WHERE record_id=?", f"consult:{c.action_id}")
                    row = con.execute(q[0], (q[1],)).fetchone()
                    if row is None or row[0]:          # never stored, or withheld already: nothing to re-decide
                        continue
                    st = store.gate(canonical_json(c), {"kind": "consult_answer", "action_id": c.action_id,
                                                        "episode_id": e["episode_id"], "task_id": e["task_id"],
                                                        "scope_key": ms.key}, ctx)
                    if st == GateStatus.error:
                        raise KnowledgeInfraError(f"leakage gate unavailable for consult {c.action_id}; re-run")
                    out["rechecked"] += 1
                    if st != GateStatus.allow:
                        with con:
                            con.execute(*(("UPDATE consult_cases SET withheld=1 WHERE action_id=?", (c.action_id,))
                                          if product else ("UPDATE records SET withheld=1 WHERE record_id=?",
                                                           (f"consult:{c.action_id}",))))
                        out["withheld"].append(c.action_id)
    finally:
        store.close()
    return out


def _set_state(db: Path, key: str, value: str) -> None:
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db)) as con, con:
        con.execute("INSERT OR REPLACE INTO state VALUES (?, ?)", (key, value))


def same_knowledge(src: Run, profile: Profile) -> list[str]:
    """U39: a re-gated state must be what this profile would build: the same KG extractor and embedder roles, the
    same knowledge config, the corpus files the source ingested (their stored hashes) and the same baseline text."""
    from .knowledge.build import BASELINE_INITIAL_ID
    from .knowledge.parse import parse_document
    a, out = src.profile, []
    for role in ("kg_extractor", "embedder"):
        if getattr(a.roles, role) != getattr(profile.roles, role):
            out.append(f"roles.{role} differs from the source run {src.manifest.run_id}")
    if a.knowledge.model_dump(exclude={"initial_state_from"}) != \
            profile.knowledge.model_dump(exclude={"initial_state_from"}):
        out.append(f"the knowledge config differs from the source run {src.manifest.run_id}")
    if a.limits.baseline_initial_text_max_chars != profile.limits.baseline_initial_text_max_chars:
        out.append("limits.baseline_initial_text_max_chars differs from the source run")
    init = src.run_dir / "initial"
    ingested = {k.removeprefix("ingested:doc:"): v
                for k, v in _state_table(init / "product" / "state" / "knowledge.sqlite").items()
                if k.startswith("ingested:doc:")}
    corpus = profile.resolve(profile.knowledge.corpus_dir)
    if ingested != {p.stem: sha256_text(p.read_text(encoding="utf-8")) for p in sorted(corpus.glob("*.md"))}:
        out.append("the corpus files differ from the ones the source run ingested")
    base = init / "baseline" / "state" / "knowledge.sqlite"
    if base.exists() and profile.knowledge.baseline_initial_text:
        import sqlite3
        from contextlib import closing
        doc = parse_document(profile.resolve(profile.knowledge.baseline_initial_text).read_text(encoding="utf-8"),
                             "baseline_initial")
        text = doc.text[doc.body:][:profile.limits.baseline_initial_text_max_chars]
        with closing(sqlite3.connect(f"file:{base.as_posix()}?mode=ro", uri=True)) as con:
            row = con.execute("SELECT content_hash FROM artifacts WHERE id=?", (BASELINE_INITIAL_ID,)).fetchone()
        if row is None or row[0] != sha256_text(text):
            out.append("the baseline initial text differs from the source run's")
    return out


def initial_state_pins(profile: Profile, plan: SetPlan) -> dict[str, str] | None:
    """U39: imported initial states are pinned by content: each condition's snapshot hash, after verifying its files."""
    from .memory.snapshot import verify_snapshot
    if not profile.knowledge.initial_state_from:
        return None
    src, out = profile.resolve(profile.knowledge.initial_state_from), {}
    for c in plan.conditions:
        try:
            out[c] = verify_snapshot(src / "initial" / c)["combined_sha256"]
        except (OSError, ValueError) as e:
            out[c] = f"unavailable: {type(e).__name__}"
    return out


def freeze_digest(profile: Profile, plan: SetPlan, tasks: dict[str, PublicTask]) -> dict[str, Any]:
    """What a frozen evaluation pins (spec §12): configuration, code, prompts AND the knowledge/evaluator inputs by
    content (corpus, baseline initial text, ontology profile, answer bundles, task validation reports). The profile
    hash excludes the frozen_manifest pointer itself. A built initial state embeds build times, so a run that builds
    its own is pinned by its inputs; an imported one (knowledge.initial_state_from, U39) by its snapshot hashes."""
    prof = profile.model_dump(mode="json", exclude={"root", "frozen_manifest"})
    root, k = Path(profile.root), profile.knowledge
    private = load_private(profile, list(tasks))
    return {"profile_sha256": payload_hash(prof), "set_plan_sha256": plan.hash,
            "tasks": {t: payload_hash(v) for t, v in tasks.items()},
            "source_tree_sha256": code_version(root)["source_tree_sha256"],
            "spec_sha256": spec_sha256(root), "prompts": prompt_versions(),
            "knowledge_inputs": {"corpus": _tree_sha256(profile.resolve(k.corpus_dir)),
                                 "baseline_initial_text": _tree_sha256(profile.resolve(k.baseline_initial_text)),
                                 "ontology_profile": _tree_sha256(profile.resolve(k.ontology_profile))},
            "answer_bundles": {t: payload_hash(a.answer_bundle) for t, a in private.items()},
            "task_validation_reports": {t: _tree_sha256(root / TASK_VALIDATION_DIR / f"{t}.json") for t in tasks},
            "initial_states": initial_state_pins(profile, plan)}


def freeze(profile: Profile, plan: SetPlan, analysis_plan: Path, out: Path) -> dict[str, Any]:
    """Write a frozen evaluation manifest. Refuses development-only components and unvalidated tasks."""
    if profile.execution_mode != "evaluation":
        raise ValueError("freeze needs an execution_mode=evaluation profile")
    tasks = load_tasks(profile, plan)
    probs = [p for p in run_preflight(profile.model_copy(update={"frozen_manifest": None}), plan)
             if not p.startswith("evaluation requires frozen_manifest")]
    if probs:
        raise ValueError("cannot freeze:\n- " + "\n- ".join(probs))
    if out.exists():
        raise FileExistsError(f"{out} exists; a changed configuration is a new evaluation version (new file)")
    body = {**freeze_digest(profile, plan, tasks), "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "analysis_plan_sha256": sha256_text(analysis_plan.read_text(encoding="utf-8")),
            "analysis_plan_path": analysis_plan.as_posix()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return body


