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

    def build_initial_states(self) -> dict[str, str]:
        """Build + snapshot each condition's initial state once per run (idempotent: existing snapshots are kept)."""
        from .knowledge.build import build_initial_state
        from .memory.snapshot import snapshot_state
        p, k = self.profile, self.profile.knowledge
        checker, embedder, onto, extractor = build_checker(p), build_embedder(p), build_ontology(p), build_extractor(p)
        for cond in self.plan.conditions:
            snap = self.run_dir / "initial" / cond
            if (snap / "manifest.json").exists():
                continue
            if snap.exists():   # interrupted snapshot (manifest is written last): keep it aside, redo
                n = 1
                while (aside := snap.parent / f"{cond}.partial-{n}").exists():
                    n += 1
                snap.rename(aside)
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
        hashes = {c: json.loads((self.run_dir / "initial" / c / "manifest.json").read_text(encoding="utf-8"))
                  ["combined_sha256"] for c in self.plan.conditions}
        if self.manifest.initial_state_hashes and self.manifest.initial_state_hashes != hashes:
            raise RuntimeError("initial state snapshots differ from the run manifest")
        self.manifest.initial_state_hashes = hashes
        self.write_manifest()
        return hashes

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
                                   tasks=self.tasks, close=close, state_hash=whole_state_hash)

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
        load_private(profile, list(tasks))
    except (OSError, ValueError) as e:
        out.append(f"private assets: {e}")
    for f, what in ((profile.knowledge.corpus_dir, "knowledge.corpus_dir"),
                    (profile.knowledge.ontology_profile, "knowledge.ontology_profile"),
                    (profile.knowledge.baseline_initial_text, "knowledge.baseline_initial_text")):
        if f is not None and not profile.resolve(f).exists():
            out.append(f"{what} not found: {f}")
    if profile.execution_mode != "offline_fixture":
        dev = [d for d in development_only_components(profile) if d != "limits.development_only"]
        if profile.execution_mode == "evaluation" or dev:
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


def freeze_digest(profile: Profile, plan: SetPlan, tasks: dict[str, PublicTask]) -> dict[str, Any]:
    """What a frozen evaluation pins (spec §12): configuration, code, prompts AND the knowledge/evaluator inputs by
    content (corpus, baseline initial text, ontology profile, answer bundles, task validation reports). The profile
    hash excludes the frozen_manifest pointer itself. Initial-state snapshot hashes are per run (they embed build
    times), so the inputs are pinned instead."""
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
            "task_validation_reports": {t: _tree_sha256(root / TASK_VALIDATION_DIR / f"{t}.json") for t in tasks}}


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


