"""Execution profiles and set plans (T00). Profiles are YAML; see configs/*.yaml."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .contracts import CONDITIONS, Condition, ExecutionMode, PublicTask, payload_hash

SPEC_MODEL = "gemini-3.1-pro-preview"
SPEC_THINKING = "high"

ProviderName = Literal["fixture", "gemini", "openai", "anthropic"]
CREDENTIAL_ENV = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleModel(Strict):
    provider: ProviderName
    model: str
    endpoint: Literal["interactions", "responses", "messages", "embed_content", "fixture"] = "fixture"
    thinking_level: str | None = None          # Gemini: generation_config.thinking_level
    reasoning_effort: str | None = None        # OpenAI Responses / Anthropic effort
    max_output_tokens: int | None = None
    dimensions: int | None = None              # embedders only
    allowed_returned_models: list[str] = []    # returned-model strings accepted as "same model"


class Roles(Strict):
    researcher: RoleModel
    advisor_baseline: RoleModel
    advisor_product: RoleModel
    internal_summarizer: RoleModel
    kg_extractor: RoleModel
    leakage_gate: RoleModel
    embedder: RoleModel


class Limits(Strict):
    """Finite internal limits (§5, §11.2). development_only until frozen for evaluation."""
    development_only: bool = True
    provider_max_attempts: int = 3
    provider_timeout_s: float = 120.0
    provider_backoff_s: float = 2.0
    researcher_max_analysis_calls: int = 4       # per planner/reviewer phase
    advisor_max_tool_calls: int = 6
    advisor_max_repairs: int = 1
    max_query_expansions: int = 2
    max_tool_response_chars: int = 8000
    baseline_initial_text_max_chars: int = 60000
    baseline_memory_max_chars: int = 60000
    retrieval_top_k: int = 8
    partial_advisor_response: Literal["retry_as_infra", "deliver_marked"] = "retry_as_infra"
    infra_max_action_retries: int = 3            # harness-level retries of one action before checkpoint


class Prices(Strict):
    input_usd_per_mtok: float
    output_usd_per_mtok: float


class CostCaps(Strict):
    """Live runs refuse to start while any cap is None. None is never 0 or unlimited."""
    max_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_usd: float | None = None
    max_wall_s: float | None = None
    max_search_calls: int | None = None          # paid search/fetch requests (required with a paid search provider)
    prices: dict[str, Prices] = {}               # model id -> unit prices (needed for USD guarantee)


class SearchConfig(Strict):
    provider: Literal["fixture", "google_cse", "none"] = "fixture"
    fixture_corpus: str | None = "tests/fixtures/web_corpus"
    max_results: int = 5


class RetrievalConfig(Strict):
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60
    candidates_per_retriever: int = 20
    reranker: Literal["none", "llm"] = "none"
    query_expansion: bool = False


class KnowledgeConfig(Strict):
    corpus_dir: str = "tests/fixtures/corpus"
    baseline_initial_text: str | None = "tests/fixtures/corpus/general_background.md"
    ontology_profile: str = "tests/fixtures/ontology/profile.yaml"
    retrieval: RetrievalConfig = RetrievalConfig()


class SimulatorConfig(Strict):
    backend: Literal["fixture", "worker"]
    fixture_function: str | None = None          # fixture backend: name in simulators.fixture
    python: str | None = None                    # worker backend: interpreter of isolated env
    worker_module: str | None = None
    worker_args: list[str] = []
    timeout_s: float = 60.0


class Paths(Strict):
    artifacts_dir: str = "artifacts"
    tasks_dir: str = "configs/tasks"
    private_dir: str = "tests/fixtures/private"


class FixtureBehaviour(Strict):
    """Offline-only scripted behaviour. Contract checks, never research-ability evidence."""
    researcher_policy: str = "coordinate_search"
    consult_every: int = 0                        # 0 = consult only first action of a visit
    invalid_request_rate_every: int = 0           # every Nth experiment deliberately invalid (0=never)


class Profile(Strict):
    profile_id: str
    execution_mode: ExecutionMode
    roles: Roles
    limits: Limits = Limits()
    cost_caps: CostCaps = CostCaps()
    search: SearchConfig = SearchConfig()
    knowledge: KnowledgeConfig = KnowledgeConfig()
    simulators: dict[str, SimulatorConfig]
    paths: Paths = Paths()
    fixture: FixtureBehaviour | None = None
    frozen_manifest: str | None = None            # evaluation mode: path to frozen manifest
    root: str = Field(default=".", exclude=True)  # repo root the relative paths resolve against

    def resolve(self, rel: str | None) -> Path | None:
        if rel is None:
            return None
        p = Path(rel)
        return p if p.is_absolute() else Path(self.root) / p

    @property
    def hash(self) -> str:
        return payload_hash(self.model_dump(mode="json", exclude={"root"}))


class SetPlan(Strict):
    set_id: str
    reps: int = 1
    conditions: list[Condition] = list(CONDITIONS)
    episodes: list[str]                           # task_id order; repeats = revisits

    def visits(self) -> list[tuple[int, str, int]]:
        """[(episode_order, task_id, visit_index)] with 1-based order and visit index."""
        seen: dict[str, int] = {}
        out = []
        for i, t in enumerate(self.episodes, start=1):
            seen[t] = seen.get(t, 0) + 1
            out.append((i, t, seen[t]))
        return out

    @property
    def hash(self) -> str:
        return payload_hash(self.model_dump(mode="json"))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_profile(path: str | Path, root: str | Path | None = None) -> Profile:
    data = _load_yaml(path)
    data["root"] = str(root if root is not None else repo_root_for(path))
    return Profile.model_validate(data)


def load_set_plan(path: str | Path) -> SetPlan:
    return SetPlan.model_validate(_load_yaml(path))


def load_public_task(profile: Profile, task_id: str) -> PublicTask:
    return PublicTask.model_validate(_load_yaml(profile.resolve(profile.paths.tasks_dir) / f"{task_id}.yaml"))


def repo_root_for(path: str | Path) -> Path:
    p = Path(path).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    return Path.cwd()


# ---------------------------------------------------------------- preflight (T00 §4, B23)

ADVISOR_SHARED_FIELDS = ("provider", "model", "endpoint", "thinking_level", "max_output_tokens")


def policy_problems(p: Profile) -> list[str]:
    """Mode-independent policy invariants (§2, §3.2, §10.1)."""
    out = []
    a, b = p.roles.advisor_baseline, p.roles.advisor_product
    for f in ADVISOR_SHARED_FIELDS:
        if getattr(a, f) != getattr(b, f):
            out.append(f"advisor_baseline.{f} != advisor_product.{f}: both advisors must share the base LLM config")
    if p.limits.provider_max_attempts < 1 or p.limits.infra_max_action_retries < 1:
        out.append("retry limits must be finite and >= 1")
    return out


def preflight_problems(p: Profile, env: dict[str, str] | None = None) -> list[str]:
    """Returns human-readable blocking problems for the profile's execution mode. Empty = ok."""
    env = dict(os.environ) if env is None else env
    out = policy_problems(p)
    roles = p.roles.model_dump()
    if p.execution_mode == "offline_fixture":
        for name, r in roles.items():
            if r["provider"] != "fixture":
                out.append(f"offline_fixture profile must not use a real provider (roles.{name}={r['provider']})")
        if p.search.provider not in ("fixture", "none"):
            out.append("offline_fixture profile must use fixture search")
        return out
    # live_development / evaluation
    for name in ("researcher", "advisor_baseline", "advisor_product"):
        r = roles[name]
        if r["provider"] != "gemini" or r["model"] != SPEC_MODEL or r["thinking_level"] != SPEC_THINKING \
                or r["endpoint"] != "interactions":
            out.append(f"roles.{name} must be gemini/{SPEC_MODEL}/interactions/thinking_level={SPEC_THINKING} (spec §10.1)")
    for name, r in roles.items():
        if r["provider"] == "fixture":
            out.append(f"roles.{name} uses a fixture provider; not allowed outside offline_fixture")
        elif not env.get(CREDENTIAL_ENV[r["provider"]]):
            out.append(f"roles.{name}: credential {CREDENTIAL_ENV[r['provider']]} is not set")
    if p.search.provider == "fixture":
        out.append("live profiles must not use fixture search")
    if p.search.provider == "google_cse":
        out += [f"search: {k} is not set" for k in ("LABGENE_SEARCH_API_KEY", "LABGENE_SEARCH_ENGINE_ID") if not env.get(k)]
        if not p.cost_caps.max_search_calls or p.cost_caps.max_search_calls <= 0:
            out.append("cost_caps.max_search_calls is unset: paid search needs an approved finite cap")
    caps = p.cost_caps
    for f in ("max_calls", "max_input_tokens", "max_output_tokens", "max_usd", "max_wall_s"):
        v = getattr(caps, f)
        if v is None or v <= 0:
            out.append(f"cost_caps.{f} is unset: an approved finite cap is required (unset is neither 0 nor unlimited)")
    for name in ("researcher", "advisor_baseline", "advisor_product", "internal_summarizer", "kg_extractor",
                 "leakage_gate", "embedder"):
        m = roles[name]["model"]
        if m not in caps.prices:
            out.append(f"cost_caps.prices has no unit price for {m} ({name}); USD cap cannot be guaranteed")
    if p.execution_mode == "evaluation":
        if p.limits.development_only:
            out.append("limits.development_only=true: freeze limits before evaluation")
        if not p.frozen_manifest or not Path(p.resolve(p.frozen_manifest)).exists():
            out.append("evaluation requires frozen_manifest pointing to an existing frozen manifest")
    return out
