"""Shared data contracts (T00). Spec: docs/superpowers/specs/2026-09-28-labgene-harness-design.md v0.6.

Only the main implementer changes this file. Other modules propose changes.
Model-facing DTOs (PublicTask, ResearcherView, ConsultRequest, AdvisorResponse,
EvidenceCard.model_projection) must never carry PrivateTaskAssets content.
"""
from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from typing import Annotated, Any, Callable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "0.1.0"

# §5 fixed policy. Not configurable per profile.
MAX_ACTIONS = 50
PROTOCOL_ERROR_LIMIT = 3  # consecutive protocol violations -> outcome=protocol_error

Condition = Literal["baseline", "product"]
CONDITIONS: tuple[Condition, ...] = ("baseline", "product")


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def canonical_json(obj: Any) -> str:
    """Stable JSON used for hashing (sorted keys, no whitespace, floats as repr)."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_hash(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


# ---------------------------------------------------------------- scopes

class MemoryScope(Frozen):
    """Set-level scope: every memory/index/cache/provider state is keyed by this."""
    run_id: str
    condition: Condition
    set_id: str
    set_rep: int

    @property
    def key(self) -> str:
        return f"{self.run_id}/{self.condition}/{self.set_id}/rep{self.set_rep}"


class RunScope(Frozen):
    """Episode-level scope (plan §4.2 RunScope)."""
    run_id: str
    condition: Condition
    set_id: str
    set_rep: int
    episode_id: str          # unique within run, harness-issued
    episode_order: int       # 1-based position in the set plan
    task_id: str
    visit_index: int         # 1 = first visit of task_id within this set rep

    @property
    def memory_scope(self) -> MemoryScope:
        return MemoryScope(run_id=self.run_id, condition=self.condition,
                           set_id=self.set_id, set_rep=self.set_rep)


# ---------------------------------------------------------------- tasks (§6)

class ContinuousParam(Frozen):
    kind: Literal["continuous"] = "continuous"
    name: str
    unit: str
    min: float
    max: float
    description: str = ""


class IntegerParam(Frozen):
    kind: Literal["integer"] = "integer"
    name: str
    unit: str = ""
    min: int
    max: int
    description: str = ""


class CategoricalParam(Frozen):
    kind: Literal["categorical"] = "categorical"
    name: str
    choices: list[str]
    description: str = ""


ParamSpec = Annotated[Union[ContinuousParam, IntegerParam, CategoricalParam], Field(discriminator="kind")]


class LinearConstraint(Frozen):
    """sum(coefficients[name] * x[name]) <op> rhs, within tolerance."""
    coefficients: dict[str, float]
    op: Literal["<=", ">=", "=="]
    rhs: float
    tolerance: float = 1e-9
    description: str = ""


class MetricSpec(Frozen):
    name: str
    unit: str
    direction: Literal["maximize", "minimize"]


class SuccessCriterion(Frozen):
    """§6.3: maximize -> observed >= target - tolerance; minimize -> observed <= target + tolerance."""
    metric: str
    direction: Literal["maximize", "minimize"]
    target: float
    tolerance: float

    def satisfied(self, value: float) -> bool:
        if value is None or not math.isfinite(value):
            return False
        if self.direction == "maximize":
            return value >= self.target - self.tolerance
        return value <= self.target + self.tolerance


class PublicTask(Frozen):
    """Everything the researcher and both advisors may see about a task (§6.1)."""
    task_id: str
    version: str
    title: str
    problem: str
    parameters: list[ParamSpec]
    constraints: list[LinearConstraint] = []
    metrics: list[MetricSpec]
    success: list[SuccessCriterion]  # ALL must hold (explicit multi-metric rule)
    simulator_id: str
    simulator_version: str

    def is_success(self, results: dict[str, float]) -> bool:
        return bool(self.success) and all(c.satisfied(results.get(c.metric)) for c in self.success)


class DocumentIdentity(Frozen):
    """Identity of a blocked document and its alternate public versions (§7.2)."""
    doi: str | None = None
    arxiv_id: str | None = None
    titles: list[str] = []
    urls: list[str] = []
    note: str = ""


class AnswerBundle(Frozen):
    """Evaluator/gate-only. Never enters a model-facing DTO or public report."""
    bundle_id: str
    version: str
    set_scope: str                       # set_id(s) this bundle applies to
    blocked_documents: list[DocumentIdentity]
    secret_markers: list[str] = []       # hidden values/phrases whose exposure means leakage
    hidden_asset_paths: list[str] = []   # full tables, model code/weights (paths on evaluator side)


class PrivateTaskAssets(Frozen):
    """Plan §4.2. Loaded only by evaluator, gate and simulator worker."""
    task_id: str
    answer_bundle: AnswerBundle
    model_artifacts: dict[str, str] = {}      # name -> sha256
    validity_evidence: dict[str, Any] = {}
    known_success_inputs: list[dict[str, Any]] = []


# ---------------------------------------------------------------- actions (§5, §11.1-11.2)

class ActionKind(str, Enum):
    consult = "consult"
    run_experiment = "run_experiment"


class ActionEnvelope(Frozen):
    """Raw model output + the action the harness identified. action_id is harness-issued as
    f"{episode_id}:a{seq:03d}" (sortable within an episode); the model can neither choose it nor mark budget state.
    Extra top-level keys in the finalizer JSON (e.g. "note") are ignored by the parser."""
    action_id: str
    scope: RunScope
    raw_text: str
    kind: ActionKind
    args: dict[str, Any]
    payload_hash: str        # payload_hash({"kind":..., "args":...}) after normalization


class ProtocolError(Frozen):
    """Uninterpretable JSON / unsupported action / provider incomplete|refusal on finalizer.
    Not an action; counts toward the consecutive-violation streak."""
    scope: RunScope
    decision_index: int
    raw_text: str | None
    reason: Literal["invalid_json", "unsupported_action", "missing_question",
                    "provider_incomplete", "provider_refusal", "multiple_actions"]
    detail: str = ""


class Observation(Frozen):
    """Immutable result of a valid experiment (§3.1.2, §7.3 source=experiment)."""
    observation_id: str
    action_id: str
    scope: RunScope
    parameters: dict[str, Any]         # normalized
    results: dict[str, float]
    units: dict[str, str]
    simulator_id: str
    simulator_version: str
    meets_success_criteria: bool
    created_at: str
    source: Literal["experiment"] = "experiment"


class ExperimentError(Frozen):
    """Invalid experiment request: counted as one action, never carries a measured value."""
    action_id: str
    scope: RunScope
    submitted_parameters: Any
    reason: str
    kind: Literal["invalid_request"] = "invalid_request"


class ValidParameters(Frozen):
    parameters: dict[str, Any]


class InvalidParameters(Frozen):
    reason: str


class AnalysisResult(Frozen):
    """Researcher-side calculation. Never an observation, never a success."""
    analysis_id: str
    input_observation_ids: list[str]
    method: str
    result: dict[str, Any]
    uncertainty: dict[str, Any] | None = None
    kind: Literal["agent_prediction", "descriptive_statistic", "analysis"]


class ResearchNote(Frozen):
    hypotheses: list[str] = []
    observation_ids: list[str] = []
    support: list[str] = []
    refute: list[str] = []
    uncertainty: str = ""
    next_action_reason: str = ""


# ---------------------------------------------------------------- consultation (§11.1, §13.4)

class CandidateSuggestion(Frozen):
    parameters: dict[str, Any]
    rationale: str = ""
    revisit: Literal["new", "current_episode_repeat", "past_episode_repeat"] | None = None
    within_public_constraints: bool | None = None   # set by harness validation


class AdvisorResponse(Frozen):
    answer: str
    cited_source_ids: list[str] = []
    cited_observation_ids: list[str] = []
    reasoning: str = ""
    limitations: str = ""
    candidates: list[CandidateSuggestion] = []   # suggestions only; each run is a separate action
    validation_issues: list[str] = []            # mechanical checks (IDs/values/units/ranges)
    status: Literal["complete", "partial"] = "complete"


class ConsultExchange(Frozen):
    action_id: str
    question: str
    response: AdvisorResponse


class ConsultRequest(Frozen):
    action_id: str
    scope: RunScope
    question: str
    task: PublicTask
    observations: list[Observation]            # exact current-episode observations
    experiment_errors: list[ExperimentError]   # current-episode invalid requests
    prior_consults: list[ConsultExchange]      # current-episode consults (explicit history resend)
    remaining_actions: int                     # after this consult is counted


# ---------------------------------------------------------------- researcher I/O (§3.1)

class HistoryItem(Frozen):
    """payload shapes (fixed; researcher reads exactly these keys, never scope/condition fields):
    consult -> {question, response: AdvisorResponse dump}; observation -> Observation dump minus scope/created_at;
    invalid_experiment -> {submitted_parameters, reason}; protocol_error -> {reason, detail}.
    Ids shown to the researcher are localized ('a003', 'obs:a003', past 'e002:a003') so no condition name leaks."""
    kind: Literal["consult", "observation", "invalid_experiment", "protocol_error", "infra_notice"]
    action_id: str | None = None
    payload: dict[str, Any]


class ResearcherView(Frozen):
    """Identical format in both conditions; carries no condition/set identity."""
    task: PublicTask
    actions_used: int
    remaining_actions: int
    history: list[HistoryItem]
    notes: list[ResearchNote]


# ---------------------------------------------------------------- providers (§11.5)

class Usage(Frozen):
    """None means the endpoint did not report it (never 0 by default).
    output_tokens EXCLUDES reasoning_tokens (CostGuard.settle adds them); adapters normalize to this."""
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None


class ProviderStatus(str, Enum):
    ok = "ok"
    incomplete = "incomplete"        # max tokens / truncated
    refusal = "refusal"
    infra_error = "infra_error"      # 429 / timeout / 5xx / network after finite retries


class FunctionCall(Frozen):
    call_id: str | None = None
    name: str
    arguments: dict[str, Any]


class ProviderResult(Frozen):
    role: str
    provider: str
    endpoint: str
    status: ProviderStatus
    text: str | None = None
    function_calls: list[FunctionCall] = []
    usage: Usage = Usage()
    request_id: str | None = None
    interaction_id: str | None = None
    model_requested: str
    model_returned: str | None = None
    sdk_version: str | None = None
    latency_s: float = 0.0
    attempt: int = 1
    error: str | None = None


class ResearcherDecision(Frozen):
    raw_action_text: str | None
    status: Literal["ok", "provider_incomplete", "provider_refusal", "infra_error"]
    note: ResearchNote | None = None
    analysis: list[AnalysisResult] = []
    error: str | None = None


class AdvisorOutcome(Frozen):
    status: Literal["ok", "partial", "infra_error"]
    response: AdvisorResponse | None = None
    error: str | None = None


# ---------------------------------------------------------------- evidence (§13.3)

class EvidenceKind(str, Enum):
    literature = "literature"
    current_observation = "current_observation"
    past_observation = "past_observation"
    code_calculation = "code_calculation"
    agent_inference = "agent_inference"


class GateStatus(str, Enum):
    allow = "allow"
    block = "block"
    hold = "hold"
    error = "error"   # checker failure -> treated as unavailable


class CardAccess(Frozen):
    """Internal-only; stripped by model_projection."""
    condition: Condition
    set_id: str
    set_rep: int
    episode_id: str | None = None
    gate_status: GateStatus
    gate_policy_version: str
    answer_bundle_version: str


class EvidenceCard(Frozen):
    card_id: str
    kind: EvidenceKind
    source_ids: list[str] = []
    observation_ids: list[str] = []
    content_hash: str
    locator: dict[str, Any] | None = None       # doc/page/section/span or experiment id
    excerpt: str | None = None                  # verbatim source text
    summary: str | None = None                  # generated; never presented as excerpt
    applicability: dict[str, Any] = {}          # task/simulator/schema/metric versions, material, range, units
    derivation: dict[str, Any] | None = None    # method, code version, input ids, model, prompt version
    access: CardAccess
    uncertainty: list[str] = []

    def model_projection(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"access"})


# ---------------------------------------------------------------- episode state (§8.1)

class Outcome(str, Enum):
    running = "running"
    success = "success"
    budget_exhausted = "budget_exhausted"
    protocol_error = "protocol_error"
    infra_incomplete = "infra_incomplete"   # checkpointed, resumable; never converted to failure


class FinalizationStatus(str, Enum):
    not_started = "not_started"
    done = "done"
    failed = "failed"


class EpisodeState(Frozen):
    """Derived view of the ledger. Counters are computed from committed actions only."""
    scope: RunScope
    outcome: Outcome
    consultation_requests: int = 0
    experiment_evaluations: int = 0
    invalid_experiment_requests: int = 0
    protocol_streak: int = 0
    protocol_errors_total: int = 0
    actions_to_success: int | None = None
    pending_action_id: str | None = None
    finalization: FinalizationStatus = FinalizationStatus.not_started
    memory_hash_start: str | None = None
    memory_hash_end: str | None = None
    outcome_reason: str | None = None        # infra_incomplete: model_changed|cost_cap|advisor_infra|simulator_infra|researcher_infra
    finalization_reason: str | None = None   # failed finalization: memory_error|cost_cap|model_changed

    @property
    def experiment_attempts(self) -> int:
        return self.experiment_evaluations + self.invalid_experiment_requests

    @property
    def actions_used(self) -> int:
        return self.consultation_requests + self.experiment_attempts


# ---------------------------------------------------------------- costs (§8.2)

class CostEvent(Frozen):
    """One physical operation: every provider attempt, simulator call, retrieval, gate check,
    finalization. phase separates prebuild from runtime (B22)."""
    kind: Literal["llm_call", "embedding", "simulator", "retrieval", "gate", "search", "finalize", "other"]
    role: str
    phase: Literal["prebuild", "runtime", "finalization", "evaluation_support"]
    scope_key: str | None = None
    episode_id: str | None = None
    action_id: str | None = None
    provider: str | None = None
    model: str | None = None
    status: str = "ok"
    request_id: str | None = None
    usage: Usage = Usage()
    latency_s: float = 0.0
    attempt: int = 1
    detail: dict[str, Any] = {}


CostSink = Callable[[CostEvent], None]


# ---------------------------------------------------------------- manifest (§8.3, §12)

ExecutionMode = Literal["offline_fixture", "live_development", "evaluation"]


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    created_at: str
    execution_mode: ExecutionMode
    contract_version: str = CONTRACT_VERSION
    spec_sha256: str | None
    code_version: dict[str, Any]            # git commit, dirty flag, source tree hash
    profile_hash: str
    profile: dict[str, Any]
    set_plan_hash: str
    set_plan: dict[str, Any]
    tasks: dict[str, str]                   # task_id -> public task hash
    initial_state_hashes: dict[str, str]    # condition -> initial snapshot manifest hash
    models: dict[str, dict[str, Any]]       # role -> model config
    analysis_plan_hash: str | None = None
    frozen: bool = False
    development_only_fields: list[str] = []
    evidence_hashes: dict[str, str] = {}
    task_defs: dict[str, dict[str, Any]] = {}   # public task definitions used by reports
    prompts: dict[str, Any] | None = None       # prompt versions + hashes per role
