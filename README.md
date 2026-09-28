# LabGene experiment harness

Compares one fixed researcher agent consulting a **general LLM advisor** vs the **LabGene product advisor**
(ontology / KG / RAG / accumulated memory) under a 50-action budget (consultations + valid + invalid experiments)
on deterministic virtual experiments. Spec: `docs/superpowers/specs/2026-09-28-labgene-harness-design.md` (v0.6);
status and evidence: `TASKS.md`; implementation choices: `docs/implementation/decisions.md`.

> Offline fixture runs are **contract checks only**. They are not evidence of research ability, retrieval quality,
> leakage-gate accuracy or product value. Real-provider and main-evaluation states are tracked separately in `TASKS.md`.

## Setup

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv -e ".[dev]"      # or: uv pip sync --python .venv requirements.lock
```

Core dependencies are only `pydantic` and `pyyaml`; provider adapters use the standard library HTTP client.
Scientific simulators run in isolated worker environments under `.envs/` (see
`docs/implementation/task-validation/README.md`).

## Offline reproduction (no API keys, no network)

```powershell
.venv\Scripts\python -m pytest tests/unit tests/integration tests/e2e -q
.venv\Scripts\python -m labgene preflight --profile configs/offline.yaml --plan configs/set_plans/smoke.yaml
.venv\Scripts\python -m labgene run-set  --profile configs/offline.yaml --plan configs/set_plans/smoke.yaml --run-id smoke-001
.venv\Scripts\python -m labgene report   --run-id smoke-001
.venv\Scripts\python -m labgene resume   --run-id smoke-001
```

Forced interruption and resume (any boundary: `before_reserve`, `after_reserve`, `after_execute_before_commit`,
`after_commit`, `before_finalize`, `during_finalize`, `after_finalize`; `name@N` = N-th hit):

```powershell
$env:LABGENE_FAULT="after_execute_before_commit@5"; .venv\Scripts\python -m labgene run-set --profile configs/offline.yaml --plan configs/set_plans/smoke.yaml --run-id crash-001   # exit 70
Remove-Item Env:LABGENE_FAULT; .venv\Scripts\python -m labgene resume --run-id crash-001                                                                                       # exit 0
```

Outputs go to `artifacts/<run_id>/` (git-ignored; each run id is created once): `manifest.json`, `ledger.sqlite`,
`guard.json` (cumulative cost-cap totals), `sessions.jsonl`, per-condition state snapshots, and `report/` (`report.json`, `report.md`, `episodes.csv`, `actions.csv`, `costs.csv`). Every manifest and report
carries `execution_mode`.

Exit codes: `0` complete, `2` blocked by preflight/input (including an existing run id: use a new id or `resume`),
`3` stopped and resumable (`infra_incomplete`, `finalization_failed`, `model_changed`, cost cap, knowledge infra),
`4` a validation ran and did not pass (`fail` / `provisional` / `incomplete`), `70` injected crash.

## Commands

| Command | Purpose |
|---|---|
| `preflight --profile P [--plan S]` | Config, credentials, approved cost caps, tasks/simulators/private assets; evaluation-only gates |
| `build-corpus --profile P --plan S --run-id ID` | Create a run and build + snapshot each condition's gated initial state (`resume` executes it) |
| `run-set --profile P --plan S --run-id ID` | Create and execute a non-evaluation run (both conditions, all set reps) |
| `resume --run-id ID [--revalidated]` | Continue a stopped run from its ledger; `--revalidated` only after re-validating a provider model change |
| `report --run-id ID` | Public JSON/CSV/Markdown report (success@50, set pairs, first visit vs revisit, costs by role/phase) |
| `validate-task TASK --profile P` | TaskValidationReport (determinism, ranges, success rule, evaluator-only success inputs) |
| `validate-gate --profile P --plan S --run-id ID --cases F` | T08.4 leakage-gate validation on a pre-fixed case file (whole file hash-pinned in `<F>.lock.json`) |
| `validate-retrieval --profile P --plan S --run-id ID --devset F` | T08.5 retrieval configs on a pre-fixed dev set over a restored copy of the product initial state |
| `qualify-researcher --profile P --plan S --run-id ID [--suite configs/qualification] [--reps 3]` | T08.3 / spec §10.3: 24 fixed-state tasks + 3 closed-loop types with the neutral general advisor (fresh memory per episode) |
| `freeze --profile P --plan S --analysis-plan F --out F` | Pin an evaluation configuration (refuses development-only components, unvalidated tasks) |
| `run-evaluation --profile P --plan S --run-id ID` | Main evaluation: requires `execution_mode: evaluation` and a matching frozen manifest |

## Live runs

Copy `configs/live.example.yaml`, fill **approved** finite cost caps and unit prices, and put keys in `.env`
(`.env.example`). `preflight` refuses to run while any cap/price/credential is missing; an unset cap is never read
as 0 or unlimited. The researcher and both advisors use `gemini-3.1-pro-preview` via Interactions with
`thinking_level=high`; a failed call is retried finitely and never replaced by another model or a fixture.

```powershell
$env:LABGENE_LIVE="1"; $env:LABGENE_LIVE_PROFILE="configs/live.yaml"; .venv\Scripts\python -m pytest tests/live -q
```

## Layout

`src/labgene/`: `contracts.py` (shared DTOs), `config.py`, `costs.py`, `app.py` (wiring), `cli.py`,
`harness/` (ledger, parser, episode/set runner), `simulators/`, `memory/`, `knowledge/` (gate, provenance, RAG, KG,
evidence cards), `providers/`, `researcher/`, `advisors/`, `evaluation/`, `reporting/`.
