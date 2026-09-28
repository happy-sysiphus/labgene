# Verification record (plan §7 B01–B23)

Run 2026-09-28 (Asia/Seoul) on Windows 11, Python 3.11.15 (`.venv`), dependencies pinned in `requirements.lock`.
Everything below is an **offline contract check** (fixture providers, fixture leakage checker, hash embedder,
artificial fixture simulators) unless marked *real env*. None of it is evidence of research ability, retrieval
quality, leakage-gate accuracy or product value.

## Commands and results

| Command | Result |
|---|---|
| `.venv/Scripts/python -m pytest tests/unit tests/integration tests/e2e -q` | 284 passed (4 min) |
| `.venv/Scripts/python -m pytest tests/science -q` | 6 passed — *real env*: aldenv (pinned clone) and Summit ReizmanSuzuki case 1 workers in `.envs/` |
| `.venv/Scripts/python -m pytest tests/live -q` | 1 skipped (needs `LABGENE_LIVE=1`, an approved live profile, credentials) |
| `python -m labgene preflight --profile configs/offline.yaml --plan configs/set_plans/smoke.yaml` | exit 0 |
| `python -m labgene run-set --profile configs/offline.yaml --plan configs/set_plans/smoke.yaml --run-id smoke-001` | exit 0, `complete`; 2 reps × 2 conditions × 3 episodes; report labelled `contract_check` |
| `python -m labgene report --run-id smoke-001` / `resume --run-id smoke-001` | exit 0 / exit 0 (no new actions) |
| `LABGENE_FAULT=after_execute_before_commit@5 … run-set … --run-id crash-001` then `resume --run-id crash-001` | exit 70, then exit 0 |
| `python -m labgene validate-gate … --cases configs/gate_cases/dev_fixture.yaml` | exit 0 (fixture checker on development cases: tool check only) |
| `python -m labgene validate-retrieval … --devset configs/retrieval_dev/fixture_dev.yaml` | exit 4: adoption `provisional` (reranker config cannot run offline; development set) |
| `python -m labgene qualify-researcher …` | exit 4: the scripted fixture researcher **fails** (structured 16/24, counterexample tasks, interaction loop) and the draft suite/criteria are pending review — the tool does not pass a weak researcher |
| `python -m labgene validate-task fixture_ridge --profile configs/offline.yaml` | exit 0, status `unvalidated` (artificial fixture) |
| `python -m labgene preflight --profile configs/live.example.yaml` | exit 2 (credentials, caps, prices, search cap missing) |

## Behaviour map

| ID | What is checked | Where (tests) |
|---|---|---|
| B01 | 49 actions then success/fail/consult → success/50 or budget_exhausted/50; no 51st execution; full CLI run with an always-consulting researcher ends at exactly 50 | `tests/unit/test_harness.py`, `tests/integration/test_episode.py`, `tests/e2e/test_offline_cli.py::test_b01_b21_budget_exhausted_at_exactly_50_without_51st_action` |
| B02 | missing/out-of-range parameters = 1 action, no value; broken JSON / unsupported / repeated action keys = protocol error; 3 consecutive → `protocol_error` | `test_harness.py`, `test_episode.py` |
| B03 | same action_id replay not re-charged; new id same parameters charged; payload conflict keeps original | `test_harness.py`, `test_episode.py`, `test_simulators.py` (adapter side) |
| B04 | forced interruption at every boundary (incl. journaling a decision, finalization) → resume equals an uninterrupted run; physical re-execution costs kept; caps hold across sessions incl. prebuild; interrupted snapshot recovered | `test_episode.py`, `tests/e2e/test_offline_cli.py::test_b04_*` (6 boundaries, subprocess CLI), `tests/e2e/test_app_integration.py` |
| B05 | consult-only and experiment-only flows; planner/reviewer cannot execute; one finalizer action per decision | `test_episode.py`, `test_researcher.py` |
| B06 | success only from a committed observation of the current episode; observation rows immutable (DB triggers) | `test_harness.py`, `test_episode.py` |
| B07 | never-consulted and post-last-consult observations saved once to own-condition memory; revisit reaches them via consultation | `test_memory.py`, `test_advisors.py`, `test_offline_cli.py::test_b07_b09_b10_*` |
| B08 | finalization failure keeps the scientific outcome, blocks the next episode, retries the same event id without duplicates | `test_memory.py`, `test_episode.py`, `test_set_runner.py` |
| B09 | new episode starts with empty researcher history/notes; past records only through the advisor | `test_episode.py`, `test_advisors.py`, `test_offline_cli.py` |
| B10 | whole-state snapshot/restore (sqlite incl. WAL, indexes, caches); every set rep starts from the same whole-state hash; no cross-condition/rep records | `test_memory.py`, `test_knowledge_rag.py`, `test_offline_cli.py::test_b07_b09_b10_*` |
| B11 | answer paper, preprint, derived text, hidden table, snippets and derived KG never reach model inputs, ledger, manifest or reports; condition parity through real wiring | `test_knowledge_gate.py`, `test_advisors.py`, `test_offline_cli.py::test_b11_*`, `test_app_integration.py::test_b17_b11_*` |
| B12 | citing background, public target, self-observation inference allowed; answer-string matching alone does not block | `test_knowledge_gate.py`, `test_evaluation_gate.py`, `test_reporting.py` |
| B13 | hold / checker error / context expansion never expose; unavailable message carries no hidden values or gate details | `test_knowledge_gate.py`, `test_knowledge_provenance.py`, `test_simulators.py` |
| B14 | blocked source invalidates chunks/summaries/relations/vectors/caches transitively; derived artifacts gated independently; bundle/policy change never reuses approvals | `test_knowledge_provenance.py` |
| B15 | table splits keep header/unit/locator; OCR-suspect numbers not verified; transcription checks | `test_knowledge_rag.py`, `test_evaluation_retrieval.py` |
| B16 | mechanical citation validity ≠ semantic support; wrong observation values flagged | `test_knowledge_cards.py`, `test_advisors.py`, `test_evaluation_retrieval.py` |
| B17 | baseline has no KG/RAG; both advisors have the same gated search/open tools and base model; no BO/optimizer; researcher prompts/tools/config identical across conditions | `test_advisors.py`, `test_knowledge_rag.py`, `test_app_integration.py::test_b17_b11_*` |
| B18 | per-role requests with explicit config; no continuation reuse across decisions/episodes; returned-model mismatch (incl. internal roles, unnamed model) stops; sticky until revalidated | `test_providers.py`, `test_researcher.py`, `test_set_runner.py`, `test_knowledge_cards.py` |
| B19 | incomplete / refusal / 429 / timeout distinguished; truncated JSON not executed; finite retries; no fallback model or fixture | `test_providers.py`, `test_researcher.py`, `test_advisors.py`, `test_memory.py` |
| B20 | analysis tools refuse file/network/simulator/other-condition access; predictions are `agent_prediction` | `test_researcher.py` |
| B21 | failures never get 51; set pairs, first visit vs revisit, incomplete sets flagged and not dropped | `test_reporting.py`, `test_episode.py`, `test_offline_cli.py` |
| B22 | internal/search/summary/finalization costs recorded; prebuild vs runtime vs finalization; unavailable usage ≠ 0; auto-save adds 0 actions; caps across sessions | `test_reporting.py`, `test_memory.py`, `test_providers.py`, `test_researcher.py`, `test_knowledge_rag.py`, `test_offline_cli.py`, `test_app_integration.py` |
| B23 | evaluation refuses unfrozen/mismatched manifests (incl. changed corpus/bundles/analysis plan), unvalidated tasks, development-only components, unset live caps; fixture output labelled contract check | `test_offline_cli.py::test_b23_*`, `test_app_integration.py::test_b23_*`, `test_simulators.py`, `test_memory.py`, `test_reporting.py` |

## Independent reviews (T10)

| Stage | Reviewed | Findings → outcome |
|---|---|---|
| Stage A | harness, simulators, memory, knowledge, providers+researcher | 9 major + 24 confirmed minor fixed with regression tests (7 minor findings the reviewer did not confirm by running were not acted on) |
| Stage B | advisors, reporting, evaluation | advisors 1 major + 3 minor; reporting 8 confirmed minor (1 unconfirmed not acted on); evaluation 1 blocker + 5 major + 10 minor — fixed with regression tests |
| Integration | app wiring, CLI, runner seams | 3 major (caps across resume/prebuild, freeze not pinning knowledge inputs, missing CLI commands) + 8 minor — fixed; plus a condition word reaching researcher inputs found while porting the parity check — fixed |

## Not verified here

- Any real provider call (Gemini Interactions, embeddings, internal Astra/Fable roles, Google search): no credentials or approved caps. `tests/live` exists but is skipped.
- Leakage-gate accuracy, retrieval/answer quality, researcher qualification, pilot, main evaluation (T08/T09).
- The Summit candidate's success rule (reachable only where the model breaks TON = yield/loading) — needs a decision.
