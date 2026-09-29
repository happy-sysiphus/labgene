# Verification record (plan §7 B01–B23)

Run 2026-09-28 (Asia/Seoul) on Windows 11, Python 3.11.15 (`.venv`), dependencies pinned in `requirements.lock`.
Everything below is an **offline contract check** (fixture providers, fixture leakage checker, hash embedder,
artificial fixture simulators) unless marked *real env*. None of it is evidence of research ability, retrieval
quality, leakage-gate accuracy or product value.

## Commands and results

| Command | Result |
|---|---|
| `.venv/Scripts/python -m pytest tests/unit tests/integration tests/e2e -q` | 353 passed (2 min 49 s; 2026-09-28 after U10–U12 and their independent review, was 284) |
| `.venv/Scripts/python -m pytest tests/science -q` | 6 passed — *real env*: aldenv (pinned clone) and Summit ReizmanSuzuki case 1 workers in `.envs/` |
| `.venv/Scripts/python -m pytest tests/live -q` | without opt-in: 1 skipped. With `LABGENE_LIVE=1 LABGENE_LIVE_PROFILE=configs/live.yaml` (approved $2 smoke allocation; that Gemini profile is now `configs/live-gemini-smoke.yaml`): **3 passed, live**, 8 calls, $0.106 list |
| `python -m labgene preflight --profile configs/offline.yaml --plan configs/set_plans/smoke.yaml` | exit 0 |
| `python -m labgene run-set --profile configs/offline.yaml --plan configs/set_plans/smoke.yaml --run-id smoke-001` | exit 0, `complete`; 2 reps × 2 conditions × 3 episodes; report labelled `contract_check` |
| `python -m labgene report --run-id smoke-001` / `resume --run-id smoke-001` | exit 0 / exit 0 (no new actions) |
| `LABGENE_FAULT=after_execute_before_commit@5 … run-set … --run-id crash-001` then `resume --run-id crash-001` | exit 70, then exit 0 |
| `python -m labgene validate-gate … --cases configs/gate_cases/dev_fixture.yaml` | exit 0 (fixture checker on development cases: tool check only) |
| `python -m labgene validate-retrieval … --devset configs/retrieval_dev/fixture_dev.yaml` | exit 4: adoption `provisional` (reranker config cannot run offline; development set) |
| `python -m labgene qualify-researcher …` | exit 4: the scripted fixture researcher **fails** (structured 16/24, counterexample tasks, interaction loop) and the draft suite/criteria are pending review — the tool does not pass a weak researcher |
| `python -m labgene validate-task fixture_ridge --profile configs/offline.yaml` | exit 0, status `unvalidated` (artificial fixture) |
| `python -m labgene preflight --profile configs/live.example.yaml` | exit 2 (the five caps and every unit price are unset; subscription models must be priced at 0) |
| `codex debug prompt-input` with the adapter's flags (`.review_scratch/prompt_input.py`) | renders the model-visible input without a model call: 7,854 → 2,705 characters of Codex context after `ISOLATION_CONFIG` (only the catalog's multi-agent role text remains) |
| `python -m labgene preflight --profile configs/live.yaml --plan configs/set_plans/smoke.yaml` | exit 0 — U10–U12 profile; runs only `codex --version`, `codex login status`, `claude auth status` (pinned codex-cli 0.158.0 with ChatGPT login; Claude Code with the `claude.ai` subscription login); no model call |
| same with `configs/live-gemini-smoke.yaml` | exit 0 (spec §10.1 configuration still accepted) |

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
| U10–U12 | Codex/Claude Code adapters, config, preflight, report | 1 major (Codex model-reroute notices ignored → now a model change) + 8 minor (adapter failures charged to the researcher, reconnect `error` discarding a good reply, empty `-o` file, helper model masking a Claude swap, CODEX_ACCESS_TOKEN, missing guards, per-call Claude version + no auto-update, reservations without the CLI's own prompt, unpriced subscription models) — fixed with regression tests; hidden Codex context reduced (`codex debug prompt-input`) |
| Integration | app wiring, CLI, runner seams | 3 major (caps across resume/prebuild, freeze not pinning knowledge inputs, missing CLI commands) + 8 minor — fixed; plus a condition word reaching researcher inputs found while porting the parity check — fixed |

## Not verified here

- Subscription CLIs (U10/U11) are verified offline only, through fake `codex exec` / `claude -p` runners: the tool protocol, the resent history, strict schemas with nullable optional fields, fail-closed isolation, effort limits and the researcher/advisor/KG paths end to end. No live call yet (waiting for the user's permission): Codex's real event stream at effort max, `--output-schema` with `anyOf`-null on gpt-6-luna, the `claude -p` result shape (`modelUsage`, `structured_output`), hidden context both CLIs may still add, latency and subscription-limit behaviour.

- Real Gemini generation is verified by the live smoke only (researcher roles). Not verified live: advisors inside a full episode run, embeddings (gemini-embedding-2), internal Astra/Fable roles, Google search.
- Leakage-gate accuracy, retrieval/answer quality, researcher qualification, pilot, main evaluation (T08/T09).
- The Summit candidate's success rule (reachable only where the model breaks TON = yield/loading) — needs a decision.
