# LabGene main evaluation v3 — analysis plan

Fixed before the v3 run (spec §12). `freeze` hashes this file into `configs/frozen/main-v3.json`; a changed file
stops `run-evaluation`. Decisions: `docs/implementation/decisions.md` (U21–U45). Earlier plans:
`analysis-plan.md` (v1), `analysis-plan-v2.md` (v2).

## Why v3
The v2 run (`artifacts/main-v2`) was stopped by the user on c1 after 21 (baseline) and 19 (product) actions, both
uncleared: each researcher consulted once, at the start, and then only ran experiments, so the comparison measured
little of the advisors. v3 keeps v2's tasks, emulators, corpus, knowledge states, roles and budgets and changes one
rule (U45). The v2 run is kept as a record and is not analyzed.

## Design
- Set plan `configs/set_plans/main_v3.yaml` (set id main_v2: the same evaluation set, so the frozen v2 knowledge
  states import unchanged): progression suzuki_ton_01 → 02 → 03 → 04; a task is retried in a new episode until
  cleared; the set ends when all four are cleared or 300 actions are used; at most 50 actions per attempt, the last
  attempt gets the remaining budget (U26). Both conditions run at the same time; one set rep.
- Rule (U45): every experiment follows exactly one consultation. Actions alternate consult, run_experiment,
  consult, ...; the researcher writes each question; both count as actions (at most 25 experiments per 50-action
  attempt). The harness names the allowed action in the researcher's view, restricts the finalizer's output schema
  to it, and records any other action as a protocol error (not an action). An invalid experiment request takes the
  experiment's turn.
- Task objective (U44, unchanged): maximize TON (= yield / loading) while keeping the yield at or above 95 % of the
  emulator's best yield. Success: the public yield threshold AND a TON within 2 % of the best TON the emulator
  reaches under that yield constraint (evaluator-held threshold, in no model input, error or public report; the rule
  itself is public). Success is judged by code on the simulator output; criteria are never changed after this plan
  is fixed.
- Conditions and roles as v2: baseline and product advisors; researcher and both advisors Codex gpt-6-luna (effort
  max, one shared config, U10); leakage gate Claude Opus 5.5 (effort low, U42); KG extraction Opus 5.5 medium (U38);
  memory summaries Opus 5.5 max; embeddings Gemini gemini-embedding-2. Researcher prompt researcher-v3 = the
  researcher-v2 texts plus the U45 rule text, used only when the rule is on.
- Initial knowledge: `artifacts/state-v2` (the corpus build of run pilot-02 with the earlier gate approvals, U43),
  imported by the run and pinned by hash at freeze. No memory is carried over from v1 or v2.

## Primary metric (U27)
Per condition: tasks cleared within the 300-action set budget; tie → fewer total actions used (consultations and
experiments). One set rep: the result is descriptive (no interval, no significance test, no general claim).

## Secondary (descriptive)
Actions and experiments to clear each task (failed attempts + actions to success), first-attempt success,
consultations per attempt, protocol errors (including refused out-of-turn actions), costs and tokens per role,
wall time.

## Execution rules (U33, U36)
- Infrastructure failure: the same episode resumes; never counted against the researcher; an episode stuck in infra
  failure is reported as incomplete with its reason.
- Protocol-error termination (3 consecutive unparseable/invalid replies): a failed attempt; attempts ending in
  protocol errors without an action use no budget and 3 in a row on one task stop that condition for inspection.
- Model change reported by a provider: the run stops until revalidated.

## Leak rule (U34)
An answer leak found after the fact is reported with a flag; the run stays in the primary result.

## Scope of claims
- Within-simulator comparison (U24); one rep; the researcher configuration failed the fixed-state qualification
  (U37) and was kept; that qualification covered the researcher-v2 prompt, without the U45 rule.
- U43: the knowledge state's gate approvals come from the v1 build (the four answer bundles block the same
  documents), with the identity check and every known verdict applied.
- U45: a consultation precedes every experiment, so the result compares the advisors when consulted at every step,
  not how often a researcher chooses to consult.
- Offline fixture results are contract checks, never research or product performance.
