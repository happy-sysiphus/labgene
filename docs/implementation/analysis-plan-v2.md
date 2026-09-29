# LabGene main evaluation v2 — analysis plan

Fixed before the v2 run (spec §12). `freeze` hashes this file into `configs/frozen/main-v2.json`; a changed file
stops `run-evaluation`. Decisions: `docs/implementation/decisions.md` (U21–U44). v1 results:
`docs/implementation/results-2026-09-29-main.md`.

## Why v2
v1 tasks were easy for an informed researcher (2–14 actions per task; both conditions cleared 4/4): the ratio of the
public yield and TON targets revealed the optimal catalyst loading (U41), and success regions reached the top of the
temperature/time ranges. v2 keeps the same paper, emulators, corpus and roles and changes only the tasks (U44).

## Design
- Set plan `configs/set_plans/main_v2.yaml`: progression suzuki_ton_01 → 02 → 03 → 04 (Reizman Suzuki cases 1–4
  emulators); a task is retried in a new episode until cleared; the set ends when all four are cleared or 300 actions
  are used; at most 50 actions per attempt, the last attempt gets the remaining budget (U26). Both conditions run at
  the same time; one set rep.
- Task objective (U44): maximize TON (= yield / loading) while keeping the yield at or above 95 % of the emulator's
  best yield. Success: the public yield threshold AND a TON within 2 % of the best TON the emulator reaches under
  that yield constraint (computed on the evaluator's 54,264-point grid). The TON threshold is evaluator-held: it is
  in no model input, error or public report; the rule itself is public. Success is judged by code on the simulator
  output; criteria are never changed after this plan is fixed.
- Difficulty check (evaluator-side, before any run): an informed heuristic (maximum temperature/time, the eight
  catalysts, then coordinate search) needs a median of 17–40 experiments per task without the loading hint; a
  uniform random search of 50 experiments succeeds with probability < 1.4 %.
- Conditions: baseline and product advisors; researcher and both advisors Codex gpt-6-luna (effort max, one shared
  config, U10); leakage gate Claude Opus 5.5 (effort low, validated 45/45, U42); KG extraction Opus 5.5 medium
  (U38); memory summaries Opus 5.5 max; embeddings Gemini gemini-embedding-2.
- Initial knowledge: the corpus build of run pilot-02 (chunks, KG relations, vectors; baseline initial text), moved
  to the v2 plan's answer bundles with the earlier gate approvals kept (U43), imported by the run and pinned by
  hash at freeze (`artifacts/state-v2`). No memory is carried over from v1.

## Primary metric (U27)
Per condition: tasks cleared within the 300-action set budget; tie → fewer total actions used. One set rep: the
result is descriptive (no interval, no significance test, no general claim).

## Secondary (descriptive)
Actions to clear each task (failed attempts + actions to success), first-attempt success, success@50 (truncated
attempts flagged), consultations per attempt, costs and tokens per role, wall time.

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
  (U37) and was kept.
- U43: the knowledge state's gate approvals come from the v1 build (the four answer bundles block the same
  documents), with the identity check and every known verdict applied.
- Offline fixture results are contract checks, never research or product performance.
