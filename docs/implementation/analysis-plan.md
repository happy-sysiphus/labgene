# LabGene main evaluation — analysis plan v1

Fixed before the main evaluation (spec §12). `freeze` hashes this file into the frozen manifest; a changed file stops
`run-evaluation`. Decisions: `docs/implementation/decisions.md` U21–U33.

## Design
- Set plan `configs/set_plans/main.yaml`: progression suzuki_flow_01 → 02 → 03 → 04 (Reizman Suzuki cases 1–4
  emulators). A task is retried in a new episode until it is cleared; the set ends when all four are cleared or
  300 actions are used; one attempt has at most 50 actions and the last attempt gets only the remaining budget (U26).
- Conditions: baseline and product, same researcher (Codex gpt-6-luna, effort max) and the same advisor model
  config (U10). Both conditions run at the same time (U26). One set rep (U26).
- Success per task: the task's public success rule (U22 calibrated tolerance), judged by code on the simulator
  output. Criteria are never changed after this plan is fixed.
- Initial knowledge: the pilot's knowledge state (corpus chunks, KG relations extracted at medium effort, vectors),
  re-gated against all four answer bundles and imported by the main run; its snapshot hashes are pinned at freeze
  (U38, U39).

## Primary metric (U27)
Per condition: tasks cleared within the 300-action set budget; tie → fewer total actions used. The condition with
more tasks cleared (then fewer actions) is the better one in this run.

With one set rep the result is **descriptive**: no confidence interval, no significance test, no claim that one
condition is better in general. Further reps are a new version with identical settings (U26).

## Secondary (descriptive)
- success@50 per condition (successes / terminal attempts; truncated attempts flagged).
- First-attempt success per task; actions to clear each task (failed attempts + actions to success).
- Consultations per attempt; costs and tokens per role; wall time.

## Execution rules (U33)
- Infrastructure failure (provider, simulator worker, rate limit): the same episode resumes; it never counts against
  the researcher. An episode stuck in infra failure is reported as incomplete with its reason, never dropped.
- Protocol-error termination (3 consecutive unparseable/invalid replies, spec §11.2): a failed attempt.
- Attempts that end in protocol errors without an action use no budget; after 3 in a row on one task that
  condition stops for inspection (U36) and is resumed after the cause is settled.
- Model change reported by a provider: the run stops until revalidated (spec §10.1).

## Leak rule (U34)
If answer-paper content is found to have reached the researcher or an advisor (a leak the gate missed), the run is
reported in full with a leak flag and stays in the primary result; the report states which condition and which
episodes were affected.

## Pilot and recall (U23, U28, U40)
- U40: if the pilot's c1 attempt (both conditions) ends without an infra stop, a protocol stall or the recall
  trigger, the pilot run becomes the main experiment: that attempt is the first attempt of the main set and the run
  continues under the main plan after an in-place re-gate against all four answer bundles. Nothing else is changed
  on the basis of the pilot. The results write-up states that the first attempt ran before the settings were shown
  to the user. Otherwise the pilot is not counted and the main experiment starts as a fresh run (U28).
- Recall probe (`recall-probe`): for each task the researcher model is asked for the paper's reported optimum.
  Recalled = the ligand matches AND at least 2 of 3 continuous conditions within ±10 °C / ±2 min / ±0.3 mol%.
  Any recall, or both conditions clearing c1 within 5 actions in the pilot, switches to full anonymisation (U23)
  before freeze.

## Scope of claims
- U43: c2-c4 were not re-checked by a model against their own answer bundles; the c1-bundle approvals stand (the
  bundles block the same documents), with the identity check and every known verdict applied.
- U42: the leakage gate model changed after c1 (Codex gpt-6-astra -> Claude Opus 5.5, both validated on the same
  45 cases); earlier approvals were kept.
- U41: the ratio of each task's public yield and TON targets reveals the catalyst loading of its anchor point, so
  the search is effectively over three variables; the U22 difficulty calibration (random search over all four)
  overstates the difficulty. Both conditions see the same targets.
- The researcher configuration failed the fixed-state qualification (22/24 structured; U37) and was kept by the
  user's decision; the results write-up states this with every result.
- Within-simulator comparison: success is defined on the emulators' outputs, not on the real reaction (U24:
  no held-out error, paper/CSV calibration mismatch, CSV failure-record completeness unverified).
- Offline fixture results are contract checks, never research or product performance.
