# LabGene main experiment — results (run pilot-02, 2026-09-29)

Run `artifacts/pilot-02` (execution_mode `live_development`; the pilot run continued as the main experiment, U40).
Settings pinned in `artifacts/pilot-02/continuation.json`; full report in `artifacts/pilot-02/report/` (git-ignored).
Analysis plan: `docs/implementation/analysis-plan.md`. Descriptive only: one set rep, no interval, no significance.

## Primary metric (U27): tasks cleared within 300 actions, tie -> fewer total actions

| condition | tasks cleared | total actions | consultations | experiments | wall time* |
|---|---|---|---|---|---|
| baseline advisor | 4/4 | 35 | 5 | 30 | ~2 h 27 min |
| product advisor (LabGene) | 4/4 | **28** | 4 | 24 | ~1 h 17 min |

Both conditions cleared every task on the first attempt, so the tie is broken by total actions: **product 28 < baseline
35 -> product better in this rep**. *Wall time is secondary information (the two conditions ran in parallel; c1 ran
as the pilot).

## Per task (actions to clear)

| task | baseline | product | note |
|---|---|---|---|
| suzuki_flow_01 (c1) | **8** | 14 | both screened catalysts at the top of the temperature/time range |
| suzuki_flow_02 (c2) | 8 | 8 | same strategy in both; the successful catalyst came last in both screens |
| suzuki_flow_03 (c3) | 5 | **2** | the product advisor recommended the right ligand citing corpus literature; first experiment succeeded |
| suzuki_flow_04 (c4) | 14 | **4** | the product researcher tuned temperature/time early; the baseline screened all eight catalysts first, then tuned |

## Where the difference came from (qualitative)
- c3: the product advisor retrieved literature chunks (dialkylbiaryl phosphine ligand review, palladacycle
  precatalyst paper) and recommended a ligand that worked at once; the baseline advisor carried over the previous
  task's catalyst and the researcher needed three more experiments.
- c4: the optimum lies inside the temperature/time range; the product side moved temperature and residence time
  after two experiments, the baseline side first exhausted the catalyst list at the range maximum.
- c1 went the other way (baseline 8, product 14): early advice differs by chance of the screening order.

## Operations
- Infra errors: 3 provider errors (one 900 s Codex timeout), all recovered by automatic retries; no protocol stall,
  no model change, no leak found.
- API spend: $0.027 (embeddings only); all other model calls on the subscriptions. Guard totals for the whole run
  (corpus build, re-gates, pilot and main): 2,314 calls, 10.4M input and 0.87M output tokens.

## Limitations (stated with every result)
- U37: the researcher configuration failed the fixed-state qualification (22/24 structured) and was kept.
- U40: the first attempt (c1) ran as the pilot, before the settings were shown to the user.
- U41: the ratio of each task's yield and TON targets reveals the anchor's catalyst loading; with optima of c1-c3 at
  the top of the temperature/time range, the search was effectively a catalyst screen. Tasks were easy for an
  informed researcher (2-14 actions per task against a 50-action attempt budget): a ceiling effect on the primary
  metric (4/4 in both), so the comparison rests on the action count.
- U42/U43: the leakage gate model changed after c1 (astra -> Opus, both validated 45/45); c2-c4 used the c1-bundle
  approvals (the four bundles block the same documents), with the identity check and known verdicts applied.
- U24: within-simulator comparison (emulator outputs, not the real reaction). One rep: no general claim.

## Suggested next version (not decided)
Hide the anchor loading (targets not tied to one point, or an independent TON target), tasks whose optimum lies
inside the ranges or penalises the range maximum, difficulty calibrated against an informed heuristic (e.g. "maximum
temperature/time + catalyst screen" must not succeed within N actions), and more set reps for an interval estimate.
