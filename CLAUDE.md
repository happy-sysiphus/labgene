# LabGene — implementation rules

Source of truth: `docs/superpowers/specs/2026-09-28-labgene-harness-design.md` (v0.6) and
`docs/superpowers/plans/2026-09-28-labgene-harness-implementation-plan.md`. Status lives in `TASKS.md`,
choices in `docs/implementation/decisions.md`.

- Env: `.venv` (uv, Python 3.11). Tests: `.venv/Scripts/python -m pytest`. Do not edit `pyproject.toml`/lockfile
  or `src/labgene/contracts.py` unless you are the main implementer; propose changes instead.
- Never weaken the researcher, force consultation, or change success criteria to favour the product.
- 50 actions = consults + valid + invalid experiments. Same action_id never re-charged; new id = new charge.
- Hidden answers/private assets never enter model-facing DTOs, errors, or public reports.
- Offline fixture results are contract checks, never research or product performance.
- No paid API calls without an approved cap in the profile. Never substitute fixtures/other models for a failed live call.
