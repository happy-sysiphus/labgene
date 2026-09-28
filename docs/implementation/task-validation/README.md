# Task validation (T02 / T08.1)

Public reports (`<task_id>.json`) come from `python -m labgene.simulators.task_validation <task_id>`.
Private halves (known success inputs, reference-point and near-target rows) go to `private/task-validation/`
(git-ignored). A task is registrable for evaluation only when `registrable_for_evaluation(report, task)` is true.

| task | status | why |
|---|---|---|
| fixture_ridge, fixture_catalyst | unvalidated | artificial fixture (offline contract check only) |
| aldenv_fastfast | unvalidated, development_only | harness connection check; mechanistic model, no measured data |
| summit_reizman_case1 | unvalidated (candidate) | success exists only where the model's TON breaks its own yield / loading (blocking reason); plus open questions in its report (tolerances, Table 1 not jointly reproduced, no held-out split, CSV/paper mismatch, ...) |

## Rebuild the isolated worker envs (git-ignored `.envs/`)

```sh
git clone https://github.com/aldsim/aldenv .envs/src/aldenv
git -C .envs/src/aldenv checkout 90055ef811134f4f3b569a088491d729746538fd
uv venv --python 3.11 .envs/aldenv
uv pip install --python .envs/aldenv -r docs/implementation/task-validation/aldenv-env.txt

git clone https://github.com/sustainable-processes/summit .envs/src/summit
git -C .envs/src/summit checkout 1de682d05e97adcfb96cd8376e876cef2d6160d3
uv venv --python 3.10 .envs/summit
uv pip install --python .envs/summit -r docs/implementation/task-validation/summit-env.txt
```

Upstream code is imported from the pinned clones (not installed) and never modified. The Summit env pins
numpy 1.23.5 instead of upstream `poetry.lock`'s 1.22.4: the torch 1.13.1 wheel needs NumPy C-API 0x10
(1.23+) and fails with 1.22.4 ("Could not infer dtype of numpy.int32"); 1.23.5 is inside upstream's `numpy ^1.21`.

## Run

```sh
.venv/Scripts/python -m labgene.simulators.task_validation summit_reizman_case1
.venv/Scripts/python -m pytest tests/science -q        # skips when .envs/ or private/ assets are missing
```

`simulators.yaml` holds the SimulatorConfig entries these runs use; live/evaluation profiles need the same entries.
Worker `simulator_version` in tasks is `opaque_version(descriptor)`; the descriptor (upstream commit, sha256 of the
upstream package tree as checked out, sha256 of the interpreter build + installed package set, model-file hash,
backend settings) is recorded in each report, never in model-facing DTOs. An edited clone or a different package
set changes the descriptor, so the worker handshake refuses the task's version instead of returning different
numbers under it. Rebuilding an env on another machine must reproduce the same hashes (same interpreter build,
the freeze files above, same line endings in the clone) or the task's `simulator_version` must be re-derived and
the task re-validated.
