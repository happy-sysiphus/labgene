"""Evaluation config (configs/rag_eval.yaml): the judge role, the sample rule and the approved caps (spec §5, §7).
The advisors' own configuration always comes from the target run's profile copy, never from here."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field

from labgene.config import CostCaps, RoleModel, Strict
from labgene.contracts import payload_hash


class EvalConfig(Strict):
    judge: RoleModel
    sample_per_task: int = Field(10, ge=1)          # spec §3.2: at most 10 consults per condition and task
    passage_max_chars: int = Field(3000, ge=200)
    bootstrap_draws: int = Field(10000, ge=100)
    seed: int = 20260929
    private_dir: str = "private/rag_eval"           # hidden thresholds / raw simulator outputs (git-ignored)
    judge_caps: CostCaps
    support_caps: CostCaps

    @property
    def hash(self) -> str:
        return payload_hash(self.model_dump(mode="json"))


def load_config(path: str | Path) -> EvalConfig:
    with open(path, encoding="utf-8") as f:
        return EvalConfig.model_validate(yaml.safe_load(f) or {})
