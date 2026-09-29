"""Fixtures for the rag_eval tests: the repo root on sys.path (rag_eval lives outside src/) and ONE finished offline
fixture run (fixture providers: a contract check, never research or product performance)."""
import hashlib
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def offline_run(tmp_path_factory) -> Path:
    """Two visits of fixture_catalyst in both conditions, a consultation every second action."""
    from labgene.app import Run
    from labgene.config import load_profile, load_set_plan
    tmp = tmp_path_factory.mktemp("rag-run")
    prof = yaml.safe_load((REPO / "configs/offline.yaml").read_text(encoding="utf-8"))
    prof.setdefault("paths", {})["artifacts_dir"] = str(tmp / "artifacts")
    prof["fixture"] = {**prof["fixture"], "consult_every": 2}
    plan = {"set_id": "rag", "reps": 1, "conditions": ["baseline", "product"],
            "episodes": ["fixture_catalyst", "fixture_catalyst"]}
    run = Run.create(load_profile(write_yaml(tmp / "profile.yaml", prof), root=REPO),
                     load_set_plan(write_yaml(tmp / "plan.yaml", plan)), "rag-offline")
    assert run.execute().status == "complete"
    return run.run_dir


@pytest.fixture
def eval_config(tmp_path) -> Path:
    """configs/rag_eval.yaml with the fixture judge, a temporary private dir and fewer bootstrap draws."""
    cfg = yaml.safe_load((REPO / "configs/rag_eval.yaml").read_text(encoding="utf-8"))
    cfg["judge"] = {"provider": "fixture", "model": "fixture-judge", "endpoint": "fixture", "max_output_tokens": 1024}
    cfg["private_dir"] = str(tmp_path / "private")
    cfg["bootstrap_draws"] = 200
    return write_yaml(tmp_path / "rag_eval.yaml", cfg)


@pytest.fixture
def digest():
    """{relative path: sha256} of every file under a directory (proves the target run is never written)."""
    def tree(root: Path) -> dict[str, str]:
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*")) if p.is_file()}
    return tree
