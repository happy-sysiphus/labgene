import json
from pathlib import Path

import pytest

CODEX_EVENTS = [{"type": "thread.started", "thread_id": "th-1"}, {"type": "turn.started"},
                {"type": "item.completed", "item": {"type": "reasoning"}},
                {"type": "item.completed", "item": {"type": "agent_message"}},
                {"type": "item.completed", "item": {"type": "error", "message": "harmless warning"}},
                {"type": "turn.completed", "usage": {"input_tokens": 7352, "cached_input_tokens": 100,
                                                     "output_tokens": 48, "reasoning_output_tokens": 8}}]


@pytest.fixture(autouse=True)
def _no_local_cli_checks(request, monkeypatch):
    """Offline tests never depend on the local codex / claude installs or logins (preflight would run their status
    commands). Live tests keep the real checks; subprocess (e2e) tests are unaffected by monkeypatching."""
    if request.node.path.parent.name == "live":
        return
    from labgene.providers import claude_cli, codex_cli
    monkeypatch.setattr(codex_cli, "cli_problem", lambda *a, **k: None)
    monkeypatch.setattr(claude_cli, "cli_problem", lambda *a, **k: None)


def codex_config(args: list[str], key: str) -> str:
    """Raw value of `-c key=value` in a codex exec command (TOML/JSON-quoted)."""
    return next(a.split("=", 1)[1] for a in args if a.startswith(key + "="))


def codex_call(args: list[str], stdin: str, cwd: Path) -> dict:
    """What one `codex exec` call received: args, stdin, instructions file, output schema, empty working dir."""
    schema = args[args.index("--output-schema") + 1] if "--output-schema" in args else None
    return {"args": args, "stdin": stdin, "cwd": cwd, "cwd_empty": not any(cwd.iterdir()),
            "instructions": Path(json.loads(codex_config(args, "model_instructions_file"))).read_text(encoding="utf-8"),
            "schema": json.loads(Path(schema).read_text(encoding="utf-8")) if schema else None}


@pytest.fixture
def codex_script():
    """Fake `codex exec` runner answering with the given final messages in order; returns (runner, calls)."""
    def make(*replies: str):
        queue, calls = list(replies), []

        def run(args, stdin, cwd, timeout):
            calls.append(codex_call(args, stdin, cwd))
            Path(args[args.index("-o") + 1]).write_text(queue.pop(0), encoding="utf-8")
            return 0, "\n".join(json.dumps(e) for e in CODEX_EVENTS), ""
        return run, calls
    return make
