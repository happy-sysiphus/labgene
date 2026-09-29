"""`python -m rag_eval run --run-dir <finished harness run> --out <new dir>` (spec §7).

Order: pin the sample -> coded metrics for every consult -> rebuild each sampled consult's advisor input -> judge ->
reports. Every step keeps its finished items, so a stopped evaluation is resumed by the same command.
Exit codes: 0 done, 2 refused input, 3 stopped (cost cap, infrastructure, model change; rerun to resume)."""
from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from labgene.app import Run, build_embedder
from labgene.config import CostCaps, load_dotenv, repo_root_for
from labgene.contracts import canonical_json, payload_hash
from labgene.costs import CallContext, CapExceeded, CostGuard
from labgene.knowledge.gate import KnowledgeInfraError
from labgene.knowledge.retrieval import embed
from labgene.providers.base import ModelChangedError
from labgene.providers.call import build_llm
from labgene.simulators.base import SimulatorInfraError

from .config import EvalConfig, load_config
from .consults import (ConsultRecord, consult_records, consult_request, draw_sample, episode_orders, open_ledger,
                       set_finished)
from .context import Blinder, Rebuilder, passages, retrieved_cards
from .judge import PROMPT_HASH, PROMPT_VERSION, Judge, fixture_judge
from .metrics import answer_relevancy, citation_metrics, claim_metrics, context_precision, issue_counts
from .report import build_items, split_sim, summarize, write_reports
from .simscore import SimScorer


def read_lines(path: Path) -> list[dict[str, Any]]:
    """JSON lines; an unreadable line (a stop in the middle of a write) is skipped, so its item is redone."""
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


class Appender:
    """Thread-safe JSON-lines append: one line per finished item."""

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def __call__(self, path: Path, row: dict[str, Any]) -> None:
        with self.lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Guards:
    """The evaluation's two approved caps (spec §7): the judge role, and everything else (embeddings; gate and
    summarizer calls while rebuilding). Totals persist in <out>/guard.json, so the caps hold across reruns."""

    def __init__(self, out: Path, cfg: EvalConfig):
        self.path, self.lock = out / "guard.json", threading.Lock()
        saved = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.judge = self._guard("judge", cfg.judge_caps, saved)
        self.support = self._guard("support", cfg.support_caps, saved)

    def _guard(self, name: str, caps: CostCaps, saved: dict[str, Any]) -> CostGuard:
        g = CostGuard(caps, enforce=True, on_change=lambda totals: self._save(name, totals))
        if name in saved:
            g.restore(saved[name])
        return g

    def _save(self, name: str, totals: dict[str, Any]) -> None:
        with self.lock:
            d = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            d[name] = totals
            tmp = self.path.with_name("guard.json.tmp")
            tmp.write_text(json.dumps(d), encoding="utf-8")
            os.replace(tmp, self.path)


def pin(out: Path, run: Run, cfg: EvalConfig, records: list[ConsultRecord], sample: list[str],
        finished: bool) -> dict[str, Any]:
    """manifest.json and sample.json, written before any judge call (spec §3.2). A rerun into the same --out must see
    the same target, consults, config, prompts and sample; anything else is refused (use a new --out)."""
    key = {"target_run": run.manifest.run_id, "records_sha256": payload_hash([r.action_id for r in records]),
           "config_sha256": cfg.hash, "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_HASH,
           "sample_sha256": payload_hash(sample)}
    path = out / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        changed = [k for k, v in key.items() if old.get(k) != v]
        if changed:
            raise ValueError(f"{out} belongs to an evaluation with another {', '.join(changed)}; use a new --out")
        return old
    m = {**key, "target_complete": finished, "target_profile_hash": run.manifest.profile_hash,
         "frozen_manifest_sha256": run.manifest.evidence_hashes.get("frozen_manifest"),
         "judge": cfg.judge.model_dump(mode="json"), "sample_size": len(sample),
         "sample_per_task": cfg.sample_per_task, "bootstrap_draws": cfg.bootstrap_draws,
         "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "sample.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")
    path.write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def code_phase(run: Run, con, records: list[ConsultRecord], out: Path, private: Path,
               append: Appender) -> dict[str, dict[str, Any]]:
    """Mechanical issue counts and the first candidate's simulator score for EVERY consult (spec §4)."""
    done = {r["action_id"]: r for r in read_lines(out / "code.jsonl")}
    todo = [r for r in records if r.action_id not in done]
    scorer = SimScorer(run) if any(r.ok for r in todo) else None
    try:
        for rec in todo:
            resp, sim = rec.exchange.response, None
            if rec.ok:
                task = run.tasks[rec.task_id]
                sim = scorer.top1(task, run.private[rec.task_id].hidden_success_thresholds, resp.candidates,
                                  consult_request(con, rec, task).observations)
            public, raw = split_sim(rec.action_id, sim)
            if raw is not None:
                append(private / "sim_raw.jsonl", raw)
            row = {"action_id": rec.action_id, "issues": issue_counts(resp.validation_issues), **public}
            append(out / "code.jsonl", row)
            done[rec.action_id] = row
    finally:
        if scorer is not None:
            scorer.close()
    return done


def context_file(out: Path, action_id: str) -> Path:
    return out / "contexts" / (action_id.replace(":", "_") + ".json")


def rebuild_phase(run: Run, con, records: list[ConsultRecord], sample: list[str], out: Path,
                  ctx_for: Callable[[ConsultRecord], CallContext]) -> None:
    """The exact advisor input of each sampled consult, saved once as <out>/contexts/<action>.json (spec §3.3)."""
    by_id = {r.action_id: r for r in records}
    builders: dict[str, Rebuilder] = {}
    try:
        for rec in (by_id[a] for a in sample if not context_file(out, a).exists()):
            rb = builders.get(rec.condition)
            if rb is None:
                rb = builders[rec.condition] = Rebuilder(run, rec.condition, episode_orders(con, rec.condition), out)
            payload, delivered = rb.rebuild(consult_request(con, rec, run.tasks[rec.task_id]), ctx_for(rec))
            f = context_file(out, rec.action_id)
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f.name + ".tmp")
            tmp.write_text(json.dumps({"payload": payload, "delivered": delivered}, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, f)
    finally:
        for rb in builders.values():
            rb.close()


def judge_one(judge: Judge, embedder: Any, rec: ConsultRecord, saved: dict[str, Any], cfg: EvalConfig,
              jctx: CallContext, sctx: CallContext) -> dict[str, Any]:
    """All judged metrics of one sampled consult (spec §4, §5): blinded passages, 3-4 judge calls, one embedding."""
    payload, delivered, r = saved["payload"], saved["delivered"], rec.exchange.response
    ps = passages(payload, cfg.passage_max_chars)
    cited = {*r.cited_source_ids, *r.cited_observation_ids}
    b = Blinder(set().union(cited, *(p.cites for p in ps)))
    question = b(rec.exchange.question)
    claims = judge.claims(question, {"answer": b(r.answer), "reasoning": b(r.reasoning),
                                     "candidate_rationales": [b(c.rationale) for c in r.candidates]}, jctx)
    facts = [c["text"] for c in claims or [] if c["type"] == "fact"]
    verdicts = judge.verify([b(p.text) for p in ps], facts, jctx) if facts else None
    cards = retrieved_cards(payload) if rec.condition == "product" else []
    useful = judge.useful(question, payload["task"], [b(canonical_json(c)) for c in cards], jctx) if cards else None
    qa = judge.questions(b(f"{r.answer}\n{r.reasoning}"), jctx)
    relevancy = None
    if qa is not None:
        vecs = embed(embedder, [question, *qa[0]], "query", sctx)
        relevancy = answer_relevancy(vecs[0], vecs[1:], qa[1])
    unavailable = [k for k, bad in (("claims", claims is None), ("verify", bool(facts) and verdicts is None),
                                    ("precision", bool(cards) and useful is None), ("questions", qa is None)) if bad]
    return {"action_id": rec.action_id, "reconstruction_match": set(r.cited_source_ids) <= set(delivered),
            "n_passages": len(ps), "n_retrieved": len(cards), "n_cited": len(cited), "unavailable": unavailable,
            **claim_metrics(claims, verdicts), **citation_metrics(verdicts, [p.cites for p in ps], cited),
            **context_precision(useful), "answer_relevancy": relevancy,
            "claims": claims, "verdicts": verdicts, "useful": useful,
            "questions": qa[0] if qa else None, "noncommittal": qa[1] if qa else None}


def judge_phase(run: Run, records: list[ConsultRecord], sample: list[str], cfg: EvalConfig, out: Path,
                guards: Guards, sink: Callable, append: Appender, workers: int) -> None:
    by_id = {r.action_id: r for r in records}
    done = {r["action_id"] for r in read_lines(out / "judged.jsonl")}
    todo = [by_id[a] for a in sample if a not in done]
    if not todo:
        return
    lim = run.profile.limits
    judge = Judge(build_llm(cfg.judge, fixture_policy=fixture_judge if cfg.judge.provider == "fixture" else None,
                            timeout_s=lim.provider_timeout_s), cfg.judge, lim)
    embedder = build_embedder(run.profile)

    def one(rec: ConsultRecord) -> None:
        base = CallContext(sink=sink, phase="evaluation_support", scope_key=f"rag_eval/{rec.condition}",
                           episode_id=rec.episode_id, action_id=rec.action_id)
        saved = json.loads(context_file(out, rec.action_id).read_text(encoding="utf-8"))
        append(out / "judged.jsonl", judge_one(judge, embedder, rec, saved, cfg, base.child(guard=guards.judge),
                                               base.child(guard=guards.support)))

    with ThreadPoolExecutor(max(1, workers)) as ex:
        futures = [ex.submit(one, r) for r in todo]
        try:
            for f in futures:
                f.result()
        except BaseException:
            for f in futures:
                f.cancel()
            raise


def _run(a: argparse.Namespace) -> dict[str, Any]:
    run_dir, out, cfg = Path(a.run_dir).resolve(), Path(a.out).resolve(), load_config(a.config)
    if out == run_dir or run_dir in out.parents:
        raise ValueError("--out must be outside the target run directory (the run is never written)")
    if cfg.judge.provider == "claude_code":
        from labgene.providers.claude_cli import cli_problem
        if problem := cli_problem():
            raise ValueError(f"judge: {problem}")
    run = Run.load(run_dir)
    if run.plan.reps != 1:
        raise ValueError("rag_eval evaluates runs with one set rep")
    private = Path(cfg.private_dir)
    private = (private if private.is_absolute() else Path(run.profile.root) / private) / out.name
    append = Appender()

    def sink(e) -> None:
        append(out / "costs.jsonl", json.loads(e.model_dump_json()))

    con = open_ledger(run_dir)
    try:
        finished = set_finished(con, run.plan)
        if not finished and not a.allow_incomplete:
            raise ValueError("the target run has not finished (open episodes, or a set short of its end); finish it "
                             "or pass --allow-incomplete")
        records = consult_records(con)
        sample = draw_sample(records, cfg.sample_per_task)
        out.mkdir(parents=True, exist_ok=True)
        private.mkdir(parents=True, exist_ok=True)
        manifest = pin(out, run, cfg, records, sample, finished)
        guards = Guards(out, cfg)
        code = code_phase(run, con, records, out, private, append)
        rebuild_phase(run, con, records, sample, out, lambda rec: CallContext(
            sink=sink, phase="evaluation_support", guard=guards.support, scope_key=f"rag_eval/{rec.condition}",
            episode_id=rec.episode_id, action_id=rec.action_id))
    finally:
        con.close()
    judge_phase(run, records, sample, cfg, out, guards, sink, append, a.workers)
    judged = {r["action_id"]: r for r in read_lines(out / "judged.jsonl")}
    items = build_items(records, code, judged, set(sample))
    paths = write_reports(out, manifest, items, summarize(items, cfg.bootstrap_draws, cfg.seed),
                          {"judge": guards.judge.summary(), "support": guards.support.summary()})
    return {"status": "done", "target_complete": finished, **paths}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rag_eval", description="Consultation RAG evaluation of a finished "
                                                                         "LabGene harness run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate one finished harness run")
    r.add_argument("--run-dir", required=True, help="the target run directory, e.g. artifacts/main-v3")
    r.add_argument("--out", required=True, help="a new directory for this evaluation (the same --out resumes it)")
    r.add_argument("--config", default="configs/rag_eval.yaml")
    r.add_argument("--workers", type=int, default=4, help="concurrent judge calls")
    r.add_argument("--allow-incomplete", action="store_true", help="evaluate a run whose sets have not finished")
    a = ap.parse_args(argv)
    load_dotenv(repo_root_for(Path.cwd() / "x") / ".env")
    try:
        print(json.dumps(_run(a), indent=1, ensure_ascii=False))
        return 0
    except (CapExceeded, KnowledgeInfraError, ModelChangedError, SimulatorInfraError) as e:
        print(json.dumps({"status": "stopped", "reason": type(e).__name__, "detail": str(e)}, ensure_ascii=False))
        return 3
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as e:
        print(json.dumps({"ok": False, "problems": [f"{type(e).__name__}: {e}"]}, ensure_ascii=False))
        return 2
