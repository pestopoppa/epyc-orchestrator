#!/usr/bin/env python3
"""Run EV-5's sequential post-scoring phase on exported T2 QuestionResult rows.

Input is a JSON array containing original prompt/answer, route_used, confidence,
confidence_source and disposition. No answer is rescored. Generation must have
finished before --generation-drained is asserted; no shared processes are stopped.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, fields
import json
import os
import signal
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/autopilot"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uncertain-n", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout-s", type=float, default=120)
    parser.add_argument("--budget-s", type=float, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-drained", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    if argv == [] or (argv is None and len(sys.argv) == 1):
        parser.print_help()
        return None
    return parser.parse_args(argv)


def restore_attribution(rows, output_dir, identity):
    """Reuse only complete fsynced records bound to this exact run plan."""
    identity_path = output_dir / "RUN-IDENTITY.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("resume input/protocol/source/model/prompt plan changed")
    else:
        with identity_path.open("x") as fh:
            json.dump(identity, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
    by_id = {row.qid or row.question_id: row for row in rows}
    if len(by_id) != len(rows) or not all(by_id):
        raise ValueError("original input IDs must be unique and nonempty")
    restored = set()
    for path in sorted(output_dir.glob("process-attribution*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)  # Torn/invalid records fail closed, never count done.
            if "kind" in record:
                continue
            qid = record.get("qid")
            if qid not in by_id or qid in restored:
                raise ValueError("duplicate or foreign attribution ID")
            feedback = record.get("details", {}).get("process_verification")
            if not isinstance(feedback, dict) or not feedback.get("verification_cot"):
                raise ValueError("incomplete attribution cannot count as completed")
            if feedback.get("generator") != by_id[qid].route_used:
                raise ValueError("attribution generator differs from original row")
            by_id[qid].details["process_verification"] = feedback
            restored.add(qid)
    return restored


def main(argv=None):
    args = parse_args(argv)
    if args is None:
        return 0
    if not 0 <= args.uncertain_n <= 100 or not 0 < args.budget_s <= 86400:
        raise ValueError("bounded uncertain-n [0,100] and budget-s (0,86400] required")
    if not args.generation_drained and not args.dry_run:
        raise ValueError("--generation-drained is required for actual verifier loading")
    from src.config import get_config
    from scripts.server.eval_process_verifier import NativeProcessVerifier, ProcessVerifierConfig

    paths = get_config().paths
    kernel = paths.llm_root / "kernels/production/cpu"
    config = ProcessVerifierConfig(
        model=paths.models_dir / "ThinkPRM-1.5B/ThinkPRM-1.5B-Q4_K_M.gguf",
        server=kernel / "llama-server",
        library_dir=kernel,
        log_path=args.output_dir / "private-verifier.log",
        max_tokens=args.max_tokens,
        context_tokens=args.context_tokens,
        timeout_s=args.request_timeout_s,
        seed=args.seed,
        generation_drained=args.generation_drained,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "config": asdict(config),
                    "input": args.input,
                    "uncertain_n": args.uncertain_n,
                },
                default=str,
            )
        )
        return 0
    from eval_tower import QuestionResult
    from scripts.autopilot.process_verification import verify_results

    payload = json.loads(args.input.read_text())
    if not isinstance(payload, list) or len(payload) > 10000:
        raise ValueError("input must be an array of at most 10000 original T2 rows")
    allowed = {f.name for f in fields(QuestionResult)}
    required = {"prompt", "answer", "route_used", "confidence", "confidence_source", "disposition"}
    for row in payload:
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(
                "each input row must preserve generator, confidence and scoring provenance"
            )
    rows = [QuestionResult(**{k: v for k, v in row.items() if k in allowed}) for row in payload]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def digest(path):
        with Path(path).open("rb") as fh:
            return hashlib.file_digest(fh, "sha256").hexdigest()

    identity = {
        "version": "ev5-resume-v1",
        "input_sha256": digest(args.input),
        "protocol": {
            "n": args.uncertain_n,
            "max_tokens": args.max_tokens,
            "context_tokens": args.context_tokens,
            "seed": args.seed,
        },
        "model": {"path": str(config.model.resolve()), "sha256": digest(config.model)},
        "server_sha256": digest(config.server),
        "source": {
            str(path.relative_to(ROOT)): digest(path)
            for path in [
                Path(__file__),
                ROOT / "scripts/autopilot/process_verification.py",
                ROOT / "scripts/server/eval_process_verifier.py",
                ROOT / "src/autopilot_core/verification_families.py",
            ]
        },
        "prompt_plan_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
    }
    restored = restore_attribution(rows, args.output_dir, identity)
    result_path = args.output_dir / "RESULT.json"
    if result_path.exists() and not json.loads(result_path.read_text())["partial"]:
        return 0  # Completed second entry does no native work and rewrites nothing.
    pending = [row for row in rows if (row.qid or row.question_id) not in restored]
    backend = NativeProcessVerifier(config)
    started = time.monotonic()

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    report = {
        "quality_evaluated": False,
        "scorer_verdicts_changed": False,
        "input": str(args.input),
        "partial": True,
        "resume_reused_n": len(restored),
        "run_identity": identity,
    }
    rc = 75
    try:
        report["process_verification"] = verify_results(
            pending,
            backend,
            n=max(0, args.uncertain_n - len(restored)),
            artifact_path=args.output_dir / f"process-attribution.{time.time_ns()}.jsonl",
            # Reserve one request + teardown before the hard window end.
            deadline_monotonic=started + max(0, args.budget_s - args.request_timeout_s - 10),
        )
        report["process_verification"]["completed_n"] += len(restored)
        report["partial"] = report["process_verification"]["partial"]
        rc = 75 if report["partial"] else 0
    except (TimeoutError, httpx.TimeoutException, KeyboardInterrupt) as exc:
        report["error"] = type(exc).__name__
    finally:
        report.update(
            elapsed_s=time.monotonic() - started,
            lifecycle=backend.lifecycle,
            question_results=[asdict(row) for row in rows],
        )
        temporary = args.output_dir / "RESULT.json.tmp"
        with temporary.open("w") as fh:
            json.dump(report, fh, indent=2, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, result_path)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
