#!/usr/bin/env python3
"""HIL-17 attribution ladder driver: per-rung accuracy / token table and aggregated nDCG@5.

``--dry-run`` runs the whole pipeline on the tiny Who&When Pro fixture with a gold-blind heuristic
FAKE judge (no server, no model; output carries ``"mock": true`` and measures plumbing only).
A real run (local judges on ``localhost:<port>/v1``) is inference and needs a window: it refuses unless
``WINDOW_CONFIRMED=1``. Inputs: ``--pro`` (Who&When Pro text split, stratified slice), ``--cases``
(labelled AttributionCase JSONL, e.g. orchestrator traces with the HIL-12 item 6 consensus), or
``--step-db`` (failed orchestrator traces from the trace store, unlabelled: picks and tokens only).
``--export-adx`` writes the AgentDebugX-schema trajectories of the loaded cases (zero inference).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trace.attribution import whowhen_pro as WP  # noqa: E402
from src.trace.attribution.adapter import from_l2_trace  # noqa: E402
from src.trace.attribution.aggregate import AggregatorConfig, aggregate_case  # noqa: E402
from src.trace.attribution.case import AttributionCase, load_cases_jsonl  # noqa: E402
from src.trace.attribution.judge import HeuristicFakeJudge, JudgeSpec, OpenAICompatJudge  # noqa: E402
from src.trace.attribution.ladder import LadderConfig, run_ladder  # noqa: E402
from src.trace.attribution.prompts import PromptConfig  # noqa: E402
from src.trace.attribution.report import markdown, ndcg_table, rung_table  # noqa: E402
from src.trace.attribution.schema import SCHEMA_PIN, write_jsonl  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "attribution" / "whowhen_pro_tiny.jsonl"


def _git_ref() -> str:
    import subprocess

    return (
        subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )


def load_inputs(a: argparse.Namespace) -> tuple[list[AttributionCase], dict]:
    if a.dry_run or a.pro:
        path = FIXTURE if a.dry_run and not a.pro else Path(a.pro)
        skips: Counter = Counter()
        cases = list(WP.load_cases(path, skips))
        if a.per_framework:
            cases = WP.select_slice(cases, a.per_framework, a.seed, a.frameworks, a.max_steps)
        return cases, {
            "set": "whowhen_pro_text",
            "path": str(path),
            "sha256": WP.file_sha256(path),
            "skipped": dict(skips),
            "slice": {
                "per_framework": a.per_framework,
                "seed": a.seed,
                "frameworks": a.frameworks,
                "max_steps": a.max_steps,
            },
        }
    if a.cases:
        return list(load_cases_jsonl(a.cases)), {"set": "cases_jsonl", "path": a.cases}
    if a.step_db:
        from src.trace.l2.loaders import load_progress_logs, load_step_events

        outcomes = None
        if a.progress_logs:
            import glob

            outcomes = {
                t.trace_id: t for t in load_progress_logs(sorted(glob.glob(a.progress_logs)))
            }
        traces = [t for t in load_step_events(a.step_db, outcomes) if t.outcome == "fail"]
        traces = sorted(traces, key=lambda t: t.trace_id)[: a.limit]
        return (
            [
                AttributionCase(t.trace_id, from_l2_trace(t), meta={"source": "step_events"})
                for t in traces
            ],
            {"set": "orchestrator_step_events", "path": a.step_db, "unlabelled": True},
        )
    raise SystemExit("one of --dry-run, --pro, --cases, --step-db is required")


def build_judges(a: argparse.Namespace) -> list[JudgeSpec]:
    if a.dry_run:
        return [
            JudgeSpec("fake-heuristic-a", HeuristicFakeJudge(100, 10), "fake", None),
            JudgeSpec("fake-heuristic-b", HeuristicFakeJudge(120, 12), "fake", None),
        ]
    if os.environ.get("WINDOW_CONFIRMED") != "1":
        raise SystemExit(
            "refusing a real judge run without WINDOW_CONFIRMED=1 (inference needs a window)"
        )
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if a.no_think else {}
    specs = [
        JudgeSpec(
            a.model_a,
            OpenAICompatJudge.local(a.port_a, a.model_a, extra_body=extra),
            a.family_a,
            a.quantised_a,
        )
    ]
    if a.port_b:
        specs.append(
            JudgeSpec(
                a.model_b,
                OpenAICompatJudge.local(a.port_b, a.model_b, extra_body=extra),
                a.family_b,
                a.quantised_b,
            )
        )
    return specs


def run(a: argparse.Namespace) -> dict:
    cases, scope = load_inputs(a)
    if a.export_adx:
        Path(a.export_adx).unlink(missing_ok=True)
        write_jsonl((c.trajectory for c in cases), a.export_adx)
    if a.export_only:
        return {"exported": len(cases), "scope": scope, "adx_schema": SCHEMA_PIN}
    judges = build_judges(a)
    pcfg = PromptConfig(with_reference=a.with_reference, max_step_chars=a.max_step_chars)
    lcfg = LadderConfig(policy=a.policy, prompt=pcfg)
    lad = [run_ladder(c.case_id, c.trajectory, judges[0].judge, lcfg, c.reference) for c in cases]
    rep: dict = {
        "set": scope["set"],
        "scope": scope,
        "mock": bool(a.dry_run),
        "policy": a.policy,
        "with_reference": a.with_reference,
        "code_ref": _git_ref(),
        "adx_schema": SCHEMA_PIN,
        "judges": [{"name": j.name, "family": j.family, "quantised": j.quantised} for j in judges],
        "ladder_judge": judges[0].name,
        "n_cases": len(cases),
        "n_labelled": sum(bool(c.relevance) for c in cases),
        "rungs": rung_table(cases, lad),
    }
    if not a.no_aggregate:
        acfg = AggregatorConfig(k=a.k, temperature=a.temperature, seed_base=a.seed, prompt=pcfg)
        agg = [
            aggregate_case(c.case_id, c.trajectory, judges, c.relevance, acfg, c.reference, a.seed)
            for c in cases
            if c.relevance
        ]
        rep["aggregate"] = ndcg_table(agg)
        rep["aggregate"]["k_per_family"], rep["aggregate"]["temperature"] = a.k, a.temperature
        rep["per_case_aggregate"] = [r.as_dict() for r in agg]
    rep["per_case_ladder"] = [r.as_dict() for r in lad]
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_argument_group("inputs")
    src.add_argument(
        "--dry-run", action="store_true", help="fixture + fake judge; no server, no model"
    )
    src.add_argument("--pro", help="Who&When Pro text.jsonl")
    src.add_argument("--cases", help="AttributionCase JSONL (labelled)")
    src.add_argument(
        "--step-db", help="trace store sqlite with tool_step rows (failed runs, unlabelled)"
    )
    src.add_argument(
        "--progress-logs", help="glob of progress logs joining outcomes to --step-db runs"
    )
    src.add_argument("--limit", type=int, default=20)
    src.add_argument(
        "--per-framework", type=int, default=0, help="stratified Pro slice size per framework"
    )
    src.add_argument("--frameworks", nargs="*")
    src.add_argument("--max-steps", type=int)
    src.add_argument("--seed", type=int, default=0)
    jd = ap.add_argument_group("judges (real run only)")
    jd.add_argument("--port-a", type=int)
    jd.add_argument("--model-a", default="judge-a")
    jd.add_argument("--family-a", default="")
    jd.add_argument("--quantised-a", action=argparse.BooleanOptionalAction, default=None)
    jd.add_argument("--port-b", type=int)
    jd.add_argument("--model-b", default="judge-b")
    jd.add_argument("--family-b", default="")
    jd.add_argument("--quantised-b", action=argparse.BooleanOptionalAction, default=None)
    jd.add_argument(
        "--no-think", action="store_true", help="send chat_template_kwargs.enable_thinking=false"
    )
    cf = ap.add_argument_group("protocol")
    cf.add_argument("--policy", choices=("calibrate", "escalate"), default="calibrate")
    cf.add_argument(
        "--with-reference",
        action="store_true",
        help="render the reference answer (stated in the report)",
    )
    cf.add_argument("--max-step-chars", type=int, default=600)
    cf.add_argument("--k", type=int, default=5, help="samples per judge family")
    cf.add_argument("--temperature", type=float, default=0.7)
    cf.add_argument("--no-aggregate", action="store_true")
    out = ap.add_argument_group("outputs")
    out.add_argument("--out", required=True, help="report JSON; a .md table is written beside it")
    out.add_argument(
        "--export-adx", help="write the cases' AgentDebugX-schema trajectories (JSONL)"
    )
    out.add_argument("--export-only", action="store_true", help="only --export-adx; no judge")
    out.add_argument(
        "--belief-out",
        help="VB-HIL17-WRITE: also write hil17_attribution.beliefs.jsonl into this dir via the root "
        "belief-kernel writer (opt-in; refuses mock, unlabelled and undigested reports)",
    )
    out.add_argument("--run-id", default="hil17-run", help="run id for the belief sidecar")
    out.add_argument("--epyc-root", help="root checkout (default: $EPYC_ROOT, epyc-root-main view, /workspace)")
    a = ap.parse_args(argv)
    if not a.dry_run and not a.export_only and (a.pro or a.cases or a.step_db) and not a.port_a:
        ap.error("a real run needs --port-a (and --port-b for the second family)")
    rep = run(a)
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    if "rungs" in rep:
        Path(a.out).with_suffix(".md").write_text(markdown(rep))
    print(json.dumps({k: rep[k] for k in ("set", "n_cases", "mock") if k in rep} | {"out": a.out}))
    if a.belief_out:
        return write_belief_sidecar(a, rep)
    return 0


def write_belief_sidecar(a: argparse.Namespace, rep: dict) -> int:
    """VB-HIL17-WRITE: emit the sidecar AFTER the report is on disk; a refusal is loud (exit 3)."""
    from scripts.lib.epyc_root import load_adapter

    try:
        Path(a.belief_out).mkdir(parents=True, exist_ok=True)
        side = load_adapter("hil17_attribution", a.epyc_root).write_belief_measurements(
            a.belief_out, run_id=a.run_id,
            producer="epyc-orchestrator/scripts/trace/attribution_ladder.py", results_path=a.out)
    except Exception as exc:  # noqa: BLE001 - report exists; surface the refusal, never hide it
        print(f"!!! BELIEF CAPTURE FAILED [hil17]: REFUSED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(f"belief sidecar: {side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
