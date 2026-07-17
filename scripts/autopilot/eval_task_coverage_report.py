#!/usr/bin/env python3
"""Read-only AutoPilot eval-task coverage report.

The fixed T1 authority core is intentionally repetitive for paired comparisons.
This report makes that repetition visible so planner-learning coverage can be
tracked separately from authority evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.autopilot_core.journal_reconstruction import fold_supersession_events
except Exception:  # pragma: no cover - import guard for standalone diagnostics
    fold_supersession_events = None  # type: ignore[assignment]

ORCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOURNAL_DIR = ORCH_ROOT / "orchestration"
DEFAULT_POOL_CANDIDATES = (
    ORCH_ROOT / "benchmarks" / "prompts" / "question_pool.jsonl",
    Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl"),
)
POOL_METADATA_KEY = "__pool_metadata__"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_pct(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) * 100.0 / float(denominator), 4)


def _stable_question_qid(suite: str, prompt_text: str) -> str:
    payload = f"{suite}\x00{prompt_text}".encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()[:16]


def _jsonl_batch_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem == "autopilot_journal":
        return (0, path.name)
    prefix = "autopilot_journal_"
    if stem.startswith(prefix):
        try:
            return (int(stem.removeprefix(prefix)), path.name)
        except ValueError:
            pass
    return (10**9, path.name)


def default_pool_path() -> Path | None:
    for path in DEFAULT_POOL_CANDIDATES:
        if path.exists():
            return path
    return None


def resolve_journal_paths(paths: list[Path] | None, journal_dir: Path) -> list[Path]:
    if paths:
        resolved: list[Path] = []
        for path in paths:
            if path.is_dir():
                resolved.extend(sorted(path.glob("autopilot_journal*.jsonl"), key=_jsonl_batch_key))
            else:
                resolved.append(path)
        return sorted(dict.fromkeys(resolved), key=_jsonl_batch_key)
    return sorted(journal_dir.glob("autopilot_journal*.jsonl"), key=_jsonl_batch_key)


def load_jsonl(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    invalid_lines: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError as exc:
                    invalid_lines.append({"path": str(path), "line": line_no, "error": str(exc)})
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows, {"raw_rows": len(rows), "invalid_lines": invalid_lines}


def fold_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if fold_supersession_events is None:
        return rows, {"folded": False, "reason": "fold_supersession_events unavailable"}
    folded, metadata = fold_supersession_events(rows)
    return [row for row in folded if isinstance(row, dict)], {
        "folded": True,
        "input_rows": len(rows),
        "folded_rows": len(folded),
        "metadata": metadata,
    }


def _eval_details(row: Mapping[str, Any]) -> Mapping[str, Any]:
    details = row.get("eval_details") or {}
    return details if isinstance(details, Mapping) else {}


def _nested_details(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = _eval_details(row).get("details") or {}
    return nested if isinstance(nested, Mapping) else {}


def question_results(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    details = _eval_details(row)
    nested = _nested_details(row)
    for key in ("question_results", "per_question_results", "per_question"):
        raw = details.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        raw = nested.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    raw = row.get("question_results")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _trial_id(row: Mapping[str, Any]) -> int | None:
    try:
        return int(row["trial_id"])
    except (KeyError, TypeError, ValueError):
        return None


def _tier(row: Mapping[str, Any]) -> str:
    raw = row.get("tier")
    if raw is None:
        raw = _eval_details(row).get("tier")
    return str(raw if raw is not None else "unknown")


def _core_id(row: Mapping[str, Any]) -> str:
    details = _eval_details(row)
    raw = row.get("core_id") or details.get("core_id") or _nested_details(row).get("core_id")
    return str(raw or "unknown")


def _action_type(row: Mapping[str, Any]) -> str:
    snapshot = row.get("config_snapshot") or {}
    if not isinstance(snapshot, Mapping):
        snapshot = {}
    raw = row.get("action_type") or row.get("action") or snapshot.get("type")
    return str(raw or "unknown")


def _config_fingerprint(row: Mapping[str, Any]) -> str | None:
    details = _eval_details(row)
    snapshot = row.get("config_snapshot") or {}
    if not isinstance(snapshot, Mapping):
        snapshot = {}
    raw = (
        row.get("config_fingerprint")
        or details.get("config_fingerprint")
        or snapshot.get("config_fingerprint")
    )
    if raw:
        return str(raw)
    if snapshot:
        payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
        return "sha1:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return None


def _hypothesis_text(row: Mapping[str, Any]) -> str | None:
    for key in ("hypothesis", "expected_mechanism", "planner_reasoning", "reasoning"):
        raw = row.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _surrogate_feedback(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return EV-10b surrogate-verifier feedback when a journal row carries it."""
    details = _eval_details(row)
    nested = _nested_details(row)
    for source in (nested, details):
        raw = source.get("surrogate_feedback")
        if isinstance(raw, Mapping):
            return raw
    return {}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _surrogate_feedback_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    proxy_rewards: list[float] = []
    rows_with_feedback = 0
    accepted = 0
    dense_feedback = 0
    opaque_only = 0
    trial_ids: list[int] = []

    for row in rows:
        feedback = _surrogate_feedback(row)
        if not feedback:
            continue
        rows_with_feedback += 1
        trial_id = _trial_id(row)
        if trial_id is not None:
            trial_ids.append(trial_id)
        reward = _float_or_none(feedback.get("proxy_reward"))
        if reward is not None:
            proxy_rewards.append(reward)
        if feedback.get("accepted") is True:
            accepted += 1
        if feedback.get("dense_feedback") is True:
            dense_feedback += 1
        if feedback.get("opaque_only") is True:
            opaque_only += 1

    if rows_with_feedback == 0:
        status = "not_present"
    elif opaque_only:
        status = "oracle_conflict"
    elif dense_feedback:
        status = "dense_feedback"
    elif accepted == rows_with_feedback:
        status = "accepted_only"
    else:
        status = "present"

    return {
        "rows": rows_with_feedback,
        "accepted": accepted,
        "dense_feedback": dense_feedback,
        "opaque_only": opaque_only,
        "proxy_reward_rows": len(proxy_rewards),
        "avg_proxy_reward": round(sum(proxy_rewards) / len(proxy_rewards), 4)
        if proxy_rewards
        else None,
        "trial_id_min": min(trial_ids) if trial_ids else None,
        "trial_id_max": max(trial_ids) if trial_ids else None,
        "status": status,
        "interpretation": (
            "EV-10b surrogate-verifier feedback is read-only here; ground-truth "
            "oracles remain authoritative when present."
        ),
    }


def _question_key(item: Mapping[str, Any]) -> tuple[str, str] | None:
    qid = str(item.get("qid") or item.get("question_id") or item.get("id") or "").strip()
    if not qid:
        return None
    suite = str(item.get("suite") or "unknown").strip() or "unknown"
    return (suite, qid)


def load_pool(pool_path: Path | None) -> dict[str, Any]:
    if pool_path is None:
        return {
            "pool_path": None,
            "pool_rows": 0,
            "metadata_total_questions": None,
            "stable_question_keys": set(),
            "raw_question_keys": set(),
            "suite_counts": {},
        }

    stable_keys: set[tuple[str, str]] = set()
    raw_keys: set[tuple[str, str]] = set()
    suite_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    metadata_total_questions: int | None = None
    pool_rows = 0

    with pool_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get(POOL_METADATA_KEY):
                try:
                    metadata_total_questions = int(row.get("total_questions"))
                except (TypeError, ValueError):
                    metadata_total_questions = None
                continue
            suite = str(row.get("suite") or "unknown").strip() or "unknown"
            tier = str(row.get("tier") if row.get("tier") is not None else "unknown")
            pool_rows += 1
            suite_counts[suite] += 1
            tier_counts[tier] += 1
            raw_id = str(row.get("id") or row.get("question_id") or "").strip()
            if raw_id:
                raw_keys.add((suite, raw_id))
                raw_keys.add((suite, f"{suite}/{raw_id}"))
            prompt = str(row.get("prompt") or "")
            if prompt:
                stable_keys.add((suite, _stable_question_qid(suite, prompt)))

    return {
        "pool_path": str(pool_path),
        "pool_rows": pool_rows,
        "metadata_total_questions": metadata_total_questions,
        "stable_question_keys": stable_keys,
        "raw_question_keys": raw_keys,
        "suite_counts": dict(sorted(suite_counts.items())),
        "tier_counts": dict(sorted(tier_counts.items())),
    }


def build_report(
    *,
    journal_paths: list[Path],
    pool_path: Path | None,
    fold_supersessions: bool = True,
) -> dict[str, Any]:
    raw_rows, load_meta = load_jsonl(journal_paths)
    rows, fold_meta = (
        fold_rows(raw_rows)
        if fold_supersessions
        else (
            raw_rows,
            {"folded": False, "reason": "disabled"},
        )
    )
    pool = load_pool(pool_path)

    trial_ids = [_trial_id(row) for row in rows]
    trial_ids = [trial_id for trial_id in trial_ids if trial_id is not None]
    eval_rows = [row for row in rows if question_results(row)]

    question_counts: Counter[tuple[str, str]] = Counter()
    partition_counts: Counter[str] = Counter()
    suite_attempt_counts: Counter[str] = Counter()
    suite_distinct: dict[str, set[str]] = defaultdict(set)
    tier_counts: Counter[str] = Counter()
    tier_question_counts: Counter[str] = Counter()
    tier_distinct: dict[str, set[tuple[str, str]]] = defaultdict(set)
    core_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    config_fingerprints: set[str] = set()
    hypotheses: set[str] = set()

    for row in rows:
        action_counts[_action_type(row)] += 1
        fingerprint = _config_fingerprint(row)
        if fingerprint:
            config_fingerprints.add(fingerprint)
        hypothesis = _hypothesis_text(row)
        if hypothesis:
            hypotheses.add(hypothesis)

    for row in eval_rows:
        tier_counts[_tier(row)] += 1
        core_counts[_core_id(row)] += 1
        for result in question_results(row):
            key = _question_key(result)
            if key is None:
                continue
            suite, qid = key
            partition = str(result.get("partition") or "core").strip() or "core"
            question_counts[key] += 1
            partition_counts[partition] += 1
            suite_attempt_counts[suite] += 1
            suite_distinct[suite].add(qid)
            tier = _tier(row)
            tier_question_counts[tier] += 1
            tier_distinct[tier].add(key)

    pool_stable_keys: set[tuple[str, str]] = pool["stable_question_keys"]
    pool_raw_keys: set[tuple[str, str]] = pool["raw_question_keys"]
    journal_keys = set(question_counts)
    matched_stable_keys = journal_keys & pool_stable_keys
    matched_raw_keys = journal_keys & pool_raw_keys
    pool_stable_count = len(pool_stable_keys)
    distinct_journal_count = len(journal_keys)
    question_rows = sum(question_counts.values())

    top_repeated = [
        {"suite": suite, "qid": qid, "attempts": attempts}
        for (suite, qid), attempts in question_counts.most_common(20)
    ]
    suite_distinct_counts = {suite: len(qids) for suite, qids in sorted(suite_distinct.items())}
    tier_distinct_counts = {tier: len(keys) for tier, keys in sorted(tier_distinct.items())}
    pool_tier_counts = pool["tier_counts"]
    tier_coverage = {}
    for tier, distinct_count in tier_distinct_counts.items():
        pool_count = int(pool_tier_counts.get(tier, 0) or 0)
        tier_coverage[tier] = {
            "eval_bearing_trials": int(tier_counts.get(tier, 0)),
            "question_result_rows": int(tier_question_counts.get(tier, 0)),
            "distinct_journal_question_keys": distinct_count,
            "pool_question_keys": pool_count,
            "distinct_vs_pool_pct": _safe_pct(distinct_count, pool_count),
        }
    for tier, pool_count in pool_tier_counts.items():
        tier_coverage.setdefault(
            tier,
            {
                "eval_bearing_trials": int(tier_counts.get(tier, 0)),
                "question_result_rows": int(tier_question_counts.get(tier, 0)),
                "distinct_journal_question_keys": 0,
                "pool_question_keys": int(pool_count or 0),
                "distinct_vs_pool_pct": 0.0,
            },
        )

    coverage = {
        "question_result_rows": question_rows,
        "distinct_journal_question_keys": distinct_journal_count,
        "pool_stable_question_keys": pool_stable_count,
        "pool_raw_question_keys": len(pool_raw_keys),
        "matched_pool_stable_question_keys": len(matched_stable_keys),
        "matched_pool_raw_question_keys": len(matched_raw_keys),
        "distinct_vs_pool_stable_upper_bound_pct": _safe_pct(
            distinct_journal_count, pool_stable_count
        ),
        "matched_pool_stable_pct": _safe_pct(len(matched_stable_keys), pool_stable_count),
        "repeat_factor": round(question_rows / distinct_journal_count, 4)
        if distinct_journal_count
        else 0.0,
        "status": "low_coverage"
        if _safe_pct(distinct_journal_count, pool_stable_count) < 10
        else "ok",
        "interpretation": (
            "Fixed authority-core repetition is acceptable for paired safety evidence; "
            "planner-learning coverage is narrow if this is the dominant optimization signal."
        ),
    }

    return {
        "schema_version": "autopilot_eval_task_coverage.v1",
        "generated_at": _now_iso(),
        "journal": {
            "paths": [str(path) for path in journal_paths],
            "load": load_meta,
            "fold": fold_meta,
            "rows": len(rows),
            "unique_trial_ids": len(set(trial_ids)),
            "trial_id_min": min(trial_ids) if trial_ids else None,
            "trial_id_max": max(trial_ids) if trial_ids else None,
            "eval_bearing_trials": len(eval_rows),
            "eval_trials_by_tier": dict(sorted(tier_counts.items())),
            "eval_trials_by_core_id": dict(core_counts.most_common(20)),
        },
        "coverage": coverage,
        "questions": {
            "partition_attempt_counts": dict(sorted(partition_counts.items())),
            "suite_attempt_counts": dict(sorted(suite_attempt_counts.items())),
            "suite_distinct_question_counts": suite_distinct_counts,
            "tier_question_counts": dict(sorted(tier_question_counts.items())),
            "tier_distinct_question_counts": tier_distinct_counts,
            "tier_coverage": dict(sorted(tier_coverage.items())),
            "top_repeated_questions": top_repeated,
        },
        "planner_diversity": {
            "action_type_counts": dict(action_counts.most_common()),
            "unique_action_types": len(action_counts),
            "unique_config_fingerprints": len(config_fingerprints),
            "unique_hypotheses": len(hypotheses),
        },
        "surrogate_verifier": _surrogate_feedback_summary(rows),
        "pool": {
            "path": pool["pool_path"],
            "rows": pool["pool_rows"],
            "metadata_total_questions": pool["metadata_total_questions"],
            "stable_question_keys": pool_stable_count,
            "raw_question_keys": len(pool_raw_keys),
            "suite_counts": pool["suite_counts"],
            "tier_counts": pool_tier_counts,
        },
        "recommendation": {
            "do_not_change_mid_w8": True,
            "lane_split": [
                "authority_core: fixed paired core for W4/W6/W8 promotion evidence",
                "exploration_coverage: rotating/advisory pool for planner learning",
                "promotion_holdout: fresh held-out T2 acceptance evidence",
            ],
            "next_step": (
                "Use this report as a guardrail before changing sampling policy; "
                "introduce any rotation behind a new instrument-era label."
            ),
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    journal = report["journal"]
    planner = report["planner_diversity"]
    questions = report["questions"]
    pool = report["pool"]
    surrogate = report["surrogate_verifier"]
    lines = [
        "# AutoPilot Eval Task Coverage",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Journal rows: `{journal['rows']}`; eval-bearing trials: `{journal['eval_bearing_trials']}`",
        f"- Trial id range: `{journal['trial_id_min']}` to `{journal['trial_id_max']}`",
        f"- Scored question rows: `{coverage['question_result_rows']}`",
        f"- Distinct scored qids: `{coverage['distinct_journal_question_keys']}`",
        f"- Pool stable qids: `{coverage['pool_stable_question_keys']}`",
        f"- Upper-bound pool coverage: `{coverage['distinct_vs_pool_stable_upper_bound_pct']}%`",
        f"- Stable-qid matches in pool: `{coverage['matched_pool_stable_question_keys']}` "
        f"(`{coverage['matched_pool_stable_pct']}%`)",
        f"- Repeat factor: `{coverage['repeat_factor']}x`",
        f"- Status: `{coverage['status']}`",
        "",
        "## Interpretation",
        "",
        coverage["interpretation"],
        "",
        "The fixed authority core should remain stable during W6/W8 collection. "
        "Planner exploration needs a separate rotating/advisory lane so tunables "
        "do not optimize only the repeated authority slice.",
        "",
        "## Planner Diversity",
        "",
        f"- Unique action types: `{planner['unique_action_types']}`",
        f"- Unique config fingerprints: `{planner['unique_config_fingerprints']}`",
        f"- Unique hypotheses: `{planner['unique_hypotheses']}`",
        "",
        "## Surrogate Verifier Feedback",
        "",
        f"- Rows: `{surrogate['rows']}`",
        f"- Accepted: `{surrogate['accepted']}`",
        f"- Dense feedback: `{surrogate['dense_feedback']}`",
        f"- Opaque-only oracle conflicts: `{surrogate['opaque_only']}`",
        f"- Average proxy reward: `{surrogate['avg_proxy_reward']}`",
        f"- Status: `{surrogate['status']}`",
        "",
        surrogate["interpretation"],
        "",
        "## Tier Coverage",
        "",
        "| tier | eval trials | scored rows | distinct qids | pool qids | coverage |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for tier, item in sorted(questions["tier_coverage"].items()):
        lines.append(
            f"| {tier} | {item['eval_bearing_trials']} | {item['question_result_rows']} | "
            f"{item['distinct_journal_question_keys']} | {item['pool_question_keys']} | "
            f"{item['distinct_vs_pool_pct']}% |"
        )
    non_sentinel_low = [
        (suite, count)
        for suite, count in sorted(
            questions["suite_distinct_question_counts"].items(),
            key=lambda item: item[1],
        )
        if not suite.startswith("sentinel_")
    ][:10]
    lines.extend(
        [
            "",
            "## Least-Covered Non-Sentinel Suites",
            "",
            "| suite | distinct qids |",
            "|---|---:|",
        ]
    )
    for suite, count in non_sentinel_low:
        lines.append(f"| {suite} | {count} |")
    lines.extend(
        [
            "",
            "## Pool",
            "",
            f"- Path: `{pool['path']}`",
            f"- Rows: `{pool['rows']}`",
            f"- Metadata total questions: `{pool['metadata_total_questions']}`",
            "",
            "## Top Repeated Questions",
            "",
            "| suite | qid | attempts |",
            "|---|---:|---:|",
        ]
    )
    for item in questions["top_repeated_questions"][:10]:
        lines.append(f"| {item['suite']} | `{item['qid']}` | {item['attempts']} |")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Keep `authority_core` fixed for paired promotion evidence.",
            "- Add a rotating `exploration_coverage` lane before treating planner learning as broad.",
            "- Fence any sampler change with a new instrument-era label.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--journal",
        action="append",
        type=Path,
        default=[],
        help="Journal JSONL file or directory. May be repeated. Defaults to orchestration/autopilot_journal*.jsonl.",
    )
    parser.add_argument(
        "--journal-dir",
        type=Path,
        default=DEFAULT_JOURNAL_DIR,
        help="Journal directory used when --journal is omitted.",
    )
    parser.add_argument("--pool", type=Path, default=None, help="Question-pool JSONL.")
    parser.add_argument("--json-out", type=Path, default=None, help="Write JSON report.")
    parser.add_argument("--markdown-out", type=Path, default=None, help="Write Markdown report.")
    parser.add_argument(
        "--no-fold-supersessions",
        action="store_true",
        help="Disable journal supersession folding.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    journal_paths = resolve_journal_paths(args.journal, args.journal_dir)
    if not journal_paths:
        raise SystemExit("no autopilot journal JSONL files found")
    pool_path = args.pool or default_pool_path()
    report = build_report(
        journal_paths=journal_paths,
        pool_path=pool_path,
        fold_supersessions=not args.no_fold_supersessions,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    if not args.json_out and not args.markdown_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        coverage = report["coverage"]
        print(
            "coverage="
            f"{coverage['distinct_journal_question_keys']}/"
            f"{coverage['pool_stable_question_keys']} "
            f"({coverage['distinct_vs_pool_stable_upper_bound_pct']}%), "
            f"repeat_factor={coverage['repeat_factor']}x, "
            f"status={coverage['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
