"""Autopilot daily digest generator.

Writes an append-only markdown digest of NumericSwarm + StructuralLab +
Pareto archive state to ``progress/YYYY-MM/YYYY-MM-DD-autopilot.md``.

Design goals:
- **Pure additive**: never edits existing handoffs; never overwrites prior
  content. Each invocation appends a new ``## YYYY-MM-DD HH:MM:SS UTC``
  section so re-running mid-day is safe.
- **No quality-judgment side effects**: the digest reports what
  NumericSwarm / StructuralLab / ParetoArchive already know. It does not
  decide verdicts or promote findings — those remain manual review steps.
- **Cheap**: no inference calls. Reads in-memory dataclass summaries plus
  the autopilot_state.json snapshot.

Triggered from ``autopilot._run_loop_inner`` once per UTC day (tracked via
``state["last_digest_date"]``). Also callable via ``autopilot.py digest``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("autopilot.digest")


# Default location: workspace progress tree. Lives outside the
# orchestrator repo because that's where session progress writes already
# go (see /workspace/progress/YYYY-MM/).
_DEFAULT_PROGRESS_ROOT = Path("/workspace/progress")


def _digest_path_for(date: datetime, root: Path | None = None) -> Path:
    """Return ``progress/YYYY-MM/YYYY-MM-DD-autopilot.md`` for ``date``."""
    base = root or _DEFAULT_PROGRESS_ROOT
    return base / f"{date.year:04d}-{date.month:02d}" / f"{date.strftime('%Y-%m-%d')}-autopilot.md"


def _fmt_num(v: Any, precision: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{precision}g}"
    return str(v)


def _surface_section(swarm: Any, surface: str) -> list[str]:
    """Render a single NumericSwarm surface as markdown lines."""
    lines: list[str] = []
    try:
        study = swarm._get_study(surface)
    except Exception as e:
        return [f"- (could not load study `{surface}`: {e})"]
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    n = len(completed)
    lines.append(f"#### `{surface}` — {n} completed trials")
    if n == 0:
        lines.append("  - no completed trials yet")
        return lines
    best_q = max((t.values[0] for t in completed if t.values), default=None)
    if best_q is not None:
        lines.append(f"  - best quality (obj 0): **{_fmt_num(best_q)}**")
    if n >= 3:
        try:
            best = swarm.best_params(surface, method="cluster")
            if best:
                fmt = ", ".join(f"`{k}={_fmt_num(v)}`" for k, v in best.items())
                lines.append(f"  - cluster-selected best: {fmt}")
        except Exception as e:
            lines.append(f"  - cluster selection unavailable: {e}")
    if n >= 10:
        try:
            imp = swarm.importance(surface)
            if imp:
                ranked = sorted(imp.items(), key=lambda kv: -kv[1])
                fmt = ", ".join(f"`{k}`={_fmt_num(v, 3)}" for k, v in ranked)
                lines.append(f"  - fANOVA importance (quality): {fmt}")
        except Exception:
            pass  # importance is optional
    return lines


def _archive_section(archive: Any) -> list[str]:
    """Render Pareto archive state as markdown lines."""
    lines = ["### Pareto archive"]
    try:
        n_entries = len(getattr(archive, "_archive", []) or [])
    except Exception:
        n_entries = -1
    lines.append(f"- entries: {n_entries if n_entries >= 0 else 'unknown'}")
    try:
        hv_slope = archive.hypervolume_slope(50)
        lines.append(f"- hypervolume slope (last 50): **{_fmt_num(hv_slope)}**")
    except Exception:
        pass
    try:
        hv = archive.hypervolume()
        lines.append(f"- current hypervolume: {_fmt_num(hv)}")
    except Exception:
        pass
    return lines


def _structural_lab_section(lab: Any) -> list[str]:
    """Render StructuralLab state as markdown lines."""
    lines = ["### StructuralLab"]
    try:
        s = lab.summary()
        for k in ("memory_count", "checkpoints", "has_production_best"):
            if k in s:
                lines.append(f"- {k.replace('_', ' ')}: **{s[k]}**")
    except Exception as e:
        lines.append(f"- (summary unavailable: {e})")
    return lines


def _state_section(state: dict[str, Any]) -> list[str]:
    """Render relevant fields from autopilot_state.json."""
    lines = ["### State snapshot"]
    keys = (
        "trial_counter",
        "session_id",
        "paused",
        "consecutive_failures",
        "species_budget",
        "last_digest_date",
    )
    for k in keys:
        if k in state:
            v = state[k]
            if isinstance(v, dict):
                fmt = ", ".join(f"{kk}={_fmt_num(vv, 2)}" for kk, vv in v.items())
                lines.append(f"- {k}: {{{fmt}}}")
            else:
                lines.append(f"- {k}: `{v}`")
    return lines


def _journal_section(journal: Any, lookback: int = 20) -> list[str]:
    """Render recent trial outcomes from ExperimentJournal (best-effort)."""
    lines = [f"### Recent trials (last {lookback})"]
    rows: list[Any] = []
    for attr in ("recent_entries", "tail", "last_n", "list_recent"):
        fn = getattr(journal, attr, None)
        if callable(fn):
            try:
                rows = list(fn(lookback))
                break
            except Exception:
                continue
    if not rows:
        lines.append("- (no recent entries readable via known journal API)")
        return lines
    for row in rows[-lookback:]:
        if is_dataclass(row):
            row = asdict(row)
        if isinstance(row, dict):
            trial = row.get("trial_id", row.get("trial", "?"))
            sp = row.get("species", "?")
            qty = row.get("quality", row.get("q", "?"))
            lines.append(f"- trial {trial} [{sp}] q={_fmt_num(qty)}")
        else:
            lines.append(f"- {row}")
    return lines


def _journal_entries_for_digest(journal: Any) -> list[Any]:
    for attr in ("entries_with_supersessions", "trustworthy_entries"):
        fn = getattr(journal, attr, None)
        if callable(fn):
            try:
                return list(fn())
            except Exception:
                continue
    entries = getattr(journal, "_entries", None)
    if entries is not None:
        try:
            return list(entries)
        except Exception:
            pass
    return []


def _mechanism_class(action_type: str) -> str:
    if action_type in {"prompt_mutation", "gepa_optimize"}:
        return "prompt_search"
    if action_type in {"numeric_trial", "structural_experiment", "code_mutation", "consult_gate_probe"}:
        return "deterministic_code_config"
    if action_type in {"train_routing_models", "distill_skillbank", "seed_batch"}:
        return "data_training"
    if action_type in {"deep_eval", "rollback"}:
        return "evaluation_control"
    return "other"


def _mechanism_effectiveness_section(journal: Any) -> list[str]:
    """Observe-only frontier-rate split by mechanism class.

    This intentionally reports only; it does not feed MetaOptimizer budgets,
    planner prompts, SafetyGate, or archive authority.
    """
    lines = ["### Mechanism effectiveness (observe-only)"]
    rows = []
    for entry in _journal_entries_for_digest(journal):
        if getattr(entry, "bug_corrupted_by", ""):
            continue
        try:
            tier = int(getattr(entry, "tier", 0))
        except (TypeError, ValueError):
            tier = 0
        if tier <= 0:
            continue
        rows.append(entry)
    if not rows:
        lines.append("- no trustworthy T1/T2 journal rows available")
        return lines

    stats: dict[str, dict[str, Any]] = {}
    for entry in rows:
        action_type = str(getattr(entry, "action_type", "") or "unknown")
        mechanism = _mechanism_class(action_type)
        item = stats.setdefault(
            mechanism,
            {"total": 0, "frontier": 0, "actions": set()},
        )
        item["total"] += 1
        item["actions"].add(action_type)
        if getattr(entry, "pareto_status", "") == "frontier":
            item["frontier"] += 1

    lines.extend(
        [
            "| Mechanism class | T1/T2 rows | Frontier rows | Frontier rate | Action types |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for mechanism, item in sorted(
        stats.items(), key=lambda kv: (-kv[1]["total"], kv[0])
    ):
        total = int(item["total"])
        frontier = int(item["frontier"])
        rate = frontier / total if total else 0.0
        action_types = ", ".join(f"`{name}`" for name in sorted(item["actions"]))
        lines.append(
            f"| `{mechanism}` | {total} | {frontier} | {_fmt_num(rate, 3)} | {action_types} |"
        )
    lines.append(
        "- note: observe-only diagnostic; no planner budget, prompt, gate, or archive authority change."
    )
    return lines


def _economics_section(now: datetime, repo_root: Path | None = None) -> list[str]:
    """Render a compact read-only economics summary for the daily digest."""
    root = repo_root or Path(__file__).resolve().parents[2]
    start = (now.astimezone(timezone.utc) - timedelta(days=6)).date()
    try:
        from scripts.economics.ledger import summarize_economics

        ledger = summarize_economics(
            week_start=start,
            planner_archive=root / "logs" / "planner_archive.jsonl",
            journal_dir=root / "orchestration",
            cloud_costs=root / "orchestration" / "cloud_costs.yaml",
            rules_path=root / "orchestration" / "economic_rules.yaml",
            progress_root=Path("/mnt/raid0/llm/epyc-root/progress"),
            orch_progress_dir=root / "logs" / "progress",
            now=now,
        )
    except Exception as e:
        return [
            "### Economics (last 7 days)",
            f"- unavailable: {e}",
        ]
    rules_source = "configured" if ledger.rules.source_exists else "built-in defaults"
    planner_status = "triggered" if ledger.review.planner_spend_triggered else "hold"
    gate_latency = ledger.review.operator_gate_latency_triggered
    if gate_latency is None:
        gate_latency_status = "not evaluated"
    else:
        gate_latency_status = "triggered" if gate_latency else "hold"
    return [
        "### Economics (last 7 days)",
        f"- planner cloud spend: ${ledger.planner.total_usd:.4f}",
        f"- manual cloud spend: ${ledger.manual.total_usd:.4f}",
        f"- total cloud spend: ${ledger.total_cloud_usd:.4f}",
        f"- local eval wall time: {ledger.local.eval_hours:.2f}h",
        f"- autopilot eval trials: {ledger.local.trials}",
        f"- decision markers: {ledger.throughput.progress_decision_markers}",
        (
            "- planner monthly projection: "
            f"${ledger.review.projected_monthly_planner_spend_usd:.2f} / "
            f"${ledger.rules.planner_monthly_spend_threshold_usd:.2f} ({planner_status})"
        ),
        f"- operator gate-latency rule: {gate_latency_status}",
        f"- economic rules source: {rules_source}",
    ]


def render_digest(
    *,
    swarm: Any,
    lab: Any,
    archive: Any,
    state: dict[str, Any],
    journal: Any = None,
    archive_source: str | None = None,
    now: datetime | None = None,
) -> str:
    """Build the markdown digest body for the current snapshot."""
    now = now or datetime.now(timezone.utc)
    body: list[str] = []
    body.append(f"## {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    body.append("")
    body.extend(_state_section(state))
    body.append("")
    body.extend(_archive_section(archive))
    if archive_source:
        body.append(f"- archive source: `{archive_source}`")
    body.append("")
    body.extend(_structural_lab_section(lab))
    body.append("")
    body.extend(_economics_section(now))
    body.append("")
    body.append("### NumericSwarm surfaces")
    body.append("")
    # Import SURFACES lazily so the digest module stays import-cheap.
    from scripts.autopilot.species.numeric_swarm import SURFACES
    for surface in SURFACES:
        body.extend(_surface_section(swarm, surface))
        body.append("")
    if journal is not None:
        body.extend(_mechanism_effectiveness_section(journal))
        body.append("")
        body.extend(_journal_section(journal))
        body.append("")
    body.append("---")
    body.append("")
    return "\n".join(body)


def generate_digest(
    *,
    swarm: Any,
    lab: Any,
    archive: Any,
    state: dict[str, Any],
    journal: Any = None,
    archive_source: str | None = None,
    output_root: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Generate and append a digest section to today's autopilot digest file.

    The file lives at ``progress/YYYY-MM/YYYY-MM-DD-autopilot.md``. If the
    file does not exist, it is created with a header comment. Each call
    appends a new ``## YYYY-MM-DD HH:MM:SS UTC`` section — multiple calls
    per day are safe and result in multiple stamped sections (useful when
    debugging the loop or after a forced re-run).
    """
    now = now or datetime.now(timezone.utc)
    path = _digest_path_for(now, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = render_digest(
        swarm=swarm,
        lab=lab,
        archive=archive,
        state=state,
        journal=journal,
        archive_source=archive_source,
        now=now,
    )
    if not path.exists():
        header = (
            f"# Autopilot Digest — {now.strftime('%Y-%m-%d')}\n\n"
            "Auto-generated by `scripts/autopilot/digest.py`. Each section below "
            "captures the autopilot state at the indicated UTC timestamp. "
            "Append-only: re-runs add new sections rather than overwriting prior ones. "
            "This file informs the next manual review (see "
            "`handoffs/active/research-evaluation-index.md` §P11 + "
            "`handoffs/active/autopilot-continuous-optimization.md` for the active "
            "search-space catalogue).\n\n---\n\n"
        )
        path.write_text(header + body)
        log.info("Wrote new digest file: %s", path)
    else:
        with path.open("a") as f:
            f.write(body)
        log.info("Appended digest section to: %s", path)
    return path


def should_generate_today(state: dict[str, Any], now: datetime | None = None) -> bool:
    """Return True iff today's UTC digest has not yet been generated.

    Reads ``state["last_digest_date"]`` (set after a successful generation).
    """
    now = now or datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    return state.get("last_digest_date") != today
