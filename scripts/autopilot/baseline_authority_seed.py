#!/usr/bin/env python3
"""Prepare a baseline-promotion seed event from current state baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from preflight_audit import JOURNAL_PATH, STATE_PATH, _load_jsonl  # noqa: E402
from src.autopilot_core.baseline_ledger import (  # noqa: E402
    BASELINE_PROMOTION_EVENT_TYPE,
    canonical_jsonable,
    reconcile_baseline_ledger,
)


POLICY_VERSION = "baseline-state-seed-v1"


@dataclass(frozen=True)
class BaselineSeedResult:
    status: str
    event: dict[str, Any] | None = None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    warning: str = ""
    live_autopilot_pids: list[int] | None = None


def _load_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"state file is not a JSON object: {path}")
    return state


def _max_trial_id(rows: list[dict[str, Any]]) -> int | None:
    max_id: int | None = None
    for row in rows:
        try:
            trial_id = int(row.get("trial_id"))
        except (AttributeError, TypeError, ValueError):
            continue
        if max_id is None or trial_id > max_id:
            max_id = trial_id
    return max_id


def _numeric_tiers(baseline_state: dict[str, Any]) -> dict[int, float]:
    tiers = baseline_state.get("baselines_by_tier")
    if not isinstance(tiers, dict):
        return {}
    parsed: dict[int, float] = {}
    for raw_tier, raw_quality in tiers.items():
        try:
            parsed[int(raw_tier)] = float(raw_quality)
        except (TypeError, ValueError):
            continue
    return parsed


def _reconciliation_view(rows: list[dict[str, Any]], baseline_state: dict[str, Any] | None) -> dict[str, Any]:
    reconciliation = reconcile_baseline_ledger(rows, baseline_state)
    return {
        "status": reconciliation.status,
        "event_count": reconciliation.event_count,
        "valid_snapshot_count": reconciliation.valid_snapshot_count,
        "cutover_ready": reconciliation.cutover_ready,
        "cutover_blockers": reconciliation.cutover_blockers,
        "warnings": reconciliation.warnings,
    }


def build_baseline_seed_event(
    state: dict[str, Any],
    journal_rows: list[dict[str, Any]],
    *,
    tier: int | None = None,
    actor: str = "baseline_authority_seed.py",
) -> BaselineSeedResult:
    """Build a baseline-promotion seed event without writing it."""
    baseline_state = state.get("baseline_state")
    if not isinstance(baseline_state, dict) or not baseline_state:
        return BaselineSeedResult(
            status="no_state_baseline",
            warning="state has no usable baseline_state to seed",
        )

    normalized_baseline = canonical_jsonable(baseline_state)
    before = _reconciliation_view(journal_rows, normalized_baseline)
    if before["cutover_ready"]:
        return BaselineSeedResult(status="already_aligned", before=before)
    if before["event_count"]:
        return BaselineSeedResult(
            status="existing_ledger_blocked",
            before=before,
            warning="existing baseline promotion ledger is not cutover-ready",
        )

    tiers = _numeric_tiers(normalized_baseline)
    if not tiers:
        return BaselineSeedResult(
            status="no_numeric_baseline_tiers",
            before=before,
            warning="baseline_state.baselines_by_tier has no numeric tier values",
        )
    selected_tier = int(tier) if tier is not None else max(tiers)
    if selected_tier not in tiers:
        return BaselineSeedResult(
            status="missing_requested_tier",
            before=before,
            warning=f"baseline_state has no numeric tier {selected_tier}",
        )

    journal_max_trial_id = _max_trial_id(journal_rows)
    try:
        state_trial_counter = int(state.get("trial_counter"))
    except (TypeError, ValueError):
        state_trial_counter = None
    source_trial_id = (
        int(journal_max_trial_id)
        if journal_max_trial_id is not None
        else max(0, int(state_trial_counter or 1) - 1)
    )
    now = datetime.now(timezone.utc).isoformat()
    event = {
        "type": BASELINE_PROMOTION_EVENT_TYPE,
        "source_trial_id": source_trial_id,
        "tier": selected_tier,
        "previous_quality": None,
        "new_quality": tiers[selected_tier],
        "reason": "seed_current_state_baseline_for_fold_authority",
        "proof": {
            "seeded_from_state_baseline": True,
            "state_trial_counter": state_trial_counter,
            "journal_max_trial_id": journal_max_trial_id,
            "baseline_tiers": sorted(tiers),
            "policy_contract": (
                "event mirrors the existing state baseline so baseline-as-fold "
                "authority can be validated before removing the state cache"
            ),
        },
        "result_metrics": {
            "quality": tiers[selected_tier],
            "source": "state_baseline_seed",
        },
        "baseline_state": normalized_baseline,
        "policy_version": POLICY_VERSION,
        "actor": actor,
        "timestamp": now,
    }
    after = _reconciliation_view([*journal_rows, event], normalized_baseline)
    if not after["cutover_ready"]:
        return BaselineSeedResult(
            status="postcheck_failed",
            event=event,
            before=before,
            after=after,
            warning="seed event did not make baseline ledger cutover-ready",
        )
    return BaselineSeedResult(status="ready", event=event, before=before, after=after)


def _autopilot_running_pids() -> list[int]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "autopilot.py start"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    except FileNotFoundError:
        return []
    pids: list[int] = []
    me = os.getpid()
    for line in out.strip().splitlines():
        parts = line.split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == me:
            continue
        if "baseline_authority_seed.py" in line:
            continue
        pids.append(pid)
    return pids


def append_baseline_seed_event(
    journal_path: Path,
    result: BaselineSeedResult,
) -> BaselineSeedResult:
    """Append a prepared baseline seed event to the JSONL journal."""
    if result.status != "ready" or result.event is None:
        return result
    live_pids = _autopilot_running_pids()
    if live_pids:
        return BaselineSeedResult(
            status="live_autopilot_running",
            event=result.event,
            before=result.before,
            after=result.after,
            warning="refusing to append baseline seed while AutoPilot is running",
            live_autopilot_pids=live_pids,
        )
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.event, sort_keys=True, default=str) + "\n")
    return BaselineSeedResult(
        status="written",
        event=result.event,
        before=result.before,
        after=result.after,
    )


def _summary_lines(result: BaselineSeedResult) -> list[str]:
    before = result.before or {}
    after = result.after or {}
    event = result.event or {}
    lines = [
        f"Baseline seed status: {result.status}",
        (
            "Before: "
            f"status={before.get('status', 'n/a')} "
            f"events={before.get('event_count', 'n/a')} "
            f"cutover_ready={before.get('cutover_ready', 'n/a')}"
        ),
    ]
    if after:
        lines.append(
            "After: "
            f"status={after.get('status', 'n/a')} "
            f"events={after.get('event_count', 'n/a')} "
            f"cutover_ready={after.get('cutover_ready', 'n/a')}"
        )
    if event:
        lines.append(
            "Prepared event: "
            f"source_trial_id={event.get('source_trial_id')} "
            f"tier={event.get('tier')} "
            f"new_quality={event.get('new_quality')}"
        )
    if result.live_autopilot_pids:
        pids = ",".join(str(pid) for pid in result.live_autopilot_pids)
        lines.append(f"Live AutoPilot PIDs: {pids}")
    if result.warning:
        lines.append(f"Warning: {result.warning}")
    return lines


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run by default. Prepare an append-only baseline_promotion seed "
            "event from the current state baseline so baseline-as-fold authority "
            "can be validated before removing baseline_state."
        )
    )
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--journal", type=Path, default=JOURNAL_PATH)
    parser.add_argument("--tier", type=int, help="Baseline tier to name on the seed event.")
    parser.add_argument("--append", action="store_true", help="Append the seed event.")
    parser.add_argument(
        "--expect-trial-counter",
        type=int,
        help="Refuse to append if state trial_counter differs from this value.",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    state_path = args.state.expanduser().resolve()
    journal_path = args.journal.expanduser().resolve()
    if not state_path.exists():
        print(f"state file does not exist: {state_path}", file=sys.stderr)
        return 2
    if not journal_path.exists():
        print(f"journal file does not exist: {journal_path}", file=sys.stderr)
        return 2
    try:
        state = _load_state(state_path)
    except (json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    actual_counter = state.get("trial_counter")
    if (
        args.append
        and args.expect_trial_counter is not None
        and actual_counter != args.expect_trial_counter
    ):
        result = BaselineSeedResult(
            status="trial_counter_mismatch",
            warning=(
                f"expected trial_counter {args.expect_trial_counter}, "
                f"found {actual_counter}"
            ),
        )
    else:
        result = build_baseline_seed_event(
            state,
            _load_jsonl(journal_path),
            tier=args.tier,
        )
        if args.append:
            result = append_baseline_seed_event(journal_path, result)

    if args.json:
        print(json.dumps(asdict(result), sort_keys=True, default=str))
    else:
        print("\n".join(_summary_lines(result)))
    if result.status in {"ready", "already_aligned", "written"}:
        return 0
    if result.status in {"trial_counter_mismatch", "live_autopilot_running"}:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
