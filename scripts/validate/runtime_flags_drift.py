#!/usr/bin/env python3
"""Diff the LIVE runtime feature flags against the tracked spec.

``orchestration/runtime_flags.json`` is gitignored live state the API rewrites,
so "what is actually on right now vs what should be" used to require
archaeology. This makes it one command.

Read-only: it never writes ``runtime_flags.json`` and never touches a process.
``--sync-spec`` writes only the tracked spec file.

Usage:
    python scripts/validate/runtime_flags_drift.py              # human report
    python scripts/validate/runtime_flags_drift.py --json       # machine report
    python scripts/validate/runtime_flags_drift.py --strict     # exit 1 on drift
    python scripts/validate/runtime_flags_drift.py --show       # full joined table
    python scripts/validate/runtime_flags_drift.py --sync-spec  # adopt new flags
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runtime_flag_spec import (  # noqa: E402
    BLOCKING_KINDS,
    SPEC_PATH,
    SpecError,
    autopilot_state_drift,
    baseline_posture,
    compute_drift,
    effective_posture,
    flag_metadata,
    live_records,
    load_spec,
    referenced_flag_names,
    registry_flag_names,
    render_spec,
    spec_coverage,
    sync_spec,
)

_KIND_ORDER = {
    "dependency_violation": 0,
    "contradicts_spec": 1,
    "undeclared_override": 2,
    "unknown_flag_in_live": 3,
    "redundant_override": 4,
}


def _bool(value: object) -> str:
    if value is None:
        return "-"
    return "on" if value else "off"


def _live_path() -> Path:
    from src.features import runtime_flags_path

    return runtime_flags_path()


def _report_text(payload: dict, *, verbose: bool) -> str:
    lines: list[str] = []
    lines.append("Runtime feature-flag drift")
    lines.append(f"  spec:   {payload['spec_path']}")
    lines.append(f"  live:   {payload['live_path']}")
    lines.append(
        f"  flags:  {payload['registry_flag_count']} declared in code, "
        f"{payload['live_override_count']} overridden in the live file"
    )

    missing, unknown = payload["coverage"]["missing_from_spec"], payload["coverage"]["unknown_in_spec"]
    if missing:
        lines.append(f"  SPEC GAP: {len(missing)} flag(s) in code but not in spec: {', '.join(missing)}")
        lines.append("            run --sync-spec")
    if unknown:
        lines.append(f"  SPEC GAP: {len(unknown)} spec entr(ies) for flags no longer in code: {', '.join(unknown)}")
    unregistered = payload["coverage"]["read_in_code_but_unregistered"]
    if unregistered:
        lines.append(
            f"  SPEC GAP: {len(unregistered)} flag(s) read off a Features object with no "
            f"FeatureSpec: {', '.join(unregistered)}"
        )

    drifts = payload["drift"]
    blocking = [d for d in drifts if d["kind"] in BLOCKING_KINDS]
    # Non-blocking findings are summarised unless asked for: 29 no-op overrides
    # would otherwise bury the 9 that actually change behaviour.
    shown = drifts if verbose else blocking
    lines.append("")
    if not drifts:
        lines.append("No drift: the live file matches the declared posture exactly.")
    else:
        lines.append(f"{len(drifts)} finding(s), {len(blocking)} of them blocking:")
        width = max(len(d["flag"]) for d in drifts)
        current_kind = None
        for item in shown:
            if item["kind"] != current_kind:
                current_kind = item["kind"]
                marker = "!!" if current_kind in BLOCKING_KINDS else "..."
                lines.append(f"\n  [{marker}] {current_kind}")
            lines.append(
                f"    {item['flag']:<{width}}  expected={_bool(item['expected']):<3} "
                f"effective={_bool(item['effective']):<3} baseline={_bool(item['baseline']):<3} "
                f"set_by={item['set_by'] or '-'} at {item['ts'] or '-'}"
            )
            if verbose and item["detail"]:
                lines.append(f"        {item['detail']}")
            if item["reason"]:
                lines.append(f"        reason: {item['reason']}")
        suppressed: dict[str, int] = {}
        for item in drifts:
            if item not in shown:
                suppressed[item["kind"]] = suppressed.get(item["kind"], 0) + 1
        if suppressed:
            lines.append("")
            for kind, count in sorted(suppressed.items()):
                lines.append(f"  [...] {kind}: {count} (use -v to list)")

    gates = payload["autopilot_state"]
    if gates:
        lines.append("")
        lines.append("autopilot_state.json gates:")
        for row in gates:
            status = row.get("status")
            lines.append(
                f"    [{status}] {row['key']}: expected={row.get('expected')!r} "
                f"live={row.get('live')!r} present={row.get('present')}"
            )

    return "\n".join(lines)


def _show_table() -> str:
    spec = load_spec()
    meta = flag_metadata()
    baseline, sources = baseline_posture()
    expected = spec.expected_posture()
    effective = effective_posture(spec)
    records = live_records()
    names = registry_flag_names()

    width = max(len(n) for n in names)
    header = (
        f"{'flag':<{width}}  {'test':<4} {'prod':<4} {'base':<4} {'expect':<6} "
        f"{'live':<4} {'eff':<4}  meaning"
    )
    lines = [header, "-" * len(header)]
    for name in names:
        record = records.get(name)
        live = "-" if record is None else _bool(record.get("value"))
        lines.append(
            f"{name:<{width}}  {_bool(meta[name]['default_test']):<4} "
            f"{_bool(meta[name]['default_prod']):<4} {_bool(baseline[name]):<4} "
            f"{_bool(expected[name]):<6} {live:<4} {_bool(effective[name]):<4}  "
            f"{meta[name]['description']}"
        )
    lines.append("")
    lines.append(
        "base = registry default_prod overridden by the stack's wave gating; "
        "expect = base overridden by spec pins;"
    )
    lines.append(
        "live = entry in runtime_flags.json ('-' = absent); eff = what the workers resolve to."
    )
    wave = [n for n, s in sources.items() if s == "stack:wave_override"]
    lines.append(f"wave-gated in orchestrator_stack.py: {', '.join(sorted(wave))}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit the machine report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when there is blocking drift or a spec coverage gap",
    )
    parser.add_argument("--show", action="store_true", help="print the full joined flag table")
    parser.add_argument(
        "--sync-spec",
        action="store_true",
        help="add flags new in code to the spec as 'baseline' and drop retired ones",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="include per-finding detail")
    parser.add_argument("--spec", type=Path, default=None, help="override the spec path")
    parser.add_argument("--live", type=Path, default=None, help="override the live flag file")
    args = parser.parse_args(argv)

    if args.sync_spec:
        spec_path = args.spec or SPEC_PATH
        try:
            # Tolerant: dropping entries for retired flags is what sync is FOR,
            # so it must be able to read a spec that still names them.
            current = load_spec(spec_path, tolerant=True)
        except SpecError as exc:
            if spec_path.exists():
                print(f"spec unreadable, refusing to overwrite: {exc}", file=sys.stderr)
                return 2
            current = None
        synced, added, removed = sync_spec(current)
        synced.path = spec_path
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = render_spec(synced)
        changed = (not spec_path.exists()) or spec_path.read_text() != rendered
        if changed:
            spec_path.write_text(rendered)
        print(f"{'wrote' if changed else 'unchanged'} {spec_path}")
        if added:
            print(f"  added   ({len(added)}): {', '.join(added)}")
        if removed:
            print(f"  removed ({len(removed)}): {', '.join(removed)}")
        return 0

    if args.show:
        print(_show_table())
        return 0

    try:
        spec = load_spec(args.spec)
    except SpecError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2

    drifts = sorted(
        (d.as_dict() for d in compute_drift(spec, args.live)),
        key=lambda d: (_KIND_ORDER.get(d["kind"], 99), d["flag"]),
    )
    missing, unknown = spec_coverage(spec)
    unregistered = sorted(referenced_flag_names() - set(registry_flag_names()))

    payload = {
        "spec_path": str(spec.path or SPEC_PATH),
        "live_path": str(args.live or _live_path()),
        "registry_flag_count": len(registry_flag_names()),
        "live_override_count": len(live_records(args.live)),
        "coverage": {
            "missing_from_spec": missing,
            "unknown_in_spec": unknown,
            "read_in_code_but_unregistered": unregistered,
        },
        "drift": drifts,
        "autopilot_state": autopilot_state_drift(spec),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_report_text(payload, verbose=args.verbose))

    if args.strict:
        blocking = [d for d in drifts if d["kind"] in BLOCKING_KINDS]
        if blocking or missing or unknown or unregistered:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
