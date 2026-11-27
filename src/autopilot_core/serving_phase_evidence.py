"""Read serving telemetry for phase reports and the advisory shadow planner."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.backends.serving_calls import SCHEMA as CALL_SCHEMA
from src.backends.serving_calls import record_digest
from src.backends.serving_phases import NATIVE_COUNTERS, SCHEMA, nonnegative, phase_breakdown

REPORT_SCHEMA = "epyc.autopilot.serving_phase_report.v1"
DECISION_SCHEMA = "epyc.autopilot.serving_phase_shadow.v1"


def serving_log_shards(log: Path) -> list[Path]:
    """The serving log and its rotated shards, oldest first."""
    rotated = []
    index = 1
    while log.with_name(f"{log.name}.{index}").is_file():
        rotated.append(log.with_name(f"{log.name}.{index}"))
        index += 1
    return list(reversed(rotated)) + ([log] if log.is_file() else [])


def _object(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _epoch(value: Any) -> float | None:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed.timestamp() if parsed.tzinfo else None
        except (ValueError, OverflowError, OSError):
            return None
    return nonnegative(value)


def _interval(start: Any, end: Any) -> tuple[float, float] | None:
    lo, hi = _epoch(start), _epoch(end)
    if lo is not None and hi is not None and hi > lo:
        return lo, hi
    return None


def _union_seconds(intervals: list[tuple[float, float]]) -> float | None:
    if not intervals:
        return None
    ordered = sorted(intervals)
    start, end = ordered[0]
    duration = 0.0
    for lo, hi in ordered[1:]:
        if lo > end:
            duration += end - start
            start = lo
        end = max(end, hi)
    return round(duration + end - start, 6)


def _native_interval(
    phase: dict, name: str, request_window: tuple[float, float] | None
) -> tuple[float, float] | None:
    raw = _object(_object(phase.get("intervals")).get(name))
    if raw.get("source") != "observed_server_phase_events" or request_window is None:
        return None
    window = _interval(raw.get("start_epoch_s"), raw.get("end_epoch_s"))
    duration = phase.get(f"{name}_ms")
    if window is None or duration is None:
        return None
    lo, hi = window
    request_start, request_end = request_window
    if lo >= request_start and hi <= request_end and abs((hi - lo) * 1000 - duration) <= 2:
        return window
    return None


def _classification(record: dict, phase: dict, trusted: bool) -> tuple[str, str]:
    if record.get("phase_breakdown") is None:
        return "insufficient_evidence", "legacy_without_phase_breakdown"
    if phase.get("schema") != SCHEMA:
        return "insufficient_evidence", "unknown_phase_schema"
    if phase.get("timing_source") == "absent":
        return "insufficient_evidence", "native_counters_absent"
    if phase.get("timing_source") != "server_native_counters":
        return "insufficient_evidence", "unknown_timing_source"
    if not trusted:
        return "insufficient_evidence", "unverified_record_digest_or_identity"
    if record.get("outcome") != "ok":
        return "insufficient_evidence", "request_did_not_complete"
    wall, prefill, decode = (phase[key] for key in ("client_wall_ms", "prefill_ms", "decode_ms"))
    fresh, cached, generated = (
        phase[key] for key in ("fresh_prompt_tokens", "cached_prompt_tokens", "generated_tokens")
    )
    if None in (wall, prefill, decode, fresh, cached, generated) or not wall or not generated:
        return "insufficient_evidence", "missing_or_zero_required_counters"
    if prefill + decode > wall + 2:
        return "insufficient_evidence", "native_durations_exceed_client_wall"
    # These cutoffs flag where to investigate. They are descriptive fractions of
    # this request's measured wall, with no claim about what an intervention saves.
    if prefill > decode and prefill >= max(wall * 0.5, 1000) and fresh > cached:
        return "fresh_prefill_dominated", "fresh_tokens_dominate_and_prefill_uses_half_of_wall"
    if decode > prefill and decode >= max(wall * 0.5, 1000):
        return "decode_dominated", "decode_uses_half_of_wall"
    return "insufficient_evidence", "no_supported_dominant_phase"


def _request(record: dict) -> dict:
    raw = _object(record.get("phase_breakdown"))
    caller = _object(record.get("caller"))
    server = _object(record.get("server"))
    provenance = _object(record.get("provenance"))
    source = _object(raw.get("source_receipt"))
    legacy_counters = {}
    if not raw and record.get("timings_source") == "server":
        legacy_counters = phase_breakdown(
            _object(record.get("timings")),
            trial_id=None,
            batch_id=None,
            request_id=None,
            stream_id="",
            streamed=None,
            wall_ms=record.get("wall_ms"),
            delivered_first_ms=None,
        )
    phase = {
        key: nonnegative((raw or legacy_counters).get(key))
        for key in (*NATIVE_COUNTERS, "client_wall_ms", "delivered_first_ms")
    }
    for key in (name for name in NATIVE_COUNTERS if name.endswith("_tokens")):
        if type(phase[key]) is not int:
            phase[key] = None
    phase.update({key: raw.get(key) for key in ("schema", "timing_source", "intervals")})
    request_id = raw.get("request_id")
    stream_id = raw.get("stream_id")
    run_id = source.get("capture_id") or provenance.get("host")
    server_id = server.get("base_url") or source.get("server_id")
    window = _interval(record.get("ts_start"), record.get("ts_end"))
    digest = record.get("record_sha256")
    trusted = (
        record.get("schema") == CALL_SCHEMA
        and raw.get("schema") == SCHEMA
        and isinstance(digest, str)
        and all(isinstance(k, str) and k for k in (request_id, stream_id))
        and stream_id == record.get("record_id")
        and digest == record_digest(record)
    )
    evidence, reason = _classification(record, phase, trusted)
    native_decode_rate = None
    if trusted and phase["decode_ms"] and phase["generated_tokens"] is not None:
        native_decode_rate = phase["generated_tokens"] * 1000 / phase["decode_ms"]
    return {
        "trial_id": raw.get("trial_id") if raw else caller.get("trial_id"),
        "batch_id": raw.get("batch_id") if raw else caller.get("batch_id"),
        "request_id": request_id or caller.get("request_id"),
        "stream_id": stream_id or record.get("record_id"),
        "streamed": raw.get("streamed"),
        "timing_source": raw.get("timing_source")
        or ("legacy_server_timings" if legacy_counters else None),
        "first_delta_scope": raw.get("first_delta_scope"),
        "wall_scope": raw.get("wall_scope"),
        "record_sha256": digest,
        "source_receipt": source or None,
        "record_digest_valid": bool(trusted),
        "status": record.get("outcome"),
        "classification": evidence,
        "reason": reason,
        **{key: phase[key] for key in (*NATIVE_COUNTERS, "client_wall_ms", "delivered_first_ms")},
        "native_decode_tps": native_decode_rate,
        "request_window": window,
        "server_scope": [run_id, server_id] if run_id and server_id else None,
        "prefill_window": _native_interval(phase, "prefill", window) if trusted else None,
        "decode_window": _native_interval(phase, "decode", window) if trusted else None,
        "replay": _object(raw.get("replay")) if trusted else {},
        "served_slot": _object(record.get("timings")).get("id_slot"),
    }


def _rollup(requests: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in requests:
        scope = row["server_scope"]
        if row["record_digest_valid"] and scope and all(isinstance(part, str) for part in scope):
            groups[tuple(scope)].append(row)
    output = []
    for scope, rows in sorted(groups.items()):
        wall = [row["request_window"] for row in rows if row["request_window"] is not None]
        decode_rows = [
            r
            for r in rows
            if r["decode_window"] is not None
            and r["generated_tokens"] is not None
            and r["status"] == "ok"
        ]
        decode_union = _union_seconds([r["decode_window"] for r in decode_rows])
        native_decode = [
            r
            for r in rows
            if r["decode_ms"] and r["generated_tokens"] is not None and r["status"] == "ok"
        ]
        native_ms = sum(r["decode_ms"] for r in native_decode)
        output.append(
            {
                "server_scope": list(scope),
                "requests": len(rows),
                "request_windows": len(wall),
                "observed_request_wall_union_s": _union_seconds(wall),
                "prefill_windows": sum(r["prefill_window"] is not None for r in rows),
                "observed_prefill_union_s": _union_seconds(
                    [r["prefill_window"] for r in rows if r["prefill_window"] is not None]
                ),
                "decode_windows": len(decode_rows),
                "observed_decode_union_s": decode_union,
                "observed_concurrent_decode_tps": (
                    sum(r["generated_tokens"] for r in decode_rows) / decode_union
                    if decode_union
                    else None
                ),
                "request_native_decode_coverage": len(native_decode),
                "request_native_decode_sum_s": native_ms / 1000 if native_decode else None,
                "request_native_decode_tps": (
                    sum(r["generated_tokens"] for r in native_decode) * 1000 / native_ms
                    if native_ms
                    else None
                ),
            }
        )
    return output


def phase_report(paths: list[Path], *, batch_ids: set[str] | None = None) -> dict:
    """Read bound source rows and deduplicate repeated receipts across shards."""
    rows: dict[str, dict] = {}
    receipt_conflicts = set()
    unparsed = repeated = 0
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("expected object")
                except ValueError:
                    unparsed += 1
                    continue
                caller = _object(record.get("caller"))
                phase = _object(record.get("phase_breakdown"))
                batch_id = phase.get("batch_id") or caller.get("batch_id")
                if batch_ids is not None and (
                    not isinstance(batch_id, str) or batch_id not in batch_ids
                ):
                    continue
                row = _request(record)
                identity = row["stream_id"]
                key = (
                    identity if isinstance(identity, str) and identity else f"{path}:{line_number}"
                )
                if key in rows:
                    if rows[key] == row:
                        repeated += 1
                    else:
                        receipt_conflicts.add(key)
                    continue
                rows[key] = row
    requests = [row for key, row in rows.items() if key not in receipt_conflicts]
    return {
        "schema": REPORT_SCHEMA,
        "files": [str(path) for path in paths],
        "unparsed_lines": unparsed,
        "duplicate_rows": repeated,
        "conflicting_stream_ids": sorted(receipt_conflicts),
        "counts": dict(sorted(Counter(r["classification"] for r in requests).items())),
        "server_aggregates": _rollup(requests),
        "requests": requests,
    }


def _has_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _overlaps(left: tuple | None, right: tuple | None) -> bool:
    return bool(left and right and max(left[0], right[0]) < min(left[1], right[1]))


def _matched_decode_evidence(rows: list[dict]) -> dict[str, list[dict]]:
    """Check measured pair receipts. Isolated arms require an attested full capture."""
    pairs = defaultdict(list)
    for row in rows:
        controls = row["replay"]
        pair_id = controls.get("pair_id")
        if isinstance(pair_id, str) and pair_id and row["record_digest_valid"]:
            pairs[pair_id].append(row)
    repeated = defaultdict(list)
    for pair_id, legs in pairs.items():
        if len(legs) != 2:
            continue
        ordered = {
            leg["replay"].get("arm"): leg
            for leg in legs
            if leg["replay"].get("arm") in ("loaded", "isolated", "isolated_reference")
        }
        if len(ordered) != 2 or "isolated" not in ordered:
            continue
        isolated = ordered["isolated"]
        family = "admission_batch_scheduling" if "loaded" in ordered else "kernel_recipe"
        fast, slow = (
            (isolated, ordered["loaded"])
            if "loaded" in ordered
            else (ordered["isolated_reference"], isolated)
        )
        controls = (fast["replay"], slow["replay"])
        shared = ("prompt_sha256", "prefix_state_sha256", "serving_family_sha256")
        if any(
            not _has_sha256(controls[0].get(k)) or controls[0].get(k) != controls[1].get(k)
            for k in shared
        ):
            continue
        recipes = tuple(k.get("recipe_sha256") for k in controls)
        if not all(_has_sha256(k) for k in recipes) or (recipes[0] == recipes[1]) != (
            family == "admission_batch_scheduling"
        ):
            continue
        if family == "admission_batch_scheduling" and fast["server_scope"] != slow["server_scope"]:
            continue
        if fast["served_slot"] != slow["served_slot"]:
            continue
        counters = ("fresh_prompt_tokens", "cached_prompt_tokens", "generated_tokens")
        if any(fast[k] is None or fast[k] != slow[k] for k in counters):
            continue
        if slow["classification"] != "decode_dominated" or not all(
            row["status"] == "ok"
            and row["decode_window"]
            and row["native_decode_tps"]
            and row["server_scope"]
            and row["served_slot"] is not None
            and _has_sha256(row["replay"].get("capture_receipt_sha256"))
            and row["replay"].get("capture_complete") is True
            for row in (fast, slow)
        ):
            continue
        if slow["native_decode_tps"] > 0.85 * fast["native_decode_tps"]:
            continue
        peer_prefill = []
        arms_consistent = True
        for row in (fast, slow):
            peers = [
                p
                for p in rows
                if p["record_digest_valid"]
                and p["stream_id"] != row["stream_id"]
                and p["server_scope"] == row["server_scope"]
                and _overlaps(row["decode_window"], p["request_window"])
            ]
            if row is slow and family == "admission_batch_scheduling":
                peer_prefill = [
                    p
                    for p in peers
                    if p["served_slot"] is not None
                    and p["served_slot"] != row["served_slot"]
                    and _overlaps(row["decode_window"], p["prefill_window"])
                ]
                arms_consistent &= bool(peer_prefill)
            else:
                arms_consistent &= not peers
        if not arms_consistent:
            continue
        key = (family, *[controls[0][k] for k in shared], *recipes)
        repeated[key].append(
            {
                "pair_id": pair_id,
                "matched_controls": {k: controls[0][k] for k in shared},
                "recipe_sha256": list(recipes),
                "slow_native_decode_tps": slow["native_decode_tps"],
                "reference_native_decode_tps": fast["native_decode_tps"],
                "records": [_receipt(row) for row in (slow, fast, *peer_prefill)],
                "capture_receipt_sha256": [k["capture_receipt_sha256"] for k in controls],
            }
        )
    return {key[0]: pairs for key, pairs in repeated.items() if len(pairs) >= 2}


def _receipt(row: dict) -> dict:
    return {
        key: row[key]
        for key in (
            "trial_id",
            "batch_id",
            "request_id",
            "stream_id",
            "record_sha256",
            "source_receipt",
            "classification",
            "reason",
            "fresh_prompt_tokens",
            "cached_prompt_tokens",
            "prefill_ms",
            "decode_ms",
            "native_decode_tps",
        )
    }


def plan_serving_phase(report: dict, *, quality_goodput: dict | None = None) -> dict:
    """Choose an investigation to test with equal-quality accepted tasks per hour."""
    rows = report["requests"]
    prefill = sorted(
        (r for r in rows if r["classification"] == "fresh_prefill_dominated"),
        key=lambda r: r["prefill_ms"],
        reverse=True,
    )
    decode = _matched_decode_evidence(rows)
    alternatives = set(decode) | ({"prefix_retention"} if prefill else set())
    reasons = []
    if report["unparsed_lines"] or report["conflicting_stream_ids"]:
        reasons.append("incomplete_or_conflicting_input")
        alternatives.clear()
    if len(alternatives) > 1:
        reasons.append("multiple_supported_investigations")
    direction = next(iter(alternatives)) if len(alternatives) == 1 else "insufficient_evidence"
    if not prefill and not decode:
        reasons.append("no_fresh_prefill_case_or_repeated_matched_decode_loss")
    reasons_by_count = dict(sorted(Counter(r["reason"] for r in rows).items()))
    return {
        "schema": DECISION_SCHEMA,
        "type": "serving_phase_shadow_decision",
        "decision": direction,
        "mode": "shadow",
        "execution_authority": False,
        "uncertainty": reasons,
        "validation_pending": ["equal_quality_goodput_effect", "paid_effort"],
        "evidence_counts": report["counts"],
        "reason_counts": reasons_by_count,
        "evidence_receipts": [_receipt(r) for r in prefill[:3]] or [_receipt(r) for r in rows[:3]],
        "matched_decode_evidence": decode,
        "evaluation_objective": "accepted_tasks_per_hour_at_equal_quality",
        "quality_qualified_goodput": quality_goodput,
        "paid_effort": None,
    }
