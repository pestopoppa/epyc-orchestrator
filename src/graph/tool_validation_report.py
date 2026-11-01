"""Report-only projection of explicitly requested native trajectory observations."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from src.graph.trajectory_artifact import SCHEMA as TRAJECTORY_SCHEMA


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def build(
    arms: dict, files: list, outcomes: dict | None, comparisons: list | None = None
) -> dict[str, Any]:
    """Bind task observations to the exact trajectory bytes used by the sidecar.

    A registry validation is not semantic correctness, and registry invocations
    do not cover every model tool attempt. No repair is inferred from ordering.
    """
    rows = []
    sources = iter(files)
    for label, arm in arms.items():
        for task in arm["tasks"]:
            _, raw = next(sources)
            tid = task["task_id"]
            try:
                events = [
                    json.loads(
                        line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
                    )
                    for line in raw.decode("utf-8").splitlines()
                    if line.strip()
                ]
                if not all(isinstance(event, dict) for event in events):
                    events = []
            except (UnicodeDecodeError, ValueError):
                events = []
            observations = [event for event in events if event.get("event") == "tool_validation"]
            finals = [event for event in events if event.get("event") == "final"]
            observation = None
            if (
                len(observations) == 1
                and finals
                and all(
                    event.get("task_id") == tid and event.get("schema") == TRAJECTORY_SCHEMA
                    for event in events
                )
                and all(
                    type(event.get("seq")) is int and event["seq"] == index + 1
                    for index, event in enumerate(events)
                )
                and observations[0].get("task_id") == tid
                and events.index(observations[0]) > events.index(finals[-1])
            ):
                candidate = observations[0].get("observation")
                if (
                    isinstance(candidate, dict)
                    and candidate.get("scope") == "request_local_registry_invoke"
                ):
                    calls = candidate.get("calls")
                    attempted = candidate.get("registry_invocations")
                    omitted = candidate.get("unrecorded_invocations")
                    censored = candidate.get("censored")
                    if (
                        isinstance(calls, list)
                        and type(censored) is bool
                        and (
                            (attempted is None and omitted is None and censored)
                            or (
                                type(attempted) is int
                                and type(omitted) is int
                                and attempted >= len(calls)
                                and omitted == attempted - len(calls)
                                and censored is (omitted > 0)
                            )
                        )
                        and all(
                            isinstance(c, dict)
                            and isinstance(c.get("call_id"), str)
                            and re.fullmatch(r"[0-9a-f]{32}", c["call_id"])
                            and type(c.get("finished")) is bool
                            and (
                                c.get("argument_validation") is None
                                or type(c.get("argument_validation")) is bool
                            )
                            for c in calls
                        )
                        and len({c["call_id"] for c in calls}) == len(calls)
                    ):
                        observation = {
                            "scope": "request_local_registry_invoke",
                            "coverage_of_all_tool_attempts": "unknown",
                            "registry_invocations": attempted,
                            "unrecorded_invocations": omitted,
                            "censored": censored,
                            "calls": [
                                {
                                    "call_id": c["call_id"],
                                    "argument_validation": c.get("argument_validation"),
                                    "finished": c["finished"],
                                }
                                for c in calls
                            ],
                            "repair_links": None,
                        }
            raw_outcome = (
                outcomes.get(label, {}).get(tid)
                if isinstance(outcomes, dict) and isinstance(outcomes.get(label), dict)
                else None
            )
            graded = raw_outcome if type(raw_outcome) is bool else None
            usage = task.get("episode_usage") or {}
            total = usage.get("total_tokens") if usage.get("status") == "complete" else None
            if type(total) is not int or total < 0:
                total = None
            rows.append(
                {
                    "arm": label,
                    "task_id": tid,
                    "trajectory_sha256": hashlib.sha256(raw).hexdigest(),
                    "graded_solved": graded,
                    "graph_episode_tokens": total,
                    "registry_observation": observation,
                }
            )
    first = [
        row["registry_observation"]["calls"][0]
        for row in rows
        if row["registry_observation"] is not None
        and not row["registry_observation"]["censored"]
        and row["registry_observation"]["calls"]
    ]
    known_first = [
        call for call in first if call["finished"] and type(call["argument_validation"]) is bool
    ]
    failures = [row for row in rows if row["graded_solved"] is False]
    known_tokens = [
        row["graph_episode_tokens"] for row in failures if row["graph_episode_tokens"] is not None
    ]
    if comparisons is not None and not isinstance(comparisons, list):
        raise ValueError("comparisons must be a list of explicit arm pairs")
    pairs = []
    seen = set()
    for pair in comparisons or []:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or pair[0] == pair[1]
            or any(not isinstance(label, str) or label not in arms for label in pair)
        ):
            raise ValueError("comparison must name distinct declared baseline and candidate arms")
        baseline, candidate = pair
        if (baseline, candidate) in seen:
            raise ValueError("duplicate comparison pair")
        seen.add((baseline, candidate))
        left = {row["task_id"]: row["graded_solved"] for row in rows if row["arm"] == baseline}
        right = {row["task_id"]: row["graded_solved"] for row in rows if row["arm"] == candidate}
        changes = {"gain": 0, "giveback": 0, "unchanged": 0, "unknown": 0, "missing": 0}
        for tid in left.keys() | right.keys():
            if tid not in left or tid not in right:
                changes["missing"] += 1
            elif left[tid] is None or right[tid] is None:
                changes["unknown"] += 1
            elif left[tid] is False and right[tid] is True:
                changes["gain"] += 1
            elif left[tid] is True and right[tid] is False:
                changes["giveback"] += 1
            else:
                changes["unchanged"] += 1
        pairs.append({"baseline": baseline, "candidate": candidate, **changes})
    return {
        "schema": "epyc.registry_validation_observation.v1",
        "rows": rows,
        "all_tool_attempt_coverage": "unknown",
        "attempted_tasks": len(rows),
        "unresolved_grade_tasks": sum(row["graded_solved"] is None for row in rows),
        "first_registry_invocation": {
            "known_denominator": len(known_first),
            "unknown_validation": len(first) - len(known_first),
            "no_registry_invocation_recorded": sum(
                row["registry_observation"] is not None
                and not row["registry_observation"]["censored"]
                and not row["registry_observation"]["calls"]
                for row in rows
            ),
            "unknown_observation": sum(row["registry_observation"] is None for row in rows),
            "censored_observation": sum(
                row["registry_observation"] is not None and row["registry_observation"]["censored"]
                for row in rows
            ),
            "valid": sum(call["argument_validation"] is True for call in known_first),
            "fraction_valid": (
                sum(call["argument_validation"] is True for call in known_first) / len(known_first)
                if known_first
                else None
            ),
        },
        "failing_graph_episode_tokens": {
            "failing_tasks": len(failures),
            "known_token_tasks": len(known_tokens),
            "unknown_token_tasks": len(failures) - len(known_tokens),
            "known_token_subtotal": sum(known_tokens) if known_tokens else None,
        },
        "first_try_valid_call_percent": None,
        "repair_recovery_percent": None,
        "failed_verification": None,
        "no_valid_tool_call": None,
        "ran_out_of_time": None,
        "paired_comparisons": pairs if comparisons is not None else None,
    }
