"""EV-15e deterministic order plan and offline distribution-level pre-aggregation.

Input distributions must already satisfy the producer's native/tokenizer/template
contract. This consumer does not establish that contract, call a model, grade a
measurement, or set a decision threshold. EV-15c/d remain separate prerequisites
for native use. Both recorded presentation orders are required before any readout.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.typed_decisions import distributional_readout as readouts
from src.typed_decisions.score_token_head import (
    score_token_head_from_top_logprobs,
    validate_score_token_ids,
)

METHODS = {
    name: getattr(readouts, name)
    for name in (
        "mode",
        "mean",
        "rounded_mean",
        "median",
        "first_percentile",
        "risk_averse_mean",
        "quantile_integral",
        "probability_of_superiority",
    )
}
_IDENTITY_KEYS = (
    "model_id",
    "scorer_id",
    "item_id",
    "context_sha256",
    "left",
    "right",
)
_LEGACY_CAPTURE_KEYS = {
    "request_id",
    "capture_id",
    *_IDENTITY_KEYS,
    "probabilities_left",
    "probabilities_right",
}
_SCORE_HEAD_CAPTURE_KEYS = {
    "request_id",
    "capture_id",
    *_IDENTITY_KEYS,
    "score_head_left",
    "score_head_right",
}
_SCORE_HEAD_KEYS = {"entries"}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value: Any, name: str) -> str:
    _text(value, name)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _keys(value: Any, keys: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} requires exactly {sorted(keys)}")


def _load_json(path: Path) -> Any:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique_object)


def make_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    """Declare every unordered candidate pair in both actual display orders.

    IDs denote pinned model/scorer versions, not display labels. Answer and task
    hashes bind the payloads a native producer must resolve. Request IDs bind all
    those identities plus the display order. No native payload is fabricated.
    """
    schema = manifest.get("schema") if isinstance(manifest, dict) else None
    if schema == "ev15-comparison-manifest.v1":
        _keys(manifest, {"schema", "model_id", "scorer_id", "items"}, "manifest")
        version = 1
        score_token_ids = None
        floor = None
    elif schema == "ev15-comparison-manifest.v2":
        _keys(
            manifest,
            {
                "schema",
                "model_id",
                "scorer_id",
                "score_token_ids",
                "minimum_retained_mass",
                "items",
            },
            "manifest",
        )
        version = 2
        score_token_ids = validate_score_token_ids(manifest["score_token_ids"])
        if len(score_token_ids) != 9:
            raise ValueError("score_token_ids must bind the ordered K=9 support")
        floor = manifest["minimum_retained_mass"]
        if isinstance(floor, bool) or not isinstance(floor, (int, float)):
            raise TypeError("minimum_retained_mass must be numeric and not bool")
        try:
            floor = float(floor)
        except OverflowError as exc:
            raise ValueError("minimum_retained_mass must be finite and in [0, 1]") from exc
        if not np.isfinite(floor) or floor < 0.0 or floor > 1.0:
            raise ValueError("minimum_retained_mass must be finite and in [0, 1]")
    else:
        raise ValueError("unsupported manifest schema")
    for key in ("model_id", "scorer_id"):
        _text(manifest[key], key)
    if not isinstance(manifest["items"], list) or not manifest["items"]:
        raise ValueError("items must be a nonempty list")
    seen_items: set[str] = set()
    orders = []
    for item in manifest["items"]:
        _keys(item, {"item_id", "context_sha256", "candidates"}, "item")
        item_id = _text(item["item_id"], "item_id")
        if item_id in seen_items:
            raise ValueError(f"duplicate item_id: {item_id}")
        seen_items.add(item_id)
        _sha(item["context_sha256"], "context_sha256")
        candidates = item["candidates"]
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError("at least two candidates required per item")
        seen_candidates: set[str] = set()
        for candidate in candidates:
            _keys(candidate, {"candidate_id", "answer_sha256"}, "candidate")
            candidate_id = _text(candidate["candidate_id"], "candidate_id")
            _sha(candidate["answer_sha256"], "answer_sha256")
            if candidate_id in seen_candidates:
                raise ValueError(f"duplicate candidate_id: {candidate_id}")
            seen_candidates.add(candidate_id)
        for first, second in combinations(candidates, 2):
            for left, right in ((first, second), (second, first)):
                order = {
                    "model_id": manifest["model_id"],
                    "scorer_id": manifest["scorer_id"],
                    "item_id": item_id,
                    "context_sha256": item["context_sha256"],
                    "left": dict(left),
                    "right": dict(right),
                }
                request_identity = dict(order)
                if version == 2:
                    request_identity.update(
                        score_token_ids=list(score_token_ids),
                        minimum_retained_mass=floor,
                    )
                orders.append({"request_id": _digest(request_identity), **order})
    # Round-trip makes the returned manifest independent of later caller mutation.
    plan = {
        "schema": f"ev15-paired-order-plan.v{version}",
        "manifest": json.loads(json.dumps(manifest)),
        "score_support": list(range(1, 10)),
        "orders": orders,
        "native_validation": "not established by this offline plan",
    }
    if version == 2:
        plan["score_token_ids"] = list(score_token_ids)
        plan["minimum_retained_mass"] = floor
    return plan


def _distribution(value: Any) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != 9
        or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value)
    ):
        raise ValueError("each distribution requires nine numeric probabilities")
    return np.asarray(value, dtype=np.float64)


def paired_readout(
    plan: dict[str, Any],
    captures: list[dict[str, Any]],
    *,
    method: str,
) -> dict[str, Any]:
    """Align distributions by candidate identity, average, then read out once.

    Never infer a reverse order, substitute an opposite preference, or average
    the nonlinear readout results. All plan/capture validation precedes readout.
    """
    if not isinstance(plan, dict) or "manifest" not in plan:
        raise ValueError("invalid plan")
    if plan != make_plan(plan["manifest"]):
        raise ValueError("plan differs from its deterministic manifest")
    if method not in METHODS:
        raise ValueError("unknown readout method")
    if not isinstance(captures, list):
        raise ValueError("captures must be a list")
    expected = {r["request_id"]: r for r in plan["orders"]}
    received: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    capture_ids: set[str] = set()
    capture_mode: str | None = None
    score_head_audit: list[dict[str, Any]] = []
    score_head_refusals: list[dict[str, Any]] = []
    for capture in captures:
        if not isinstance(capture, dict):
            raise ValueError("capture must be a mapping")
        keys = set(capture)
        if keys == _LEGACY_CAPTURE_KEYS:
            mode = "legacy_probability_vectors"
        elif keys == _SCORE_HEAD_CAPTURE_KEYS:
            mode = "per_call_score_heads"
        else:
            raise ValueError("capture must contain exactly one supported capture schema")
        if plan["schema"] == "ev15-paired-order-plan.v1" and mode != "legacy_probability_vectors":
            raise ValueError("raw score heads require an ev15-comparison-manifest.v2 plan")
        if plan["schema"] == "ev15-paired-order-plan.v2" and mode != "per_call_score_heads":
            raise ValueError("v2 plans require per-call score-head captures")
        if capture_mode is not None and capture_mode != mode:
            raise ValueError("legacy vectors and per-call score heads cannot be mixed")
        capture_mode = mode
        request_id = _text(capture["request_id"], "request_id")
        if request_id not in expected or request_id in received:
            raise ValueError("unexpected or duplicate presentation order")
        if any(capture[k] != expected[request_id][k] for k in _IDENTITY_KEYS):
            raise ValueError("capture identity differs from planned presentation")
        capture_id = _text(capture["capture_id"], "capture_id")
        if capture_id in capture_ids:
            raise ValueError("a capture cannot stand in for two presentation orders")
        capture_ids.add(capture_id)
        if mode == "legacy_probability_vectors":
            left = _distribution(capture["probabilities_left"])
            right = _distribution(capture["probabilities_right"])
            # This path remains a math-only fixture API. It carries no native
            # head or retained-mass warrant and is labeled as such in output.
            readouts.probability_of_superiority(left, right)
        else:
            parsed = []
            for side in ("left", "right"):
                raw_head = capture[f"score_head_{side}"]
                _keys(raw_head, _SCORE_HEAD_KEYS, f"score_head_{side}")
                head = score_token_head_from_top_logprobs(
                    raw_head["entries"],
                    plan["score_token_ids"],
                    minimum_retained_mass=plan["minimum_retained_mass"],
                )
                audit = {
                    "request_id": request_id,
                    "candidate_id": expected[request_id][side]["candidate_id"],
                    "side": side,
                    **head.to_dict(),
                }
                score_head_audit.append(audit)
                if head.status == "refused":
                    score_head_refusals.append(
                        {
                            "request_id": request_id,
                            "candidate_id": audit["candidate_id"],
                            "side": side,
                            "reason": head.refusal_reason,
                        }
                    )
                parsed.append(head)
            left = np.asarray(parsed[0].probabilities or (0.0,) * 9, dtype=np.float64)
            right = np.asarray(parsed[1].probabilities or (0.0,) * 9, dtype=np.float64)
            # Refused calls retain their per-call audit above and cannot enter
            # a readout. Accepted distributions were normalized only after
            # each call independently passed its retained-mass floor.
            if not score_head_refusals:
                readouts.probability_of_superiority(left, right)
        received[request_id] = left, right
    if set(received) != set(expected):
        raise ValueError("missing planned presentation orders")
    if score_head_refusals:
        return {
            "schema": "ev15-paired-distributional-readout.v2",
            "status": "refused",
            "method": method,
            "plan_sha256": _digest(plan),
            "captures_sha256": _digest(captures),
            "model_id": plan["manifest"]["model_id"],
            "scorer_id": plan["manifest"]["scorer_id"],
            "refusals": score_head_refusals,
            "score_head_captures": score_head_audit,
            "comparisons": [],
            "capture_mode": capture_mode,
            "native_validation": "not established by this offline consumer",
            "grade": None,
        }
    results = []
    for forward, reverse in zip(plan["orders"][::2], plan["orders"][1::2], strict=True):
        left_ab, right_ab = received[forward["request_id"]]
        left_ba, right_ba = received[reverse["request_id"]]
        first = (left_ab + right_ba) / 2.0
        second = (right_ab + left_ba) / 2.0
        results.append(
            {
                "item_id": forward["item_id"],
                "first": forward["left"],
                "second": forward["right"],
                "request_ids": [forward["request_id"], reverse["request_id"]],
                "first_distribution": first.tolist(),
                "second_distribution": second.tolist(),
                "readout": float(METHODS[method](first, second)),
            }
        )
    result = {
        "schema": f"ev15-paired-distributional-readout.v{2 if capture_mode == 'per_call_score_heads' else 1}",
        "method": method,
        "plan_sha256": _digest(plan),
        "captures_sha256": _digest(captures),
        "model_id": plan["manifest"]["model_id"],
        "scorer_id": plan["manifest"]["scorer_id"],
        "comparisons": results,
        "native_validation": "not established by this offline consumer",
        "grade": None,
    }
    if capture_mode == "per_call_score_heads":
        result.update(
            status="computed",
            capture_mode=capture_mode,
            score_head_captures=score_head_audit,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="declare required AB and BA requests")
    planning.add_argument("manifest", type=Path)
    planning.add_argument("--output", required=True, type=Path)
    scoring = commands.add_parser("readout", help="consume every planned order before readout")
    scoring.add_argument("plan", type=Path)
    scoring.add_argument("captures", type=Path)
    scoring.add_argument("--method", choices=METHODS, required=True)
    scoring.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = make_plan(_load_json(args.manifest))
        else:
            result = paired_readout(
                _load_json(args.plan),
                _load_json(args.captures),
                method=args.method,
            )
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        # Refuse unrelated outputs; validate and serialize before exclusive creation.
        with args.output.open("x") as output:
            output.write(payload)
        if result.get("status") == "refused":
            return 2
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
