"""Default-off native crossover donor diagnostics, independent of acceptance policy."""
from __future__ import annotations
import hashlib
import logging
import os
from typing import Any
from experiment_journal import build_error_scope, is_learning_eligible
from mutation_pair_features import compare_mutation_features

log = logging.getLogger(__name__)
FLAG = "AUTOPILOT_CROSSOVER_COMPLEMENTARITY_DIAGNOSTICS"
SCHEMA = "epyc.autopilot.crossover_features.v2"


def enabled() -> bool:
    return os.environ.get(FLAG) == "1"


def _current_native_scope(row: Any) -> dict[str, Any] | None:
    """Validate the producer's recorded scope against this row's native fields."""
    native = getattr(row, "error_scope", None)
    pin = getattr(row, "baseline_pin", None)
    details = row.eval_details if isinstance(getattr(row, "eval_details", None), dict) else {}
    comparability = getattr(row, "comparability", None)
    if type(native) is not dict or type(pin) is not dict:
        return None
    revision = pin.get("baseline_revision")
    if type(revision) is not int:
        return None
    core_id = native.get("core_id")
    expected = build_error_scope(
        core_id=core_id, baseline_pin=pin,
        regime_digest=details.get("infra_regime_digest"), comparability=comparability)
    # The existing journal writer records error_scope from these same native
    # fields. A missing or divergent source remains unknown; no field is inferred.
    if expected is None or native != expected:
        return None
    return {**native, "baseline_revision": revision}


def _metadata(row: Any) -> dict[str, Any] | None:
    if (type(getattr(row, "trial_id", None)) is not int
            or not is_learning_eligible(row) or row.pareto_status != "frontier"):
        return None
    details = row.eval_details if isinstance(row.eval_details, dict) else {}
    value = details.get("crossover_features")
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        return None
    if set(value) != {
        "schema", "trial_id", "scope", "features", "signature",
        "delta_reference_trial_id", "feature_semantics", "sentinel_outcome_source",
    }:
        return None
    scope = value.get("scope")
    keys = {"schema_version", "comparability", "core_id", "infra_regime_digest",
            "eval_quality_era", "autopilot_speed_era", "baseline_revision"}
    if type(scope) is not dict or set(scope) != keys:
        return None
    if (type(scope["schema_version"]) is not int or scope["schema_version"] != 1
            or scope["comparability"] != "COMPARABLE"
            or type(scope["baseline_revision"]) is not int):
        return None
    if not all(type(scope[k]) is str and scope[k].strip() for k in (
            "core_id", "infra_regime_digest", "eval_quality_era", "autopilot_speed_era")):
        return None
    if scope != _current_native_scope(row):
        return None
    if type(value.get("feature_semantics")) is not str:
        return None
    if type(value.get("trial_id")) is not int or value["trial_id"] != row.trial_id:
        return None
    sentinel_source = value.get("sentinel_outcome_source")
    if type(sentinel_source) is not str or sentinel_source not in {
        "question_results", "suite_quality_proxy",
    }:
        return None
    features = value.get("features")
    if type(features) is not dict or set(features) != {
        "version", "trial_id", "action_type", "subsystem", "files_touched",
        "prompt_sections_touched", "feature_flags", "behavior_signature_delta",
        "parent_trial", "pareto_status", "archive_member_id",
    }:
        return None
    from bsv_observe import BSV_CONFLICT_VERSION
    if features.get("version") != BSV_CONFLICT_VERSION:
        return None
    if (type(features.get("subsystem")) is not str or not features["subsystem"]
            or type(features.get("action_type")) is not str
            or features.get("pareto_status") != "frontier"):
        return None
    if (features.get("parent_trial") is not None
            and type(features.get("parent_trial")) is not int):
        return None
    if (features.get("archive_member_id") is not None
            and type(features.get("archive_member_id")) is not str):
        return None
    if type(features.get("trial_id")) is not int or features["trial_id"] != row.trial_id:
        return None
    for key in ("files_touched", "prompt_sections_touched"):
        if type(features.get(key)) is not list or not all(type(x) is str for x in features[key]):
            return None
    if (type(features.get("feature_flags")) is not dict
            or not all(type(key) is str for key in features["feature_flags"])):
        return None
    delta = features.get("behavior_signature_delta")
    if type(delta) is not dict or set(delta) != {
        "severity", "reasons", "signature_hash", "signature_confidence",
        "changed_fields", "improved_sentinels", "regressed_sentinels",
    }:
        return None
    if ((delta.get("severity") is not None
         and (type(delta.get("severity")) is not str
              or delta.get("severity") not in {"benign", "watch", "blocking"}))
            or type(delta.get("reasons")) is not list
            or not all(type(x) is str for x in delta["reasons"])
            or delta.get("signature_confidence") != "partial"
            or (delta.get("signature_hash") is not None
                and type(delta.get("signature_hash")) is not str)):
        return None
    for key in ("changed_fields", "improved_sentinels", "regressed_sentinels"):
        if type(delta.get(key)) is not list or not all(type(x) is str for x in delta[key]):
            return None
    reference = value.get("delta_reference_trial_id")
    if reference is None:
        if delta.get("severity") is not None:
            return None
    elif type(reference) is not int or delta.get("severity") not in {"benign", "watch", "blocking"}:
        return None
    signature = value.get("signature")
    if type(signature) is not dict or set(signature) != {
        "archive_member_id", "trial_id", "event_id", "sentinel_outcomes", "answer_hash",
        "route_path_hash", "tool_sequence_hash", "escalation_path_hash", "latency_bucket",
        "token_bucket", "signature_hash", "signature_confidence",
    }:
        return None
    if type(signature.get("trial_id")) is not int or signature["trial_id"] != row.trial_id:
        return None
    if (type(signature.get("archive_member_id")) is not str
            or type(signature.get("signature_hash")) is not str
            or not signature["signature_hash"]
            or signature.get("signature_confidence") != "partial"
            or type(signature.get("sentinel_outcomes")) is not dict
            or not all(type(k) is str and type(v) is str
                       and v in {"pass", "fail", "error", "skip", "pass_via_shortcut"}
                       for k, v in signature["sentinel_outcomes"].items())):
        return None
    if any(signature.get(key) is not None and type(signature.get(key)) is not str for key in (
            "answer_hash", "route_path_hash", "tool_sequence_hash", "escalation_path_hash",
            "latency_bucket", "token_bucket")):
        return None
    if signature.get("event_id") is not None and type(signature.get("event_id")) is not int:
        return None
    if delta.get("signature_hash") != signature.get("signature_hash"):
        return None
    return value


def capture_trial_features(row: Any, *, verdict: Any, action: dict[str, Any],
                           eval_result: Any, history: list[Any]) -> None:
    """Writer-time only: capture native declared features with a native observed anchor."""
    content = getattr(eval_result, "_crossover_donor_content", None)
    try:
        delattr(eval_result, "_crossover_donor_content")
    except AttributeError:
        pass
    except Exception as exc:
        log.warning("Crossover donor consume unavailable (%s)", type(exc).__name__)
        return
    if not enabled():
        return
    try:
        if not verdict.passed or not is_learning_eligible(row) or row.pareto_status != "frontier":
            return
        scope = _current_native_scope(row)
        if (scope is None or type(getattr(eval_result, "core_id", None)) is not str
                or eval_result.core_id != scope["core_id"]):
            return
        from bsv_observe import compute_bsv_observe_payload, build_mutation_dependency_entry
        # Determine which native sentinel source this evaluation actually supplied
        # before choosing an anchor. Question outcomes and suite proxies are distinct
        # observation classes even when their labels happen to coincide.
        payload = compute_bsv_observe_payload(
            eval_result, species_name=row.species, trial_id=row.trial_id,
            archive_member_id=f"trial:{row.trial_id}", incumbent_signature=None,
            incumbent_archive_member_id=None)
        sentinel_source = payload.get("sentinel_outcome_source")
        if sentinel_source not in {"question_results", "suite_quality_proxy"}:
            return
        # Earliest recorded same-scope, same-source anchor; never mix proxy and native question rows.
        candidates = [value for prior in history if (value := _metadata(prior)) is not None
                      and prior.trial_id < row.trial_id and value["scope"] == scope
                      and value["sentinel_outcome_source"] == sentinel_source]
        anchor = min(candidates, key=lambda value: value["trial_id"]) if candidates else None
        incumbent = anchor.get("signature") if anchor else None
        if anchor is not None:
            payload = compute_bsv_observe_payload(
                eval_result, species_name=row.species, trial_id=row.trial_id,
                archive_member_id=f"trial:{row.trial_id}", incumbent_signature=incumbent,
                incumbent_archive_member_id=f"trial:{anchor['trial_id']}")
        features = build_mutation_dependency_entry(
            trial_id=row.trial_id, action=action, parent_trial=row.parent_trial,
            bsv_payload=payload, incumbent_signature=incumbent, pareto_status=row.pareto_status)
        row.eval_details["crossover_features"] = {
            "schema": SCHEMA, "trial_id": row.trial_id, "scope": scope,
            "features": features, "signature": payload.get("signature"),
            "delta_reference_trial_id": anchor["trial_id"] if anchor else None,
            "sentinel_outcome_source": payload.get("sentinel_outcome_source"),
            "feature_semantics": "native_partial observations; sentinel source explicitly recorded",
        }
        if (os.environ.get("AUTOPILOT_CROSSOVER_DONOR_SELECTION") == "1"
                and action.get("type") == "prompt_mutation"
                and isinstance(content, tuple) and len(content) == 3
                and type(content[0]) is str and content[0] in features["files_touched"]
                and type(content[1]) is str and type(content[2]) is str
                and 0 < len(content[2].encode("utf-8")) <= 4096):
            data = content[2].encode("utf-8")
            digest = hashlib.sha256(data).hexdigest()
            served = row.eval_details.get("served_content") or {}
            if (isinstance(served, dict) and isinstance(served.get("files"), dict)
                    and served["files"].get(content[1]) == digest):
                row.eval_details["crossover_donor_content"] = {
                    "trial_id": row.trial_id, "target": content[0], "source": content[1],
                    "content": content[2], "sha256": digest,
                }
    except Exception as exc:
        log.warning("Crossover feature capture unavailable (%s)", type(exc).__name__)


def donor_pairs(history: list[Any], target_file: str, *, top_n: int = 2) -> list[dict[str, Any]]:
    """Rank observed complementary pairs, never invert the BSV conflict verdict."""
    recorded = [value for row in history if (value := _metadata(row)) is not None]
    candidates = [value for value in recorded
                  if type(value.get("delta_reference_trial_id")) is int
                  and target_file in value["features"]["files_touched"]
                  and any(anchor["trial_id"] == value["delta_reference_trial_id"]
                          and anchor["trial_id"] < value["trial_id"]
                          and anchor["scope"] == value["scope"]
                          and anchor["sentinel_outcome_source"] == value["sentinel_outcome_source"]
                          and anchor.get("delta_reference_trial_id") is None
                          for anchor in recorded)][-30:]
    pairs = []
    for i, left in enumerate(candidates):
        for right in candidates[i + 1:]:
            if left["trial_id"] == right["trial_id"] or left["scope"] != right["scope"]:
                continue
            if left["sentinel_outcome_source"] != right["sentinel_outcome_source"]:
                continue
            if left["delta_reference_trial_id"] != right["delta_reference_trial_id"]:
                continue
            facts = compare_mutation_features(left["features"], right["features"])
            if (not facts["disjoint_improvements"] or facts["opposing_movement"]
                    or facts["has_regressions"] or facts["delta_blocking"]):
                continue
            pairs.append({"donor_trial_ids": [left["trial_id"], right["trial_id"]],
                          "scope": left["scope"], "features": facts,
                          "sentinel_outcome_source": left["sentinel_outcome_source"],
                          "policy": "advisory-disjoint-observed-improvements-v1"})
    pairs.sort(key=lambda pair: (len(pair["features"]["overlap_reasons"]), pair["donor_trial_ids"]))
    return pairs[:max(0, top_n)]


def select_donor_pairs(history: list[Any], target_file: str, source: str, *, top_n: int = 1) -> list[dict[str, Any]]:
    """Select only verifiable, comparable content, using the BSV-3 verdict."""
    from bsv_observe import _conflict_severity

    recorded = [(row, value) for row in history if (value := _metadata(row)) is not None]
    candidates = []
    for row, value in recorded:
        reference = value.get("delta_reference_trial_id")
        feature = value["features"]
        content = row.eval_details.get("crossover_donor_content")
        if (type(reference) is not int or target_file not in feature["files_touched"]
                or not feature["prompt_sections_touched"] or type(content) is not dict
                or set(content) != {"trial_id", "target", "source", "content", "sha256"}
                or type(content["trial_id"]) is not int or content["trial_id"] != row.trial_id
                or content["target"] != target_file or type(source) is not str or not source
                or content["source"] != source
                or type(content["content"]) is not str or type(content["sha256"]) is not str):
            continue
        raw = content["content"].encode("utf-8")
        served = row.eval_details.get("served_content")
        if (not raw or len(raw) > 4096 or hashlib.sha256(raw).hexdigest() != content["sha256"]
                or type(served) is not dict or type(served.get("files")) is not dict
                or served["files"].get(content["source"]) != content["sha256"]):
            continue
        delta = feature["behavior_signature_delta"]
        if not (delta["changed_fields"] or delta["improved_sentinels"] or delta["regressed_sentinels"]):
            continue
        if not any(anchor["trial_id"] == reference and anchor["trial_id"] < value["trial_id"]
                   and anchor["scope"] == value["scope"]
                   and anchor["sentinel_outcome_source"] == value["sentinel_outcome_source"]
                   and anchor["delta_reference_trial_id"] is None for _, anchor in recorded):
            continue
        candidates.append((value, content))
    # Match the advisory path's bounded pair-search window.
    candidates = candidates[-30:]
    ranked = []
    rank = {"none": 0, "watch": 1, "blocking": 2}
    for i, (left, left_content) in enumerate(candidates):
        for right, right_content in candidates[i + 1:]:
            if (left["trial_id"] == right["trial_id"] or left["scope"] != right["scope"]
                    or left["sentinel_outcome_source"] != right["sentinel_outcome_source"]
                    or left["delta_reference_trial_id"] != right["delta_reference_trial_id"]):
                continue
            severity, reasons = _conflict_severity(left["features"], right["features"])
            ranked.append((rank[severity], left["trial_id"], right["trial_id"], {
                "trial_ids": [left["trial_id"], right["trial_id"]], "severity": severity,
                "reasons": reasons, "source": source,
                "sha256": [left_content["sha256"], right_content["sha256"]],
                "contents": [left_content["content"], right_content["content"]],
            }))
    ranked.sort(key=lambda item: item[:3])
    return [item[3] for item in ranked[:max(0, top_n)]]


def crossover_context(journal: Any, target_file: str) -> str:
    if not enabled():
        return ""
    try:
        pairs = donor_pairs(journal.entries_with_supersessions(), target_file)
        if not pairs:
            return "## Crossover donor diagnostics\nUnknown: no eligible recorded comparable donor pair."
        lines = ["## Crossover donor diagnostics (native partial observations; advisory only; not ground truth)"]
        for pair in pairs:
            facts = pair["features"]
            source = pair["sentinel_outcome_source"]
            source_label = ("native per-question outcomes (partial)" if source == "question_results"
                            else "native suite-quality proxy outcomes (partial proxy)")
            improved_left = ", ".join(x[:64] for x in facts["new_improvements"][:4]) or "unknown"
            improved_right = ", ".join(x[:64] for x in facts["prior_improvements"][:4]) or "unknown"
            shared = "; ".join(x[:160] for x in facts["overlap_reasons"][:3]) or "none recorded"
            lines.append(
                f"- donor trials {pair['donor_trial_ids']}; source={source_label}; "
                f"recorded improvements [{improved_left}] / [{improved_right}]; "
                f"shared recorded features: {shared}; existing acceptance gates unchanged"
            )
        return "\n".join(lines)
    except Exception as exc:
        log.warning("Crossover donor context unavailable (%s)", type(exc).__name__)
        return ""
