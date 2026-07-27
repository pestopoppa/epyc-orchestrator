#!/usr/bin/env python3
"""Read-only semantic validator candidate for E8 quality protocol v5."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any


RUNNER_PATH = Path(__file__).with_name("run_e8_quality_baseline_v5.py")
EXPECTED_EVIDENCE_KEYS = {
    "schema",
    "eval_quality_era",
    "source_records",
    "replacement",
    "protocol_candidate",
    "runner",
    "run_seal_path",
    "generation_tail_contract",
}
EXPECTED_CHECKS = {
    "six_observations",
    "all_vectors_identical_per_tier",
    "post_e8_timestamps",
    "frozen_endpoints",
    "no_state_registry_lineup_mutation",
    "numeric_rerun_unchanged",
    "frozen_runtime_binding",
    "continuous_clean_monitor",
    "all_clean_repetitions",
    "v5_semantic_replay",
}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("e8_v5_validator_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import v5 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_artifact(root: Path, value: Any, label: str) -> Path:
    path = Path(str(value or "")).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes the sealed evidence directory")
    return path


def validate_tail_trace_replacement(
    *,
    original_trace_lines: list[bytes],
    final_trace_lines: list[bytes],
    retry_traces: dict[int, list[dict[str, Any]]],
    target_ordinals: set[int],
    scoring_questions: list[dict[str, Any]],
    expected_qids: list[str],
    tier: int,
    repetition: int,
) -> None:
    old_traces: dict[int, bytes] = {}
    final_traces: dict[int, bytes] = {}
    for label, lines, destination in (
        ("original", original_trace_lines, old_traces),
        ("final", final_trace_lines, final_traces),
    ):
        for line in lines:
            ordinal = int(json.loads(line)["fixed_vector_row"]["ordinal"])
            if ordinal in destination:
                raise ValueError(f"generation tail has duplicate {label} judge-trace identities")
            destination[ordinal] = line
    llm_target_ordinals = {
        ordinal
        for ordinal in target_ordinals
        if scoring_questions[ordinal].get("scoring_method") == "llm_judge"
    }
    if set(final_traces) != set(old_traces) | llm_target_ordinals:
        raise ValueError("generation tail changed judge-trace identities")
    for ordinal, old_trace_line in old_traces.items():
        if ordinal not in target_ordinals and final_traces.get(ordinal) != old_trace_line:
            raise ValueError("generation tail changed a non-target judge trace")
    for ordinal in target_ordinals:
        focused = retry_traces[ordinal]
        if ordinal in llm_target_ordinals:
            if len(focused) != 1:
                raise ValueError("LLM-judge generation tail lacks one focused trace")
            expected_trace = dict(focused[0])
            expected_trace["fixed_vector_row"] = {
                "tier": tier,
                "repetition": repetition,
                "ordinal": ordinal,
                "qid": expected_qids[ordinal],
            }
            expected_line = (json.dumps(expected_trace, sort_keys=True) + "\n").encode()
            if final_traces.get(ordinal) != expected_line:
                raise ValueError("final LLM-judge trace differs from focused retry")
        elif focused or final_traces.get(ordinal) != old_traces.get(ordinal):
            raise ValueError("non-judge generation tail changed judge-trace evidence")


def validate(
    evidence_path: Path,
    *,
    expected_runner_sha256: str,
    expected_base_runner_sha256: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_runner_sha256):
        raise ValueError("expected runner SHA-256 is malformed")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_base_runner_sha256):
        raise ValueError("expected base-runner SHA-256 is malformed")
    if sha256_path(RUNNER_PATH) != expected_runner_sha256:
        raise ValueError("v5 runner differs from the externally reviewed hash")
    runner = load_runner()
    if (
        sha256_path(runner.V4_PATH) != expected_base_runner_sha256
        or runner.GENERATION_TAIL_CONTRACT.get("v4_base_runner_sha256")
        != expected_base_runner_sha256
    ):
        raise ValueError("v4 base runner differs from the externally reviewed hash")
    evidence_path = evidence_path.resolve(strict=True)
    evidence_root = evidence_path.parent
    evidence = load_json(evidence_path, "evidence")
    if (
        set(evidence) != EXPECTED_EVIDENCE_KEYS
        or evidence.get("schema") != "epyc.e8_quality_baseline_evidence.v2"
        or evidence.get("eval_quality_era") != "E8"
    ):
        raise ValueError("evidence schema differs")
    if evidence.get("generation_tail_contract") != runner.GENERATION_TAIL_CONTRACT:
        raise ValueError("evidence generation-tail contract differs")
    runner_ref = evidence.get("runner")
    if runner_ref != {"path": str(RUNNER_PATH), "sha256": expected_runner_sha256}:
        raise ValueError("evidence runner binding differs")
    candidate_ref = evidence.get("protocol_candidate")
    if not isinstance(candidate_ref, dict):
        raise ValueError("protocol candidate reference is missing")
    candidate_path = resolve_artifact(
        evidence_root,
        candidate_ref.get("path"),
        "protocol candidate",
    )
    if candidate_ref.get("sha256") != sha256_path(candidate_path):
        raise ValueError("protocol candidate hash differs")
    proposal = load_json(candidate_path, "protocol candidate")
    protocol = proposal.get("protocol")
    if (
        proposal.get("schema") != runner.PROPOSAL_SCHEMA
        or not isinstance(protocol, dict)
        or protocol.get("protocol_id") != runner.PROTOCOL_ID
        or protocol.get("generation_tail_contract") != runner.GENERATION_TAIL_CONTRACT
    ):
        raise ValueError("v5 protocol candidate differs")
    report = load_json(evidence_path.parent / "runner_report.json", "runner report")
    postconditions = report.get("postconditions")
    if not isinstance(postconditions, dict):
        raise ValueError("runner report postconditions are missing")
    checks = postconditions.get("checks")
    samples = postconditions.get("watcher_samples")
    if (
        report.get("mode") != "executed"
        or report.get("protocol_id") != runner.PROTOCOL_ID
        or report.get("decision_grade") is not True
        or not isinstance(checks, dict)
        or set(checks) != EXPECTED_CHECKS
        or any(value is not True for value in checks.values())
        or not isinstance(samples, list)
        or len(samples) < 2
        or any(not isinstance(sample, dict) or sample.get("ok") is not True for sample in samples)
        or any("watcher_exception" in sample for sample in samples)
    ):
        raise ValueError("runner report is not a clean decision-grade v5 run")
    watcher_path = resolve_artifact(
        evidence_root,
        postconditions.get("watcher_path"),
        "runtime watcher ledger",
    )
    if (
        postconditions.get("watcher_sha256") != sha256_path(watcher_path)
        or runner.V4.load_jsonl(watcher_path) != samples
    ):
        raise ValueError("runtime watcher ledger differs from the runner report")
    try:
        watcher_started = [
            datetime.fromisoformat(str(sample["started_at"]).replace("Z", "+00:00"))
            for sample in samples
        ]
    except (KeyError, ValueError) as exc:
        raise ValueError("runtime watcher timestamps are invalid") from exc
    if any(
        (later - earlier).total_seconds() > 7.0
        for earlier, later in zip(watcher_started, watcher_started[1:])
    ):
        raise ValueError("runtime watcher has a sampling gap")
    details = report.get("observations")
    if not isinstance(details, dict) or set(details) != {"1", "2"}:
        raise ValueError("runner report does not contain both tiers")
    vectors = {
        tier: load_json(evidence_path.parent / f"question_vector.T{tier}.json", f"T{tier} vector")
        for tier in (1, 2)
    }
    scoring = {
        tier: load_json(
            evidence_path.parent / f"scoring_vector.T{tier}.json", f"T{tier} scoring vector"
        )
        for tier in (1, 2)
    }
    for tier, expected_n in ((1, 50), (2, 500)):
        vector_questions = vectors[tier].get("questions")
        scoring_questions = scoring[tier].get("questions")
        if (
            vectors[tier].get("schema") != "epyc.e8_quality_question_vector.v1"
            or vectors[tier].get("era") != "E8"
            or vectors[tier].get("tier") != tier
            or vectors[tier].get("n") != expected_n
            or not isinstance(vector_questions, list)
            or len(vector_questions) != expected_n
            or scoring[tier].get("schema") != "epyc.e8_quality_scoring_vector.v1"
            or scoring[tier].get("era") != "E8"
            or scoring[tier].get("tier") != tier
            or scoring[tier].get("n") != expected_n
            or not isinstance(scoring_questions, list)
            or len(scoring_questions) != expected_n
            or [row.get("qid") for row in vector_questions]
            != [row.get("qid") for row in scoring_questions]
        ):
            raise ValueError(f"T{tier} fixed/scoring vector differs")
    total = 0
    for tier_text, expected_n in (("1", 50), ("2", 500)):
        rows = details[tier_text]
        if (
            not isinstance(rows, list)
            or len(rows) != 3
            or {row.get("repetition") for row in rows if isinstance(row, dict)} != {1, 2, 3}
        ):
            raise ValueError(f"T{tier_text} does not contain three repetitions")
        for detail in rows:
            total += 1
            tier = int(tier_text)
            repetition = int(detail.get("repetition", 0))
            tail = detail.get("generation_tail")
            if (
                detail.get("n_results") != expected_n
                or detail.get("response_vector_matches_input") is not True
                or detail.get("per_suite_counts_match_input") is not True
                or detail.get("runtime_binding_matches_pre") is not True
                or detail.get("all_routes_frontdoor") is not True
                or detail.get("error_classification") != {}
                or detail.get("scoring_audit", {}).get("matches") is not True
                or not isinstance(tail, dict)
                or tail.get("schema") != runner.TAIL_SCHEMA
                or tail.get("retry_count") != len(tail.get("targets") or [])
            ):
                raise ValueError(f"T{tier_text} repetition detail differs")
            response_path = resolve_artifact(
                evidence_root,
                detail.get("response_path"),
                "response ledger",
            )
            sidecar_path = resolve_artifact(
                evidence_root,
                detail.get("sidecar_path"),
                "sidecar ledger",
            )
            trace_path = resolve_artifact(
                evidence_root,
                detail.get("judge_trace_path"),
                "judge-trace ledger",
            )
            if (
                detail.get("response_sha256") != sha256_path(response_path)
                or detail.get("sidecar_sha256") != sha256_path(sidecar_path)
                or detail.get("judge_trace_sha256") != sha256_path(trace_path)
            ):
                raise ValueError("repetition artifact hash differs")
            pristine = detail.get("pristine_full_run")
            pristine_artifacts = pristine.get("artifacts") if isinstance(pristine, dict) else None
            if (
                not isinstance(pristine, dict)
                or pristine.get("schema") != "epyc.e8_quality_pristine_full_run.v1"
                or not isinstance(pristine_artifacts, dict)
                or set(pristine_artifacts)
                != {response_path.name, sidecar_path.name, trace_path.name}
            ):
                raise ValueError("pristine full-run snapshot is missing")
            pristine_dir = resolve_artifact(
                evidence_root,
                pristine.get("path"),
                "pristine full-run directory",
            )
            if {path.name for path in pristine_dir.iterdir() if path.is_file()} != set(
                pristine_artifacts
            ) or any(path.is_dir() for path in pristine_dir.iterdir()):
                raise ValueError("pristine full-run directory has an unexpected artifact set")
            pristine_paths: dict[str, Path] = {}
            for name, artifact in pristine_artifacts.items():
                if not isinstance(artifact, dict):
                    raise ValueError("pristine full-run artifact reference is malformed")
                artifact_path = resolve_artifact(
                    evidence_root,
                    artifact.get("path"),
                    "pristine full-run artifact",
                )
                if artifact_path != pristine_dir / name or artifact.get("sha256") != sha256_path(
                    artifact_path
                ):
                    raise ValueError("pristine full-run artifact differs")
                pristine_paths[name] = artifact_path
            scorer_tail = detail.get("scorer_tail_replay")
            scorer_replacements = detail.get("scorer_sidecar_replacement_ordinals")
            if not isinstance(scorer_tail, list) or not isinstance(scorer_replacements, list):
                raise ValueError("scorer-tail replacement disposition is missing")
            scorer_targets: dict[int, str] = {}
            for row in scorer_tail:
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("ordinal"), int)
                    or isinstance(row.get("ordinal"), bool)
                    or not 0 <= row["ordinal"] < expected_n
                    or row.get("ordinal") in scorer_targets
                    or row.get("outcome") != "recovered"
                ):
                    raise ValueError("scorer-tail replacement disposition differs")
                scorer_targets[int(row["ordinal"])] = str(row.get("qid") or "")
            if scorer_replacements != sorted(scorer_targets):
                raise ValueError("scorer-tail sidecar replacement allowlist differs")
            generation_targets = tail.get("targets")
            if not isinstance(generation_targets, list):
                raise ValueError("generation-tail target list differs")
            generation_target_map: dict[int, str] = {}
            for row in generation_targets:
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("ordinal"), int)
                    or isinstance(row.get("ordinal"), bool)
                    or not 0 <= row["ordinal"] < expected_n
                    or row.get("ordinal") in generation_target_map
                ):
                    raise ValueError("generation-tail target identity differs")
                generation_target_map[int(row["ordinal"])] = str(row.get("qid") or "")
            generation_ordinals = set(generation_target_map)
            scorer_ordinals = set(scorer_targets)
            if generation_ordinals & scorer_ordinals:
                raise ValueError("generation and scorer tail targets overlap")
            responses = runner.V4.load_jsonl(response_path)
            expected_qids = [row["qid"] for row in vectors[tier]["questions"]]
            if any(expected_qids[ordinal] != qid for ordinal, qid in scorer_targets.items()) or any(
                expected_qids[ordinal] != qid for ordinal, qid in generation_target_map.items()
            ):
                raise ValueError("tail target differs from fixed-vector identity")
            if [row.get("qid") for row in responses] != expected_qids or any(
                row.get("error") is not None
                or row.get("partial") is not False
                or row.get("degraded") is not False
                or row.get("route_used") != "frontdoor"
                or not str(row.get("answer") or "").strip()
                for row in responses
            ):
                raise ValueError("final response ledger differs")
            _parsed_sidecar, final_sidecars = runner.sidecar_question_rows(
                sidecar_path, expected_n=expected_n
            )
            if any(
                not runner.validate_clean_sidecar_result(
                    responses[ordinal],
                    final_sidecars[ordinal][1],
                    qid=expected_qids[ordinal],
                )
                for ordinal in range(expected_n)
            ):
                raise ValueError("final response and sidecar ledgers are not coherent")
            pristine_response_lines = (
                pristine_paths[response_path.name].read_bytes().splitlines(keepends=True)
            )
            final_response_lines = response_path.read_bytes().splitlines(keepends=True)
            if (
                len(pristine_response_lines) != expected_n
                or len(final_response_lines) != expected_n
                or any(
                    pristine_line != final_line
                    for ordinal, (pristine_line, final_line) in enumerate(
                        zip(pristine_response_lines, final_response_lines)
                    )
                    if ordinal not in generation_ordinals
                )
                or any(
                    pristine_response_lines[ordinal] == final_response_lines[ordinal]
                    for ordinal in generation_ordinals
                )
            ):
                raise ValueError("final response bytes exceed the generation-tail allowlist")
            pristine_sidecar_path = pristine_paths[sidecar_path.name]
            _pristine_parsed, pristine_sidecars = runner.sidecar_question_rows(
                pristine_sidecar_path,
                expected_n=expected_n,
            )
            pristine_sidecar_lines = pristine_sidecar_path.read_bytes().splitlines(keepends=True)
            final_sidecar_lines = sidecar_path.read_bytes().splitlines(keepends=True)
            allowed_sidecar_ordinals = scorer_ordinals | generation_ordinals
            if (
                len(pristine_sidecar_lines) != len(final_sidecar_lines)
                or any(
                    pristine_sidecars[ordinal][0] != final_sidecars[ordinal][0]
                    for ordinal in range(expected_n)
                )
                or any(
                    pristine_line != final_line
                    for line_index, (pristine_line, final_line) in enumerate(
                        zip(pristine_sidecar_lines, final_sidecar_lines)
                    )
                    if line_index
                    not in {pristine_sidecars[ordinal][0] for ordinal in allowed_sidecar_ordinals}
                )
                or any(
                    pristine_sidecar_lines[pristine_sidecars[ordinal][0]]
                    == final_sidecar_lines[final_sidecars[ordinal][0]]
                    for ordinal in allowed_sidecar_ordinals
                )
            ):
                raise ValueError("final sidecar bytes exceed the declared tail allowlists")
            pristine_trace_lines = (
                pristine_paths[trace_path.name].read_bytes().splitlines(keepends=True)
            )
            if (
                not generation_ordinals
                and trace_path.read_bytes() != pristine_paths[trace_path.name].read_bytes()
            ):
                raise ValueError("no-generation-tail run changed judge-trace bytes")
            runner.V4.validate_response_scoring(
                responses,
                scoring[tier]["questions"],
                trace_path,
                default_api_url="http://127.0.0.1:8000",
                tier=tier,
                repetition=repetition,
            )
            if tail["retry_count"]:
                attempts_path = resolve_artifact(
                    evidence_root,
                    tail.get("attempt_path"),
                    "generation-tail attempt ledger",
                )
                attempts = runner.V4.load_jsonl(attempts_path)
                if (
                    tail.get("attempt_sha256") != sha256_path(attempts_path)
                    or len(attempts) != tail["retry_count"]
                    or any(
                        row.get("outcome") != "recovered"
                        or row.get("concurrency") != 1
                        or row.get("request_timeout_s") != 300
                        for row in attempts
                    )
                ):
                    raise ValueError("generation-tail attempt ledger differs")
                targets = {(int(row["ordinal"]), str(row["qid"])): row for row in tail["targets"]}
                if set(targets) != {(int(row["ordinal"]), str(row["qid"])) for row in attempts}:
                    raise ValueError("generation-tail attempts do not cover exact targets")
                retry_traces: dict[int, list[dict[str, Any]]] = {}
                retry_sidecar_rows: dict[int, dict[str, Any]] = {}
                for attempt in attempts:
                    ordinal = int(attempt["ordinal"])
                    target = targets[(ordinal, str(attempt["qid"]))]
                    retry_sidecar_path = resolve_artifact(
                        evidence_root,
                        attempt.get("retry_sidecar_path"),
                        "generation-tail retry sidecar",
                    )
                    retry_trace_path = resolve_artifact(
                        evidence_root,
                        attempt.get("retry_judge_trace_path"),
                        "generation-tail retry judge trace",
                    )
                    _retry_parsed, retry_sidecars = runner.sidecar_question_rows(
                        retry_sidecar_path,
                        expected_n=1,
                    )
                    retry_sidecar_rows[ordinal] = retry_sidecars[0][1]
                    retry_trace_rows = runner.V4.load_jsonl(retry_trace_path)
                    retry_traces[ordinal] = retry_trace_rows
                    if (
                        attempt.get("failure_fingerprint") != target.get("failure_fingerprint")
                        or attempt.get("original_response_sha256") != target.get("response_sha256")
                        or attempt.get("original_sidecar_sha256") != target.get("sidecar_sha256")
                        or attempt.get("retry_response_sha256")
                        != runner.canonical_hash(responses[ordinal])
                        or attempt.get("retry_sidecar_sha256")
                        != runner.canonical_hash(retry_sidecars[0][1])
                        or attempt.get("merged_sidecar_sha256")
                        != runner.canonical_hash(final_sidecars[ordinal][1])
                        or attempt.get("retry_judge_trace_sha256") != sha256_path(retry_trace_path)
                        or not runner.validate_clean_sidecar_result(
                            responses[ordinal],
                            retry_sidecars[0][1],
                            qid=str(attempt["qid"]),
                        )
                        or retry_sidecars[0][1].get("result")
                        != final_sidecars[ordinal][1].get("result")
                    ):
                        raise ValueError("generation-tail attempt provenance differs")
                original_dir = resolve_artifact(
                    evidence_root,
                    tail.get("original_artifact_dir"),
                    "generation-tail original artifact directory",
                )
                original_response_path = original_dir / response_path.name
                original_responses = original_response_path.read_bytes().splitlines(keepends=True)
                final_response_lines = response_path.read_bytes().splitlines(keepends=True)
                target_ordinals = {ordinal for ordinal, _qid in targets}
                if len(original_responses) != len(final_response_lines) or any(
                    original != final
                    for ordinal, (original, final) in enumerate(
                        zip(original_responses, final_response_lines)
                    )
                    if ordinal not in target_ordinals
                ):
                    raise ValueError("generation tail changed non-target response bytes")
                old_response_rows = runner.V4.load_jsonl(original_response_path)
                original_sidecar_path = original_dir / sidecar_path.name
                _old_parsed, old_sidecars = runner.sidecar_question_rows(
                    original_sidecar_path, expected_n=expected_n
                )
                original_sidecar_lines = original_sidecar_path.read_bytes().splitlines(
                    keepends=True
                )
                final_sidecar_lines = sidecar_path.read_bytes().splitlines(keepends=True)
                for ordinal in range(expected_n):
                    if (
                        ordinal not in target_ordinals
                        and original_sidecar_lines[old_sidecars[ordinal][0]]
                        != final_sidecar_lines[final_sidecars[ordinal][0]]
                    ):
                        raise ValueError("generation tail changed a non-target sidecar row")
                for (ordinal, qid), target in targets.items():
                    old_result = old_sidecars[ordinal][1].get("result")
                    retry_result = retry_sidecar_rows[ordinal].get("result")
                    final_result = final_sidecars[ordinal][1].get("result")
                    if (
                        not isinstance(old_result, dict)
                        or not isinstance(retry_result, dict)
                        or not isinstance(final_result, dict)
                        or not isinstance(old_result.get("question_id"), str)
                        or not old_result["question_id"].strip()
                        or retry_result.get("question_id") != old_result["question_id"]
                        or final_result.get("question_id") != old_result["question_id"]
                    ):
                        raise ValueError("generation-tail question identity differs")
                    source = {
                        "ordinal": ordinal,
                        "qid": qid,
                        "error": runner.classify_generation_failure(
                            old_response_rows[ordinal],
                            old_sidecars[ordinal][1],
                        ),
                        "response_sha256": runner.canonical_hash(old_response_rows[ordinal]),
                        "sidecar_sha256": runner.canonical_hash(old_sidecars[ordinal][1]),
                    }
                    if source["error"] is None or target != {
                        **source,
                        "failure_fingerprint": runner.canonical_hash(source),
                    }:
                        raise ValueError(
                            "generation-tail target is not bound to original artifacts"
                        )
                final_trace_lines = trace_path.read_bytes().splitlines(keepends=True)
                validate_tail_trace_replacement(
                    original_trace_lines=pristine_trace_lines,
                    final_trace_lines=final_trace_lines,
                    retry_traces=retry_traces,
                    target_ordinals=target_ordinals,
                    scoring_questions=scoring[tier]["questions"],
                    expected_qids=expected_qids,
                    tier=tier,
                    repetition=repetition,
                )
    if total != 6:
        raise ValueError("v5 does not contain exactly six repetitions")
    records = evidence.get("source_records")
    if (
        not isinstance(records, list)
        or len(records) != 2
        or {record.get("tier") for record in records if isinstance(record, dict)} != {1, 2}
    ):
        raise ValueError("evidence source records differ")
    expected_baselines: dict[str, float] = {}
    expected_suite_quality: dict[str, dict[str, float]] = {}
    expected_suite_counts: dict[str, dict[str, int]] = {}
    expected_histories: dict[str, list[float]] = {}
    expected_provenance: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        tier = int(record["tier"])
        expected_n = 50 if tier == 1 else 500
        summary_path = resolve_artifact(
            evidence_root,
            record.get("path"),
            "tier summary",
        )
        summary = load_json(summary_path, "tier summary")
        observations = summary.get("observations")
        if (
            record.get("sha256") != sha256_path(summary_path)
            or record.get("protocol_id") != runner.PROTOCOL_ID
            or record.get("era") != "E8"
            or record.get("n") != expected_n
            or record.get("question_vector_sha256") != runner.V4.vector_sha256(vectors[tier])
            or record.get("scoring_vector_sha256") != runner.canonical_hash(scoring[tier])
            or summary.get("tier") != tier
            or summary.get("core_id") != record.get("core_id")
            or summary.get("n") != expected_n
            or summary.get("era") != "E8"
            or summary.get("decision_grade") is not True
            or not isinstance(observations, list)
            or len(observations) != 3
            or summary.get("question_vector_sha256")
            != sha256_path(evidence_root / f"question_vector.T{tier}.json")
            or summary.get("scoring_vector_sha256")
            != sha256_path(evidence_root / f"scoring_vector.T{tier}.json")
        ):
            raise ValueError("tier summary is not decision-grade")
        raw_rows: list[dict[str, Any]] = []
        for repetition, observation in enumerate(observations, 1):
            raw_path = resolve_artifact(
                evidence_root,
                observation.get("path"),
                "raw observation",
            )
            raw = load_json(raw_path, "raw observation")
            if (
                raw_path.name != f"raw.T{tier}.r{repetition}.json"
                or observation.get("sha256") != sha256_path(raw_path)
                or observation
                != {
                    "path": str(raw_path),
                    "sha256": sha256_path(raw_path),
                    "q": raw.get("q"),
                    "ts": raw.get("ts"),
                    "core_id": raw.get("core_id"),
                    "protocol_id": raw.get("protocol_id"),
                    "n": raw.get("n"),
                    "era": raw.get("era"),
                }
                or raw.get("protocol_id") != runner.PROTOCOL_ID
                or raw.get("era") != "E8"
                or raw.get("n") != expected_n
                or not isinstance(raw.get("q"), (int, float))
                or not isinstance(raw.get("per_suite_quality"), dict)
                or not isinstance(raw.get("per_suite_counts"), dict)
            ):
                raise ValueError("raw observation differs from its summary")
            raw_rows.append(raw)
        tier_quality = sorted(float(row["q"]) for row in raw_rows)[1]
        suites = set(raw_rows[0]["per_suite_quality"])
        per_suite_quality = {
            suite: sorted(float(row["per_suite_quality"][suite]) for row in raw_rows)[1]
            for suite in suites
        }
        per_suite_counts = dict(raw_rows[0]["per_suite_counts"])
        if (
            any(
                set(row["per_suite_quality"]) != suites
                or row["per_suite_counts"] != per_suite_counts
                for row in raw_rows
            )
            or summary.get("quality") != tier_quality
            or summary.get("per_suite_quality") != per_suite_quality
            or summary.get("per_suite_counts") != per_suite_counts
            or record.get("quality") != tier_quality
            or record.get("timestamp") != raw_rows[-1]["ts"]
        ):
            raise ValueError("tier aggregate differs from raw observations")
        key = str(tier)
        expected_baselines[key] = tier_quality
        expected_suite_quality[key] = per_suite_quality
        expected_suite_counts[key] = per_suite_counts
        expected_histories[key] = [row["q"] for row in raw_rows]
        expected_provenance[key] = [
            {name: row[name] for name in ("q", "ts", "era", "core_id")} for row in raw_rows
        ]
    if evidence.get("replacement") != {
        "baseline_state": {
            "eval_quality_era": "E8",
            "baselines_by_tier": expected_baselines,
            "per_suite_quality_by_tier": expected_suite_quality,
            "per_suite_counts_by_tier": expected_suite_counts,
        },
        "quality_history_by_tier": expected_histories,
        "quality_history_provenance_by_tier": expected_provenance,
    }:
        raise ValueError("state replacement is not derived from sealed observations")
    seal_path = evidence_path.parent / "run_seal.json"
    if evidence.get("run_seal_path") != str(seal_path):
        raise ValueError("evidence run-seal path differs")
    seal = load_json(seal_path, "run seal")
    if (
        seal.get("schema") != "epyc.e8_quality_baseline_run_seal.v1"
        or seal.get("status") != "complete"
        or seal.get("manifest_sha256") != sha256_path(evidence_path)
        or seal.get("runner_report_sha256")
        != sha256_path(evidence_path.parent / "runner_report.json")
        or seal.get("protocol_candidate_sha256") != sha256_path(candidate_path)
        or seal.get("runner_sha256") != expected_runner_sha256
    ):
        raise ValueError("v5 run seal differs")
    bundle = seal.get("bundle_sha256")
    if not isinstance(bundle, dict) or not bundle:
        raise ValueError("v5 bundle seal is missing")
    actual_paths = {
        str(path.resolve())
        for path in evidence_path.parent.rglob("*")
        if path.is_file() and path.name != "run_seal.json"
    }
    if set(bundle) != actual_paths:
        raise ValueError("v5 bundle seal does not have the exact artifact set")
    for path_text, expected in bundle.items():
        path = Path(path_text)
        if not path.is_absolute() or not path.is_file() or sha256_path(path) != expected:
            raise ValueError(f"sealed bundle member differs: {path_text}")
    return {"valid": True, "protocol_id": runner.PROTOCOL_ID, "repetitions": total}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-base-runner-sha256", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            validate(
                args.evidence,
                expected_runner_sha256=args.expected_runner_sha256,
                expected_base_runner_sha256=args.expected_base_runner_sha256,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
