"""Focused contract tests for the separate E8 recovered-r2 validator context."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = PROJECT_ROOT / "scripts/benchmark/validate_e8_quality_baseline_v5.py"
FINALIZER_PATH = PROJECT_ROOT / "scripts/benchmark/finalize_e8_quality_baseline_v5_recovery_r2.py"
spec = importlib.util.spec_from_file_location("e8_recovery_context_validator", VALIDATOR_PATH)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)
finalizer_spec = importlib.util.spec_from_file_location("e8_recovery_finalizer", FINALIZER_PATH)
assert finalizer_spec is not None and finalizer_spec.loader is not None
finalizer = importlib.util.module_from_spec(finalizer_spec)
sys.modules[finalizer_spec.name] = finalizer
finalizer_spec.loader.exec_module(finalizer)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _context(tmp_path: Path) -> tuple[dict, dict]:
    root = tmp_path / "bundle"
    snapshot = root / "intermediate/source_snapshot"
    _write_json(
        snapshot / "question_vector.T1.json",
        {
            "tier": 1,
            "n": 1,
            "core_id": "sealed-t1-core",
            "questions": [{"qid": "t1-q"}],
        },
    )
    _write_json(snapshot / "question_vector.T2.json", {"fixed": "source"})
    scoring_questions = [
        {
            "qid": f"q{ordinal}",
            "suite": "thinking",
            "scoring_method": "exact_match",
            "scoring_config": {},
            "expected": str(ordinal),
            "prompt_sha256": hashlib.sha256(f"prompt-{ordinal}".encode()).hexdigest(),
        }
        for ordinal in range(500)
    ]
    _write_json(
        snapshot / "scoring_vector.T2.json",
        {
            "schema": "epyc.e8_quality_scoring_vector.v1",
            "tier": 2,
            "n": 500,
            "questions": scoring_questions,
        },
    )
    saved_rows = [
        {
            "row_type": "question_result",
            "ordinal": ordinal,
            "answer": str(ordinal),
            "result": {"qid": f"q{ordinal}"},
        }
        for ordinal in (59, 60, 61)
    ]
    _write_jsonl(
        snapshot / "eval_sidecars/question_results.e8-t2-r2.jsonl",
        saved_rows,
    )
    source_hashes = {
        str(path.relative_to(snapshot)): _sha(path)
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    }
    _write_json(
        snapshot / "source_binding.json",
        {
            "source_sha256": source_hashes,
            "source_tree_sha256": validator.canonical_hash(source_hashes),
        },
    )
    reuse = list(range(59))
    replay = [59, 60, 61]
    generation = list(range(62, 500))
    plan = {
        "schema": validator.RECOVERY_R2_PLAN_SCHEMA,
        "protocol_id": "e8_quality_full_pool_tier_baseline.v5",
        "source_sha256": source_hashes,
        "source_tree_sha256": validator.canonical_hash(source_hashes),
        "tier": 2,
        "repetition": 2,
        "n": 500,
        "generation_concurrency": 3,
        "t1_core_id": "sealed-t1-core",
        "reuse_ordinals": reuse,
        "scorer_replay_ordinals": replay,
        "generation_ordinals": generation,
    }
    _write_json(root / "intermediate/partial_r2_plan.json", plan)
    claim = {
        "claims": [{"payload": {"request_tag": "e8", "region": "q0"}}],
        "global_claims": [{"region": "q0"}],
    }
    proposal = {
        "schema": validator.RECOVERY_R2_PROPOSAL_SCHEMA,
        "status": "observation_only",
        "protocol_id": "e8_quality_full_pool_tier_baseline.v5",
        "source_tree_sha256": plan["source_tree_sha256"],
        "generation_concurrency": 3,
        "generation_ordinals_sha256": validator.canonical_hash(generation),
        "scorer_replay_ordinals_sha256": validator.canonical_hash(replay),
        "instrument": {
            "commit": "c",
            "runner_sha256": "a" * 64,
            "measurement_source_sha256": {
                "/a": "a" * 64,
                "/b": "b" * 64,
                "/c": "c" * 64,
                "/d": "d" * 64,
            },
        },
        "output_namespace": "/tmp/recovery",
        "region_claim": {"tag": "e8", "regions": ["q0"]},
        "frontdoor_capacity": {"capacity": 3},
        "application": "requires_separate_human_finalizer",
    }
    _write_json(root / "intermediate/recovery_proposal.json", proposal)
    response = root / "intermediate/responses.T2.r2.jsonl"
    sidecar = root / "intermediate/eval_sidecars/question_results.e8-t2-r2.jsonl"
    trace = root / "intermediate/judge_traces.T2.r2.jsonl"
    raw = root / "intermediate/raw.T2.r2.json"
    _write_jsonl(response, [])
    _write_jsonl(sidecar, [])
    _write_jsonl(trace, [])
    _write_json(raw, {"q": 0.0})
    watcher_path = root / "intermediate/runtime_watch.r2.jsonl"
    watcher_rows = [
        {
            "ok": True,
            "started_at": "2026-01-01T00:00:00Z",
            "active_load": {"tier": 2, "repetition": 2},
            "api_probe_urls": {},
            "runtime_artifacts": {},
        },
        {
            "ok": True,
            "started_at": "2026-01-01T00:00:05Z",
            "active_load": None,
            "api_probe_urls": {},
            "runtime_artifacts": {},
        },
    ]
    _write_jsonl(watcher_path, watcher_rows)
    complete = {
        "schema": validator.RECOVERY_R2_COMPLETE_SCHEMA,
        "status": "intermediate_r2_complete",
        "plan_sha256": _sha(root / "intermediate/partial_r2_plan.json"),
        "responses_sha256": _sha(response),
        "sidecar_sha256": _sha(sidecar),
        "trace_sha256": _sha(trace),
        "raw_sha256": _sha(raw),
        "watcher": {
            "path": str(watcher_path),
            "sha256": _sha(watcher_path),
            "samples": 2,
            "claim_before": claim,
            "claim_after": claim,
            "proposal_sha256": _sha(root / "intermediate/recovery_proposal.json"),
            "binding_sha256": validator._monitor_binding_sha256(watcher_rows[0]),
            "observed_gap_count_over_7s": 0,
            "observed_max_gap_s": 5.0,
        },
        "claim": claim,
    }
    _write_json(root / "intermediate/r2_complete.json", complete)
    journal = [
        {"ordinal": ordinal, "source": source, "response": {"qid": f"q{ordinal}"}}
        for source, ordinals in (
            ("reuse", reuse),
            ("scorer_replay", replay),
            ("generation", generation),
        )
        for ordinal in ordinals
    ]
    _write_jsonl(root / "intermediate/recovery_rows.T2.r2.jsonl", journal)
    scorer_attempts = [
        {
            "schema": validator.RECOVERY_R2_SCORER_ATTEMPTS_SCHEMA,
            "ordinal": ordinal,
            "qid": f"q{ordinal}",
            "saved_sidecar_sha256": validator.canonical_hash(saved_rows[ordinal - 59]),
            "scoring_question_sha256": validator.canonical_hash(scoring_questions[ordinal]),
            "state": state,
        }
        for ordinal in replay
        for state in ("started", "succeeded")
    ]
    scorer_attempts_path = root / "intermediate/scorer_attempts.T2.r2.jsonl"
    _write_jsonl(scorer_attempts_path, scorer_attempts)
    complete["journal_sha256"] = _sha(root / "intermediate/recovery_rows.T2.r2.jsonl")
    complete["scorer_attempts_sha256"] = _sha(scorer_attempts_path)
    complete["scorer_attempts"] = {
        "path": "scorer_attempts.T2.r2.jsonl",
        "sha256": _sha(scorer_attempts_path),
        "records": 6,
        "expected_terminal_count": 3,
        "terminal_states": {"succeeded": 3},
    }
    _write_json(root / "intermediate/r2_complete.json", complete)
    _write_json(root / "partial_resume_plan.json", {"schema": "partial"})
    _write_jsonl(
        root / "generation_tail_attempts.T2.r1.jsonl",
        [{"schema": "attempt", "ordinal": 98}],
    )
    context = {
        "schema": validator.RECOVERY_R2_CONTEXT_SCHEMA,
        "recovery_runner": {"path": "/reviewed/recovery.py", "sha256": "a" * 64},
        "finalizer_runner": {
            "path": "/reviewed/finalizer.py",
            "sha256": _sha(validator.FINALIZER_PATH),
        },
        "dependency_sha256": {"v5": "c" * 64, "resume": "d" * 64, "recovery": "a" * 64},
        "banked_t2_r1_repair_history": {
            "partial_resume_plan.json": {
                "path": str(root / "partial_resume_plan.json"),
                "sha256": _sha(root / "partial_resume_plan.json"),
            },
            "generation_tail_attempts.T2.r1.jsonl": {
                "path": str(root / "generation_tail_attempts.T2.r1.jsonl"),
                "sha256": _sha(root / "generation_tail_attempts.T2.r1.jsonl"),
            },
        },
        "source_binding": str(snapshot / "source_binding.json"),
        "source_binding_sha256": _sha(snapshot / "source_binding.json"),
        "source_tree_sha256": plan["source_tree_sha256"],
        "plan_path": str(root / "intermediate/partial_r2_plan.json"),
        "plan_sha256": _sha(root / "intermediate/partial_r2_plan.json"),
        "proposal_path": str(root / "intermediate/recovery_proposal.json"),
        "proposal_sha256": _sha(root / "intermediate/recovery_proposal.json"),
        "complete_path": str(root / "intermediate/r2_complete.json"),
        "complete_sha256": _sha(root / "intermediate/r2_complete.json"),
        "watcher_path": str(watcher_path),
        "watcher_sha256": _sha(watcher_path),
        "response_path": str(response),
        "sidecar_path": str(sidecar),
        "trace_path": str(trace),
        "raw_path": str(raw),
        "journal_path": str(root / "intermediate/recovery_rows.T2.r2.jsonl"),
        "journal_sha256": _sha(root / "intermediate/recovery_rows.T2.r2.jsonl"),
        "scorer_attempts_path": str(scorer_attempts_path),
        "scorer_attempts_sha256": _sha(scorer_attempts_path),
    }
    return root, context


def _validate(root: Path, context: dict) -> dict:
    return validator.validate_recovery_r2_context(
        {"recovery_r2": context},
        evidence_root=root,
        expected_recovery_runner_sha256="a" * 64,
        expected_finalizer_runner_sha256=_sha(validator.FINALIZER_PATH),
        expected_v5_runner_sha256="c" * 64,
        expected_base_runner_sha256="b" * 64,
        expected_resume_runner_sha256="d" * 64,
    )


def test_recovery_r2_context_accepts_hash_bound_59_3_438_bundle(tmp_path: Path) -> None:
    root, context = _context(tmp_path)
    accepted = _validate(root, context)
    assert accepted is not None
    assert len(accepted["plan"]["generation_ordinals"]) == 438


def test_recovery_r2_context_rejects_changed_on_disk_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, context = _context(tmp_path)
    changed = tmp_path / "changed-finalizer.py"
    changed.write_text("# unreviewed finalizer\n")
    monkeypatch.setattr(validator, "FINALIZER_PATH", changed)
    with pytest.raises(ValueError, match="finalizer instrument differs"):
        _validate(root, context)


def test_finalizer_conditionally_recomputes_mixed_tail_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = {
        "schema": "epyc.e8_quality_v5_partial_r2_mixed_tail_chain.v1",
        "descriptor_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        finalizer.RACE_RETRY,
        "validate_mixed_predecessor",
        lambda *_args: chain,
    )
    race_plan = {"mixed_tail_repair": chain}

    accepted = finalizer._validate_optional_mixed_tail_chain(
        tmp_path,
        {},
        [],
        {},
        race_plan,
    )

    assert accepted == chain
    with pytest.raises(ValueError, match="mixed-tail chain"):
        finalizer._validate_optional_mixed_tail_chain(
            tmp_path,
            {},
            [],
            {},
            {},
        )


def test_validator_requires_exact_mixed_runner_and_nested_bindings(
    tmp_path: Path,
) -> None:
    race_root = tmp_path / "race"
    evidence = race_root / "predecessor_snapshot/mixed_tail_repair.json"
    binding = (
        race_root
        / "predecessor_snapshot/predecessor_snapshot/source_binding.json"
    )
    _write_json(evidence, {"sealed": True})
    _write_json(binding, {"sealed": True})
    runner_sha = _sha(validator.MIXED_TAIL_REPAIR_RUNNER_PATH)
    mixed = {
        "repair_runner_sha256": runner_sha,
        "descriptor_sha256": "b" * 64,
        "original_source": {"tree_sha256": "c" * 64},
    }
    context = {
        "mixed_tail_repair_runner": {
            "path": str(validator.MIXED_TAIL_REPAIR_RUNNER_PATH),
            "sha256": runner_sha,
        },
        "mixed_tail_repair_descriptor_sha256": "b" * 64,
        "mixed_tail_repair_evidence_path": str(evidence),
        "mixed_tail_repair_evidence_sha256": _sha(evidence),
        "mixed_tail_original_source_binding": str(binding),
        "mixed_tail_original_source_binding_sha256": _sha(binding),
        "mixed_tail_original_source_tree_sha256": "c" * 64,
    }

    with pytest.raises(ValueError, match="externally reviewed hash"):
        validator.validate_mixed_tail_repair_context(
            context,
            evidence_root=tmp_path,
            race_root=race_root,
            mixed=mixed,
            expected_mixed_tail_repair_runner_sha256=None,
        )
    with pytest.raises(ValueError, match="externally reviewed hash"):
        validator.validate_mixed_tail_repair_context(
            context,
            evidence_root=tmp_path,
            race_root=race_root,
            mixed=mixed,
            expected_mixed_tail_repair_runner_sha256="0" * 64,
        )
    validator.validate_mixed_tail_repair_context(
        context,
        evidence_root=tmp_path,
        race_root=race_root,
        mixed=mixed,
        expected_mixed_tail_repair_runner_sha256=runner_sha,
    )
    context["mixed_tail_repair_evidence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="nested source or evidence"):
        validator.validate_mixed_tail_repair_context(
            context,
            evidence_root=tmp_path,
            race_root=race_root,
            mixed=mixed,
            expected_mixed_tail_repair_runner_sha256=runner_sha,
        )


def test_completed_successor_context_reaches_external_validator(tmp_path: Path) -> None:
    """Exercise a completed successor through the final validator, not source text.

    This is deliberately synthetic, but every ledger/hash/watcher relation is
    materialized.  It covers the successor's different 59/3/128/12/298
    disposition and confirms the external gate accepts it only with a pinned
    successor runner and an excluded failed-watcher snapshot.
    """
    root, _context_value = _context(tmp_path)
    intermediate = root / "intermediate"
    snapshot = intermediate / "source_snapshot"
    scoring = json.loads((snapshot / "scoring_vector.T2.json").read_text())["questions"]
    reuse = list(range(59))
    inherited = list(range(59, 62))
    defects = [97, 138]
    imported = [ordinal for ordinal in range(62, 192) if ordinal not in defects]
    replay = list(range(192, 204))
    generation = [*defects, *range(204, 500)]
    plan_path = intermediate / "partial_r2_plan.json"
    plan = {
        "schema": validator.RECOVERY_R2_SUCCESSOR_PLAN_SCHEMA,
        "protocol_id": "e8_quality_full_pool_tier_baseline.v5",
        "source_sha256": json.loads((snapshot / "source_binding.json").read_text())["source_sha256"],
        "source_tree_sha256": json.loads((snapshot / "source_binding.json").read_text())["source_tree_sha256"],
        "successor_runner_sha256": _sha(validator.SUCCESSOR_RUNNER_PATH),
        "tier": 2,
        "repetition": 2,
        "n": 500,
        "generation_concurrency": 3,
        "t1_core_id": "sealed-t1-core",
        "reuse_ordinals": reuse,
        "inherited_scorer_replay_ordinals": inherited,
        "imported_generation_ordinals": imported,
        "scorer_replay_ordinals": replay,
        "generation_defect_ordinals": defects,
        "generation_ordinals": generation,
        "successor_watcher_path": "runtime_watch.r2.successor.jsonl",
    }
    failed = intermediate / "failed_source_snapshot"
    for ordinal in replay:
        scoring[ordinal]["scoring_method"] = "llm_judge"
    scoring_vector = json.loads((snapshot / "scoring_vector.T2.json").read_text())
    scoring_vector["questions"] = scoring
    _write_json(snapshot / "scoring_vector.T2.json", scoring_vector)
    source_hashes = {
        str(path.relative_to(snapshot)): _sha(path)
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path.name != "source_binding.json"
    }
    _write_json(
        snapshot / "source_binding.json",
        {"source_sha256": source_hashes, "source_tree_sha256": validator.canonical_hash(source_hashes)},
    )
    plan["source_sha256"] = source_hashes
    plan["source_tree_sha256"] = validator.canonical_hash(source_hashes)

    def clean_sidecar(ordinal: int) -> dict:
        answer = f"answer-{ordinal}"
        return {
            "row_type": "question_result",
            "ordinal": ordinal,
            "answer": answer,
            "result": {
                "qid": f"q{ordinal}",
                "question_id": f"q{ordinal}",
                "tokens_generated": 1,
                "correct": False,
                "route": "frontdoor",
                "answer_hash": validator.load_runner()._normalized_answer_hash(answer),
                "suite": "thinking",
                **({"scoring_method": "llm_judge"} if ordinal in replay else {}),
            },
        }

    failed_rows = [clean_sidecar(ordinal) for ordinal in imported]
    failed_rows.extend(
        {
            **clean_sidecar(ordinal),
            "result": {
                **clean_sidecar(ordinal)["result"],
                "error": True,
                "error_detail": "scoring_unavailable: synthetic",
            },
        }
        for ordinal in replay
    )
    failed_rows.extend(
        {
            "row_type": "question_result",
            "ordinal": ordinal,
            "answer": "",
            "result": {
                "qid": f"q{ordinal}", "question_id": f"q{ordinal}",
                "tokens_generated": 0, "error": True, "error_detail": "timed out",
                "route": "frontdoor", "suite": "thinking",
            },
        }
        for ordinal in defects
    )
    _write_jsonl(failed / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl", failed_rows)
    failed_watcher = failed / "runtime_watch.r2.jsonl"
    _write_jsonl(failed_watcher, [{"ok": False, "started_at": "2026-01-01T00:00:00Z"}])
    _write_json(
        failed / "partial_r2_plan.json",
        {
            "schema": validator.RECOVERY_R2_PLAN_SCHEMA,
            "protocol_id": plan["protocol_id"], "n": 500,
            "reuse_ordinals": reuse, "scorer_replay_ordinals": inherited,
        },
    )
    _write_jsonl(
        failed / "recovery_rows.T2.r2.jsonl",
        [
            {"ordinal": ordinal, "source": source, "response": {"qid": scoring[ordinal]["qid"]}}
            for source, ordinals in (("reuse", reuse), ("scorer_replay", inherited))
            for ordinal in ordinals
        ],
    )
    failed_hashes = {
        str(path.relative_to(failed)): _sha(path)
        for path in sorted(failed.rglob("*"))
        if path.is_file()
    }
    _write_json(
        failed / "source_binding.json",
        {"source_sha256": failed_hashes, "source_tree_sha256": validator.canonical_hash(failed_hashes)},
    )
    plan["failed_source_sha256"] = failed_hashes
    plan["failed_source_tree_sha256"] = validator.canonical_hash(failed_hashes)
    plan["failed_watcher"] = {
        "path": "runtime_watch.r2.jsonl",
        "sha256": _sha(failed_watcher),
        "eligibility": "excluded_audit_evidence",
    }
    _write_json(plan_path, plan)
    claim = {
        "claims": [{"payload": {"request_tag": "e8", "region": "q0"}}],
        "global_claims": [{"region": "q0"}],
    }
    proposal_path = intermediate / "recovery_proposal.json"
    _write_json(
        proposal_path,
        {
            "schema": validator.RECOVERY_R2_SUCCESSOR_PROPOSAL_SCHEMA,
            "status": "observation_only",
            "protocol_id": plan["protocol_id"],
            "source_tree_sha256": plan["source_tree_sha256"],
            "failed_source_tree_sha256": plan["failed_source_tree_sha256"],
            "failed_watcher": plan["failed_watcher"],
            "successor_runner_sha256": plan["successor_runner_sha256"],
            "generation_concurrency": 3,
            "generation_ordinals_sha256": validator.canonical_hash(generation),
            "scorer_replay_ordinals_sha256": validator.canonical_hash(replay),
            "instrument": {},
            "region_claim": {"tag": "e8", "regions": ["q0"]},
            "frontdoor_capacity": {"capacity": 3},
            "application": "requires_separate_human_finalizer",
        },
    )
    journal = [
        {"ordinal": ordinal, "source": source, "response": {"qid": scoring[ordinal]["qid"]}}
        for source, ordinals in (
            ("reuse", reuse),
            ("scorer_replay", inherited),
            ("imported_generation", imported),
            ("scorer_replay", replay),
            ("generation", generation),
        )
        for ordinal in ordinals
    ]
    journal_path = intermediate / "recovery_rows.T2.r2.jsonl"
    _write_jsonl(journal_path, journal)
    rows_by_ordinal = {row["ordinal"]: row for row in failed_rows}
    responses = []
    final_sidecars = []
    for ordinal in range(500):
        source_row = rows_by_ordinal.get(ordinal, clean_sidecar(ordinal))
        answer = str(source_row.get("answer") or f"answer-{ordinal}")
        response = {
            "qid": scoring[ordinal]["qid"],
            "suite": "thinking",
            "scoring_method": scoring[ordinal]["scoring_method"],
            "answer": answer,
            "correct": False,
            "error": None,
            "partial": False,
            "degraded": False,
            "route_used": "frontdoor",
            "scoring_config_sha256": validator.canonical_hash({}),
        }
        responses.append(response)
        result = {
            **source_row["result"],
            "qid": response["qid"],
            "question_id": response["qid"],
            "tokens_generated": 1,
            "correct": False,
            "route": "frontdoor",
            "answer_hash": validator.load_runner()._normalized_answer_hash(answer),
        }
        result.pop("error", None)
        result.pop("error_detail", None)
        final_sidecars.append(
            {"row_type": "question_result", "ordinal": ordinal, "answer": answer, "result": result}
        )
    response = intermediate / "responses.T2.r2.jsonl"
    sidecar = intermediate / "eval_sidecars/question_results.e8-t2-r2.jsonl"
    trace = intermediate / "judge_traces.T2.r2.jsonl"
    raw = intermediate / "raw.T2.r2.json"
    _write_jsonl(response, responses)
    _write_jsonl(sidecar, final_sidecars)
    _write_jsonl(trace, [])
    for row in journal:
        row["response"] = responses[row["ordinal"]]
    _write_jsonl(journal_path, journal)
    _write_jsonl(
        failed / "recovery_rows.T2.r2.jsonl",
        [row for row in journal if row["ordinal"] in {*reuse, *inherited}],
    )
    failed_hashes = {
        str(path.relative_to(failed)): _sha(path)
        for path in sorted(failed.rglob("*"))
        if path.is_file() and path.name != "source_binding.json"
    }
    _write_json(
        failed / "source_binding.json",
        {"source_sha256": failed_hashes, "source_tree_sha256": validator.canonical_hash(failed_hashes)},
    )
    plan["failed_source_sha256"] = failed_hashes
    plan["failed_source_tree_sha256"] = validator.canonical_hash(failed_hashes)
    _write_json(plan_path, plan)
    proposal = json.loads(proposal_path.read_text())
    proposal["failed_source_tree_sha256"] = plan["failed_source_tree_sha256"]
    proposal["failed_watcher"] = plan["failed_watcher"]
    _write_json(proposal_path, proposal)
    attempts = [
        {
            "schema": validator.RECOVERY_R2_SCORER_ATTEMPTS_SCHEMA,
            "ordinal": ordinal,
            "qid": scoring[ordinal]["qid"],
            "saved_sidecar_sha256": validator.canonical_hash(
                next(row for row in failed_rows if row["ordinal"] == ordinal)
            ),
            "scoring_question_sha256": validator.canonical_hash(scoring[ordinal]),
            "state": state,
        }
        for ordinal in replay
        for state in ("started", "succeeded")
    ]
    attempts_path = intermediate / "scorer_attempts.T2.r2.jsonl"
    _write_jsonl(attempts_path, attempts)
    watcher_path = intermediate / "runtime_watch.r2.jsonl"
    watcher_rows = [
        {
            "ok": True,
            "started_at": "2026-01-01T00:00:00Z",
            "active_load": {"tier": 2, "repetition": 2},
            "api_probe_urls": {},
            "runtime_artifacts": {},
        },
        {
            "ok": True,
            "started_at": "2026-01-01T00:00:05Z",
            "active_load": None,
            "api_probe_urls": {},
            "runtime_artifacts": {},
        },
    ]
    _write_jsonl(watcher_path, watcher_rows)
    _write_jsonl(intermediate / "runtime_watch.r2.successor.jsonl", watcher_rows)
    complete_path = intermediate / "r2_complete.json"
    complete = {
        "schema": validator.RECOVERY_R2_COMPLETE_SCHEMA,
        "status": "intermediate_r2_successor_complete",
        "plan_sha256": _sha(plan_path),
        "responses_sha256": _sha(response), "sidecar_sha256": _sha(sidecar),
        "trace_sha256": _sha(trace), "raw_sha256": _sha(raw),
        "journal_sha256": _sha(journal_path), "scorer_attempts_sha256": _sha(attempts_path),
        "failed_watcher": plan["failed_watcher"], "claim": claim,
        "watcher": {
            "sha256": _sha(watcher_path), "samples": 2,
            "claim_before": claim, "claim_after": claim,
            "proposal_sha256": _sha(proposal_path),
            "binding_sha256": validator._monitor_binding_sha256(watcher_rows[0]),
            "observed_gap_count_over_7s": 0, "observed_max_gap_s": 5.0,
        },
        "scorer_attempts": {
            "path": attempts_path.name, "sha256": _sha(attempts_path),
            "records": 24, "expected_terminal_count": 12,
            "terminal_states": {"succeeded": 12},
        },
    }
    _write_json(complete_path, complete)
    finalized = finalizer.validate_intermediate(intermediate)
    assert finalized["successor"] is True
    staging = root / "finalizer-staging"
    for source_path in snapshot.rglob("*"):
        if source_path.is_file():
            target = staging / "source_snapshot" / source_path.relative_to(snapshot)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source_path.read_bytes())
    _write_json(staging / "partial_resume_plan.json", {"source_tree_sha256": "synthetic"})
    _write_jsonl(staging / "generation_tail_attempts.T2.r1.jsonl", [])
    _write_jsonl(staging / "judge_traces.T2.r1.jsonl", [])
    _write_json(staging / "recovery_finalizer_source_plan.json", {"source": "synthetic"})
    _write_json(staging / "e8_quality_baseline_evidence.json", {"schema": "synthetic"})
    _write_json(staging / "run_seal.json", {"schema": "synthetic"})
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    source_resume_rows = []
    for ordinal in range(411):
        elapsed = ordinal * 5 + (3 if ordinal >= 200 else 0)
        source_resume_rows.append(
            {
                "ok": True,
                "started_at": (base_time + timedelta(seconds=elapsed)).isoformat().replace("+00:00", "Z"),
                "active_load": {"tier": 2, "repetition": 1 if ordinal == 0 else 2 if ordinal == 1 else 3},
                "api_probe_urls": {},
                "runtime_artifacts": {},
            }
        )
    _write_jsonl(staging / "source_resume_runtime_watch.jsonl", source_resume_rows)
    historical = {"source": "historical", "sample_indexes": [0]}
    resume = {"source": "resume", "sample_indexes": [1]}
    report = {
        "postconditions": {
            "watcher_samples": [source_resume_rows[0], source_resume_rows[-1]],
            "segmented_monitor": [historical, resume],
            "held_region_claim": claim,
        }
    }
    _write_json(staging / "runner_report.json", report)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(finalizer, "SOURCE_RESUME_WATCHER_SHA256", _sha(staging / "source_resume_runtime_watch.jsonl"))
    monkeypatch.setattr(finalizer, "SOURCE_RESUME_BINDING_SHA256", finalizer.RESUME._monitor_binding_sha256(source_resume_rows[0]))
    monkeypatch.setattr(finalizer, "SOURCE_RESUME_MAX_GAP_S", 8.0)
    monkeypatch.setattr(finalizer.RESUME, "_scorer_recovery_rows", lambda *_args, **_kwargs: [])
    try:
        finalizer._rewrite_for_recovery(staging, staging, finalized)
    finally:
        monkeypatch.undo()
    for name in ("responses.T2.r2.jsonl", "judge_traces.T2.r2.jsonl", "raw.T2.r2.json"):
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((intermediate / name).read_bytes())
    sidecar_target = staging / "eval_sidecars/question_results.e8-t2-r2.jsonl"
    sidecar_target.parent.mkdir(parents=True, exist_ok=True)
    sidecar_target.write_bytes(sidecar.read_bytes())
    emitted = json.loads((staging / "runner_report.json").read_text())["recovery_r2"]
    accepted = validator.validate_recovery_r2_context(
        {"recovery_r2": emitted}, evidence_root=staging,
        expected_recovery_runner_sha256=_sha(finalizer.RECOVERY_PATH),
        expected_finalizer_runner_sha256=_sha(finalizer.FINALIZER_PATH) if hasattr(finalizer, "FINALIZER_PATH") else _sha(FINALIZER_PATH),
        expected_successor_runner_sha256=_sha(validator.SUCCESSOR_RUNNER_PATH),
        expected_v5_runner_sha256=_sha(finalizer.V5_PATH),
        expected_base_runner_sha256=_sha(finalizer.V5.V4_PATH),
        expected_resume_runner_sha256=_sha(finalizer.RESUME_PATH),
    )
    assert accepted and accepted["successor"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "epyc.e8_quality_v5_partial_r2_plan.v1"),
        ("t1_core_id", None),
        ("t1_core_id", "wrong-core"),
    ],
)
def test_finalizer_rejects_legacy_or_unbound_t1_plan(
    tmp_path: Path, field: str, value: str | None
) -> None:
    root, _context_value = _context(tmp_path)
    plan_path = root / "intermediate/partial_r2_plan.json"
    plan = json.loads(plan_path.read_text())
    if value is None:
        plan.pop(field)
    else:
        plan[field] = value
    _write_json(plan_path, plan)
    with pytest.raises(ValueError, match="plan differs"):
        finalizer.validate_intermediate(root / "intermediate")


def test_independent_validator_rejects_mismatched_t1_plan(tmp_path: Path) -> None:
    root, context = _context(tmp_path)
    plan_path = Path(context["plan_path"])
    plan = json.loads(plan_path.read_text())
    plan["t1_core_id"] = "wrong-core"
    _write_json(plan_path, plan)
    context["plan_sha256"] = _sha(plan_path)
    with pytest.raises(ValueError, match="T1 core binding"):
        _validate(root, context)


@pytest.mark.parametrize("field", ["schema", "plan_sha256", "source_tree_sha256"])
def test_recovery_r2_context_rejects_schema_or_hash_drift(tmp_path: Path, field: str) -> None:
    root, context = _context(tmp_path)
    context[field] = "wrong" if field == "schema" else "0" * 64
    with pytest.raises(ValueError):
        _validate(root, context)


def test_recovery_r2_context_rejects_ordinal_allowlist_drift(tmp_path: Path) -> None:
    root, context = _context(tmp_path)
    plan_path = Path(context["plan_path"])
    plan = json.loads(plan_path.read_text())
    plan["generation_ordinals"][-1] = 61
    _write_json(plan_path, plan)
    context["plan_sha256"] = _sha(plan_path)
    with pytest.raises(ValueError, match="allowlist"):
        _validate(root, context)


def test_recovery_r2_context_rejects_failed_or_extra_scorer_attempts(tmp_path: Path) -> None:
    root, context = _context(tmp_path)
    attempts_path = Path(context["scorer_attempts_path"])
    attempts = [json.loads(line) for line in attempts_path.read_text().splitlines()]
    attempts[1]["state"] = "failed"
    _write_jsonl(attempts_path, attempts)
    digest = _sha(attempts_path)
    context["scorer_attempts_sha256"] = digest
    complete_path = Path(context["complete_path"])
    complete = json.loads(complete_path.read_text())
    complete["scorer_attempts_sha256"] = digest
    complete["scorer_attempts"]["sha256"] = digest
    _write_json(complete_path, complete)
    context["complete_sha256"] = _sha(complete_path)
    with pytest.raises(ValueError, match="scorer-attempt record differs"):
        _validate(root, context)


@pytest.mark.parametrize("field", ["saved_sidecar_sha256", "scoring_question_sha256"])
def test_recovery_r2_context_derives_scorer_attempt_hashes_from_snapshot(
    tmp_path: Path, field: str
) -> None:
    root, context = _context(tmp_path)
    attempts_path = Path(context["scorer_attempts_path"])
    attempts = [json.loads(line) for line in attempts_path.read_text().splitlines()]
    attempts[0][field] = "0" * 64
    attempts[1][field] = "0" * 64
    _write_jsonl(attempts_path, attempts)
    digest = _sha(attempts_path)
    context["scorer_attempts_sha256"] = digest
    complete_path = Path(context["complete_path"])
    complete = json.loads(complete_path.read_text())
    complete["scorer_attempts_sha256"] = digest
    complete["scorer_attempts"]["sha256"] = digest
    _write_json(complete_path, complete)
    context["complete_sha256"] = _sha(complete_path)

    with pytest.raises(ValueError, match="scorer-attempt record differs"):
        _validate(root, context)


def test_finalizer_scorer_contract_uses_producer_canonical_source_semantics(
    tmp_path: Path,
) -> None:
    root, context = _context(tmp_path)
    snapshot = Path(context["source_binding"]).parent
    plan = json.loads(Path(context["plan_path"]).read_text())
    expected = finalizer._expected_scorer_attempt_inputs(
        snapshot, plan["scorer_replay_ordinals"], expected_n=plan["n"]
    )
    saved = [
        row
        for row in finalizer.V4.load_jsonl(
            snapshot / "eval_sidecars/question_results.e8-t2-r2.jsonl"
        )
        if row.get("row_type") == "question_result"
    ]
    scoring = finalizer.V4.load_json(snapshot / "scoring_vector.T2.json")["questions"]

    for ordinal in plan["scorer_replay_ordinals"]:
        assert expected[ordinal] == {
            "qid": f"q{ordinal}",
            "saved_sidecar_sha256": finalizer.RECOVERY.canonical_hash(
                next(row for row in saved if row["ordinal"] == ordinal)
            ),
            "scoring_question_sha256": finalizer.RECOVERY.canonical_hash(scoring[ordinal]),
        }


def test_recovery_r2_context_rejects_nonexistent_repair_history(tmp_path: Path) -> None:
    root, context = _context(tmp_path)
    entry = context["banked_t2_r1_repair_history"]["partial_resume_plan.json"]
    entry["path"] = str(root / "missing-partial-plan.json")
    with pytest.raises(ValueError, match="repair-history"):
        _validate(root, context)


def test_recovery_r2_context_rejects_source_watcher_and_claim_drift(tmp_path: Path) -> None:
    root, context = _context(tmp_path)
    source = Path(context["source_binding"]).parent / "question_vector.T2.json"
    source.write_text("tampered\n")
    with pytest.raises(ValueError, match="source binding"):
        _validate(root, context)
    root, context = _context(tmp_path / "watcher")
    complete_path = Path(context["complete_path"])
    complete = json.loads(complete_path.read_text())
    complete["watcher"]["sha256"] = "0" * 64
    _write_json(complete_path, complete)
    context["complete_sha256"] = _sha(complete_path)
    with pytest.raises(ValueError, match="watcher"):
        _validate(root, context)
    root, context = _context(tmp_path / "claim")
    complete_path = Path(context["complete_path"])
    complete = json.loads(complete_path.read_text())
    complete["watcher"]["claim_after"] = {"changed": True}
    _write_json(complete_path, complete)
    context["complete_sha256"] = _sha(complete_path)
    with pytest.raises(ValueError, match="claim"):
        _validate(root, context)


def test_finalizer_plan_accepts_the_preserved_completed_r1_source() -> None:
    source = Path(
        "/mnt/raid0/llm/epyc-root/artifacts/operator/"
        ".e8_quality_baseline_v5_partial_resume_promptfix_20260728.staging-b0d7ce62d6e04509a1cec7849aa68832"
    )
    plan = finalizer.build_plan(source)
    assert plan["banked"] == {"tiers": [1], "t2_r1": True}
    assert plan["fresh_collection"] == [{"tier": 2, "repetition": 3}]


def test_real_source_tail_is_rebound_to_the_new_bundle_without_regeneration(tmp_path: Path) -> None:
    source = Path(
        "/mnt/raid0/llm/epyc-root/artifacts/operator/"
        ".e8_quality_baseline_v5_partial_resume_promptfix_20260728.staging-"
        "b0d7ce62d6e04509a1cec7849aa68832"
    )
    plan = finalizer.build_plan(source)
    staging = tmp_path / "staging"
    destination = tmp_path / "published"
    finalizer._copy_composite_source(source, staging, plan)
    tail = finalizer._canonical_t2r1_tail(staging, destination)
    attempts = finalizer.V4.load_jsonl(staging / "generation_tail_attempts.T2.r1.jsonl")

    assert tail["retry_count"] == 2
    assert [row["ordinal"] for row in attempts] == [98, 99]
    assert [row["retry_sidecar_path"] for row in attempts] == [
        str(destination / "eval_sidecars/question_results.e8-v5-tail-t2-r1-o98.jsonl"),
        str(destination / "eval_sidecars/question_results.e8-v5-tail-t2-r1-o99.jsonl"),
    ]
    assert [row["retry_judge_trace_path"] for row in attempts] == [
        str(destination / "generation_tail_judge_traces/T2.r1.o98.jsonl"),
        str(destination / "generation_tail_judge_traces/T2.r1.o99.jsonl"),
    ]
    for ordinal in (98, 99):
        assert (
            staging / f"eval_sidecars/question_results.e8-v5-tail-t2-r1-o{ordinal}.jsonl"
        ).read_bytes() == (
            source / f"eval_sidecars/question_results.e8-v5-tail-t2-r1-o{ordinal}.jsonl"
        ).read_bytes()
        assert (staging / f"generation_tail_judge_traces/T2.r1.o{ordinal}.jsonl").read_bytes() == (
            source / f"generation_tail_judge_traces/T2.r1.o{ordinal}.jsonl"
        ).read_bytes()


def test_real_source_tail_rejects_broadened_ordinals(tmp_path: Path) -> None:
    source = Path(
        "/mnt/raid0/llm/epyc-root/artifacts/operator/"
        ".e8_quality_baseline_v5_partial_resume_promptfix_20260728.staging-"
        "b0d7ce62d6e04509a1cec7849aa68832"
    )
    staging = tmp_path / "staging"
    finalizer._copy_composite_source(source, staging, finalizer.build_plan(source))
    plan_path = staging / "partial_resume_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["generation_tail"]["targets"][1]["ordinal"] = 100
    _write_json(plan_path, plan)

    with pytest.raises(ValueError, match="tail targets"):
        finalizer._canonical_t2r1_tail(staging, tmp_path / "published")


def test_layered_context_is_limited_to_the_exact_composite_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_plan = finalizer.build_plan(validator.COMPOSITE_SOURCE_DIR)
    assert validator.canonical_hash(source_plan["source_sha256"]) == (
        validator.COMPOSITE_SOURCE_TREE_SHA256
    )
    source_plan_path = tmp_path / "recovery_finalizer_source_plan.json"
    _write_json(source_plan_path, source_plan)
    context = {
        "context": {
            "composite_source_plan_path": str(source_plan_path),
            "composite_source_plan_sha256": _sha(source_plan_path),
        }
    }
    validator.validate_composite_context(context, evidence_root=tmp_path)
    recycled_source = tmp_path / "recycled-source"
    monkeypatch.setattr(validator, "COMPOSITE_SOURCE_DIR", recycled_source)
    source_plan["source"] = str(recycled_source)
    for name, entry in source_plan["t2_r1_repair_history"].items():
        entry["path"] = str(recycled_source / name)
    _write_json(source_plan_path, source_plan)
    context["context"]["composite_source_plan_sha256"] = _sha(source_plan_path)
    validator.validate_composite_context(context, evidence_root=tmp_path)
    source_plan["source_sha256"] = {}
    _write_json(source_plan_path, source_plan)
    context["context"]["composite_source_plan_sha256"] = _sha(source_plan_path)
    with pytest.raises(ValueError, match="exact reviewed composite"):
        validator.validate_composite_context(context, evidence_root=tmp_path)


def test_composite_context_requires_both_recovery_layers() -> None:
    partial = {"partial": {"plan_sha256": validator.COMPOSITE_PARTIAL_RESUME_PLAN_SHA256}}
    ordinary_partial = {"partial": {"plan_sha256": "0" * 64}}
    ordinary_recovery = {"context": {}}
    composite_recovery = {
        "context": {
            "composite_source_plan_path": "/bundle/recovery_finalizer_source_plan.json",
            "composite_source_plan_sha256": "a" * 64,
        }
    }
    assert validator.composite_context_state(None, ordinary_recovery) is False
    assert validator.composite_context_state(ordinary_partial, None) is False
    assert validator.composite_context_state(partial, composite_recovery) is True
    with pytest.raises(ValueError, match="requires recovery-r2"):
        validator.composite_context_state(partial, None)
    with pytest.raises(ValueError, match="requires the partial-resume"):
        validator.composite_context_state(None, composite_recovery)
    with pytest.raises(ValueError, match="exact reviewed partial-resume"):
        validator.composite_context_state(ordinary_partial, composite_recovery)
    with pytest.raises(ValueError, match="require the composite source plan"):
        validator.composite_context_state(partial, ordinary_recovery)
    incomplete = {
        "context": {"composite_source_plan_path": "/bundle/recovery_finalizer_source_plan.json"}
    }
    with pytest.raises(ValueError, match="incomplete"):
        validator.composite_context_state(None, incomplete)


def test_four_monitor_segments_pin_the_source_resume_gap_and_order(tmp_path: Path) -> None:
    source = Path(
        "/mnt/raid0/llm/epyc-root/artifacts/operator/"
        ".e8_quality_baseline_v5_partial_resume_promptfix_20260728.staging-"
        "b0d7ce62d6e04509a1cec7849aa68832"
    )
    historical = finalizer.V4.load_jsonl(source / "historical_runtime_watch.jsonl")
    source_resume = finalizer.V4.load_jsonl(source / "resume_runtime_watch.jsonl")
    recovery = [dict(source_resume[0]), dict(source_resume[1])]
    recovery[0]["active_load"] = {"tier": 2, "repetition": 2}
    recovery[1]["active_load"] = None
    final_resume = [dict(source_resume[index]) for index in range(2)]
    for index, row in enumerate(final_resume):
        row["started_at"] = f"2026-07-28T00:00:{index * 5:02d}Z"
        row["active_load"] = {"tier": 2, "repetition": 3}
    paths = {
        "historical": tmp_path / "historical_runtime_watch.jsonl",
        "source_resume": tmp_path / "source_resume_runtime_watch.jsonl",
        "recovery_r2": tmp_path / "recovery_runtime_watch.jsonl",
        "resume": tmp_path / "resume_runtime_watch.jsonl",
    }
    for name, rows in (
        ("historical", historical),
        ("source_resume", source_resume),
        ("recovery_r2", recovery),
        ("resume", final_resume),
    ):
        if name == "historical":
            paths[name].write_bytes((source / "historical_runtime_watch.jsonl").read_bytes())
        elif name == "source_resume":
            paths[name].write_bytes((source / "resume_runtime_watch.jsonl").read_bytes())
        else:
            _write_jsonl(paths[name], rows)
    recovery_gap_count, recovery_max_gap = validator._monitor_gap_stats(recovery)
    final_gap_count, final_max_gap = validator._monitor_gap_stats(final_resume)
    starts = 0
    segments = []
    for name, rows, maximum, gap_count, binding in (
        (
            "historical",
            historical,
            validator.HISTORICAL_MAX_GAP_S,
            validator.HISTORICAL_EXPECTED_GAP_COUNT,
            validator.HISTORICAL_BINDING_SHA256,
        ),
        (
            "source_resume",
            source_resume,
            7.0,
            1,
            validator.SOURCE_RESUME_BINDING_SHA256,
        ),
        (
            "recovery_r2",
            recovery,
            7.0,
            recovery_gap_count,
            validator._monitor_binding_sha256(recovery[0]),
        ),
        (
            "resume",
            final_resume,
            7.0,
            final_gap_count,
            validator._monitor_binding_sha256(final_resume[0]),
        ),
    ):
        segment = {
            "source": name,
            "source_path": str(paths[name]),
            "source_sha256": _sha(paths[name]),
            "binding_sha256": binding,
            "sample_indexes": list(range(starts, starts + len(rows))),
            "max_gap_s": maximum,
            "observed_gap_count_over_7s": gap_count,
            "observed_max_gap_s": validator._monitor_gap_stats(rows)[1],
        }
        if name == "source_resume":
            segment["source_sha256"] = validator.SOURCE_RESUME_WATCHER_SHA256
            segment["observed_max_gap_s"] = validator.SOURCE_RESUME_MAX_GAP_S
            segment["pending_human_amendment"] = validator.source_resume_pending_amendment()
        segments.append(segment)
        starts += len(rows)
    samples = [*historical, *source_resume, *recovery, *final_resume]
    validator.validate_segmented_monitor(samples, segments, evidence_root=tmp_path)
    segments[1]["observed_max_gap_s"] = 7.0
    with pytest.raises(ValueError, match="source-resume"):
        validator.validate_segmented_monitor(samples, segments, evidence_root=tmp_path)


def test_install_recovered_r2_replaces_only_hash_bound_partial_source_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    snapshot = staging / "source_snapshot"
    recovered = tmp_path / "recovered"
    partial = {
        "eval_sidecars/question_results.e8-t2-r2.jsonl": b"partial-sidecar\n",
        "judge_traces.T2.r2.jsonl": b"partial-trace\n",
    }
    hashes: dict[str, str] = {}
    for relative, payload in partial.items():
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        (staging / relative).parent.mkdir(parents=True, exist_ok=True)
        (staging / relative).write_bytes(payload)
        hashes[relative] = _sha(path)
    _write_json(snapshot / "source_binding.json", {"source_sha256": hashes})
    _write_json(staging / "recovery_finalizer_source_plan.json", {"source_sha256": hashes})
    replacement = {
        "responses.T2.r2.jsonl": b"recovered-responses\n",
        "eval_sidecars/question_results.e8-t2-r2.jsonl": b"recovered-sidecar\n",
        "judge_traces.T2.r2.jsonl": b"recovered-trace\n",
        "raw.T2.r2.json": b"recovered-raw\n",
    }
    for relative, payload in replacement.items():
        path = recovered / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _write_json(staging / "question_vector.T2.json", {"core_id": "core"})
    _write_json(
        staging / "scoring_vector.T2.json",
        {"questions": [{"qid": f"q{ordinal}"} for ordinal in range(500)]},
    )
    monkeypatch.setattr(finalizer.RESUME, "_pristine_reference", lambda **_: {"artifacts": {}})
    monkeypatch.setattr(finalizer.RESUME, "_questions", lambda *_: [])
    monkeypatch.setattr(
        finalizer.RESUME,
        "_banked_observation_and_detail",
        lambda **_: (
            {"q": 1},
            {"scorer_tail_replay": ["old"], "scorer_sidecar_replacement_ordinals": [1]},
        ),
    )
    observation, detail = finalizer._install_recovered_r2(
        {"root": recovered, "plan": {"scorer_replay_ordinals": [1, 2, 3]}},
        staging,
        tmp_path / "published",
        object(),
    )
    assert observation == {"q": 1}
    assert detail["scorer_tail_replay"] == [
        {"ordinal": ordinal, "qid": f"q{ordinal}", "outcome": "recovered"} for ordinal in (1, 2, 3)
    ]
    assert all(
        (staging / relative).read_bytes() == payload for relative, payload in replacement.items()
    )


def test_install_recovered_r2_rejects_mutated_partial_source_file(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    snapshot = staging / "source_snapshot"
    relative = "eval_sidecars/question_results.e8-t2-r2.jsonl"
    source = snapshot / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("sealed\n")
    _write_json(snapshot / "source_binding.json", {"source_sha256": {relative: _sha(source)}})
    _write_json(
        staging / "recovery_finalizer_source_plan.json", {"source_sha256": {relative: _sha(source)}}
    )
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("tampered\n")
    for name in ("responses.T2.r2.jsonl", "judge_traces.T2.r2.jsonl", "raw.T2.r2.json"):
        path = tmp_path / "recovered" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new\n")
    (tmp_path / "recovered" / relative).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "recovered" / relative).write_text("new\n")
    with pytest.raises(ValueError, match="pre-existing partial r2 artifact"):
        finalizer._install_recovered_r2(
            {"root": tmp_path / "recovered"}, staging, tmp_path / "published", object()
        )


def test_validate_intermediate_rejects_lexical_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        finalizer.validate_intermediate(link)


def test_final_c1_intermediate_requires_complete_and_preserves_mixed_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "final-c1"
    (root / "predecessor_snapshot").mkdir(parents=True)
    mixed = {"schema": "mixed-tail"}
    (root / "predecessor_snapshot/partial_r2_plan.json").write_text(
        json.dumps({"mixed_tail_repair": mixed}) + "\n", encoding="utf-8"
    )
    plan = {"schema": finalizer.FINAL_C1_RETRY.PLAN_SCHEMA, "mixed_tail_repair": mixed}
    seen: list[bool] = []

    def validate_output(path: Path, *, require_complete: bool = False) -> dict:
        assert path == root
        seen.append(require_complete)
        return {"proposal": {"sealed": True}, "complete": {"status": "complete"}}

    monkeypatch.setattr(finalizer.FINAL_C1_RETRY, "validate_output", validate_output)
    validated = finalizer._validate_final_c1_intermediate(root, plan)
    assert seen == [True]
    assert validated["final_c1_retry"] is True
    assert validated["mixed_tail_repair"] == mixed


def test_final_c1_terminal_output_cannot_enter_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "terminal"
    (root / "predecessor_snapshot").mkdir(parents=True)

    def reject_terminal(_path: Path, *, require_complete: bool = False) -> dict:
        assert require_complete is True
        raise ValueError("terminal final-c1 output is not complete")

    monkeypatch.setattr(finalizer.FINAL_C1_RETRY, "validate_output", reject_terminal)
    with pytest.raises(ValueError, match="not complete"):
        finalizer._validate_final_c1_intermediate(
            root, {"schema": finalizer.FINAL_C1_RETRY.PLAN_SCHEMA}
        )
