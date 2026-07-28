#!/usr/bin/env python3
"""Seal a failed E8 v5 successor for deterministic tail repair.

This bridge is deliberately one-use and copy-only.  It accepts a fully
captured but unsealed v1 successor, corrects its root self-binding contract in
a fresh namespace, records every terminal non-clean row, and leaves inference
to the existing mixed-tail/race-retry chain.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RACE_PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_race_retry.py"
MIXED_PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_mixed_tail_repair.py"
RECOVERY_PATH = ROOT / "scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py"
SCHEMA = "epyc.e8_quality_v5_partial_r2_terminalization.v1"
TRANSITION_NAME = "terminalization_transition.json"
COMPLETION_NAME = "terminalization_complete.json"
REWRITTEN_SOURCE_PATHS = (
    "source_snapshot/source_binding.json",
    "partial_r2_plan.json",
    "recovery_proposal.json",
    "recovery_rows.T2.r2.jsonl",
)
SIDECARE_RELATIVE = "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl"
AT_FDCWD = -100
RENAME_NOREPLACE = 1


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RACE = _load(RACE_PATH, "e8_terminal_bridge_race")
MIXED = _load(MIXED_PATH, "e8_terminal_bridge_mixed")
RECOVERY = _load(RECOVERY_PATH, "e8_terminal_bridge_recovery")
V4 = RECOVERY.V4


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _root_bound_hashes(snapshot: Path) -> dict[str, str]:
    binding = snapshot / "source_binding.json"
    return {
        str(path.relative_to(snapshot)): sha256_path(path)
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path != binding
    }


def _assert_no_symlink_parents(path: Path) -> None:
    """Reject a destination whose existing ancestry can redirect publication."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if not current.exists():
            break
        if current.is_symlink():
            raise ValueError(f"terminal bridge destination has a symlink parent: {current}")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_durable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"terminal bridge path is not a real directory: {path}")
    _fsync_dir(path.parent)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    """Atomically replace a staging file after its bytes reach durable media."""
    _assert_no_symlink_parents(path.parent)
    _mkdir_durable(path.parent)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        V4._write_full_record(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Publish exactly once; never replace a path created by a concurrent actor."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("terminal bridge requires Linux renameat2(RENAME_NOREPLACE)")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"terminal bridge output namespace already exists: {destination}")
        raise OSError(error, os.strerror(error), destination)


def _write_completion_seal(output: Path, transition: dict[str, Any]) -> None:
    """Create the sole consumption marker after the directory publication is durable."""
    path = output / COMPLETION_NAME
    if path.exists() or path.is_symlink():
        raise FileExistsError("terminal bridge completion seal already exists")
    value = {
        "schema": SCHEMA,
        "status": "published_complete",
        "transition": {
            "path": TRANSITION_NAME,
            "sha256": sha256_path(output / TRANSITION_NAME),
        },
        "terminalizer_runner": transition["terminalizer_runner"],
        "source_tree_sha256": transition["source_tree_sha256"],
        "output_payload_tree_sha256": transition["output_payload_tree_sha256"],
    }
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        V4._write_full_record(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    linked = False
    try:
        os.link(temporary, path)
        linked = True
        _fsync_dir(output)
    except Exception:
        if linked:
            path.unlink(missing_ok=True)
            try:
                _fsync_dir(output)
            except Exception:
                pass
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _load_binding(snapshot: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    binding = snapshot / "source_binding.json"
    if not binding.is_file() or binding.is_symlink():
        raise ValueError("terminal bridge snapshot lacks a real source binding")
    data = V4.load_json(binding)
    declared = data.get("source_sha256")
    if not isinstance(declared, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in declared.items()
    ):
        raise ValueError("terminal bridge source binding is malformed")
    return data, declared, _root_bound_hashes(snapshot)


def _validate_input(source: Path, expected_tree: str) -> dict[str, Any]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("terminal bridge source must be a real directory")
    if canonical_hash(RACE.source_hashes(source)) != expected_tree:
        raise ValueError("terminal bridge source differs from explicit full tree hash")
    if (source / "r2_complete.json").exists():
        raise ValueError("terminal bridge accepts only an unsealed failed successor")
    required = (
        "partial_r2_plan.json", "recovery_proposal.json", "recovery_rows.T2.r2.jsonl",
        "runtime_watch.r2.successor.jsonl", "scorer_attempts.T2.r2.jsonl",
        "generation_judge_traces.T2.r2.jsonl", "scorer_replay_traces.T2.r2.jsonl",
        "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl",
        "source_snapshot/source_binding.json", "failed_source_snapshot/source_binding.json",
    )
    if any(not (source / relative).is_file() or (source / relative).is_symlink() for relative in required):
        raise ValueError("terminal bridge source lacks required successor evidence")
    plan = V4.load_json(source / "partial_r2_plan.json")
    proposal = V4.load_json(source / "recovery_proposal.json")
    base_data, base_declared, base_actual = _load_binding(source / "source_snapshot")
    failed_data, failed_declared, failed_actual = _load_binding(source / "failed_source_snapshot")
    if (
        set(base_declared) - set(base_actual) != {"source_binding.json"}
        or set(base_actual) - set(base_declared)
        or any(base_declared[key] != value for key, value in base_actual.items())
        or base_data.get("source_tree_sha256") != canonical_hash(base_declared)
        or plan.get("source_sha256") != base_declared
        or plan.get("source_tree_sha256") != canonical_hash(base_declared)
        or proposal.get("source_tree_sha256") != plan.get("source_tree_sha256")
    ):
        raise ValueError("terminal bridge input is not the exact root-self-binding mismatch")
    if (
        failed_declared != failed_actual
        or failed_data.get("source_tree_sha256") != canonical_hash(failed_declared)
        or plan.get("failed_source_sha256") != failed_declared
        or plan.get("failed_source_tree_sha256") != canonical_hash(failed_declared)
        or proposal.get("failed_source_tree_sha256") != plan.get("failed_source_tree_sha256")
    ):
        raise ValueError("terminal bridge failed-source binding differs")
    questions = V4.load_json(source / "source_snapshot/scoring_vector.T2.json").get("questions")
    generation = plan.get("generation_ordinals")
    if (
        not isinstance(questions, list) or len(questions) != MIXED.N
        or not isinstance(generation, list)
        or any(type(ordinal) is not int for ordinal in generation)
    ):
        raise ValueError("terminal bridge source plan/vector is invalid")
    _lines, sidecars = MIXED._rows_with_bytes(
        source / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl"
    )
    if set(sidecars) != set(generation):
        raise ValueError("terminal bridge requires one saved sidecar per generation ordinal")
    kinds = {ordinal: MIXED._classify(sidecars[ordinal][1], questions[ordinal]) for ordinal in generation}
    if set(kinds.values()) - set(MIXED.ALLOWED_CLASSES):
        raise ValueError("terminal bridge found an unapproved successor failure class")
    rows = RECOVERY._load_journal(source / "recovery_rows.T2.r2.jsonl")
    if any(ordinal in rows for ordinal, kind in kinds.items() if kind == "clean"):
        raise ValueError("terminal bridge source unexpectedly records clean generation rows")
    return {
        "plan": plan,
        "proposal": proposal,
        "base_declared": base_declared,
        "base_actual": base_actual,
        "failed_declared": failed_declared,
        "sidecars": sidecars,
        "questions": questions,
        "kinds": kinds,
        "rows": rows,
    }


def _copy_tree(source: Path, destination: Path, source_sha256: dict[str, str]) -> None:
    _mkdir_durable(destination)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            raise ValueError("terminal bridge source contains a symlink")
        if path.is_dir():
            _mkdir_durable(target)
        elif path.is_file():
            contents = path.read_bytes()
            digest = hashlib.sha256(contents).hexdigest()
            if source_sha256.get(str(relative)) != digest:
                raise ValueError("terminal bridge source changed while copying")
            _write_bytes_atomic(target, contents)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes_atomic(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _append_clean_journal(
    output: Path,
    rows: dict[int, dict[str, Any]],
    sidecars: dict[int, tuple[int, dict[str, Any]]],
    questions: list[dict[str, Any]],
    kinds: dict[int, str],
) -> list[int]:
    journal = output / "recovery_rows.T2.r2.jsonl"
    appended: list[int] = []
    for ordinal in sorted(ordinal for ordinal, kind in kinds.items() if kind == "clean"):
        response = RECOVERY._response_from_sidecar(sidecars[ordinal][1], questions[ordinal])
        RECOVERY._record(journal, rows, ordinal, response, "generation")
        appended.append(ordinal)
    return appended


def _write_failure_ledger(
    output: Path,
    sidecars: dict[int, tuple[int, dict[str, Any]]],
    kinds: dict[int, str],
) -> dict[str, Any]:
    failures = MIXED._terminal_failures(sidecars, kinds)
    ledger = {
        "failures": failures,
        "disposition": "failed_closed_no_automatic_retry",
    }
    path = output / "generation_failed_attempts.T2.r2.jsonl"
    _write_bytes_atomic(path, (json.dumps(ledger, sort_keys=True) + "\n").encode())
    return {"path": path.name, "sha256": sha256_path(path), "failures": failures}


def terminalize(source_dir: Path, output_dir: Path, expected_source_tree_sha256: str) -> Path:
    source = source_dir.resolve(strict=True)
    output = output_dir.absolute()
    _assert_no_symlink_parents(output.parent)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"terminal bridge output namespace already exists: {output}")
    if output == source or source in output.parents:
        raise ValueError("terminal bridge output must not overlap source")
    validated = _validate_input(source, expected_source_tree_sha256)
    source_sha256 = RACE.source_hashes(source)
    staging = output.with_name(f".{output.name}.terminalizing-{uuid.uuid4().hex}")
    published = False
    try:
        _copy_tree(source, staging, source_sha256)
        plan = V4.load_json(staging / "partial_r2_plan.json")
        proposal = V4.load_json(staging / "recovery_proposal.json")
        base_binding = staging / "source_snapshot/source_binding.json"
        before = source_sha256["source_snapshot/source_binding.json"]
        corrected = validated["base_actual"]
        _write_json(base_binding, {"source_sha256": corrected, "source_tree_sha256": canonical_hash(corrected)})
        plan["source_sha256"] = corrected
        plan["source_tree_sha256"] = canonical_hash(corrected)
        proposal["source_tree_sha256"] = plan["source_tree_sha256"]
        _write_json(staging / "partial_r2_plan.json", plan)
        _write_json(staging / "recovery_proposal.json", proposal)
        rows = RECOVERY._load_journal(staging / "recovery_rows.T2.r2.jsonl")
        appended = _append_clean_journal(staging, rows, validated["sidecars"], validated["questions"], validated["kinds"])
        ledger = _write_failure_ledger(staging, validated["sidecars"], validated["kinds"])
        rewritten = {
            relative: {
                "before_sha256": source_sha256[relative],
                "after_sha256": sha256_path(staging / relative),
            }
            for relative in REWRITTEN_SOURCE_PATHS
        }
        unchanged = {
            relative: digest
            for relative, digest in source_sha256.items()
            if relative not in REWRITTEN_SOURCE_PATHS
        }
        output_before_transition = RACE.source_hashes(staging)
        output_unchanged = {
            relative: output_before_transition[relative]
            for relative in unchanged
        }
        if output_unchanged != unchanged:
            raise ValueError("terminal bridge changed an unlisted copied source artifact")
        if set(output_before_transition) != set(source_sha256) | {ledger["path"]}:
            raise ValueError("terminal bridge output has an unlisted source artifact before transition")
        transition = {
            "schema": SCHEMA,
            "status": "terminal_failed",
            "source_tree_sha256": expected_source_tree_sha256,
            "source_sha256": source_sha256,
            "terminalizer_runner": {
                "path": Path(__file__).name,
                "sha256": sha256_path(Path(__file__)),
            },
            "rewritten_artifacts": rewritten,
            "unchanged_copied_sha256": unchanged,
            "unchanged_copied_tree_sha256": canonical_hash(unchanged),
            "output_payload_sha256": output_before_transition,
            "output_payload_tree_sha256": canonical_hash(output_before_transition),
            "root_self_binding_correction": {
                "snapshot": "source_snapshot",
                "before_binding_sha256": before,
                "after_binding_sha256": sha256_path(base_binding),
                "declared_entries_before": len(validated["base_declared"]),
                "content_entries_after": len(corrected),
                "excluded_path": "source_binding.json",
                "nested_bindings_retained": sorted(key for key in corrected if key.endswith("source_binding.json")),
            },
            "saved_sidecar_byte_preservation": {
                "path": SIDECARE_RELATIVE,
                "source_sha256": source_sha256[SIDECARE_RELATIVE],
                "output_sha256": sha256_path(staging / SIDECARE_RELATIVE),
            },
            "journal": {
                "path": "recovery_rows.T2.r2.jsonl",
                "clean_generation_ordinals": appended,
                "appended_count": len(appended),
                "before_sha256": source_sha256["recovery_rows.T2.r2.jsonl"],
                "before_byte_length": (source / "recovery_rows.T2.r2.jsonl").stat().st_size,
                "after_sha256": sha256_path(staging / "recovery_rows.T2.r2.jsonl"),
            },
            "failure_ledger": ledger,
            "classified_ordinals": {
                kind: sorted(ordinal for ordinal, value in validated["kinds"].items() if value == kind)
                for kind in MIXED.ALLOWED_CLASSES
            },
        }
        if transition["saved_sidecar_byte_preservation"]["source_sha256"] != transition["saved_sidecar_byte_preservation"]["output_sha256"]:
            raise ValueError("terminal bridge altered a saved sidecar byte")
        _write_json(staging / TRANSITION_NAME, transition)
        if RACE.source_hashes(source) != source_sha256:
            raise ValueError("terminal bridge source changed before publication")
        terminal_hash = canonical_hash(RACE.source_hashes(staging))
        # Validate the unpublished staging payload, but never let a consumer
        # use it: only publication creates the completion seal.
        MIXED._validate_predecessor(staging, terminal_hash, require_completion=False)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"terminal bridge output namespace already exists: {output}")
        _rename_noreplace(staging, output)
        published = True
        try:
            _fsync_dir(output.parent)
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            try:
                _fsync_dir(output.parent)
            except Exception:
                pass
            raise
        _write_completion_seal(output, transition)
        return output
    except Exception:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-tree-sha256", required=True)
    args = parser.parse_args(argv)
    print(terminalize(args.source_dir, args.output_dir, args.expected_source_tree_sha256))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
