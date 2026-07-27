#!/usr/bin/env python3
"""Repair terminal E8 v4 generation failures without rerunning a repetition.

The source bundle is immutable.  A terminalizer must write ``terminal_source.json``
after every T1/T2 repetition has completed; this tool deliberately refuses a live or
partial staging directory.  It never writes baseline state or an apply receipt.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = PROJECT_ROOT / "scripts/benchmark/run_e8_quality_baseline_reseed.py"
TERMINAL_SCHEMA = "epyc.e8_quality_terminal_source.v1"
PLAN_SCHEMA = "epyc.e8_quality_multirow_repair_plan.v1"
SEAL_SCHEMA = "epyc.e8_quality_multirow_repair_seal.v1"
FOCUSED_SEAL_SCHEMA = "epyc.e8_quality_focused_collection_seal.v1"
REPETITIONS = 3
REQUEST_TIMEOUT_S = 300


def _runner() -> Any:
    spec = importlib.util.spec_from_file_location("e8_reseed_repair_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned E8 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _runner()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(row)
    return rows


def _write_create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Durably append one focused result so later failures retain prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise OSError("focused attempt write made no progress")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_create(path: Path, value: Any) -> None:
    _write_create(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def write_jsonl_create(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_create(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode())


def _reject_link_or_alias(path: Path, *, seen: set[tuple[int, int]] | None = None) -> None:
    if path.is_symlink():
        raise ValueError(f"symlink is not permitted: {path}")
    stat = path.stat(follow_symlinks=False)
    if stat.st_nlink != 1:
        raise ValueError(f"hardlink is not permitted: {path}")
    identity = (stat.st_dev, stat.st_ino)
    if seen is not None:
        if identity in seen:
            raise ValueError(f"source aliases another artifact: {path}")
        seen.add(identity)


def _reject_overlap(*paths: Path) -> None:
    resolved = [path.absolute() for path in paths]
    for index, left in enumerate(resolved):
        for right in resolved[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("source, focused, staging, and destination paths must not overlap")


def _copy_file_create(source: Path, destination: Path) -> None:
    _reject_link_or_alias(source)
    _write_create(destination, source.read_bytes())
    if sha256_path(source) != sha256_path(destination):
        raise ValueError(f"copy differs from source: {source}")


@dataclass(frozen=True)
class RunFiles:
    tier: int
    repetition: int
    raw: Path
    responses: Path
    judge_traces: Path
    sidecar: Path


@dataclass(frozen=True)
class RepairRow:
    tier: int
    repetition: int
    ordinal: int
    qid: str
    fingerprint: str
    reasons: tuple[str, ...]
    source_response_sha256: str
    source_sidecar_sha256: str


@dataclass(frozen=True)
class SourceBundle:
    root: Path
    manifest: dict[str, Any]
    vectors: dict[int, dict[str, Any]]
    scoring_vectors: dict[int, dict[str, Any]]
    runs: tuple[RunFiles, ...]


def _required_paths(root: Path, manifest: dict[str, Any] | None = None) -> list[Path]:
    if manifest is not None and isinstance(manifest.get("source_artifacts"), list):
        paths = [root / "terminal_source.json"]
        for value in manifest["source_artifacts"]:
            if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError("terminal source artifact path is invalid")
            paths.append(root / value)
        return paths
    paths = [root / f"{kind}_vector.T{tier}.json" for tier in (1, 2) for kind in ("question", "scoring")]
    paths += [root / "runtime_watch.jsonl", root / "terminal_source.json"]
    if manifest is not None and isinstance(manifest.get("runs"), list):
        for row in manifest["runs"]:
            if not isinstance(row, dict):
                raise ValueError("terminal source run mapping is invalid")
            for key in ("raw", "responses", "judge_traces", "sidecar"):
                value = row.get(key)
                if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
                    raise ValueError("terminal source run path is invalid")
                paths.append(root / value)
        return paths
    for tier in (1, 2):
        for repetition in range(1, REPETITIONS + 1):
            paths.extend((
                root / f"raw.T{tier}.r{repetition}.json",
                root / f"responses.T{tier}.r{repetition}.jsonl",
                root / f"judge_traces.T{tier}.r{repetition}.jsonl",
                root / "eval_sidecars" / f"question_results.e8-t{tier}-r{repetition}.jsonl",
            ))
    return paths


def _run_files(root: Path, manifest: dict[str, Any] | None = None) -> tuple[RunFiles, ...]:
    if manifest is not None and isinstance(manifest.get("runs"), list):
        runs: list[RunFiles] = []
        for row in manifest["runs"]:
            if not isinstance(row, dict) or not isinstance(row.get("tier"), int) or not isinstance(row.get("repetition"), int):
                raise ValueError("terminal source run identity is invalid")
            runs.append(RunFiles(
                row["tier"], row["repetition"], *(root / str(row[key]) for key in ("raw", "responses", "judge_traces", "sidecar"))
            ))
        if sorted((run.tier, run.repetition) for run in runs) != [(tier, repetition) for tier in (1, 2) for repetition in range(1, 4)]:
            raise ValueError("terminal source must map every T1/T2 repetition exactly once")
        return tuple(sorted(runs, key=lambda run: (run.tier, run.repetition)))
    return tuple(
        RunFiles(
            tier, repetition, root / f"raw.T{tier}.r{repetition}.json",
            root / f"responses.T{tier}.r{repetition}.jsonl",
            root / f"judge_traces.T{tier}.r{repetition}.jsonl",
            root / "eval_sidecars" / f"question_results.e8-t{tier}-r{repetition}.jsonl",
        )
        for tier in (1, 2) for repetition in range(1, REPETITIONS + 1)
    )


def _source_hashes(root: Path, manifest: dict[str, Any]) -> None:
    declared = manifest.get("source_sha256")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("terminal source has no source_sha256 ledger")
    required = _required_paths(root, manifest)
    expected_keys = {str(path.relative_to(root)) for path in required if path.name != "terminal_source.json"}
    if set(declared) != expected_keys:
        raise ValueError("terminal source hash key set differs from required artifact set")
    seen: set[tuple[int, int]] = set()
    for path in required:
        if path.name == "terminal_source.json":
            continue
        _reject_link_or_alias(path, seen=seen)
        relative = str(path.relative_to(root))
        expected = declared.get(relative)
        if not isinstance(expected, str) or sha256_path(path) != expected:
            raise ValueError(f"terminal source hash differs for {relative}")


def _vector_qids(vector: dict[str, Any], *, tier: int) -> list[str]:
    n = 50 if tier == 1 else 500
    if vector.get("tier") != tier or vector.get("n") != n or not isinstance(vector.get("questions"), list):
        raise ValueError(f"T{tier} vector is not the E8 fixed n={n} vector")
    qids = [str(row.get("qid") or "") for row in vector["questions"] if isinstance(row, dict)]
    if len(qids) != n or any(not qid for qid in qids) or len(set(qids)) != n:
        raise ValueError(f"T{tier} fixed vector qids are invalid")
    return qids


def _sidecar_by_ordinal(path: Path, qids: list[str]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    rows = load_jsonl(path)
    starts = [row for row in rows if row.get("row_type") == "batch_start"]
    terminals = [row for row in rows if row.get("row_type") in {"batch_complete", "batch_end"}]
    if len(starts) != 1 or starts[0].get("requested_n") != len(qids):
        raise ValueError(f"{path} has no exact batch start")
    if len(terminals) != 1 or terminals[0].get("complete") is not True:
        raise ValueError(f"{path} has no complete terminal marker")
    for row in rows:
        if row.get("row_type") != "question_result":
            continue
        ordinal = row.get("ordinal")
        payload = row.get("result")
        if not isinstance(ordinal, int) or not isinstance(payload, dict):
            raise ValueError(f"{path} has malformed result row")
        if ordinal in result or ordinal < 0 or ordinal >= len(qids):
            raise ValueError(f"{path} has invalid ordinal")
        if str(payload.get("qid") or payload.get("question_id") or "") != qids[ordinal]:
            raise ValueError(f"{path} qid order differs at ordinal {ordinal}")
        result[ordinal] = row
    if len(result) != len(qids):
        raise ValueError(f"{path} does not contain every fixed-vector result")
    return result


def validate_source(source_dir: Path) -> SourceBundle:
    if source_dir.is_symlink():
        raise ValueError("source directory symlink is not permitted")
    root = source_dir.absolute()
    if not root.is_dir():
        raise ValueError("source directory does not exist")
    required_base = [root / f"{kind}_vector.T{tier}.json" for tier in (1, 2) for kind in ("question", "scoring")]
    required_base += [root / "runtime_watch.jsonl", root / "terminal_source.json"]
    missing = [str(path.relative_to(root)) for path in required_base if not path.is_file()]
    if missing:
        raise ValueError("terminal source is incomplete: " + ", ".join(missing))
    manifest = load_json(root / "terminal_source.json")
    if manifest.get("schema") != TERMINAL_SCHEMA or manifest.get("status") != "terminal_failed":
        raise ValueError("source is not a terminal failed E8 v4 bundle")
    protocol = manifest.get("protocol")
    if manifest.get("protocol_id") != RUNNER.PROTOCOL_ID or not isinstance(manifest.get("runtime_binding"), dict):
        raise ValueError("terminal source protocol or runtime binding is missing")
    if (
        not isinstance(protocol, dict)
        or protocol.get("protocol_id") != RUNNER.PROTOCOL_ID
        or protocol.get("repetitions") != REPETITIONS
        or protocol.get("request_timeout_s") != REQUEST_TIMEOUT_S
        or protocol.get("frontdoor_request_contract") != RUNNER.FRONTDOOR_REQUEST_CONTRACT
        or protocol.get("runtime_binding") != manifest["runtime_binding"]
    ):
        raise ValueError("terminal source protocol contract differs from E8 v4")
    missing = [str(path.relative_to(root)) for path in _required_paths(root, manifest) if not path.is_file()]
    if missing:
        raise ValueError("terminal source is incomplete: " + ", ".join(missing))
    _source_hashes(root, manifest)
    if "runner_report.json" in manifest.get("source_artifacts", []) and "run_seal.json" in manifest.get("source_artifacts", []):
        report, seal = load_json(root / "runner_report.json"), load_json(root / "run_seal.json")
        if report.get("decision_grade") is True or seal.get("status") != "failed":
            raise ValueError("terminal source runner report or seal is not failed")
    vectors = {tier: load_json(root / f"question_vector.T{tier}.json") for tier in (1, 2)}
    scoring = {tier: load_json(root / f"scoring_vector.T{tier}.json") for tier in (1, 2)}
    for tier in (1, 2):
        qids = _vector_qids(vectors[tier], tier=tier)
        score_qids = _vector_qids(scoring[tier], tier=tier)
        if qids != score_qids:
            raise ValueError(f"T{tier} question and scoring vector order differs")
        declared_vector = manifest.get("vector_sha256", {}).get(str(tier))
        if declared_vector != canonical_hash(vectors[tier]):
            raise ValueError(f"T{tier} vector hash differs from terminal source")
    watcher = load_jsonl(root / "runtime_watch.jsonl")
    if not watcher or any(row.get("ok") is not True for row in watcher):
        raise ValueError("terminal source runtime watcher is not clean")
    runs = _run_files(root, manifest)
    for run in runs:
        qids = _vector_qids(vectors[run.tier], tier=run.tier)
        responses = load_jsonl(run.responses)
        if len(responses) != len(qids) or [str(row.get("qid") or "") for row in responses] != qids:
            raise ValueError(f"T{run.tier}/r{run.repetition} response qid order differs")
        _sidecar_by_ordinal(run.sidecar, qids)
        raw = load_json(run.raw)
        if raw.get("n") != len(qids) or raw.get("protocol_id") != RUNNER.PROTOCOL_ID:
            raise ValueError(f"T{run.tier}/r{run.repetition} raw record differs from source contract")
    return SourceBundle(root, manifest, vectors, scoring, runs)


def _terminal_run_files(root: Path) -> tuple[RunFiles, ...]:
    """Resolve normal rows and the pinned runner's migrated T1/r1 layout."""
    runs: list[RunFiles] = []
    for tier in (1, 2):
        for repetition in range(1, REPETITIONS + 1):
            standard = RunFiles(
                tier, repetition, root / f"raw.T{tier}.r{repetition}.json",
                root / f"responses.T{tier}.r{repetition}.jsonl",
                root / f"judge_traces.T{tier}.r{repetition}.jsonl",
                root / "eval_sidecars" / f"question_results.e8-t{tier}-r{repetition}.jsonl",
            )
            if tier == 1 and repetition == 1 and not standard.responses.is_file():
                migrated = root / "migration.T1.r1"
                standard = RunFiles(
                    tier, repetition, root / "raw.T1.r1.json", migrated / "responses.T1.r1.jsonl",
                    migrated / "judge_traces.T1.r1.jsonl", migrated / "legacy_question_results.T1.r1.jsonl",
                )
            runs.append(standard)
    return tuple(runs)


def terminalize_source(staging_dir: Path, destination: Path) -> Path:
    """Copy a terminal failed runner staging tree into an immutable repair input."""
    if staging_dir.is_symlink() or destination.is_symlink():
        raise ValueError("staging and destination symlinks are not permitted")
    source = staging_dir.absolute()
    _reject_overlap(source, destination.absolute())
    if destination.exists():
        raise FileExistsError(f"terminal destination already exists: {destination}")
    report_path, seal_path = source / "runner_report.json", source / "run_seal.json"
    required = [
        source / f"{kind}_vector.T{tier}.json" for tier in (1, 2) for kind in ("question", "scoring")
    ] + [source / "runtime_watch.jsonl", report_path, seal_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("staging is not terminal: " + ", ".join(missing))
    report, seal = load_json(report_path), load_json(seal_path)
    if report.get("protocol_id") != RUNNER.PROTOCOL_ID or report.get("mode") not in {"executed", "failed"}:
        raise ValueError("staging report is not an E8 terminal run")
    if seal.get("status") != "failed" or report.get("decision_grade") is True:
        raise ValueError("terminalizer accepts only a failed source run")
    protocol = report.get("protocol") or report.get("protocol_candidate", {}).get("protocol")
    binding = report.get("postconditions", {}).get("runtime_binding") or report.get("preconditions", {}).get("runtime_binding")
    if not isinstance(protocol, dict) or not isinstance(binding, dict):
        raise ValueError("staging report lacks protocol or runtime binding")
    runs = _terminal_run_files(source)
    required.extend(path for run in runs for path in (run.raw, run.responses, run.judge_traces, run.sidecar))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("staging is incomplete: " + ", ".join(missing))
    vectors = {tier: load_json(source / f"question_vector.T{tier}.json") for tier in (1, 2)}
    for run in runs:
        qids = _vector_qids(vectors[run.tier], tier=run.tier)
        if [str(row.get("qid") or "") for row in load_jsonl(run.responses)] != qids:
            raise ValueError(f"staging T{run.tier}/r{run.repetition} response qid order differs")
        _sidecar_by_ordinal(run.sidecar, qids)
    watcher = load_jsonl(source / "runtime_watch.jsonl")
    if not watcher or any(row.get("ok") is not True for row in watcher):
        raise ValueError("staging runtime watcher is not clean")
    publish_dir = destination.with_name(f".{destination.name}.terminalizing-{uuid.uuid4().hex}")
    publish_dir.mkdir(mode=0o700)
    artifacts = sorted({str(path.relative_to(source)) for path in required})
    try:
        for relative in artifacts:
            _copy_file_create(source / relative, publish_dir / relative)
        copied_runs = [
            {"tier": run.tier, "repetition": run.repetition,
             **{name: str(getattr(run, name).relative_to(source)) for name in ("raw", "responses", "judge_traces", "sidecar")}}
            for run in runs
        ]
        hashes = {relative: sha256_path(publish_dir / relative) for relative in artifacts}
        manifest = {
            "schema": TERMINAL_SCHEMA, "status": "terminal_failed", "protocol_id": RUNNER.PROTOCOL_ID,
            "protocol": protocol, "runtime_binding": binding, "runs": copied_runs,
            "source_artifacts": artifacts, "source_sha256": hashes,
            "vector_sha256": {str(tier): canonical_hash(vectors[tier]) for tier in (1, 2)},
            "source_staging_sha256": canonical_hash({relative: sha256_path(source / relative) for relative in artifacts}),
        }
        write_json_create(publish_dir / "terminal_source.json", manifest)
        validate_source(publish_dir)
        _atomic_publish_noreplace(publish_dir, destination)
    except Exception:
        # Do not erase forensic terminalization evidence.
        raise
    return destination


def _accepted_generation_error(value: str) -> bool:
    return value in {
        "[ERROR: Inference failed: chat_completions failed: timed out]",
        "[ERROR: Inference failed: chat_completions failed: timeout]",
    }


def _generation_reasons(response: dict[str, Any], sidecar: dict[str, Any], question: dict[str, Any]) -> tuple[str, ...]:
    """Classify only an explicit infrastructure generation failure."""
    result = sidecar["result"]
    answer = str(response.get("answer") or "")
    method = str(question.get("scoring_method") or "")
    error = str(result.get("error_detail") or "")
    if method == "llm_judge" and answer.strip() and str(response.get("error") or error).startswith("scoring_unavailable:"):
        return ()
    # The admitted sentinel is transport evidence, not generated content.
    blank_or_sentinel = not answer.strip() or answer == error
    if (
        result.get("tokens_generated") == 0
        and blank_or_sentinel
        and result.get("error") is True
        and _accepted_generation_error(error)
    ):
        return ("infrastructure_generation_timeout",)
    return ()


def build_plan(bundle: SourceBundle) -> dict[str, Any]:
    targets: list[RepairRow] = []
    for run in bundle.runs:
        questions = bundle.scoring_vectors[run.tier]["questions"]
        sidecars = _sidecar_by_ordinal(run.sidecar, _vector_qids(bundle.vectors[run.tier], tier=run.tier))
        for ordinal, (response, question) in enumerate(zip(load_jsonl(run.responses), questions)):
            reasons = _generation_reasons(response, sidecars[ordinal], question)
            if reasons:
                fingerprint = canonical_hash({"response": response, "sidecar_result": sidecars[ordinal]["result"], "reasons": reasons})
                targets.append(RepairRow(run.tier, run.repetition, ordinal, str(response["qid"]), fingerprint, reasons, sha256_path(run.responses), sha256_path(run.sidecar)))
    targets.sort(key=lambda row: (row.tier, row.repetition, row.ordinal, row.qid))
    return {
        "schema": PLAN_SCHEMA,
        "source_dir": str(bundle.root),
        "source_terminal_sha256": sha256_path(bundle.root / "terminal_source.json"),
        "source_sha256": dict(sorted(bundle.manifest["source_sha256"].items())),
        "runtime_binding": bundle.manifest["runtime_binding"],
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "targets": [row.__dict__ | {"reasons": list(row.reasons)} for row in targets],
        "target_count": len(targets),
    }


def write_plan(plan: dict[str, Any], destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"plan already exists: {destination}")
    write_json_create(destination, plan)


@contextmanager
def _focused_environment(sidecar_dir: Path, api_url: str):
    """Keep the runner's scoring contract while forcing a single request lane."""
    with RUNNER.fixed_baseline_environment(sidecar_dir, api_url):
        old = os.environ.get("AUTOPILOT_EVAL_CONCURRENCY")
        os.environ["AUTOPILOT_EVAL_CONCURRENCY"] = "1"
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("AUTOPILOT_EVAL_CONCURRENCY", None)
            else:
                os.environ["AUTOPILOT_EVAL_CONCURRENCY"] = old


def _reconstruct_questions(bundle: SourceBundle, runner_args: argparse.Namespace) -> dict[int, list[dict[str, Any]]]:
    tower = RUNNER.EvalTower(url=runner_args.api_url.rstrip("/"), timeout=REQUEST_TIMEOUT_S)
    result: dict[int, list[dict[str, Any]]] = {}
    for tier in (1, 2):
        vector = bundle.vectors[tier]
        questions, core_id = RUNNER.question_vector(
            tower,
            tier=tier,
            t1_core_id=str(vector["core_id"]) if tier == 1 else runner_args.t1_core_id,
            n=vector["n"],
            seed=vector["seed"],
        )
        questions = RUNNER.apply_context_replacement_map(runner_args, questions, tier=tier)
        public = RUNNER.public_vector(questions, tier=tier, core_id=core_id, seed=vector["seed"])
        scoring = RUNNER.scoring_vector(questions, tier=tier, core_id=core_id, seed=vector["seed"])
        if canonical_hash(public) != canonical_hash(vector) or canonical_hash(scoring) != canonical_hash(bundle.scoring_vectors[tier]):
            raise ValueError(f"T{tier} reconstructed question/scoring contract differs from terminal source")
        result[tier] = questions
    return result


def collect_focused(source: Path, output: Path, *, api_url: str) -> Path:
    """Run exactly one 300-second direct-frontdoor generation per planned row.

    A failed probe is terminal.  The partially written output is deliberately
    retained as failure evidence and is never promoted to a repair candidate.
    """
    bundle = validate_source(source)
    plan = build_plan(bundle)
    if output.exists():
        raise FileExistsError(f"collection output already exists: {output}")
    output.mkdir(mode=0o700)
    write_json_create(output / "repair_plan.json", plan)
    watcher: Any | None = None
    attempts_path = output / "focused_attempts.jsonl"
    failure: Exception | None = None
    try:
        runner_args = RUNNER.parse_args(["--protocol-proposal", "--api-url", api_url, "--evaltower-timeout-s", "300"])
        questions = _reconstruct_questions(bundle, runner_args)
        watcher = RUNNER.RuntimeWatcher(
            runner_args, bundle.manifest["runtime_binding"], output / "focused_runtime_watch.jsonl", include_receipt=False
        )
        watcher.start()
        RUNNER.require_clean_watcher(watcher)
        tower = RUNNER.EvalTower(url=api_url.rstrip("/"), timeout=REQUEST_TIMEOUT_S)
        tower._question_artifact_dir = output / "eval_sidecars"
        for target in plan["targets"]:
            tier, repetition, ordinal = target["tier"], target["repetition"], target["ordinal"]
            question = dict(questions[tier][ordinal])
            question.update({"qid": target["qid"], **RUNNER.FRONTDOOR_REQUEST_CONTRACT})
            trace = output / "focused_judge_traces" / f"T{tier}.r{repetition}.o{ordinal}.jsonl"
            _write_create(trace, b"")
            with watcher.active_load(tier=tier, repetition=repetition):
                with (
                    RUNNER.httpx.Client(timeout=REQUEST_TIMEOUT_S) as client,
                    _focused_environment(output / "eval_sidecars", api_url),
                    RUNNER.capture_llm_judge_traces(trace, default_api_url=api_url),
                    RUNNER.bind_eval_tower_scorer_identities(tower),
                ):
                    results = tower._eval_batch([question], client, log_every=1, label=f"e8-focused-t{tier}-r{repetition}-o{ordinal}")
                    RUNNER.replay_llm_judge_scorer_tail_once(results, [question])
            if len(results) != 1:
                raise ValueError("focused generation did not return exactly one row")
            response = RUNNER.response_rows(results, [question])[0]
            result = results[0]
            if (
                response.get("error") is not None or not str(response.get("answer") or "").strip()
                or response.get("route_used") != "frontdoor" or int(getattr(result, "eval_concurrency", 0)) != 1
            ):
                raise RuntimeError(f"focused probe failed closed for T{tier}/r{repetition}/o{ordinal}")
            response.update({
                "tier": tier,
                "repetition": repetition,
                "ordinal": ordinal,
                "request_timeout_s": REQUEST_TIMEOUT_S,
                "focused": True,
                "score_terminal": True,
                "source_terminal_sha256": plan["source_terminal_sha256"],
                "repair_plan_sha256": sha256_path(output / "repair_plan.json"),
                "failure_fingerprint": target["fingerprint"],
                "eval_concurrency": int(getattr(result, "eval_concurrency", 0)),
            })
            _append_jsonl(attempts_path, response)
            RUNNER.require_clean_watcher(watcher)
    except Exception as exc:
        failure = exc
    finally:
        samples = watcher.stop() if watcher is not None else []
        if watcher is not None and not (output / "focused_runtime_watch.jsonl").exists():
            write_jsonl_create(output / "focused_runtime_watch.jsonl", samples)
    attempt_count = len(load_jsonl(attempts_path)) if attempts_path.exists() else 0
    if failure is None and (watcher is None or watcher.fatal_error or any(row.get("ok") is not True for row in watcher.samples)):
        failure = RuntimeError("focused runtime watcher failed")
    if failure is not None:
        write_json_create(output / "collection_failure.json", {"error": str(failure), "attempt_count": attempt_count, "request_timeout_s": REQUEST_TIMEOUT_S})
    write_json_create(output / "focused_collection_seal.json", {
        "schema": FOCUSED_SEAL_SCHEMA,
        "status": "complete" if failure is None else "failed",
        "bundle_sha256": seal_directory(output, exclude={"focused_collection_seal.json"}),
    })
    if failure is not None:
        raise failure
    return output


def _atomic_publish_noreplace(staging: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    RUNNER.atomic_publish_noreplace(staging, destination)


def splice_responses(bundle: SourceBundle, replacements: dict[tuple[int, int, int], dict[str, Any]], output: Path) -> dict[str, str]:
    """Write merged ledgers and prove that every untouched JSONL byte is identical."""
    written: dict[str, str] = {}
    for run in bundle.runs:
        source_bytes = run.responses.read_bytes()
        source_lines = source_bytes.splitlines(keepends=True)
        rows = load_jsonl(run.responses)
        changed = False
        for ordinal, row in enumerate(rows):
            replacement = replacements.get((run.tier, run.repetition, ordinal))
            if replacement is not None:
                if replacement.get("qid") != row.get("qid"):
                    raise ValueError("replacement qid differs from source row")
                rows[ordinal] = replacement
                changed = True
        relative = run.responses.relative_to(bundle.root)
        target = output / "final_ledgers" / relative
        if changed:
            if len(source_lines) != len(rows):
                raise ValueError("source response ledger has no stable physical rows")
            new_lines = list(source_lines)
            for ordinal, row in enumerate(rows):
                if (run.tier, run.repetition, ordinal) in replacements:
                    new_lines[ordinal] = json.dumps(row, sort_keys=True).encode() + b"\n"
            _write_create(target, b"".join(new_lines))
        else:
            _write_create(target, source_bytes)
        written[str(relative)] = sha256_path(target)
    return written


def regenerate_derived_artifacts(bundle: SourceBundle, output: Path, published: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Use the runner's evidence writer with mixed-window detail contracts."""
    for tier in (1, 2):
        _copy_file_create(bundle.root / f"question_vector.T{tier}.json", output / f"question_vector.T{tier}.json")
        _copy_file_create(bundle.root / f"scoring_vector.T{tier}.json", output / f"scoring_vector.T{tier}.json")
    observations: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    details: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    for run in bundle.runs:
        ledger = output / "final_ledgers" / run.responses.relative_to(bundle.root)
        rows = load_jsonl(ledger)
        suites: dict[str, list[bool]] = {}
        for row in rows:
            suites.setdefault(str(row["suite"]), []).append(bool(row["correct"]))
        source_raw = load_json(run.raw)
        raw = {
            "q": sum(bool(row["correct"]) for row in rows) * 3.0 / len(rows),
            "ts": RUNNER.utc_now(),
            "core_id": source_raw["core_id"],
            "protocol_id": RUNNER.PROTOCOL_ID,
            "n": len(rows),
            "era": source_raw["era"],
            "per_suite_quality": {key: sum(values) * 3.0 / len(values) for key, values in sorted(suites.items())},
            "per_suite_counts": {key: len(values) for key, values in sorted(suites.items())},
        }
        path = output / f"raw.T{run.tier}.r{run.repetition}.json"
        write_json_create(path, raw)
        trace = output / "migration" / "source_judge_traces" / run.judge_traces.relative_to(bundle.root)
        sidecar = output / "migration" / "source_sidecars" / run.sidecar.relative_to(bundle.root)
        _copy_file_create(run.judge_traces, trace)
        _copy_file_create(run.sidecar, sidecar)
        audit = RUNNER.validate_response_scoring(
            rows, bundle.scoring_vectors[run.tier]["questions"], trace,
            default_api_url="http://127.0.0.1:8000", tier=run.tier, repetition=run.repetition,
        )
        observations[run.tier].append({
            "path": str(published / path.relative_to(output)), "sha256": sha256_path(path),
            "q": raw["q"], "ts": raw["ts"], "core_id": raw["core_id"], "protocol_id": raw["protocol_id"], "n": raw["n"], "era": raw["era"],
        })
        details[run.tier].append({
            "tier": run.tier, "repetition": run.repetition, "n_results": len(rows), "mixed_window_contract": True,
            "response_vector_matches_input": [row["qid"] for row in rows] == _vector_qids(bundle.vectors[run.tier], tier=run.tier),
            "per_suite_counts_match_input": raw["per_suite_counts"] == {key: len(values) for key, values in sorted(suites.items())},
            "all_routes_frontdoor": all(row.get("route_used") == "frontdoor" for row in rows),
            "response_path": str(published / ledger.relative_to(output)), "response_sha256": sha256_path(ledger),
            "sidecar_path": str(published / sidecar.relative_to(output)), "sidecar_sha256": sha256_path(sidecar),
            "judge_trace_path": str(published / trace.relative_to(output)), "judge_trace_sha256": sha256_path(trace),
            "scoring_audit": audit,
        })
    evidence, aggregates = RUNNER.build_evidence(
        output_dir=output, published_dir=published, vectors=bundle.vectors, scoring_vectors=bundle.scoring_vectors,
        observations=observations, details=details, globally_eligible=True,
    )
    return evidence, aggregates, {"observations": observations, "details": details}


def seal_directory(root: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = {"run_seal.json"} if exclude is None else set(exclude)
    return {
        str(path.relative_to(root)): sha256_path(path)
        for path in sorted(root.rglob("*")) if path.is_file() and path.name not in excluded
    }


def _copy_snapshot(source: Path, destination: Path, hashes: dict[str, str] | None = None) -> None:
    if destination.exists():
        raise FileExistsError(f"snapshot destination exists: {destination}")
    if hashes is None:
        seal = load_json(source / "focused_collection_seal.json")
        hashes = dict(seal["bundle_sha256"])
    names = sorted({*hashes, "terminal_source.json" if (source / "terminal_source.json").is_file() else "focused_collection_seal.json"})
    for relative in names:
        path = source / relative
        if not path.is_file() or path.is_symlink() or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"snapshot source file is invalid: {relative}")
        _copy_file_create(path, destination / relative)
    if hashes is not None:
        for relative, expected in hashes.items():
            copied = destination / relative
            if not copied.is_file() or sha256_path(copied) != expected:
                raise ValueError(f"source snapshot differs for {relative}")


def _validate_focused_collection(bundle: SourceBundle, focused_dir: Path, plan: dict[str, Any]) -> dict[tuple[int, int, int], dict[str, Any]]:
    focused = focused_dir.absolute()
    _reject_overlap(bundle.root, focused)
    if focused.is_symlink() or not focused.is_dir():
        raise ValueError("focused collection directory is invalid")
    seal_path = focused / "focused_collection_seal.json"
    if not seal_path.is_file():
        raise ValueError("focused collection has no seal")
    seal = load_json(seal_path)
    if seal.get("schema") != FOCUSED_SEAL_SCHEMA or seal.get("status") != "complete":
        raise ValueError("focused collection is not complete")
    actual = seal_directory(focused, exclude={"focused_collection_seal.json"})
    if seal.get("bundle_sha256") != actual:
        raise ValueError("focused collection seal differs from its bundle")
    focused_plan = load_json(focused / "repair_plan.json")
    if canonical_hash(focused_plan) != canonical_hash(plan):
        raise ValueError("focused collection plan differs from current source plan")
    plan_sha = sha256_path(focused / "repair_plan.json")
    watcher_path = focused / "focused_runtime_watch.jsonl"
    if not watcher_path.is_file():
        raise ValueError("focused collection has no runtime watcher")
    watcher = load_jsonl(watcher_path)
    if not watcher or any(row.get("ok") is not True for row in watcher):
        raise ValueError("focused collection runtime watcher is not clean")
    expected = {(row["tier"], row["repetition"], row["ordinal"]): row for row in plan["targets"]}
    attempts_path = focused / "focused_attempts.jsonl"
    if not attempts_path.is_file():
        raise ValueError("focused collection has no attempts")
    replacements: dict[tuple[int, int, int], dict[str, Any]] = {}
    required_response = {"qid", "suite", "scoring_method", "answer", "correct", "error", "partial", "degraded", "route_used", "scoring_config_sha256"}
    for row in load_jsonl(attempts_path):
        key = (row.get("tier"), row.get("repetition"), row.get("ordinal"))
        target = expected.get(key)
        if target is None or key in replacements:
            raise ValueError("focused collection has an unexpected or duplicate replacement")
        if (
            row.get("source_terminal_sha256") != plan["source_terminal_sha256"]
            or row.get("repair_plan_sha256") != plan_sha
            or row.get("failure_fingerprint") != target["fingerprint"]
            or row.get("qid") != target["qid"]
            or row.get("route_used") != "frontdoor"
            or row.get("request_timeout_s") != REQUEST_TIMEOUT_S
            or row.get("score_terminal") is not True
            or row.get("error") is not None
            or not isinstance(row.get("correct"), bool)
            or not str(row.get("answer") or "").strip()
            or not required_response <= set(row)
        ):
            raise ValueError("focused replacement does not satisfy the sealed contract")
        # EvalTower records the actual semaphore width; a one-item request must report one.
        if row.get("eval_concurrency") != 1:
            raise ValueError("focused replacement did not use concurrency one")
        replacements[key] = {name: row[name] for name in required_response}
    if set(replacements) != set(expected):
        raise ValueError("focused collection does not exactly cover the repair plan")
    return replacements


def publish_candidate(source: Path, destination: Path, *, focused_dir: Path) -> Path:
    """Offline finalization hook used after focused collection has succeeded."""
    bundle = validate_source(source)
    plan = build_plan(bundle)
    replacements = _validate_focused_collection(bundle, focused_dir, plan)
    _reject_overlap(bundle.root, focused_dir.absolute(), destination.absolute())
    if destination.is_symlink():
        raise ValueError("destination symlink is not permitted")
    staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    try:
        _copy_snapshot(bundle.root, staging / "source_snapshot", bundle.manifest["source_sha256"])
        _copy_snapshot(focused_dir.absolute(), staging / "focused_snapshot")
        write_json_create(staging / "repair_plan.json", plan)
        ledgers = splice_responses(bundle, replacements, staging)
        evidence, aggregates, repair_detail = regenerate_derived_artifacts(bundle, staging, destination)
        evidence.update({
            "mixed_window_provenance": True,
            "source_terminal_sha256": plan["source_terminal_sha256"],
            "repair_plan_sha256": sha256_path(staging / "repair_plan.json"),
            "replacement_count": len(replacements), "final_ledgers": ledgers,
        })
        evidence_path = staging / "e8_quality_baseline_evidence.json"
        write_json_create(evidence_path, evidence)
        report = {
            "schema": "epyc.e8_quality_baseline_reseed_runner.v1", "mode": "repair_candidate",
            "era": RUNNER.E8_ERA, "protocol_id": RUNNER.PROTOCOL_ID, "required_repetitions": REPETITIONS,
            "decision_grade": True, "mixed_window_provenance": True, "no_state_write": True,
            "observations": repair_detail["details"], "aggregates": aggregates,
            "evidence_manifest": str(destination / evidence_path.name), "evidence_manifest_sha256": sha256_path(evidence_path),
        }
        report_path = staging / "runner_report.json"
        write_json_create(report_path, report)
        seal = {
            "schema": "epyc.e8_quality_baseline_run_seal.v1", "status": "complete",
            "manifest_sha256": sha256_path(evidence_path), "runner_report_sha256": sha256_path(report_path),
            "bundle_sha256": seal_directory(staging), "mixed_window_provenance": True,
        }
        write_json_create(staging / "run_seal.json", seal)
        _atomic_publish_noreplace(staging, destination)
    except Exception:
        # Preserve this no-replace staging directory as forensic evidence.
        raise
    return destination


def validate_candidate(candidate_dir: Path) -> dict[str, Any]:
    """Replay immutable provenance, seals, byte preservation, and score ledgers."""
    if candidate_dir.is_symlink():
        raise ValueError("candidate symlink is not permitted")
    root = candidate_dir.absolute()
    seal = load_json(root / "run_seal.json")
    if seal.get("schema") != "epyc.e8_quality_baseline_run_seal.v1" or seal.get("status") != "complete":
        raise ValueError("candidate run seal is invalid")
    if seal.get("bundle_sha256") != seal_directory(root):
        raise ValueError("candidate run seal differs from bundle")
    source = validate_source(root / "source_snapshot")
    plan = load_json(root / "repair_plan.json")
    replacements = _validate_focused_collection(source, root / "focused_snapshot", plan)
    for run in source.runs:
        source_lines = run.responses.read_bytes().splitlines(keepends=True)
        final = root / "final_ledgers" / run.responses.relative_to(source.root)
        final_lines = final.read_bytes().splitlines(keepends=True)
        if len(source_lines) != len(final_lines):
            raise ValueError("candidate response cardinality differs from source")
        for ordinal, (old, new) in enumerate(zip(source_lines, final_lines)):
            if (run.tier, run.repetition, ordinal) not in replacements and old != new:
                raise ValueError("candidate changed a non-target response byte")
        rows = load_jsonl(final)
        RUNNER.validate_response_scoring(
            rows, source.scoring_vectors[run.tier]["questions"],
            root / "migration" / "source_judge_traces" / run.judge_traces.relative_to(source.root),
            default_api_url="http://127.0.0.1:8000", tier=run.tier, repetition=run.repetition,
        )
    evidence = load_json(root / "e8_quality_baseline_evidence.json")
    report = load_json(root / "runner_report.json")
    if evidence.get("mixed_window_provenance") is not True or report.get("decision_grade") is not True:
        raise ValueError("candidate derived artifacts do not satisfy mixed-window contract")
    return {"valid": True, "candidate": str(root), "replacement_count": len(replacements)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--collect", action="store_true", help="collect focused rows; no publication")
    mode.add_argument("--repair-candidate", action="store_true")
    mode.add_argument("--validate-candidate", action="store_true")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focused-dir", type=Path, help="sealed focused collection directory")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--evaltower-timeout-s", type=int, choices=(REQUEST_TIMEOUT_S,), default=REQUEST_TIMEOUT_S)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = validate_source(args.source_dir)
    plan = build_plan(bundle)
    if args.plan:
        write_plan(plan, args.output_dir)
        return 0
    if args.collect:
        collect_focused(args.source_dir, args.output_dir, api_url=args.api_url)
        return 0
    if args.validate_candidate:
        validate_candidate(args.output_dir)
        return 0
    if args.focused_dir is None:
        raise ValueError("--repair-candidate requires --focused-dir")
    publish_candidate(args.source_dir, args.output_dir, focused_dir=args.focused_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
