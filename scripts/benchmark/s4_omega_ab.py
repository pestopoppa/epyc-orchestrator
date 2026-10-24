#!/usr/bin/env python3
"""Paired S4 Omega prompt A/B runner.

The live path sends every question to ``frontdoor:repl`` and selects the prompt
with the API's request-local ``x_orchestrator_prompt_root`` override. It never
changes process-wide prompt flags. The checked-in three-question suite is a
synthetic dry-run fixture only; it is not suitable for an efficacy claim.

Live execution is fail-closed until the deployed API advertises native
``ChatResponse.prompt_tokens`` and the root Vidya source row plus adapter exist.
Prompt-token totals are never inferred from ``tokens_used`` or shared process
counters. Missing response metrics remain null and mark the captured row incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
DEFAULT_SUITE = HERE / "s4_omega_questions.json"
DEFAULT_SUITE_SHA256 = "851947fefbde4aba42960d766dae4f42c48623100a8ead601f4abfb9ddd34686"
PROMPT_DIR = HERE.parent.parent / "orchestration" / "prompts"
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_ROOT_REPO = Path(os.environ.get("EPYC_ROOT_REPO", "/workspace"))
DEFAULT_PROMPT_OVERRIDE_BASE = Path(
    os.environ.get(
        "ORCHESTRATOR_PROMPT_ROOT_OVERRIDE_BASE",
        "/workspace/repos/epyc-orchestrator/tmp/gepa_prompt_roots",
    )
)
ADAPTER_RELATIVE_PATH = Path("scripts/vidya/adapters/s4_omega_ab.py")
SOURCE_TABLE_RELATIVE_PATH = Path("scripts/vidya/adapters/README.md")
SOURCE_TASK_RELATIVE_PATH = Path("handoffs/active/vidya-belief-substrate-program.md")
RECORD_SCHEMA = "epyc.s4_omega_ab.record.v1"
MANIFEST_SCHEMA = "epyc.s4_omega_ab.manifest.v1"
SUITE_SCHEMA = "epyc.s4_omega_question_suite.v1"
TURNS_UNIT = "orchestrator_repl_turn"

sys.path.insert(0, str(HERE))
from seeding_orchestrator import call_orchestrator_forced  # noqa: E402
from seeding_scoring import score_answer_or_error  # noqa: E402


class RunnerNotReady(RuntimeError):
    """The live path lacks a required measurement capability or source wire."""


class LiveRunInterrupted(RuntimeError):
    """A live run stopped after persisting its completed native records."""

    def __init__(self, message: str, *, manifest_path: Path, records_path: Path):
        super().__init__(message)
        self.manifest_path = manifest_path
        self.records_path = records_path


class RequestCaptureClient:
    """Capture the exact JSON body the existing seeding HTTP helper submits."""

    def __init__(self, delegate: Any):
        self.delegate = delegate
        self.last_payload: dict[str, Any] | None = None

    def capture_payload(self, payload: dict[str, Any]) -> None:
        self.last_payload = dict(payload)

    def post(self, *args: Any, **kwargs: Any) -> Any:
        payload = kwargs.get("json")
        if isinstance(payload, dict):
            self.capture_payload(payload)
        return self.delegate.post(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def with_content_hash(value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_sha256", None)
    result = dict(body)
    result["content_sha256"] = sha256_bytes(canonical_json_bytes(body))
    return result


def verify_content_hash(value: dict[str, Any]) -> bool:
    expected = value.get("content_sha256")
    body = dict(value)
    body.pop("content_sha256", None)
    return isinstance(expected, str) and expected == sha256_bytes(canonical_json_bytes(body))


def load_suite(path: Path = DEFAULT_SUITE, *, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if path.resolve() == DEFAULT_SUITE.resolve() and expected_sha256 is None:
        expected_sha256 = DEFAULT_SUITE_SHA256
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"suite digest mismatch: expected {expected_sha256}, got {digest}")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema") != SUITE_SCHEMA:
        raise ValueError(f"suite must use schema {SUITE_SCHEMA}")
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("suite questions must be a non-empty list")
    seen: set[str] = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("each suite question must be an object")
        qid = question.get("id")
        if not isinstance(qid, str) or not qid.strip() or qid in seen:
            raise ValueError(f"question ids must be unique non-empty strings: {qid!r}")
        seen.add(qid)
        for field in ("prompt", "expected", "scoring_method"):
            if not isinstance(question.get(field), str):
                raise ValueError(f"question {qid} requires string field {field}")
        if not isinstance(question.get("scoring_config", {}), dict):
            raise ValueError(f"question {qid} scoring_config must be an object")
    return data, digest


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _metric_value(response: dict[str, Any], key: str) -> tuple[int | None, str | None]:
    if key not in response:
        return None, f"response_missing_{key}"
    value = _nonnegative_int(response.get(key))
    if value is None:
        return None, f"response_invalid_{key}"
    return value, None


def _passed(response: dict[str, Any], question: dict[str, Any]) -> tuple[bool | None, str | None]:
    error = response.get("error") or response.get("error_detail")
    if error:
        return False, str(error)
    answer = response.get("answer")
    if not isinstance(answer, str):
        return None, "response_missing_answer"
    passed, reason = score_answer_or_error(
        answer,
        question["expected"],
        question["scoring_method"],
        question.get("scoring_config", {}),
    )
    return passed, reason


def make_record(
    *,
    run_id: str,
    suite_id: str,
    suite_sha256: str,
    question: dict[str, Any],
    arm_id: str,
    prompt_sha256: str,
    response: dict[str, Any],
    elapsed_seconds: float,
    synthetic: bool,
    order_within_pair: int = 0,
    batch_id: str | None = None,
    request_contract: dict[str, Any] | None = None,
    request_sha256: str | None = None,
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    turns, turns_error = _metric_value(response, "turns")
    generated, generated_error = _metric_value(response, "tokens_generated")
    prompt_tokens, prompt_error = _metric_value(response, "prompt_tokens")
    passed, score_error = _passed(response, question)
    errors = [e for e in (turns_error, generated_error, prompt_error, score_error) if e]
    request_contract = request_contract or {
        "prompt": question["prompt"],
        "role": "frontdoor",
        "mode": "repl",
        "arm_id": arm_id,
        "prompt_sha256": prompt_sha256,
        "suite_sha256": suite_sha256,
        "question_id": question["id"],
    }
    if synthetic and request_sha256 is None:
        request_sha256 = sha256_bytes(canonical_json_bytes(request_contract))
    request_error = None if request_sha256 else "request_payload_not_captured"
    if request_error:
        errors.append(request_error)
    row = {
        "schema": RECORD_SCHEMA,
        "run_id": run_id,
        "suite_id": suite_id,
        "suite_sha256": suite_sha256,
        "question_id": question["id"],
        "order_within_pair": order_within_pair,
        "batch_id": batch_id,
        "request_sha256": request_sha256,
        "request_payload": request_payload if request_payload is not None else (request_contract if synthetic else None),
        "request_sha256_source": "synthetic_contract" if synthetic else ("http_json_payload" if request_sha256 else None),
        "request_sha256_unavailable_reason": request_error,
        "arm_id": arm_id,
        "role": "frontdoor",
        "mode": "repl",
        "turns": turns,
        "turns_unit": TURNS_UNIT,
        "tokens_generated": generated,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_unavailable_reason": prompt_error,
        "passed": passed,
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        "score_unavailable_reason": score_error,
        "schema_validation": "synthetic_fixture" if synthetic else ("complete" if not errors else "incomplete"),
        "metric_unavailable_reasons": errors,
        "error": str(response.get("error") or response.get("error_detail") or "") or None,
        "response": dict(response),
        "question": dict(question),
        "synthetic": bool(synthetic),
        "prompt_sha256": prompt_sha256,
        "protocol_id": None,
        "protocol_id_unavailable_reason": "not_pre_registered",
        "promotion_claim": False,
        "model_identity": None,
        "model_identity_unavailable_reason": "not_exposed_by_chatresponse",
        "server_identity": None,
        "server_identity_unavailable_reason": "not_exposed_by_chatresponse",
    }
    return with_content_hash(row)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        tmp.unlink(missing_ok=True)


def write_artifacts(
    output_dir: Path,
    *,
    run_id: str,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"{run_id}.records.jsonl"
    manifest_path = output_dir / f"{run_id}.manifest.json"
    if any(not verify_content_hash(record) for record in records):
        raise ValueError("refusing to write a record with an invalid self hash")
    records_bytes = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    _atomic_write(records_path, records_bytes)
    sealed_manifest = with_content_hash(
        {
            **manifest,
            "schema": MANIFEST_SCHEMA,
            "run_id": run_id,
            "record_count": len(records),
            "records_file": records_path.name,
            "records_file_sha256": sha256_bytes(records_bytes),
            "record_sha256s": [record["content_sha256"] for record in records],
        }
    )
    _atomic_write(manifest_path, canonical_json_bytes(sealed_manifest) + b"\n")
    return manifest_path, records_path


class NativeRunWriter:
    """Append-only per-question writer with a durable, self-hashed run manifest."""

    def __init__(self, output_dir: Path, *, run_id: str, manifest: dict[str, Any]):
        self.run_id = run_id
        self.manifest = dict(manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = output_dir / f"{run_id}.records.jsonl"
        self.manifest_path = output_dir / f"{run_id}.manifest.json"
        if self.records_path.exists() or self.manifest_path.exists():
            raise FileExistsError(f"run artifacts already exist for {run_id}")
        _atomic_write(self.records_path, b"")
        self._publish(status="running")

    def append(self, record: dict[str, Any]) -> None:
        if not verify_content_hash(record):
            raise ValueError("refusing to append a record with an invalid self hash")
        line = canonical_json_bytes(record) + b"\n"
        with self.records_path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        # Publish only after the record line is durable. If the process stops
        # between these writes, `reopen_native_run` reconstructs the manifest.
        self._publish(status="running")

    def seal(self, *, status: str, failure: str | None = None) -> dict[str, Any]:
        return self._publish(status=status, failure=failure)

    def _publish(self, *, status: str, failure: str | None = None) -> dict[str, Any]:
        rows = _read_native_records(self.records_path)
        reasons = [
            reason
            for row in rows
            for reason in row.get("metric_unavailable_reasons", [])
        ]
        schema_validation = "incomplete" if reasons else ("complete" if rows else "pending")
        records_bytes = self.records_path.read_bytes()
        manifest = with_content_hash(
            {
                **self.manifest,
                "schema": MANIFEST_SCHEMA,
                "run_id": self.run_id,
                "run_status": status,
                "schema_validation": schema_validation,
                "measurement_status": "candidate_observation",
                "protocol_id": None,
                "protocol_id_unavailable_reason": "not_pre_registered",
                "promotion_claim": False,
                "model_identity": None,
                "model_identity_unavailable_reason": "not_exposed_by_chatresponse",
                "server_identity": None,
                "server_identity_unavailable_reason": "not_exposed_by_chatresponse",
                "failure": failure,
                "record_count": len(rows),
                "records_file": self.records_path.name,
                "records_file_sha256": sha256_bytes(records_bytes),
                "record_sha256s": [row["content_sha256"] for row in rows],
            }
        )
        _atomic_write(self.manifest_path, canonical_json_bytes(manifest) + b"\n")
        return manifest


def _read_native_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"incomplete final JSONL record in {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not verify_content_hash(value):
            raise ValueError(f"invalid native record hash at {path}:{line_number}")
        rows.append(value)
    return rows


def reopen_native_run(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify a complete or partial run from its durable manifest and JSONL."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not verify_content_hash(manifest):
        raise ValueError(f"invalid native manifest hash: {manifest_path}")
    records_name = manifest.get("records_file")
    if not isinstance(records_name, str) or Path(records_name).name != records_name:
        raise ValueError("manifest records_file must be a local filename")
    records_path = manifest_path.parent / records_name
    rows = _read_native_records(records_path)
    records_bytes = records_path.read_bytes()
    if (
        manifest.get("record_count") != len(rows)
        or manifest.get("record_sha256s") != [row["content_sha256"] for row in rows]
        or manifest.get("records_file_sha256") != sha256_bytes(records_bytes)
    ):
        raise ValueError("native manifest does not match its records JSONL")
    return manifest, rows


def recover_native_run(
    manifest_path: Path, *, source_stopped: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create a separate recovery view without modifying original run bytes."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not verify_content_hash(manifest):
        raise ValueError(f"invalid native manifest hash: {manifest_path}")
    records_name = manifest.get("records_file")
    if not isinstance(records_name, str) or Path(records_name).name != records_name:
        raise ValueError("manifest records_file must be a local filename")
    records_path = manifest_path.parent / records_name
    original_bytes = records_path.read_bytes()
    if manifest.get("run_status") in {"complete", "partial", "refused", "synthetic"}:
        return reopen_native_run(manifest_path)
    if not source_stopped:
        raise ValueError("recovery requires explicit confirmation that the writer is stopped")
    newline = original_bytes.rfind(b"\n")
    valid_prefix = original_bytes[: newline + 1] if newline >= 0 else b""
    tail = original_bytes[len(valid_prefix):]
    rows = []
    for line_number, line in enumerate(valid_prefix.splitlines(), start=1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not verify_content_hash(value):
            raise ValueError(f"invalid native record hash at {records_path}:{line_number}")
        rows.append(value)
    stem = manifest_path.name.removesuffix(".manifest.json")
    recovered_records = manifest_path.parent / f"{stem}.recovered.records.jsonl"
    recovery_tail = manifest_path.parent / f"{stem}.recovery-tail.bin"
    recovered_manifest = manifest_path.parent / f"{stem}.recovered.manifest.json"
    _atomic_write(recovered_records, valid_prefix)
    if tail:
        _atomic_write(recovery_tail, tail)
    else:
        recovery_tail.unlink(missing_ok=True)
    dynamic_fields = {
        "content_sha256", "schema", "run_status", "schema_validation",
        "measurement_status", "failure", "record_count", "records_file",
        "records_file_sha256", "record_sha256s",
    }
    recovery = with_content_hash({
        **{key: value for key, value in manifest.items() if key not in dynamic_fields},
        "schema": MANIFEST_SCHEMA,
        "run_id": manifest.get("run_id"),
        "run_status": "partial",
        "failure": "recovered_from_stopped_writer_original_preserved",
        "recovery_original_manifest_sha256": sha256_file(manifest_path),
        "recovery_original_records_sha256": sha256_bytes(original_bytes),
        "recovery_tail_file": recovery_tail.name if tail else None,
        "recovery_tail_sha256": sha256_bytes(tail) if tail else None,
        "record_count": len(rows),
        "records_file": recovered_records.name,
        "records_file_sha256": sha256_bytes(valid_prefix),
        "record_sha256s": [row["content_sha256"] for row in rows],
        "schema_validation": "incomplete" if any(row.get("schema_validation") != "complete" for row in rows) else ("complete" if rows else "pending"),
        "measurement_status": "candidate_observation",
    })
    _atomic_write(recovered_manifest, canonical_json_bytes(recovery) + b"\n")
    return recovery, rows


def synthetic_response(question: dict[str, Any], arm_id: str, ordinal: int) -> dict[str, Any]:
    """Fixed fake response for schema/durability tests; never a model observation."""
    return {
        "answer": question["expected"],
        "turns": (ordinal % 4) + (1 if arm_id == "baseline" else 2),
        "tokens_generated": 20 + ordinal,
        "prompt_tokens": 100 + ordinal,
    }


def run_dry_run(
    *,
    suite: dict[str, Any],
    suite_sha256: str,
    output_dir: Path,
    response_factory: Callable[[dict[str, Any], str, int], dict[str, Any]] = synthetic_response,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    run_id = f"dry-{uuid.uuid4().hex}"
    records: list[dict[str, Any]] = []
    arm_specs = _arm_specs()
    ordinal = 0
    for question_index, question in enumerate(suite["questions"]):
        ordered_arms = arm_specs if question_index % 2 == 0 else list(reversed(arm_specs))
        for pair_order, (arm_id, prompt_sha256, _) in enumerate(ordered_arms):
            response = response_factory(question, arm_id, ordinal)
            records.append(
                make_record(
                    run_id=run_id,
                    suite_id=suite["suite_id"],
                    suite_sha256=suite_sha256,
                    question=question,
                    arm_id=arm_id,
                    prompt_sha256=prompt_sha256,
                    response=response,
                    elapsed_seconds=0.001 * (ordinal + 1),
                    synthetic=True,
                    order_within_pair=pair_order,
                    batch_id=f"{run_id}:{question['id']}:{arm_id}",
                )
            )
            ordinal += 1
    manifest = {
        "suite_id": suite["suite_id"],
        "suite_sha256": suite_sha256,
        "question_ids": [q["id"] for q in suite["questions"]],
        "arms": [
            {"arm_id": arm_id, "prompt_source": f"orchestration/prompts/{source.name}", "prompt_sha256": digest}
            for arm_id, digest, source in arm_specs
        ],
        "role": "frontdoor",
        "mode": "repl",
        "turns_unit": TURNS_UNIT,
        "arm_order_strategy": "alternating baseline-first and s4_omega-first by suite order",
        "request_policy": {
            "allow_delegation": False,
            "cache_prompt": False,
            "request_priority": "background",
            "workload_class": "eval_batch",
        },
        "synthetic": True,
        "run_status": "synthetic",
        "schema_validation": "synthetic_fixture",
        "measurement_status": "not_a_measurement",
        "protocol_id": None,
        "protocol_id_unavailable_reason": "not_pre_registered",
        "promotion_claim": False,
        "model_identity": None,
        "model_identity_unavailable_reason": "not_exposed_by_chatresponse",
        "server_identity": None,
        "server_identity_unavailable_reason": "not_exposed_by_chatresponse",
        "runner_source_sha256": sha256_file(Path(__file__)),
    }
    paths = write_artifacts(output_dir, run_id=run_id, manifest=manifest, records=records)
    return paths[0], paths[1], records


def _arm_specs(prompt_dir: Path = PROMPT_DIR) -> list[tuple[str, str, Path]]:
    baseline_path = prompt_dir / "root_lm_system.md"
    omega_path = prompt_dir / "root_lm_system.s4_omega.md"
    for path in (baseline_path, omega_path):
        if not path.is_file():
            raise FileNotFoundError(f"required S4 prompt file is missing: {path}")
    return [
        ("baseline", sha256_file(baseline_path), baseline_path),
        ("s4_omega", sha256_file(omega_path), omega_path),
    ]


def check_chatresponse_capabilities(api_url: str, *, client: Any) -> str:
    response = client.get(f"{api_url.rstrip('/')}/openapi.json", timeout=10)
    response.raise_for_status()
    schema = response.json()
    properties = (
        schema.get("components", {})
        .get("schemas", {})
        .get("ChatResponse", {})
        .get("properties", {})
    )
    required = {"turns", "tokens_generated", "prompt_tokens"}
    missing = sorted(required - set(properties))
    if missing:
        raise RunnerNotReady(
            "live run refused before inference: ChatResponse does not advertise "
            + ", ".join(missing)
        )
    return sha256_bytes(canonical_json_bytes(schema))


def check_source_wiring(root_repo: Path) -> tuple[Path, str]:
    adapter = root_repo / ADAPTER_RELATIVE_PATH
    source_table = root_repo / SOURCE_TABLE_RELATIVE_PATH
    source_task = root_repo / SOURCE_TASK_RELATIVE_PATH
    if not adapter.is_file() or not source_table.is_file() or not source_task.is_file():
        raise RunnerNotReady("live run refused before inference: S4 Omega Vidya adapter/source table missing")
    table = source_table.read_text(encoding="utf-8")
    if "S4 Omega REPL A/B runner records" not in table:
        raise RunnerNotReady("live run refused before inference: S4 Omega source row is not enrolled")
    task = source_task.read_text(encoding="utf-8")
    parent_complete = re.search(
        r"(?m)^- \[x\] \*\*VB-S4-OMEGA-WIRE(?:\s|—|\*)", task
    )
    if parent_complete is None:
        raise RunnerNotReady("live run refused before inference: VB-S4-OMEGA-WIRE is not complete")
    return adapter, sha256_file(adapter)


def prepare_prompt_roots(
    *,
    run_id: str,
    override_base: Path = DEFAULT_PROMPT_OVERRIDE_BASE,
    prompt_dir: Path = PROMPT_DIR,
) -> dict[str, Path]:
    arms = _arm_specs(prompt_dir)
    override_base.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix=f"s4-omega-{run_id}-", dir=override_base))
    roots: dict[str, Path] = {}
    try:
        for arm_id, _digest, source in arms:
            root = run_root / arm_id
            shutil.copytree(prompt_dir, root)
            # The server's process environment may select a prompt variant. Pin
            # this one prompt per request while preserving every other prompt
            # file identically across the two arms.
            for variant in root.glob("root_lm_system.*.md"):
                variant.unlink()
            shutil.copyfile(source, root / "root_lm_system.md")
            if arm_id == "s4_omega":
                shutil.copyfile(source, root / "root_lm_system.s4_omega.md")
            roots[arm_id] = root
    except Exception:
        shutil.rmtree(run_root, ignore_errors=True)
        raise
    return roots


def run_live(
    *,
    suite: dict[str, Any],
    suite_sha256: str,
    output_dir: Path,
    api_url: str = DEFAULT_API_URL,
    root_repo: Path = DEFAULT_ROOT_REPO,
    override_base: Path = DEFAULT_PROMPT_OVERRIDE_BASE,
    prompt_dir: Path = PROMPT_DIR,
    timeout_s: int = 300,
    client: Any | None = None,
    call_fn: Callable[..., dict[str, Any]] = call_orchestrator_forced,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    import httpx

    own_client = client is None
    client = client or httpx.Client()
    prompt_run_root: Path | None = None
    writer: NativeRunWriter | None = None
    records: list[dict[str, Any]] = []
    try:
        openapi_sha256 = check_chatresponse_capabilities(api_url, client=client)
        adapter_path, adapter_sha256 = check_source_wiring(root_repo)
        run_id = f"live-{uuid.uuid4().hex}"
        prompt_roots = prepare_prompt_roots(
            run_id=run_id, override_base=override_base, prompt_dir=prompt_dir
        )
        prompt_run_root = next(iter(prompt_roots.values())).parent
        arms = _arm_specs(prompt_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir = output_dir / f"{run_id}.sources"
        snapshot_dir.mkdir()
        suite_bytes = canonical_json_bytes(suite) + b"\n"
        _atomic_write(snapshot_dir / "suite.json", suite_bytes)
        prompt_snapshots = {}
        for arm_id, _digest, source in arms:
            snapshot_name = f"{arm_id}.root_lm_system.md"
            _atomic_write(snapshot_dir / snapshot_name, source.read_bytes())
            prompt_snapshots[arm_id] = {
                "file": snapshot_name,
                "sha256": sha256_file(snapshot_dir / snapshot_name),
            }
        base_manifest = {
            "suite_id": suite["suite_id"],
            "suite_sha256": suite_sha256,
            "suite_snapshot": f"{snapshot_dir.name}/suite.json",
            "suite_snapshot_sha256": sha256_bytes(suite_bytes),
            "source_snapshot_dir": snapshot_dir.name,
            "prompt_snapshots": prompt_snapshots,
            "question_ids": [q["id"] for q in suite["questions"]],
            "arms": [
                {"arm_id": arm_id, "prompt_source": f"orchestration/prompts/{source.name}", "prompt_sha256": digest}
                for arm_id, digest, source in arms
            ],
            "role": "frontdoor",
            "mode": "repl",
            "turns_unit": TURNS_UNIT,
            "arm_order_strategy": "alternating baseline-first and s4_omega-first by suite order",
            "request_policy": {
                "allow_delegation": False,
                "cache_prompt": False,
                "request_priority": "background",
                "workload_class": "eval_batch",
                "timeout_s": timeout_s,
            },
            "synthetic": False,
            "vidya_adapter": str(adapter_path),
            "vidya_adapter_sha256": adapter_sha256,
            "openapi_sha256": openapi_sha256,
            "api_url": api_url,
            "runner_source_sha256": sha256_file(Path(__file__)),
        }
        writer = NativeRunWriter(output_dir, run_id=run_id, manifest=base_manifest)
        for question_index, question in enumerate(suite["questions"]):
            ordered_arms = arms if question_index % 2 == 0 else list(reversed(arms))
            for pair_order, (arm_id, prompt_sha256, _source) in enumerate(ordered_arms):
                batch_id = f"{run_id}:{question['id']}:{arm_id}"
                started = time.perf_counter()
                request_capture = RequestCaptureClient(client)
                response = call_fn(
                    prompt=question["prompt"],
                    force_role="frontdoor",
                    force_mode="repl",
                    url=api_url,
                    timeout=timeout_s,
                    client=request_capture,
                    allow_delegation=False,
                    cache_prompt=False,
                    scoring_method=question["scoring_method"],
                    workload_class="eval_batch",
                    request_priority="background",
                    prompt_root=str(prompt_roots[arm_id]),
                    batch_id=batch_id,
                )
                elapsed = time.perf_counter() - started
                record = make_record(
                    run_id=run_id,
                    suite_id=suite["suite_id"],
                    suite_sha256=suite_sha256,
                    question=question,
                    arm_id=arm_id,
                    prompt_sha256=prompt_sha256,
                    response=response,
                    elapsed_seconds=elapsed,
                    synthetic=False,
                    order_within_pair=pair_order,
                    batch_id=batch_id,
                    request_sha256=(
                        sha256_bytes(canonical_json_bytes(request_capture.last_payload))
                        if request_capture.last_payload is not None
                        else None
                    ),
                    request_payload=request_capture.last_payload,
                )
                writer.append(record)
                records.append(record)
                if record["schema_validation"] != "complete":
                    break
            if records and records[-1]["schema_validation"] != "complete":
                break
        run_status = (
            "complete"
            if len(records) == 2 * len(suite["questions"])
            and all(record["schema_validation"] == "complete" for record in records)
            else "refused"
        )
        writer.seal(status=run_status)
        return writer.manifest_path, writer.records_path, records
    except Exception as exc:
        if writer is None:
            raise
        try:
            writer.seal(
                status="partial",
                failure=f"{type(exc).__name__}: {str(exc)[:300]}",
            )
        except Exception as persist_error:
            raise LiveRunInterrupted(
                f"live run failed ({type(exc).__name__}) and partial manifest refresh failed "
                f"({type(persist_error).__name__}); inspect durable records",
                manifest_path=writer.manifest_path,
                records_path=writer.records_path,
            ) from exc
        raise LiveRunInterrupted(
            f"live run interrupted after {len(records)} durable records: {type(exc).__name__}: {exc}",
            manifest_path=writer.manifest_path,
            records_path=writer.records_path,
        ) from exc
    finally:
        if prompt_run_root is not None:
            shutil.rmtree(prompt_run_root, ignore_errors=True)
        if own_client:
            client.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="write marked synthetic fixtures; never contacts the API")
    parser.add_argument("--live", action="store_true", help="run real frontdoor:repl inference after fail-closed preflight")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--root-repo", type=Path, default=DEFAULT_ROOT_REPO)
    parser.add_argument("--prompt-override-base", type=Path, default=DEFAULT_PROMPT_OVERRIDE_BASE)
    parser.add_argument("--timeout-s", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dry_run == args.live:
        raise SystemExit("choose exactly one of --dry-run or --live")
    suite, digest = load_suite(args.suite)
    if args.dry_run:
        manifest_path, records_path, records = run_dry_run(
            suite=suite, suite_sha256=digest, output_dir=args.output_dir
        )
        print(json.dumps({"manifest": str(manifest_path), "records": str(records_path), "record_count": len(records), "schema_validation": "synthetic_fixture", "synthetic": True}))
        return 0
    try:
        manifest_path, records_path, records = run_live(
            suite=suite,
            suite_sha256=digest,
            output_dir=args.output_dir,
            api_url=args.api_url,
            root_repo=args.root_repo,
            override_base=args.prompt_override_base,
            timeout_s=args.timeout_s,
        )
    except LiveRunInterrupted as exc:
        manifest, persisted_records = recover_native_run(exc.manifest_path)
        print(
            json.dumps(
                {
                    "manifest": str(exc.manifest_path),
                    "records": str(exc.records_path),
                    "record_count": len(persisted_records),
                    "run_status": manifest["run_status"],
                    "schema_validation": manifest["schema_validation"],
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 2
    except RunnerNotReady as exc:
        print(str(exc), file=sys.stderr)
        return 2
    manifest, persisted_records = reopen_native_run(manifest_path)
    success = manifest.get("run_status") == "complete" and manifest.get("schema_validation") == "complete"
    print(json.dumps({"manifest": str(manifest_path), "records": str(records_path), "record_count": len(persisted_records), "run_status": manifest["run_status"], "schema_validation": manifest["schema_validation"], "synthetic": False}))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
