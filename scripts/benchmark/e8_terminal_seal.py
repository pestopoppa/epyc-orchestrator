#!/usr/bin/env python3
"""Shared fail-closed terminal sealing for E8 writer aborts."""

from __future__ import annotations

import functools
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable
import uuid


RUN_SEAL_SCHEMA = "epyc.e8_quality_baseline_run_seal.v1"
COMPLETE_STATUS = "complete"
STAGED_COMPLETE_STATUS = "staged_complete_pending_publish"
TERMINAL_STATUS = "terminal_aborted_no_admission"


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_full(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError("short write while sealing E8 abort")
        offset += written


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        _write_full(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _bundle_hashes(namespace: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(namespace.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"E8 aborted namespace contains a symlink: {path}")
        if path.is_file() and path != namespace / "run_seal.json":
            hashes[path.relative_to(namespace).as_posix()] = _sha256_path(path)
    return hashes


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def record_complete(
    namespace: Path,
    *,
    writer: str,
    manifest_name: str,
    runner_path: Path | None = None,
) -> Path:
    """Atomically seal a validated successor namespace as complete."""
    namespace = namespace.absolute()
    if namespace.is_symlink() or not namespace.is_dir():
        raise ValueError(f"cannot seal unsafe E8 namespace: {namespace}")
    if namespace.name.startswith(".") and ".staging-" in namespace.name:
        raise ValueError("cannot seal an E8 staging namespace complete")
    if Path(manifest_name).name != manifest_name or manifest_name == "run_seal.json":
        raise ValueError("E8 completion manifest must be one safe basename")

    manifest = namespace / manifest_name
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError(f"E8 completion manifest must be a regular file: {manifest}")
    seal_path = namespace / "run_seal.json"
    if seal_path.exists() or seal_path.is_symlink():
        raise FileExistsError(f"E8 run seal already exists: {seal_path}")
    seal = {
        "schema": RUN_SEAL_SCHEMA,
        "status": COMPLETE_STATUS,
        "writer": writer,
        "completion_manifest_path": manifest_name,
        "completion_manifest_sha256": _sha256_path(manifest),
        "runner_sha256": (
            _sha256_path(runner_path)
            if runner_path is not None
            and runner_path.is_file()
            and not runner_path.is_symlink()
            else None
        ),
        "bundle_sha256": _bundle_hashes(namespace),
        "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    _write_json_atomic(seal_path, seal)
    return seal_path


def promote_staged_complete(namespace: Path) -> Path:
    """Atomically make a just-published staged seal admissible."""
    namespace = namespace.absolute()
    if namespace.is_symlink() or not namespace.is_dir():
        raise ValueError(f"cannot promote unsafe E8 namespace: {namespace}")
    seal_path = namespace / "run_seal.json"
    if seal_path.is_symlink() or not seal_path.is_file():
        raise ValueError("published E8 namespace lacks a safe root run seal")
    seal = _load_object(seal_path, label="staged E8 run seal")
    if (
        seal.get("schema") != RUN_SEAL_SCHEMA
        or seal.get("status") != STAGED_COMPLETE_STATUS
    ):
        raise ValueError("published E8 namespace does not have a staged-complete seal")
    seal["status"] = COMPLETE_STATUS
    seal["published_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    _write_json_atomic(seal_path, seal)
    return seal_path


def record_terminal_abort(
    namespace: Path,
    *,
    writer: str,
    error: BaseException,
    marker_name: str = "writer_abort.json",
    marker_payload: dict[str, Any] | None = None,
    runner_path: Path | None = None,
) -> Path:
    """Atomically seal one real namespace as terminal and non-admissible."""
    namespace = namespace.absolute()
    if namespace.is_symlink() or not namespace.is_dir():
        raise ValueError(f"cannot terminalize unsafe E8 namespace: {namespace}")
    if Path(marker_name).name != marker_name or marker_name == "run_seal.json":
        raise ValueError("E8 abort marker must be one safe basename")

    error_type = f"{type(error).__module__}.{type(error).__qualname__}"
    error_sha256 = hashlib.sha256(f"{error_type}:{error}".encode()).hexdigest()
    payload = marker_payload or {
        "schema": "epyc.e8_quality_writer_abort.v1",
        "status": TERMINAL_STATUS,
        "writer": writer,
        "error_type": error_type,
        "error_sha256": error_sha256,
        "no_auto_retry": True,
        "no_admission": True,
    }
    required_marker = {
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "writer": writer,
        "error_type": error_type,
        "error_sha256": error_sha256,
        "no_auto_retry": True,
        "no_admission": True,
    }
    if any(value is None for value in required_marker.values()):
        raise ValueError("E8 abort marker payload omits a required field")
    payload = {**payload, **required_marker}

    marker = namespace / marker_name
    if marker.is_symlink():
        raise ValueError("E8 abort marker must not be a symlink")
    if not marker.exists():
        _write_json_atomic(marker, payload)
    elif not marker.is_file():
        raise ValueError("E8 abort marker must be a regular file")
    else:
        existing_marker = _load_object(marker, label="E8 abort marker")
        mismatches = {
            key: {"expected": expected, "actual": existing_marker.get(key)}
            for key, expected in required_marker.items()
            if existing_marker.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "existing E8 abort marker contradicts requested terminal seal: "
                + json.dumps(mismatches, sort_keys=True)
            )

    seal_path = namespace / "run_seal.json"
    if seal_path.is_symlink():
        raise ValueError("E8 run seal must not be a symlink")
    prior_seal_sha256 = _sha256_path(seal_path) if seal_path.is_file() else None
    prior_status: Any = None
    if prior_seal_sha256 is not None:
        try:
            prior_status = _load_object(
                seal_path,
                label="superseded E8 run seal",
            ).get("status")
        except ValueError:
            prior_status = "unparseable"

    seal = {
        "schema": RUN_SEAL_SCHEMA,
        "status": TERMINAL_STATUS,
        "writer": writer,
        "error_type": error_type,
        "error_sha256": error_sha256,
        "abort_marker_path": marker_name,
        "abort_marker_sha256": _sha256_path(marker),
        "runner_sha256": (
            _sha256_path(runner_path)
            if runner_path is not None and runner_path.is_file() and not runner_path.is_symlink()
            else None
        ),
        "bundle_sha256": _bundle_hashes(namespace),
        "superseded_run_seal_sha256": prior_seal_sha256,
        "superseded_run_seal_status": prior_status,
        "no_auto_retry": True,
        "no_admission": True,
        "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    _write_json_atomic(seal_path, seal)
    return seal_path


def durable_candidate_writer(
    writer: str,
    *,
    marker_name: str,
    marker_schema: str,
    marker_status: str,
    runner_path: Path | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Seal every output namespace created by a failed writer invocation."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapped(args: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            output_value = getattr(args, "output_dir", None)
            if output_value is None:
                return function(args, *call_args, **call_kwargs)
            output = Path(output_value).absolute()
            staging_pattern = f".{output.name}.staging-*"
            existing_staging = set(output.parent.glob(staging_pattern))
            output_existed = output.exists() or output.is_symlink()

            def seal_created(failure: BaseException) -> None:
                candidates = sorted(
                    set(output.parent.glob(staging_pattern)) - existing_staging,
                    key=str,
                )
                if not output_existed and (output.exists() or output.is_symlink()):
                    candidates.append(output)
                for candidate in candidates:
                    try:
                        record_terminal_abort(
                            candidate,
                            writer=writer,
                            error=failure,
                            marker_name=marker_name,
                            marker_payload={
                                "schema": marker_schema,
                                "status": marker_status,
                                "writer": writer,
                                "error_type": (
                                    f"{type(failure).__module__}."
                                    f"{type(failure).__qualname__}"
                                ),
                                "error_sha256": hashlib.sha256(
                                    (
                                        f"{type(failure).__module__}."
                                        f"{type(failure).__qualname__}:{failure}"
                                    ).encode()
                                ).hexdigest(),
                                "no_auto_retry": True,
                                "no_admission": True,
                            },
                            runner_path=runner_path,
                        )
                    except BaseException as seal_error:
                        failure.add_note(
                            f"failed to terminally seal {candidate}: {seal_error}"
                        )

            try:
                result = function(args, *call_args, **call_kwargs)
            except BaseException as exc:
                seal_created(exc)
                raise
            status = (
                result[-1]
                if isinstance(result, tuple)
                and result
                and isinstance(result[-1], int)
                and not isinstance(result[-1], bool)
                else None
            )
            if result is False or (status is not None and status != 0):
                seal_created(
                    RuntimeError(
                        f"{writer} returned non-success status "
                        f"{status if status is not None else result!r}"
                    )
                )
            return result

        return wrapped

    return decorate
