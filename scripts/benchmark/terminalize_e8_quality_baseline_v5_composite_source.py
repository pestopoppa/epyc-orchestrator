#!/usr/bin/env python3
"""Publish the reviewed E8 composite source as a sealed non-staging copy."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINALIZER_PATH = (
    PROJECT_ROOT / "scripts/benchmark/finalize_e8_quality_baseline_v5_recovery_r2.py"
)
TERMINAL_SEAL_PATH = PROJECT_ROOT / "scripts/benchmark/e8_terminal_seal.py"
MANIFEST_NAME = "composite_source_manifest.json"
MANIFEST_SCHEMA = "epyc.e8_quality_v5_composite_source_terminal.v1"
WRITER = "terminalize_e8_quality_baseline_v5_composite_source"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FINALIZER = _load(FINALIZER_PATH, "e8_composite_source_terminalizer_finalizer")
TERMINAL_SEAL = _load(TERMINAL_SEAL_PATH, "e8_composite_source_terminal_seal")


def _validate_source_unchanged(source: Path, hashes: dict[str, str]) -> None:
    actual = {
        item.relative_to(source).as_posix(): FINALIZER.sha256_path(item)
        for item in sorted(source.rglob("*"))
        if item.is_file() and not item.is_symlink()
    }
    if any(item.is_symlink() for item in source.rglob("*")) or actual != hashes:
        raise ValueError("composite source changed during terminalization")


def _copy_exact(source: Path, destination: Path, hashes: dict[str, str]) -> None:
    for relative, digest in hashes.items():
        origin = source / relative
        target = destination / relative
        if FINALIZER.sha256_path(origin) != digest:
            raise ValueError("composite source changed before terminalization")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        if (
            FINALIZER.sha256_path(origin) != digest
            or FINALIZER.sha256_path(target) != digest
        ):
            raise ValueError("composite source terminal copy differs")
    copied = {
        item.relative_to(destination).as_posix(): FINALIZER.sha256_path(item)
        for item in sorted(destination.rglob("*"))
        if item.is_file() and not item.is_symlink()
    }
    if copied != hashes:
        raise ValueError("composite source terminal copy artifact set differs")
    _validate_source_unchanged(source, hashes)


@TERMINAL_SEAL.durable_candidate_writer(
    WRITER,
    marker_name="writer_abort.json",
    marker_schema="epyc.e8_quality_writer_abort.v1",
    marker_status=TERMINAL_SEAL.TERMINAL_STATUS,
    runner_path=Path(__file__).resolve(),
)
def execute(args: argparse.Namespace) -> Path:
    destination = args.output_dir.absolute()
    if destination.name.startswith(".") or ".staging-" in destination.name:
        raise ValueError("composite source successor destination must be non-staging")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"composite source successor already exists: {destination}")
    validated = FINALIZER.validate_legacy_composite_source(args.source_dir)
    source = Path(validated["source"]).resolve(strict=True)
    staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    FINALIZER.V4.fsync_dir(staging.parent)
    try:
        _copy_exact(
            source,
            staging / "source_snapshot",
            validated["source_sha256"],
        )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "complete",
            "historical_source": str(source),
            "source_snapshot": "source_snapshot",
            "source_tree_sha256": validated["source_tree_sha256"],
            "source_sha256": validated["source_sha256"],
            "terminalizer": {
                "path": str(Path(__file__).resolve()),
                "sha256": FINALIZER.sha256_path(Path(__file__).resolve()),
            },
        }
        FINALIZER.RECOVERY._write_json(staging / MANIFEST_NAME, manifest)
        FINALIZER.V4.fsync_dir(staging)
        FINALIZER.V4.atomic_publish_noreplace(staging, destination)
        FINALIZER.V4.fsync_dir(destination.parent)
        _validate_source_unchanged(source, validated["source_sha256"])
        TERMINAL_SEAL.record_complete(
            destination,
            writer=WRITER,
            manifest_name=MANIFEST_NAME,
            runner_path=Path(__file__).resolve(),
        )
        FINALIZER._validate_standard_complete_seal(
            destination,
            expected_writer=WRITER,
            expected_manifest_name=MANIFEST_NAME,
            expected_manifest_schema=MANIFEST_SCHEMA,
        )
        _validate_source_unchanged(source, validated["source_sha256"])
        return destination
    except BaseException:
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=FINALIZER.COMPOSITE_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    output = execute(parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
