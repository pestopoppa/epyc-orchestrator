"""Build an external, source-derived clone for NI63's import-capacity child probes.

This fixture copies the Python modules in the import-time caller closure plus the
lazy compiler/resolver modules instrumented by the side-effect probe. It also
copies the immutable declared GPU-capacity policy input and owned YAML fixtures.
It reads real /proc/meminfo; it never edits or substitutes host memory values.
The module closure is fixed, while source hashes are captured from the current
checkout and written into the private clone's provenance record.
"""

from __future__ import annotations

import hashlib
import json
import os
import sysconfig
import tempfile
from pathlib import Path

# The closure membership is the independent contract; content hashes are recorded
# from the active checkout for each clone so routine accepted edits do not stale the
# test harness or weaken the behavioral probes below.
SOURCE_CLOSURE: tuple[str, ...] = (
    "scripts/server/bench_core_claim.py",
    "scripts/server/fleet_markers.py",
    "scripts/server/orchestrator_stack.py",
    "scripts/server/realized_fleet.py",
    "scripts/server/runtime_facts_manifest.py",
    "scripts/server/stack_env.py",
    "scripts/server/stack_health.py",
    "scripts/server/stack_log_banner.py",
    "scripts/server/stack_manifest.py",
    "scripts/server/stack_numa.py",
    "scripts/server/stack_numa_evict.py",
    "scripts/server/stack_numa_mode.py",
    "scripts/server/stack_paths.py",
    "scripts/server/stack_processes.py",
    "scripts/server/stack_runtime.py",
    "scripts/server/stack_state.py",
    "scripts/server/launch_affinity.py",
    "src/__init__.py",
    "src/config/__init__.py",
    "src/config/models.py",
    "src/config/validation.py",
    "src/env_parsing.py",
    "src/registry/__init__.py",
    "src/registry/kernel_paths.py",
    "src/registry/registry_loader.py",
    "src/registry/model_descriptors.py",
    "src/registry/research_origin.py",
    "src/registry/stack_priors.py",
    "src/registry_loader.py",
    "src/roles.py",
    "src/scheduling/__init__.py",
    "src/scheduling/device_model.py",
)
CAPACITY_ARTIFACT_PIN = (
    "orchestration/gpu_shadow_lane_np_ceiling.yaml",
    "0bf348c1dae4bd44698a2ed9d3920c98d9eb1bc0",
    "deeb744c4a411bc0f364d1209c402c29504c23a19c96eddc22b9b0bd0326c04f",
)
FIXTURE_PINS: dict[str, str] = {
    "launch_manifest.yaml": "583bbb9ee8910c5633e454f5a3528e4f9a86146b6641448e92ebf6b573cfd41b",
    "stack_topology.yaml": "d4fcf1ee0583df9072450129278b55339df3f9f115d3c18529e72f29b44135e4",
    "model_registry_fit.yaml": "b1afae32f0cd1256eef7e43e55fb25241e96f0b962adc87896fd2ce3a4c41080",
    "model_registry_oversized.yaml": "5547c56b5730ca528dfd446a7f1c655d64eecb4fe3e303275f2f73e0526c8b64",
}


def _git_blob_sha1(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def read_real_meminfo() -> dict[str, int]:
    """Read the runner's actual MemTotal and MemAvailable in kB, without writes."""
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            result[key] = int(rest.split()[0])
    if set(result) != {"MemTotal", "MemAvailable"}:
        raise RuntimeError("/proc/meminfo did not expose both required host facts")
    return result


def assert_real_available_headroom(
    required_gib: float, reserve_gib: float, meminfo_kb: dict[str, int]
) -> None:
    """Require actual MemAvailable to cover fixture arithmetic plus its reserve."""
    total_gib = meminfo_kb["MemTotal"] / (1024.0 * 1024.0)
    available_gib = meminfo_kb["MemAvailable"] / (1024.0 * 1024.0)
    if available_gib > total_gib:
        raise AssertionError("actual MemAvailable exceeds actual MemTotal")
    if required_gib + reserve_gib > available_gib:
        raise AssertionError(
            f"fixture needs {required_gib:.6f} GiB plus {reserve_gib:.3f} GiB reserve, "
            f"but actual MemAvailable is {available_gib:.3f} GiB "
            f"of {total_gib:.3f} GiB MemTotal"
        )


def materialize_source_clone(
    source_root: Path,
    private_parent: Path,
    *,
    variant: str,
    project_venv_python: Path,
) -> Path:
    """Copy only pinned import-time Python sources and the selected fixture data."""
    if variant not in {"fit", "oversized"}:
        raise ValueError("variant must be 'fit' or 'oversized'")
    source_root = source_root.resolve(strict=True)
    private_parent = private_parent.resolve(strict=True)
    if private_parent == source_root or source_root in private_parent.parents:
        raise ValueError("fixture source clone must live outside the checked-out repository")
    data_root = Path(__file__).resolve().parent
    clone = Path(tempfile.mkdtemp(prefix="ni63-capacity-child-", dir=private_parent))
    clone.chmod(0o700)

    source_records: dict[str, dict[str, str]] = {}
    for relative in SOURCE_CLOSURE:
        source = source_root / relative
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"source is absent or linked: {relative}")
        contents = source.read_bytes()
        source_records[relative] = {
            "git_blob_sha1": _git_blob_sha1(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }
        target = clone / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(contents)
        target.chmod(0o600)

    provenance = {
        "schema": "ni63-source-provenance.v1",
        "source_root": str(source_root),
        "files": source_records,
    }
    (clone / "source-provenance.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (clone / "source-provenance.json").chmod(0o600)

    for fixture_name, expected_sha256 in FIXTURE_PINS.items():
        source = data_root / fixture_name
        contents = source.read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected_sha256:
            raise RuntimeError(f"fixture bytes changed without updating the pin: {fixture_name}")
        if fixture_name.startswith("model_registry_"):
            if (variant == "fit") != (fixture_name == "model_registry_fit.yaml"):
                continue
            target_name = "model_registry.yaml"
        else:
            target_name = fixture_name
        target = clone / "orchestration" / target_name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(contents)
        target.chmod(0o600)

    capacity_path, capacity_blob, capacity_sha256 = CAPACITY_ARTIFACT_PIN
    capacity_source = source_root / capacity_path
    contents = capacity_source.read_bytes()
    if (_git_blob_sha1(contents), hashlib.sha256(contents).hexdigest()) != (
        capacity_blob,
        capacity_sha256,
    ):
        raise RuntimeError("declared GPU-capacity source differs from accepted APP pin")
    capacity_target = clone / capacity_path
    capacity_target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    capacity_target.write_bytes(contents)
    capacity_target.chmod(0o600)

    real_venv_python = project_venv_python.resolve(strict=True)
    venv_link = clone / ".venv" / "bin" / "python"
    venv_link.parent.mkdir(mode=0o700, parents=True)
    venv_link.symlink_to(real_venv_python)
    (clone / "scratch" / "tmp").mkdir(mode=0o700, parents=True)
    (clone / "scratch" / "logs").mkdir(mode=0o700, parents=True)
    (clone / "scratch" / "cache").mkdir(mode=0o700, parents=True)
    return clone


def probe_environment(clone: Path) -> dict[str, str]:
    """Explicit child allowlist; preserve inherited HOME and exclude provider/memory overrides."""
    clone = clone.resolve(strict=True)
    purelib = sysconfig.get_paths()["purelib"]
    scratch = clone / "scratch"
    absent = clone / "absent"
    env = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(Path(purelib).resolve(strict=True)),
        "TMPDIR": str(scratch / "tmp"),
        "ORCHESTRATOR_PATHS_PROJECT_ROOT": str(clone),
        "ORCHESTRATOR_PATHS_LLM_ROOT": str(absent / "llm"),
        "ORCHESTRATOR_PATHS_MODELS_DIR": str(absent / "models"),
        "ORCHESTRATOR_PATHS_MODEL_BASE": str(absent / "model-base"),
        "ORCHESTRATOR_PATHS_LLAMA_CPP_BIN": str(absent / "kernel-bin"),
        "ORCHESTRATOR_PATHS_LLAMA_MTMD": str(absent / "llama-mtmd"),
        "ORCHESTRATOR_PATHS_LOG_DIR": str(scratch / "logs"),
        "ORCHESTRATOR_PATHS_CACHE_DIR": str(scratch / "cache"),
        "ORCHESTRATOR_PATHS_TMP_DIR": str(scratch / "tmp"),
        "ORCHESTRATOR_PATHS_REGISTRY_PATH": str(clone / "orchestration" / "model_registry.yaml"),
        "ORCHESTRATOR_PATHS_TOOL_REGISTRY_PATH": str(absent / "tool_registry.yaml"),
        "ORCHESTRATOR_PATHS_SCRIPT_REGISTRY_DIR": str(absent / "script-registry"),
        "ORCHESTRATOR_PATHS_STACK_PRIORS_PATH": str(absent / "stack-priors.yaml"),
        "ORCHESTRATOR_PATHS_VISION_DIR": str(absent / "vision"),
        "ORCHESTRATOR_PATHS_RAID_PREFIX": str(absent / "raid"),
    }
    forbidden = {
        "HOME",
        "ORCHESTRATOR_STACK_REEXEC",
        "ORCHESTRATOR_HOST_MEMTOTAL_GIB",
        "ORCHESTRATOR_HOST_MEMAVAILABLE_GIB",
        "ORCHESTRATOR_VRAM_TOTAL_GIB",
        "ORCHESTRATOR_VRAM_HEADROOM_GIB",
    }
    if forbidden & env.keys():
        raise AssertionError(f"probe allowlist contains forbidden overrides: {sorted(forbidden & env.keys())}")
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    if (absent / "kernel-bin").exists():
        raise AssertionError("fixture kernel path unexpectedly exists")
    return env
