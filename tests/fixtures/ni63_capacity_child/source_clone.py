"""Build an external, source-pinned clone for NI63's import-capacity child probes.

This fixture copies the Python modules in the import-time caller closure plus the
lazy compiler/resolver modules instrumented by the side-effect probe. It also
copies the immutable declared GPU-capacity policy input and owned YAML fixtures.
It reads real /proc/meminfo; it never edits or substitutes host memory values.
"""

from __future__ import annotations

import hashlib
import os
import sysconfig
import tempfile
from pathlib import Path

PINNED_APP_COMMIT = "6bb8d860d410217895efb7806e9ef907a62a4350"  # original NI63 fixture baseline
# (Git blob SHA-1, content SHA-256), pinned per reviewed import-closure source.
SOURCE_PINS: dict[str, tuple[str, str]] = {
    "scripts/server/bench_core_claim.py": ("7c03af5a90825ec7acd46b4ab3b4600828a75eba", "252c6dfc37bb4c34416cd7ba9f16158eb8a248ad4b56a9bfe94fd21466a34f16"),
    "scripts/server/fleet_markers.py": ("070b6ab8af62fbd33cffa4069133cca915c3258e", "8ef8df7703252f41f52c2f9033e941d28d4fea7017d220a7e3dc617e1a2aec68"),
    "scripts/server/orchestrator_stack.py": ("a95a0dd5cddc09f44069241198414aac246fd2e5", "a2e8e1cf04cb5218aac16d7d0c9858aa4a09c1e68637533da03db8c608bfbcd1"),
    "scripts/server/realized_fleet.py": ("12f01d3c17a2ccb13c193728d79e24d52148de4e", "9099aa115a22799ba2446bf2082e0d7ebb360a0df5f57aac654b6286c1ea9a31"),
    "scripts/server/runtime_facts_manifest.py": ("c18bb563aabfa092e6b378fdddbd5f516352cd5d", "e2bfec7bb580fac727ca184b2dd4d56bd487e443e89d4720cc037743ab1dc374"),
    "scripts/server/stack_env.py": ("0ad2da92f14a50e2145e36fe32e31011d0da8606", "c2dd2a40abf2ac3048e392a11bd910a27f6991e2a99cefdbd232172b67ea06a4"),
    "scripts/server/stack_health.py": ("99d48f391565e0befb10040a0fea252087fc0313", "19ede42e88efeda2ab1fc5eb82c1272ac9d992f59313103c7d8d83917d4dfcad"),
    "scripts/server/stack_log_banner.py": ("1c67ef1c1b4a48b8afa315e34b88511e0fa012ed", "cfec940567bdd359459537e2e3c6e964b05e3ea1136947ba67f3d0cc6b611221"),
    "scripts/server/stack_manifest.py": ("6b92dec62271fe9936d6bd08caeb8816da821297", "1fb4257afddc16307d4df81bffa9312b95f606cda2dc41461bddf5f96407ca41"),
    "scripts/server/stack_numa.py": ("78ba16a1d289fa7f44480aa5f0f225ce70103908", "edfdb41035ec638d3eab60631024ebece56a5e2d9b577851bf02e05a446e0c32"),
    "scripts/server/stack_numa_evict.py": ("09116f5040a575405f0d7807bb7af211085f906b", "575b811b72b31ed95f7296ee7aecb9e245aae9d603496e4baa1137ab42cb77de"),
    "scripts/server/stack_numa_mode.py": ("115bb4c3d5b61802452512f98819fbf86ae67e9c", "e17e1c7260dda722b5e1f555e5fc92204a4e8d93a397a7128a0ed76c057afe63"),
    "scripts/server/stack_paths.py": ("7cd5921700b10ebaebd91752a744631d97fae3a9", "3980a9c9860fca6555d345b26f63ed5bf98d671ff2e54969cd49e9f0b2686a58"),
    "scripts/server/stack_processes.py": ("df278a70d5801b81dc54c5ec663d19859ad4a50e", "e9ac8d7951fb6a11664c1d69e71050dfd44ed51ae3e215858ffd698198459812"),
    "scripts/server/stack_runtime.py": ("2d4489a3d4e6046973255dbf2b9a2d47b65c6a7e", "984d27262e1f1d5a5332d0b4a3e6fb96fb8095fc890ddaa93da07bff52eb1c99"),
    "scripts/server/stack_state.py": ("5d7ad9483260e4e56f8105df702fc5ff1b7f782f", "a047f152ac478e8fead80700b73bcb782dfe61969f513aa69da422518c89da8c"),
    "scripts/server/launch_affinity.py": ("c78ceb5f327491696d314153b4e3280a5ad96ee0", "f1ef273e86a4f7121e588d79a34a9d72ec9150889c72344b950f4608cca091da"),
    "src/__init__.py": ("a43657f16f7063ccc1d61f7048b64c55433eddce", "26adaa6226a4b680574c4a105785b954465ea8a532a91d76aa9944e985bfb3ad"),
    "src/config/__init__.py": ("644ecc9e4ee3e0afeb3235be651976a3562eb8ca", "e2c868f3481591c5b61a420b92bc2dba7e6a0699ef5fa93bade89a9672302831"),
    "src/config/models.py": ("fe17d91959c26dac8065f3b98d941392b2e75928", "49a29f9156e5112b5344e6f5ddc6b759674eab9d95f4d4224deeaafb51674cc0"),
    "src/config/validation.py": ("eab4945f939d5aa2f35cd167dcece1a55c5546ff", "e69b9b2ab0b38e0e2b50bcc435a19695ed20eb627b69fbb4dd302bea680bc51d"),
    "src/env_parsing.py": ("11e3b7c80df1a6f921d467f7a924d287e1ac28dc", "2626fb2c47012c56a7f7e99fe03f088977f9b061c77bcd1edf0dfa1e12b7d52f"),
    "src/registry/__init__.py": ("287333a63889217a52c0082a7e316dc64670ee1f", "50ddc75e291c2caaae688bbe94c4f3144a5751eb11e0a65ee35b5632ea3a9c6d"),
    "src/registry/kernel_paths.py": ("fb006e99f7ab8f3706f1371934e51800256988eb", "3993e914d5436f753a114bd6d92aaf727c75230baa1efe32d28351183b42a379"),
    "src/registry/registry_loader.py": ("dd035c4e19fe1ee7c39fd842e7cb1578e0d4091a", "e8ae270fb6fc04a7bc1501c3207eb96f1df4a0fde47949ee0be50b3de36496a9"),
    "src/registry/model_descriptors.py": ("924ff6c98279a8e59ec7d006f3edc20dcb80e523", "f7eac72c2cf8b9549005f4195c2c143fd22ae9e1dad9de65bef141cce4a193b1"),
    "src/registry/research_origin.py": ("f41a8de1d31942d223ad1e86a8415c43e6a005ec", "6b2f3aa7844887b491f8883878f97f2563cced61c5bb2bd51dc5e23f8027d226"),
    "src/registry/stack_priors.py": ("c4f48f3d67eacf3fc067ae73dd206d73025f7f5d", "a5bf6f82c3ba9ae50ce8e034e1c71d51ba3275486518b07b7cdf98b767a55ef7"),
    "src/registry_loader.py": ("c68c5918b94e4a54a07a7c6642d92365480bf656", "81f3dc46dea4af9eb8df9adb4a73d2e900e1d040d107a73a04053f8ea72b77be"),
    "src/roles.py": ("59432d52c3c5ebb661c336f70627267073cfa4cb", "408f10be964705638562ec9a6d0e1cd56b26ac849fdb964e8bac0c9a7fe4be61"),
    "src/scheduling/__init__.py": ("e81d5c94b0479ae82472dfceec8393cd8f8286d2", "a65b963b3d38ca2d680827bfe3debe3e7d91095ea50723c5c40ce6b19dd0df51"),
    "src/scheduling/device_model.py": ("78e0ede88801c39a842fadcbf6b4322d1c1a83ac", "90e3f9e001bbcb940c4cb493caa4df29b4da244b7d9834e6b09632c9fb49171b"),
}
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

    for relative, (expected_blob, expected_sha256) in SOURCE_PINS.items():
        source = source_root / relative
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"pinned source is absent or linked: {relative}")
        contents = source.read_bytes()
        got_blob = _git_blob_sha1(contents)
        got_sha256 = hashlib.sha256(contents).hexdigest()
        if (got_blob, got_sha256) != (expected_blob, expected_sha256):
            raise RuntimeError(f"source differs from accepted APP pin: {relative}")
        target = clone / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(contents)
        target.chmod(0o600)

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
