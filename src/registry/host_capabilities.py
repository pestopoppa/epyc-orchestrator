"""RTG-VAR-1: host-capability manifest, a compiled input of the registry compiler.

``orchestration/host_capabilities.yaml`` records what the (single, fixed) serving host can
do: CPU model and the ISA flags the quant kernels branch on, logical CPU count, NUMA nodes
(cpulist and RAM per node), total RAM, GPU architecture and VRAM, ROCm version. It is the
input RTG-VAR-2 needs to decide which (model, quant, backend, device) variants are
servable here.

Design:

* ``build_manifest`` is PURE: it turns raw probe text (``/proc/cpuinfo``, ``/proc/meminfo``,
  per-node ``cpulist``/``meminfo``, ``rocminfo``, ``rocm-smi --json``, the ROCm version
  file) into the manifest dict. No clock, no host name, no PIDs: the same probe text always
  yields byte-identical YAML, which is what "generated reproducibly" means here.
* ``probe_host`` only READS: sysfs/procfs files and two read-only ROCm queries. It never
  starts, stops or reloads anything. A missing GPU stack yields ``gpu: []`` with a recorded
  ``probe_gaps`` entry, never an invented value.
* Quantities that wobble between boots (``MemTotal`` loses a few MiB to firmware and
  kernel reservations) are rounded to whole GiB so the committed manifest does not churn.
* The registry compiler consumes it through ``load_host_capabilities`` (opt-in:
  ``compile_descriptors.py --host-capabilities``), which records the manifest's sha256 and
  headline facts in the descriptor metadata. Default compile output is unchanged.

CLI: ``scripts/registry/compile_host_capabilities.py`` (``--dry-run`` prints, ``--check``
exits non-zero when the committed file differs from a fresh probe).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = _REPO_ROOT / "orchestration/host_capabilities.yaml"
MANIFEST_VERSION = 1

# The ISA flags the quantized kernels (and the iqk/AVX-512 paths) branch on. Other cpuinfo
# flags are deliberately not recorded: they would churn the manifest without informing any
# variant decision.
RELEVANT_CPU_FLAGS = frozenset(
    {
        "avx", "avx2", "fma", "f16c", "bmi1", "bmi2", "avx_vnni",
        "avx512f", "avx512dq", "avx512cd", "avx512bw", "avx512vl", "avx512ifma",
        "avx512vbmi", "avx512_vbmi2", "avx512_vnni", "avx512_bf16", "avx512_bitalg",
        "avx512_vpopcntdq", "avx512_fp16", "amx_tile", "amx_int8", "amx_bf16",
    }
)

_KIB_PER_GIB = 1024 * 1024
_BYTES_PER_GIB = 1024**3


def _gib_from_kib(kib: int) -> int:
    return int(round(kib / _KIB_PER_GIB))


def _gib_from_bytes(n: int) -> int:
    return int(round(n / _BYTES_PER_GIB))


def parse_cpuinfo(text: str) -> dict[str, Any]:
    """Model name, sorted relevant ISA flags and logical CPU count from ``/proc/cpuinfo``."""
    model: str | None = None
    flags: set[str] = set()
    logical = 0
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if key == "processor":
            logical += 1
        elif key == "model name" and model is None:
            model = value
        elif key == "flags" and not flags:
            flags = set(value.split()) & RELEVANT_CPU_FLAGS
    return {"model": model, "logical_cpus": logical, "isa_flags": sorted(flags)}


def parse_meminfo_total_kib(text: str) -> int | None:
    m = re.search(r"^(?:Node\s+\d+\s+)?MemTotal:\s+(\d+)\s*kB", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def parse_rocminfo_gpus(text: str) -> list[dict[str, Any]]:
    """GPU agents (``Device Type: GPU``) from ``rocminfo``: arch, marketing name, CU count."""
    gpus: list[dict[str, Any]] = []
    for block in re.split(r"\*{5,}\s*\nAgent \d+\s*\n\*{5,}", text):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() and key.strip() not in fields:
                fields[key.strip()] = value.strip()
        if fields.get("Device Type") != "GPU":
            continue
        cu = fields.get("Compute Unit")
        gpus.append(
            {
                "arch": fields.get("Name"),
                "name": fields.get("Marketing Name"),
                "compute_units": int(cu) if cu and cu.isdigit() else None,
            }
        )
    return gpus


def parse_rocm_smi_vram_gib(text: str) -> dict[int, int]:
    """``{card_index: VRAM GiB}`` from ``rocm-smi --showmeminfo vram --json``."""
    out: dict[int, int] = {}
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return out
    if not isinstance(data, dict):
        return out
    for key, val in data.items():
        m = re.fullmatch(r"card(\d+)", str(key))
        if not m or not isinstance(val, dict):
            continue
        raw = val.get("VRAM Total Memory (B)")
        try:
            out[int(m.group(1))] = _gib_from_bytes(int(raw))
        except (TypeError, ValueError):
            continue
    return out


def build_manifest(
    *,
    cpuinfo: str | None,
    meminfo: str | None,
    numa_nodes: Mapping[int, tuple[str, str]] | None,
    rocminfo: str | None,
    rocm_smi_vram_json: str | None,
    rocm_version: str | None,
) -> dict[str, Any]:
    """Pure: assemble the manifest from raw probe text. ``None`` means the probe was unavailable."""
    gaps: list[str] = []

    if cpuinfo:
        cpu = parse_cpuinfo(cpuinfo)
    else:
        cpu = {"model": None, "logical_cpus": None, "isa_flags": []}
        gaps.append("cpuinfo unavailable")

    ram_kib = parse_meminfo_total_kib(meminfo) if meminfo else None
    if ram_kib is None:
        gaps.append("meminfo unavailable")

    nodes: list[dict[str, Any]] = []
    for idx in sorted(numa_nodes or {}):
        cpulist, node_meminfo = numa_nodes[idx]  # type: ignore[index]
        node_kib = parse_meminfo_total_kib(node_meminfo)
        nodes.append(
            {
                "node": idx,
                "cpulist": cpulist.strip(),
                "ram_gib": _gib_from_kib(node_kib) if node_kib is not None else None,
            }
        )
    if not nodes:
        gaps.append("numa topology unavailable")

    gpus = parse_rocminfo_gpus(rocminfo) if rocminfo else []
    vram = parse_rocm_smi_vram_gib(rocm_smi_vram_json) if rocm_smi_vram_json else {}
    gpu_rows = []
    for i, gpu in enumerate(gpus):
        row = dict(gpu)
        row["index"] = i
        row["vram_gib"] = vram.get(i)
        if row["vram_gib"] is None:
            gaps.append(f"gpu {i} vram unavailable")
        gpu_rows.append(row)
    if not gpu_rows:
        gaps.append("no gpu agent reported")

    return {
        "manifest_version": MANIFEST_VERSION,
        "cpu": {
            "model": cpu["model"],
            "logical_cpus": cpu["logical_cpus"],
            "isa_flags": cpu["isa_flags"],
        },
        "ram_gib": _gib_from_kib(ram_kib) if ram_kib is not None else None,
        "numa_nodes": nodes,
        "gpu": gpu_rows,
        "rocm_version": (rocm_version or "").strip() or None,
        "probe_gaps": sorted(set(gaps)),
    }


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _run(argv: list[str]) -> str | None:
    """READ-ONLY query; any failure is a probe gap, not an error."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def probe_host(
    *,
    proc_root: Path = Path("/proc"),
    node_root: Path = Path("/sys/devices/system/node"),
    rocm_version_path: Path = Path("/opt/rocm/.info/version"),
    run: Callable[[list[str]], str | None] = _run,
) -> dict[str, Any]:
    """Read the live host (read-only) and build the manifest."""
    numa: dict[int, tuple[str, str]] = {}
    if node_root.is_dir():
        for d in node_root.glob("node[0-9]*"):
            cpulist = _read(d / "cpulist")
            meminfo = _read(d / "meminfo")
            if cpulist is not None and meminfo is not None:
                numa[int(d.name[4:])] = (cpulist, meminfo)
    return build_manifest(
        cpuinfo=_read(proc_root / "cpuinfo"),
        meminfo=_read(proc_root / "meminfo"),
        numa_nodes=numa,
        rocminfo=run(["rocminfo"]),
        rocm_smi_vram_json=run(["rocm-smi", "--showmeminfo", "vram", "--json"]),
        rocm_version=_read(rocm_version_path),
    )


def render_manifest(manifest: Mapping[str, Any]) -> str:
    """Deterministic YAML text (fixed key order, no timestamp)."""
    return yaml.safe_dump(
        dict(manifest), sort_keys=False, default_flow_style=False, allow_unicode=True, width=200
    )


def load_host_capabilities(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Load the committed manifest and return the compiler-facing summary (with sha256)."""
    raw = path.read_bytes()
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict) or data.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"{path}: not a v{MANIFEST_VERSION} host-capability manifest")
    gpus = data.get("gpu") or []
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "cpu_model": (data.get("cpu") or {}).get("model"),
        "ram_gib": data.get("ram_gib"),
        "numa_nodes": len(data.get("numa_nodes") or []),
        "gpu_arch": sorted({g.get("arch") for g in gpus if g.get("arch")}),
        "gpu_vram_gib": [g.get("vram_gib") for g in gpus],
        "rocm_version": data.get("rocm_version"),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Probe the host and emit host_capabilities.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--dry-run", action="store_true", help="print instead of writing")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 when the committed file differs from a fresh probe"
    )
    args = parser.parse_args(argv)
    text = render_manifest(probe_host())
    if args.dry_run:
        sys.stdout.write(text)
        return 0
    if args.check:
        current = _read(args.output)
        if current != text:
            print(f"DRIFT: {args.output} differs from a fresh probe", file=sys.stderr)
            return 1
        print(f"OK: {args.output} matches a fresh probe")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"OK: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
