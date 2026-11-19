"""RTG-VAR-1: host-capability manifest (pure builder, reproducibility, golden shape, compiler hook)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.registry import host_capabilities as hc

CPUINFO = (
    "processor\t: 0\nmodel name\t: AMD EPYC 9655 96-Core Processor\n"
    "flags\t\t: fpu sse4_2 fma avx avx2 avx512f avx512_bf16 avx512_vnni bmi2 hypervisor\n\n"
    "processor\t: 1\nmodel name\t: AMD EPYC 9655 96-Core Processor\n"
    "flags\t\t: fpu sse4_2 fma avx avx2 avx512f avx512_bf16 avx512_vnni bmi2 hypervisor\n"
)
MEMINFO = "MemTotal:       1188473556 kB\nMemFree:  100 kB\n"
NODES = {
    1: ("24-47", "Node 1 MemTotal:       297254428 kB\n"),
    0: ("0-23", "Node 0 MemTotal:       296773668 kB\n"),
}
ROCMINFO = """\
*******
Agent 1
*******
  Name:                    AMD EPYC 9655 96-Core Processor
  Marketing Name:          AMD EPYC 9655 96-Core Processor
  Device Type:             CPU
  Compute Unit:            48
*******
Agent 2
*******
  Name:                    gfx90a
  Marketing Name:          AMD Instinct MI210
  Device Type:             GPU
  Compute Unit:            104
"""
SMI = json.dumps({"card0": {"VRAM Total Memory (B)": "68702699520"}})


def _manifest(**over):
    kw = dict(
        cpuinfo=CPUINFO, meminfo=MEMINFO, numa_nodes=NODES,
        rocminfo=ROCMINFO, rocm_smi_vram_json=SMI, rocm_version="6.2.0-66\n",
    )
    kw.update(over)
    return hc.build_manifest(**kw)


def test_golden_shape_and_values():
    m = _manifest()
    assert list(m) == [
        "manifest_version", "cpu", "ram_gib", "numa_nodes", "gpu", "rocm_version", "probe_gaps",
    ]
    assert list(m["cpu"]) == ["model", "logical_cpus", "isa_flags"]
    assert m["cpu"] == {
        "model": "AMD EPYC 9655 96-Core Processor",
        "logical_cpus": 2,
        # only the relevant flags, sorted; sse4_2/fpu/hypervisor are dropped
        "isa_flags": ["avx", "avx2", "avx512_bf16", "avx512_vnni", "avx512f", "bmi2", "fma"],
    }
    assert m["ram_gib"] == 1133
    assert m["numa_nodes"] == [
        {"node": 0, "cpulist": "0-23", "ram_gib": 283},
        {"node": 1, "cpulist": "24-47", "ram_gib": 283},
    ]
    assert m["gpu"] == [
        {"arch": "gfx90a", "name": "AMD Instinct MI210", "compute_units": 104, "index": 0, "vram_gib": 64}
    ]
    assert m["rocm_version"] == "6.2.0-66"
    assert m["probe_gaps"] == []


def test_reproducible_byte_identical():
    a = hc.render_manifest(_manifest())
    b = hc.render_manifest(_manifest(numa_nodes=dict(reversed(list(NODES.items())))))
    assert a == b


def test_missing_gpu_stack_is_a_recorded_gap_not_an_invented_value():
    m = _manifest(rocminfo=None, rocm_smi_vram_json=None, rocm_version=None)
    assert m["gpu"] == [] and m["rocm_version"] is None
    assert "no gpu agent reported" in m["probe_gaps"]


def test_ram_rounding_absorbs_boot_to_boot_wobble():
    a = _manifest(meminfo="MemTotal: 1188473556 kB\n")["ram_gib"]
    b = _manifest(meminfo="MemTotal: 1188400000 kB\n")["ram_gib"]
    assert a == b


def test_committed_manifest_has_the_pinned_shape():
    data = yaml.safe_load(hc.DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert list(data) == list(_manifest())
    assert data["manifest_version"] == hc.MANIFEST_VERSION
    assert set(data["cpu"]["isa_flags"]) <= hc.RELEVANT_CPU_FLAGS
    assert all(set(g) == {"arch", "name", "compute_units", "index", "vram_gib"} for g in data["gpu"])
    assert all(set(n) == {"node", "cpulist", "ram_gib"} for n in data["numa_nodes"])


def test_loader_summary_and_rejects_foreign_file(tmp_path: Path):
    s = hc.load_host_capabilities()
    assert len(s["sha256"]) == 64 and s["numa_nodes"] >= 1
    bad = tmp_path / "x.yaml"
    bad.write_text("manifest_version: 99\n", encoding="utf-8")
    try:
        hc.load_host_capabilities(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("foreign manifest accepted")


def test_registry_compiler_consumes_manifest_opt_in(tmp_path: Path):
    from src.registry.model_descriptors import compile_model_descriptors

    lean = tmp_path / "lean.yaml"
    lean.write_text(yaml.safe_dump({"roles": {}, "server_mode": {}}), encoding="utf-8")
    base = compile_model_descriptors(lean, None, active_roles=set(), allow_incomplete=True)
    assert "host_capabilities" not in base
    with_hc = compile_model_descriptors(
        lean, None, active_roles=set(), allow_incomplete=True,
        host_capabilities_path=hc.DEFAULT_MANIFEST_PATH,
    )
    assert with_hc["host_capabilities"]["sha256"] == hc.load_host_capabilities()["sha256"]
