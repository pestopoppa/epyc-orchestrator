"""MMID-UB-CEIL-1: a GPU launch whose effective ubatch exceeds 16384 tokens is refused.

HIP sets smpbo from sharedMemPerBlock (65536 B on the MI210) and mm_ids_helper
asserts n_tokens * 4 B <= smpbo, so mul_mat_id aborts above 16384 tokens per ubatch.
"""

from __future__ import annotations

import pytest

from scripts.server import orchestrator_stack as oss


def _cmd(device: str | None, ub: int | None, b: int | None = None) -> list[str]:
    cmd = ["llama-server", "-m", "model.gguf"]
    if ub is not None:
        cmd += ["-ub", str(ub)]
    if b is not None:
        cmd += ["-b", str(b)]
    if device is not None:
        cmd += ["--device", device]
    return cmd


def test_ceiling_constant_is_derived_value():
    assert oss.GPU_MMID_UBATCH_CEILING == 65536 // 4


def test_gpu_at_ceiling_passes():
    oss._refuse_gpu_ubatch_above_mmid_ceiling("r", _cmd("ROCm0", 16384, 16384))


def test_gpu_above_ceiling_refused():
    with pytest.raises(ValueError, match="MMID-UB-CEIL-1"):
        oss._refuse_gpu_ubatch_above_mmid_ceiling("r", _cmd("ROCm0", 16385, 32768))


def test_long_form_flags_refused():
    cmd = ["llama-server", "--ubatch-size", "20000", "--batch-size", "20000", "-dev", "ROCm0"]
    with pytest.raises(ValueError):
        oss._refuse_gpu_ubatch_above_mmid_ceiling("r", cmd)


def test_ubatch_clamped_by_default_batch_passes():
    # No -b emitted: llama.cpp clamps n_ubatch to the 2048 default (K4), so it is inert.
    oss._refuse_gpu_ubatch_above_mmid_ceiling("r", _cmd("ROCm0", 32768))


def test_ubatch_clamped_by_small_batch_passes():
    oss._refuse_gpu_ubatch_above_mmid_ceiling("r", _cmd("ROCm0", 32768, 8192))


@pytest.mark.parametrize("device", [None, "none", "CPU"])
def test_cpu_launch_never_refused(device):
    oss._refuse_gpu_ubatch_above_mmid_ceiling("r", _cmd(device, 65536, 65536))


def test_no_ubatch_flag_passes():
    oss._refuse_gpu_ubatch_above_mmid_ceiling("r", _cmd("ROCm0", None, 65536))


def test_build_server_command_calls_guard(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr(oss, "_refuse_gpu_ubatch_above_mmid_ceiling",
                        lambda role, cmd: seen.append(list(cmd)))
    monkeypatch.setattr(oss, "_build_dev_command", lambda port: ["llama-server", "-ub", "512"])
    monkeypatch.setattr(oss, "_verify_artifact_integrity", lambda *a, **k: None)
    oss.build_server_command({}, 8080, dev_mode=True)
    assert seen and "--device" in seen[0]
