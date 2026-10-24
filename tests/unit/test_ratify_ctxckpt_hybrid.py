"""RATIFY-CTXCKPT-HYBRID-20261024: `--ctx-checkpoints 32` declared on every hybrid server.

The declaration (`server_mode.<role>.ctx_checkpoints`) must travel registry -> compiled
priors -> launcher argv -> runtime attestation. 32 is the v10 llama-server default, so
behaviour is unchanged; the point is that a kernel-default change can no longer silently
change production (KPF-64.4: checkpoints off lose 3.9 pp prefix-hit share on a hybrid
role). Offline: synthetic priors, no server, no inference.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.server import orchestrator_stack as oss
from scripts.server import stack_commands
from src.registry.stack_priors import _launch_runtime_record

REPO = Path(__file__).resolve().parents[2]

# llm_arch_is_hybrid() in the frozen v10 tree (src/llama-arch.cpp:1040-1060), by GGUF
# `general.architecture` string. Recurrent-only archs (mamba/rwkv) are not served.
HYBRID_ARCHS = frozenset({
    "jamba", "falcon-h1", "plamo2", "granitehybrid", "lfm2", "lfm2moe", "nemotron_h",
    "nemotron_h_moe", "qwen3next", "kimi-linear", "qwen35", "qwen35moe", "qwen4exp",
    "deepseek4",
})


# -- compiler ------------------------------------------------------------------------


def _record(server_cfg: dict) -> dict:
    return _launch_runtime_record(
        role="frontdoor",
        descriptor={},
        server_cfg=server_cfg,
        role_cfg={},
        launch_cfg={
            "effective_context_tokens": 8192,
            "launch": {
                "primary_roles": ["frontdoor"],
                "modes": ["default"],
                "requirements": {},
                "runtime": {},
            },
        },
    )


@pytest.mark.parametrize("declared", [32, 0])
def test_declared_ctx_checkpoints_is_compiled_verbatim(declared: int) -> None:
    """0 is a DECLARED value (checkpoints off), not an absent one."""
    runtime = _record({"cache_ram": 32768, "ctx_checkpoints": declared})
    assert runtime["flags"]["ctx_checkpoints"] == declared
    assert runtime["flags"]["cache_ram"] == 32768


def test_undeclared_ctx_checkpoints_is_absent_not_none() -> None:
    """Absent keeps every non-hybrid role's compiled record byte-identical."""
    runtime = _record({"cache_ram": 32768})
    assert "ctx_checkpoints" not in runtime["flags"]


@pytest.mark.parametrize("bad", [True, -1, "32"])
def test_non_integer_or_negative_declaration_is_not_compiled(bad) -> None:
    runtime = _record({"ctx_checkpoints": bad})
    assert "ctx_checkpoints" not in runtime["flags"]


# -- launcher ------------------------------------------------------------------------


def _write_prior(tmp_path: Path, flags: dict) -> Path:
    path = tmp_path / "stack_priors.yaml"
    path.write_text(yaml.safe_dump({"roles": {"frontdoor": {
        "deployment_status": "live_stack",
        "serving": {"launch": {"requirements": {}, "runtime": {
            "binary_path": "/prior/llama-server",
            "cache": {"context_tokens": 262144, "slots": 4, "ubatch": 2048},
            "flags": {"flash_attn": True, "jinja": True, "spec": {"enabled": False}, **flags},
        }}},
    }}}, sort_keys=False), encoding="utf-8")
    return path


def _role() -> SimpleNamespace:
    return SimpleNamespace(
        name="frontdoor",
        model=SimpleNamespace(full_path="/m/q36.gguf"),
        acceleration=SimpleNamespace(type="none", experts=None, draft_role=None),
    )


def test_launcher_emits_the_declared_value_right_after_cache_ram(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", _write_prior(
        tmp_path, {"cache_ram": 32768, "ctx_checkpoints": 32}))
    cmd = oss._build_role_command(_role(), port=9070, prepare_runtime_dirs=False)
    assert cmd.count("--ctx-checkpoints") == 1
    i = cmd.index("--ctx-checkpoints")
    assert cmd[i + 1] == "32"
    # a pure insertion among the declared serving flags: immediately after --cache-ram N
    assert cmd[i - 2:i] == ["--cache-ram", "32768"]


def test_launcher_emits_a_declared_zero(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", _write_prior(tmp_path, {"ctx_checkpoints": 0}))
    cmd = oss._build_role_command(_role(), port=9070, prepare_runtime_dirs=False)
    assert cmd[cmd.index("--ctx-checkpoints") + 1] == "0"


def test_launcher_emits_nothing_when_undeclared(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", _write_prior(tmp_path, {"cache_ram": 32768}))
    cmd = oss._build_role_command(_role(), port=9070, prepare_runtime_dirs=False)
    assert "--ctx-checkpoints" not in cmd


def test_undeclared_argv_is_identical_to_the_pre_change_shape(tmp_path, monkeypatch) -> None:
    """Declaring the flag adds exactly two tokens and changes nothing else."""
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", _write_prior(tmp_path, {"cache_ram": 32768}))
    without = oss._build_role_command(_role(), port=9070, prepare_runtime_dirs=False)
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", _write_prior(
        tmp_path, {"cache_ram": 32768, "ctx_checkpoints": 32}))
    with_flag = oss._build_role_command(_role(), port=9070, prepare_runtime_dirs=False)
    i = with_flag.index("--ctx-checkpoints")
    assert with_flag[:i] + with_flag[i + 2:] == without


# -- runtime attestation -------------------------------------------------------------


def _info():
    return stack_commands.ProcessInfo(role="frontdoor", pid=321, port=8070,
                                      started_at="now", model_path="/models/m.gguf",
                                      log_file="a.log")


_BASE = ["/opt/llama/bin/llama-server", "-m", "/models/m.gguf", "--cache-ram", "32768"]


def _contract(flags: dict) -> dict:
    return {"requirements": {"model_path": "/models/m.gguf"},
            "runtime": {"binary_path": "/opt/llama/bin/llama-server", "cache": {},
                        "flags": {"cache_ram": 32768, **flags}},
            "ports": [8070]}


def test_attestation_clean_when_live_carries_the_flag() -> None:
    assert stack_commands._runtime_attestation_warnings(
        "frontdoor", _info(), _BASE + ["--ctx-checkpoints", "32"],
        _contract({"ctx_checkpoints": 32})) == []


def test_attestation_clean_for_a_pre_reload_server_on_the_default() -> None:
    """Landing the declaration does not turn every running hybrid server into drift."""
    assert stack_commands._runtime_attestation_warnings(
        "frontdoor", _info(), _BASE, _contract({"ctx_checkpoints": 32})) == []


def test_attestation_reports_checkpoints_off_on_a_role_declaring_32() -> None:
    assert stack_commands._runtime_attestation_warnings(
        "frontdoor", _info(), _BASE + ["-ctxcp", "0"], _contract({"ctx_checkpoints": 32})) == [
        "frontdoor pid 321 runtime ctx_checkpoints expected 32; live cmdline has 0"
    ]


def test_live_resolution_rules() -> None:
    assert stack_commands._live_ctx_checkpoints(["x"]) == stack_commands.SERVER_DEFAULT_CTX_CHECKPOINTS == 32
    assert stack_commands._live_ctx_checkpoints(["x", "--ctx-checkpoints", "8"]) == 8
    assert stack_commands._live_ctx_checkpoints(["x", "--swa-checkpoints", "4", "-ctxcp", "16"]) == 16
    assert stack_commands._live_ctx_checkpoints(["x", "--ctx-checkpoints", "many"]) is None


def test_the_new_field_is_mapped_for_attestation() -> None:
    assert "runtime.flags.ctx_checkpoints" in stack_commands._RUNTIME_FIELD_CHECKS


# -- this tree: master declaration -> compiled priors (recompute, not a literal) -------


def _lean_server_mode() -> dict:
    lean = yaml.safe_load((REPO / "orchestration/model_registry.yaml").read_text())
    return lean.get("server_mode") or {}


def _priors_roles() -> dict:
    return yaml.safe_load((REPO / "orchestration/derived/stack_priors.yaml").read_text())["roles"]


def test_this_trees_priors_carry_the_master_declaration_verbatim() -> None:
    server_mode = _lean_server_mode()
    """Each launch record carries its SERVING host's declaration (aliases inherit it)."""
    for role, rec in _priors_roles().items():
        serving = rec.get("serving") or {}
        flags = ((serving.get("launch") or {}).get("runtime") or {}).get("flags")
        host = serving.get("server_role")
        if not isinstance(flags, dict) or host not in server_mode:
            continue
        declared = (server_mode.get(host) or {}).get("ctx_checkpoints")
        assert flags.get("ctx_checkpoints") == declared, (role, host)


def test_this_trees_hybrid_host_servers_declare_32_and_nothing_else_does() -> None:
    """Every launched (non-alias) live server is declared iff its model is hybrid."""
    server_mode = _lean_server_mode()
    seen_hybrid = set()
    for role, rec in _priors_roles().items():
        if rec.get("deployment_status") != "live_stack":
            continue
        serving = rec.get("serving") or {}
        runtime = (serving.get("launch") or {}).get("runtime") or {}
        flags = runtime.get("flags")
        if not isinstance(flags, dict) or serving.get("server_role") != role:
            continue
        if (server_mode.get(role) or {}).get("alias_of"):
            continue   # served by its alias_of host; never launched on its own
        arch = (rec.get("model") or {}).get("arch")
        if arch in HYBRID_ARCHS:
            seen_hybrid.add(role)
            assert flags.get("ctx_checkpoints") == 32, (role, arch)
        else:
            assert "ctx_checkpoints" not in flags, (role, arch)
    if seen_hybrid:
        assert {"frontdoor", "architect_critic", "architect_general"} <= seen_hybrid
