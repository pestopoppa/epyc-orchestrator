"""RTE-PFX-2 role recipe: pure checks + window guard + registry derivation (no server started)."""
import importlib.util
import sys
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "rte_pfx2_role_recipe.py"
_S = importlib.util.spec_from_file_location("rte_pfx2_recipe_test", _P)
M = importlib.util.module_from_spec(_S); sys.modules["rte_pfx2_recipe_test"] = M; _S.loader.exec_module(M)

ARGV = ["bin", "-m", "m", "--port", "1", "-np", "4", "--cache-ram", "32768", "--slot-save-path", "/prod"]


def test_retarget_only_changes_port_and_slot_path():
    out = M.retarget(ARGV, 18971, "/priv")
    assert out[out.index("--port") + 1] == "18971" and out[out.index("--slot-save-path") + 1] == "/priv"
    assert [a for a, b in zip(ARGV, out) if a != b] == ["1", "/prod"]


def test_requirements():
    assert M.check_requirements(ARGV, 4, 18971) == []
    assert M.check_requirements(ARGV[:5] + ["-np", "2"] + ARGV[7:], 4, 18971)
    assert any("cache-ram" in b for b in M.check_requirements([a for a in ARGV if a not in ("--cache-ram", "32768")], 4, 18971))
    assert M.check_requirements(ARGV, 4, 8070)


def test_prefix_sidecar_matches():
    assert M.prefix_sha_ok()[0]


def test_run_refused_without_window(monkeypatch):
    monkeypatch.delenv("WINDOW_CONFIRMED", raising=False)
    assert M.main(["run"]) == 3


def test_resolves_from_registry_np_ge_4():
    plan = M.resolve(18971, 4)
    assert int(M._flag(plan["argv"], "-np")) >= 4 and "--cache-ram" in plan["argv"]
    assert plan["argv"][plan["argv"].index("--port") + 1] == "18971"
