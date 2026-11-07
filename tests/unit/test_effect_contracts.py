"""UFH-TE-4: every registry tool has an authored effect contract; unknown blocks reuse."""

from pathlib import Path

import yaml

from src.typed_decisions.effect_contracts import EFFECT_CLASSES, load_contracts, reuse_allowed

REG = Path(__file__).resolve().parents[2] / "orchestration" / "tool_registry.yaml"


def test_every_registry_tool_has_a_complete_contract():
    tools = set(yaml.safe_load(REG.read_text())["tools"])
    contracts = load_contracts()
    assert tools == set(contracts)
    for name, c in contracts.items():
        assert c["effect_class"] in EFFECT_CLASSES, name
        assert c["destination"] and c["principal"] and c["basis"], name


def test_unknown_and_missing_block_reuse_negative_control():
    contracts = load_contracts()
    unknown = [n for n, c in contracts.items() if c["effect_class"] == "unknown"]
    assert unknown and not any(reuse_allowed(n, contracts) for n in unknown)
    assert not reuse_allowed("tool_not_in_table", contracts)
    assert reuse_allowed("read_file", contracts)
    # a hand-edited unknown row claiming reuse must still be refused
    assert not reuse_allowed("x", {"x": {"effect_class": "unknown", "reuse_allowed": True}})


def test_declared_side_effects_are_not_downgraded():
    reg = yaml.safe_load(REG.read_text())["tools"]
    contracts = load_contracts()
    for name, tool in reg.items():
        if tool.get("side_effects") == ["system_state"]:
            assert contracts[name]["effect_class"] != "read-only", name
