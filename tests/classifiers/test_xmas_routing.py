from __future__ import annotations

import pytest
import yaml

from src.classifiers.xmas_routing import (
    XMAS_DOMAINS,
    XMAS_FUNCTIONS,
    WinnerTable,
    XmasRoutingConfig,
    XmasCell,
    build_xmas_routing_metadata,
    classify_xmas_cell,
    get_xmas_routing_config,
    load_winner_table,
)
from src.classifiers.config_loader import reset_classifier_config


def _complete_evidence_backed_table_payload() -> dict:
    cells = {
        domain: {function: "frontdoor" for function in XMAS_FUNCTIONS}
        for domain in XMAS_DOMAINS
    }
    evidence = {
        domain: {
            function: {
                "cell": f"{domain}:{function}",
                "winner": "frontdoor",
                "sample_count": 3,
                "source_summary_path": f"summary.table.{domain}.frontdoor",
                "candidates": {
                    "frontdoor": {
                        "correct": 3,
                        "total": 3,
                        "accuracy": 1.0,
                        "wall_mean_s": 1.0,
                    }
                },
            }
            for function in XMAS_FUNCTIONS
        }
        for domain in XMAS_DOMAINS
    }
    return {
        "version": "xmas-test",
        "fallback_role": "frontdoor",
        "provenance": {
            "source_results": [
                "data/research/2026-05-20-xmas-v3-25tasks-nothink/results.json"
            ],
            "generator": "scripts/research/xmas_winner_table.py",
        },
        "cells": cells,
        "evidence": evidence,
    }


def test_classify_math_verify_cell() -> None:
    result = classify_xmas_cell("Verify this proof and check the integral equation.")

    assert result.cell == XmasCell(domain="math", function="verify")
    assert result.is_confident()
    assert "integral" in result.matched_terms["domain"]
    assert "verify" in result.matched_terms["function"]


def test_classify_code_refine_cell() -> None:
    result = classify_xmas_cell("Refactor this Python function and fix the bug.")

    assert result.domain == "code"
    assert result.function == "refine"
    assert result.confidence >= 0.55


def test_classify_long_context_extract_with_context_bonus() -> None:
    context = "document " * 9000
    result = classify_xmas_cell("Extract the key points from this report.", context)

    assert result.domain == "long_context"
    assert result.function == "extract"
    assert result.domain_confidence > 0.0


def test_low_signal_prompt_falls_back_to_default_cell_with_zero_confidence() -> None:
    result = classify_xmas_cell("hello there")

    assert result.cell == XmasCell(domain="knowledge", function="solve")
    assert result.confidence == 0.0
    assert not result.is_confident()


def test_winner_table_loads_nested_yaml(tmp_path) -> None:
    table_path = tmp_path / "xmas.yaml"
    table_path.write_text(
        yaml.safe_dump(
            {
                "version": "xmas-v1",
                "fallback_role": "frontdoor",
                "cells": {
                    "math": {
                        "solve": "architect_general",
                        "verify": "worker_math",
                    },
                    "code": {
                        "refine": "coder_escalation",
                    },
                },
            }
        )
    )

    table = load_winner_table(table_path)

    assert table.version == "xmas-v1"
    assert table.winner_for("math", "solve") == "architect_general"
    assert table.winner_for("math", "verify") == "worker_math"
    assert table.winner_for("code", "refine") == "coder_escalation"
    assert table.winner_for("reasoning", "plan") == "frontdoor"


def test_winner_table_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="unknown X-MAS domain"):
        WinnerTable.from_mapping(
            {
                "cells": {
                    "music": {
                        "solve": "frontdoor",
                    },
                },
            }
        )


def test_winner_table_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="unknown orchestrator role"):
        WinnerTable.from_mapping(
            {
                "cells": {
                    "math": {
                        "solve": "not_a_role",
                    },
                },
            }
        )


def test_winner_table_can_require_complete_5x5_table() -> None:
    with pytest.raises(ValueError, match="missing 24 cells"):
        WinnerTable.from_mapping(
            {
                "cells": {
                    "math": {
                        "solve": "architect_general",
                    },
                },
            },
            require_complete=True,
        )


def test_winner_table_can_require_evidence_backed_5x5_table() -> None:
    table = WinnerTable.from_mapping(
        _complete_evidence_backed_table_payload(),
        require_evidence=True,
    )

    assert table.winner_for("knowledge", "solve") == "frontdoor"
    assert table.provenance["generator"] == "scripts/research/xmas_winner_table.py"


def test_winner_table_requires_evidence_provenance() -> None:
    payload = _complete_evidence_backed_table_payload()
    payload["provenance"] = {}

    with pytest.raises(ValueError, match="provenance.source_results"):
        WinnerTable.from_mapping(payload, require_evidence=True)


def test_winner_table_requires_cell_evidence_matching_winner() -> None:
    payload = _complete_evidence_backed_table_payload()
    payload["evidence"]["math"]["solve"]["winner"] = "architect_general"

    with pytest.raises(ValueError, match="missing evidence for 1 cells: math:solve"):
        WinnerTable.from_mapping(payload, require_evidence=True)


def test_xmas_config_defaults_off(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "classifier_config.yaml"
    cfg_path.write_text("xmas_routing:\n  mode: off\n", encoding="utf-8")
    monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_CONFIG", str(cfg_path))
    monkeypatch.delenv("ORCHESTRATOR_XMAS_ROUTING_MODE", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_XMAS_WINNER_TABLE_PATH", raising=False)
    reset_classifier_config()

    try:
        cfg = get_xmas_routing_config()
    finally:
        reset_classifier_config()

    assert cfg.mode == "off"
    assert not cfg.enabled
    assert build_xmas_routing_metadata("Refactor this function", config=cfg) is None


def test_xmas_config_env_override(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "classifier_config.yaml"
    table_path = tmp_path / "winner.yaml"
    cfg_path.write_text(
        "xmas_routing:\n  mode: off\n  confidence_threshold: 0.2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_CONFIG", str(cfg_path))
    monkeypatch.setenv("ORCHESTRATOR_XMAS_ROUTING_MODE", "shadow")
    monkeypatch.setenv("ORCHESTRATOR_XMAS_WINNER_TABLE_PATH", str(table_path))
    reset_classifier_config()

    try:
        cfg = get_xmas_routing_config()
    finally:
        reset_classifier_config()

    assert cfg.mode == "shadow"
    assert cfg.enabled
    assert cfg.confidence_threshold == 0.2
    assert cfg.winner_table_path == table_path


def test_build_xmas_metadata_with_winner_table(tmp_path) -> None:
    table_path = tmp_path / "xmas.yaml"
    table_path.write_text(
        yaml.safe_dump(
            {
                "version": "xmas-test",
                "fallback_role": "frontdoor",
                "cells": {
                    "code": {"refine": "coder_escalation"},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = XmasRoutingConfig(
        mode="shadow",
        confidence_threshold=0.55,
        winner_table_path=table_path,
    )

    meta = build_xmas_routing_metadata(
        "Refactor this Python function and fix the bug.",
        config=cfg,
    )

    assert meta is not None
    assert meta["mode"] == "shadow"
    assert meta["cell"] == "code:refine"
    assert meta["is_confident"] is True
    assert meta["suggested_role"] == "coder_escalation"
    assert meta["winner_table_version"] == "xmas-test"
    assert meta["winner_table_status"] == "loaded"
    assert meta["applied"] is False


def test_enforce_metadata_requires_complete_winner_table(tmp_path) -> None:
    table_path = tmp_path / "xmas.yaml"
    table_path.write_text(
        yaml.safe_dump(
            {
                "version": "xmas-test",
                "fallback_role": "frontdoor",
                "cells": {
                    "code": {"refine": "coder_escalation"},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = XmasRoutingConfig(
        mode="enforce",
        confidence_threshold=0.55,
        winner_table_path=table_path,
    )

    meta = build_xmas_routing_metadata(
        "Refactor this Python function and fix the bug.",
        config=cfg,
    )

    assert meta is not None
    assert meta["mode"] == "enforce"
    assert meta["suggested_role"] is None
    assert meta["winner_table_status"] == "invalid"
    assert "missing 24 cells" in meta["winner_table_error"]
    assert meta["applied"] is False


def test_enforce_metadata_requires_evidence_backed_winner_table(tmp_path) -> None:
    table_path = tmp_path / "xmas.yaml"
    payload = _complete_evidence_backed_table_payload()
    payload.pop("evidence")
    table_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    cfg = XmasRoutingConfig(
        mode="enforce",
        confidence_threshold=0.55,
        winner_table_path=table_path,
    )

    meta = build_xmas_routing_metadata(
        "Refactor this Python function and fix the bug.",
        config=cfg,
    )

    assert meta is not None
    assert meta["mode"] == "enforce"
    assert meta["suggested_role"] is None
    assert meta["winner_table_status"] == "invalid"
    assert "missing evidence for 25 cells" in meta["winner_table_error"]


def test_enforce_metadata_loads_complete_evidence_backed_table(tmp_path) -> None:
    table_path = tmp_path / "xmas.yaml"
    table_path.write_text(
        yaml.safe_dump(_complete_evidence_backed_table_payload()),
        encoding="utf-8",
    )
    cfg = XmasRoutingConfig(
        mode="enforce",
        confidence_threshold=0.55,
        winner_table_path=table_path,
    )

    meta = build_xmas_routing_metadata(
        "Refactor this Python function and fix the bug.",
        config=cfg,
    )

    assert meta is not None
    assert meta["mode"] == "enforce"
    assert meta["suggested_role"] == "frontdoor"
    assert meta["winner_table_status"] == "loaded"


def test_build_xmas_metadata_survives_missing_table(tmp_path) -> None:
    cfg = XmasRoutingConfig(
        mode="shadow",
        winner_table_path=tmp_path / "missing.yaml",
    )

    meta = build_xmas_routing_metadata("Verify this proof.", config=cfg)

    assert meta is not None
    assert meta["cell"] == "math:verify"
    assert meta["suggested_role"] is None
    assert meta["winner_table_status"] == "missing"
    assert meta["applied"] is False
