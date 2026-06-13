from __future__ import annotations

import pytest
import yaml

from src.classifiers.xmas_routing import (
    WinnerTable,
    XmasCell,
    classify_xmas_cell,
    load_winner_table,
)


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
