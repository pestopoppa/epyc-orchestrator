from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))

import eval_tower  # noqa: E402


def test_oversized_eval_prompt_retrieves_relevant_segments_and_preserves_question() -> None:
    filler = ("ordinary unrelated material " * 180) + "\n"
    relevant = "The complete graph fixtures use K5, K10, and K20 but never K15.\n"
    prompt = "Context:\n" + filler * 20 + relevant + filler * 20
    prompt += "\nQuestion: Which complete graph was not used?\nA) K5\nB) K10\nC) K15\nD) K20"

    compact, provenance = eval_tower._compact_oversized_eval_prompt(prompt)

    assert provenance["applied"] is True
    assert provenance["original_chars"] == len(prompt)
    assert provenance["output_chars"] < len(prompt)
    assert relevant.strip() in compact
    assert compact.endswith("Question: Which complete graph was not used?\nA) K5\nB) K10\nC) K15\nD) K20")


def test_short_eval_prompt_is_unchanged() -> None:
    prompt = "Context:\nsmall\nQuestion: value?"
    compact, provenance = eval_tower._compact_oversized_eval_prompt(prompt)
    assert compact == prompt
    assert provenance == {
        "applied": False,
        "original_chars": len(prompt),
        "output_chars": len(prompt),
    }


def test_inband_placement_timeout_has_typed_provenance() -> None:
    provenance = eval_tower._inband_error_provenance(
        "[ERROR: placement timeout role=frontdoor reason=placement_topology_overlap_timeout holders=[] after 300.0s]",
        role="frontdoor",
    )
    assert provenance["class"] == "admission_timeout"
    assert provenance["code"] == "placement_topology_overlap_timeout"
    assert provenance["role"] == "frontdoor"


def test_inband_http_400_has_typed_provenance() -> None:
    provenance = eval_tower._inband_error_provenance(
        "[ERROR: Inference failed: llama-server HTTP 400]",
        role="ingest_long_context",
    )
    assert provenance["class"] == "backend_request_rejected"
    assert provenance["code"] == "llama_http_400"
