"""OAB-6 prompt diet: identity when nothing repeats, reduction when it does."""
from __future__ import annotations

import json

from src.prompt_builders import json_diet as jd


def _launch(tag: str) -> dict:
    return {"dsos": ["libggml-hip.so", "libggml-base.so"] * 20, "capability": tag * 80,
            "template": {"flags": [f"--flag-{i}" for i in range(30)]}}


def test_nothing_repeats_is_the_same_object():
    block = {"rows": [{"mechanism": f"m{i}", "note": "x" * 100} for i in range(80)]}
    assert len(json.dumps(block)) > jd.DEFAULT_MIN_BLOCK_BYTES
    assert jd.slim_json_block(block) is block


def test_small_block_untouched_even_with_ids_and_repeats():
    block = {"attempt_id": "a", "x": {"y": 1}, "z": {"y": 1}}
    assert jd.slim_json_block(block) is block


def test_repeated_subtree_points_at_first_copy_and_saves_size():
    launch = _launch("c")
    block = {"recipe": launch, "common_cpu_scope": {"full_transfer_target": launch},
             "pad": "p" * 5000}
    slim = jd.slim_json_block(block)
    assert slim is not block
    assert slim["common_cpu_scope"]["full_transfer_target"] == "<same as $.recipe>"
    assert slim["recipe"] == launch
    assert jd.reduction(block, slim)["saved_fraction"] > 0.2


def test_ids_dropped_and_prose_clipped_with_a_marker():
    rows = [{"attempt_id": f"id{i}", "result_sha256": "f" * 64, "status": "kept",
             "refusal": "r" * 900} for i in range(10)]
    slim = jd.slim_json_block({"rows": rows}, min_bytes=100)
    row = slim["rows"][0]
    assert "attempt_id" not in row and "result_sha256" not in row
    assert row["status"] == "kept"
    assert row["refusal"].endswith("…[clipped 400 chars]") and len(row["refusal"]) < 560


def test_non_container_values_pass_through_and_blocks_map():
    assert jd.slim_json_block("text") == "text"
    out = jd.slim_context_blocks({"a": [1, 2], "b": "s"})
    assert out == {"a": [1, 2], "b": "s"}
