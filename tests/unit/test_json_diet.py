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


def test_non_json_input_fails_open_unchanged():
    class Opaque:
        pass

    block = {"rows": [Opaque(), {"pad": "p" * 5000}]}
    assert jd.slim_json_block(block) is block            # TypeError -> unchanged
    loop: dict = {"pad": "p" * 5000}
    loop["self"] = loop
    assert jd.slim_json_block(loop) is loop              # circular ValueError -> unchanged
    bad_keys = {(1, 2): "p" * 5000}
    assert jd.slim_json_block(bad_keys) is bad_keys
    assert jd.slim_json_block(b"raw") == b"raw" and jd.slim_json_block(None) is None


def test_nested_repeats_point_at_first_copy_and_order_is_preserved():
    sub = {"z": "q" * 150, "a": ["r" * 150]}
    block = {"zeta": {"deep": {"copy": sub}}, "alpha": [sub, {"again": sub}], "pad": "p" * 5000}
    slim = jd.slim_json_block(block)
    assert list(slim) == ["zeta", "alpha", "pad"], "key order is preserved, never sorted"
    assert slim["zeta"]["deep"]["copy"] == sub                    # first copy kept intact
    assert slim["alpha"][0] == "<same as $.zeta.deep.copy>"
    assert slim["alpha"][1]["again"] == "<same as $.zeta.deep.copy>"
    assert list(slim["zeta"]["deep"]["copy"]) == ["z", "a"]


def test_dedupe_subtrees_does_not_sort_keys():
    out = jd.dedupe_subtrees({"b": 1, "a": {"y": "k" * 300, "x": 2}, "c": {"y": "k" * 300, "x": 2}})
    assert list(out) == ["b", "a", "c"] and list(out["a"]) == ["y", "x"]
    assert out["c"] == "<same as $.a>"
