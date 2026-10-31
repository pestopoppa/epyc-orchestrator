"""EV-15e uses both actual orders before the frozen nonlinear readout."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.typed_decisions import distributional_readout as original
from src.typed_decisions.paired_distributional_readout import METHODS, make_plan, paired_readout


def manifest():
    return {
        "schema": "ev15-comparison-manifest.v1",
        "model_id": "model@pinned",
        "scorer_id": "scorer@pinned",
        "items": [
            {
                "item_id": "task-1",
                "context_sha256": "0" * 64,
                "candidates": [
                    {"candidate_id": c, "answer_sha256": h * 64}
                    for c, h in (("A", "a"), ("B", "b"), ("C", "c"))
                ],
            }
        ],
    }


def score_head_manifest():
    result = manifest()
    result.update(
        schema="ev15-comparison-manifest.v2",
        score_token_ids=list(range(100, 109)),
        minimum_retained_mass=0.9,
    )
    return result


def point(index):
    return [float(i == index) for i in range(9)]


def captures(plan):
    distributions = {"A": point(8), "B": point(4), "C": point(0)}
    return [
        {
            **order,
            "capture_id": f"capture-{i}",
            "probabilities_left": distributions[order["left"]["candidate_id"]],
            "probabilities_right": distributions[order["right"]["candidate_id"]],
        }
        for i, order in enumerate(plan["orders"])
    ]


def score_head(values):
    token_ids = range(100, 109)
    return {
        "entries": [
            {"id": token_id, "prob": value}
            for token_id, value in zip(token_ids, values, strict=True)
        ]
    }


def score_head_captures(plan, *, override=None):
    distributions = {"A": point(8), "B": point(4), "C": point(0)}
    rows = []
    for i, order in enumerate(plan["orders"]):
        left_id = order["left"]["candidate_id"]
        right_id = order["right"]["candidate_id"]
        left = score_head(distributions[left_id])
        right = score_head(distributions[right_id])
        if override:
            override(i, left, right)
        rows.append(
            {
                **order,
                "capture_id": f"head-capture-{i}",
                "score_head_left": left,
                "score_head_right": right,
            }
        )
    return rows


def test_plan_is_deterministic_complete_and_binds_payloads():
    m = manifest()
    plan = make_plan(m)
    assert plan == make_plan(deepcopy(m))
    assert plan["schema"] == "ev15-paired-order-plan.v1"
    assert set(plan) == {"schema", "manifest", "score_support", "orders", "native_validation"}
    assert [(r["left"]["candidate_id"], r["right"]["candidate_id"]) for r in plan["orders"]] == [
        ("A", "B"),
        ("B", "A"),
        ("A", "C"),
        ("C", "A"),
        ("B", "C"),
        ("C", "B"),
    ]
    assert len({r["request_id"] for r in plan["orders"]}) == 6
    first = plan["orders"][0]
    original_v1_identity = {
        key: first[key]
        for key in ("model_id", "scorer_id", "item_id", "context_sha256", "left", "right")
    }
    original_v1_request_id = hashlib.sha256(
        json.dumps(
            original_v1_identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    assert first["request_id"] == original_v1_request_id
    old = deepcopy(plan)
    m["items"][0]["candidates"][0]["answer_sha256"] = "f" * 64
    assert plan == old  # caller mutation cannot change the declaration
    changed = make_plan(m)
    assert changed["orders"][0]["request_id"] != plan["orders"][0]["request_id"]
    assert changed["orders"][-1] == plan["orders"][-1]


def test_plan_binds_score_token_order_and_retained_mass_floor():
    m = score_head_manifest()
    original = make_plan(m)
    m["score_token_ids"][0], m["score_token_ids"][1] = (
        m["score_token_ids"][1],
        m["score_token_ids"][0],
    )
    reordered = make_plan(m)
    assert reordered["score_token_ids"] != original["score_token_ids"]
    assert reordered["orders"][0]["request_id"] != original["orders"][0]["request_id"]
    tampered = deepcopy(original)
    tampered["score_token_ids"] = reordered["score_token_ids"]
    with pytest.raises(ValueError, match="deterministic"):
        paired_readout(tampered, [], method="mean")

    m = score_head_manifest()
    m["minimum_retained_mass"] = 0.8
    lower_floor = make_plan(m)
    assert lower_floor["minimum_retained_mass"] != original["minimum_retained_mass"]
    assert lower_floor["orders"][0]["request_id"] != original["orders"][0]["request_id"]
    tampered = deepcopy(original)
    tampered["minimum_retained_mass"] = lower_floor["minimum_retained_mass"]
    with pytest.raises(ValueError, match="deterministic"):
        paired_readout(tampered, [], method="mean")


@pytest.mark.parametrize("method", METHODS)
def test_all_frozen_readouts_receive_identity_aligned_average(method):
    plan = make_plan(manifest())
    result = paired_readout(plan, list(reversed(captures(plan))), method=method)
    assert len(result["comparisons"]) == 3
    for row in result["comparisons"]:
        a = np.array(row["first_distribution"])
        b = np.array(row["second_distribution"])
        assert row["readout"] == float(getattr(original, method)(a, b))
    assert result["grade"] is None
    assert "not established" in result["native_validation"]


def test_nonlinear_readout_occurs_after_distribution_aggregation():
    m = manifest()
    m["items"][0]["candidates"] = m["items"][0]["candidates"][:2]
    plan = make_plan(m)
    rows = captures(plan)
    rows[0]["probabilities_left"] = [0.51, 0, 0, 0, 0, 0, 0, 0, 0.49]
    rows[1]["probabilities_right"] = [0.10, 0, 0, 0, 0, 0, 0, 0, 0.90]
    result = paired_readout(plan, rows, method="mode")["comparisons"][0]
    assert result["first_distribution"] == pytest.approx([0.305, 0, 0, 0, 0, 0, 0, 0, 0.695])
    assert result["readout"] == 1
    post = (
        original.mode(np.array(rows[0]["probabilities_left"]), np.array(point(4)))
        + original.mode(np.array(rows[1]["probabilities_right"]), np.array(point(4)))
    ) / 2
    assert post == 0  # averaging readout outputs would give the wrong answer


def test_swapping_declared_candidate_order_preserves_identity_and_reverses_readout():
    m = manifest()
    plan = make_plan(m)
    forward = paired_readout(plan, captures(plan), method="mean")["comparisons"][0]
    m["items"][0]["candidates"][:2] = list(reversed(m["items"][0]["candidates"][:2]))
    swapped = make_plan(m)
    backward = paired_readout(swapped, captures(swapped), method="mean")["comparisons"][0]
    assert backward["first"] == forward["second"]
    assert backward["readout"] == -forward["readout"]


@pytest.mark.parametrize("defect", ["missing", "whole_pair_missing", "duplicate", "reused_capture"])
def test_missing_or_replaced_orders_refuse(defect):
    plan = make_plan(manifest())
    rows = captures(plan)
    if defect == "missing":
        rows.pop()
    elif defect == "whole_pair_missing":
        rows = rows[:-2]
    elif defect == "duplicate":
        rows.append(deepcopy(rows[0]))
    else:
        rows[1]["capture_id"] = rows[0]["capture_id"]
    with pytest.raises(ValueError):
        paired_readout(plan, rows, method="mean")


@pytest.mark.parametrize(
    "key", ["model_id", "scorer_id", "item_id", "context_sha256", "left", "right"]
)
def test_mismatched_capture_identity_refuses(key):
    plan = make_plan(manifest())
    rows = captures(plan)
    rows[0][key] = deepcopy(rows[1]["left"]) if key in ("left", "right") else "wrong"
    if key == "right":
        rows[0][key] = deepcopy(rows[0]["left"])
    with pytest.raises(ValueError, match="identity"):
        paired_readout(plan, rows, method="mean")


@pytest.mark.parametrize(
    "bad",
    [
        [0.0] * 9,
        [1.0] * 8,
        [True] + [0] * 8,
        ["1"] + [0] * 8,
        [float("nan")] + [0] * 8,
        [float("inf")] + [0] * 8,
        [-0.1] + [0] * 7 + [1.1],
    ],
)
def test_bad_distributions_refuse_before_averaging(bad):
    plan = make_plan(manifest())
    rows = captures(plan)
    rows[0]["probabilities_left"] = bad
    with pytest.raises((ValueError, TypeError)):
        paired_readout(plan, rows, method="mean")


def test_tampered_plan_and_duplicate_item_refuse():
    m = manifest()
    plan = make_plan(m)
    plan["orders"].pop()
    with pytest.raises(ValueError, match="deterministic"):
        paired_readout(plan, [], method="mean")
    m["items"].append(deepcopy(m["items"][0]))
    with pytest.raises(ValueError, match="duplicate item"):
        make_plan(m)


def test_invalid_individual_distributions_cannot_cancel_when_averaged():
    plan = make_plan(manifest())
    rows = captures(plan)
    rows[0]["probabilities_left"] = [-0.1] + [0] * 7 + [1.1]
    rows[1]["probabilities_right"] = [0.1] + [0] * 7 + [0.9]
    with pytest.raises(ValueError):
        paired_readout(plan, rows, method="mean")


def test_multiple_items_cannot_fill_each_others_missing_orders():
    m = manifest()
    other = deepcopy(m["items"][0])
    other["item_id"] = "task-2"
    m["items"].append(other)
    plan = make_plan(m)
    rows = captures(plan)
    assert len(paired_readout(plan, rows, method="mean")["comparisons"]) == 6
    rows[-1] = deepcopy(rows[0])
    with pytest.raises(ValueError):
        paired_readout(plan, rows, method="mean")


def test_actual_cli_plan_readout_refusal_and_no_clobber(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-B", "-m", "src.typed_decisions.paired_distributional_readout"]

    def run(*args):
        return subprocess.run([*command, *map(str, args)], cwd=root, text=True, capture_output=True)

    source, plan_path, rows_path, output = [
        tmp_path / name
        for name in (
            "manifest.json",
            "plan.json",
            "captures.json",
            "result.json",
        )
    ]
    source.write_text(json.dumps(manifest()))
    assert run("plan", source, "--output", plan_path).returncode == 0
    plan = json.loads(plan_path.read_text())
    rows = captures(plan)
    rows_path.write_text(json.dumps(rows))
    assert (
        run("readout", plan_path, rows_path, "--method", "mean", "--output", output).returncode == 0
    )
    result = json.loads(output.read_text())
    assert result == paired_readout(plan, rows, method="mean")
    before = hashlib.sha256(output.read_bytes()).hexdigest()
    assert (
        run("readout", plan_path, rows_path, "--method", "mean", "--output", output).returncode == 2
    )
    assert hashlib.sha256(output.read_bytes()).hexdigest() == before
    rows_path.write_text(json.dumps(rows[:-1]))
    refused = tmp_path / "refused.json"
    failed = run("readout", plan_path, rows_path, "--method", "mean", "--output", refused)
    assert failed.returncode == 2 and "missing planned" in failed.stderr
    assert not refused.exists()
    source.write_text('{"schema": "bad", "schema": "ev15-comparison-manifest.v1"}')
    duplicate_output = tmp_path / "duplicate.json"
    failed = run("plan", source, "--output", duplicate_output)
    assert failed.returncode == 2 and "duplicate JSON key" in failed.stderr
    assert not duplicate_output.exists()


def test_cli_readout_never_connects_network_or_starts_native_process(tmp_path):
    plan = make_plan(manifest())
    plan_path, rows_path, output = [tmp_path / name for name in ("plan", "rows", "out")]
    plan_path.write_text(json.dumps(plan))
    rows_path.write_text(json.dumps(captures(plan)))
    # Install the tripwire before importing the actual CLI/package. Imports of
    # existing native adapters must not become a model/HTTP execution path.
    launcher = (
        "import sys,runpy\n"
        "def guard(event,args):\n"
        " if event in {'socket.connect','socket.getaddrinfo','subprocess.Popen','os.exec','os.posix_spawn'}:\n"
        "  raise RuntimeError('forbidden execution: '+event)\n"
        "sys.addaudithook(guard)\n"
        "runpy.run_module('src.typed_decisions.paired_distributional_readout',run_name='__main__')\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            launcher,
            "readout",
            str(plan_path),
            str(rows_path),
            "--method",
            "mean",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["comparisons"]


def test_raw_score_heads_are_checked_per_call_then_used_by_paired_consumer():
    plan = make_plan(score_head_manifest())
    heads = score_head_captures(plan)
    result = paired_readout(plan, heads, method="mean")

    assert result["status"] == "computed"
    assert result["capture_mode"] == "per_call_score_heads"
    assert result["native_validation"] == "not established by this offline consumer"
    assert result["grade"] is None
    assert len(result["score_head_captures"]) == 2 * len(plan["orders"])
    assert {row["retained_mass"] for row in result["score_head_captures"]} == {1.0}
    assert all(row["status"] == "accepted" for row in result["score_head_captures"])


def test_one_underfloor_call_refuses_pair_without_pooling_with_reverse_order():
    m = score_head_manifest()
    m["items"][0]["candidates"] = m["items"][0]["candidates"][:2]
    plan = make_plan(m)

    def lower_one_call(index, left, _right):
        if index == 0:
            left["entries"] = [{"id": row["id"], "prob": 0.08} for row in left["entries"]]

    rows = score_head_captures(plan, override=lower_one_call)
    result = paired_readout(plan, rows, method="mean")

    assert result["status"] == "refused"
    assert result["comparisons"] == []
    assert result["refusals"] == [
        {
            "request_id": rows[0]["request_id"],
            "candidate_id": "A",
            "side": "left",
            "reason": "below_retained_mass_floor",
        }
    ]
    assert len(result["score_head_captures"]) == 4
    assert result["score_head_captures"][0]["retained_mass"] == pytest.approx(0.72)
    assert result["score_head_captures"][0]["probabilities"] is None
    assert result["score_head_captures"][0]["status"] == "refused"


def test_legacy_math_vectors_are_marked_and_cannot_be_mixed_with_raw_heads():
    plan = make_plan(manifest())
    result = paired_readout(plan, captures(plan), method="mean")
    assert result["schema"] == "ev15-paired-distributional-readout.v1"
    assert set(result) == {
        "schema",
        "method",
        "plan_sha256",
        "captures_sha256",
        "model_id",
        "scorer_id",
        "comparisons",
        "native_validation",
        "grade",
    }
    assert result["native_validation"] == "not established by this offline consumer"

    rows = score_head_captures(plan)
    with pytest.raises(ValueError, match="require an ev15-comparison-manifest.v2"):
        paired_readout(plan, rows, method="mean")

    raw_plan = make_plan(score_head_manifest())
    with pytest.raises(ValueError, match="v2 plans require per-call score-head captures"):
        paired_readout(raw_plan, captures(raw_plan), method="mean")

    rows = score_head_captures(raw_plan)
    rows[0]["score_head_left"]["minimum_retained_mass"] = 0.0
    with pytest.raises(ValueError, match="exactly"):
        paired_readout(raw_plan, rows, method="mean")

    rows = score_head_captures(raw_plan)
    rows[0]["score_head_left"]["score_token_ids"] = list(range(101, 110))
    with pytest.raises(ValueError, match="exactly"):
        paired_readout(raw_plan, rows, method="mean")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score_token_ids", [1] * 9),
        ("score_token_ids", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("minimum_retained_mass", True),
        ("minimum_retained_mass", float("nan")),
        ("minimum_retained_mass", 1.1),
    ],
)
def test_plan_refuses_invalid_score_token_contract(field, value):
    m = score_head_manifest()
    m[field] = value
    with pytest.raises((TypeError, ValueError)):
        make_plan(m)


def test_cli_writes_typed_underfloor_refusal_and_exits_nonzero(tmp_path):
    m = score_head_manifest()
    m["items"][0]["candidates"] = m["items"][0]["candidates"][:2]
    plan = make_plan(m)
    plan_path = tmp_path / "plan.json"
    rows_path = tmp_path / "heads.json"
    output = tmp_path / "refused.json"
    plan_path.write_text(json.dumps(plan))
    rows = score_head_captures(plan)
    rows[0]["score_head_left"]["entries"] = [
        {"id": row["id"], "prob": 0.05} for row in rows[0]["score_head_left"]["entries"]
    ]
    rows_path.write_text(json.dumps(rows))

    command = [
        sys.executable,
        "-B",
        "-m",
        "src.typed_decisions.paired_distributional_readout",
        "readout",
        str(plan_path),
        str(rows_path),
        "--method",
        "mean",
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    refusal = json.loads(output.read_text())
    assert refusal["status"] == "refused"
    assert refusal["comparisons"] == []
    assert refusal["score_head_captures"][0]["retained_mass"] == pytest.approx(0.45)
    assert refusal["score_head_captures"][0]["probabilities"] is None


def test_cli_accepts_plan_bound_score_heads_and_emits_per_call_audits(tmp_path):
    m = score_head_manifest()
    m["items"][0]["candidates"] = m["items"][0]["candidates"][:2]
    plan = make_plan(m)
    plan_path = tmp_path / "plan.json"
    rows_path = tmp_path / "heads.json"
    output = tmp_path / "accepted.json"
    plan_path.write_text(json.dumps(plan))
    rows_path.write_text(json.dumps(score_head_captures(plan)))

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "src.typed_decisions.paired_distributional_readout",
            "readout",
            str(plan_path),
            str(rows_path),
            "--method",
            "mean",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    accepted = json.loads(output.read_text())
    assert accepted["status"] == "computed"
    assert accepted["capture_mode"] == "per_call_score_heads"
    assert len(accepted["comparisons"]) == 1
    assert len(accepted["score_head_captures"]) == 4
    assert {row["minimum_retained_mass"] for row in accepted["score_head_captures"]} == {0.9}
