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
        "schema": "ev15-comparison-manifest.v1", "model_id": "model@pinned",
        "scorer_id": "scorer@pinned",
        "items": [{
            "item_id": "task-1", "context_sha256": "0" * 64,
            "candidates": [
                {"candidate_id": c, "answer_sha256": h * 64}
                for c, h in (("A", "a"), ("B", "b"), ("C", "c"))
            ],
        }],
    }


def point(index):
    return [float(i == index) for i in range(9)]


def captures(plan):
    distributions = {"A": point(8), "B": point(4), "C": point(0)}
    return [{
        **order, "capture_id": f"capture-{i}",
        "probabilities_left": distributions[order["left"]["candidate_id"]],
        "probabilities_right": distributions[order["right"]["candidate_id"]],
    } for i, order in enumerate(plan["orders"])]


def test_plan_is_deterministic_complete_and_binds_payloads():
    m = manifest()
    plan = make_plan(m)
    assert plan == make_plan(deepcopy(m))
    assert [(r["left"]["candidate_id"], r["right"]["candidate_id"])
            for r in plan["orders"]] == [
        ("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C"), ("C", "B"),
    ]
    assert len({r["request_id"] for r in plan["orders"]}) == 6
    old = deepcopy(plan)
    m["items"][0]["candidates"][0]["answer_sha256"] = "f" * 64
    assert plan == old  # caller mutation cannot change the declaration
    changed = make_plan(m)
    assert changed["orders"][0]["request_id"] != plan["orders"][0]["request_id"]
    assert changed["orders"][-1] == plan["orders"][-1]


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
    assert result["first_distribution"] == pytest.approx([.305, 0, 0, 0, 0, 0, 0, 0, .695])
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


@pytest.mark.parametrize("key", ["model_id", "scorer_id", "item_id", "context_sha256", "left", "right"])
def test_mismatched_capture_identity_refuses(key):
    plan = make_plan(manifest())
    rows = captures(plan)
    rows[0][key] = deepcopy(rows[1]["left"]) if key in ("left", "right") else "wrong"
    if key == "right":
        rows[0][key] = deepcopy(rows[0]["left"])
    with pytest.raises(ValueError, match="identity"):
        paired_readout(plan, rows, method="mean")


@pytest.mark.parametrize("bad", [
    [0.0] * 9, [1.0] * 8, [True] + [0] * 8, ["1"] + [0] * 8,
    [float("nan")] + [0] * 8, [float("inf")] + [0] * 8,
    [-.1] + [0] * 7 + [1.1],
])
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
    rows[0]["probabilities_left"] = [-.1] + [0] * 7 + [1.1]
    rows[1]["probabilities_right"] = [.1] + [0] * 7 + [.9]
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

    source, plan_path, rows_path, output = [tmp_path / name for name in (
        "manifest.json", "plan.json", "captures.json", "result.json",
    )]
    source.write_text(json.dumps(manifest()))
    assert run("plan", source, "--output", plan_path).returncode == 0
    plan = json.loads(plan_path.read_text())
    rows = captures(plan)
    rows_path.write_text(json.dumps(rows))
    assert run("readout", plan_path, rows_path, "--method", "mean", "--output", output).returncode == 0
    result = json.loads(output.read_text())
    assert result == paired_readout(plan, rows, method="mean")
    before = hashlib.sha256(output.read_bytes()).hexdigest()
    assert run("readout", plan_path, rows_path, "--method", "mean", "--output", output).returncode == 2
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
        [sys.executable, "-B", "-c", launcher, "readout", str(plan_path), str(rows_path),
         "--method", "mean", "--output", str(output)],
        cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["comparisons"]
