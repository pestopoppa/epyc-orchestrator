"""HS-TD-4: prepared-action mutation and receipt conformance fixtures."""

from __future__ import annotations

import json

import pytest

from src.typed_decisions.prepared_action import (
    SCHEMA_VERSION,
    append_receipt,
    make_receipt,
    prepare_action,
    validate_prepared_action,
)

CATALOG = {"inspect": {"tool": "read"}, "apply": {"tool": "write"}}
READ_SET = {"task": "revision-1", "permission_scope": "repo-a"}


def _prepared():
    return prepare_action(
        task_revision="revision-1",
        catalog=CATALOG,
        read_set=READ_SET,
        expires_at_ns=100,
        requested_model="local",
        resolved_model="local-v1",
    )


def _validate(prepared, **overrides):
    args = {
        "selected_id": "inspect",
        "current_task_revision": "revision-1",
        "current_catalog": CATALOG,
        "current_read_set": READ_SET,
        "authorized": True,
        "model_available": True,
        "now_ns": 99,
    }
    args.update(overrides)
    return validate_prepared_action(prepared, **args)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"selected_id": None}, "abstained"),
        ({"selected_id": "invented"}, "unknown_id"),
        ({"now_ns": 100}, "expired"),
        ({"current_task_revision": "revision-2"}, "stale"),
        ({"current_offered_choices": ["inspect"]}, "stale"),
        (
            {"current_catalog": {"inspect": {"tool": "changed"}, "apply": {"tool": "write"}}},
            "stale",
        ),
        ({"current_read_set": {"task": "revision-1", "permission_scope": "repo-b"}}, "stale"),
        ({"authorized": False}, "unauthorized"),
        ({"model_available": False}, "model_unavailable"),
    ],
)
def test_rejected_selections_use_incumbent_fallback(mutation, expected):
    result = _validate(_prepared(), **mutation)
    assert result.status == expected
    assert result.fallback == "incumbent_fallback"


def test_accepted_selection_and_receipt_are_complete(tmp_path):
    prepared = _prepared()
    result = _validate(prepared)
    assert result.status == "accepted"
    assert result.fallback is None
    receipt = make_receipt(
        prepared,
        result,
        authorization_result="allowed",
        component_timing_ms={"selection": 1.0, "validation": 2.0, "dispatch": 3.0},
        total_timing_ms=6.0,
        observed_downstream_outcome={"status": "returned", "detail": "dict"},
    )
    path = tmp_path / "decision_receipts.jsonl"
    append_receipt(path, receipt)
    row = json.loads(path.read_text().strip())
    assert row == {
        "schema_version": SCHEMA_VERSION,
        "task_revision": "revision-1",
        "offered_choices": ["apply", "inspect"],
        "catalog_fingerprint": prepared.catalog_fingerprint,
        "read_set_fingerprint": prepared.read_set_fingerprint,
        "expires_at_ns": 100,
        "selected_id": "inspect",
        "requested_model": "local",
        "resolved_model": "local-v1",
        "validation_result": "accepted",
        "authorization_result": "allowed",
        "fallback": None,
        "component_timing_ms": {"selection": 1.0, "validation": 2.0, "dispatch": 3.0},
        "total_timing_ms": 6.0,
        "observed_downstream_outcome": {"status": "returned", "detail": "dict"},
    }


def test_only_host_eligible_choices_are_offered():
    prepared = prepare_action(
        task_revision="revision-1",
        catalog=CATALOG,
        read_set=READ_SET,
        expires_at_ns=100,
        offered_choices=["inspect"],
    )
    assert prepared.offered_choices == ("inspect",)
    assert _validate(prepared, selected_id="apply").status == "unknown_id"


# UFH-TE-1 / UFH-TE-2
SERVING = {
    "descriptor_digest": "d1",
    "handler_artifact_digest": "h1",
    "config_digest": "c1",
    "serving_tree_commit": "abc",
    "process_start_ns": 1,
}


def _prepared_serving():
    return prepare_action(
        task_revision="revision-1", catalog=CATALOG, read_set=READ_SET,
        expires_at_ns=100, serving_identity=SERVING,
    )


def test_benign_restart_is_not_stale_but_handler_swap_is():
    prepared = _prepared_serving()
    restart = dict(SERVING, process_start_ns=999)
    assert _validate(prepared, current_serving_identity=restart).status == "accepted"
    assert _validate(prepared, current_serving_identity=SERVING).status == "accepted"
    for key in ("descriptor_digest", "handler_artifact_digest", "config_digest", "serving_tree_commit"):
        swapped = dict(SERVING, **{key: "changed"})
        result = _validate(prepared, current_serving_identity=swapped)
        assert result.status == "stale", key  # negative control per key
        assert result.fallback == "incumbent_fallback"
    assert _validate(prepared).status == "stale"  # identity missing at dispatch


def test_no_serving_identity_keeps_legacy_behaviour():
    assert _validate(_prepared(), current_serving_identity={"descriptor_digest": "x"}).status == "accepted"


def test_validator_fields_optional_and_emitted_when_set(tmp_path):
    prepared = _prepared()
    result = _validate(prepared)
    kw = dict(
        authorization_result="allowed", component_timing_ms={}, total_timing_ms=0.0,
        observed_downstream_outcome={},
    )
    path = tmp_path / "r.jsonl"
    append_receipt(path, make_receipt(prepared, result, **kw))
    legacy = json.loads(path.read_text().strip())
    assert "validator_id" not in legacy and "conclusion_valid_until_ns" not in legacy
    path2 = tmp_path / "r2.jsonl"
    append_receipt(
        path2,
        make_receipt(prepared, result, validator_id="v", validator_version="1",
                     conclusion_valid_until_ns=5, **kw),
    )
    row = json.loads(path2.read_text().strip())
    assert (row["validator_id"], row["validator_version"], row["conclusion_valid_until_ns"]) == ("v", "1", 5)
    assert row["schema_version"] == SCHEMA_VERSION


_JOINT_KEYS = (
    "joint_status", "joint_reason", "proposed_tool", "executed_tool",
    "operation_differs", "ignored_unused_failures", "joint_arguments",
)


def test_joint_fields_absent_for_old_rows_and_emitted_when_set(tmp_path):
    prepared = _prepared()
    result = _validate(prepared)
    kw = dict(
        authorization_result="allowed", component_timing_ms={}, total_timing_ms=0.0,
        observed_downstream_outcome={},
    )
    old = tmp_path / "old.jsonl"
    append_receipt(old, make_receipt(prepared, result, **kw))
    row = json.loads(old.read_text().strip())
    assert not any(k in row for k in _JOINT_KEYS)
    # Byte identity: the line equals the pre-seam serialisation (new keys stripped from asdict).
    from dataclasses import asdict

    legacy = asdict(make_receipt(prepared, result, **kw))
    for k in _JOINT_KEYS:
        legacy.pop(k)
    for k in ("proposed_arguments", "executed_arguments", "selector_status", "selector_reason",
              "selector_calls", "shadow_mode", "validator_id", "validator_version",
              "conclusion_valid_until_ns"):
        legacy.pop(k)
    assert old.read_text() == json.dumps(legacy, sort_keys=True, allow_nan=False) + "\n"

    new = tmp_path / "new.jsonl"
    append_receipt(
        new,
        make_receipt(
            prepared, result, selector_status="proposed", joint_status="proposed",
            joint_reason=None, proposed_tool="read", executed_tool="write",
            operation_differs=True, ignored_unused_failures=("a",), **kw,
        ),
    )
    row = json.loads(new.read_text().strip())
    assert row["joint_status"] == "proposed" and row["proposed_tool"] == "read"
    assert row["executed_tool"] == "write" and row["operation_differs"] is True
    assert row["ignored_unused_failures"] == ["a"] and row["joint_reason"] is None
