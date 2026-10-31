import pytest

from scripts.autopilot.judge_exposure import summarize, summarize_results


def test_distinct_counts_and_selected_row_weights_are_separate():
    rows = [
        {"__pool_metadata__": True},
        {"suite": "a", "prompt": "judge", "scoring_method": "llm_judge"},
        {"suite": "a", "prompt": "judge", "scoring_method": "llm_judge"},
        {"suite": "a", "prompt": "exact"},
        {"suite": "b", "prompt": "rubric", "scoring_method": "rubric"},
    ]
    inventory = summarize(rows)
    assert inventory["distinct_judge_fraction"] == 2 / 3
    assert inventory["potential_quality_axis_share"] is None
    draw = summarize(rows, selected=True)
    assert draw["potential_quality_axis_share"] == 3 / 4
    assert draw["suites"]["a"]["potential_quality_axis_share"] == 1 / 2
    assert draw["suites"]["a"]["duplicate_rows"] == 1
    assert draw["realized_scored_quality_axis_share"] is None


def test_local_id_collisions_context_and_image_inputs():
    rows = [
        {"suite": "a", "id": "same", "prompt": "look", "image_path": image}
        for image in ("one.png", "two.png")
    ]
    rows.append({"suite": "a", "id": "same", "prompt": "look", "context": "different"})
    assert summarize(rows)["distinct_inputs"] == 3


def test_conflicting_methods_and_malformed_rows_fail_loudly():
    with pytest.raises(ValueError, match="conflicting"):
        summarize(
            [
                {"suite": "a", "prompt": "q"},
                {"suite": "a", "prompt": "q", "scoring_method": "rubric"},
            ]
        )
    with pytest.raises(ValueError, match="suite"):
        summarize([{"prompt": "q"}])
    assert summarize([])["distinct_judge_fraction"] is None


def batch():
    rows = [
        {"row_type": "batch_start", "requested_n": 3},
        {
            "row_type": "question_result",
            "ordinal": 0,
            "result": {"suite": "a", "qid": "1", "scoring_method": "rubric"},
        },
        {
            "row_type": "question_result",
            "ordinal": 1,
            "result": {"suite": "b", "qid": "2", "error": True, "disposition": "task_failed"},
        },
        {
            "row_type": "question_result",
            "ordinal": 2,
            "result": {
                "suite": "a",
                "qid": "3",
                "error": True,
                "scoring_method": "llm_judge",
                "disposition": "scoring_failed",
            },
        },
        {"row_type": "batch_complete", "complete": True, "completed_n": 3},
    ]
    return [{**row, "eval_batch_id": "one"} for row in rows]


def test_native_denominator_preserves_task_failed_and_excludes_scoring_errors():
    report = summarize_results(batch())
    assert report["quality_denominator"] == 2
    assert report["excluded_error_rows"] == 1
    assert report["realized_scored_quality_axis_share"] == 0.5
    assert report["suites"]["a"]["realized_scored_quality_axis_share"] == 0.5


def test_shadow_audit_partition_does_not_inflate_decision_exposure():
    rows = batch()
    rows[1]["result"]["partition"] = "audit"
    report = summarize_results(rows)
    assert report["quality_denominator"] == 1
    assert report["excluded_partition_rows"] == 1
    assert report["realized_scored_quality_axis_share"] == 0
    assert summarize_results(rows, partition="audit")["realized_scored_quality_axis_share"] == 1


@pytest.mark.parametrize("mutation", ["incomplete", "duplicate", "mixed", "short"])
def test_result_batches_refuse_ambiguous_or_incomplete_evidence(mutation):
    rows = batch()
    if mutation == "incomplete":
        rows.pop()
    elif mutation == "duplicate":
        rows[2]["ordinal"] = 0
    elif mutation == "mixed":
        rows[2]["eval_batch_id"] = "other"
    else:
        rows[0]["requested_n"] = 4
    with pytest.raises(ValueError):
        summarize_results(rows)
