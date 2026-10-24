"""B5-01: the exact hypervolume kernel is memoized, never changed.

The digests pin reconstruct_archive_from_journal_rows on the real 1,372-row replay fixture
for every objective policy, captured on the code BEFORE the memo landed. Segment snapshots
persist hv_history_by_tier and are verified by exact prefix replay, so any drift in a
single HV value would invalidate them: these digests must never be re-captured to make a
change pass.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.autopilot_core import pareto_math
from src.autopilot_core.journal_reconstruction import reconstruct_archive_from_journal_rows
from src.autopilot_core.tier_specs import DEFAULT_REFERENCE_POINT, TASK_RATE_REFERENCE_POINT

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "gate_frontier_replay.json"
GOLDEN = {
    ("legacy_4d_v1", None): "3d04f5798fa2ae91bb2ae96e6c3504f6c55fd1bbf9fc33aee55f5ecd6fc9f29e",
    ("task_rate_3d_v1", None): "c37ece6c7a983702f5511deeb4b3800aa9afd114a97239e66655799d2beb6db2",
    ("task_rate_4d_v7_physical_cohort_exclusion", None): (
        "ec2fa464be005a534a5b7b4cd8e8c6cd5d61143688a14272fd9285d6cd67fd9b"
    ),
    ("task_rate_4d_v1", None): "e3a5ef4201a6d5abbbefa4d14f555280ebe0de19c87f98466acf3424b501725e",
    ("task_rate_4d_v2_resource_lanes", None): (
        "9180611de7ba649e0600f7898f43541082c488d6e5a339244a01729814e5e193"
    ),
    # exclude_before_ts = the fixture's median row timestamp.
    ("task_rate_4d_v2_resource_lanes", 1781408825.709817): (
        "8a963ee6ba95c999dea19a6025294e07b8bf52860f7ab47c4a94487b339430fb"
    ),
}
# Fixture rows dropped for an unmeasured objective axis, per policy (B5-12).
UNMEASURED = {
    "legacy_4d_v1": 85,
    "task_rate_3d_v1": 85,
    "task_rate_4d_v7_physical_cohort_exclusion": 88,
    "task_rate_4d_v1": 88,
    "task_rate_4d_v2_resource_lanes": 88,
}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return json.loads(FIXTURE.read_text())["rows"]


@pytest.mark.parametrize("policy, exclude_before_ts", sorted(GOLDEN, key=str))
def test_replay_payload_matches_pre_memo_golden(rows, policy, exclude_before_ts):
    payload = reconstruct_archive_from_journal_rows(
        rows, None, objective_policy=policy, exclude_before_ts=exclude_before_ts
    )
    # B5-12 added this telemetry slot after the capture; everything else stays pinned.
    unmeasured = payload["exclusions"].pop("unmeasured_objectives")
    if exclude_before_ts is None:
        assert unmeasured["count"] == UNMEASURED[policy]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    assert digest == GOLDEN[(policy, exclude_before_ts)]


A4, B4, C4 = (1.0, 2.0, -0.5, 1.0), (2.0, 1.0, -0.5, 1.0), (1.5, 1.5, -0.75, 0.5)
A3, B3, C3 = (1.0, 2.0, 1.0), (2.0, 1.0, 1.0), (1.5, 1.5, 0.5)


@pytest.mark.parametrize(
    "points, ref, expected",
    [
        ([A4], DEFAULT_REFERENCE_POINT, 1.0),
        ([A4, B4], DEFAULT_REFERENCE_POINT, 1.5),
        ([A4, B4, C4], DEFAULT_REFERENCE_POINT, 1.53125),
        ([A3], TASK_RATE_REFERENCE_POINT, 2.0),
        ([A3, B3], TASK_RATE_REFERENCE_POINT, 3.0),
        ([A3, B3, C3], TASK_RATE_REFERENCE_POINT, 3.125),
    ],
)
def test_hypervolume_of_overlapping_boxes(points, ref, expected):
    assert pareto_math.hypervolume(points, ref) == pytest.approx(expected, abs=1e-12)


def test_replay_reuses_the_memo(rows):
    pareto_math._exact_hypervolume.cache_clear()
    reconstruct_archive_from_journal_rows(rows, None, objective_policy="legacy_4d_v1")
    info = pareto_math._exact_hypervolume.cache_info()
    assert info.hits > 0
    assert info.misses <= 200
