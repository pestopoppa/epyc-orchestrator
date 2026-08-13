from __future__ import annotations

import copy
import json

import pytest

from scripts.autopilot import planner_roster as roster


def test_default_roster_is_exactly_two_codex_and_zero_claude() -> None:
    payload, digest, path = roster.load_roster(roster.DEFAULT_ROSTER)
    assert [cell["provider"] for cell in payload["cells"]] == ["codex", "codex_critic"]
    assert payload["constraints"]["claude_planner_count"] == 0
    assert len(digest) == 64
    assert path == roster.DEFAULT_ROSTER


def test_claude_cell_is_rejected_even_if_counts_claim_zero() -> None:
    payload = json.loads(roster.DEFAULT_ROSTER.read_text(encoding="utf-8"))
    payload["cells"][1]["provider"] = "claude"
    with pytest.raises(roster.PlannerRosterError, match="Claude is forbidden"):
        roster.validate_roster(payload)


def test_active_environment_is_bound_to_roster_sha_and_exact_roles(tmp_path) -> None:
    path = tmp_path / "roster.json"
    path.write_bytes(roster.DEFAULT_ROSTER.read_bytes())
    env = roster.apply_roster({"AUTOPILOT_PLANNER_PRIMARY": "claude"}, path)
    roster.validate_active_environment(env)
    assert env["AUTOPILOT_PLANNER_PRIMARY"] == "codex"
    assert env["AUTOPILOT_PLANNER_CRITIC"] == "codex_critic"
    assert env["AUTOPILOT_PLANNER_CRITIC_FALLBACK"] == "none"

    drifted = copy.deepcopy(env)
    drifted["AUTOPILOT_PLANNER_CRITIC"] = "claude"
    with pytest.raises(roster.PlannerRosterError, match="differs from sealed roster"):
        roster.validate_active_environment(drifted)
