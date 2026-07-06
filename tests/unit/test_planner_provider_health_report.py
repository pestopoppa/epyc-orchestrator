from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "autopilot" / "planner_provider_health_report.py"
spec = importlib.util.spec_from_file_location("planner_provider_health_report", SCRIPT)
assert spec is not None and spec.loader is not None
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def _write_tap(path: Path, blocks: list[str]) -> None:
    path.write_text("\n========================================================================\n".join(blocks))


def test_build_report_counts_local_draft_and_critique_success(tmp_path: Path) -> None:
    tap = tmp_path / "planner_tap.log"
    _write_tap(
        tap,
        [
            """[2026-07-06T06:03:11] PLANNER provider=local_frontdoor role=draft start
url: http://127.0.0.1:8000/v1/chat/completions
------------------------------------------------------------------------
[local:result:local_frontdoor:draft] ```json:autopilot_actions
{"type": "train_routing_models", "min_memories": 500}
```
```json:autopilot_rationale
{"hypothesis": "routing refresh is useful", "falsifier": "no gain"}
```
[END provider=local_frontdoor role=draft] result_chars=622""",
            """[2026-07-06T06:06:48] PLANNER provider=local_worker role=critique start
url: http://127.0.0.1:8000/v1/chat/completions
------------------------------------------------------------------------
[local:result:local_worker:critique] ```json:autopilot_critique
{"decision": "approve", "confidence": 1.0, "issues": []}
```
[END provider=local_worker role=critique] result_chars=137""",
        ],
    )

    report = reporter.build_report(tap_path=tap, stale_after_s=999999)

    assert report["ok"] is True
    assert report["status"] == "healthy"
    assert report["local"]["draft_successes"] == 1
    assert report["local"]["critique_successes"] == 1
    assert report["draft_actions"] == {"train_routing_models": 1}
    assert report["critic_decisions"] == {"approve": 1}
    assert report["providers"]["local_frontdoor"]["draft_successes"] == 1
    assert report["providers"]["local_worker"]["critique_successes"] == 1


def test_build_report_surfaces_local_failures_and_critic_rejections(tmp_path: Path) -> None:
    tap = tmp_path / "planner_tap.log"
    _write_tap(
        tap,
        [
            """[2026-07-06T05:39:53] PLANNER provider=local_frontdoor role=draft start
------------------------------------------------------------------------
[FAIL provider=local_frontdoor role=draft] Server disconnected without sending a response.""",
            """[2026-07-06T05:43:11] PLANNER provider=local_worker role=critique start
------------------------------------------------------------------------
[local:result:local_worker:critique] ```json:autopilot_critique
{"decision": "reject", "confidence": 0.9, "issues": ["Action type deep_eval is not recognized by the declared action space."], "revised_action": null}
```
[END provider=local_worker role=critique] result_chars=288""",
        ],
    )

    report = reporter.build_report(tap_path=tap, stale_after_s=999999)

    assert report["ok"] is False
    assert report["status"] == "attention"
    assert "no successful local draft event in planner tap window" in report["blockers"]
    assert report["local"]["failures"] == 1
    assert report["critic_decisions"] == {"reject": 1}
    assert any(
        issue["kind"] == "critic_issue" and "deep_eval" in issue["message"]
        for issue in report["recent_issues"]
    )


def test_build_report_missing_tap_is_not_ok(tmp_path: Path) -> None:
    report = reporter.build_report(tap_path=tmp_path / "missing.log")

    assert report["ok"] is False
    assert report["status"] == "missing"
    assert report["providers"] == {}
    assert report["blockers"]
