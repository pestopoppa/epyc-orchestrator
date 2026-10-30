"""The N-way non-ALLOW counter reaches the contention-gate panel compatibly.

The counter is computed in `src/scheduling/contention_gate.py::ContentionGate.evaluate()`
(J4c: the measured N-way matrix can return non-ALLOW even when every pairwise
combination in the active set is ALLOW) and was already wired into
`ContentionGate.metrics_snapshot()` and the `/dashboard/api/contention` JSON payload
(`src/api/routes/dashboard.py::contention_gate_snapshot`). The producer keeps the old
wire key as an alias during rolling upgrades; updated dashboards prefer the accurate
new key and fall back to the alias while older servers are still running.

These tests exercise the REAL browser-side renderer (via `node`), not a description of
it. They require the label and tooltip to state exactly what is counted and cover old,
new, and mixed-version payloads. There is no
matching denominator for N-way checks, so the display does not use the unrelated
admitted-decision counter as one.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

_HTML_PATH = (
    Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
)


def _extract_update_contention_gate() -> str:
    body = _HTML_PATH.read_text()
    start = body.index("async function updateContentionGate(")
    end = body.index("async function updateRegionLocks(", start)
    return body[start:end]


def _run_with_payload(payload: dict) -> str:
    """Execute the real `updateContentionGate()` JS against a stubbed fetch/DOM."""
    fn = _extract_update_contention_gate()
    script = f"""
let _regionLocksRefreshSeq = 1;
const escapeHTML = value => String(value);
const fetchJSON = async () => ({json.dumps(payload)});
const el = {{innerHTML: ''}};
const document = {{getElementById: (id) => (id === 'contention-gate-compact' ? el : null)}};
{fn}
(async () => {{
  await updateContentionGate(1);
  process.stdout.write(JSON.stringify({{html: el.innerHTML}}));
}})().catch(err => {{ console.error(err); process.exit(1); }});
"""
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["html"]


_BASE_PAYLOAD = {
    "matrix_status": "ok",
    "contention_admitted_count": 42,
    "contention_timeout_count": 0,
    "contention_degraded_allow_count": 0,
    "contention_wait_seconds": 1.5,
    "contention_unknown_pair_count": 0,
    "contention_blocked_count": {},
    "per_role_scheduling": {},
}


def test_dashboard_html_reads_nway_non_allow_key() -> None:
    """The new key is preferred, while old servers remain readable during rollout."""
    fn = _extract_update_contention_gate()
    assert "contention_nway_non_allow_count" in fn
    assert "contention_nway_restricted_count" in fn


def test_nway_non_allow_count_reaches_rendered_panel_html() -> None:
    """The real JS renderer must put the count and accurate tooltip in the DOM."""
    payload = dict(_BASE_PAYLOAD, contention_nway_non_allow_count=3)
    html = _run_with_payload(payload)

    assert "nway-non-allow" in html
    assert ">3<" in html
    assert "3 N-way check(s) returned non-ALLOW" in html
    assert "pairwise decision" in html
    assert "of 42 admitted decision" not in html


def test_nway_non_allow_count_zero_still_renders_without_warning() -> None:
    """Zero case: still rendered (not silently dropped), not colored as a warning."""
    payload = dict(_BASE_PAYLOAD, contention_nway_non_allow_count=0)
    html = _run_with_payload(payload)

    assert "nway-non-allow" in html
    assert ">0<" in html


def test_new_dashboard_reads_old_api_server_alias_during_rolling_upgrade() -> None:
    """New browser code can run against an old server that only emits the old key."""
    payload = dict(_BASE_PAYLOAD, contention_nway_restricted_count=4)
    html = _run_with_payload(payload)

    assert "nway-non-allow" in html
    assert ">4<" in html
    assert "4 N-way check(s) returned non-ALLOW" in html


def test_new_dashboard_key_wins_when_both_versions_are_present() -> None:
    """The deprecated alias cannot override the new field during mixed rollout."""
    payload = dict(
        _BASE_PAYLOAD,
        contention_nway_non_allow_count=2,
        contention_nway_restricted_count=9,
    )
    html = _run_with_payload(payload)

    assert ">2<" in html
    assert "2 N-way check(s) returned non-ALLOW" in html
    assert ">9<" not in html


def test_new_zero_value_is_not_replaced_by_nonzero_deprecated_alias() -> None:
    """Zero is a valid new-server value, not a cue to fall back to the old alias."""
    payload = dict(
        _BASE_PAYLOAD,
        contention_nway_non_allow_count=0,
        contention_nway_restricted_count=9,
    )
    html = _run_with_payload(payload)

    assert ">0<" in html
    assert ">9<" not in html
