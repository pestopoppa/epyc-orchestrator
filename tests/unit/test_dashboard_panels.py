"""Guard tests for the dashboard panel registry + freshness health endpoint.

These are the anti-regression core of the dashboard-freshness work: they fail
loudly if a panel is displayed without a registered/monitored source, if a
registered panel is never stamped, if the registry's path literals drift from
dashboard.py, or if the health endpoint stops folding every panel. Each past
"dashboard panel stale" incident was a different producer dying unnoticed; this
suite makes "add a panel, forget its freshness" a test failure.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from src.api.routes import dashboard as d
from src.api.routes import dashboard_panels as P

_DASHBOARD_SRC = Path(d.__file__).read_text()


def _call(coro):
    return json.loads(asyncio.run(coro).body.decode())


# --- registry integrity -------------------------------------------------------

def test_registry_paths_match_dashboard_constants():
    """The registry re-derives producer paths; they must equal dashboard.py's."""
    assert P.AUTOPILOT_PHASE_PATH == d.AUTOPILOT_PHASE_PATH
    assert P.ORCHESTRATOR_STATE_PATH == d.ORCHESTRATOR_STATE_PATH
    assert P.AUTOPILOT_LOG_PATH == d.AUTOPILOT_LOG
    assert P.AUTOPILOT_STATE_PATH == d._AUTOPILOT_STATE_PATH
    assert P.AUTOPILOT_JOURNAL_PATH == d._AUTOPILOT_JOURNAL_PATH


def test_panel_keys_unique():
    keys = [p.key for p in P.PANELS]
    assert len(keys) == len(set(keys)), f"duplicate panel keys: {keys}"


def test_every_registered_panel_is_stamped_by_an_endpoint():
    # Forward direction: each registered panel key is used as a _stamp() arg.
    for spec in P.PANELS:
        assert f'"{spec.key}"' in _DASHBOARD_SRC, f"panel {spec.key} registered but never stamped"


def test_no_stamp_uses_an_unregistered_key():
    # Reverse direction: every _stamp(payload, "key", ...) key is registered.
    # Scan a bounded window after each _stamp( call to find its panel-key arg
    # (a bare "identifier" string followed by ) or , now=...), skipping dict
    # keys which are always followed by ':'.
    key_arg = re.compile(r',\s*"([a-z_]+)"\s*(?:,\s*now=[^)]*)?\)')
    found: set[str] = set()
    for m in re.finditer(r'_stamp\(', _DASHBOARD_SRC):
        window = _DASHBOARD_SRC[m.end(): m.end() + 900]
        km = key_arg.search(window)
        if km:
            found.add(km.group(1))
    assert found, "no _stamp() keys extracted — regex or call sites changed"
    registered = set(P.PANELS_BY_KEY)
    assert found <= registered, f"stamped with unregistered keys: {found - registered}"


# --- route existence ----------------------------------------------------------

def test_every_registry_endpoint_is_a_real_route():
    route_paths = {getattr(r, "path", None) for r in d.router.routes}
    for spec in P.PANELS:
        assert spec.endpoint in route_paths, f"{spec.key}: {spec.endpoint} not a registered route"


def test_health_endpoint_is_registered():
    route_paths = {getattr(r, "path", None) for r in d.router.routes}
    assert "/dashboard/api/health" in route_paths


# --- health endpoint ----------------------------------------------------------

def test_health_folds_every_panel():
    h = _call(d.dashboard_health())
    assert h["panel_count"] == len(P.PANELS)
    assert {p["key"] for p in h["panels"]} == set(P.PANELS_BY_KEY)
    assert h["worst_class"] in {"fresh", "aging", "stale", "dead"}
    assert h["status"] in {"ok", "degraded"}
    # status must agree with worst_class
    assert (h["status"] == "degraded") == (h["worst_class"] in {"stale", "dead"})
    # degraded_panels must be exactly the stale/dead ones
    expected = {p["key"] for p in h["panels"] if p["staleness_class"] in ("stale", "dead")}
    assert set(h["degraded_panels"]) == expected


def test_health_live_panels_never_dead_on_their_own():
    # A live panel (topology/region_locks/...) has no gating file source, so it
    # can never be the reason the dashboard is 'degraded'.
    h = _call(d.dashboard_health())
    by_key = {p["key"]: p for p in h["panels"]}
    for spec in P.PANELS:
        if spec.live:
            assert by_key[spec.key]["staleness_class"] == "fresh"


# --- endpoints actually emit the envelope ------------------------------------

# --- journal shard following (the trial-999-frozen-panel bug) -----------------

def _write_journal(path: Path, trial_ids: list[int]) -> None:
    path.write_text("\n".join(json.dumps({"trial_id": t}) for t in trial_ids) + "\n")


def test_journal_shards_ordered_and_filtered(tmp_path, monkeypatch):
    base = tmp_path / "autopilot_journal.jsonl"
    _write_journal(base, [1])
    _write_journal(tmp_path / "autopilot_journal_1.jsonl", [2])
    _write_journal(tmp_path / "autopilot_journal_2.jsonl", [3])
    _write_journal(tmp_path / "autopilot_journal_10.jsonl", [4])
    # noise files that must NOT be treated as rotation shards
    _write_journal(tmp_path / "autopilot_journal_snapshot_123.jsonl", [99])
    _write_journal(tmp_path / "autopilot_journal_before_stall.jsonl", [98])
    monkeypatch.setattr(d, "_AUTOPILOT_JOURNAL_PATH", base)
    shards = [p.name for p in d._autopilot_journal_shards()]
    # numeric-suffix order (base first, then 1, 2, 10 — not lexical 1,10,2)
    assert shards == [
        "autopilot_journal.jsonl",
        "autopilot_journal_1.jsonl",
        "autopilot_journal_2.jsonl",
        "autopilot_journal_10.jsonl",
    ]


def test_read_journal_merges_across_rotation(tmp_path, monkeypatch):
    base = tmp_path / "autopilot_journal.jsonl"
    _write_journal(base, [997, 998, 999])
    _write_journal(tmp_path / "autopilot_journal_1.jsonl", [1000, 1001])
    monkeypatch.setattr(d, "_AUTOPILOT_JOURNAL_PATH", base)
    rows = d._read_autopilot_journal_rows()
    tids = [r["trial_id"] for r in rows]
    assert tids == [997, 998, 999, 1000, 1001], "journal must follow the rotation, not freeze at the base file"


def test_read_journal_none_when_no_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "_AUTOPILOT_JOURNAL_PATH", tmp_path / "autopilot_journal.jsonl")
    assert d._read_autopilot_journal_rows() is None


def test_explicit_path_reads_single_shard(tmp_path, monkeypatch):
    base = tmp_path / "autopilot_journal.jsonl"
    _write_journal(base, [1])
    other = tmp_path / "autopilot_journal_1.jsonl"
    _write_journal(other, [2])
    monkeypatch.setattr(d, "_AUTOPILOT_JOURNAL_PATH", base)
    # explicit path keeps single-file behaviour (no merge)
    rows = d._read_autopilot_journal_rows(other)
    assert [r["trial_id"] for r in rows] == [2]


_ENVELOPE_KEYS = {"generated_at", "sources", "worst_age_s", "staleness_class", "reason"}


@pytest.mark.parametrize("coro_factory,key", [
    (lambda: d.topology(), "topology"),
    (lambda: d.region_locks_snapshot(), "region_locks"),
    (lambda: d.inference_tap_snapshot(3), "inference_tap"),
    (lambda: d.autopilot_progress(), "autopilot_progress"),
    (lambda: d.process_status(), "process_status"),
    (lambda: d.gepa_status(), "gepa"),
])
def test_stamped_endpoint_emits_wellformed_freshness(coro_factory, key):
    body = _call(coro_factory())
    assert "_freshness" in body, f"{key} endpoint missing _freshness"
    env = body["_freshness"]
    assert _ENVELOPE_KEYS <= set(env), f"{key} envelope missing keys: {_ENVELOPE_KEYS - set(env)}"
    assert env["staleness_class"] in {"fresh", "aging", "stale", "dead"}
    # sources carry the gating flag so the frontend can separate gating vs info.
    for s in env["sources"]:
        assert "gating" in s and "class" in s and "label" in s
