"""Route-level test for the /dashboard endpoint after Tranche-3 HTML extraction.

Verifies that the route returns the HTML loaded from dashboard.html (the
extracted static file) — guards against the file going missing or the
loader path breaking after future refactors.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


@pytest.fixture
def dashboard_route():
    """Import dashboard.py + return the route handler function for /dashboard."""
    from src.api.routes import dashboard
    # Find the GET /dashboard route in the router and return its endpoint
    for route in dashboard.router.routes:
        if getattr(route, "path", None) == "/dashboard":
            return route.endpoint
    pytest.skip("/dashboard route not registered")


def test_dashboard_route_returns_extracted_html(dashboard_route) -> None:
    """The /dashboard endpoint serves the same HTML body that's in dashboard.html."""
    response = asyncio.run(dashboard_route())
    # FastAPI HTMLResponse has .body as bytes
    body = response.body.decode("utf-8") if isinstance(response.body, bytes) else response.body
    expected = (Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html").read_text()
    assert body == expected


def test_dashboard_html_response_starts_with_doctype(dashboard_route) -> None:
    response = asyncio.run(dashboard_route())
    body = response.body.decode("utf-8") if isinstance(response.body, bytes) else response.body
    assert body.startswith("<!doctype html>")


def test_dashboard_html_response_ends_with_closing_tags(dashboard_route) -> None:
    response = asyncio.run(dashboard_route())
    body = response.body.decode("utf-8") if isinstance(response.body, bytes) else response.body
    assert body.rstrip().endswith("</body></html>")


def test_dashboard_html_file_exists_at_expected_path() -> None:
    """The extracted dashboard.html file must live alongside dashboard.py."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    assert html_path.exists()
    assert html_path.stat().st_size > 40_000  # ~43KB after extraction


def test_dashboard_html_loaded_at_module_import() -> None:
    """_DASHBOARD_HTML constant in dashboard.py should be populated."""
    from src.api.routes import dashboard
    assert len(dashboard._DASHBOARD_HTML) > 40_000


def test_dashboard_html_distinguishes_waiting_tap_from_active_locks() -> None:
    """The live tap can show CPU-lock wait states without counting them as holders."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert "waiting_cpu_lock" in body
    assert "tap-inferred active stream" in body
    assert "TAP ACTIVE" in body
    assert "structuredTapPrimaryRole" in body
    assert "function inferStructuredTapLockIdentity(req, byRole = null)" in body
    assert "const topoNode = (topology && Number.isFinite(port))" in body
    assert "const identity = inferStructuredTapLockIdentity(req" in body
    assert "status === 'quiet'" in body
    assert "blocked_by_roles" in body
    assert "lockOnlyStructuredTapHolders" in body
    assert "chat.* tap absent" in body
    assert "holders · ${tapped} tapped · ${offTap} off-tap" in body


def test_dashboard_run_state_active_inference_overrides_quiet_log() -> None:
    """Active tap/lock work should not render the top run-state as quiet."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert "activeInferenceCount > 0 || activeLockCount > 0" in body
    assert "runState = 'inference active'" in body
    assert "runState = hasActiveInference ? 'orphan inference' : 'down'" in body


def test_dashboard_autopilot_log_render_dedupes_adjacent_lines() -> None:
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert "function _dedupeAdjacentAutopilotLines(lines)" in body
    assert "_dedupeAdjacentAutopilotLines(_autopilotLogBuffer.split('\\n'))" in body


def test_dashboard_topology_activity_stats_refresh_with_live_age_tick() -> None:
    """Topology activity text should not lag behind lock/tap freshness signals."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert "TOPOLOGY_ACTIVITY_WINDOW_S = 60" in body
    assert "TOPOLOGY_ACTIVITY_POLL_MS = 1500" in body
    assert "TOPOLOGY_ACTIVITY_AGE_TICK_MS = 1000" in body
    assert "topologyActivityAgeS" in body
    assert "renderTopologyActivity" in body
    assert "fetchJSON(`/dashboard/api/topology_activity?window_s=${TOPOLOGY_ACTIVITY_WINDOW_S}`)" in body
    assert "scheduleRegionLocksRefresh(true)" in body
    assert "snap.region_locks" in body
    assert "return updateRegionLocks(refreshSeq, snap.region_locks);" in body
    assert "const refreshSeq = ++_regionLocksRefreshSeq;" in body
    assert "updateContentionGate(refreshSeq);" in body
    assert "let _topologyActivityRefreshSeq = 0;" in body
    assert "async function updateTopologyActivity(refreshSeq = ++_topologyActivityRefreshSeq)" in body
    assert "updateTopologyActivity();" in body
    assert "setInterval(() => scheduleRegionLocksRefresh(true), 1500)" in body
    assert "setInterval(renderTopologyActivity, TOPOLOGY_ACTIVITY_AGE_TICK_MS)" in body
    assert "lockActivitySignature" in body
    assert "const liveActiveCount = Math.max(lockActiveCount, tapActiveCount);" in body
    assert "active CPU-region/tap holder(s)" in body
    assert "summary ${stats.avg_tps_recent.toFixed(2)} t/s" in body
    assert "renderRegionLocksBasicGrid(grid, d);" in body
    assert "grid.dataset.regionLocksPainted = '1';" in body
    assert "basic matrix fallback; rich overlay still initializing" in body
    assert "loading CPU region-lock matrix" in body
    assert "function startPanelSafely(name, fn)" in body
    assert "startPanelSafely('region-locks-primer', ensureRegionLocksPanelPainted);" in body
    assert "startPanelSafely('region-locks-refresh', () => scheduleRegionLocksRefresh(true));" in body
    assert "startPanelSafely('topology', loadTopology);" in body
    assert "startPanelSafely('snapshot-poll', updateSnapshotPoll);" in body
    assert "setTimeout(ensureRegionLocksPanelPainted, 750)" in body
    assert "_latestSnapshot = snap || {};" in body
    assert "buildSlotInferredRegionLocks(byRole)" in body
    assert "SLOT ACTIVE" in body
    assert "slot-inferred active instance(s)" in body
    assert "Tap PIDs identify backend llama-server processes" in body
    assert "active.instanceIdxs && active.instanceIdxs.has(Number(idx))" in body
    assert "function repaintRegionLocksFromStructuredTapFrame()" in body
    assert "repaintRegionLocksFromStructuredTapFrame();" in body
    assert "function structuredTapLogicalAliasesForIdentity(role, idx)" in body
    assert "formatPhysicalRoleWithLogicalAliases(holder.role, holder.shape, holder.idx)" in body
    assert "logical route(s): ${aliases.join(', ')}" in body
    assert "logical ${escapeHTML(req.role)}" in body


def test_dashboard_live_panel_refreshes_ignore_stale_responses_where_possible() -> None:
    """Live inference and CPU-lock refreshes should not repaint older responses."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert "let _processStatusFetchSeq = 0;" in body
    assert "const requestSeq = ++_processStatusFetchSeq;" in body
    assert "if (requestSeq !== _processStatusFetchSeq) return;" in body
    assert "let _regionLocksRefreshSeq = 0;" in body
    assert "const refreshSeq = ++_regionLocksRefreshSeq;" in body
    assert "updateRegionLocks(refreshSeq);" in body
    assert "updateContentionGate(refreshSeq);" in body
    assert "updateTopologyActivity();" in body
    assert "function applyDashboardSnapshot(snap, source)" in body
    assert "updatePanelSafely('completed', () => updateTasks(snap));" in body
    assert "updatePanelSafely('decisions', () => updateDecisions(snap));" in body
    assert "const result = fn();" in body
    assert "result.catch((err) =>" in body
    assert "return updateRegionLocks(refreshSeq, snap.region_locks);" in body
    assert "let _lastRegionLocksPayload = null;" in body
    assert "rich overlay failed" in body
    assert "fetchJSON('/dashboard/api/snapshot'" in body
    assert "timeoutMs: _SNAPSHOT_POLL_TIMEOUT_MS" in body
    assert "setInterval(updateSnapshotPoll, 2500)" in body
    assert "window_s=${TOPOLOGY_ACTIVITY_WINDOW_S}" in body
    assert "fetchJSON('/dashboard/api/contention')" in body
    assert "if (refreshSeq !== _regionLocksRefreshSeq) return;" in body


def test_dashboard_transport_self_heals_without_page_reload() -> None:
    """Wedge-killers + watchdog: a dead stream, hung fetch, or poisoned frame
    watermark must recover on its own — the trio of stream-fed panels
    (topology / region locks / live tap) froze permanently on these before."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    # B1: timeout-bounded snapshot poll with a self-expiring in-flight guard.
    assert "async function fetchJSON(url, { timeoutMs = _FETCH_JSON_TIMEOUT_MS } = {})" in body
    assert "let _snapshotPollInFlightSince = 0;" in body
    assert "_snapshotPollInFlightSince &&" in body
    assert "now - _snapshotPollInFlightSince < _SNAPSHOT_POLL_TIMEOUT_MS + 2000" in body
    assert "_snapshotPollInFlight = false" not in body  # old permanent-wedge boolean

    # B2: no client-clock fallback for the frame watermark; null frames apply
    # but never advance it; watermark resets on every (re)connect.
    assert "if (!Number.isFinite(ts)) return null;" in body
    assert "Date.now() / 1000 + 120" in body
    assert body.count("_latestSnapshotFrameTs = 0;") >= 3  # decl init + both stream starters + watchdog
    assert "if (frameTs !== null) {" in body

    # B3: universal watchdog rebuilds the stream + fires a poll when snapshots
    # stop applying; hooks cover sleep/wake and network flaps.
    assert "function snapshotTransportWatchdog()" in body
    assert "setInterval(snapshotTransportWatchdog, 5000)" in body
    assert "document.addEventListener('visibilitychange'" in body
    assert "window.addEventListener('online', snapshotTransportWatchdog)" in body
    assert "let _lastSnapshotAppliedAt = Date.now();" in body

    # B7: every legacy EventSource reconnect is guarded (no stampedes) and
    # identity-checked (no stale-closure restarts).
    assert body.count("es._reconnectScheduled = true;") >= 5
    assert "if (_autopilotLogStream === es) startAutopilotLogStream();" in body
    assert "if (_rawTapStream === es) startRawTapStream();" in body
    assert "if (_structuredTapStream === es) startStructuredTapStream();" in body
    assert "if (_plannerTapStream === es) startPlannerTapStream();" in body


def test_dashboard_pareto_plot_uses_journal_sources_and_nonnegative_axes() -> None:
    """Quality/speed axes should not render negative tick labels from padding."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert "startsWith('journal_')" in body
    assert "const xLo = Math.max(0, xMin - xPad)" in body
    assert "const yLo = Math.max(0, yMin - yPad)" in body
    assert "d.canonical_tier" in body
    assert "d.frontiers_by_tier" in body
    assert "tiers ${tierKeys.map" in body
    assert "const legendY = VB.h - PAD.b - 4;" in body
    assert "paretoTierLegend(legendKeys, VB.w - 44, legendY, eraMode)" in body
    assert "paretoTierLegend(series.map(s => s[0]), VB.w - 38, legendY)" in body
    # All-era scope wiring: toggle, era-labeled fetch, underlay clouds + bands.
    assert "setParetoScope('all_eras')" in body
    assert "/dashboard/api/pareto?scope=" in body
    assert "function convexHull2D(pts)" in body
    assert "paretoEraLegend(eras, PAD.l + 6, PAD.t + 12)" in body


def test_dashboard_gepa_and_pareto_surface_real_suite_metrics() -> None:
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert "function realSuiteBadge(metric)" in body
    assert "real_suite_v1 · q=" in body
    assert "realSuiteBadge(t.real_suite_v1)" in body
    assert "const suiteTip = p =>" in body
    assert "p.real_suite_v1" in body
    assert "real_suite_v1 q=" in body


def test_dashboard_autopilot_progress_includes_eval_label() -> None:
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert "const evalLabel = prog.eval_label || phase.eval_label || '';" in body
    assert "evalLabel ? `trial #${d.trial_id} (${action}, ${evalLabel})`" in body
    assert "${evalLabel || 'T?'} tower ${lp.completed}/${lp.total}" in body
    assert "${escapeHTML(String(evalLabel)) || 'T?'} tower ${prog.log_tail_progress.completed}/${prog.log_tail_progress.total}" in body
    assert "const promotions = d.baseline_promotions || {};" in body
    assert "baseline promotions ${promotions.count}" in body
    assert "const outcome = d.outcome_kpis || {};" in body
    assert "keepable ${fmtRate(keepable)}" in body
    assert "wasted-eval ${fmtRate(wasted)}" in body
    assert "learning-excluded ${fmtRate(excluded)}" in body
    assert "_currentCodeHealthLabel(d.current_code_health)" in body
    assert "current code ${health.status}" in body


def test_dashboard_repo_readiness_panel_is_advisory_only() -> None:
    """Repo-readiness queue can render in the dashboard without becoming a gate."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert 'id="repo-readiness-panel"' in body
    assert 'id="repo-readiness-open"' in body
    assert 'id="repo-readiness-list"' in body
    assert "/dashboard/api/repo_readiness" in body
    assert "function renderRepoReadiness(data)" in body
    assert "setInterval(updateRepoReadiness, 60000)" in body
    assert "data.authority || 'advisory'" in body
    assert "data.autopilot_gate ? 'true' : 'false'" in body


def test_dashboard_insight_graph_panel_is_wired_read_only() -> None:
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert 'id="insight-graph-panel"' in body
    assert "planner insight graph" in body
    assert 'id="insight-graph-svg"' in body
    assert 'id="insight-graph-query"' in body
    assert "focus by trial, strategy, hint, campaign, species, or handoff" in body
    assert "/dashboard/api/insight_graph" in body
    assert "function renderInsightGraph(graph)" in body
    assert "function updateInsightGraph(focusOverride)" in body
    assert "setInterval(() => updateInsightGraph(_insightGraphFocus), 20000)" in body


def test_dashboard_operational_panels_stay_directly_under_topology() -> None:
    """Completed/routing must stay directly under topology in the right-column scan path."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    topology_idx = body.index('id="topology-strip"')
    completed_idx = body.index('id="completed-tasks"')
    routing_idx = body.index('id="decision-feed"')
    readiness_idx = body.index('id="repo-readiness-panel"')

    assert topology_idx < completed_idx < routing_idx < readiness_idx


# ----- dashboard_tasks: timezone-aware UTC (Tranche-8 polish) -----


def test_task_text_snapshot_uses_timezone_aware_utc(monkeypatch) -> None:
    """Tranche-8 fix: datetime.utcnow() replaced with datetime.now(timezone.utc).

    No DeprecationWarning should be emitted by _task_text_snapshot.
    """
    import warnings
    from src.api.routes import dashboard_tasks

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Run with an empty event list — exercises the timestamp formatting path
        out = dashboard_tasks._task_text_snapshot("chat-test", [], None)
    # Header should still contain the "Z" suffix marker
    assert "@ " in out
    assert "Z ===" in out
