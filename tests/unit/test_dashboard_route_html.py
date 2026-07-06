"""Route-level test for the /dashboard endpoint after Tranche-3 HTML extraction.

Verifies that the route returns the HTML loaded from dashboard.html (the
extracted static file) — guards against the file going missing or the
loader path breaking after future refactors.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import textwrap
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
    assert "tap-inferred request(s)" in body
    assert "TAP ACTIVE" in body
    assert "structuredTapPrimaryRole" in body
    assert "function inferStructuredTapLockIdentity(req, byRole = null)" in body
    assert "const topoNode = (topology && Number.isFinite(port))" in body
    assert "const roleCandidates = [" in body
    assert "roleCandidates.find(r => byRole[r])" in body
    assert "const identity = inferStructuredTapLockIdentity(req" in body
    assert "status === 'quiet'" in body
    assert "blocked_by_roles" in body
    assert "lockOnlyStructuredTapHolders" in body
    assert "chat.* tap absent" in body
    assert "holders · ${tapped} tapped · ${offTap} off-tap" in body
    assert "live tap request(s)" in body
    assert "prefill/decode pending" in body
    assert "tapLiveRequestCount" in body


def test_dashboard_html_separates_proc_holders_from_live_tap_requests() -> None:
    """The lock panel should distinguish real /proc holders from tap inference."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert "procHolderCount" in body
    assert "tapInferredCount" in body
    assert "slotInferredCount" in body
    assert "tapLiveStateBits" in body
    assert "live tap request(s)" in body
    assert "/proc holder instance(s)" in body
    assert "structuredTapLockCandidates" in body
    assert "tap-inferred request(s)" in body


def test_dashboard_html_infers_quiet_tap_requests_into_region_lock_candidates() -> None:
    """Quiet open tap requests can still hold locks, so the lock overlay must paint them."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()
    start = body.index("function currentAutopilotActionSource()")
    end = body.index("function refreshLiveDotsFromStructuredTap()")
    snippet = body[start:end]

    script = textwrap.dedent(
        f"""
        const vm = require('vm');
        const ctx = {{
          _latestProcessStatus: {{}},
          _latestRegionLocksByRole: {{}},
          _latestTapInferredRegionLocksByRole: {{}},
          _lastRegionLocksPayload: {{
            by_role: {{
              worker_general: {{
                instances: [
                  {{ idx: 0, shape: 'full', regions: ['q0', 'q1', 'q2', 'q3'] }},
                  {{ idx: 3, shape: 'q2', regions: ['q2'] }},
                ],
              }},
            }},
          }},
          _structuredTapRequests: [
            {{
              request_id: 'streaming',
              role: 'frontdoor',
              lock_role: 'worker_general',
              instance_idx: 0,
              instance_shape: 'full',
              instance_regions: ['q0', 'q1', 'q2', 'q3'],
              status: 'running',
              chunk_count: 2,
              response_len: 64,
              quiet_s: 0,
            }},
            {{
              request_id: 'quiet',
              role: 'architect',
              lock_role: 'worker_general',
              instance_idx: 3,
              instance_shape: 'q2',
              instance_regions: ['q2'],
              status: 'quiet',
              chunk_count: 2,
              response_len: 64,
              quiet_s: 0,
            }},
            {{
              request_id: 'waiting',
              role: 'ingest',
              lock_role: 'worker_general',
              instance_idx: 3,
              instance_shape: 'q2',
              instance_regions: ['q2'],
              status: 'running',
              chunk_count: 0,
              response_len: 0,
              quiet_s: 0,
            }},
          ],
          _STRUCTURED_TAP_STALLED_S: 999999,
          escapeHTML: (s) => String(s),
          clientBaseRole: (s) => String(s || ''),
          roleColor: () => 'x',
          inferStructuredTapLockIdentity: (req) => ({{
            role: req.lock_role || req.topology_role || req.role || '',
            idx: Number(req.instance_idx),
            regions: Array.isArray(req.instance_regions) ? req.instance_regions : [],
            shape: req.instance_shape || '',
          }}),
          structuredTapHasActiveCpuLock: (req) => req.request_id !== 'waiting',
          structuredTapCpuBlockers: (req) => req.request_id === 'waiting' ? ['worker_general'] : [],
          structuredTapCpuBlockerSummary: (blockers) => blockers.join(','),
          topology: {{ nodes: [] }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        const candidateIds = vm.runInContext(
          'structuredTapLockCandidates().map((req) => req.request_id).sort().join(\",\")',
          ctx,
        );
        const inferred = vm.runInContext(`
          const out = buildTapInferredRegionLocks();
          JSON.stringify(Object.fromEntries(Object.entries(out).map(([role, bucket]) => [role, {{
            instanceIdxs: [...bucket.instanceIdxs].map(Number).sort((a, b) => a - b),
            requestIds: bucket.sources.map((req) => req.request_id).sort(),
          }}])));
        `, ctx);
        vm.runInContext('_latestTapInferredRegionLocksByRole = buildTapInferredRegionLocks();', ctx);
        const summary = vm.runInContext('formatRegionLockHolderSummary()', ctx);
        if (candidateIds !== 'quiet,streaming') {{
          throw new Error(`unexpected candidate ids: ${{candidateIds}}`);
        }}
        const parsed = JSON.parse(inferred);
        if (!parsed.worker_general) {{
          throw new Error('missing worker_general bucket');
        }}
        if (parsed.worker_general.instanceIdxs.join(',') !== '0,3') {{
          throw new Error(`unexpected instanceIdxs: ${{parsed.worker_general.instanceIdxs.join(',')}}`);
        }}
        if (parsed.worker_general.requestIds.join(',') !== 'quiet,streaming') {{
          throw new Error(`unexpected requestIds: ${{parsed.worker_general.requestIds.join(',')}}`);
        }}
        if (!summary.includes('tap-inferred quiet/held')) {{
          throw new Error(`summary did not include quiet holder: ${{summary}}`);
        }}
        if (!summary.includes('tap-inferred streaming')) {{
          throw new Error(`summary did not include streaming holder: ${{summary}}`);
        }}
        if (!summary.includes('frontdoor') || !summary.includes('architect')) {{
          throw new Error(`summary did not include logical aliases: ${{summary}}`);
        }}
        """
    )

    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout


def test_dashboard_html_preserves_tap_inferred_holders_even_when_proc_label_matches() -> None:
    """A tapped holder should still appear when an older /proc label matches it."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()
    start = body.index("function currentAutopilotActionSource()")
    end = body.index("function refreshLiveDotsFromStructuredTap()")
    snippet = body[start:end]

    script = textwrap.dedent(
        f"""
        const vm = require('vm');
        const ctx = {{
          _latestProcessStatus: {{}},
          _latestRegionLocksByRole: {{
            worker_general: {{
              instanceIdxs: new Set([0]),
              procInstanceIdxs: [0],
              regions: new Set(['q0', 'q1', 'q2', 'q3']),
              holderPids: new Set(['111']),
              pidsByInstanceIdx: new Map([[0, new Set(['111'])]]),
              instances: [{{ idx: 0, shape: 'full', regions: ['q0', 'q1', 'q2', 'q3'] }}],
            }},
          }},
          _latestTapInferredRegionLocksByRole: {{
            worker_general: {{
              instanceIdxs: new Set([0]),
              regions: new Set(['q0', 'q1', 'q2', 'q3']),
              holderPids: new Set(['222']),
              pidsByInstanceIdx: new Map([[0, new Set(['222'])]]),
              sources: [{{
                request_id: 'streaming',
                role: 'frontdoor',
                lock_role: 'worker_general',
                instance_idx: 0,
                instance_shape: 'full',
                instance_regions: ['q0', 'q1', 'q2', 'q3'],
                status: 'running',
                chunk_count: 2,
                response_len: 64,
                quiet_s: 0,
              }}],
            }},
          }},
          _lastRegionLocksPayload: {{
            by_role: {{
              worker_general: {{
                instances: [{{ idx: 0, shape: 'full', regions: ['q0', 'q1', 'q2', 'q3'] }}],
              }},
            }},
          }},
          _STRUCTURED_TAP_STALLED_S: 999999,
          _structuredTapRequests: [{{
            request_id: 'streaming',
            role: 'frontdoor',
            lock_role: 'worker_general',
            instance_idx: 0,
            instance_shape: 'full',
            instance_regions: ['q0', 'q1', 'q2', 'q3'],
            status: 'running',
            chunk_count: 2,
            response_len: 64,
            quiet_s: 0,
          }}],
          escapeHTML: (s) => String(s),
          clientBaseRole: (s) => String(s || ''),
          roleColor: () => 'x',
          inferStructuredTapLockIdentity: () => null,
          structuredTapHasActiveCpuLock: () => true,
          structuredTapCpuBlockers: () => [],
          structuredTapCpuBlockerSummary: (blockers) => blockers.join(','),
          topology: {{ nodes: [] }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        const summary = vm.runInContext('formatRegionLockHolderSummary()', ctx);
        const occurrences = (summary.match(/worker_general\\.full/g) || []).length;
        if (occurrences < 2) {{
          throw new Error(`expected both proc and tap-inferred entries, got: ${{summary}}`);
        }}
        if (!summary.includes('tap-inferred streaming')) {{
          throw new Error(`expected tap-inferred holder in summary, got: ${{summary}}`);
        }}
        """
    )

    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout


def test_dashboard_html_merges_unique_tap_inferred_holders_into_region_lock_summary() -> None:
    """The Regions Lock header should list tap-inferred holders alongside real /proc holders."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert "function formatRegionLockHolderSummary()" in body
    assert "const displayHeldBySummary = formatRegionLockHolderSummary();" in body
    assert "function tapInferredHolderLabel(role, shape, idx, sources)" in body
    assert "tap-inferred active/pending" in body
    assert "tap-inferred streaming" in body
    assert "tap-inferred quiet/held" in body
    assert "let tapInferredRequestCount = 0" in body
    assert "tapInferredRequestCount += tapInferredSourceCount(inferred)" in body
    assert "const tapInferredCountSuffix = tapInferredRequestCount" in body
    assert "const countSuffix = holderSources.length > 1 ? ` ×${holderSources.length}` : ''" in body
    assert "/proc holder instance(s)${tapInferredCountSuffix}: ${displayHeldBySummary" in body
    assert "~${formatPhysicalRoleWithLogicalAliases(role, shape, idx)}${countSuffix} (${inferredState})" in body


def test_dashboard_html_counts_multiple_tap_requests_on_one_physical_holder() -> None:
    """Eval batches can stream multiple requests through one lock; do not hide that behind one cell."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()
    start = body.index("function currentAutopilotActionSource()")
    end = body.index("function refreshLiveDotsFromStructuredTap()")
    snippet = body[start:end]

    script = textwrap.dedent(
        f"""
        const vm = require('vm');
        const ctx = {{
          _latestProcessStatus: {{}},
          _latestRegionLocksByRole: {{
            worker_general: {{
              instanceIdxs: new Set([0]),
              procInstanceIdxs: new Set([0]),
              regions: new Set(['q0', 'q1', 'q2', 'q3']),
              holderPids: new Set(['111']),
              pidsByInstanceIdx: new Map([[0, new Set(['111'])]]),
              instances: [{{ idx: 0, shape: 'full', regions: ['q0', 'q1', 'q2', 'q3'] }}],
            }},
          }},
          _latestTapInferredRegionLocksByRole: {{}},
          _lastRegionLocksPayload: {{
            by_role: {{
              worker_general: {{
                instances: [{{ idx: 0, shape: 'full', regions: ['q0', 'q1', 'q2', 'q3'] }}],
              }},
            }},
          }},
          _structuredTapRequests: [
            {{
              request_id: 'a',
              role: 'worker_general',
              lock_role: 'worker_general',
              instance_idx: 0,
              instance_shape: 'full',
              instance_regions: ['q0', 'q1', 'q2', 'q3'],
              status: 'running',
              chunk_count: 2,
              response_len: 64,
              quiet_s: 0,
            }},
            {{
              request_id: 'b',
              role: 'worker_general',
              lock_role: 'worker_general',
              instance_idx: 0,
              instance_shape: 'full',
              instance_regions: ['q0', 'q1', 'q2', 'q3'],
              status: 'running',
              chunk_count: 2,
              response_len: 64,
              quiet_s: 0,
            }},
          ],
          _STRUCTURED_TAP_STALLED_S: 999999,
          escapeHTML: (s) => String(s),
          clientBaseRole: (s) => String(s || ''),
          roleColor: () => 'x',
          inferStructuredTapLockIdentity: (req) => ({{
            role: req.lock_role || req.topology_role || req.role || '',
            idx: Number(req.instance_idx),
            regions: Array.isArray(req.instance_regions) ? req.instance_regions : [],
            shape: req.instance_shape || '',
          }}),
          structuredTapHasActiveCpuLock: () => true,
          structuredTapCpuBlockers: () => [],
          structuredTapCpuBlockerSummary: (blockers) => blockers.join(','),
          topology: {{ nodes: [] }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        vm.runInContext('_latestTapInferredRegionLocksByRole = buildTapInferredRegionLocks();', ctx);
        const inferredCount = vm.runInContext(
          'tapInferredSourceCount(_latestTapInferredRegionLocksByRole.worker_general)',
          ctx,
        );
        const summary = vm.runInContext('formatRegionLockHolderSummary()', ctx);
        if (inferredCount !== 2) {{
          throw new Error(`expected two tap-inferred requests, got ${{inferredCount}}`);
        }}
        if (!summary.includes('worker_general.full ×2')) {{
          throw new Error(`expected source count in summary, got: ${{summary}}`);
        }}
        """
    )

    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout


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


def test_dashboard_planner_status_distinguishes_history_from_active_stream() -> None:
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert "function _plannerTapStatusText(lines)" in body
    assert "planner history · autopilot stopped" in body
    assert "planner active" in body
    assert "plannerStatus.textContent = _plannerTapStatusText(plannerLines);" in body


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
    assert "const snapshotSeq = ++_latestSnapshotSeq;" in body
    assert "return updateRegionLocks(refreshSeq, snap.region_locks, snapshotSeq, snap.in_flight_tasks || [])" in body
    assert "const refreshSeq = ++_regionLocksRefreshSeq;" in body
    assert "updatePanelSafely('contention', () => updateContentionGate(refreshSeq));" in body
    assert "function updateTopology(activity, inflight, snapshotSeq = null)" in body
    assert "updateTopology(snap.display_activity || snap.activity || {}, snap.in_flight_tasks || [], snapshotSeq)" in body
    assert "let _latestStructuredTapFrameTs = 0;" in body
    assert "function applyStructuredTapFrame(data, {" in body
    assert "if (Array.isArray(snap.structured_requests))" in body
    assert "applyStructuredTapFrame(snap, {" in body
    assert "requestSnapshot: false" in body
    assert "acceptCoherentSnapshot: true" in body
    assert "if (!acceptCoherentSnapshot && frameTs + 0.001 < _latestStructuredTapFrameTs) return false;" in body
    assert "tap-inferred CPU holders and the lock grid reflect the same instant" in body
    assert "let _topologyActivityRefreshSeq = 0;" in body
    assert "async function updateTopologyActivity(refreshSeq = ++_topologyActivityRefreshSeq)" in body
    assert "if (snap.topology_activity) applyTopologyActivityPayload(snap.topology_activity);" in body
    assert "idle, not an active-holder signal" in body
    assert "last <span class=\"stat-stale\">${formatTopologyActivityAge(age)}</span>" in body
    assert "not an active-holder signal" in body
    assert "historical avg ${stats.avg_tps_recent.toFixed(2)} t/s" in body
    assert "history <span class=\"stat-stale\">${nCompleted}</span> done" in body
    assert "t/s hist" in body
    assert "servers up${suffix}" in body
    assert "setInterval(() => scheduleRegionLocksRefresh(true), 1500)" in body
    assert "setInterval(renderTopologyActivity, TOPOLOGY_ACTIVITY_AGE_TICK_MS)" in body
    assert "lockActivitySignature" in body
    assert "const liveActiveCount = Math.max(lockActiveCount, tapActiveCount);" in body
    assert "active CPU-region/tap holder(s)" in body
    assert "historical summary ${stats.avg_tps_recent.toFixed(2)} t/s" in body
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
    assert "_latestSlotInferredRegionLocksByRole = {};" in body
    assert "never promote raw llama /slots" in body
    assert "Tap PIDs identify backend llama-server processes" in body
    assert "active.instanceIdxs && active.instanceIdxs.has(Number(idx))" in body
    assert "function requestCoherentDashboardSnapshot(reason = '')" in body
    assert "requestCoherentDashboardSnapshot('structured_tap')" in body
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
    assert "async function updateRegionLocks(" in body
    assert "snapshotSeq = null," in body
    assert "snapshotInflightTasks = null" in body
    assert "updatePanelSafely('contention', () => updateContentionGate(refreshSeq));" in body
    assert "function applyTopologyActivityPayload(d)" in body
    assert "function applyDashboardSnapshot(snap, source)" in body
    assert "updatePanelSafely('completed', () => updateTasks(snap));" in body
    assert "updatePanelSafely('decisions', () => updateDecisions(snap));" in body
    assert "const result = fn();" in body
    assert "result.catch((err) =>" in body
    assert "return updateRegionLocks(refreshSeq, snap.region_locks, snapshotSeq, snap.in_flight_tasks || [])" in body
    assert "function requestCoherentDashboardSnapshot(reason = '')" in body
    assert "requestCoherentDashboardSnapshot('region_locks_refresh')" in body
    assert "function updateTopologyInflight(inflight, snapshotSeq = null)" in body
    assert "const safeInflight = snapshotSeq == null ? [] : (inflight || []);" in body
    assert "let _lastRegionLocksPayload = null;" in body
    assert "rich overlay failed" in body
    assert "fetchJSON('/dashboard/api/snapshot'" in body
    assert "timeoutMs: _SNAPSHOT_POLL_TIMEOUT_MS" in body
    assert "setInterval(updateSnapshotPoll, 2500)" in body
    assert "window_s=${TOPOLOGY_ACTIVITY_WINDOW_S}" in body
    assert "fetchJSON('/dashboard/api/contention')" in body
    assert "if (refreshSeq !== _regionLocksRefreshSeq) return;" in body


def test_dashboard_snapshot_to_region_lock_overlay_choreography() -> None:
    """Region-lock refreshes should only reuse in-flight tasks from the same snapshot frame."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    assert "let _latestSnapshotSeq = 0;" in body
    assert "const snapshotSeq = ++_latestSnapshotSeq;" in body
    assert "return updateRegionLocks(refreshSeq, snap.region_locks, snapshotSeq, snap.in_flight_tasks || []);" in body
    assert "function updateTopology(activity, inflight, snapshotSeq = null)" in body
    assert "updateTopologyInflight(inflight, snapshotSeq)" in body
    assert "const safeInflight = snapshotSeq == null ? [] : (inflight || []);" in body
    assert "const overlayInflight = snapshotSeq != null" in body
    assert "updateTopologyInflight(overlayInflight, snapshotSeq);" in body


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
    assert "const currentCodeHealthChip = _autopilotStatusChip(" in body
    assert "_currentCodeHealthLabel(currentCodeHealth)" in body
    assert "healthChips ? `<div style=\"margin-top:3px\">${healthChips}</div>` : ''" in body
    assert "current code ${health.status}" in body
    assert "const advice = health.restart_advice || {};" in body
    assert "restart wait for boundary" in body
    assert "restart ready" in body


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


def test_dashboard_folds_devices_into_regions_lock_and_tap_panels() -> None:
    """MI210/extern servers bypass the orchestrator pipeline; the Regions Lock
    grid and the live tap panel must still surface their occupancy instead of
    rendering an idle machine while a device is visibly working."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    body = html_path.read_text()

    # Panel renamed + device fold (operator request 2026-07-05).
    assert "regions lock" in body
    assert "cpu region locks" not in body
    assert "function gpuDeviceRegionRows()" in body
    assert body.count("rows.push(...gpuRows);") >= 2  # basic + rich renderers
    assert "device occupancy from /slots, not a CPU region lock" in body

    # Orphan (off-pipeline) inference cards in the live tap panel.
    assert "function orphanDeviceSlots()" in body
    assert "function orphanDeviceSlotCards()" in body
    assert "orphan inference" in body
    assert "no token tap — off-pipeline" in body
    assert "!requests.length && !lockOnlyHolders.length && !orphanCards.length" in body

    # A degraded contention matrix reads as an incident, not a status chip.
    assert "admission gate degraded" in body
    assert "contention-matrix-v6-quarter-refresh.md" in body
