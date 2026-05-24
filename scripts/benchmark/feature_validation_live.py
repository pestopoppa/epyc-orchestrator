"""LiveValidator — orchestrator-driven feature validation against running stack.

Extracted from scripts/benchmark/feature_validation.py during the 2026-05-22
Task-D refactor. Also hosts the LiveValidator-specific helpers
(_read_meminfo_mb, _hot_reload_feature, _ensure_stack_running, etc.) since
they're only used here.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from feature_validation_profiles import (
    ComparisonReport,
    FeatureProfile,
    MetricSnapshot,
    TestSpec,
    _build_profiles,
)

logger = logging.getLogger("feature_validation")

import os
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "runs" / "feature_validation"
MANIFESTS_DIR = PROJECT_ROOT / "benchmarks" / "prompts" / "v1" / "feature_validation"
API_URL = os.environ.get("ORCHESTRATOR_API_URL", "http://localhost:8000")



def _read_meminfo_mb() -> float:
    """Read current RSS from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0




def _hot_reload_feature(feature: str, enabled: bool) -> bool:
    """Toggle a feature via POST /config and verify the change took effect."""
    try:
        import httpx
        resp = httpx.post(
            f"{API_URL}/config",
            json={feature: enabled},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.error("Hot-reload %s=%s returned %d", feature, enabled, resp.status_code)
            return False
        # Verify the feature state matches what we requested
        body = resp.json()
        actual = body.get("features", {}).get(feature)
        if actual is not None and actual != enabled:
            logger.error("Hot-reload %s=%s: server reports %s (mismatch!)", feature, enabled, actual)
            return False
        return True
    except Exception as e:
        logger.error("Failed to hot-reload %s=%s: %s", feature, enabled, e)
        return False




def _ensure_stack_running() -> bool:
    """Check if the orchestrator stack is running; attempt auto-start if not.

    Returns True if the stack is reachable after this call.
    """
    try:
        import httpx
        resp = httpx.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    logger.warning("Stack not reachable at %s — attempting auto-start", API_URL)
    stack_script = PROJECT_ROOT / "scripts" / "server" / "orchestrator_stack.py"
    if not stack_script.exists():
        logger.error("Stack script not found: %s", stack_script)
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(stack_script), "start", "--hot-only"],
            capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            logger.error("Stack start failed: %s", result.stderr[-500:])
            return False
        # Wait for health
        import httpx
        for _ in range(30):
            time.sleep(2)
            try:
                resp = httpx.get(f"{API_URL}/health", timeout=5)
                if resp.status_code == 200:
                    logger.info("Stack started successfully")
                    return True
            except Exception:
                continue
        logger.error("Stack started but health check never passed")
        return False
    except Exception as e:
        logger.error("Failed to start stack: %s", e)
        return False




def _verify_health_mid_run(client: "Any") -> bool:
    """Quick health check between prompts. Returns False if stack is down."""
    try:
        resp = client.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False




def _load_prompt_manifest(manifest_name: str) -> list[dict[str, Any]]:
    """Load a prompt manifest JSON from the feature_validation directory."""
    path = MANIFESTS_DIR / manifest_name
    if not path.exists():
        logger.warning("Manifest not found: %s", path)
        return []
    with open(path) as f:
        return json.load(f)




def _write_incremental(path: Path, data: dict[str, Any]) -> None:
    """Append one JSON line to incremental results file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data, default=str) + "\n")




def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Offline validation (mock mode + replay) ──────────────────────────




class LiveValidator:
    """Run live A/B tests against the running orchestrator stack."""

    def __init__(self) -> None:
        try:
            import httpx
            self._client = httpx.Client(timeout=120)
        except ImportError:
            self._client = None

    def _check_stack(self) -> bool:
        """Verify orchestrator API is reachable; auto-start if not."""
        if not self._client:
            return False
        return _ensure_stack_running()

    def _run_prompts(self, prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Send prompts to /chat and collect responses.

        Includes mid-run health checks: if the stack goes down between prompts,
        remaining prompts are skipped with an error marker.

        When self.watcher is set (autopilot-managed feature validation), each
        POST uses resilient_post for exogenous-reload detection + retry;
        watcher-aware metadata is captured in each result's `_meta` key so
        downstream consumers (autopilot's feature-validation gate) can tell
        which prompt results were contaminated by operator-initiated reloads.
        """
        # Lazy import: pulls resilient_post + watcher only when needed.
        watcher = getattr(self, "watcher", None)
        resilient = None
        if watcher is not None:
            try:
                import sys
                from pathlib import Path
                _ap = Path(__file__).resolve().parents[1] / "autopilot"
                if str(_ap) not in sys.path:
                    sys.path.insert(0, str(_ap))
                from resilient_http import resilient_post as resilient  # type: ignore
            except Exception as exc:
                logger.warning("feature_validation_live: resilient_post unavailable (%s); falling back to legacy", exc)
                resilient = None

        results = []
        for i, p in enumerate(prompts):
            # Mid-run health check every 3 prompts (avoid overhead on every call)
            if i > 0 and i % 3 == 0:
                if not _verify_health_mid_run(self._client):
                    logger.error("Stack went down after prompt %d/%d — aborting run",
                                 i, len(prompts))
                    for remaining in prompts[i:]:
                        results.append({
                            "prompt_id": remaining.get("id", ""),
                            "error": "stack_unreachable_mid_run",
                        })
                    break
            try:
                payload = {"prompt": p.get("prompt", p.get("message", "")),
                           "role": p.get("role", "frontdoor"),
                           "mode": p.get("mode", "direct"),
                           "mock_mode": False,
                           "real_mode": True}
                start = time.monotonic()
                meta: dict[str, Any] = {}
                if resilient is not None:
                    data, meta = resilient(
                        f"{API_URL}/chat",
                        json=payload,
                        timeout=120.0,
                        client=self._client,
                        watcher=watcher,
                        llama_role=p.get("role", "frontdoor"),
                    )
                    elapsed = time.monotonic() - start
                    status = 200 if not data.get("error") else 0
                else:
                    resp = self._client.post(f"{API_URL}/chat", json=payload)
                    elapsed = time.monotonic() - start
                    data = resp.json() if resp.status_code == 200 else {}
                    status = resp.status_code
                results.append({
                    "prompt_id": p.get("id", ""),
                    "elapsed_seconds": elapsed,
                    "predicted_tps": data.get("predicted_tps", 0),
                    "answer": data.get("answer", ""),
                    "status": status,
                    "raw": data,
                    "_meta": meta,
                })
            except Exception as e:
                results.append({"prompt_id": p.get("id", ""), "error": str(e)})
        return results

    def capture_baseline(self, prompts: list[dict[str, Any]]) -> MetricSnapshot:
        """Capture baseline metrics with current production flags."""
        snap = MetricSnapshot(feature="baseline", enabled=False, timestamp=_now_iso())
        mem_before = _read_meminfo_mb()
        results = self._run_prompts(prompts)
        snap.memory_delta_mb = _read_meminfo_mb() - mem_before
        snap.prompts_run = len(results)

        elapsed_list = [r["elapsed_seconds"] for r in results if "elapsed_seconds" in r]
        if elapsed_list:
            elapsed_list.sort()
            snap.latency_p50_s = elapsed_list[len(elapsed_list) // 2]
            snap.latency_p95_s = elapsed_list[int(len(elapsed_list) * 0.95)]

        tps_list = [r["predicted_tps"] for r in results if r.get("predicted_tps", 0) > 0]
        if tps_list:
            snap.predicted_tps = sum(tps_list) / len(tps_list)

        snap.raw = {"results": results}
        return snap

    def validate_feature(self, profile: FeatureProfile,
                         baseline: MetricSnapshot) -> ComparisonReport:
        """Run live A/B test for a single feature."""
        report = ComparisonReport(feature=profile.name, baseline=baseline)

        if not self._check_stack():
            report.verdict = "SKIP_NO_STACK"
            return report

        # Enable feature + deps
        all_flags = {d: True for d in profile.deps}
        all_flags[profile.name] = True
        for flag, val in all_flags.items():
            if not _hot_reload_feature(flag, val):
                report.verdict = "SKIP_RELOAD_FAIL"
                return report

        time.sleep(0.5)  # settle

        # Load prompts
        prompts = []
        for test in profile.live_tests:
            if test.prompt_manifest:
                prompts.extend(_load_prompt_manifest(test.prompt_manifest))
        if not prompts:
            # Fallback: use a small general set
            prompts = _load_prompt_manifest("general_5.json")

        # Candidate run
        candidate = self.capture_baseline(prompts)
        candidate.feature = profile.name
        candidate.enabled = True
        report.candidate = candidate

        # Revert and verify baseline restored
        revert_ok = True
        for flag in all_flags:
            if not _hot_reload_feature(flag, False):
                logger.error("REVERT FAILED for %s — baseline may be contaminated", flag)
                revert_ok = False
        if not revert_ok:
            report.verdict = "REVERT_FAILED"
            return report

        # Compute deltas
        if baseline and candidate:
            report.quality_delta = candidate.quality_score - baseline.quality_score
            report.latency_delta_s = candidate.latency_p50_s - baseline.latency_p50_s
            report.tps_delta = candidate.predicted_tps - baseline.predicted_tps
            report.memory_delta_mb = candidate.memory_delta_mb - baseline.memory_delta_mb

        # Verdict
        report.verdict = self._judge_verdict(profile, report)

        # Write incremental — include raw per-prompt results for quality scoring
        baseline_responses = self._summarize_responses(baseline.raw.get("results", []))
        candidate_responses = self._summarize_responses(candidate.raw.get("results", []))
        _write_incremental(
            RESULTS_DIR / "live" / f"{profile.name}.jsonl",
            {
                "feature": profile.name,
                "verdict": report.verdict,
                "quality_delta": report.quality_delta,
                "latency_delta_s": report.latency_delta_s,
                "tps_delta": report.tps_delta,
                "memory_delta_mb": report.memory_delta_mb,
                "timestamp": _now_iso(),
                "baseline": {
                    "p50_s": baseline.latency_p50_s,
                    "avg_tps": baseline.predicted_tps,
                    "prompts_run": baseline.prompts_run,
                    "responses": baseline_responses,
                },
                "candidate": {
                    "p50_s": candidate.latency_p50_s,
                    "avg_tps": candidate.predicted_tps,
                    "prompts_run": candidate.prompts_run,
                    "responses": candidate_responses,
                },
            },
        )
        return report

    @staticmethod
    def _summarize_responses(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract key fields from raw prompt results for persistence."""
        summaries = []
        for r in results:
            raw = r.get("raw", {})
            elapsed = r.get("elapsed_seconds", 0)
            tokens = raw.get("tokens_generated", 0)
            summary = {
                "prompt_id": r.get("prompt_id", ""),
                "status": r.get("status", 0),
                "elapsed_s": round(elapsed, 2),
                "tokens_generated": tokens,
                "client_tps": round(tokens / elapsed, 1) if elapsed > 0 and tokens > 0 else 0,
                "routed_to": raw.get("routed_to", ""),
                "turns": raw.get("turns", 0),
                "role_history": raw.get("role_history", []),
                "answer": r.get("answer", "")[:500],  # truncate for storage
            }
            if r.get("error"):
                summary["error"] = r["error"]
            summaries.append(summary)
        return summaries

    @staticmethod
    def _judge_verdict(profile: FeatureProfile, report: ComparisonReport) -> str:
        """Determine PASS/FAIL/BORDERLINE from deltas and pass criteria."""
        if not report.candidate:
            return "NO_DATA"
        # Quality must not regress
        if report.quality_delta < -0.05:
            return "FAIL_QUALITY"
        # Latency must not increase by more than 2s for tier 1, 5s otherwise
        max_latency = 2.0 if profile.tier <= 1 else 5.0
        if report.latency_delta_s > max_latency:
            return "FAIL_LATENCY"
        # Borderline: small regression
        if report.quality_delta < 0 or report.latency_delta_s > 1.0:
            return "BORDERLINE"
        return "PASS"


# ── Report generation ────────────────────────────────────────────────


