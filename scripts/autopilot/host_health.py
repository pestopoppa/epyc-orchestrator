"""Autopilot host-health detection + remediation.

Catches host-level performance regressions (CPU throttle, page-cache
fragmentation, sustained-load slowdown per `feedback_host_throttle_check.md`)
and triggers the canonical fix (`sudo sync && echo 3 > /proc/sys/vm/drop_caches`)
BEFORE attributing the regression to whatever config the autopilot was testing.

Without this, multi-day autopilot runs misattribute host-throttle to whichever
config was in flight, contaminating the Pareto archive with false-negative
entries (per the 2026-05-09 incident: frontdoor measured at 7.48 t/s = 1/3 of
expected after 9 hours of mlocked sustained load).

Usage (CLI):
    python scripts/autopilot/host_health.py            # status only
    python scripts/autopilot/host_health.py --remediate # detect + fix if needed

Wire-in (autopilot loop):
    from scripts.autopilot.host_health import is_throttled, remediate, HostHealthState

    state = HostHealthState.snapshot()
    if state.is_throttled():
        if remediate():
            state = HostHealthState.snapshot()  # re-baseline
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("autopilot.host_health")

# Canonical drop_caches path — root-owned wrapper installed via the sudoers
# helper. See ../scripts/host_health_install.md for setup.
_CANONICAL_FLUSH_HELPER = "/usr/local/sbin/autopilot-flush-cache"

# Detection thresholds. Tuned conservatively — false-positive (run drop_caches
# unnecessarily) costs ~5 s of bench time; false-negative (let throttled host
# poison data) costs trial integrity.
_LOADAVG_FRACTION_OF_CORES_OK = 1.5  # loadavg/n_cores must stay below this when no
                                      # active inference is expected; >1.5× = throttle suspect
_CPU_FREQ_FRACTION_OF_BASE_OK = 0.80  # mean cur_freq must stay above 80% of base_freq;
                                      # below = thermal/power throttling
_MIN_PAGE_CACHE_AVAILABLE_MB = 4096   # arbitrary lower bound; if Cached drops below this
                                      # under sustained load, fragmentation is likely


@dataclass(frozen=True)
class HostHealthState:
    """Snapshot of host CPU + memory state."""

    loadavg_1min: float
    n_cores_online: int
    mean_cur_mhz: float | None  # None if not exposed by kernel
    base_mhz: float | None
    page_cache_mb: float
    mem_available_mb: float
    timestamp: float

    @classmethod
    def snapshot(cls) -> "HostHealthState":
        return cls(
            loadavg_1min=_read_loadavg_1min(),
            n_cores_online=_read_online_cores(),
            mean_cur_mhz=_read_mean_cur_mhz(),
            base_mhz=_read_base_mhz(),
            page_cache_mb=_read_meminfo_mb("Cached"),
            mem_available_mb=_read_meminfo_mb("MemAvailable"),
            timestamp=time.time(),
        )

    @property
    def loadavg_per_core(self) -> float:
        return self.loadavg_1min / max(1, self.n_cores_online)

    @property
    def cpu_freq_fraction(self) -> float | None:
        if self.mean_cur_mhz is None or self.base_mhz is None or self.base_mhz <= 0:
            return None
        return self.mean_cur_mhz / self.base_mhz

    def is_throttled(self) -> tuple[bool, list[str]]:
        """Return (throttled?, list of triggers). Caller may log the triggers."""
        triggers = []
        ff = self.cpu_freq_fraction
        if ff is not None and ff < _CPU_FREQ_FRACTION_OF_BASE_OK:
            triggers.append(
                f"cpu_freq={self.mean_cur_mhz:.0f} MHz < {_CPU_FREQ_FRACTION_OF_BASE_OK:.0%} "
                f"of base {self.base_mhz:.0f} MHz"
            )
        if self.loadavg_per_core > _LOADAVG_FRACTION_OF_CORES_OK:
            triggers.append(
                f"loadavg/cores={self.loadavg_per_core:.2f} > {_LOADAVG_FRACTION_OF_CORES_OK}"
            )
        if self.page_cache_mb < _MIN_PAGE_CACHE_AVAILABLE_MB:
            triggers.append(
                f"page_cache={self.page_cache_mb:.0f} MB < {_MIN_PAGE_CACHE_AVAILABLE_MB} MB"
            )
        return (bool(triggers), triggers)


def _read_loadavg_1min() -> float:
    try:
        with open("/proc/loadavg") as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def _read_online_cores() -> int:
    try:
        with open("/sys/devices/system/cpu/online") as f:
            ranges = f.read().strip()
        n = 0
        for part in ranges.split(","):
            if "-" in part:
                a, b = part.split("-")
                n += int(b) - int(a) + 1
            elif part:
                n += 1
        return n or os.cpu_count() or 1
    except (OSError, ValueError):
        return os.cpu_count() or 1


def _read_meminfo_mb(field: str) -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(f"{field}:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _read_mean_cur_mhz() -> float | None:
    """Median of per-core scaling_cur_freq, in MHz.

    Median (not mean) so a few idle cores at min-freq don't drag the value
    below the throttle threshold on an otherwise-busy system. Sampled
    across all online CPUs (HT siblings included — they ramp identically
    on AMD).
    """
    base = Path("/sys/devices/system/cpu")
    vals = []
    for cpu_dir in sorted(base.glob("cpu[0-9]*")):
        f = cpu_dir / "cpufreq" / "scaling_cur_freq"
        if f.exists():
            try:
                vals.append(int(f.read_text().strip()) / 1000.0)
            except (OSError, ValueError):
                continue
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def _read_base_mhz() -> float | None:
    """All-core sustained-load frequency baseline.

    Throttle detection compares mean current freq against this. AMD pstate
    exposes `amd_pstate_lowest_nonlinear_freq` — the lowest "efficient
    performance" point; sustained workloads should stay above this. Intel
    typically exposes `base_frequency` directly. We try in order and fall
    back to None if neither is available (in which case the freq check is
    skipped to avoid false positives).

    NB: do NOT use `cpuinfo_max_freq` as the baseline — that's the absolute
    single-core boost peak; sustained all-core clocks are normally well
    below it, producing constant false-positives on healthy systems.
    """
    base = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    for fname in ("base_frequency", "cpuinfo_base_freq", "amd_pstate_lowest_nonlinear_freq"):
        f = base / fname
        if f.exists():
            try:
                v = f.read_text().strip()
                if v:
                    return int(v) / 1000.0
            except (OSError, ValueError):
                continue
    return None


def remediate() -> bool:
    """Run the canonical sync + drop_caches via passwordless sudo helper.

    Returns True on success, False if the helper is missing or sudo failed.
    """
    if not shutil.which("sudo"):
        log.warning("sudo not found; cannot remediate")
        return False
    if not Path(_CANONICAL_FLUSH_HELPER).exists():
        log.warning("flush helper not installed at %s — see "
                    "scripts/autopilot/host_health_install.md", _CANONICAL_FLUSH_HELPER)
        return False
    try:
        result = subprocess.run(
            ["sudo", "-n", _CANONICAL_FLUSH_HELPER],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            log.info("drop_caches OK (helper output: %s)", result.stdout.strip()[:120])
            return True
        log.error("drop_caches helper failed (rc=%d): %s",
                  result.returncode, result.stderr.strip()[:200])
        return False
    except subprocess.TimeoutExpired:
        log.error("drop_caches helper timed out after 30 s")
        return False
    except OSError as exc:
        log.error("drop_caches helper invocation error: %s", exc)
        return False


def is_throttled() -> tuple[bool, list[str]]:
    """One-shot check. Convenience wrapper around HostHealthState.snapshot."""
    return HostHealthState.snapshot().is_throttled()


# ---------------------------------------------------------------------------
# Pause-around-flush wrapper (2026-05-24)
# ---------------------------------------------------------------------------
#
# The bare remediate() above only runs `sync && drop_caches` — it does NOT pause
# the autopilot trial loop nor warm the GGUFs back in NUMA-interleaved. Either
# of those omissions can corrupt a trial that lands during the flush window:
# - Trial in flight when sync runs: completes with degraded throughput (cold cache)
# - Trial that starts immediately after: hits cold cache, throughput tanks ~50%
# - Non-NUMA-aware re-warm (naïve cat) pins all pages to ONE NUMA node, halving
#   sustained t/s per `feedback_drop_caches_numa_eviction`.
#
# flush_cache_with_pause() handles all three: set state["paused"]=True (relies on
# the 2026-05-24 autopilot loop fix that reloads state at the top of every
# iteration), run flush, NUMA-interleave-rewarm all active role GGUFs serially
# (parallel rewarm would defeat the interleave benefit), restore paused state.
# Any trial that DID complete during the window gets tagged with
# DeficiencyCategory.EXOGENOUS_CACHE_FLUSH by the safety_gate wire-in.

# Active role GGUFs to rewarm post-flush. Source-of-truth would be NUMA_CONFIG +
# the model registry; this list is the conservative fallback used when the
# launcher modules aren't importable (e.g., when running host_health standalone).
# Models too small to need NUMA-interleave rewarm (embedders, drafters) are
# omitted — bare `cat` on a 4 GB file is fine.
_DEFAULT_REWARM_GGUFS = (
    "/mnt/raid0/llm/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",                              # frontdoor / coder_escalation / worker_summarize
    "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-Q4_K_M.gguf",                              # worker_general
    "/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf",  # architect_general (multipart; cat pulls all 3 via mmap follow)
    "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf",  # ingest_long_context
    "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf",      # vision_escalation
    "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",            # worker_vision
)


def _numa_interleave_rewarm(gguf_paths: tuple[str, ...] = _DEFAULT_REWARM_GGUFS,
                            timeout_per_gguf: int = 120) -> dict[str, bool]:
    """Warm each GGUF back into the page cache with `numactl --interleave=all`.

    Serial, not parallel — running multiple `numactl --interleave=all cat`
    concurrently would interleave on a per-process basis but the kernel page
    cache is shared, so the second process re-reads pages the first already
    placed and the interleave property degrades. Serial keeps the
    one-process-one-walk invariant.

    Returns {gguf_path: success_bool} for the operator to see what got warmed.
    """
    if not shutil.which("numactl"):
        log.warning("numactl not found; cannot do NUMA-interleaved rewarm")
        return {p: False for p in gguf_paths}

    results: dict[str, bool] = {}
    for gguf in gguf_paths:
        if not Path(gguf).exists():
            log.info("skip rewarm (missing): %s", gguf)
            results[gguf] = False
            continue
        try:
            t0 = time.monotonic()
            # numactl --interleave=all cat <gguf> > /dev/null
            proc = subprocess.run(
                ["numactl", "--interleave=all", "cat", gguf],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout_per_gguf,
            )
            elapsed = time.monotonic() - t0
            ok = (proc.returncode == 0)
            size_gb = Path(gguf).stat().st_size / (1024 ** 3)
            log.info("rewarm %s: %.1f GB in %.1fs (%.1f GB/s) %s",
                     Path(gguf).name, size_gb, elapsed,
                     size_gb / max(elapsed, 0.001), "OK" if ok else "FAIL")
            results[gguf] = ok
        except subprocess.TimeoutExpired:
            log.error("rewarm timed out after %ds: %s", timeout_per_gguf, gguf)
            results[gguf] = False
        except OSError as exc:
            log.error("rewarm failed for %s: %s", gguf, exc)
            results[gguf] = False
    return results


def flush_cache_with_pause(
    *,
    state_path: Path | None = None,
    rewarm: bool = True,
    rewarm_paths: tuple[str, ...] = _DEFAULT_REWARM_GGUFS,
) -> dict[str, object]:
    """The robust pause+flush+rewarm+resume sequence.

    Used by both the autopilot safety_gate path (when it detects throttle) and
    the operator-facing `flush_cache_safely.py` wrapper. Returns a dict with
    `paused_pre`, `flush_ok`, `rewarm` (dict per-gguf), `elapsed_s` so callers
    can log + journal the outcome.

    Pause mechanic depends on the 2026-05-24 loop fix that reloads state at
    the top of every iteration. Without that fix, an externally-set paused=True
    is clobbered by save_state at trial end and the trial loop never honors it.

    `state_path` defaults to the autopilot state file the same way the rest of
    the autopilot module locates it (env override `AUTOPILOT_STATE`, then
    default under orchestration/).
    """
    import json
    if state_path is None:
        # Resolve via the same path the autopilot CLI uses, without importing
        # the heavy autopilot module here.
        env_override = os.environ.get("AUTOPILOT_STATE")
        if env_override:
            state_path = Path(env_override)
        else:
            # Default: <repo>/orchestration/autopilot_state.json. host_health
            # lives at scripts/autopilot/, so go two parents up + orchestration.
            state_path = Path(__file__).resolve().parents[2] / "orchestration" / "autopilot_state.json"

    started = time.monotonic()
    result: dict[str, object] = {"paused_pre": None, "flush_ok": False, "rewarm": {}, "elapsed_s": 0.0}

    # Step 1: set paused=True via atomic write (mirror AP-39 atomic state pattern).
    paused_pre = None
    try:
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            paused_pre = state.get("paused", False)
            state["paused"] = True
            tmp = state_path.with_suffix(state_path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, state_path)
            log.info("autopilot paused via state.json (pre=%s)", paused_pre)
        else:
            log.warning("state file %s does not exist; flush will proceed without pause", state_path)
    except Exception as exc:
        log.error("could not set paused=True on %s: %s", state_path, exc)
    result["paused_pre"] = paused_pre

    # Step 2: brief grace window for the trial loop to notice the new paused state
    # (loop reloads at top of every iteration, so 11s covers the 10s sleep inside
    # the paused branch plus jitter).
    time.sleep(11)

    # Step 3: run the canonical flush.
    flush_ok = remediate()
    result["flush_ok"] = flush_ok

    # Step 4: NUMA-interleave-rewarm.
    if rewarm and flush_ok:
        result["rewarm"] = _numa_interleave_rewarm(rewarm_paths)

    # Step 5: restore previous paused state (if there was one).
    try:
        with open(state_path) as f:
            state = json.load(f)
        # Only restore if we set it ourselves AND nothing else changed it
        # to a stricter value (operator may have run `autopilot.py pause` mid-flush).
        if state.get("paused") is True and paused_pre is False:
            state["paused"] = False
            tmp = state_path.with_suffix(state_path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, state_path)
            log.info("autopilot resume (paused=False)")
    except Exception as exc:
        log.error("could not restore paused state: %s", exc)

    result["elapsed_s"] = time.monotonic() - started
    log.info("flush_cache_with_pause done in %.1fs (flush=%s, %d/%d gguf rewarmed)",
             result["elapsed_s"], flush_ok,
             sum(1 for v in result["rewarm"].values() if v),
             len(result["rewarm"]))
    return result


# --- CLI -------------------------------------------------------------------

def _format_state(state: HostHealthState) -> str:
    ff = state.cpu_freq_fraction
    ff_pct = f"{ff:.0%}" if ff is not None else "N/A"
    return (
        f"loadavg(1m)={state.loadavg_1min:.2f}  "
        f"loadavg/cores={state.loadavg_per_core:.2f}  "
        f"cpu_freq={state.mean_cur_mhz:.0f}/{state.base_mhz:.0f} MHz ({ff_pct})  "
        f"page_cache={state.page_cache_mb:.0f} MB  "
        f"mem_avail={state.mem_available_mb:.0f} MB"
        if state.mean_cur_mhz and state.base_mhz else
        f"loadavg(1m)={state.loadavg_1min:.2f}  "
        f"loadavg/cores={state.loadavg_per_core:.2f}  "
        f"cpu_freq=N/A  "
        f"page_cache={state.page_cache_mb:.0f} MB  "
        f"mem_avail={state.mem_available_mb:.0f} MB"
    )


def _main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Host health check + drop_caches remediation")
    p.add_argument("--remediate", action="store_true",
                   help="run drop_caches if throttle detected")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    state = HostHealthState.snapshot()
    print(_format_state(state))
    throttled, triggers = state.is_throttled()
    if throttled:
        print(f"\nTHROTTLED: {len(triggers)} trigger(s):")
        for t in triggers:
            print(f"  - {t}")
        if args.remediate:
            print("\nRemediating (sync + drop_caches)…")
            ok = remediate()
            if ok:
                state2 = HostHealthState.snapshot()
                print(_format_state(state2))
                print("OK" if not state2.is_throttled()[0] else "STILL THROTTLED after remediation")
                return 0 if not state2.is_throttled()[0] else 3
            return 2
        return 1
    print("OK — no throttle detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
