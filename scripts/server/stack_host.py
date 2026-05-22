"""Host-prerequisite audit/apply helpers for orchestrator stack.

Canonical inference host state — sysctl values, transparent hugepage flags,
CPU governor — that every llama-server launch depends on. Sourced from
handoffs/active/cpu-kernel-env-flags-inventory.md §211 +
handoffs/active/model-registry-v5-deployment-draft.yaml host_prerequisites.

`check_host_prerequisites()` audits without side effects; `apply_host_prerequisites()`
optionally repairs via `sudo -n`. `orchestrator_stack.py` re-imports both so
existing call sites keep working.
"""

from __future__ import annotations

import subprocess


# Required sysctl values. /etc/sysctl.d/99-epyc-inference.conf gives the boot-time
# default; these are re-verified per session per feedback_numa_balancing_self_reset.
_HOST_PREREQ_SYSCTLS = {
    "kernel.numa_balancing": "0",
    "kernel.perf_event_paranoid": "1",
}

# Required THP state — both enabled and defrag must read "always".
_HOST_PREREQ_THP = {
    "/sys/kernel/mm/transparent_hugepage/enabled": "always",
    "/sys/kernel/mm/transparent_hugepage/defrag": "always",
}

# Required CPU governor.
_HOST_PREREQ_GOVERNOR = "performance"


def _read_sysctl(key: str) -> str | None:
    path = "/proc/sys/" + key.replace(".", "/")
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _read_thp_active(path: str) -> str | None:
    # /sys/kernel/mm/transparent_hugepage/enabled has format e.g. "always [madvise] never"
    # The bracketed token is the active value.
    try:
        with open(path) as f:
            content = f.read().strip()
    except OSError:
        return None
    for token in content.split():
        if token.startswith("[") and token.endswith("]"):
            return token[1:-1]
    return content


def _read_governor() -> str | None:
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except OSError:
        return None


def check_host_prerequisites() -> tuple[bool, list[str]]:
    """Audit canonical host state. Returns (all_pass, list_of_drift_messages)."""
    drift: list[str] = []

    for key, want in _HOST_PREREQ_SYSCTLS.items():
        got = _read_sysctl(key)
        if got != want:
            drift.append(f"sysctl {key}={got} (want {want})")

    for path, want in _HOST_PREREQ_THP.items():
        got = _read_thp_active(path)
        if got != want:
            drift.append(f"{path} active={got} (want {want})")

    gov = _read_governor()
    if gov != _HOST_PREREQ_GOVERNOR:
        drift.append(f"cpu0 scaling_governor={gov} (want {_HOST_PREREQ_GOVERNOR})")

    return (len(drift) == 0, drift)


def apply_host_prerequisites(auto_fix: bool = True) -> bool:
    """Verify and (optionally) apply canonical host settings.

    Returns True if host is canonical (or was successfully fixed). Returns False
    if any prereq could not be applied — caller should refuse to launch.
    """
    print("[host_prereq] Auditing canonical host state...")
    ok, drift = check_host_prerequisites()
    if ok:
        print("  [OK] All host prerequisites satisfied "
              "(numa_balancing=0, THP=always, governor=performance, perf_paranoid=1)")
        return True

    print(f"  [DRIFT] {len(drift)} setting(s) need correction:")
    for msg in drift:
        print(f"    - {msg}")

    if not auto_fix:
        print("  [SKIP] auto_fix disabled. Pass --apply-host-prereqs or fix manually.")
        return False

    print("  [FIX] Applying canonical settings (sudo -n)...")
    ok_fix = True
    for key, val in _HOST_PREREQ_SYSCTLS.items():
        if _read_sysctl(key) == val:
            continue
        try:
            subprocess.run(["sudo", "-n", "sysctl", "-w", f"{key}={val}"],
                           check=True, capture_output=True, text=True, timeout=5)
            print(f"    ✓ sysctl {key}={val}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"    ✗ FAILED to set sysctl {key}: {exc}")
            ok_fix = False

    for path, val in _HOST_PREREQ_THP.items():
        if _read_thp_active(path) == val:
            continue
        try:
            # tee with sudo: echo "always" | sudo tee /sys/kernel/...
            subprocess.run(
                ["sudo", "-n", "tee", path],
                input=val + "\n", check=True, capture_output=True, text=True, timeout=5,
            )
            print(f"    ✓ {path} = {val}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"    ✗ FAILED to set {path}: {exc}")
            ok_fix = False

    if _read_governor() != _HOST_PREREQ_GOVERNOR:
        try:
            subprocess.run(
                ["sudo", "-n", "cpupower", "frequency-set", "-g", _HOST_PREREQ_GOVERNOR],
                check=True, capture_output=True, text=True, timeout=10,
            )
            print(f"    ✓ cpu governor = {_HOST_PREREQ_GOVERNOR}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"    ✗ FAILED to set governor: {exc}")
            ok_fix = False

    # Re-audit
    ok, drift_after = check_host_prerequisites()
    if ok:
        print("  [OK] All host prerequisites now satisfied after fix")
        return True

    print(f"  [FAIL] {len(drift_after)} setting(s) STILL drifted after fix attempt:")
    for msg in drift_after:
        print(f"    - {msg}")
    return False
