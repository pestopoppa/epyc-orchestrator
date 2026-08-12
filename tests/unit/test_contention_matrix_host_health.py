"""The contention matrix must carry its own host-health provenance.

ORIGIN. Found 2026-08-12 while dry-running the OP-21 overlapping contention
re-bench BEFORE spending a post-reboot window on it.

`_host_metadata()` collected `uptime` and `kernel`; `_emit_yaml` wrote only
`host: <hostname>` and discarded both. The emitted matrix had no
`host_health_warnings` and no `decision_grade`, and stamped
`verdict: allow/block` with `samples: 1` regardless of host state.

CONSEQUENCE, and the reason this is a test and not a comment: a run at 14.1 d
uptime landed BYTE-INDISTINGUISHABLE from a clean-host measurement. Nobody
reading the artifact later could tell which one they were holding — so a
reboot window bought a number that could not prove the host was clean, which
is the only thing the reboot was for.

THE FAILURE MODE THIS FILE GUARDS is not "the field is wrong". It is "the
field is ABSENT, and absence renders as a pass". Every test below that
exercises an unknown asserts on `decision_grade is False` AND on the presence
of a warning that names what could not be established, because a silent
`decision_grade: true` on an unprobed host is exactly the vacuity class this
repo tracks.

NO INFERENCE, NO SERVER, NO BENCH. Every test here calls the emitter and the
probe directly.
"""
from __future__ import annotations

import types

import pytest
import yaml

import scripts.server.contention_matrix as cm


# ── Rule-set access ──────────────────────────────────────────────────
#
# The tests below feed SYNTHETIC attestations to the REAL rule set
# (epyc-inference-research host_health_warnings). Re-implementing the
# thresholds in the test would let the module and the test drift together into
# agreement about the wrong answer.


def _real_rules():
    try:
        return cm._load_host_health_rules()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"research-repo host-health rule set unavailable: {exc}")


def _attestation(**overrides):
    """A synthetic attestation shaped like server_np_sweep.collect_attestation()."""
    np_sweep, _ = _real_rules()
    fresh = np_sweep.MAX_DECISION_GRADE_UPTIME_SECONDS / 2
    base = {
        "host": "TestHost",
        "kernel": "Linux TestHost 6.14.0-37-generic",
        "uptime_seconds": fresh,
        "numa_balancing": np_sweep.REQUIRED_NUMA_BALANCING,
        "scaling_governors": ["performance"],
        "loadavg": "0.10 0.20 0.30 1/100 1234",
        "existing_llama_processes": [],
    }
    base.update(overrides)
    return base


def _rules_loader(attestation, freq_warnings=()):
    """A loader returning the REAL warning rules over a synthetic attestation."""
    np_sweep, _ = _real_rules()
    np_mod = types.SimpleNamespace(
        collect_attestation=lambda: dict(attestation),
        host_health_warnings=np_sweep.host_health_warnings,
    )
    numa_mod = types.SimpleNamespace(
        cpu_freq_static_warnings=lambda: list(freq_warnings),
    )
    return lambda: (np_mod, numa_mod)


def _emit(host_health, **kwargs):
    return cm._emit_yaml(
        [],
        topology_hash="deadbeef",
        binary={"git_commit": "abc1234"},
        host="TestHost",
        host_health=host_health,
        **kwargs,
    )


# ── 1. The original defect: collected metadata was discarded ─────────


def test_emitted_matrix_carries_kernel_and_uptime() -> None:
    """`_host_metadata()` collected these; the emitter used to throw them away."""
    probe = cm._host_health_probe(
        host_meta={"hostname": "TestHost", "kernel": "6.14.0-testkernel", "uptime": "up 14 days"},
        load_rules=_rules_loader(_attestation()),
    )
    doc = yaml.safe_load(_emit(probe))
    assert doc["host_kernel"] == "6.14.0-testkernel"
    assert doc["host_uptime"] == "up 14 days"


def test_emitted_matrix_carries_health_verdict_fields() -> None:
    probe = cm._host_health_probe(
        host_meta={"hostname": "TestHost", "kernel": "k", "uptime": "u"},
        load_rules=_rules_loader(_attestation()),
    )
    doc = yaml.safe_load(_emit(probe))
    for key in (
        "host_health_status",
        "host_health_warnings",
        "host_health_structural_for_harness",
        "decision_grade",
        "decision_grade_blockers",
        "host_provenance",
    ):
        assert key in doc, f"{key} missing from emitted matrix"
    assert isinstance(doc["decision_grade"], bool)
    assert isinstance(doc["host_health_warnings"], list)


# ── 2. COMPLIANT-PATH GUARD ──────────────────────────────────────────
#
# A gate that can never be satisfied is not a gate. These two prove the clean
# path is REACHABLE — including with the stack up, which is the only way a
# contention matrix is ever measured.


def test_a_clean_host_is_decision_grade() -> None:
    probe = cm._host_health_probe(load_rules=_rules_loader(_attestation()))
    assert probe["warnings"] == [], probe["warnings"]
    assert probe["status"] == cm.HOST_HEALTH_CLEAN
    assert probe["decision_grade"] is True
    assert probe["decision_grade_blockers"] == []
    doc = yaml.safe_load(_emit(probe))
    assert doc["decision_grade"] is True
    assert doc["host_health_status"] == "clean"
    assert doc["host_health_warnings"] == []


def test_a_clean_host_with_the_live_stack_up_is_still_decision_grade() -> None:
    """The matrix benches live servers BY DESIGN — that is the instrument.

    If llama-server presence gated, `decision_grade` could never be true for
    this harness and the field would be decorative.
    """
    probe = cm._host_health_probe(
        load_rules=_rules_loader(
            _attestation(existing_llama_processes=[{"pid": "1", "cmd": "llama-server"}])
        )
    )
    assert probe["status"] == cm.HOST_HEALTH_CLEAN
    assert probe["decision_grade"] is True
    # ...but the presence is RECORDED, never hidden.
    assert probe["warnings"], "live llama processes must still appear in the record"
    assert probe["structural_for_harness"] == probe["warnings"]
    assert probe["llama_processes_at_attestation"] == 1
    doc = yaml.safe_load(_emit(probe))
    assert doc["host_health_structural_for_harness"] == probe["warnings"]


# ── 3. Real host state actually gates ────────────────────────────────


def test_stale_uptime_blocks_decision_grade() -> None:
    """The condition that made OP-21 worth a reboot in the first place."""
    np_sweep, _ = _real_rules()
    stale = np_sweep.MAX_DECISION_GRADE_UPTIME_SECONDS + 1
    probe = cm._host_health_probe(load_rules=_rules_loader(_attestation(uptime_seconds=stale)))
    assert probe["status"] == cm.HOST_HEALTH_WARN
    assert probe["decision_grade"] is False
    assert any("uptime" in w for w in probe["decision_grade_blockers"])
    doc = yaml.safe_load(_emit(probe))
    assert doc["decision_grade"] is False
    assert doc["decision_grade_blockers"]


def test_numa_balancing_on_blocks_decision_grade() -> None:
    probe = cm._host_health_probe(load_rules=_rules_loader(_attestation(numa_balancing="1")))
    assert probe["decision_grade"] is False
    assert any("numa_balancing" in w for w in probe["decision_grade_blockers"])


def test_cpu_throttle_warning_blocks_decision_grade() -> None:
    probe = cm._host_health_probe(
        load_rules=_rules_loader(_attestation(), freq_warnings=["cpufreq global boost flag is 0"])
    )
    assert probe["decision_grade"] is False
    assert "cpufreq global boost flag is 0" in probe["decision_grade_blockers"]
    # A throttle is NOT structural for this harness — it must not be waived.
    assert probe["structural_for_harness"] == []


# ── 4. FAIL-SAFE: an unknown must never render as a pass ─────────────


def test_probe_failure_renders_unknown_not_clean() -> None:
    def _boom():
        raise ImportError("no research repo on this host")

    probe = cm._host_health_probe(
        host_meta={"hostname": "TestHost", "kernel": "k", "uptime": "u"},
        load_rules=_boom,
    )
    assert probe["status"] == cm.HOST_HEALTH_UNKNOWN
    assert probe["decision_grade"] is False
    assert probe["decision_grade_blockers"], "an unknown must name what it could not establish"
    assert "no research repo on this host" in probe["attestation_error"]
    doc = yaml.safe_load(_emit(probe))
    assert doc["host_health_status"] == "unknown"
    assert doc["decision_grade"] is False
    assert any("could not be determined" in w for w in doc["host_health_warnings"])


def test_unreadable_uptime_renders_unknown_not_clean() -> None:
    """The sharpest case: the rule SILENTLY does not fire on a missing input.

    `host_health_warnings` guards its uptime branch with an isinstance check,
    so `uptime_seconds: None` produces NO warning. Without an explicit refusal
    here, an unreadable /proc/uptime would have rendered as a clean host.
    """
    probe = cm._host_health_probe(load_rules=_rules_loader(_attestation(uptime_seconds=None)))
    assert probe["status"] == cm.HOST_HEALTH_UNKNOWN
    assert probe["decision_grade"] is False
    assert any("uptime could not be read" in w for w in probe["decision_grade_blockers"])
    assert probe["attestation_status"] == "incomplete"


def test_emitter_without_a_probe_renders_unknown_not_clean() -> None:
    """A caller that forgets to probe must not silently produce a clean-looking file."""
    doc = yaml.safe_load(_emit(None))
    assert doc["host_health_status"] == "unknown"
    assert doc["decision_grade"] is False
    assert doc["host_health_warnings"], "the omission must be stated in the artifact"
    assert doc["decision_grade_blockers"]


def test_probe_never_raises_and_is_always_fully_populated() -> None:
    """Against the live host, with no mocks: the record is total."""
    probe = cm._host_health_probe()
    for key in (
        "status",
        "warnings",
        "structural_for_harness",
        "decision_grade",
        "decision_grade_blockers",
        "attestation_status",
        "kernel",
        "uptime",
    ):
        assert key in probe, f"{key} missing from probe record"
    assert probe["status"] in {
        cm.HOST_HEALTH_CLEAN,
        cm.HOST_HEALTH_WARN,
        cm.HOST_HEALTH_UNKNOWN,
    }
    assert isinstance(probe["decision_grade"], bool)
    if probe["status"] != cm.HOST_HEALTH_CLEAN:
        assert probe["decision_grade"] is False


# ── 5. Overrides DEMOTE, they never rescue ───────────────────────────


def test_a_role_restricted_run_is_demoted_even_on_a_clean_host() -> None:
    probe = cm._host_health_probe(
        extra_blockers=["run was role-restricted via --roles (frontdoor)"],
        load_rules=_rules_loader(_attestation()),
    )
    assert probe["decision_grade"] is False
    assert any("--roles" in b for b in probe["decision_grade_blockers"])


# ── 6. A stale "clean" must never be carried forward ─────────────────


def test_host_health_keys_are_emitter_owned() -> None:
    """Carry-forward preserves hand-authored policy; provenance is NOT policy.

    If any of these keys were absent from `_EMITTER_OWNED_SECTIONS`, a re-bench
    would preserve the PREVIOUS run's health block verbatim while emitting a
    fresh one — a duplicate key whose first (stale) value is the one PyYAML
    keeps is the same defect this file exists to prevent.
    """
    for key in (
        "host_kernel",
        "host_uptime",
        "host_health_status",
        "host_health_warnings",
        "host_health_structural_for_harness",
        "decision_grade",
        "decision_grade_blockers",
        "host_provenance",
    ):
        assert key in cm._EMITTER_OWNED_SECTIONS, f"{key} would be carried forward stale"


def test_a_stale_clean_block_is_regenerated_not_preserved(tmp_path) -> None:
    clean = cm._host_health_probe(load_rules=_rules_loader(_attestation()))
    first = _emit(clean) + '\nnway_light_roles: ["frontdoor"]\n'
    path = tmp_path / "contention_matrix.yaml"
    path.write_text(first)
    assert yaml.safe_load(first)["decision_grade"] is True

    np_sweep, _ = _real_rules()
    dirty = cm._host_health_probe(
        load_rules=_rules_loader(
            _attestation(uptime_seconds=np_sweep.MAX_DECISION_GRADE_UPTIME_SECONDS + 1)
        )
    )
    preserved = cm._carry_forward_sections(path)
    assert [k for k, _ in preserved] == ["nway_light_roles"]

    second = _emit(dirty, preserve_sections=preserved)
    top_level = [
        line.split(":", 1)[0]
        for line in second.splitlines()
        if line and not line[0].isspace() and not line.startswith("#") and ":" in line
    ]
    assert len(top_level) == len(set(top_level)), f"duplicate top-level keys: {top_level}"
    doc = yaml.safe_load(second)
    assert doc["decision_grade"] is False, "the stale clean verdict survived a re-bench"
    assert doc["nway_light_roles"] == ["frontdoor"], "hand-authored policy was dropped"


# ── 7. Read side: an unstamped matrix reads UNKNOWN, never clean ─────


def test_a_matrix_without_a_stamp_reads_unknown(tmp_path) -> None:
    """Every matrix written before today is in this class, including the one on disk."""
    path = tmp_path / "old.yaml"
    path.write_text('version: 1\nhost: "Beelzebub"\npairs: []\n')
    record = cm.read_matrix_host_health(path)
    assert record["status"] == cm.HOST_HEALTH_UNKNOWN
    assert record["decision_grade"] is False
    assert "UNKNOWN, not clean" in record["reason"]
    assert any("UNKNOWN" in line for line in cm.describe_matrix_host_health(path))


def test_an_unreadable_matrix_reads_unknown(tmp_path) -> None:
    record = cm.read_matrix_host_health(tmp_path / "does-not-exist.yaml")
    assert record["status"] == cm.HOST_HEALTH_UNKNOWN
    assert record["decision_grade"] is False
    assert record["reason"]


def test_a_stamped_matrix_round_trips_through_the_reader(tmp_path) -> None:
    probe = cm._host_health_probe(load_rules=_rules_loader(_attestation()))
    path = tmp_path / "new.yaml"
    path.write_text(_emit(probe))
    record = cm.read_matrix_host_health(path)
    assert record["status"] == cm.HOST_HEALTH_CLEAN
    assert record["decision_grade"] is True

    np_sweep, _ = _real_rules()
    dirty = cm._host_health_probe(
        load_rules=_rules_loader(
            _attestation(uptime_seconds=np_sweep.MAX_DECISION_GRADE_UPTIME_SECONDS + 1)
        )
    )
    path.write_text(_emit(dirty))
    record = cm.read_matrix_host_health(path)
    assert record["status"] == cm.HOST_HEALTH_WARN
    assert record["decision_grade"] is False
    assert any("uptime" in w for w in record["warnings"])
