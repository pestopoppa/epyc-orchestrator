"""RTG-DUR-1: the duration-baseline refresh rule and its refusals.

Acceptance: a test that FAILS when a role's bound model changed after the baseline
window (`test_model_change_after_window_is_refused`), and derivation that stamps window
and n per role. Nothing here regenerates the live artifact; all inputs are synthetic.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from src.registry import duration_baseline_freshness as fresh

REPO_ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
FLOOR = datetime(2026, 9, 24, 10, 2, 39, tzinfo=timezone.utc)


def _load_derive():
    spec = importlib.util.spec_from_file_location(
        "derive_duration_baselines_under_test",
        REPO_ROOT / "scripts" / "analysis" / "derive_duration_baselines.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _entry(model="m-a", first="2026-10-01T00:00:00+00:00", n=100):
    return {"n": n, "p50_s": 5.0, "p90_s": 20.0, "mean_s": 8.0,
            "first_ts": first, "last_ts": "2026-11-10T00:00:00+00:00", "bound_model_id": model}


def _doc(roles, generated="2026-11-15T00:00:00+00:00"):
    return {"generated_at_utc": generated, "roles": roles}


def _eval(doc, bindings, **kw):
    return fresh.evaluate(doc, bindings, floor=FLOOR, floor_id="E18-test", now=NOW, **kw)


def test_fresh_entry_is_usable():
    rep = _eval(_doc({"frontdoor": _entry("m-a")}), {"frontdoor": "m-a"})
    assert rep.ok and rep.usable_roles == {"frontdoor"}


def test_model_change_after_window_is_refused():
    """The RTG-DUR-1 acceptance: role re-bound to another model after the window."""
    rep = _eval(_doc({"architect_general": _entry("old-model")}), {"architect_general": "flash-next"})
    assert rep.stale_roles == {"architect_general": ["model_changed"]}
    assert rep.verdicts["architect_general"].detail == {
        "stamped_model": "old-model", "current_model": "flash-next"}
    assert not rep.ok


def test_retired_role_is_refused():
    rep = _eval(_doc({"architect_coding": _entry("old")}), {"frontdoor": "m-a"})
    assert "role_not_bound" in rep.stale_roles["architect_coding"]
    assert rep.unbaselined_live_roles == ["frontdoor"]


def test_v1_entry_without_stamp_is_unverifiable():
    legacy = {"n": 5000, "p50_s": 9.1, "p90_s": 75.4, "mean_s": 24.9}  # the 2026-09-24 shape
    rep = _eval(_doc({"coder_escalation": legacy}), {"coder_escalation": "27b"})
    reasons = rep.stale_roles["coder_escalation"]
    assert "no_model_stamp" in reasons and "window_before_era" in reasons


def test_window_straddling_era_boundary_is_refused():
    rep = _eval(_doc({"frontdoor": _entry("m-a", first="2026-09-01T00:00:00+00:00")}),
                {"frontdoor": "m-a"})
    assert rep.stale_roles == {"frontdoor": ["window_before_era"]}


def test_expired_and_low_n():
    rep = _eval(_doc({"frontdoor": _entry("m-a", n=10)}, generated="2026-08-01T00:00:00+00:00"),
                {"frontdoor": "m-a"})
    assert set(rep.stale_roles["frontdoor"]) == {"expired", "low_n"}


def test_missing_generated_stamp_is_expired():
    rep = _eval({"roles": {"frontdoor": _entry("m-a")}}, {"frontdoor": "m-a"})
    assert "expired" in rep.stale_roles["frontdoor"]


def test_era_floor_uses_fence_scopes_and_skips_unrooted_rows(tmp_path):
    eras = {"eras": [
        {"id": "E18", "scope": "routing_reward", "from": "2026-09-24T10:02:39Z"},
        {"id": "E20", "scope": "eval_quality", "from": "2026-10-21T15:19:42Z"},  # not a fence scope
        {"id": "cand", "scope": "cpu_bench", "binary_version": 1},               # no `from`
        {"id": "future", "scope": "autopilot_speed", "from": "2026-12-01T00:00:00Z"},
    ]}
    path = tmp_path / "eras.yaml"
    path.write_text(yaml.safe_dump(eras))
    floor, era_id = fresh.era_floor(path, now=NOW)
    assert era_id == "E18" and floor == FLOOR
    # once the relaunch era is in the past, it becomes the floor
    floor2, id2 = fresh.era_floor(path, now=datetime(2026, 12, 2, tzinfo=timezone.utc))
    assert id2 == "future"


def test_current_bindings_only_live_stack(tmp_path):
    priors = {"roles": {
        "frontdoor": {"role": "frontdoor", "deployment_status": "live_stack", "model_id": "m-a"},
        "cand": {"role": "cand", "deployment_status": "benchmark_or_candidate", "model_id": "m-b"},
    }}
    path = tmp_path / "priors.yaml"
    path.write_text(yaml.safe_dump(priors))
    assert fresh.current_role_bindings(path) == {"frontdoor": "m-a"}


def test_check_files_missing_artifact_is_all_stale(tmp_path):
    (tmp_path / "priors.yaml").write_text(yaml.safe_dump({"roles": {
        "frontdoor": {"role": "frontdoor", "deployment_status": "live_stack", "model_id": "m-a"}}}))
    (tmp_path / "eras.yaml").write_text(yaml.safe_dump({"eras": []}))
    rep = fresh.check_files(tmp_path / "absent.json", tmp_path / "priors.yaml",
                            tmp_path / "eras.yaml", now=NOW)
    assert rep.unbaselined_live_roles == ["frontdoor"] and not rep.ok


def test_live_artifact_is_reported_stale_today():
    """The shipped 2026-09-24 artifact (v1, unstamped) must be refused, not silently used."""
    rep = fresh.check_files(
        REPO_ROOT / "orchestration" / "derived" / "duration_baselines_by_role.json",
        REPO_ROOT / "orchestration" / "derived" / "stack_priors.yaml",
        REPO_ROOT / "orchestration" / "instrument_eras.yaml",
        now=NOW,
    )
    assert "architect_coding" in rep.stale_roles
    assert all("no_model_stamp" in why for why in rep.stale_roles.values())
    assert rep.usable_roles == set()


# --------------------------------------------------------------- derivation (v2)

def _write_log(dirpath: Path, day: str, rows):
    """rows: (task_id, role, start_iso, wall_s)."""
    lines = []
    for tid, role, start, wall in rows:
        t0 = datetime.fromisoformat(start)
        lines.append(json.dumps({"event_type": "task_started", "task_id": tid,
                                 "timestamp": t0.isoformat(), "data": {}}))
        lines.append(json.dumps({"event_type": "task_completed", "task_id": tid,
                                 "timestamp": (t0 + timedelta(seconds=wall)).isoformat(),
                                 "data": {"producer_role": role}}))
    (dirpath / f"{day}.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture
def logs(tmp_path):
    d = tmp_path / "progress"
    d.mkdir()
    pre = [(f"a{i}", "frontdoor", "2026-09-10T00:00:00+00:00", 100.0) for i in range(40)]
    post = [(f"b{i}", "frontdoor", "2026-10-05T00:00:00+00:00", 10.0 + i % 3) for i in range(40)]
    swapped = [(f"c{i}", "architect_general", "2026-10-05T00:00:00+00:00", 50.0) for i in range(40)]
    swapped += [(f"d{i}", "architect_general", "2026-10-20T00:00:00+00:00", 7.0) for i in range(40)]
    retired = [(f"e{i}", "architect_coding", "2026-10-05T00:00:00+00:00", 9.0) for i in range(40)]
    _write_log(d, "2026-09-10", pre)
    _write_log(d, "2026-10-05", post + swapped[:40] + retired)
    _write_log(d, "2026-10-20", swapped[40:])
    return d


def test_derive_floors_window_stamps_model_and_drops_unbound(logs):
    mod = _load_derive()
    bindings = {"frontdoor": "m-a", "architect_general": "flash-next"}
    out = mod.derive(logs, since=FLOOR, bindings=bindings, era_floor_utc=FLOOR, era_floor_id="E18-test")
    fd = out["roles"]["frontdoor"]
    assert fd["n"] == 40 and fd["p50_s"] < 20          # the pre-floor 100 s rows are gone
    assert fd["bound_model_id"] == "m-a" and fd["first_ts"].startswith("2026-10-05")
    assert out["n_pairs_excluded_before_window"] == 40
    assert out["excluded_unbound_roles"] == {"architect_coding": 40}
    assert "architect_coding" not in out["roles"]
    assert out["source_window"]["era_floor_id"] == "E18-test"
    assert out["protocol_id"].endswith("-v2")


def test_role_since_starts_a_swapped_roles_window_at_the_swap(logs):
    mod = _load_derive()
    swap = datetime(2026, 10, 15, tzinfo=timezone.utc)
    out = mod.derive(logs, since=FLOOR, role_since={"architect_general": swap},
                     bindings={"architect_general": "flash-next", "frontdoor": "m-a"})
    ag = out["roles"]["architect_general"]
    assert ag["n"] == 40 and ag["p50_s"] == 7.0         # only post-swap traffic
    assert ag["role_since_utc"] == swap.isoformat()
    # the other role is unaffected by the per-role floor
    assert out["roles"]["frontdoor"]["n"] == 40


def test_derived_artifact_passes_the_freshness_rule(logs, tmp_path):
    mod = _load_derive()
    out = mod.derive(logs, since=FLOOR, bindings={"frontdoor": "m-a"})
    out["generated_at_utc"] = NOW.isoformat()
    rep = fresh.evaluate(out, {"frontdoor": "m-a"}, floor=FLOOR, now=NOW)
    assert rep.ok
    # ...and the same artifact is refused the moment the role's model changes
    rep2 = fresh.evaluate(out, {"frontdoor": "m-b"}, floor=FLOOR, now=NOW)
    assert rep2.stale_roles == {"frontdoor": ["model_changed"]}


def test_since_before_era_floor_is_refused(monkeypatch, tmp_path, capsys):
    mod = _load_derive()
    (tmp_path / "eras.yaml").write_text(yaml.safe_dump({"eras": [
        {"id": "E18", "scope": "routing_reward", "from": "2026-09-24T10:02:39Z"}]}))
    (tmp_path / "priors.yaml").write_text(yaml.safe_dump({"roles": {}}))
    argv = ["derive", "--log-dir", str(tmp_path), "--out", str(tmp_path / "o.json"),
            "--eras", str(tmp_path / "eras.yaml"), "--stack-priors", str(tmp_path / "priors.yaml"),
            "--since", "2026-08-01T00:00:00Z"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert "before the era floor" in str(exc.value)
    assert not (tmp_path / "o.json").exists()
    # explicit override is allowed (and would proceed to read logs)
    monkeypatch.setattr("sys.argv", argv + ["--allow-cross-era"])
    with pytest.raises(SystemExit) as exc2:
        mod.main()
    assert "no progress logs" in str(exc2.value)


def test_check_mode_exits_nonzero_on_stale_and_writes_nothing(monkeypatch, tmp_path, capsys):
    mod = _load_derive()
    art = tmp_path / "art.json"
    art.write_text(json.dumps(_doc({"frontdoor": _entry("old")}, generated=NOW.isoformat())))
    before = art.read_text()
    (tmp_path / "eras.yaml").write_text(yaml.safe_dump({"eras": []}))
    (tmp_path / "priors.yaml").write_text(yaml.safe_dump({"roles": {
        "frontdoor": {"role": "frontdoor", "deployment_status": "live_stack", "model_id": "new"}}}))
    monkeypatch.setattr("sys.argv", ["derive", "--check", "--out", str(art),
                                     "--eras", str(tmp_path / "eras.yaml"),
                                     "--stack-priors", str(tmp_path / "priors.yaml"),
                                     "--max-age-days", "100000"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1
    assert "frontdoor: STALE (model_changed)" in capsys.readouterr().out
    assert art.read_text() == before


# --------------------------------------------------------------- loader + pipeline wiring

def _write_fixture_tree(tmp_path, stamped_model, current_model):
    (tmp_path / "art.json").write_text(json.dumps(_doc({
        "frontdoor": _entry(stamped_model, first="2026-10-05T00:00:00+00:00"),
        "worker_general": _entry("m-a", first="2026-10-05T00:00:00+00:00"),
    }, generated=datetime.now(timezone.utc).isoformat())))
    return tmp_path / "art.json"


def test_q_scorer_loader_default_off_is_unchanged(monkeypatch):
    from orchestration.repl_memory.q_scorer import load_duration_baselines_by_role

    monkeypatch.delenv("ORCHESTRATOR_DURATION_BASELINE_FRESHNESS", raising=False)
    path = REPO_ROOT / "orchestration" / "derived" / "duration_baselines_by_role.json"
    out, reason = load_duration_baselines_by_role(path)
    assert reason is None and "architect_coding" in out and len(out) == 10


def test_q_scorer_loader_warn_changes_nothing_enforce_drops_stale(monkeypatch):
    from orchestration.repl_memory.q_scorer import load_duration_baselines_by_role

    path = REPO_ROOT / "orchestration" / "derived" / "duration_baselines_by_role.json"
    monkeypatch.setenv("ORCHESTRATOR_DURATION_BASELINE_FRESHNESS", "warn")
    out, reason = load_duration_baselines_by_role(path)
    assert reason is None and len(out) == 10

    monkeypatch.setenv("ORCHESTRATOR_DURATION_BASELINE_FRESHNESS", "enforce")
    out, reason = load_duration_baselines_by_role(path)
    assert out == {} and "stale, dropped" in reason and "no_model_stamp" in reason


def test_pipeline_step_off_by_default_and_modes(tmp_path):
    from scripts.registry import stack_change_pipeline as pipe

    cfg = pipe.StackChangePipelineConfig(mode="check", repo_root=REPO_ROOT)
    assert cfg.duration_baselines == "off"
    warn = pipe._duration_baselines_step(
        pipe.StackChangePipelineConfig(mode="check", repo_root=REPO_ROOT, duration_baselines="warn"))
    assert warn.ok and warn.warnings and warn.status == "warning"
    assert any("derive_duration_baselines.py" in d for d in warn.details)
    enforce = pipe._duration_baselines_step(
        pipe.StackChangePipelineConfig(mode="update", repo_root=REPO_ROOT, duration_baselines="enforce"))
    assert not enforce.ok and enforce.errors
