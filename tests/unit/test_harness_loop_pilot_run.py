"""HIL-6j: the pilot-only entrypoint. Refusals, the exclusion set/clear, resumability and a full
dry run, all on injected fakes: no model, no CLI, no network, no live exclusions file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.autopilot_core import mutation_exclusions as mx
from src.harness_loop import pilot_run as pr
from src.harness_loop import substrate
from src.harness_loop.interfaces import ArmResult
from src.harness_loop.pilots import first_feature as ff

ROOT = Path(__file__).resolve().parents[2]
GOOD_ENV = {"WINDOW_CONFIRMED": "1"}


@pytest.fixture(autouse=True)
def _private_exclusions(tmp_path, monkeypatch):
    monkeypatch.setenv(mx.ENV_FILE, str(tmp_path / "excl" / "exclusions.json"))
    # the real MHS-3 guard parses the 1.35 GB pool; the screen's protected layer still runs
    monkeypatch.setattr(substrate, "leakage_reason", lambda *a, **k: None)


def _pool(tmp_path: Path) -> Path:
    path = tmp_path / "pool.jsonl"
    need = sum(n for _, n in ff.PER_STRATUM) + 5
    with open(path, "w") as fh:
        fh.write(json.dumps({"__pool_metadata__": True, "generator": "test", "total_questions": 0}) + "\n")
        for s in ff.STRATA:
            for i in range(need):
                fh.write(json.dumps({"id": f"{s}_{i:04d}", "suite": s, "prompt": f"Q {s} {i}: answer?",
                                     "context": "", "expected": f"ans{i}", "image_path": None,
                                     "scoring_method": "exact_match", "scoring_config": {}}) + "\n")
    return path


def _live_target(tmp_path: Path) -> Path:
    p = tmp_path / "live" / "architect_investigate.md"
    p.parent.mkdir(parents=True)
    p.write_text((ROOT / ff.TARGET_PATH).read_text())
    return p


def _argv(phase, run_dir, *extra, lift=pr.LIFT_TOKEN, design=pr.PILOT_DESIGN_ID):
    out = ["--design", design, "--phase", phase, "--run-dir", str(run_dir)]
    if lift is not None:
        out += ["--operator-lift", lift]
    return out + list(extra)


# ── refusals ──────────────────────────────────────────────────────────────────


def test_refuses_without_window(tmp_path, capsys):
    assert pr.main(_argv("preflight", tmp_path), env={}) == pr.EXIT_REFUSED
    assert "WINDOW_CONFIRMED" in capsys.readouterr().err


def test_refuses_without_or_with_wrong_lift_token(tmp_path, capsys):
    assert pr.main(_argv("run", tmp_path, lift=None), env=GOOD_ENV) == pr.EXIT_REFUSED
    assert pr.main(_argv("run", tmp_path, lift="2026-10-22"), env=GOOD_ENV) == pr.EXIT_REFUSED
    assert capsys.readouterr().err.count("--operator-lift") == 2


def test_refuses_any_other_design(tmp_path, capsys):
    for design in ("hil6i-second-feature", "HS4-3.1-profile-memory", ""):
        assert pr.main(_argv("freeze", tmp_path, design=design), env=GOOD_ENV) == pr.EXIT_REFUSED
    assert "not the HIL-6i pilot" in capsys.readouterr().err
    assert pr.refusal_reasons(design=ff.FEATURE_ID, lift=pr.LIFT_TOKEN, env=GOOD_ENV) == []


def test_refuses_when_the_validator_fails(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    assert pr.main(_argv("freeze", run, "--pool", str(_pool(tmp_path)),
                         "--live-target", str(_live_target(tmp_path))), env=GOOD_ENV) == pr.EXIT_OK
    monkeypatch.setattr(pr, "unmet_requirements", lambda d: [("P3", "judge shares the editor family")])
    for phase in ("preflight", "run", "status"):
        assert pr.main(_argv(phase, run, "--fake-backends"), env=GOOD_ENV) == pr.EXIT_REFUSED
    assert "REFUSED (validator)" in capsys.readouterr().err
    assert not mx.exclusions_file().exists()  # nothing was set by a refused run


def test_validator_passes_for_both_cycles_and_covers_items_1_to_22(tmp_path):
    rows = [json.loads(l) for l in _pool(tmp_path).read_text().splitlines()[1:]]
    assert pr.validator_reasons(ff.build_splits(rows)) == []


def test_no_scheduler_hook_and_no_other_entrypoint():
    pkg = ROOT / "src" / "harness_loop"
    mains = sorted(f.name for f in pkg.rglob("*.py") if re.search(r"__name__\s*==\s*[\"']__main__", f.read_text()))
    assert mains == ["pilot_run.py"]
    pat = re.compile(r"harness_loop[./]pilot_run")
    for base in ("src", "scripts", "orchestration"):
        for f in (ROOT / base).rglob("*"):
            if f.is_file() and f.suffix in {".py", ".sh", ".yaml", ".yml", ".toml", ".json"} and pkg not in f.parents:
                assert not pat.search(f.read_text(errors="ignore")), f


# ── exclusion set / clear ─────────────────────────────────────────────────────


def test_exclusion_set_and_clear_phases(tmp_path):
    assert pr.main(_argv("exclude-set", tmp_path), env=GOOD_ENV) == pr.EXIT_OK
    assert mx.load() == frozenset({ff.TARGET_PATH})
    # clear needs no window and no token: the job's EXIT trap must always be able to clear
    assert pr.main(["--design", "x", "--phase", "exclude-clear"], env={}) == pr.EXIT_OK
    assert not mx.exclusions_file().exists()


def test_exclusion_set_refused_without_lift(tmp_path):
    assert pr.main(_argv("exclude-set", tmp_path, lift=None), env=GOOD_ENV) == pr.EXIT_REFUSED
    assert not mx.exclusions_file().exists()


def test_run_clears_the_exclusion_on_failure(tmp_path, monkeypatch):
    run = tmp_path / "run"
    pr.main(_argv("freeze", run, "--pool", str(_pool(tmp_path)), "--live-target", str(_live_target(tmp_path))),
            env=GOOD_ENV)
    seen = {}

    def boom(ctx):
        seen["during"] = mx.load()
        raise RuntimeError("mid-run failure")

    monkeypatch.setattr(pr, "run_pilot", boom)
    with pytest.raises(RuntimeError):
        pr.main(_argv("run", run, "--fake-backends"), env=GOOD_ENV)
    assert seen["during"] == frozenset({ff.TARGET_PATH})
    assert not mx.exclusions_file().exists()


# ── pieces ────────────────────────────────────────────────────────────────────


def test_run_mean_sign_test_uses_run_means():
    def res(outs):
        return ArmResult("a", "V", ff.PRIMARY_METRIC, None, ff.PRIMARY_UNIT, ff.PRIMARY_DIRECTION,
                         len(outs), len(outs), item_outcomes=outs)

    inc = {f"q{i}@r{r}": False for i in range(80) for r in (1, 2)}
    cand = {f"q{i}@r{r}": True for i in range(80) for r in (1, 2)}
    assert pr.RunMeanSignTest().decide(res(inc), res(cand)).keep
    half = {f"q{i}@r1": True for i in range(80)} | {f"q{i}@r2": False for i in range(80)}
    d = pr.RunMeanSignTest().decide(res(half), res(half))
    assert not d.keep and "paired_wins:0/0" in d.reason


def test_resuming_ledger_does_not_reserve_alpha_twice(tmp_path):
    j = substrate.open_loop_journal(tmp_path / "j")
    led = pr.ResumingLedger(j, alpha_budget=0.05)
    a = led.open("cand-1", 0.00625, cycle=1, split="V")
    assert led.open("cand-1", 0.00625, cycle=1, split="V") == a
    assert abs(led.alpha_spent() - 0.00625) < 1e-12
    led.close(a, "keep_base")
    assert led.open("cand-1", 0.00625, cycle=1, split="V") != a  # a closed attempt is not resumed


def test_cached_transport_stops_before_uncached_requests(tmp_path):
    cache = pr.ChatCache(tmp_path / "c.jsonl")
    calls = []
    stop = {"now": False}
    t = pr.cached_transport(lambda b: calls.append(b) or {"answer": "x"}, cache, lambda: stop["now"])
    t({"request_id": "r1"})
    stop["now"] = True
    assert t({"request_id": "r1"}) == {"answer": "x"}  # cached: replayed even when stopping
    with pytest.raises(pr.PilotStopped):
        t({"request_id": "r2"})
    assert len(calls) == 1
    assert pr.ChatCache(tmp_path / "c.jsonl").get("r1")["response"] == {"answer": "x"}


def test_qwen_editor_transport_disables_thinking_and_refuses_empty():
    sent = {}

    def inner(body):
        sent.update(body)
        return {"choices": [{"message": {"content": ""}}]}

    with pytest.raises(pr.TransportError):
        pr.qwen_editor_transport(inner)({"messages": []})
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_estimates_are_stated():
    c = pr.estimate_cloud_calls()
    assert c["total_upper_bound"] == 2 + 2 + 80
    e = pr.estimate_requests()
    assert e["requests_total_upper"] > e["requests_total_lower"] > 0
    assert e["hours_total_upper"] > 0


def test_cloud_live_verification_records_and_never_edits_registry():
    fakes = pr.FakeBackends()
    from src.typed_decisions.cloud_judges import load_registry

    before = (ROOT / "orchestration" / "cloud_judges.yaml").read_bytes()
    out = pr.cloud_live_verification(load_registry(), run_fn=fakes.cloud_run)
    assert [r["entry"] for r in out] == [pr.REFERENCE_EDITOR_JUDGE_ENTRY, pr.JUDGE_ENTRY]
    assert all(r["ok"] for r in out) and fakes.calls["cloud"] == 2
    assert (ROOT / "orchestration" / "cloud_judges.yaml").read_bytes() == before


# ── full dry run on fakes ─────────────────────────────────────────────────────


def test_full_dry_run_on_fakes_is_resumable(tmp_path):
    run = tmp_path / "run"
    live = _live_target(tmp_path)
    assert pr.main(_argv("freeze", run, "--pool", str(_pool(tmp_path)), "--live-target", str(live)),
                   env=GOOD_ENV) == pr.EXIT_OK
    assert pr.main(_argv("preflight", run, "--fake-backends"), env=GOOD_ENV) == pr.EXIT_OK
    manifest = json.loads((run / "manifest.json").read_text())
    assert all(r["ok"] for r in manifest["cloud_live_verification"][-1]["results"])

    # first window: stop after a few hundred requests
    stopfile = tmp_path / "STOP"
    fakes = pr.FakeBackends(run / "items.jsonl")
    orig = fakes.chat

    def chat_then_stop(body):
        if fakes.calls["chat"] >= 300:
            stopfile.write_text("stop")
        return orig(body)

    fakes.chat = chat_then_stop
    rc = pr.main(_argv("run", run, "--fake-backends", "--live-target", str(live), "--stopfile", str(stopfile)),
                 env=GOOD_ENV, fakes=fakes)
    assert rc == pr.EXIT_STOPPED and not (run / "PILOT_DONE").exists()
    assert not mx.exclusions_file().exists()
    first = fakes.calls["chat"]

    stopfile.unlink()
    fakes2 = pr.FakeBackends(run / "items.jsonl")
    rc = pr.main(_argv("run", run, "--fake-backends", "--live-target", str(live), "--ignore-power-gate"),
                 env=GOOD_ENV, fakes=fakes2)
    assert rc == pr.EXIT_OK and (run / "PILOT_DONE").exists()
    report = json.loads((run / "report.json").read_text())
    assert report["fake_backends"] and report["decision_grade"] == "observation" and report["protocol_id"] is None
    assert set(report["cycles"]) == {"c1.local_editor", "c1.reference_editor", "c2.local_editor", "c2.reference_editor"}
    assert report["test"]
    assert not mx.exclusions_file().exists()
    # the second window never re-asked what the first one cached
    cache = pr.ChatCache(run / "cache" / "chat.jsonl")
    assert len(cache) == first + fakes2.calls["chat"]
    # a third invocation is a no-op replay
    fakes3 = pr.FakeBackends(run / "items.jsonl")
    assert pr.main(_argv("run", run, "--fake-backends", "--live-target", str(live), "--ignore-power-gate"),
                   env=GOOD_ENV, fakes=fakes3) == pr.EXIT_OK
    assert fakes3.calls["chat"] == 0 and fakes3.calls["cloud"] == 0 and fakes3.calls["editor"] == 0


def test_live_target_change_requires_refreeze(tmp_path):
    run = tmp_path / "run"
    live = _live_target(tmp_path)
    pr.main(_argv("freeze", run, "--pool", str(_pool(tmp_path)), "--live-target", str(live)), env=GOOD_ENV)
    live.write_text(live.read_text() + "\nmutated by someone\n")
    fakes = pr.FakeBackends(run / "items.jsonl")
    rc = pr.main(_argv("run", run, "--fake-backends", "--live-target", str(live), "--ignore-power-gate"),
                 env=GOOD_ENV, fakes=fakes)
    assert rc == pr.EXIT_REFREEZE
    assert not mx.exclusions_file().exists()
