"""serving-config artifact -> registry diff -> scratch dry-run (PROMOTION-REGISTRY-SERVING-1)."""
from __future__ import annotations

import os
import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.registry import serving_config_registry as scr

ROOT_WORKTREE = Path(os.environ.get("EPYC_ROOT", "/workspace"))
REAL_C22 = ROOT_WORKTREE / "docs/reference/kernel-manifests/serving-configs/C22.serving-config.json"
ROOT_VALIDATOR = ROOT_WORKTREE / "scripts/champion/serving_config/validate_serving_config.py"
needs_real = pytest.mark.skipif(
    not (REAL_C22.is_file() and ROOT_VALIDATOR.is_file()), reason="epyc-root C22 artifact/validator not on this host"
)

PROV = {"row_id": "ROW-1", "result_json": "/x/RESULT.json", "result_sha256": "c" * 64, "store": "/x/store"}
SHAPE = {"n_ctx": 262144, "batch": 512, "ubatch": 512, "kv_unified": True, "kv_quant": {"k": "q8_0", "v": "q8_0"}}


def _entry(**cfg) -> dict:
    """A MEASURED entry in the real artifact's shape (copy of C22's architect_critic optimal)."""
    config = {
        "n_ctx": 262144, "np": 4, "batch": 512, "ubatch": 512, "kv_k": "q8_0", "kv_v": "q8_0",
        "kv_unified": True, "ctx_checkpoints": 32, "cache_ram_mib": 65536,
        "drafter": {"spec_type": "draft-dflash", "model": "/m/d.gguf", "n_max": 7},
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    }
    config.update(cfg)
    return {"status": "measured", "config": config, "argv_delta": [], "provenance": dict(PROV)}


def _config() -> dict:
    return {
        "schema": scr.CONFIG_SCHEMA,
        "champion": "fixture-c22",
        "generated_utc": "2026-11-24T20:00:00Z",
        "servers": [
            {
                "server_mode": "frontdoor", "port": 8070, "model": "m", "device": "CPU",
                "registry_serving_shape": dict(SHAPE),
                "optimal": _entry(np=6, ctx_checkpoints=16, n_ctx=131072, kv_k="q4_0")
                | {"argv_delta": [{"flag": "-c", "value": 131072, "reason": "r"},
                                  {"flag": "-ctk", "value": "q4_0", "reason": "r"}]},
                "alternates": [
                    {"name": "latency", "status": "measured", "config": _entry()["config"],
                     "argv_delta": [{"flag": "-ub", "value": 256, "reason": "latency arm"},
                                    {"flag": "--no-warmup", "reason": "bare flag"}],
                     "provenance": dict(PROV)},
                    {"name": "throughput", "status": "pending", "row_id": "ROW-SWEEP",
                     "pending_reason": "sweep has no RESULT.json yet"},
                ],
                "tradeoffs": [],
            },
            {
                "server_mode": "architect_general", "port": 8084, "model": "m2", "device": "CPU",
                "registry_serving_shape": {},
                "optimal": {"status": "pending", "row_id": "ROW-Q38", "pending_reason": "no RESULT.json"},
                "alternates": [], "tradeoffs": [],
            },
        ],
    }


def _master() -> dict:
    return {
        "server_mode": {
            "frontdoor": {
                "slots": 4,
                "cache_ram": 65536,
                "ctx_checkpoints": 32,
                "serving_shape": {
                    **copy.deepcopy(SHAPE),
                    "slots_by_shape": {"full": 4},
                    "alternates": {
                        "stale": {"argv_delta": {"x": 1}, "provenance": dict(PROV)},
                        "throughput": {"argv_delta": {"-b": 1024}, "provenance": dict(PROV)},
                    },
                },
            },
            "architect_general": {"slots": 4, "serving_shape": {"n_ctx": 1}},
            "architect_critic": {"slots": 4},
        }
    }


# ----------------------------------------------------------------------- artifact


def test_valid_config_loads(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_config()))
    assert scr.load_serving_config(p)["champion"] == "fixture-c22"


def test_missing_artifact_is_an_error(tmp_path: Path):
    with pytest.raises(scr.ServingConfigError, match="not found"):
        scr.load_serving_config(tmp_path / "absent.json")


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda c: c.update(schema="other"), "schema"),
        (lambda c: c.update(servers=[]), "non-empty"),
        (lambda c: c["servers"][0].pop("server_mode"), "server_mode"),
        (lambda c: c["servers"][0]["optimal"].update(status="done"), "measured|pending"),
        (lambda c: c["servers"][0]["alternates"][0].update(status="?"), "measured|pending"),
    ],
)
def test_malformed_config_is_rejected(mutate, needle):
    cfg = _config()
    mutate(cfg)
    with pytest.raises(scr.ServingConfigError, match=needle):
        scr.check_shape(cfg)


@needs_real
def test_the_root_validator_is_the_parser_when_handed_over(tmp_path: Path):
    doc = json.loads(REAL_C22.read_text())
    good = tmp_path / "good.json"
    good.write_text(json.dumps(doc))
    assert scr.load_serving_config(good, validator=ROOT_VALIDATOR)["champion"] == doc["champion"]
    doc["servers"][0]["optimal"]["provenance"].pop("result_sha256")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(scr.ServingConfigError, match="validate_serving_config.py"):
        scr.load_serving_config(bad, validator=ROOT_VALIDATOR)


@needs_real
def test_real_c22_against_a_registry_equal_to_its_shape_is_an_empty_diff():
    doc = json.loads(REAL_C22.read_text())
    srv = {s["server_mode"]: s for s in doc["servers"]}["architect_critic"]
    cfg = srv["optimal"]["config"]
    master = {"server_mode": {"architect_critic": {
        "slots": cfg["np"], "cache_ram": cfg["cache_ram_mib"], "ctx_checkpoints": cfg["ctx_checkpoints"],
        "serving_shape": copy.deepcopy(srv["registry_serving_shape"]),
    }, "architect_general": {"slots": 1}}}
    diff = scr.build_registry_diff(doc, master)
    assert diff["ops"] == []  # a measured optimum that IS the served recipe changes nothing
    skipped = {(s["server_mode"], s["entry"]) for s in diff["skipped_pending"]}
    assert ("architect_general", "optimal") in skipped
    assert {("architect_critic", n) for n in ("latency", "throughput", "low-temperature")} <= skipped


# ----------------------------------------------------------------------- diff / apply


def test_diff_writes_measured_entries_only_and_skips_pending():
    diff = scr.build_registry_diff(_config(), _master(), config_sha256="d" * 64)
    by_path = {".".join(op["path"]): op for op in diff["ops"]}
    assert by_path["server_mode.frontdoor.slots"]["old"] == 4
    assert by_path["server_mode.frontdoor.slots"]["new"] == 6
    assert by_path["server_mode.frontdoor.ctx_checkpoints"]["new"] == 16
    assert by_path["server_mode.frontdoor.serving_shape.n_ctx"]["new"] == 131072
    assert by_path["server_mode.frontdoor.serving_shape.kv_quant.k"]["new"] == "q4_0"
    latency = by_path["server_mode.frontdoor.serving_shape.alternates.latency"]
    assert latency["had_old"] is False
    assert latency["new"]["argv_delta"] == {"-ub": 256, "--no-warmup": True}
    assert latency["new"]["provenance"]["row_id"] == "ROW-1"
    assert by_path["server_mode.frontdoor.serving_shape.alternates.stale"]["op"] == "remove"
    # equal-to-registry config values, drafter and sampling are never ops
    assert not any(k.endswith((".batch", ".ubatch", ".cache_ram", "kv_unified", "drafter", "sampling")) for k in by_path)
    # PENDING: never written, never removed, and listed so the package shows what is not covered
    assert not any("throughput" in k for k in by_path)
    assert not any("architect_general" in k for k in by_path)
    assert {(s["server_mode"], s["entry"]) for s in diff["skipped_pending"]} == {
        ("frontdoor", "throughput"), ("architect_general", "optimal")}
    assert diff["summary"] == {"ops": 6, "roles": {"frontdoor": {"knobs": 4, "alternates": 2}},
                               "skipped_pending": 2}
    assert diff["serving_config_sha256"] == "d" * 64


def test_diff_skips_values_already_in_the_registry():
    master = _master()
    master["server_mode"]["frontdoor"]["slots"] = 6
    diff = scr.build_registry_diff(_config(), master)
    assert "server_mode.frontdoor.slots" not in {".".join(o["path"]) for o in diff["ops"]}


def test_diff_refuses_a_measured_role_the_registry_does_not_serve():
    cfg = _config()
    cfg["servers"][0]["server_mode"] = "ghost"
    with pytest.raises(scr.ServingConfigError, match="ghost"):
        scr.build_registry_diff(cfg, _master())


def test_a_pending_role_may_be_absent_from_the_registry():
    cfg = _config()
    cfg["servers"][1]["server_mode"] = "not_in_registry"
    scr.build_registry_diff(cfg, _master())


def test_measured_alternate_with_empty_delta_is_refused():
    cfg = _config()
    cfg["servers"][0]["alternates"][0]["argv_delta"] = []
    with pytest.raises(scr.ServingAlternatesError, match="non-empty"):
        scr.build_registry_diff(cfg, _master())


def test_alternates_write_path_end_to_end_with_a_measured_alternate(tmp_path: Path):
    """A MEASURED alternate: artifact -> diff op -> patched master -> the pipeline's alternates step (wiring-audit
    defect A3: the real C22 diff is empty, so no preflight ever exercised this)."""
    from scripts.registry import stack_change_pipeline as pipeline

    master = _master()
    diff = scr.build_registry_diff(_config(), master)
    assert any(".alternates.latency" in ".".join(o["path"]) and o["op"] == "set" for o in diff["ops"])
    patched = scr.apply_registry_diff(master, diff)
    reg = tmp_path / "patched.yaml"
    reg.write_text(yaml.safe_dump(patched, sort_keys=False))
    step = pipeline._serving_alternates_step(pipeline.StackChangePipelineConfig(
        mode="check", repo_root=tmp_path, lean_registry=reg, research_registry=reg))
    assert step.status == "ok", step
    assert any("server_mode.frontdoor.serving_shape.alternates" in d for d in step.details)
    # the same step also runs when the diff is EMPTY (patched is already the served recipe) and still catches a bad slot
    assert scr.build_registry_diff(_config(), patched)["ops"] == []
    patched["server_mode"]["frontdoor"]["serving_shape"]["alternates"]["latency"]["argv_delta"]["-b"] = True
    reg.write_text(yaml.safe_dump(patched, sort_keys=False))
    bad = pipeline._serving_alternates_step(pipeline.StackChangePipelineConfig(
        mode="check", repo_root=tmp_path, lean_registry=reg, research_registry=reg))
    assert bad.status == "failed" and any("takes a value" in e for e in bad.errors)


def test_a_flag_without_its_value_cannot_reach_a_signed_diff():
    cfg = _config()
    cfg["servers"][0]["alternates"][0]["argv_delta"] = [{"flag": "-b", "reason": "no value"}]
    with pytest.raises(scr.ServingAlternatesError, match="takes a value"):
        scr.build_registry_diff(cfg, _master())
    cfg["servers"][0]["alternates"][0]["argv_delta"] = [{"flag": "-ub", "value": "1024|2048", "reason": "sweep"}]
    with pytest.raises(scr.ServingAlternatesError, match="'\\|'"):
        scr.build_registry_diff(cfg, _master())


@needs_real
def test_a_fabricated_sha_with_a_missing_result_json_is_refused_strict(tmp_path: Path):
    """Defect 5: the audit's fabricated-sha probe. Callers validate strict by default."""
    doc = json.loads(REAL_C22.read_text())
    prov = doc["servers"][0]["optimal"]["provenance"]
    prov["result_json"], prov["result_sha256"] = "/nonexistent/fabricated/RESULT.json", "f" * 64
    p = tmp_path / "fab.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(scr.ServingConfigError, match="not present on this host"):
        scr.load_serving_config(p, validator=ROOT_VALIDATOR)
    # the lax mode is opt-in only, and is the behaviour the audit caught
    assert scr.load_serving_config(p, validator=ROOT_VALIDATOR, strict_files=False)["champion"] == doc["champion"]


def test_apply_patches_a_copy_and_keeps_pending_and_siblings():
    master = _master()
    before = copy.deepcopy(master)
    diff = scr.build_registry_diff(_config(), master)
    patched = scr.apply_registry_diff(master, diff)
    assert master == before  # the input is never mutated
    fd = patched["server_mode"]["frontdoor"]
    assert fd["slots"] == 6 and fd["ctx_checkpoints"] == 16
    assert fd["serving_shape"]["n_ctx"] == 131072
    assert fd["serving_shape"]["kv_quant"] == {"k": "q4_0", "v": "q8_0"}
    assert fd["serving_shape"]["slots_by_shape"] == {"full": 4}  # untouched sibling
    # measured latency written, stale removed, PENDING throughput's existing entry left alone
    assert set(fd["serving_shape"]["alternates"]) == {"latency", "throughput"}
    assert patched["server_mode"]["architect_general"] == before["server_mode"]["architect_general"]
    assert patched["server_mode"]["architect_critic"] == {"slots": 4}


def test_apply_refuses_a_master_that_drifted_from_the_signed_diff():
    diff = scr.build_registry_diff(_config(), _master())
    drifted = _master()
    drifted["server_mode"]["frontdoor"]["slots"] = 5
    with pytest.raises(scr.ServingConfigError, match="drifted"):
        scr.apply_registry_diff(drifted, diff)


def test_render_diff_is_one_readable_line_per_op_plus_skips():
    lines = scr.render_diff(scr.build_registry_diff(_config(), _master()))
    assert "server_mode.frontdoor.slots: 4 -> 6" in lines
    assert any(line.endswith("alternates.latency: <absent> -> " + json.dumps({"-ub": 256, "--no-warmup": True}))
               for line in lines)
    assert any(line.startswith("SKIPPED (pending, never written) architect_general.optimal") for line in lines)


# ----------------------------------------------------------------------- dry run


def _tiny_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "orchestration").mkdir(parents=True)
    (repo / "orchestration" / "model_registry.yaml").write_text("live: lean\n")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    return repo


def _stub(exit_codes, seen):
    codes = iter(exit_codes)

    def run(cmd, **kw):
        seen.append((cmd, kw["cwd"]))
        return subprocess.CompletedProcess(cmd, next(codes), stdout="", stderr="")

    return run


def _master_file(tmp_path: Path) -> Path:
    p = tmp_path / "master.yaml"
    p.write_text(yaml.safe_dump(_master()))
    return p


def test_dry_run_applies_to_a_copy_and_runs_update_then_check(tmp_path: Path):
    repo, master = _tiny_repo(tmp_path), _master_file(tmp_path)
    live_master_before = master.read_text()
    diff = scr.build_registry_diff(_config(), _master())
    seen: list = []
    rec = scr.dry_run(diff, master, repo_root=repo, scratch=tmp_path / "scratch",
                      python="py", runner=_stub([0, 0], seen))
    assert rec["status"] == "pass" and rec["preflight_exit"] == 0
    assert rec["host_state_carve_out"]["failures_excluded"] == []  # present even when nothing was excluded
    assert [c[0][2:3] for c in seen] == [["update"], ["check"]]  # `check` is the verdict step
    assert all(str(tmp_path / "scratch" / "repo") == str(cwd) for _, cwd in seen)
    patched = yaml.safe_load(Path(rec["patched_master"]).read_text())
    assert patched["server_mode"]["frontdoor"]["slots"] == 6
    assert master.read_text() == live_master_before  # the input master is never written
    assert (tmp_path / "scratch" / "repo" / "orchestration" / "model_registry.yaml").exists()
    assert rec["live_registry_untouched"] is True
    assert rec["repo_commit"]


def test_dry_run_fails_when_check_fails(tmp_path: Path):
    repo, master = _tiny_repo(tmp_path), _master_file(tmp_path)
    diff = scr.build_registry_diff(_config(), _master())
    rec = scr.dry_run(diff, master, repo_root=repo, scratch=tmp_path / "s",
                      python="py", runner=_stub([0, 1], []))
    assert rec["status"] == "fail" and rec["preflight_exit"] == 1


def test_dry_run_excludes_host_state_failures_but_not_registry_ones(tmp_path: Path):
    repo, master = _tiny_repo(tmp_path), _master_file(tmp_path)
    diff = scr.build_registry_diff(_config(), _master())

    def run_with(report: str):
        def run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 1, stdout=report, stderr="")
        return run

    host = "lean_registry: ok\nruntime_attestation: failed\n  error: live process drift: x\nsummary: failed\n"
    rec = scr.dry_run(diff, master, repo_root=repo, scratch=tmp_path / "a", python="py", runner=run_with(host))
    assert rec["status"] == "pass" and rec["host_state_failures"] == ["runtime_attestation"]
    carve = rec["host_state_carve_out"]  # recorded explicitly, with what it excludes and why
    assert carve["failures_excluded"] == ["runtime_attestation"]
    assert set(carve["excluded_steps"]) == {"runtime_attestation", "declared_env_attestation"}
    assert "live host" in carve["rationale"]

    bad = host + "serving_shape_capacity: failed\n  error: too big\n"
    rec = scr.dry_run(diff, master, repo_root=repo, scratch=tmp_path / "b", python="py", runner=run_with(bad))
    assert rec["status"] == "fail"
    assert "serving_shape_capacity" in rec["steps"][-1]["failed_steps"]


def test_dry_run_reports_a_drifted_master_as_failure(tmp_path: Path):
    repo = _tiny_repo(tmp_path)
    drifted = _master()
    drifted["server_mode"]["frontdoor"]["slots"] = 5
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(drifted))
    diff = scr.build_registry_diff(_config(), _master())
    rec = scr.dry_run(diff, p, repo_root=repo, scratch=tmp_path / "s", python="py", runner=_stub([0, 0], []))
    assert rec["status"] == "fail" and "drifted" in rec["error"]


def test_dry_run_preflight_cmd_seam(tmp_path: Path, monkeypatch):
    repo, master = _tiny_repo(tmp_path), _master_file(tmp_path)
    diff = scr.build_registry_diff(_config(), _master())
    monkeypatch.setenv(scr.PREFLIGHT_CMD_ENV, "true")
    assert scr.dry_run(diff, master, repo_root=repo, scratch=tmp_path / "t")["status"] == "pass"
    monkeypatch.setenv(scr.PREFLIGHT_CMD_ENV, "false")
    assert scr.dry_run(diff, master, repo_root=repo, scratch=tmp_path / "f")["status"] == "fail"


def test_failed_steps_parser_ignores_derived_rows_and_indented_lines():
    text = "a: ok\nb: failed\n  error: b: failed\nsummary: failed\nacceptance: blocked\n"
    assert scr.failed_steps(text) == ["b"]


def test_package_sections_cli(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(_config()))
    master = _master_file(tmp_path)
    sec = scr.build_package_sections(cfg, master, run_dry=False)
    assert sec["serving_config"]["champion"] == "fixture-c22"
    assert len(sec["serving_config"]["sha256"]) == 64
    assert sec["registry_diff"]["summary"]["ops"] == 6
    assert sec["registry_diff"]["summary"]["skipped_pending"] == 2
    assert sec["registry_dry_run"]["status"] == "skipped"
