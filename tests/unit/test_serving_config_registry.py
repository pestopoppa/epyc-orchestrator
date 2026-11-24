"""serving-config artifact -> registry diff -> scratch dry-run (PROMOTION-REGISTRY-SERVING-1)."""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.registry import serving_config_registry as scr

PROV = {"row_id": "ROW-1", "result_json": "x/RESULT.json", "result_sha256": "c" * 64, "store": "kernels/production/cpu"}


def _config() -> dict:
    return {
        "schema": scr.CONFIG_SCHEMA,
        "champion": "fixture-c22",
        "roles": {
            "frontdoor": {
                "optimal": {
                    "server": {"slots": 6, "ctx_checkpoints": 16},
                    "serving_shape": {"n_ctx": 131072},
                    "provenance": dict(PROV),
                },
                "alternates": {
                    "latency": {"argv_delta": {"--parallel": 1}, "provenance": dict(PROV)},
                    "throughput": {"argv_delta": {"--parallel": 8}, "provenance": dict(PROV)},
                },
                "tradeoffs": [{"config": "optimal", "tps": 10.0}],
            }
        },
    }


def _master() -> dict:
    return {
        "server_mode": {
            "frontdoor": {
                "slots": 4,
                "ctx_checkpoints": 32,
                "serving_shape": {
                    "n_ctx": 262144,
                    "slots_by_shape": {"full": 4},
                    "alternates": {
                        "stale": {"argv_delta": {"x": 1}, "provenance": dict(PROV)},
                    },
                },
            },
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
        (lambda c: c.update(roles={}), "non-empty"),
        (lambda c: c["roles"]["frontdoor"].pop("optimal"), "optimal is required"),
        (lambda c: c["roles"]["frontdoor"]["optimal"].update(provenance={}), "provenance"),
        (lambda c: c["roles"]["frontdoor"].pop("tradeoffs"), "tradeoffs"),
        (lambda c: c["roles"]["frontdoor"]["optimal"]["serving_shape"].update(alternates={}), "alternates"),
        (lambda c: c["roles"]["frontdoor"]["alternates"]["latency"].pop("provenance"), "provenance"),
    ],
)
def test_malformed_config_is_rejected(mutate, needle):
    cfg = _config()
    mutate(cfg)
    with pytest.raises(scr.ServingConfigError, match=needle):
        scr.validate_serving_config(cfg)


# ----------------------------------------------------------------------- diff / apply


def test_diff_covers_knobs_shape_alternates_and_removes_stale():
    diff = scr.build_registry_diff(_config(), _master(), config_sha256="d" * 64)
    by_path = {".".join(op["path"]): op for op in diff["ops"]}
    assert by_path["server_mode.frontdoor.slots"]["old"] == 4
    assert by_path["server_mode.frontdoor.slots"]["new"] == 6
    assert by_path["server_mode.frontdoor.ctx_checkpoints"]["new"] == 16
    assert by_path["server_mode.frontdoor.serving_shape.n_ctx"]["new"] == 131072
    assert by_path["server_mode.frontdoor.serving_shape.alternates.latency"]["had_old"] is False
    assert by_path["server_mode.frontdoor.serving_shape.alternates.stale"]["op"] == "remove"
    assert diff["summary"] == {"ops": 6, "roles": {"frontdoor": {"knobs": 3, "alternates": 3}}}
    assert diff["serving_config_sha256"] == "d" * 64
    assert not any("architect_critic" in k for k in by_path)  # roles not in the artifact are untouched


def test_diff_skips_values_already_in_the_registry():
    cfg = _config()
    master = _master()
    master["server_mode"]["frontdoor"]["slots"] = 6
    diff = scr.build_registry_diff(cfg, master)
    assert "server_mode.frontdoor.slots" not in {".".join(o["path"]) for o in diff["ops"]}


def test_diff_refuses_a_role_the_registry_does_not_serve():
    cfg = _config()
    cfg["roles"]["ghost"] = cfg["roles"].pop("frontdoor")
    with pytest.raises(scr.ServingConfigError, match="ghost"):
        scr.build_registry_diff(cfg, _master())


def test_apply_patches_a_copy_and_keeps_alternates():
    master = _master()
    before = copy.deepcopy(master)
    diff = scr.build_registry_diff(_config(), master)
    patched = scr.apply_registry_diff(master, diff)
    assert master == before  # the input is never mutated
    fd = patched["server_mode"]["frontdoor"]
    assert fd["slots"] == 6 and fd["ctx_checkpoints"] == 16
    assert fd["serving_shape"]["n_ctx"] == 131072
    assert fd["serving_shape"]["slots_by_shape"] == {"full": 4}  # untouched sibling
    assert set(fd["serving_shape"]["alternates"]) == {"latency", "throughput"}
    assert patched["server_mode"]["architect_critic"] == {"slots": 4}


def test_apply_refuses_a_master_that_drifted_from_the_signed_diff():
    diff = scr.build_registry_diff(_config(), _master())
    drifted = _master()
    drifted["server_mode"]["frontdoor"]["slots"] = 5
    with pytest.raises(scr.ServingConfigError, match="drifted"):
        scr.apply_registry_diff(drifted, diff)


def test_render_diff_is_one_readable_line_per_op():
    lines = scr.render_diff(scr.build_registry_diff(_config(), _master()))
    assert "server_mode.frontdoor.slots: 4 -> 6" in lines
    assert any(line.endswith("alternates.latency: <absent> -> " + json.dumps({"--parallel": 1})) for line in lines)


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
    assert sec["registry_dry_run"]["status"] == "skipped"
