"""RTE-PFX-2 role recipe: pure checks + window guard + registry derivation (no server started)."""
import importlib.util
import sys
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "rte_pfx2_role_recipe.py"
_S = importlib.util.spec_from_file_location("rte_pfx2_recipe_test", _P)
M = importlib.util.module_from_spec(_S); sys.modules["rte_pfx2_recipe_test"] = M; _S.loader.exec_module(M)

ARGV = ["bin", "-m", "m", "--port", "1", "-np", "4", "--cache-ram", "32768", "--slot-save-path", "/prod"]


def test_retarget_only_changes_port_and_slot_path():
    out = M.retarget(ARGV, 18971, "/priv")
    assert out[out.index("--port") + 1] == "18971" and out[out.index("--slot-save-path") + 1] == "/priv"
    assert [a for a, b in zip(ARGV, out) if a != b] == ["1", "/prod"]


def test_requirements():
    assert M.check_requirements(ARGV, 4, 18971) == []
    assert M.check_requirements(ARGV[:5] + ["-np", "2"] + ARGV[7:], 4, 18971)
    assert any("cache-ram" in b for b in M.check_requirements([a for a in ARGV if a not in ("--cache-ram", "32768")], 4, 18971))
    assert M.check_requirements(ARGV, 4, 8070)


def test_prefix_sidecar_matches():
    assert M.prefix_sha_ok()[0]


def test_run_refused_without_window(monkeypatch):
    monkeypatch.delenv("WINDOW_CONFIRMED", raising=False)
    assert M.main(["run"]) == 3


def test_resolves_from_registry_np_ge_4():
    plan = M.resolve(18971, 4)
    assert int(M._flag(plan["argv"], "-np")) >= 4 and "--cache-ram" in plan["argv"]
    assert plan["argv"][plan["argv"].index("--port") + 1] == "18971"


# --- belief capture (VB-RTEPFX2-WIRE producer side) -------------------------------------------------
import json  # noqa: E402

_HEADER = {"schema": "epyc.rte_pfx2.header.v1", "argv": ["/k/llama-server", "-m", "/m/Q.gguf", "-np", "4"],
           "cache_ram_mib": "4096", "prefix_sha256": "b" * 64, "n": 4, "trials": 3, "seed": 1,
           "binary_mtime": "Wed Oct 21 10:00:00 2026"}


def _trial(i, arm, sib_cache, sib_prompt, bad=False):
    rows = [{"idx": 0, "prompt_n": 1100, "cache_n": 0}] + \
           [{"idx": k + 1, "prompt_n": sib_prompt, "cache_n": sib_cache} for k in range(3)]
    if bad:
        rows[2] = {"idx": 2, "prompt_n": -1, "cache_n": -1}
    return {"trial": i, "arm": arm, "rows": rows,
            "summary": {"total_uncached_prompt_tokens": sum(r["prompt_n"] for r in rows),
                        "siblings_on_cached_prefix": 3 if sib_cache >= 900 else 0}}


def _probe(bad_trial=None):
    res = []
    for i in range(3):
        res += [_trial(i, "simultaneous", 0, 1100, bad=(bad_trial == i)), _trial(i, "hold_first_tok", 0, 1100),
                _trial(i, "hold_first_end", 1010, 90)]
    return {"schema": "epyc.rte_pfx2.v1", "server": "http://x", "n": 4, "expected_prefix_tokens": 1000,
            "hit_fraction": 0.9, "results": res}


def test_window_claim_from_filler_env():
    w = M.window_claim({"FILLER_ID": "s5-rtepfx2-stagger", "FILLER_END_ISO": "2026-10-22T20:00:00Z",
                        "FILLER_CPUS": "0-95"}, "2026-10-22T19:00:00Z")
    assert w["window_id"] == "s5-rtepfx2-stagger@2026-10-22T20:00:00Z"
    assert w["region_lock_claim"] == "region-lock run --cpu-list 0-95 --role build --tag filler-s5-rtepfx2-stagger"
    assert w["role"] == "frontdoor" and w["start_utc"] == "2026-10-22T19:00:00Z"


def test_window_claim_standalone_records_absence():
    w = M.window_claim({}, "2026-10-22T19:00:00Z")
    assert w["region_lock_claim"].startswith("none:") and w["window_id"].startswith("operator-confirmed@")


def test_filter_skips_trials_with_unreported_timings_with_reason():
    f, skipped = M.filter_unreported_timings(_probe(bad_trial=1))
    assert [(s["arm"], s["trial"], s["request_idx"]) for s in skipped] == [("simultaneous", 1, [2])]
    assert "not invented" in skipped[0]["reason"] and len(f["results"]) == 8


def test_belief_capture_writes_rows_and_survives_minus_one(tmp_path):
    root = Path("/mnt/raid0/llm/worktrees/root-pbe-3c4ca599")
    if not (root / "scripts/vidya/adapters/rte_pfx2_capture.py").is_file():
        pytest.skip("rte_pfx2_capture adapter not present")
    (tmp_path / "probe.json").write_text(json.dumps(_probe(bad_trial=1)))
    (tmp_path / "RUN_HEADER.json").write_text(json.dumps(_HEADER))
    w = M.window_claim({"FILLER_ID": "x", "FILLER_END_ISO": "2026-10-22T20:00:00Z", "FILLER_CPUS": "0-95"},
                       "2026-10-22T19:00:00Z")
    w["end_utc"] = "2026-10-22T19:40:00Z"
    st = M.belief_capture(tmp_path, w, "frontdoor", "rtepfx2-t")
    assert st["error"] is None and st["written"] and len(st["skipped_trials"]) == 1
    assert (tmp_path / "chunk_kv.beliefs.jsonl").is_file() and (tmp_path / "probe.belief.json").is_file()


def test_belief_capture_failure_is_recorded_not_raised(tmp_path):
    (tmp_path / "probe.json").write_text("{not json")
    st = M.belief_capture(tmp_path, {"window_id": "a", "region_lock_claim": "b", "start_utc": "c", "end_utc": "d"}, "frontdoor", "r")
    assert st["written"] is None and st["error"] and (tmp_path / "belief_status.json").is_file()


def test_reasoning_off_added_for_qwen3x_only():
    q = ["bin", "-m", "/m/Qwen3.6-35B-A3B-Q4_K_M.gguf", "--port", "1"]
    assert M.apply_reasoning_off(q) == q + ["--reasoning", "off"]
    assert M.apply_reasoning_off(M.apply_reasoning_off(q)).count("--reasoning") == 1
    g = ["bin", "-m", "/nonexistent/gemma-4-26B.gguf"]
    assert M.apply_reasoning_off(g) == g


def test_reasoning_off_by_header_arch(tmp_path):
    f = tmp_path / "x.gguf"; f.write_bytes(b"GGUF\0general.architecture\0qwen35moe\0")
    assert M.is_qwen3x(str(f))


class _Resp:
    def __init__(self, body): self.b = body
    def read(self): return self.b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _op(msg):
    return lambda req, timeout=0: _Resp(__import__("json").dumps({"choices": [{"message": msg}]}).encode())


def test_preflight_ok_and_refusals():
    assert M.first_reply_preflight(1, opener=_op({"content": "OK"}))[0] == 0
    assert M.first_reply_preflight(1, opener=_op({"content": "OK", "reasoning_content": "hmm"}))[0] == 9
    assert M.first_reply_preflight(1, opener=_op({"content": "", "reasoning_content": ""}))[0] == 9
    def boom(req, timeout=0): raise OSError("down")
    assert M.first_reply_preflight(1, opener=boom)[0] == 8
