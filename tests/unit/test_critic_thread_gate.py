"""Pure parts of the DAR-LAT-3h gate driver: argv rewrite, env arms, THP expectation, decision rule."""

from __future__ import annotations

import pytest

from scripts.server import critic_thread_gate as g

LIVE = ["/k/llama-server", "-m", "/m.gguf", "--port", "8074", "-np", "1", "-t", "96",
        "--slot-save-path", "/cache/architect_critic", "--mlock"]


def test_rewrite_changes_only_threads_and_port_and_drops_slot_dir() -> None:
    out = g.rewrite_argv(LIVE, 48)
    assert out == ["/k/llama-server", "-m", "/m.gguf", "--port", "18074", "-np", "1", "-t", "48", "--mlock"]


def test_rewrite_refuses_argv_without_threads() -> None:
    with pytest.raises(ValueError):
        g.rewrite_argv(["/k/llama-server", "--port", "8074"], 48)


def test_arm_envs() -> None:
    live = {"OMP_PLACES": "cores", "GGML_IQK": "1", "GGML_FA_SPLIT_KV": "1"}
    assert g.arm_env(live, "L") == {"OMP_PLACES": "cores", "GGML_IQK": "1"}
    assert g.arm_env(live, "T48") == {"OMP_PLACES": "cores", "GGML_IQK": "1", "GGML_FUSED_DECODE_OFF": "1"}
    assert g.arm_env(live, "T96N") == {"OMP_PLACES": "cores", "GGML_IQK": "1", "GGML_FUSED_DECODE_OFF": "1",
                                       "GGML_NOHUGEPAGE_PROCESS": "1"}
    # FA_SPLIT_KV is dropped from the package (DAR-LAT-3i): no arm sets it.
    assert all("GGML_FA_SPLIT_KV" not in arm["env"] for arm in g.ARMS.values())


def test_thp_expectation() -> None:
    assert [g.expected_thp_enabled(a) for a in ("L", "T96", "T96N", "T48", "T48N")] == [1, 1, 0, 1, 0]


def test_schedule_is_balanced_blocks() -> None:
    assert sorted(g.SCHEDULE) == sorted(list(g.ARMS) * 3)
    for block in range(3):
        assert sorted(g.SCHEDULE[block * 5:block * 5 + 5]) == sorted(g.ARMS)


def _stats(wall: float, ttft: float = 100.0, coherent: int = 24) -> dict:
    return {"wall_s_median": wall, "ttft_ms_median": ttft, "coherent": coherent, "decode_tps_tokw": 30.0}


def _arms(L=10.0, T96=10.0, T96N=10.0, T48=10.0, T48N=10.0) -> dict:
    return {a: [_stats(w)] * 3 for a, w in {"L": L, "T96": T96, "T96N": T96N, "T48": T48, "T48N": T48N}.items()}


def test_parity_everywhere_gives_48_without_shim() -> None:
    assert g.decide(_arms())["outcome"] == "T48"


def test_48_needs_parity_at_both_shim_levels() -> None:
    assert g.decide(_arms(T48N=10.5))["outcome"] == "T96"


def test_48_slower_gives_96() -> None:
    assert g.decide(_arms(T48=10.5, T48N=10.5))["outcome"] == "T96"


def test_shim_adopted_only_on_a_real_gain_at_the_chosen_threads() -> None:
    assert g.decide(_arms(T48N=9.7, T96N=9.7))["outcome"] == "T48N"
    assert g.decide(_arms(T48N=9.9, T96N=9.9))["outcome"] == "T48"          # 1% is not enough
    assert g.decide(_arms(T48=10.5, T48N=10.5, T96N=9.7))["outcome"] == "T96N"


def test_shim_with_worse_ttft_is_not_adopted() -> None:
    arms = _arms()
    arms["T48N"] = [_stats(9.5, ttft=120.0)] * 3
    arms["T96N"] = [_stats(9.5, ttft=120.0)] * 3
    assert g.decide(arms)["outcome"] == "T48"


def test_fused_decode_off_not_inert_invalidates() -> None:
    assert g.decide(_arms(T96=10.5))["outcome"] == "INVALID-PREMISE"


def test_too_few_launches_is_inconclusive() -> None:
    arms = _arms()
    arms["T48N"] = arms["T48N"][:2]
    assert g.decide(arms)["outcome"] == "INCONCLUSIVE"


def test_coherence_regression_blocks_48() -> None:
    arms = _arms()
    arms["T48"] = [_stats(9.0, coherent=20)] * 3
    assert g.decide(arms)["outcome"] == "T96"


def test_coherent_classifier() -> None:
    assert g.coherent("The plan fails because step two assumes the cache is warm.", "stop")
    assert not g.coherent("", "stop")
    assert not g.coherent("a b c d " * 50, "length")
