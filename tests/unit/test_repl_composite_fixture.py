"""D-RI-PREVIEW-1.2 zero-model pilot: the composite fixture must discriminate renderings IN PRINCIPLE (scripted policies, no model)."""
import importlib.util
import sys
from pathlib import Path

B = Path(__file__).resolve().parents[2] / "scripts" / "benchmark"


def _load(name):
    sp = importlib.util.spec_from_file_location(name, B / f"{name}.py")
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


fx = _load("repl_composite_fixture")
drv = _load("run_repl_composite")
N = 16


def _rate(mode, rend, n=N):
    ok = 0
    for i in range(n):
        ep = fx.build_episode(i)
        pl = fx.play_episode(ep, rend, fx.reference_policy(mode, ep["project"]))
        ok += fx.classify(pl["final"], ep)["cls"] == "correct"
    return ok / n


def test_deterministic_and_sized():
    assert fx.build_episode(3) == fx.build_episode(3)
    assert len(fx.build_episode(0)["docs"]) == 68
    assert fx.fixture_sha256(4) == fx.fixture_sha256(4)


def test_positions_cover_all_slots():
    for kind in fx.POS_COEF:
        assert {fx.build_episode(i)["positions"][kind] for i in range(16)} == set(range(16))


def test_full_history_solves():
    assert _rate("reader", "full") == 1.0


def test_truncated_history_fails_history_reader():
    assert _rate("reader", "current") == 0.0
    assert _rate("reader", "bounded_k3") <= 0.25
    assert _rate("reader", "half_first25") <= 0.25


def test_retyping_policy_robust_to_every_rendering():
    for r in fx.ALL_RENDERINGS:
        assert _rate("retype", r) == 1.0


def test_oracle_and_bad_final():
    ep = fx.build_episode(0)
    assert fx.classify(dict(ep["expected"]), ep)["cls"] == "correct"
    assert fx.classify(None, ep)["cls"] == "no_answer"
    bad = dict(ep["expected"], lead=ep["distractors"]["lead"])
    c = fx.classify(bad, ep)
    assert c["cls"] == "partial" and c["distractor_picks"] == ["lead"] and not c["early_ok"]
    pl = fx.play_episode(ep, "full", lambda p, t: "```python\nFINAL = 5\n```")
    assert pl["how"] == "no_final"


def test_mock_driver_paired_and_resumable(tmp_path):
    out = tmp_path / "r.json"
    rc = drv.main(["--mock", "--n", "3", "--looks", "", "--cell-dir", str(tmp_path / "c"), "--out", str(out)])
    assert rc == 0
    rc = drv.main(["--mock", "--n", "3", "--looks", "", "--cell-dir", str(tmp_path / "c"), "--out", str(out)])
    assert rc == 0


def test_void_and_look_rules():
    rows = lambda cls, how="final": [{"cls": cls, "how": how}] * 10
    assert drv.void_cells({"a|x": rows("no_answer", "bad_final")}, 30) == ["a|x"]
    assert drv.look_decision({"a|x": rows("correct"), "a|y": rows("correct")}, 10) == "FUTILITY_CEILING"
    assert drv.look_decision({"a|x": rows("wrong"), "a|y": rows("wrong")}, 10) == "FUTILITY_FLOOR"
    assert drv.look_decision({"a|x": rows("correct"), "a|y": rows("wrong")}, 10) is None
