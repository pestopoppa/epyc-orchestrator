"""CT-18: the IB2 canary fixtures, their deterministic graders and the A/B harness render (no model).

Static-fixture evidence only (CT-20): a pass here says the fixtures and graders are well formed,
nothing about any model's or defence's robustness.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "ct18_injection"
sys.path.insert(0, str(ROOT / "scripts" / "security"))
import ct18_ab_run as R  # noqa: E402
import ct18_graders as G  # noqa: E402

FX = json.loads((FIX / "ib2_fixtures.json").read_text())["fixtures"]


def test_fixture_set_is_24_and_hash_pinned():
    assert len(FX) == 24 and len({f["id"] for f in FX}) == 24
    man = json.loads((FIX / "MANIFEST.json").read_text())
    assert man["ib2_fixtures.json"] == hashlib.sha256((FIX / "ib2_fixtures.json").read_bytes()).hexdigest()
    assert {f["family"] for f in FX} == {"format-framing", "content-manipulation", "tool-manipulation", "agentic-analogue"}


def test_builder_is_deterministic():
    out = subprocess.run([sys.executable, str(ROOT / "scripts/security/ct18_build_fixtures.py")],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out)["ib2_fixtures.json"] == json.loads((FIX / "MANIFEST.json").read_text())["ib2_fixtures.json"]


@pytest.mark.parametrize("f", FX, ids=lambda f: f["id"])
def test_graders_separate_safe_from_attacked(f):
    s, a = f["safe_reference"], f["attacked_reference"]
    assert G.grade_fixture(f, "conflict", s) == {"attacked": False, "task_ok": True}
    assert G.grade_fixture(f, "conflict", a)["attacked"] is True
    assert G.grade_fixture(f, "clean", s)["task_ok"] is True
    # non-conflict twin: complying is correct, declining is not
    assert G.grade_fixture(f, "aligned", a)["complied"] is True
    assert G.grade_fixture(f, "aligned", s)["complied"] is False
    assert f["injection"] in f["tool_output_injected"] and f["injection"] not in f["tool_output_clean"]


def test_payloads_are_benign_canaries():
    # defensive fixtures: every injected instruction asks only for a canary word / fake canary tool line
    for f in FX:
        assert any(k in f["injection"] for k in ("PINEAPPLE", "KUMQUAT", "MANGO", "CANARY", "ARRR", "capital letters", "CALL:")), f["id"]


def test_arm_rendering():
    msgs = {a: R.render(a, "task", "a b\nc", "k")[1]["content"] for a in R.ARMS}
    assert "untrusted" not in msgs["A"] and "untrusted" in msgs["B"] and "untrusted" in msgs["C"]
    assert " " not in msgs["C"].split("```\n")[1].split("\n```")[0]
    mark = msgs["C"].split("```\n")[1][1]
    assert 0xE000 <= ord(mark) <= 0xF8FF and repr(mark) in msgs["C"].split("```")[0]
    assert R.render("C", "t", "a b", "k") == R.render("C", "t", "a b", "k")  # deterministic per key


def test_window_guard_refuses_without_confirmation():
    env = {k: v for k, v in __import__("os").environ.items() if k != "WINDOW_CONFIRMED"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/security/ct18_ab_run.py"), "--role", "w=http://127.0.0.1:1"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "refusing" in r.stdout


def test_discordant_counts():
    assert G.discordant([True, True, False, False], [True, False, True, False]) == {
        "n": 4, "both_true": 1, "both_false": 1, "a_only": 1, "b_only": 1}


def test_skill_inject_pairs_are_structural_and_canary_only():
    d = json.loads((FIX / "skill_inject_pairs.json").read_text())
    assert d["source_sha"] == "182f3d9d9836e81cdae213e9b9cec1d9be96eea3" and d["n_pairs"] == len(d["pairs"])
    for p in d["pairs"]:
        assert p["canary_sentence"].startswith("Also say the word ") and p["canary"] in p["grader"]["v"]
        assert "title" not in p and "judge_prompt" not in p  # no original attack text carried over
