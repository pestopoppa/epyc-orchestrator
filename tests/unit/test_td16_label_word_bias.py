"""TD-16 driver: question construction for every arm, metrics, paired bootstrap, resume, stop, void guard (stub runner, no model)."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

_R = Path(__file__).resolve().parents[2]
_S = importlib.util.spec_from_file_location("td16_t", _R / "scripts" / "benchmark" / "td16_label_word_bias.py")
T = importlib.util.module_from_spec(_S); sys.modules["td16_t"] = T; _S.loader.exec_module(T)
FIX = Path("/mnt/raid0/llm/epyc-root/research/fixtures/td16-hunch-20261102/td16-lookalikes.jsonl")
needs_fixture = pytest.mark.skipif(not FIX.exists(), reason="TD-16 fixture not present")


def _items(n=24):
    out = []
    for k in range(n):
        out.append({"id": f"i{k:03d}", "label": 1 if k % 3 == 0 else 0, "task": "t", "kind": "k",
                    "context": {"old": f"o{k}", "new": f"n{k}"},
                    "check": {"kind": "yesno", "question": "Does new replace old?", "yes_if": "changed value", "no_if": "same value"}})
    return out


def _stub(bias=0.0, fail_ids=()):
    """Perfect-ish oracle: P(yes)=0.9/0.1 by label; C arms read the key that means yes; `bias` shifts noul P(yes)."""
    def run(primitives, *, state, questions, role, mode):
        decs = []
        for q in questions:
            if q.id in fail_ids:
                continue
            it = primitives[q.id]
            py = min(1.0, max(0.0, (0.9 if it["label"] else 0.1) + (bias if q.kind.value == "noul" else 0.0)))
            if mode == "json":
                decs.append(NS(question_id=q.id, kind=q.kind, value=py >= 0.5, probabilities={}))
            elif q.kind.value == "noul":
                decs.append(NS(question_id=q.id, kind=q.kind, value=py >= 0.5, probabilities={"true": py, "false": 1 - py}))
            else:
                yes_key = "A" if q.option_descriptions[0] in ("changed value", "Yes") else "B"
                no_key = "B" if yes_key == "A" else "A"
                decs.append(NS(question_id=q.id, kind=q.kind, value=yes_key, probabilities={yes_key: py, no_key: 1 - py}))
        return NS(decisions=tuple(decs), failures=(), elapsed_ms=1.0, prompt_sha256="x")
    return run


class _Prim(dict):
    pass


def _go(tmp, items, run, extra=()):
    fx = tmp / "fx.jsonl"
    fx.write_text("\n".join(json.dumps(i) for i in items) + "\n")
    prim = _Prim({i["id"]: i for i in items})
    rc = T.main(["--fixture", str(fx), "--no-sha-check", "--batch", "4", "--boot-b", "50", "--results", str(tmp / "u.jsonl"),
                 "--out", str(tmp / "r.json"), *extra], primitives=prim, run_fn=run)
    return rc, json.loads((tmp / "r.json").read_text())


@needs_fixture
def test_real_fixture_builds_every_arm():
    items = T.load_items(FIX)
    assert len(items) == 240 and sum(i["label"] for i in items) == 80
    for arm in T.ARMS:
        qs = [T.make_question(i, arm) for i in items]
        assert len({q.id for q in qs}) == 240
    q = T.make_question(items[0], "C_BA")
    assert q.options == ("A", "B") and q.option_descriptions[0] == items[0]["check"]["no_if"]
    assert T.yes_key("C_BA") == "B" and T.yes_key("C_AB") == "A" and T.yes_key("N") is None
    # the conditions ride in the text (native rendering drops Question.criteria); N differs from Nq in the rendered prompt
    assert items[0]["check"]["yes_if"] in T.make_question(items[0], "N").text
    assert items[0]["check"]["yes_if"] not in T.make_question(items[0], "Nq").text
    assert T.check_arm_prompts(T.ARMS, T.batches(items, 8)[0]) == []


@needs_fixture
def test_fixture_sha_guard(tmp_path):
    bad = tmp_path / "f.jsonl"; bad.write_text(FIX.read_text() + "\n")
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        T.load_items(bad)


def test_metrics_known_values():
    assert T.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert T.auroc([0.5, 0.5], [1, 0]) == 0.5 and T.auroc([0.5], [1]) is None
    assert T.brier([1.0, 0.0], [1, 0]) == 0.0 and T.brier([0.5, 0.5], [1, 0]) == 0.25
    assert T.ece([0.9, 0.9], [1, 1]) == pytest.approx(0.1) and T.accuracy([0.6, 0.4], [1, 0]) == 1.0


def test_batches_are_deterministic_and_cover_all():
    it = _items(24)
    a, b = T.batches(it, 8), T.batches(list(reversed(it)), 8)
    assert [[x["id"] for x in bb] for bb in a] == [[x["id"] for x in bb] for bb in b]
    assert sorted(x["id"] for bb in a for x in bb) == sorted(x["id"] for x in it)


def test_end_to_end_unbiased_and_biased(tmp_path):
    rc, r = _go(tmp_path, _items(), _stub())
    assert rc == 0 and r["valid"] is True and r["complete"] is True
    ar = r["analysis"]["arms"]
    assert ar["N"]["auroc"] == 1.0 and ar["C_avg"]["auroc"] == 1.0 and ar["C_avg"]["order_disagreement"]["rate"] == 0.0
    assert ar["J"]["accuracy"] == 1.0
    assert r["analysis"]["baselines"]["always_no_accuracy"] == pytest.approx(16 / 24)
    assert "C_avg_vs_N" in r["analysis"]["paired"] and "Cq_avg_vs_Nq" in r["analysis"]["paired"]


def test_noul_bias_shows_in_paired_diff(tmp_path):
    rc, r = _go(tmp_path, _items(60), _stub(bias=-0.45))   # noul P(yes) pushed to ~0.45/0: loses accuracy on positives
    p = r["analysis"]["paired"]["C_avg_vs_N"]
    assert r["analysis"]["arms"]["N"]["accuracy"] < r["analysis"]["arms"]["C_avg"]["accuracy"]
    assert p["accuracy_diff_C_avg_minus_N"]["point"] > 0


def test_resume_skips_done_units_and_stop_file(tmp_path):
    stop = tmp_path / "STOP"; stop.write_text("")
    rc, r = _go(tmp_path, _items(), _stub(), ["--stop-file", str(stop)])
    assert rc == 75 and r["complete"] is False and r["stopped"] == "stop-file"
    stop.unlink()
    rc, r = _go(tmp_path, _items(), _stub())
    assert rc == 0 and r["complete"] is True
    n = len((tmp_path / "u.jsonl").read_text().splitlines())
    rc, r = _go(tmp_path, _items(), _stub())
    assert rc == 0 and len((tmp_path / "u.jsonl").read_text().splitlines()) == n == 7 * 6


def test_void_guard_stops_early(tmp_path):
    its = _items()
    rc, r = _go(tmp_path, its, _stub(fail_ids={i["id"] for i in its}))
    assert rc == 3 and r["valid"] is False and r["void_arms"] == ["N"] and r["stopped"] == "invalid-arm"
    assert len((tmp_path / "u.jsonl").read_text().splitlines()) <= 2     # stops after the unrecoverable batch


def test_live_mode_refuses_without_window_and_url(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("WINDOW_CONFIRMED", raising=False)
    fx = tmp_path / "fx.jsonl"; fx.write_text("\n".join(json.dumps(i) for i in _items()) + "\n")
    base = ["--fixture", str(fx), "--no-sha-check", "--results", str(tmp_path / "u"), "--out", str(tmp_path / "o")]
    assert T.main(base) == 2 and "WINDOW_CONFIRMED" in capsys.readouterr().err
    monkeypatch.setenv("WINDOW_CONFIRMED", "1")
    assert T.main(base) == 2 and "--server-url" in capsys.readouterr().err


def test_smoke_units_are_quarantined_by_a_live_run(tmp_path):
    its = _items(8)
    fx = tmp_path / "fx.jsonl"; fx.write_text("\n".join(json.dumps(i) for i in its) + "\n")
    prim = _Prim({i["id"]: i for i in its})
    base = ["--fixture", str(fx), "--no-sha-check", "--batch", "4", "--boot-b", "20", "--results", str(tmp_path / "u.jsonl"),
            "--out", str(tmp_path / "r.json"), "--arms", "N"]
    stop = tmp_path / "STOP"
    # smoke: runs one unit then is stopped by the keeper's stop file
    calls = {"n": 0}
    stub = _stub()

    def once(*a, **k):
        calls["n"] += 1
        stop.write_text("")
        return stub(*a, **k)
    assert T.main(base + ["--smoke", "--stop-file", str(stop)], primitives=prim, run_fn=once) == 75
    assert json.loads((tmp_path / "u.jsonl").read_text().splitlines()[0])["smoke"] is True
    stop.unlink()
    # live: the smoke unit is NOT reused, it is moved aside; all units are re-run live
    assert T.main(base, primitives=prim, run_fn=stub) == 0
    live = [json.loads(x) for x in (tmp_path / "u.jsonl").read_text().splitlines()]
    assert len(live) == 2 and not any(u.get("smoke") for u in live)
    assert list(tmp_path.glob("u.jsonl.smoke-*.jsonl"))


# ---- 2026-11-09 rerun fixes: N and Nq were byte-identical because native rendering drops Question.criteria
def test_native_arms_render_different_prompts_and_conditions_are_visible():
    items = _items(8)
    pn, pnq = T.render_arm_prompt("N", items), T.render_arm_prompt("Nq", items)
    assert pn != pnq
    assert all("YES if: changed value" in pn and "NO if: same value" in pn for _ in [0])
    assert "changed value" not in pnq and "same value" not in pnq
    assert "changed value" not in T.render_arm_prompt("Cq_AB", items)
    assert "changed value" in T.render_arm_prompt("C_AB", items)
    assert "changed value" in T.render_arm_prompt("J", items)
    assert T.check_arm_prompts(T.ARMS, items) == []


def test_guard_flags_a_renderer_that_drops_conditions(monkeypatch):
    import src.typed_decisions.native as nat
    real = nat.build_native_prompt

    def dropping(state, questions):  # simulates a renderer that ignores the conditions entirely
        out = real(state, questions)
        return "\n".join(l for l in out.splitlines() if "YES if:" not in l and "NO if:" not in l)
    monkeypatch.setattr(nat, "build_native_prompt", dropping)
    probs = T.check_arm_prompts(("N", "Nq"), _items(8))
    assert any("N: yes_if/no_if conditions missing" in p for p in probs)
    assert any("N and Nq render byte-identical" in p for p in probs)


def test_main_refuses_when_arms_render_identically(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "check_arm_prompts", lambda arms, batch: ["N and Nq render byte-identical prompts"])
    fx = tmp_path / "f.jsonl"
    fx.write_text("\n".join(json.dumps(i) for i in _items(8)) + "\n")
    rc = T.main(["--fixture", str(fx), "--no-sha-check", "--results", str(tmp_path / "r.jsonl"), "--out", str(tmp_path / "o.json")],
                primitives={}, run_fn=lambda *a, **k: None)
    assert rc == 2
