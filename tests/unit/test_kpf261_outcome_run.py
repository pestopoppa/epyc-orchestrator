"""KPF-26.1 outcome runner: fake HTTP endpoint (no model, no server), decision rule table, guards, resume."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "trace"))
import kpf261_outcome_run as K  # noqa: E402


def _fake_server(answer_for):
    """Loopback server answering /v1/chat/completions; answer_for(prompt) -> str."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = body["messages"][0]["content"]
            seen.append(body)
            out = json.dumps({"choices": [{"message": {"content": answer_for(prompt)}, "finish_reason": "stop"}],
                              "timings": {"prompt_n": 10, "cache_n": 5}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def _oracle(waves):
    table = [(s["action"], re.sub(r"\\b|\(\?i\)", "", s["expect"])) for w in waves for s in w["steps"]]
    return lambda p: next((e for a, e in table if a in p), "?")


@pytest.fixture
def waves():
    return K.build_waves(n_waves=3, n_steps=6)


def test_build_waves_deterministic_persona_bearing():
    a, b = K.build_waves(), K.build_waves()
    assert a == b
    steps = [s for w in a for s in w["steps"]]
    assert sum(1 for s in steps if s["persona"]) * 2 >= len(steps)
    assert len({s["action"] for s in steps}) == len(steps)


def test_treatment_really_differs(waves):
    t = K.treatment_check(waves)
    assert t["applied"] and t["persona_moved_after_trunk"]
    for w in t["per_wave"].values():
        assert w["shared_chars_B"] > w["shared_chars_A"] + 100


def test_render_restores_env_and_matches_apply_persona_prefix(waves, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PREFIX_INDEX", "0")
    monkeypatch.delenv("ORCHESTRATOR_PREFIX_INDEX_FORK", raising=False)
    w = waves[0]
    s = next(s for s in w["steps"] if s["persona"])
    pa, _ = K.render(w, s, "A")
    assert os.environ["ORCHESTRATOR_PREFIX_INDEX"] == "0" and "ORCHESTRATOR_PREFIX_INDEX_FORK" not in os.environ
    from src.llm_primitives.persona import PersonaMixin
    from src.features import features as _f
    from unittest.mock import MagicMock, patch
    from src.parallel_step_executor import StepExecutor
    ex = StepExecutor(primitives=MagicMock())
    ex.step_outputs = dict(w["step_outputs"])
    raw = ex._build_step_prompt({"objective": w["objective"]}, s)
    with patch.object(_f(), "personas", True, create=True):
        class M(PersonaMixin):
            pass
        try:
            expect = M()._apply_persona_prefix(raw, s["persona"])
        except Exception:
            pytest.skip("features().personas not patchable here")
    assert pa == expect


def test_full_run_unchanged_and_resumable(waves, tmp_path):
    srv, seen = _fake_server(_oracle(waves))
    url = f"http://127.0.0.1:{srv.server_port}"
    s = K.run(waves, K.HttpClient(url), tmp_path, reps=3, concurrency=3)
    srv.shutdown()
    assert s["verdict"] == "UNCHANGED" and s["complete"] and s["n_cases"] == 54
    assert (tmp_path / "DONE").exists()
    assert all(b["temperature"] == 0.0 and b["chat_template_kwargs"] == {"enable_thinking": False} for b in seen)
    n = len((tmp_path / "cases.jsonl").read_text().splitlines())
    assert n == 3 * 3 * 18
    # resume: nothing re-sent
    srv2, seen2 = _fake_server(_oracle(waves))
    K.run(waves, K.HttpClient(f"http://127.0.0.1:{srv2.server_port}"), tmp_path, reps=3, concurrency=3)
    srv2.shutdown()
    assert seen2 == [] and len((tmp_path / "cases.jsonl").read_text().splitlines()) == n


def test_run_detects_a_persona_after_trunk_regression(waves, tmp_path):
    ora = _oracle(waves)
    persona_texts = [K._persona_text(p) for p in K.PERSONAS if p]
    # the fake model breaks exactly when the persona text sits AFTER the trunk (i.e. the prompt does not start with it)
    def ans(p):
        has = [t for t in persona_texts if t and t.strip() in p]
        if has and not p.startswith(has[0].strip()):
            return "garbled"
        return ora(p)
    srv, _ = _fake_server(ans)
    s = K.run(waves, K.HttpClient(f"http://127.0.0.1:{srv.server_port}"), tmp_path, reps=3, concurrency=3)
    srv.shutdown()
    assert s["verdict"] == "CHANGED" and s["reorder_decision"] == "KEEP_OFF" and s["regress"] > s["ctrl_regress"] == 0 and s["improve"] == 0


def test_stopfile_and_deadline_stop_between_waves(waves, tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("")
    s = K.run(waves, lambda p: {"answer": "", "finish": None, "elapsed_s": 0, "prompt_n": 0, "cache_n": 0},
              tmp_path / "o", reps=1, concurrency=1, stopfile=str(stop))
    assert s["stopped"] == "stopfile" and not s["complete"] and not (tmp_path / "o" / "DONE").exists()
    s = K.run(waves, lambda p: {}, tmp_path / "o2", reps=1, concurrency=1, deadline=0.0)
    assert s["stopped"] == "deadline"


def _case(rep, arm, w, st, ok, ans="x", err=None):
    return {"rep": rep, "arm": arm, "wave": w, "step": st, "pass": ok, "answer": ans, "error": err}


def _table(n, a, b, a2):
    cs = []
    for i in range(n):
        for arm, f in (("A", a), ("B", b), ("A2", a2)):
            cs.append(_case(0, arm, "w", f"s{i}", f(i)))
    return cs

APPLIED = {"applied": True}


@pytest.mark.parametrize("a,b,a2,verdict,decision", [
    (lambda i: True, lambda i: True, lambda i: True, "UNCHANGED", "KEEP_ON_ELIGIBLE"),
    # one B regression, but A vs A2 also differ by one case: the drop equals the spread
    (lambda i: True, lambda i: i != 0, lambda i: i != 1, "WITHIN_SPREAD", "KEEP_ON_ELIGIBLE"),
    # three regressions, control identical: drop 3/60 > spread 0
    (lambda i: True, lambda i: i > 2, lambda i: True, "CHANGED", "KEEP_OFF"),
    # flips both ways, net drop 0: rate unchanged
    (lambda i: i % 4 != 3, lambda i: i % 4 != 0, lambda i: i % 4 != 3, "WITHIN_SPREAD", "KEEP_ON_ELIGIBLE"),
    (lambda i: i % 4 == 0, lambda i: True, lambda i: i % 4 == 0, "INVALID", None),   # baseline pass 25% < 60%
])
def test_decide_table(a, b, a2, verdict, decision):
    d = K.decide(_table(60, a, b, a2), APPLIED, 6, 6)
    assert d["verdict"] == verdict and d["reorder_decision"] == decision, d


def test_b_errors_are_regressions_not_a_void():
    ok = lambda i: True  # noqa: E731
    cs = _table(60, ok, ok, ok)
    for c in cs:
        if c["arm"] == "B":
            c["pass"], c["error"] = False, "timeout"
    d = K.decide(cs, APPLIED, 6, 6)
    assert d["verdict"] == "CHANGED" and d["B_transport_errors"] == 60 and d["reorder_decision"] == "KEEP_OFF"


def test_decide_invalid_gates():
    ok = lambda i: True  # noqa: E731
    assert K.decide(_table(60, ok, ok, ok), {"applied": False}, 6, 6)["verdict"] == "INVALID"
    assert K.decide(_table(10, ok, ok, ok), APPLIED, 6, 6)["verdict"] == "INVALID"       # < MIN_CASES
    assert K.decide(_table(60, ok, ok, ok), APPLIED, 1, 6)["verdict"] == "INVALID"       # persona share
    cs = _table(60, ok, ok, ok)
    for c in cs:
        if c["arm"] == "A":
            c["error"] = "boom"
    assert K.decide(cs, APPLIED, 6, 6)["verdict"] == "INVALID"


def test_port_ownership(monkeypatch):
    monkeypatch.setattr(K, "listening_pid", lambda port: 4242)
    with pytest.raises(SystemExit, match="production stack port"):
        K.prove_port_ownership("http://127.0.0.1:8000", 4242)
    with pytest.raises(SystemExit, match="owned by pid 4242"):
        K.prove_port_ownership("http://127.0.0.1:18961", 1)
    with pytest.raises(SystemExit, match="loopback"):
        K.prove_port_ownership("http://10.0.0.5:18961", 4242)
    assert K.prove_port_ownership("http://127.0.0.1:18961", 4242) == 18961


def _cli(args, env_extra=None):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("WINDOW_CONFIRMED", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(ROOT / "scripts/trace/kpf261_outcome_run.py"), *args],
                          capture_output=True, text=True, env=env, timeout=300)


def test_cli_live_refuses_without_window_and_proof(tmp_path):
    r = _cli(["--out-dir", str(tmp_path)])
    assert r.returncode != 0 and "WINDOW_CONFIRMED=1" in r.stderr + r.stdout
    r = _cli(["--out-dir", str(tmp_path)], {"WINDOW_CONFIRMED": "1"})
    assert r.returncode != 0 and "--server-pid" in r.stderr + r.stdout


def test_cli_dry_run_and_mock(tmp_path):
    r = _cli(["--out-dir", str(tmp_path / "d"), "--dry-run"])
    assert r.returncode == 0 and '"treatment_applied": true' in r.stdout
    r = _cli(["--out-dir", str(tmp_path / "m"), "--mock-llm", "--n-waves", "3", "--reps", "3"])
    assert r.returncode == 0 and "UNCHANGED" in r.stdout and (tmp_path / "m" / "DONE").exists()


def test_manifest_frozen_and_mismatch_aborts(waves, tmp_path):
    ok = lambda p: {"answer": "", "finish": None, "elapsed_s": 0, "prompt_n": 0, "cache_n": 0}  # noqa: E731
    K.run(waves, ok, tmp_path, reps=1, concurrency=1, deadline=0.0, seed=261)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["n_tasks"] == 18 and len(m["tasks"][0]["sha256_A"]) == 64 and m["tasks"][0]["sha256_A"] != m["tasks"][0]["sha256_B"]
    K.run(waves, ok, tmp_path, reps=1, concurrency=1, deadline=0.0, seed=261)      # same prompts: resumes fine
    other = K.build_waves(n_waves=3, n_steps=6, seed=262)
    with pytest.raises(K.ManifestMismatch):
        K.run(other, ok, tmp_path, reps=1, concurrency=1, deadline=0.0, seed=262)


def test_seed_gives_distinct_waves():
    assert K.build_waves(2, 6, 261) != K.build_waves(2, 6, 262)


def test_final_answer_extraction_and_grade():
    st = {"expect": r"\b664\b"}
    assert K.grade(st, "Sum: 34+85=119 ... total 664.\nANSWER: 664")
    assert not K.grade(st, "work mentions 664 along the way\nANSWER: 613")         # intermediate value must not pass
    assert not K.grade(st, "x " * 100 + "664")                                      # long reply, no marker -> no answer
    assert K.grade(st, "664")                                                       # short bare reply (oracle form)
    assert K.grade({"expect": r"(?i)\boak\b"}, "Largest is oak.\nanswer: **Oak**")
    assert K.grade(st, "ANSWER: 1\nANSWER: 664")                                     # last marker wins


def test_prompts_ask_for_work_and_marker(waves):
    for w in waves:
        for s in w["steps"]:
            assert "ANSWER:" in s["action"]
