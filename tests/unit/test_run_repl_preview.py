"""D-RI-PREVIEW-1 driver: reply execution, guards, mock end-to-end (no model)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_R = Path(__file__).resolve().parents[2]
_S = importlib.util.spec_from_file_location("run_repl_preview_t", _R / "scripts" / "benchmark" / "run_repl_preview.py")
M = importlib.util.module_from_spec(_S); sys.modules["run_repl_preview_t"] = M; _S.loader.exec_module(M)
FX = M.fx_mod.build_fixture()


def test_reply_forms():
    c = FX["expected"]["correct"]
    assert M.run_reply("```python\nFINAL(sum(-x['qty'] if x['tag']=='R' else x['qty'] for x in batch))\n```", FX["batch"]) == (c, "code")
    assert M.run_reply("The net is 5 or rather 123", FX["batch"]) == (123, "text_int")
    assert M.run_reply("```python\nraise ValueError('x')\n```", FX["batch"])[0] is None
    assert M.run_reply("```python\nx = 1\n```", FX["batch"])[0] is None          # no FINAL
    assert M.run_reply("```python\nwhile True: pass\n```", FX["batch"], timeout_s=1)[1].startswith("exec_failed")


def test_prompt_contains_rendering_only_not_batch_dump():
    p = M.build_prompt(FX, "bounded_k3")
    assert "len=50" in p and "ord-025" not in p


def test_guards(tmp_path):
    for args, msg in ([["--out", str(tmp_path / "o")], "WINDOW_CONFIRMED=1"],):
        r = subprocess.run([sys.executable, str(_R / "scripts/benchmark/run_repl_preview.py"), *args], capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
        assert r.returncode != 0 and msg in r.stderr
    with pytest.raises(SystemExit, match="production stack port"):
        M.prove_port_ownership("http://127.0.0.1:8070", 1, {8070})
    with pytest.raises(SystemExit):
        M.parse_role("bad")


def test_mock_end_to_end(tmp_path):
    out = tmp_path / "r.json"
    assert M.main(["--mock", "--n", "4", "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    cnt = r["counts"]
    assert cnt["mock_role|bounded_k3"]["correct"] == 4
    assert cnt["mock_role|current"]["retyping_loss"] + cnt["mock_role|current"]["other"] == 4
    assert cnt["mock_role|half_first25"]["blind_rule_fit"] == 2 and cnt["mock_role|half_first25"]["retyping_loss"] == 2
    assert r["mock"] is True


def test_resume_and_stop(tmp_path):
    out, cells = tmp_path / "r.json", tmp_path / "cells"
    stop = tmp_path / "STOP"
    stop.write_text("")
    assert M.main(["--mock", "--n", "4", "--cell-dir", str(cells), "--stop-file", str(stop), "--out", str(out)]) == 75
    assert json.loads(out.read_text())["stopped"] == "stop-file"
    stop.unlink()
    assert M.main(["--mock", "--n", "4", "--cell-dir", str(cells), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["complete"] is True and all(c["n"] == 4 for c in r["counts"].values())
    # a second pass over a finished cell dir adds no rows
    assert M.main(["--mock", "--n", "4", "--cell-dir", str(cells), "--out", str(out)]) == 0
    assert sum(len(p.read_text().splitlines()) for p in cells.glob("*.jsonl")) == 12


def _complete_result(tmp_path, mock=False):
    out = tmp_path / "r.json"
    assert M.main(["--mock", "--n", "4", "--no-belief", "--out", str(out)]) == 0
    r = json.loads(out.read_text()); r["mock"] = mock; out.write_text(json.dumps(r))
    return out, r


def test_belief_skipped_for_mock_and_partial(tmp_path):
    out, r = _complete_result(tmp_path, mock=True)
    st = M.belief_capture(r, out, [("mock_role", None, None)], {}, M.fx_mod.build_fixture(), "rid", None)
    assert st["written"] is None and st["skipped"] == "mock result" and not (tmp_path / "repl_preview.beliefs.jsonl").exists()
    out, r = _complete_result(tmp_path)
    r["complete"] = False
    st = M.belief_capture(r, out, [("mock_role", None, None)], {}, M.fx_mod.build_fixture(), "rid", None)
    assert "partial" in st["skipped"]


def test_belief_written_for_complete_nonmock(tmp_path):
    from pathlib import Path as P
    root = P("/mnt/raid0/llm/worktrees/root-pbe-3c4ca599")
    if not (root / "scripts/vidya/adapters/repl_preview.py").is_file():
        pytest.skip("repl_preview adapter not present")
    out, r = _complete_result(tmp_path)
    st = M.belief_capture(r, out, [("mock_role", None, None)], {"mock_role": "/m/x.gguf"}, M.fx_mod.build_fixture(),
                          "rid", str(root))
    assert st["error"] is None and st["written"]
    rows = [json.loads(x) for x in (tmp_path / "repl_preview.beliefs.jsonl").read_text().splitlines()]
    assert rows and all(x["extra"]["gguf_path"] == "/m/x.gguf" and len(x["extra"]["fixture_sha256"]) == 64 for x in rows)


def test_belief_missing_gguf_is_recorded_not_raised(tmp_path):
    out, r = _complete_result(tmp_path)
    st = M.belief_capture(r, out, [("mock_role", None, None)], {}, M.fx_mod.build_fixture(), "rid",
                          "/mnt/raid0/llm/worktrees/root-pbe-3c4ca599")
    assert st["written"] is None and "no GGUF" in (st["error"] or "")
