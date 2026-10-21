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
                           env={"PATH": "/usr/bin:/bin"})
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
