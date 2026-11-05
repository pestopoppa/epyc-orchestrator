#!/usr/bin/env python3
"""V11-122: acceptance parity of the turn-1 GBNF vs its %llguidance Lark translation.

For both turn-1 shapes (legacy, call) the real ToolRegistry grammar is translated by
src.backends.llg_grammar.gbnf_to_llguidance. Every SPEC-GRAM corpus tool call is
rendered into turn-1 texts (tool call, FINAL, code line, plus negative mutations) and
each text is judged by BOTH grammars: GBNF by llama.cpp's ``test-gbnf-validator`` (the
stock grammar engine) and Lark by llguidance's LLMatcher (byte-level, via a Qwen
tokenizer). Parity = the two verdicts agree on every text. Needs a Python with
``llguidance`` and ``tokenizers``; runs no model.

Usage: python llg_gbnf_lark_parity.py --corpus corpus.jsonl --validator <dir with test-gbnf-validator>
       --tokenizer tokenizer.json [--limit N] --out result.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.backends.llg_grammar import gbnf_to_llguidance  # noqa: E402


def registry_grammars() -> dict[str, str]:
    from src.registry.tool_registry import Tool, ToolCategory, ToolRegistry

    reg = ToolRegistry()
    for name, param in (("run_shell", "command"), ("web_search", "query"), ("python_eval", "code")):
        reg.register_tool(Tool(name=name, description="d", category=ToolCategory.WEB, parameters={param: {"type": "string"}}))
    return {shape: reg.generate_gbnf_grammar(None, shape=shape) for shape in ("legacy", "call")}


def texts_for(cmd: str) -> list[str]:
    one = cmd.replace("\n", " ")
    out = [
        f'# run it\nTOOL("run_shell", command="{one}")\n',
        f'# run it\nres = CALL("run_shell", command="{one}")\n',
        f'# run it\nCALL("run_shell", command="{one}")\n',
        f'# done\nFINAL("{one[:40]}")\n',
        f"# step\n{one}\n",
        f'# run it\nTOOL("run_shell", command="{one}")',          # missing final newline
        f'# run it\nTOOL("rm_rf", command="{one}")\n',            # unknown tool
        f'TOOL("run_shell", command="{one}")\n',                  # missing thought
        f'# x\nres = CALL("web_search", query="{one[:30]}", n=3)\n',
    ]
    return out


def gbnf_accepts(validator: Path, gpath: str, text: str, tmpdir: str) -> bool:
    ipath = os.path.join(tmpdir, "in.txt")
    with open(ipath, "w", encoding="utf-8") as f:
        f.write(text)
    env = {**os.environ, "LD_LIBRARY_PATH": str(validator)}
    r = subprocess.run([str(validator / "test-gbnf-validator"), gpath, ipath], capture_output=True, text=True, env=env, timeout=30)
    if "is valid according" in r.stdout:
        return True
    if "is invalid according" in r.stdout:
        return False
    raise RuntimeError(f"validator gave no verdict: rc={r.returncode} {r.stdout[-200:]} {r.stderr[-200:]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--validator", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import llguidance as L
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(a.tokenizer)
    lt = L.LLTokenizer(open(a.tokenizer).read())
    cmds = []
    for line in open(a.corpus):
        if line.strip():
            inp = json.loads(line).get("input") or {}
            c = inp.get("command") or inp.get("query") or inp.get("code")
            if isinstance(c, str) and c:
                cmds.append(c)
    if a.limit:
        cmds = cmds[: a.limit]
    result = {"schema": "v11_122.gbnf_lark_parity.v1", "corpus": a.corpus, "n_cmds": len(cmds), "shapes": {}}
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(a.out))) as td:
        for shape, gbnf in registry_grammars().items():
            lark = gbnf_to_llguidance(gbnf)
            gpath = os.path.join(td, f"{shape}.gbnf")
            Path(gpath).write_text(gbnf)
            base = L.LLMatcher(lt, lark)
            if base.is_error():
                result["shapes"][shape] = {"error": base.get_error()[:400]}
                continue
            n = agree = acc = 0
            mism = []
            for cmd in cmds:
                for t in texts_for(cmd):
                    g = gbnf_accepts(a.validator, gpath, t, td)
                    m = base.deep_copy()
                    ok = m.consume_tokens(tk.encode(t, add_special_tokens=False).ids) and not m.is_error()
                    l_ = bool(ok and m.is_accepting())
                    n += 1
                    acc += g
                    agree += g == l_
                    if g != l_ and len(mism) < 20:
                        mism.append({"text": t[:300], "gbnf": g, "lark": l_})
            result["shapes"][shape] = {"texts": n, "gbnf_accepted": acc, "agree": agree, "disagree": n - agree, "mismatches": mism}
    result["verdict"] = "PASS" if all(s.get("disagree", 1) == 0 and not s.get("error") for s in result["shapes"].values()) else "FAIL"
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: (v if k != "shapes" else {s: {kk: vv for kk, vv in d.items() if kk != "mismatches"} for s, d in v.items()}) for k, v in result.items()}))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
