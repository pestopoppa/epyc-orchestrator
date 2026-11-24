#!/usr/bin/env python3
"""HAR-COMPACT-LEDGER-1 replay over RETAINED real trajectories (no inference).

Corpus: the TOC-SP-5 / BEP sandbox traces (``data/bep_sandbox/*/traces/*.jsonl``, ``INVALID-*`` skipped).
Each trace is one real agent trajectory (per turn: the model's tool-call text plus the prompt tail that
carries the previous tool output / errors). A pre-compaction context is rebuilt by accumulating, turn by
turn, the action text and the not-yet-seen lines of the prompt tail.

The real ``_maybe_compact_context`` runs with ``compaction_failed_action_ledger`` ON and a deterministic
stub for the WORKER_GENERAL summary call. Two stubs are exercised: ``lossy`` (generic one-line summary)
and ``partial`` (summary that keeps every third line of the externalized head, so some entries survive on
their own and the "already present" branch is exercised too).

ORACLE (independent of ``compaction.py``'s regexes): a lexical line classifier for failed actions and a
whitespace/punctuation-split token classifier for pointers. Neither imports or reuses the extractor's
patterns. Misses are classified afterwards: ``capped`` (the extractor would match it but its documented
bound dropped it) versus ``gap`` (the extractor never matches it).

Usage:
    PYTHONPATH=<repo> python scripts/har_compact_ledger_replay.py --out artifacts/har_compact_ledger_replay.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
# HAR_SRC_ROOT points the replay at another checkout's ``src`` (e.g. a pre-fix baseline).
sys.path.insert(0, os.environ.get("HAR_SRC_ROOT", str(REPO)))

DEFAULT_CORPUS = Path("/mnt/raid0/llm/epyc-orchestrator/data/bep_sandbox")

# ------------------------------------------------------------------ independent oracle

_FAIL_WORDS = (
    "error", "fail", "traceback", "exception", "refused", "denied", "not found", "no such",
    "rejected", "invalid", "halted", "timed out", "timeout", "cannot", "can't", "unable", "aborted",
    "non-zero", "nonzero", "❌", "✗", "not permitted", "not allowed", "mismatch",
)
#: Lines that mention a failure word without recording a failed action: source code being written
#: (print/comment/raise lines inside a file body) and the standing prompt rules text.
_NOISE_PREFIXES = ("print(", "#", "Rules:", "raise ", "assert ", "def ", "return ", "except ", '"postconditions"')
_SPLIT = re.compile(r"""[\s=,;()\[\]{}<>"'`|]+""")
_EXTS = {"py", "md", "json", "jsonl", "yaml", "yml", "txt", "sh", "gguf", "log", "toml", "cfg", "ini", "csv", "sqlite"}


def oracle_failed_lines(text: str) -> list[str]:
    out, seen = [], set()
    for ln in text.split("\n"):
        low = ln.lower()
        if ln.strip() and not ln.strip().startswith(_NOISE_PREFIXES) and any(w in low for w in _FAIL_WORDS) and ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out


def _is_uuid(tok: str) -> bool:
    parts = tok.split("-")
    return [len(p) for p in parts] == [8, 4, 4, 4, 12] and all(_ishex(p) for p in parts)


def _ishex(s: str) -> bool:
    return bool(s) and all(c in "0123456789abcdefABCDEF" for c in s)


def oracle_pointer_tokens(text: str) -> dict[str, str]:
    """token -> kind."""
    found: dict[str, str] = {}
    for raw in _SPLIT.split(text):
        tok = raw.strip(".:!?*#~")
        if len(tok) < 2:
            continue
        kind = None
        stem = tok.rstrip("%")
        if tok.startswith("//"):
            continue  # URL tail
        if tok.startswith("/") and tok.count("/") >= 2 and any(c.isalnum() for c in tok):
            kind = "abs_path"
        elif tok.startswith("/") and tok.count("/") == 1 and tok.rsplit(".", 1)[-1].lower() in _EXTS and "." in tok:
            kind = "abs_path"  # root-level file, e.g. /shapes.py
        elif "/" in tok and "." in tok.rsplit("/", 1)[-1] and tok.rsplit(".", 1)[-1].lower() in _EXTS and not tok.startswith(("http", "//")):
            kind = "rel_path"
        elif "." in tok and "/" not in tok and tok.rsplit(".", 1)[-1].lower() in _EXTS and tok.split(".")[0].replace("_", "").isalnum():
            kind = "bare_file"
        elif _is_uuid(tok):
            kind = "uuid"
        elif 7 <= len(tok) <= 128 and _ishex(tok) and any(c.isdigit() for c in tok) and any(c in "abcdefABCDEF" for c in tok):
            kind = "hex"
        elif tok.count("-") >= 1 and tok.split("-")[0].isalnum() and tok.split("-")[0].isupper() and len(tok.split("-")[0]) >= 2 and tok[0].isalpha() and all(p.isalnum() for p in tok.split("-")):
            kind = "box_id"
        else:
            try:
                float(stem)
                digits = stem.lstrip("+-").replace(".", "")
                if ("." in stem and digits.replace(".", "").isdigit()) or (stem.lstrip("+-").isdigit() and len(digits) >= 4):
                    kind = "number"
                    tok = stem.lstrip("+-")
            except ValueError:
                pass
        if kind:
            found.setdefault(tok, kind)
    return found


# ------------------------------------------------------------------ corpus

def list_traces(corpus: Path) -> list[Path]:
    out = []
    for run in sorted(corpus.iterdir()):
        if run.name.startswith("INVALID-") or not run.is_dir():
            continue
        out.extend(sorted((run / "traces").glob("*.jsonl")) if (run / "traces").is_dir() else [])
    return out


def build_context(trace: Path) -> str:
    seen: set[str] = set()
    chunks: list[str] = []
    for raw in trace.read_text(errors="replace").splitlines():
        if not raw.strip():
            continue
        t = json.loads(raw)
        chunks.append(f"[Turn {t.get('turn')}] role={t.get('role')}")
        for key in ("raw_output", "extracted_code"):
            if t.get(key):
                chunks.append(f"ACTION: {t[key]}" if key == "raw_output" else str(t[key]))
        for ln in (t.get("prompt_tail") or "").split("\n"):
            if ln not in seen:
                seen.add(ln)
                chunks.append(ln)
    return "\n".join(chunks)


# ------------------------------------------------------------------ compaction harness

def _stub(kind: str):
    def llm_call(prompt: str, role: str = "", **_kw):
        if kind == "lossy":
            return "- Session compacted: earlier turns summarised; consult the externalized file.\n"
        body = prompt.split("\n---\n\n", 1)[-1].split("\n")
        return "- Partial summary:\n" + "\n".join(body[::3])
    return llm_call


def compact(old: str, stub_kind: str, tmpdir: str) -> str:
    from src.graph.compaction import _maybe_compact_context

    os.environ["ORCHESTRATOR_PATHS_TMP_DIR"] = tmpdir
    state = SimpleNamespace(
        task_id="replay", context=old, turns=10, compaction_count=0, last_compaction_turn=0,
        context_file_paths=[], compaction_tokens_saved=0, aggregate_tokens=0, current_role="worker",
    )
    prim = SimpleNamespace(llm_call=_stub(stub_kind), registry=None, server_urls={})
    ctx = SimpleNamespace(state=state, deps=SimpleNamespace(primitives=prim, repl=None))
    feats = SimpleNamespace(
        session_compaction=True, session_compaction_llm_index=True, context_arm_ab=False,
        session_token_budget=False, compaction_failed_action_ledger=True,
    )
    with patch("src.features.features", return_value=feats):
        asyncio.run(_maybe_compact_context(ctx, force=True))
    return state.context


def compact_sim(old: str, stub_kind: str) -> str:
    """Same assembly as ``_maybe_compact_context`` but with a keep-recent floor of 20% of the text, for
    single traces too short (< 3000 chars) to clear the real path's 3000-char floor."""
    from src.graph.compaction import apply_integrity

    keep = max(1, len(old) // 5)
    head, tail = old[:-keep], old[-keep:]
    index = _stub(stub_kind)("x\n---\n\n" + head, role="worker_general")
    new = f"[Context Index (compaction #1)]\n{index}\n\n[Recent Context]\n{tail}\n\nFull context available: read_file(\"/x\")"
    return apply_integrity(old, new)


def build_units(corpus: Path, min_chars: int):
    """(name, kind, context): every single trace, plus one multi-task session per run directory
    (its traces concatenated), the latter clearing the real compaction floor."""
    traces = list_traces(corpus)
    runs: dict[Path, list[Path]] = {}
    for t in traces:
        runs.setdefault(t.parent.parent, []).append(t)
    units = [(str(t), "trace", build_context(t)) for t in traces]
    units += [(str(r) + "/traces/*", "run_session", "\n".join(build_context(t) for t in ts)) for r, ts in runs.items()]
    return [(n, k, c) for n, k, c in units if len(c) >= min_chars or k == "trace"]


def classify_misses(fail_missing: list[str], pin_missing: dict[str, str]):
    from src.graph.compaction import _FAILED_ACTION_RE, _PINNED_RES

    f_gap = [ln for ln in fail_missing if not _FAILED_ACTION_RE.search(ln)]
    p_gap = {t: k for t, k in pin_missing.items() if not any(rx.fullmatch(t) for rx in _PINNED_RES)}
    return f_gap, p_gap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "har_compact_ledger_replay.json")
    ap.add_argument("--min-chars", type=int, default=3500, help="skip contexts too short to externalize anything")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rows, gap_f, gap_p = [], {}, {}
    for trace, kind, old in build_units(args.corpus, args.min_chars):
        o_fail = oracle_failed_lines(old)
        o_pins = oracle_pointer_tokens(old)
        for stub in ("lossy", "partial"):
            if kind == "run_session":  # real _maybe_compact_context
                with tempfile.TemporaryDirectory() as td:
                    new = compact(old, stub, td)
            else:
                new = compact_sim(old, stub)
            f_missing = [ln for ln in o_fail if ln not in new]
            p_missing = {t: k for t, k in o_pins.items() if t not in new}
            f_gap, p_gap = classify_misses(f_missing, p_missing)
            for ln in f_gap:
                gap_f.setdefault(ln, str(trace))
            for t, k in p_gap.items():
                gap_p.setdefault(t, (k, str(trace)))
            rows.append({
                "trajectory": str(trace), "kind": kind, "stub": stub, "context_chars": len(old),
                "oracle_failed": len(o_fail), "failed_survived": len(o_fail) - len(f_missing),
                "failed_missing_capped": len(f_missing) - len(f_gap), "failed_missing_gap": len(f_gap),
                "oracle_pointers": len(o_pins), "pointers_survived": len(o_pins) - len(p_missing),
                "pointers_missing_capped": len(p_missing) - len(p_gap), "pointers_missing_gap": len(p_gap),
            })
            if args.verbose:
                print(rows[-1])
    ran = [r for r in rows if "stub" in r]
    summary = {
        "single_traces_replayed": len({r["trajectory"] for r in ran if r["kind"] == "trace"}),
        "run_sessions_replayed_real_path": len({r["trajectory"] for r in ran if r["kind"] == "run_session"}),
        "oracle_failed_total": sum(r["oracle_failed"] for r in ran if r["stub"] == "lossy"),
        "oracle_pointers_total": sum(r["oracle_pointers"] for r in ran if r["stub"] == "lossy"),
        "failed_gap_total": sum(r["failed_missing_gap"] for r in ran if r["stub"] == "lossy"),
        "pointer_gap_total": sum(r["pointers_missing_gap"] for r in ran if r["stub"] == "lossy"),
        "failed_capped_total": sum(r["failed_missing_capped"] for r in ran if r["stub"] == "lossy"),
        "pointer_capped_total": sum(r["pointers_missing_capped"] for r in ran if r["stub"] == "lossy"),
    }
    result = {
        "summary": summary, "rows": rows,
        "unique_failed_gap_lines": [{"line": k, "trajectory": v} for k, v in sorted(gap_f.items())][:200],
        "unique_pointer_gaps": [{"token": k, "kind": v[0], "trajectory": v[1]} for k, v in sorted(gap_p.items())][:300],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
