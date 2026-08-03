#!/usr/bin/env python3
"""No-inference edit-transaction validation for the BEP sandbox.

Default mode is module-only: it exercises ``src.edit_transaction.run_edit_transaction`` against a
deterministic, inference-free solution render so the benchmark can emit clean-window evidence
without touching live inference. A guarded live mode is still available for explicit operator use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ORCH = Path(__file__).resolve().parents[2]
TASKS_PATH = ORCH / "data" / "bep_sandbox" / "tasks.jsonl"
SOLUTIONS_PATH = ORCH / "data" / "bep_sandbox" / "solutions.jsonl"
DEFAULT_SCRATCH = Path("/mnt/raid0/llm/tmp/bep_edittxn/work")
DEFAULT_API_URL = "http://localhost:8000/chat"

sys.path.insert(0, str(ORCH))
from src.edit_transaction import EditResult, run_edit_transaction  # noqa: E402


@dataclass(frozen=True)
class ChatHTTPError(RuntimeError):
    status_code: int
    body: str

    def __str__(self) -> str:
        return f"http {self.status_code}: {self.body}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _orch_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ORCH,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return ""


def _reset_scratch(root: Path, files: dict[str, str] | None) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for rel, content in (files or {}).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _render_solution(solution: dict[str, Any]) -> str:
    blocks: list[str] = []
    for rel, content in sorted((solution.get("write") or {}).items()):
        blocks.append(f"<<<FILE: {rel}>>>\n{content}\n<<<END>>>")
    for rel in sorted(solution.get("delete") or []):
        blocks.append(f"<<<DELETE: {rel}>>>")
    return "\n".join(blocks)


def _live_llm(api_url: str, prompt: str, *, session_id: str) -> str:
    payload = {
        "prompt": prompt,
        "force_role": "coder_escalation",
        "force_mode": "direct",
        "max_turns": 1,
        "cache_prompt": False,
        "mock_mode": False,
        "real_mode": True,
        "session_id": session_id,
    }
    try:
        resp = httpx.post(api_url, json=payload, timeout=220)
    except httpx.HTTPError as exc:
        raise ChatHTTPError(-1, str(exc)) from exc
    if resp.status_code == 412:
        raise ChatHTTPError(resp.status_code, resp.text)
    resp.raise_for_status()
    data = resp.json()
    return data.get("answer") or data.get("response") or data.get("content") or ""


def _attested_header(*, mode: str, scratch_root: Path, task_ids: list[str], probe_ids: list[str]) -> dict[str, Any]:
    return {
        "kind": "attested-run-header",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "orch_head": _orch_head(),
        "scratch_root": str(scratch_root),
        "edit_root": str(scratch_root),
        "task_ids": task_ids,
        "probe_ids": probe_ids,
        "mode": mode,
    }


def _classify_failure(result: EditResult | None, *, raw: str, verifier_ok: bool,
                      llm_error: Exception | None) -> str:
    if llm_error is not None:
        if isinstance(llm_error, ChatHTTPError) and llm_error.status_code == 412:
            return "412/precondition"
        return "chat/http error"

    if result is None:
        return "chat/http error"

    err = (result.error or "").lower()
    raw = raw or ""
    if not result.ok:
        if "scope too large" in err:
            return "scope-cap reject"
        if "no valid file blocks parsed" in err or not raw.strip():
            return "parse/no blocks"
        if "functional verifier failed" in err:
            return "verifier fail"
        return "rollback/self-check"
    if not verifier_ok:
        if "scope too large" in err or "scope too large" in raw.lower():
            return "scope-cap reject"
        if "no valid file blocks parsed" in err or "no valid file blocks" in raw.lower() or not raw.strip():
            return "parse/no blocks"
        if "syntaxerror" in err or "rollback" in err or "self-check" in err:
            return "rollback/self-check"
        return "verifier fail"
    return "pass"


def _run_module_task(task: dict[str, Any], solution: dict[str, Any], scratch: Path
                     ) -> tuple[EditResult, str, Exception | None]:
    def _stub_llm(_prompt: str) -> str:
        return _render_solution(solution)

    try:
        return (*run_edit_transaction(_stub_llm, task["prompt"], scratch,
                                      target_files=list((task.get("files") or {}).keys()) or None,
                                      verify_fn=lambda root: _run_verifier(root, task["verifier_cmd"])),
                None)
    except Exception as exc:  # pragma: no cover - defensive, exercised via live-path tests if needed
        return EditResult(ok=False, error=f"{type(exc).__name__}: {exc}"), "", exc


def _run_live_task(task: dict[str, Any], scratch: Path, api_url: str, session_id: str
                   ) -> tuple[EditResult, str, Exception | None]:
    def _llm(prompt: str) -> str:
        return _live_llm(api_url, prompt, session_id=session_id)

    try:
        return (*run_edit_transaction(_llm, task["prompt"], scratch,
                                      target_files=list((task.get("files") or {}).keys()) or None,
                                      verify_fn=lambda root: _run_verifier(root, task["verifier_cmd"])),
                None)
    except Exception as exc:
        return EditResult(ok=False, error=f"{type(exc).__name__}: {exc}"), "", exc


def _run_verifier(root: Path, cmd: str) -> bool:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True, timeout=60, env=env)
        return r.returncode == 0 and "PASS" in (r.stdout + r.stderr)
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("module", "live"), default="module",
                    help="module = no-inference run_edit_transaction path; live = guarded /chat call")
    ap.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH)
    ap.add_argument("--api-url", default=DEFAULT_API_URL)
    ap.add_argument("--live-confirmed", action="store_true",
                    help="required to use the live /chat path")
    args = ap.parse_args(argv)

    tasks = _load_jsonl(TASKS_PATH)
    solutions = {row["id"]: row for row in _load_jsonl(SOLUTIONS_PATH)}
    task_ids = [t["id"] for t in tasks]
    header = _attested_header(mode=args.mode, scratch_root=args.scratch_root,
                              task_ids=task_ids, probe_ids=[])
    print("[attest] " + json.dumps(header, sort_keys=True))

    if args.mode == "live" and not args.live_confirmed:
        print("REFUSING live inference: pass --live-confirmed explicitly.", file=sys.stderr)
        return 2

    npass = 0
    for task in tasks:
        _reset_scratch(args.scratch_root, task.get("files"))
        result: EditResult | None
        raw = ""
        llm_error: Exception | None = None
        if args.mode == "module":
            result, raw, llm_error = _run_module_task(task, solutions[task["id"]], args.scratch_root)
        else:
            result, raw, llm_error = _run_live_task(task, args.scratch_root, args.api_url,
                                                    session_id=f"edittxn-{task['id']}")

        verifier_ok = bool(result and result.ok and _run_verifier(args.scratch_root, task["verifier_cmd"]))
        bucket = _classify_failure(result, raw=raw, verifier_ok=verifier_ok, llm_error=llm_error)
        npass += int(bucket == "pass")
        print(json.dumps({
            "task_id": task["id"],
            "bucket": bucket,
            "txn_ok": bool(result and result.ok),
            "verifier_ok": verifier_ok,
            "written": result.written if result else [],
            "deleted": result.deleted if result else [],
            "rejected": result.rejected if result else [],
            "error": result.error if result else "",
            "mode": args.mode,
        }, sort_keys=True))

    print(f"[summary] edit-transaction {npass}/{len(tasks)} pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
