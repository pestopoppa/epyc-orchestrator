#!/usr/bin/env python3
"""Edit-mode wiring probe for the BEP sandbox.

Default mode is stubbed and inference-free so the script can emit clean-window evidence without
touching the live server. A guarded live /chat path remains available behind --live-confirmed.
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

ORCH = Path("/mnt/raid0/llm/epyc-orchestrator")
TASKS_PATH = ORCH / "data" / "bep_sandbox" / "tasks.jsonl"
SOLUTIONS_PATH = ORCH / "data" / "bep_sandbox" / "solutions.jsonl"
DEFAULT_ROOT = Path(os.environ.get("ORCHESTRATOR_EDIT_ROOT", "/mnt/raid0/llm/tmp/bep_editwire/work"))
DEFAULT_API_URL = "http://localhost:8000/chat"
PROBE_IDS = ["t1_create_util", "t2_add_and_use", "t4_rename_module"]

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


def _reset_root(root: Path, files: dict[str, str] | None) -> None:
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


def _attested_header(*, mode: str, edit_root: Path, task_ids: list[str], probe_ids: list[str]) -> dict[str, Any]:
    return {
        "kind": "attested-run-header",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "orch_head": _orch_head(),
        "scratch_root": str(edit_root),
        "edit_root": str(edit_root),
        "task_ids": task_ids,
        "probe_ids": probe_ids,
        "mode": mode,
    }


def _run_verifier(root: Path, cmd: str) -> bool:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True, timeout=60, env=env)
        return r.returncode == 0 and "PASS" in (r.stdout + r.stderr)
    except Exception:
        return False


def _live_chat(prompt: str, *, api_url: str, session_id: str) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "force_role": "coder_escalation",
        "force_mode": "edit",
        "max_turns": 1,
        "cache_prompt": False,
        "mock_mode": False,
        "real_mode": True,
        "session_id": session_id,
    }
    try:
        resp = httpx.post(api_url, json=payload, timeout=240)
    except httpx.HTTPError as exc:
        raise ChatHTTPError(-1, str(exc)) from exc
    if resp.status_code == 412:
        raise ChatHTTPError(resp.status_code, resp.text)
    resp.raise_for_status()
    return resp.json()


def _classify_failure(*, result: EditResult | None, response_mode: str | None, response_text: str,
                      verifier_ok: bool, llm_error: Exception | None) -> str:
    if llm_error is not None:
        if isinstance(llm_error, ChatHTTPError) and llm_error.status_code == 412:
            return "412/precondition"
        return "chat/http error"
    if result is not None and not result.ok:
        text = (result.error or "").lower()
        if "scope too large" in text:
            return "scope-cap reject"
        if "no valid file blocks" in text or not response_text.strip():
            return "parse/no blocks"
        if "functional verifier failed" in text:
            return "verifier fail"
        return "rollback/self-check"
    if response_mode != "edit":
        text = response_text.lower()
        if not text.strip() or "no valid file blocks" in text:
            return "parse/no blocks"
        if "scope too large" in text:
            return "scope-cap reject"
        if "syntaxerror" in text or "rollback" in text or "self-check" in text:
            return "rollback/self-check"
        return "rollback/self-check"
    if not verifier_ok:
        text = response_text.lower()
        if "scope too large" in text:
            return "scope-cap reject"
        if "no valid file blocks" in text or not text.strip():
            return "parse/no blocks"
        if "syntaxerror" in text or "rollback" in text or "self-check" in text:
            return "rollback/self-check"
        return "verifier fail"
    return "pass"


def _run_stub_task(task: dict[str, Any], solution: dict[str, Any], root: Path
                   ) -> tuple[EditResult, dict[str, Any], Exception | None]:
    def _stub_llm(_prompt: str) -> str:
        return _render_solution(solution)

    try:
        result, _raw = run_edit_transaction(_stub_llm, task["prompt"], root,
                                            target_files=list((task.get("files") or {}).keys()) or None,
                                            verify_fn=lambda tx_root: _run_verifier(
                                                tx_root, task["verifier_cmd"]
                                            ))
        return result, {"mode": "edit", "answer": "[STUB]"}, None
    except Exception as exc:
        return EditResult(ok=False, error=f"{type(exc).__name__}: {exc}"), {"mode": None, "answer": ""}, exc


def _run_live_task(task: dict[str, Any], root: Path, api_url: str
                   ) -> tuple[EditResult, dict[str, Any], Exception | None]:
    try:
        body = _live_chat(task["prompt"], api_url=api_url, session_id=f"editwire-{task['id']}")
        mode = body.get("mode")
        text = (body.get("answer") or body.get("response") or body.get("content") or "")
        return EditResult(ok=True), {"mode": mode, "answer": text}, None
    except Exception as exc:
        return EditResult(ok=False, error=f"{type(exc).__name__}: {exc}"), {"mode": None, "answer": ""}, exc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("stub", "live"), default="stub",
                    help="stub = no-inference local edit transaction; live = guarded /chat call")
    ap.add_argument("--edit-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--api-url", default=DEFAULT_API_URL)
    ap.add_argument("--live-confirmed", action="store_true",
                    help="required to use the live /chat path")
    args = ap.parse_args(argv)

    tasks = {t["id"]: t for t in _load_jsonl(TASKS_PATH)}
    solutions = {row["id"]: row for row in _load_jsonl(SOLUTIONS_PATH)}
    probe_tasks = [tasks[tid] for tid in PROBE_IDS]
    header = _attested_header(mode=args.mode, edit_root=args.edit_root,
                              task_ids=sorted(tasks), probe_ids=PROBE_IDS)
    print("[attest] " + json.dumps(header, sort_keys=True))

    if args.mode == "live" and not args.live_confirmed:
        print("REFUSING live inference: pass --live-confirmed explicitly.", file=sys.stderr)
        return 2

    npass = 0
    for task in probe_tasks:
        _reset_root(args.edit_root, task.get("files"))
        if args.mode == "stub":
            result, response, llm_error = _run_stub_task(task, solutions[task["id"]], args.edit_root)
        else:
            result, response, llm_error = _run_live_task(task, args.edit_root, args.api_url)
        verifier_ok = bool(result and result.ok and _run_verifier(args.edit_root, task["verifier_cmd"]))
        response_mode = response.get("mode")
        response_text = response.get("answer") or ""
        bucket = _classify_failure(result=result, response_mode=response_mode, response_text=response_text,
                                   verifier_ok=verifier_ok, llm_error=llm_error)
        npass += int(bucket == "pass")
        print(json.dumps({
            "task_id": task["id"],
            "response_mode": response_mode,
            "bucket": bucket,
            "verifier_ok": verifier_ok,
            "answer": response_text[:120],
            "mode": args.mode,
        }, sort_keys=True))

    print(f"[summary] edit-mode wiring {npass}/{len(probe_tasks)} pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
