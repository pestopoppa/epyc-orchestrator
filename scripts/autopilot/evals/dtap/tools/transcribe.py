#!/usr/bin/env python3
"""EVL-46 TU-DTAP-1 authoring tool: transcribe a bounded DTAP subset from a
disposable clone of https://github.com/AI-secure/DecodingTrust-Agent.

Run against a DISPOSABLE clone pinned at commit e0323a521ba4ef88f8e14c1eccf68d0a3d19a458.
This tool is an authoring-time conversion script only; it is never part of the
runtime harness and must never be run inside the upstream checkout on a host
that will serve live traffic. It does not execute any upstream code.

Outputs (into the repo tree under scripts/autopilot/evals/dtap/):
  cases.json     - verbatim task configs (upstream config.yaml -> JSON) + metadata
  manifest.json  - per-file provenance: upstream paths, SHA-256 digests, rewrites
  judges/<case_id>/judge.py - judge sources with the import shim applied

Judge transcription policy (logic-preserving, reviewable):
  * Every line of judge logic is kept byte-identical to upstream.
  * Only the module-import prologue is rewritten: the upstream sys.path dance
    (`sys.path.insert` / `from judge import BaseJudge` /
    `from dt_arena.utils.X import Y`) becomes a direct import of the local
    harness shims (`harness.base_judge`, `harness.shims.*`) which expose the
    same names with the same signatures.
  * `import sys` is dropped only when the body never references `sys.`.
  * An attribution header (upstream path, commit, file SHA-256) is prepended.
  * The rewritten file's SHA-256 is recorded in manifest.json; the *upstream*
    SHA-256 is also recorded so the transcription can be re-verified at any time.

Usage:
  python3 tools/transcribe.py --upstream-root /tmp/dtap-import-20260825 \
      --cases cases.tsv --out <dtap-dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import yaml

SELECTED = [
    ("finance", "benign/trade_execution/1"),
    ("finance", "benign/analyst_review/1"),
    ("finance", "malicious/direct/churning/5"),
    ("finance", "malicious/direct/churning/4"),
    ("finance", "malicious/direct/churning/1"),
    ("finance", "malicious/direct/fake_security_alert/7"),
    ("finance", "malicious/indirect/action_reversal/2"),
    ("finance", "malicious/indirect/action_reversal/3"),
    ("finance", "malicious/indirect/action_reversal/12"),
    ("finance", "malicious/indirect/symbol_substitution/7"),
    ("finance", "malicious/indirect/data_exfiltration/4"),
    ("finance", "malicious/indirect/quantity_inflation/14"),
    ("crm", "benign/1"),
    ("crm", "malicious/direct/social-scoring/10"),
    ("crm", "malicious/direct/general-ai-restrictions/1"),
    ("crm", "malicious/indirect/social-scoring/3"),
    ("crm", "malicious/indirect/misinformation-hallucination/1"),
    ("crm", "malicious/indirect/general-ai-restrictions/4"),
]

UPSTREAM_COMMIT = "e0323a521ba4ef88f8e14c1eccf68d0a3d19a458"
UPSTREAM_TREE = "fd5a107aedb8971c346fc0e85d4789bf510e3f5f"
UPSTREAM_REPO = "https://github.com/AI-secure/DecodingTrust-Agent"
UPSTREAM_LICENSE = "Apache-2.0"

IMPORT_RE = re.compile(r"^from dt_arena\.utils\.([\w]+)(?:\.([\w]+))? import (.*)$")
IMPORT_JUDGE_RE = re.compile(r"^from judge import BaseJudge$")
SYS_PATH_RE = re.compile(r"^sys\.path\.insert\(")
SYSPATH_DANCE_END = "from dt_arena.utils"  # first such import is inside the dance


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def threat_class(rel: str) -> str:
    if rel.startswith("benign"):
        return "benign"
    return "indirect" if "/indirect/" in rel else "direct"


def task_id_of(config: dict, domain: str, rel: str) -> str:
    tid = ((config.get("Task") or {}).get("task_id") or "").strip()
    if tid:
        return tid
    return f"{domain}-{threat_class(rel)}-{rel.split('/')[-1]}"


def families_of(config: dict) -> list[str]:
    attack = config.get("Attack") or {}
    turns = attack.get("attack_turns") or []
    fams: set[str] = set()
    for turn in turns:
        for step in turn.get("attack_steps", []):
            t = step.get("type")
            if t in ("prompt", "tool", "skill", "environment"):
                fams.add(t)
    return sorted(fams)


def transcribe_judge(src: Path, case_id: str, upstream_sha: str) -> str:
    lines = src.read_text().splitlines()
    out: list[str] = []
    i = 0
    seen_imports = False
    while i < len(lines):
        line = lines[i]
        if i == 0 and line.startswith("#!"):
            out.append(line)
            i += 1
            continue
        stripped = line.strip()
        if SYS_PATH_RE.match(stripped) or stripped.startswith("_root = Path("):
            i += 1
            continue
        # `_task_dir` must survive: the harness shim's load_task_config() resolves
        # the case id from the transcribed judge directory's name.
        m = IMPORT_JUDGE_RE.match(stripped)
        if m:
            if not seen_imports:
                out.append("from harness.base_judge import BaseJudge")
                seen_imports = True
            i += 1
            continue
        m = IMPORT_RE.match(stripped)
        if m:
            pkg, sub, names = m.group(1), m.group(2), m.group(3)
            if sub == "helpers" or pkg == "utils":
                mod = "utils" if pkg == "utils" else f"shims.{pkg}"
                indent = line[: len(line) - len(line.lstrip())]
                out.append(f"{indent}from harness.{mod} import {names}")
                seen_imports = True
                i += 1
                continue
        if stripped == "import sys" and "sys." not in "\n".join(lines[i + 1 :]):
            i += 1
            continue
        out.append(line)
        i += 1
    body = "\n".join(out).rstrip() + "\n"
    header = (
        "# Transcribed for EVL-46 (TU-DTAP-1) from https://github.com/AI-secure/DecodingTrust-Agent\n"
        f"# upstream path: dataset/{case_id}/judge.py\n"
        f"# upstream commit: {UPSTREAM_COMMIT} (tree {UPSTREAM_TREE})\n"
        f"# upstream file SHA-256: {upstream_sha}\n"
        "# License: Apache-2.0 (https://github.com/AI-secure/DecodingTrust-Agent)\n"
        "# Import prologue rewritten to local harness shims; judge logic byte-identical\n"
        "# to upstream (see tools/transcribe.py).\n"
        "# ruff: noqa\n"
    )
    return header + body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    root: Path = args.upstream_root
    out: Path = args.out
    if not (root / "LICENSE").exists():
        print(f"error: {root} does not look like the upstream checkout", file=sys.stderr)
        return 1
    if sha256_file(root / "LICENSE") != "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4":
        print("error: LICENSE hash mismatch; refusing to transcribe", file=sys.stderr)
        return 1
    head = ""
    try:
        head = (
            (root / ".git" / "refs" / "heads" / "main").read_text().strip()
            if (root / ".git" / "refs" / "heads" / "main").exists()
            else "detached"
        )
    except Exception:
        pass

    judges_dir = out / "judges"
    fixtures_dir = out / "fixtures"
    if judges_dir.exists():
        shutil.rmtree(judges_dir)
    judges_dir.mkdir(parents=True)
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    cases: dict[str, dict] = {}
    manifest: dict[str, dict] = {"meta": {}, "cases": {}}
    manifest["meta"] = {
        "source_repo": UPSTREAM_REPO,
        "commit": UPSTREAM_COMMIT,
        "tree": UPSTREAM_TREE,
        "license": UPSTREAM_LICENSE,
        "git_head_at_transcription": head,
        "selected_cases": len(SELECTED),
    }

    for domain, rel in SELECTED:
        case_dir = root / "dataset" / domain / rel
        cfg_path = case_dir / "config.yaml"
        judge_path = case_dir / "judge.py"
        if not cfg_path.exists() or not judge_path.exists():
            print(f"error: missing files for {domain}/{rel}", file=sys.stderr)
            return 1
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        case_id = task_id_of(cfg, domain, rel)
        fams = families_of(cfg)
        cfg_sha = sha256_file(cfg_path)
        judge_sha = sha256_file(judge_path)
        cases[case_id] = {
            "upstream_path": f"dataset/{domain}/{rel}",
            "domain": domain,
            "threat": threat_class(rel),
            "risk_category": (cfg.get("Task") or {}).get("task_category") or "",
            "injection_families": fams,
            "config": cfg,
        }
        transcribed = transcribe_judge(judge_path, case_id, judge_sha)
        jdir = judges_dir / case_id
        jdir.mkdir(parents=True)
        (jdir / "judge.py").write_text(transcribed)
        manifest["cases"][case_id] = {
            "upstream_path": f"dataset/{domain}/{rel}",
            "upstream_judge_sha256": judge_sha,
            "upstream_config_sha256": cfg_sha,
            "transcribed_judge_sha256": sha256_bytes(transcribed.encode()),
            "injection_families": fams,
            "threat": threat_class(rel),
            "rewrites": ["import prologue -> harness shims", "attribution header prepended"],
        }

    (out / "cases.json").write_text(json.dumps({"meta": manifest["meta"], "cases": cases}, indent=2, sort_keys=True) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"transcribed {len(cases)} cases -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
