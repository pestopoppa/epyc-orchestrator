#!/usr/bin/env python3
"""Source 5: bug-reports miner -> known-bad code candidates (RC-3). NO inference.

Checks the two configured bug-report roots. Each report is a (description .md,
fix .patch) pair. The KNOWN-BAD candidate = the pre-patch code reconstructed
from the unified diff (context + removed '-' lines, dropping the '+' additions);
the human-diagnosed root cause (.md) = the gold cause; the verified fix patch =
a corroborating oracle. NATURAL defects.

If no bug-report dirs/pairs exist, emits 0 rows and records the absence (the
orchestrator repo has no bug-reports/ dir as of 2026-07-16).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from corpus_v1 import common
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from corpus_v1 import common


def _reconstruct_prepatch(patch_text: str) -> str:
    """Buggy (pre-patch) source: keep context + removed lines, drop additions."""
    out = []
    for ln in patch_text.split("\n"):
        if ln.startswith(("+++", "---", "@@", "diff ", "index ")):
            continue
        if ln.startswith("+"):
            continue  # the fix; excluded from the buggy state
        if ln.startswith("-"):
            out.append(ln[1:])
        elif ln.startswith(" "):
            out.append(ln[1:])
        else:
            out.append(ln)
    return "\n".join(out).strip()


def _md_title(md_text: str) -> str:
    for ln in md_text.split("\n"):
        s = ln.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()
    return ""


def _pair_key(name: str) -> str:
    m = re.match(r"(\d+)", name)
    return m.group(1) if m else name


def mine() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    scanned = []
    dirs_present = []
    for root in common.BUGREPORT_DIRS:
        if not root.exists():
            scanned.append({"dir": str(root), "present": False})
            continue
        dirs_present.append(str(root))
        patches = sorted(root.rglob("*.patch"))
        scanned.append({"dir": str(root), "present": True, "patches": len(patches)})
        for patch in patches:
            group = patch.parent
            key = _pair_key(patch.stem)
            md = None
            for cand in sorted(group.glob("*.md")):
                if _pair_key(cand.stem) == key and cand.name.lower() != "readme.md":
                    md = cand
                    break
            patch_text = patch.read_text(encoding="utf-8", errors="replace")
            buggy = _reconstruct_prepatch(patch_text)
            if not buggy:
                continue
            desc = md.read_text(encoding="utf-8", errors="replace") if md else ""
            title = _md_title(desc) or patch.stem
            component = group.name
            rows.append(
                common.make_row(
                    source_benchmark="bug-report",
                    source_suite="code",
                    domain="code",
                    task=common.truncate(
                        f"Review this code region for defects.\nComponent: {component}\n"
                        f"Reported issue: {title}",
                        4000,
                    ),
                    candidate=common.truncate(buggy, 20000),
                    gold_label="reject",
                    gold_source="human_bug_report+verified_fix_patch",
                    gold_confidence="single_oracle",  # functional verification deferred
                    defect_origin="natural",
                    row_key=f"bug|{group.name}|{patch.name}",
                    executable_oracle={
                        "verdict": "fail",
                        "oracle_type": "reported_bug_fixed_by_compile_verified_patch",
                        "source": str(patch.relative_to(root)),
                    },
                    reasoning_module_labels={
                        "source": "human_bug_report",
                        "title": title,
                        "description": common.truncate(desc, 4000),
                        "fix_patch": common.truncate(patch_text, 4000),
                    },
                    rationale_gold_cause=common.truncate(title + "\n" + desc, 4000),
                    ambiguous_tail=True,  # single-oracle -> arbitration
                    natural_defect_control=True,
                    decontamination={"repo": component, "source_dir": str(group)},
                    provenance={
                        "component": component,
                        "patch": patch.name,
                        "md": md.name if md else None,
                        "candidate_is": "reconstructed_prepatch_code",
                    },
                )
            )
    stats = {"dirs_scanned": scanned, "dirs_present": dirs_present, "rows": len(rows)}
    return rows, stats


def main() -> int:
    rows, stats = mine()
    out = common.STAGING_DIR / "bugreports.jsonl"
    n = common.write_jsonl(out, rows)
    print(f"[mine_bugreports] wrote {n} rows -> {out}  stats={stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
