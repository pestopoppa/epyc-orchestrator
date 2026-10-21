"""Advisory literal-constant lint for distilled skills (handoff M-24).

Flags skills that embed literal parameters, paths, file names, column names, ids,
ports or role/model names that were copied from the source trajectory instead of
being generalised.  ADVISORY ONLY: this is not an admission gate and not a
quality score (intake-2181#03, intake-2190#07); nothing here changes what the
pipeline stores.  Stdlib only, no inference, reads files read-only.

The live teacher prompts (``prompts.py``: 'Preserve specific model names, tier
names, and port numbers') ask for exactly these literals, so ``ROLE_MODEL`` and
``PORT`` fire on them by design; the finding is reported, not suppressed.

Categories (rule set fixed 2026-10-21 before the calibration run):
  PATH        absolute path with >= 2 segments (system roots /usr /bin /etc /dev /proc /sys /lib ignored)
  FILE        file name with a data/doc/code extension
  ID          hex id >= 8 chars, or TICKET-123 style id
  NUM         number with >= 4 digits or >= 3 decimals (calendar years ignored)
  PORT        ``:NNNN`` / ``port NNNN`` with 4-5 digits
  COL         backticked identifier on a line that mentions column/field/key/header/param
  ROLE_MODEL  EPYC role/tier names and model-family names with size tags
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_SYS_ROOTS = ("usr", "bin", "etc", "dev", "proc", "sys", "lib", "sbin")
_EXT = (
    r"csv|tsv|xlsx?|json|jsonl|parquet|db|sqlite|txt|pdf|docx?|pptx?|png|jpe?g|xml|ya?ml|md|py|ipynb|toml|sh|html|log"
)
RULES: dict[str, re.Pattern[str]] = {
    "PATH": re.compile(r"(?<![\w./:-])/(?:[\w.\-]+/)+[\w.\-]+"),
    "FILE": re.compile(rf"(?<![\w])[\w\-]+(?:\.[\w\-]+)*\.(?:{_EXT})\b", re.I),
    "ID": re.compile(r"\b[0-9a-f]{8,}\b|\b[A-Z]{2,}-\d+\b"),
    "NUM": re.compile(r"(?<![\w.])(?:\d{4,}|\d+\.\d{3,})(?![\w.]*\d)"),
    "PORT": re.compile(r"(?::|\bport\s+)(\d{4,5})\b", re.I),
    "ROLE_MODEL": re.compile(
        r"\b(?:frontdoor|architect_\w+|worker_\w+|coder_\w+|ingest_\w+|"
        r"qwen\d[\w.\-]*|gemma\d?[\w.\-]*|llama[\w.\-]*|glm[\w.\-]*|deepseek[\w.\-]*|"
        r"\d+b-a\d+b|\d+b)\b",
        re.I,
    ),
}
_COL_LINE = re.compile(r"column|field|key|header|param", re.I)
_BACKTICK = re.compile(r"`([A-Za-z_][\w.\-]{1,40})`")
_YEAR = re.compile(r"^(?:19|20)\d{2}$")


# ── M-27 (a): negative-capability claims and unlabelled-success provenance ──
# Kept OUT of RULES so the M-24 calibration numbers stay reproducible.
_NEG_CAP_RE = re.compile(
    r"\b(?:does(?:\s+not|n'?t)\s+(?:work|support|exist|apply)|do(?:es)?\s+not\s+(?:use|support)|"
    r"cannot\s+(?:use|be\s+used|handle|run|access)|can'?t\s+(?:use|handle|run|access)|"
    r"(?:is|are)\s+not\s+(?:supported|available|possible|allowed)|unable\s+to|never\s+works?|"
    r"not\s+possible\s+to|fails?\s+to\s+(?:work|run))\b",
    re.I,
)


def extract_negative_capability(text: str) -> list[str]:
    """Matched negative-capability phrases ("X does not work", "cannot use Y"), with 40 chars of context."""
    out = []
    for m in _NEG_CAP_RE.finditer(text):
        a, b = max(0, m.start() - 40), min(len(text), m.end() + 40)
        out.append(" ".join(text[a:b].split()))
    return out


def unlabelled_success(source_outcome: str | None) -> bool:
    """True when a skill's source trajectory did not end in a labelled success."""
    return (source_outcome or "").strip().lower() != "success"


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :]
    return text


def extract_literals(text: str) -> dict[str, list[str]]:
    """Return {category: sorted unique literals} found in ``text``."""
    out: dict[str, set[str]] = defaultdict(set)
    for cat, rx in RULES.items():
        for m in rx.finditer(text):
            val = m.group(1) if cat == "PORT" else m.group(0)
            if cat == "PATH" and val.split("/")[1] in _SYS_ROOTS:
                continue
            if cat == "NUM" and _YEAR.match(val):
                continue
            if cat == "ID" and val.isdigit():
                continue
            out[cat].add(val.lower() if cat in ("FILE", "ID", "ROLE_MODEL") else val)
    for line in text.splitlines():
        if _COL_LINE.search(line):
            for m in _BACKTICK.finditer(line):
                out["COL"].add(m.group(1))
    return {k: sorted(v) for k, v in out.items()}


@dataclass
class LintResult:
    literals: dict[str, list[str]]
    grounded: dict[str, list[str]] = field(default_factory=dict)  # also present in the source text

    @property
    def flagged(self) -> bool:
        return bool(self.literals)

    @property
    def flagged_grounded(self) -> bool:
        return bool(self.grounded)


def lint_skill(skill_text: str, source_text: str | None = None) -> LintResult:
    """Lint one skill body.  With ``source_text`` also report which literals are
    verbatim in the source (copied, not merely specific)."""
    lits = extract_literals(strip_frontmatter(skill_text))
    grounded: dict[str, list[str]] = {}
    if source_text is not None:
        src = source_text.lower()
        for cat, vals in lits.items():
            hit = [v for v in vals if v.lower() in src]
            if hit:
                grounded[cat] = hit
    return LintResult(lits, grounded)


# ── CLI: corpus calibration and read-only skills.db run ─────────────────────

_TEXT_SUFFIX = {".md", ".txt", ".json", ".csv", ".py", ".toml", ".yaml", ".yml", ".sh", ".xml", ".html", ".tsv", ""}


def _source_text(task_dir: Path, cap: int = 400_000) -> str:
    parts: list[str] = []
    for p in sorted(task_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in _TEXT_SUFFIX and "tests" not in p.relative_to(task_dir).parts:
            try:
                parts.append(p.read_text(errors="ignore")[:cap])
            except OSError:
                continue
        parts.append(p.name if p.is_file() else "")
    return "\n".join(parts)


def run_skilllearnbench(repo: Path) -> dict:
    """skills/<config>/<task>/<skill>/SKILL.md vs tasks/<task>/*/ (instruction + environment names/text; tests excluded)."""
    src_cache: dict[str, str] = {}
    rows: dict[str, Counter] = defaultdict(Counter)
    cat_rows: dict[str, Counter] = defaultdict(Counter)
    for skill in sorted((repo / "skills").glob("*/*/*/SKILL.md")):
        cfg, task = skill.parts[-4], skill.parts[-3]
        if task not in src_cache:
            td = repo / "tasks" / task
            src_cache[task] = _source_text(td) if td.is_dir() else ""
        res = lint_skill(skill.read_text(errors="ignore"), src_cache[task] or None)
        r = rows[cfg]
        r["n"] += 1
        r["flagged_any"] += res.flagged
        r["flagged_grounded"] += res.flagged_grounded
        r["no_source"] += not src_cache[task]
        for c in res.literals:
            cat_rows[cfg][c] += 1
    return {"rows": {k: dict(v) for k, v in rows.items()}, "categories": {k: dict(v) for k, v in cat_rows.items()}}


def run_skills_db(path: Path) -> dict:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        n = flagged = 0
        cats: Counter = Counter()
        examples: dict[str, list[str]] = defaultdict(list)
        for title, principle, when in con.execute("SELECT title, principle, when_to_apply FROM skills"):
            res = lint_skill(f"{title}\n{principle}\n{when}")
            n += 1
            flagged += res.flagged
            for c, v in res.literals.items():
                cats[c] += 1
                if len(examples[c]) < 4:
                    examples[c].append(v[0])
        return {"n": n, "flagged_any": flagged, "by_category": dict(cats), "examples": dict(examples)}
    finally:
        con.close()


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skilllearnbench", type=Path, help="SkillLearnBench repo root")
    ap.add_argument("--skills-db", type=Path, help="skills.db (opened read-only)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(list(argv) if argv is not None else None)
    result: dict = {"advisory_only": True}
    if a.skilllearnbench:
        result["skilllearnbench"] = run_skilllearnbench(a.skilllearnbench)
    if a.skills_db:
        result["skills_db"] = run_skills_db(a.skills_db)
    text = json.dumps(result, indent=1, sort_keys=True)
    if a.out:
        a.out.write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
