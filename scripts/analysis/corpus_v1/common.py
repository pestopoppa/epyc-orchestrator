#!/usr/bin/env python3
"""Shared schema, helpers, and IO for the near-miss decision corpus v1 (H4 RC-3).

This module is the single source of truth for the row schema (RC-3), the
domain map, the eval-harness qid derivation (so journal outcomes can be joined
back to the research question pool), and the row validator used by both the
miners and the tests.

Design invariants (see reviewer-calibration-accounting.md RC-3):
  * Dual gold labels: ``executable_oracle`` + ``reasoning_module_labels`` — either
    may be null per source.
  * ``defect_origin in {natural, seeded}``; the natural-defect *control slice* is
    tagged explicitly via ``natural_defect_control``.
  * Decontamination metadata (repo/base_commit/pull_number/created_at) preserved
    wherever the source provides it (SWE-Bench-Illusion applies to derivatives).
  * ``rationale_gold_cause`` records the WHY (right-for-wrong-reason discipline).
  * Gate-worthy rows need >=2 oracles or arbitration -> ``gold_confidence`` +
    ``ambiguous_tail``.

NO inference is performed anywhere in this package.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

# --------------------------------------------------------------------------- #
# Versioning / identity
# --------------------------------------------------------------------------- #
CORPUS_ID = "nearmiss-v1"
SCHEMA_VERSION = "nearmiss_corpus_row.v1"

# Version tags for each gold-label instrument (part of the instrument identity;
# changing any of these = a new instrument version per P-REV-1).
GOLD_INSTRUMENT_VERSIONS = {
    "c-crab": "c-crab-swe-care-2026-07-16",              # human review comments + testgen oracles
    "swe-care": "swe-care-inclusionAI-2026-07-16",       # human review comments
    "autopilot-journal": "autopilot-journal-scorer-v1",  # programmatic per-question scorers
    "seeded-mutation": "nearmiss-seed-rules-v1",          # rule-based mutations (this package)
}
MUTATION_RULES_VERSION = "nearmiss-seed-rules-v1"
QID_SCHEME = "sha1(suite\\x00prompt)[:16]"

# --------------------------------------------------------------------------- #
# Source paths (as-of 2026-07-16 acquisition; see datasets/BENCHMARKS.md)
# --------------------------------------------------------------------------- #
DATASETS = Path("/mnt/raid0/llm/datasets")
CCRAB_DIR = DATASETS / "c-crab"
CCRAB_PREPROCESS = CCRAB_DIR / "results_preprocessed" / "preprocess_dataset.jsonl"
CCRAB_FUNNEL = CCRAB_DIR / "results_pipeline_funnel"
SWECARE_DIR = DATASETS / "swe-care" / "data"
SWECARE_TEST = SWECARE_DIR / "test-00000-of-00001.parquet"

ORCH_ROOT = Path(__file__).resolve().parents[3]
JOURNAL_DIR = ORCH_ROOT / "orchestration"
RESEARCH_ROOT = Path("/mnt/raid0/llm/epyc-inference-research")
QUESTION_POOL = RESEARCH_ROOT / "benchmarks" / "prompts" / "question_pool.jsonl"

BUGREPORT_DIRS = [
    ORCH_ROOT / "bug-reports",
    Path("/mnt/raid0/llm/epyc-root") / "bug-reports",
]

OUTPUT_DIR = DATASETS / "nearmiss-corpus-v1"
STAGING_DIR = OUTPUT_DIR / "_staging"

# Candidate interpreters that carry pyarrow (SWE-CARE parquet mining). The
# orchestrator venv deliberately lacks pyarrow (no new pip deps), so the swecare
# miner self-relocates to one of these.
PYARROW_PYTHONS = [
    RESEARCH_ROOT / ".venv" / "bin" / "python",
    Path("/mnt/raid0/llm/omnidocbench") / ".venv" / "bin" / "python",
    Path("/mnt/raid0/llm/delta-Mem") / ".venv" / "bin" / "python",
]

# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
DOMAINS = {"code", "general", "hotpotqa", "simpleqa", "instruction_precision", "thinking"}
DEFECT_ORIGINS = {"natural", "seeded"}
GOLD_LABELS = {"accept", "reject", "pass", "fail", None}
GOLD_CONFIDENCES = {"multi_oracle", "single_oracle", "observation"}
SOURCE_BENCHMARKS = {"c-crab", "swe-care", "autopilot-journal", "seeded-mutation", "bug-report"}

# Map the eval-harness suite label (or dataset language) onto the RC-3 6-domain
# enum. The exact suite is always preserved in ``source_suite`` so this mapping
# is reversible / re-bucketable.
DOMAIN_MAP = {
    # code
    "code": "code", "coder": "code", "bigcodebench": "code", "cruxeval": "code",
    "debugbench": "code", "livecodebench": "code", "python": "code",
    # verbatim
    "hotpotqa": "hotpotqa", "simpleqa": "simpleqa",
    "instruction_precision": "instruction_precision", "thinking": "thinking",
    # reasoning-heavy -> thinking
    "math": "thinking", "gpqa": "thinking",
    # everything else -> general
    "general": "general", "vl": "general", "long_context": "general",
    "agentic": "general", "tool_use": "general", "mode_advantage": "general",
    "mode_advantage_hard": "general", "skill_transfer": "general",
    "real_suite_v1": "general",
}


def map_domain(suite: str | None) -> str:
    if not suite:
        return "general"
    return DOMAIN_MAP.get(str(suite).strip().lower(), "general")


# --------------------------------------------------------------------------- #
# Hashing / ids
# --------------------------------------------------------------------------- #
def stable_qid(suite: str, prompt_text: str) -> str:
    """Reproduce the eval-tower qid so journal outcomes join the question pool.

    Mirrors ``scripts/autopilot/eval_tower.py::_stable_question_qid`` exactly.
    """
    payload = f"{suite}\x00{prompt_text}".encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()[:16]


def _sha1(s: str, n: int = 16) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()[:n]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_row_id(source: str, key: str) -> str:
    return f"{CORPUS_ID}:{source}:{_sha1(key)}"


# --------------------------------------------------------------------------- #
# Row builder + validator
# --------------------------------------------------------------------------- #
_REQUIRED_KEYS = (
    "row_id", "corpus_id", "schema_version", "source_benchmark", "source_suite",
    "domain", "task", "candidate", "gold_label", "gold_source",
    "gold_instrument_version", "gold_confidence", "executable_oracle",
    "reasoning_module_labels", "rationale_gold_cause", "defect_origin",
    "ambiguous_tail", "natural_defect_control", "decontamination", "provenance",
)


def make_row(
    *,
    source_benchmark: str,
    source_suite: str | None,
    domain: str,
    task: str | None,
    candidate: str | None,
    gold_label: str | None,
    gold_source: str,
    gold_confidence: str,
    defect_origin: str,
    row_key: str,
    executable_oracle: dict | None = None,
    reasoning_module_labels: dict | None = None,
    rationale_gold_cause: str | None = None,
    ambiguous_tail: bool = False,
    natural_defect_control: bool = False,
    decontamination: dict | None = None,
    provenance: dict | None = None,
    gold_instrument_version: str | None = None,
) -> dict[str, Any]:
    """Assemble a schema-conformant row. ``row_key`` seeds the deterministic id."""
    if gold_instrument_version is None:
        gold_instrument_version = GOLD_INSTRUMENT_VERSIONS.get(source_benchmark, "unknown")
    return {
        "row_id": make_row_id(source_benchmark, row_key),
        "corpus_id": CORPUS_ID,
        "schema_version": SCHEMA_VERSION,
        "source_benchmark": source_benchmark,
        "source_suite": source_suite,
        "domain": domain,
        "task": task,
        "candidate": candidate,
        "gold_label": gold_label,
        "gold_source": gold_source,
        "gold_instrument_version": gold_instrument_version,
        "gold_confidence": gold_confidence,
        "executable_oracle": executable_oracle,
        "reasoning_module_labels": reasoning_module_labels,
        "rationale_gold_cause": rationale_gold_cause,
        "defect_origin": defect_origin,
        "ambiguous_tail": bool(ambiguous_tail),
        "natural_defect_control": bool(natural_defect_control),
        "decontamination": decontamination,
        "provenance": provenance or {},
    }


def validate_row(row: dict[str, Any]) -> list[str]:
    """Return a list of schema-violation messages (empty == valid)."""
    errs: list[str] = []
    for k in _REQUIRED_KEYS:
        if k not in row:
            errs.append(f"missing key: {k}")
    if errs:
        return errs
    if row["corpus_id"] != CORPUS_ID:
        errs.append(f"corpus_id != {CORPUS_ID}")
    if row["schema_version"] != SCHEMA_VERSION:
        errs.append("schema_version mismatch")
    if row["source_benchmark"] not in SOURCE_BENCHMARKS:
        errs.append(f"bad source_benchmark: {row['source_benchmark']}")
    if row["domain"] not in DOMAINS:
        errs.append(f"bad domain: {row['domain']}")
    if row["defect_origin"] not in DEFECT_ORIGINS:
        errs.append(f"bad defect_origin: {row['defect_origin']}")
    if row["gold_label"] not in GOLD_LABELS:
        errs.append(f"bad gold_label: {row['gold_label']}")
    if row["gold_confidence"] not in GOLD_CONFIDENCES:
        errs.append(f"bad gold_confidence: {row['gold_confidence']}")
    if not isinstance(row["ambiguous_tail"], bool):
        errs.append("ambiguous_tail must be bool")
    if not isinstance(row["natural_defect_control"], bool):
        errs.append("natural_defect_control must be bool")
    for k in ("executable_oracle", "reasoning_module_labels", "decontamination"):
        if row[k] is not None and not isinstance(row[k], dict):
            errs.append(f"{k} must be dict or null")
    if not isinstance(row["provenance"], dict):
        errs.append("provenance must be dict")
    if not row["gold_source"]:
        errs.append("gold_source is required (non-empty)")
    # At least one gold field must carry signal (dual-gold: either may be null,
    # but not both when the row claims a hard label).
    if row["gold_label"] in {"accept", "reject", "pass", "fail"}:
        if row["executable_oracle"] is None and row["reasoning_module_labels"] is None:
            errs.append("labeled row has neither executable_oracle nor reasoning_module_labels")
    # Single-oracle gate-worthy rows must be routed to arbitration.
    if row["gold_confidence"] == "single_oracle" and not row["ambiguous_tail"]:
        # allowed but flagged: single-oracle should generally be ambiguous_tail.
        # We enforce it as a hard rule to keep the gate honest.
        errs.append("single_oracle rows must set ambiguous_tail=true (arbitration)")
    return errs


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def truncate(text: str | None, limit: int = 20000) -> str | None:
    """Bound very large candidate/task payloads (diffs) to keep the corpus lean."""
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


# --------------------------------------------------------------------------- #
# Question pool loader (shared by journals + seeded mutations)
# --------------------------------------------------------------------------- #
def load_question_pool(path: str | os.PathLike[str] | None = None) -> dict[str, dict]:
    """Load the research question pool keyed by the eval-harness qid.

    Returns ``{qid: {suite, prompt, expected, scoring_method, scoring_config,
    tier, dataset_source, id}}``. The first line is a ``__pool_metadata__``
    header and is skipped.
    """
    p = Path(path) if path else QUESTION_POOL
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i == 0:  # metadata header
                continue
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("__pool_metadata__"):
                continue
            suite = str(r.get("suite", ""))
            prompt = str(r.get("prompt", ""))
            qid = stable_qid(suite, prompt)
            out[qid] = {
                "id": r.get("id"),
                "suite": suite,
                "prompt": prompt,
                "expected": r.get("expected"),
                "scoring_method": r.get("scoring_method"),
                "scoring_config": r.get("scoring_config"),
                "tier": r.get("tier"),
                "dataset_source": r.get("dataset_source"),
            }
    return out


def journal_shards() -> list[Path]:
    """All journal shards, base + rotated, sorted (read-only inputs)."""
    shards = []
    base = JOURNAL_DIR / "autopilot_journal.jsonl"
    if base.exists():
        shards.append(base)
    for p in sorted(JOURNAL_DIR.glob("autopilot_journal_*.jsonl")):
        shards.append(p)
    return shards


def iter_journal_question_results() -> Iterator[tuple[dict, dict]]:
    """Yield ``(trial_row, question_result)`` across all journal shards.

    ``eval_details`` may be a JSON string or a dict; question results live under
    ``eval_details.question_results``.
    """
    for shard in journal_shards():
        with open(shard, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ed = row.get("eval_details")
                if isinstance(ed, str):
                    try:
                        ed = json.loads(ed)
                    except json.JSONDecodeError:
                        continue
                if not isinstance(ed, dict):
                    continue
                qrs = ed.get("question_results")
                if not isinstance(qrs, list):
                    continue
                for qr in qrs:
                    if isinstance(qr, dict) and qr.get("qid"):
                        yield row, qr
