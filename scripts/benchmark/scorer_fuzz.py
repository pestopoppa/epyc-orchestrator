#!/usr/bin/env python3
"""EV-SCORER-FUZZ-1: offline fuzz harness for the deterministic eval_tower scorers.

Source: intake-2889 ("Before the Model Learns the Bug: Fuzzing RLVR Verifiers").
The paper differential-tests a verifier against a stricter reference contract using
mutation categories of a COMPLETION: ``partial_answer`` (gold value in prose, no final
answer marker), ``missing_marker``, ``contradictory_finals``, ``loose_float`` and
wrong-then-correct. A deterministic verifier that accepts a mutated completion the strict
contract rejects has a false-positive region an optimiser can find in a handful of queries.

This harness replays those categories over GSM8K / MATH-500 gold answers against the two
scorers that gate the math suites in ``debug_scorer.score_answer``:

* ``math_verify`` (EV-11; the production math scorer), and
* ``exact_match`` (the extract-and-compare scorer: ``<answer>``/``####``/``\\boxed``/last-line).

No model is called and no server is touched: pure Python over ``score_answer``.
The strict contract is declared PER TEMPLATE (``Case.strict_accept``), so every FPR/FNR is
relative to that declared contract, not to ground truth about the scorer's intent. A scorer
behaviour that is documented as by-design (for example exact_match's last-line fallback) is
still reported against the strict contract and is labelled ``by_design`` in the findings.

Metrics per (scorer, source, category[, template]):

* FPR = accepted / strict-reject cases (the verifier accepts what the contract rejects),
* FNR = rejected / strict-accept cases,
* acceptance = accepted / scored, coverage = scored / n (a case the scorer refuses, e.g.
  ``ScoringUnavailableError`` on a gold it cannot parse, is EXCLUDED, never counted wrong).

Usage::

    # one-off: regenerate the committed gold fixture (needs pyarrow for GSM8K parquet)
    python scripts/benchmark/scorer_fuzz.py --build-gold tests/fixtures/scorer_fuzz/gold_v1.jsonl

    # the fuzz run (needs the ``math-verify`` package; refuses without it, EV-11)
    python scripts/benchmark/scorer_fuzz.py --out-dir artifacts/scorer_fuzz

Default OFF: nothing imports or runs this from the runtime; it is an operator/CLI tool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "scorer_fuzz" / "gold_v1.jsonl"

SCORERS = ("math_verify", "exact_match")
# The v1 report also ran two candidate variants (math_verify_goldwrap,
# math_verify_strict_candidate) that sized the fixes. Both landed in production at scorer
# semantics v2 (debug_scorer.SCORER_SEMANTICS_VERSION), so the variants were removed.
SCHEMA = "epyc.scorer_fuzz.v2"

CATEGORIES = (
    "clean_correct",  # control: strict accept
    "clean_wrong",  # control: strict reject
    "format_variants",  # control: strict accept (formatting noise around a correct final)
    "partial_answer",
    "missing_marker",
    "contradictory_finals",
    "loose_float",
    "wrong_then_correct",
    "gold_prefix_collision",  # extra (not in the intake-2889 five): a gold-parse defect probe
)

_NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")
# A gold suffix that only decorates the leading number: degrees, a base subscript, percent.
_CONVENTION_SUFFIX = re.compile(r"\^\{?\\circ\}?|_\{?\d+\}?|\\%")

# Per-source scorer config exactly as the dataset adapters declare it
# (scripts/benchmark/dataset_adapter_modules/math_adapter.py).
_MV_CONFIG = {"gsm8k": {"extraction_mode": "expr"}, "math500": {"extraction_mode": "latex"}}


# ── scorer loading ───────────────────────────────────────────────────────

_SCORER_KEY = "epyc_orch_debug_scorer"


def load_debug_scorer():
    """Load THIS repo's debug_scorer by path under the private key seeding_scoring uses."""
    cached = sys.modules.get(_SCORER_KEY)
    if cached is not None:
        return cached
    path = REPO_ROOT / "scripts" / "benchmark" / "debug_scorer.py"
    spec = importlib.util.spec_from_file_location(_SCORER_KEY, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load debug_scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SCORER_KEY] = module
    spec.loader.exec_module(module)
    return module


def require_math_verify() -> None:
    try:
        import math_verify  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "scorer_fuzz: the math-verify package is required (EV-11 hard-fail: a fuzz run "
            "against a silently degraded scorer would be meaningless)"
        ) from exc


# ── gold fixture ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Gold:
    item_id: str
    source: str  # gsm8k | math500
    gold: str


def build_gold(n_per_source: int = 100, seed: int = 20261124) -> list[Gold]:
    """Sample a seeded gold set from the on-disk GSM8K test split and MATH-500 (one-off)."""
    import glob

    hub = Path("/mnt/raid0/llm/hf-home/hub")
    out: list[Gold] = []
    rng = random.Random(seed)

    import pyarrow.parquet as pq  # one-off builder only

    parquet = sorted(glob.glob(str(hub / "datasets--gsm8k/snapshots/*/main/test-*.parquet")))[0]
    rows = pq.read_table(parquet).to_pylist()
    for idx in sorted(rng.sample(range(len(rows)), n_per_source)):
        m = re.search(r"####\s*(.+)", rows[idx]["answer"])
        assert m, f"gsm8k row {idx} has no #### answer"
        out.append(Gold(f"gsm8k_{idx:05d}", "gsm8k", m.group(1).strip().replace(",", "")))

    jsonl = sorted(glob.glob(str(hub / "datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl")))[0]
    mrows = [json.loads(line) for line in Path(jsonl).read_text().splitlines() if line.strip()]
    for idx in sorted(rng.sample(range(len(mrows)), n_per_source)):
        out.append(Gold(f"math500_{idx:05d}", "math500", str(mrows[idx]["answer"]).strip()))
    return out


def write_gold(path: Path, golds: Sequence[Gold]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for g in golds:
            fh.write(json.dumps(asdict(g), ensure_ascii=False, sort_keys=True) + "\n")


def read_gold(path: Path = DEFAULT_FIXTURE) -> list[Gold]:
    golds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            golds.append(Gold(d["item_id"], d["source"], d["gold"]))
    return golds


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── mutation generators ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    case_id: str
    item_id: str
    source: str
    category: str
    template: str
    answer: str  # the mutated completion handed to the scorer
    expected: str  # gold
    strict_accept: bool  # the strict contract's verdict for this completion


def _wrong_for(gold: str, others: Sequence[str]) -> str:
    """A deterministic wrong final that differs from gold."""
    if _NUMERIC.match(gold):
        d = Decimal(gold)
        return format(d + (7 if d == d.to_integral() else Decimal("7.5")), "f")
    for cand in others:
        if cand != gold:
            return cand
    return "0"


def _drift(gold: str, delta: str) -> str:
    return format(Decimal(gold) + Decimal(delta), "f")


def _templates(gold: Gold, wrong: str) -> list[tuple[str, str, str, bool]]:
    """(category, template, completion, strict_accept) for one gold item."""
    g, w = gold.gold, wrong
    reason = "Working through the problem step by step."
    t: list[tuple[str, str, str, bool]] = [
        # controls
        ("clean_correct", "boxed", f"{reason} Therefore the final answer is $\\boxed{{{g}}}$.", True),
        ("clean_wrong", "boxed", f"{reason} Therefore the final answer is $\\boxed{{{w}}}$.", False),
        # formatting noise around a CORRECT single final: the strict contract accepts all of these
        ("format_variants", "boxed_bare", f"{reason}\n\\boxed{{{g}}}", True),
        ("format_variants", "boxed_spaced", f"{reason}\nFinal answer: \\boxed{{ {g} }}", True),
        ("format_variants", "boxed_bold", f"{reason}\n**\\boxed{{{g}}}**", True),
        # partial_answer: the gold value appears in prose, but no final answer is committed
        (
            "partial_answer",
            "considered",
            f"{reason} I considered whether the result could be {g}, but I did not finish the computation.",
            False,
        ),
        (
            "partial_answer",
            "intermediate",
            f"{reason} In step 2 the intermediate quantity {g} appears; the rest is left undone.",
            False,
        ),
        # missing_marker: an uncommitted (unmarked) correct claim
        ("missing_marker", "sentence", f"{reason} So the answer is {g}.", False),
        ("missing_marker", "bare_last_line", f"{reason}\n{g}", False),
        # contradictory_finals: two conflicting committed finals, either order
        (
            "contradictory_finals",
            "wrong_then_gold_boxed",
            f"\\boxed{{{w}}} is my first result. On reflection another route gives \\boxed{{{g}}}.",
            False,
        ),
        (
            "contradictory_finals",
            "gold_then_wrong_boxed",
            f"\\boxed{{{g}}} is my first result. On reflection another route gives \\boxed{{{w}}}.",
            False,
        ),
        (
            "contradictory_finals",
            "gold_then_wrong_answer_tags",
            f"<answer>{g}</answer> Hmm, recomputing: <answer>{w}</answer>",
            False,
        ),
        (
            "contradictory_finals",
            "wrong_then_gold_hashes",
            f"#### {w}\nRe-checking the arithmetic.\n#### {g}",
            False,
        ),
        # wrong_then_correct: a wrong attempt in prose, explicitly retracted, ONE committed final
        (
            "wrong_then_correct",
            "retracted",
            f"First attempt: the result is {w}. Wait, that is wrong because I mis-added. "
            f"Correcting: the final answer is \\boxed{{{g}}}.",
            True,
        ),
        (
            "wrong_then_correct",
            "retracted_tags",
            f"My first guess was {w}; I now see that is a mistake. <answer>{g}</answer>",
            True,
        ),
    ]
    lead = re.match(r"-?\d+", g)
    suffix = g[lead.end() :] if lead else ""
    if not _NUMERIC.match(g) and lead and _CONVENTION_SUFFIX.fullmatch(suffix):
        # NOT a collision: the gold is the leading number plus a unit/base decoration
        # (106^\circ, 2516_8, 25\%), which MATH grading conventions (and math-verify's
        # normaliser) treat as the same answer. Under schema v1 these were labelled
        # strict-reject collisions; they were the "2 residual" cases.
        t.append(
            (
                "format_variants",
                "suffix_dropped",
                f"{reason} Therefore the final answer is $\\boxed{{{lead.group()}}}$.",
                True,
            )
        )
    elif not _NUMERIC.match(g) and lead and lead.group() != g:
        # gold_prefix_collision: a WRONG boxed final equal to the leading number of a
        # non-numeric gold (3 for 3\sqrt{13}). Strict reject; it is accepted when the scorer
        # mis-parses a bare LaTeX gold down to its leading number.
        t.append(
            (
                "gold_prefix_collision",
                "leading_number",
                f"{reason} Therefore the final answer is $\\boxed{{{lead.group()}}}$.",
                False,
            )
        )
    if _NUMERIC.match(g):
        # loose_float: numerically near but not equal; the strict contract is exact equality
        for label, delta in (("1e-3", "0.001"), ("1e-5", "0.00001"), ("1e-7", "0.0000001")):
            t.append(
                (
                    "loose_float",
                    f"drift_{label}",
                    f"{reason} Therefore the final answer is $\\boxed{{{_drift(g, delta)}}}$.",
                    False,
                )
            )
        t.append(
            (
                "loose_float",
                "trailing_zero_equal",
                f"{reason} Therefore the final answer is $\\boxed{{{g}.0}}$.",
                True,  # same value: the strict contract accepts it
            )
        )
    return t


def generate_cases(golds: Sequence[Gold]) -> list[Case]:
    by_source: dict[str, list[str]] = {}
    for g in golds:
        by_source.setdefault(g.source, []).append(g.gold)
    cases: list[Case] = []
    for i, g in enumerate(golds):
        pool = by_source[g.source]
        others = pool[i % len(pool) :] + pool[: i % len(pool)]
        wrong = _wrong_for(g.gold, others[1:] or others)
        for category, template, answer, strict in _templates(g, wrong):
            cases.append(
                Case(
                    case_id=f"{g.item_id}:{category}:{template}",
                    item_id=g.item_id,
                    source=g.source,
                    category=category,
                    template=template,
                    answer=answer,
                    expected=g.gold,
                    strict_accept=strict,
                )
            )
    return cases


def cases_digest(cases: Iterable[Case]) -> str:
    h = hashlib.sha256()
    for c in cases:
        h.update(json.dumps(asdict(c), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ── scoring and aggregation ──────────────────────────────────────────────


def score_case(scorer: str, case: Case, module: Any | None = None) -> str:
    """Return ``accept`` | ``reject`` | ``excluded`` | ``error``."""
    module = module or load_debug_scorer()
    if scorer not in SCORERS:
        raise ValueError(f"unknown scorer {scorer!r}")
    cfg: dict[str, Any] = dict(_MV_CONFIG[case.source]) if scorer == "math_verify" else {}
    unavailable = getattr(module, "ScoringUnavailableError", ())
    try:
        return "accept" if module.score_answer(case.answer, case.expected, scorer, cfg) else "reject"
    except unavailable:
        return "excluded"
    except Exception:  # noqa: BLE001 - an unexpected scorer crash is its own outcome
        return "error"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n) ** 0.5
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def run_fuzz(cases: Sequence[Case], scorers: Sequence[str] = SCORERS) -> list[dict[str, Any]]:
    module = load_debug_scorer()
    rows = []
    for scorer in scorers:
        for case in cases:
            rows.append(
                {
                    "scorer": scorer,
                    "case_id": case.case_id,
                    "source": case.source,
                    "category": case.category,
                    "template": case.template,
                    "strict_accept": case.strict_accept,
                    "outcome": score_case(scorer, case, module),
                }
            )
    return rows


def aggregate(rows: Sequence[dict[str, Any]], by_template: bool = False) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["scorer"], r["source"], r["category"]) + ((r["template"],) if by_template else ())
        groups.setdefault(key, []).append(r)
    out = []
    for key in sorted(groups):
        rs = groups[key]
        scored = [r for r in rs if r["outcome"] in ("accept", "reject")]
        neg = [r for r in scored if not r["strict_accept"]]
        pos = [r for r in scored if r["strict_accept"]]
        fp = sum(1 for r in neg if r["outcome"] == "accept")
        fn = sum(1 for r in pos if r["outcome"] == "reject")
        accepted = sum(1 for r in scored if r["outcome"] == "accept")
        row = {
            "scorer": key[0],
            "source": key[1],
            "category": key[2],
            "n": len(rs),
            "scored": len(scored),
            "excluded": sum(1 for r in rs if r["outcome"] == "excluded"),
            "errors": sum(1 for r in rs if r["outcome"] == "error"),
            "n_strict_reject": len(neg),
            "n_strict_accept": len(pos),
            "fp": fp,
            "fn": fn,
            "fpr": (fp / len(neg)) if neg else None,
            "fpr_ci95": wilson(fp, len(neg)) if neg else None,
            "fnr": (fn / len(pos)) if pos else None,
            "fnr_ci95": wilson(fn, len(pos)) if pos else None,
            "acceptance": (accepted / len(scored)) if scored else None,
            "coverage": (len(scored) / len(rs)) if rs else None,
        }
        if by_template:
            row["template"] = key[3]
        out.append(row)
    return out


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def render_markdown(table: Sequence[dict[str, Any]], by_template: bool = False) -> str:
    head = ["scorer", "source", "category"] + (["template"] if by_template else [])
    head += ["n", "FPR (fp/neg)", "FNR (fn/pos)", "acceptance", "coverage"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in table:
        cells = [r["scorer"], r["source"], r["category"]] + ([r["template"]] if by_template else [])
        cells += [
            str(r["n"]),
            f"{_fmt(r['fpr'])} ({r['fp']}/{r['n_strict_reject']})",
            f"{_fmt(r['fnr'])} ({r['fn']}/{r['n_strict_accept']})",
            _fmt(r["acceptance"]),
            _fmt(r["coverage"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def findings(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every (scorer, category, template) cell with a non-zero FPR or FNR, with an example."""
    table = aggregate(rows, by_template=True)
    example: dict[tuple, str] = {}
    for r in rows:
        bad = (r["outcome"] == "accept" and not r["strict_accept"]) or (
            r["outcome"] == "reject" and r["strict_accept"]
        )
        if bad:
            example.setdefault((r["scorer"], r["source"], r["category"], r["template"]), r["case_id"])
    out = []
    for t in table:
        if t["fp"] or t["fn"]:
            key = (t["scorer"], t["source"], t["category"], t["template"])
            out.append({**t, "example_case_id": example.get(key)})
    return out


def run_report(
    fixture: Path = DEFAULT_FIXTURE, scorers: Sequence[str] = SCORERS
) -> dict[str, Any]:
    golds = read_gold(fixture)
    cases = generate_cases(golds)
    rows = run_fuzz(cases, scorers)
    return {
        "schema": SCHEMA,
        "fixture": str(fixture.relative_to(REPO_ROOT)) if fixture.is_relative_to(REPO_ROOT) else str(fixture),
        "gold_fixture_sha256": file_sha256(fixture),
        "cases_digest_sha256": cases_digest(cases),
        "n_gold": len(golds),
        "n_cases": len(cases),
        "scorers": list(scorers),
        "table": aggregate(rows),
        "table_by_template": aggregate(rows, by_template=True),
        "findings": findings(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--out-dir", type=Path, default=None, help="write report.json + report.md here")
    ap.add_argument("--build-gold", type=Path, default=None, help="(re)generate the gold fixture and exit")
    ap.add_argument("--n-per-source", type=int, default=100)
    args = ap.parse_args(argv)

    if args.build_gold:
        golds = build_gold(args.n_per_source)
        write_gold(args.build_gold, golds)
        print(f"wrote {len(golds)} gold rows to {args.build_gold} sha256={file_sha256(args.build_gold)}")
        return 0

    require_math_verify()
    report = run_report(args.fixture)
    md = render_markdown(report["table"])
    print(md)
    print(f"cases_digest_sha256={report['cases_digest_sha256']} gold_sha256={report['gold_fixture_sha256']}")
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (args.out_dir / "report.md").write_text(
            md + "\n## By template\n\n" + render_markdown(report["table_by_template"], by_template=True)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
