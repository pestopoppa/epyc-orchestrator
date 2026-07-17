#!/usr/bin/env python3
"""Source 4: rule-based seeded-defect mutations (RC-3). NO inference.

Both natural sources (c-CRAB, SWE-CARE) are 100% *natural* defects and are
almost all "reject" candidates. To make FALSE-REJECT measurable we need clean
"accept" candidates, and to get clean-cause "reject" candidates with a KNOWN WHY
we synthesize seeded defects deterministically from known-good items:

  * Non-code (source 3 gold answers, recovered via the journal<->pool join):
    known-good = the reference gold answer for a really-evaluated question.
      - seeded reject  = a rule-mutated answer (off-by-one / negation / wrong
                         option / entity substitution)   -> defect_origin=seeded
      - accept control = the gold answer itself           -> defect_origin=natural
  * Code (source 1/2 merged_patch, the accepted revision): a single same-line
    operator/boolean flip inside an added diff line       -> defect_origin=seeded

All mutations are deterministic (content-hash-seeded) and rule-based; the
mutation rule IS the ground-truth cause (recorded in rationale_gold_cause and
reasoning_module_labels). ``gold_confidence="multi_oracle"`` here denotes
synthetic ground truth (reference gold + deterministic rule).

instruction_precision gold answers are mostly empty strings (scoring is
programmatic on prompt constraints), so IP seeded coverage is limited by design
-- IP is still covered by the journal-outcome partition.
"""
from __future__ import annotations

import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from corpus_v1 import common
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from corpus_v1 import common


def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8", "replace")).hexdigest(), 16)


# --------------------------------------------------------------------------- #
# Pure mutation primitives (imported by tests)
# --------------------------------------------------------------------------- #
_LETTERS = ["A", "B", "C", "D", "E"]
_BOOL_FLIP = {"yes": "no", "no": "yes", "true": "false", "false": "true"}


def mutate_multiple_choice(expected: str, salt: str = ""):
    e = expected.strip().upper()
    if e in _LETTERS:
        others = [x for x in _LETTERS if x != e]
        return others[_h(salt + e) % len(others)], "wrong_option_selected"
    return None


def mutate_numeric(expected: str):
    s = expected.strip().replace(",", "")
    if re.fullmatch(r"-?\d+", s):
        return str(int(s) + 1), "numeric_off_by_one"
    if re.fullmatch(r"-?\d+\.\d+", s):
        decimals = len(s.split(".", 1)[1])
        bumped = float(s) + 10 ** (-decimals)
        return f"{bumped:.{decimals}f}", "numeric_off_by_one"
    return None


def mutate_boolean(expected: str):
    key = expected.strip().lower()
    if key in _BOOL_FLIP:
        flipped = _BOOL_FLIP[key]
        if expected.strip().istitle():
            flipped = flipped.capitalize()
        elif expected.strip().isupper():
            flipped = flipped.upper()
        return flipped, "negation_flip"
    return None


def mutate_entity(expected: str, alt_pool, salt: str = ""):
    e = expected.strip().lower()
    cands = [
        a for a in alt_pool
        if a and len(a.strip()) > 1 and a.strip().lower() != e
        and e not in a.strip().lower() and a.strip().lower() not in e
    ]
    if not cands:
        return None
    cands = sorted(set(cands))
    return cands[_h(salt + expected) % len(cands)], "entity_substitution"


# same-length or line-count-preserving in-line flips keep a diff structurally valid
_OP_FLIPS = [("==", "!="), ("!=", "=="), ("<=", ">="), (">=", "<=")]


def mutate_code_token(code: str):
    """Flip one operator/boolean token inside the first eligible added ('+') line."""
    lines = code.split("\n")
    for i, ln in enumerate(lines):
        if not ln.startswith("+") or ln.startswith("+++"):
            continue
        body = ln[1:]
        for a, b in _OP_FLIPS:
            if a in body:
                lines[i] = "+" + body.replace(a, b, 1)
                return "\n".join(lines), f"code_operator_flip[{a}->{b}]"
        for tok, rep in (("True", "False"), ("False", "True")):
            m = re.search(rf"\b{tok}\b", body)
            if m:
                lines[i] = "+" + body[: m.start()] + rep + body[m.end():]
                return "\n".join(lines), f"code_bool_flip[{tok}->{rep}]"
    return None


def seed_answer(scoring_method, expected, alt_pool, salt: str):
    """Dispatch to the first applicable rule. Returns (candidate, cause) or None."""
    e = str(expected).strip()
    if not e:
        return None
    if scoring_method == "multiple_choice":
        return mutate_multiple_choice(e, salt)
    for fn in (mutate_numeric, mutate_boolean):
        out = fn(e)
        if out:
            return out
    return mutate_entity(e, alt_pool, salt)


# --------------------------------------------------------------------------- #
# Miner
# --------------------------------------------------------------------------- #
def _read_staging_accept_code_rows() -> list[dict]:
    rows = []
    for name in ("ccrab.jsonl", "swecare.jsonl"):
        p = common.STAGING_DIR / name
        if not p.exists():
            continue
        for r in common.read_jsonl(p):
            prov = r.get("provenance") or {}
            if prov.get("candidate_is") == "merged_patch" and r.get("candidate"):
                rows.append(r)
    return rows


def mine() -> tuple[list[dict], dict]:
    pool = common.load_question_pool()

    # distinct journal-evaluated qids -> known-run questions
    qids: set[str] = set()
    for _trial, qr in common.iter_journal_question_results():
        qids.add(qr["qid"])

    # per-suite alternate-answer pools for entity substitution
    alt_by_suite: dict[str, list[str]] = defaultdict(list)
    for qid in qids:
        pinfo = pool.get(qid)
        if pinfo and pinfo.get("expected") not in (None, ""):
            alt_by_suite[pinfo["suite"]].append(str(pinfo["expected"]))

    rows: list[dict] = []
    stats = {"noncode_pairs": 0, "noncode_skipped_unmutatable": 0, "code_seeded": 0}

    for qid in sorted(qids):
        pinfo = pool.get(qid)
        if not pinfo:
            continue
        suite = pinfo["suite"]
        domain = common.map_domain(suite)
        expected = pinfo.get("expected")
        scoring_method = pinfo.get("scoring_method")
        task = common.truncate(pinfo["prompt"], 8000)
        mutated = seed_answer(scoring_method, expected, alt_by_suite.get(suite, []), salt=qid)
        if not mutated:
            stats["noncode_skipped_unmutatable"] += 1
            continue
        cand, cause = mutated
        ref = str(expected)

        # seeded reject
        rows.append(
            common.make_row(
                source_benchmark="seeded-mutation",
                source_suite=suite,
                domain=domain,
                task=task,
                candidate=common.truncate(cand, 4000),
                gold_label="reject",
                gold_source="seeded_mutation_rule",
                gold_confidence="multi_oracle",  # synthetic ground truth
                defect_origin="seeded",
                row_key=f"seed|{qid}|{cause}",
                executable_oracle=None,
                reasoning_module_labels={
                    "source": "seed_rule",
                    "cause": cause,
                    "reference_gold_answer": common.truncate(ref, 2000),
                },
                rationale_gold_cause=f"seeded {cause}: gold='{common.truncate(ref, 200)}' "
                f"mutated to '{common.truncate(cand, 200)}'",
                ambiguous_tail=False,
                natural_defect_control=False,
                decontamination=None,
                provenance={
                    "qid": qid,
                    "suite": suite,
                    "scoring_method": scoring_method,
                    "mutation_rule": cause,
                    "base_source": "autopilot-journal/question_pool",
                    "dataset_source": pinfo.get("dataset_source"),
                    "reference_gold_answer": common.truncate(ref, 2000),
                },
            )
        )
        # paired accept-control (the natural reference gold answer)
        rows.append(
            common.make_row(
                source_benchmark="seeded-mutation",
                source_suite=suite,
                domain=domain,
                task=task,
                candidate=common.truncate(ref, 4000),
                gold_label="accept",
                gold_source="reference_gold_answer",
                gold_confidence="multi_oracle",
                defect_origin="natural",
                row_key=f"seed-accept|{qid}",
                executable_oracle=None,
                reasoning_module_labels={
                    "source": "reference_gold_answer",
                    "note": "verified reference answer (clean accept control)",
                },
                rationale_gold_cause=None,
                ambiguous_tail=False,
                natural_defect_control=False,
                decontamination=None,
                provenance={
                    "qid": qid,
                    "suite": suite,
                    "scoring_method": scoring_method,
                    "clean_control": True,
                    "paired_with_seeded_rule": cause,
                },
            )
        )
        stats["noncode_pairs"] += 1

    # code seeded: single-token flips on accepted merged patches (source 1/2)
    for base in _read_staging_accept_code_rows():
        out = mutate_code_token(base["candidate"])
        if not out:
            continue
        mutated_code, cause = out
        decontam = base.get("decontamination")
        inst = (base.get("provenance") or {}).get("instance_id")
        rows.append(
            common.make_row(
                source_benchmark="seeded-mutation",
                source_suite=base.get("source_suite"),
                domain="code",
                task=base.get("task"),
                candidate=common.truncate(mutated_code, 20000),
                gold_label="reject",
                gold_source="seeded_mutation_rule",
                gold_confidence="multi_oracle",
                defect_origin="seeded",
                row_key=f"seed-code|{inst}|{cause}",
                executable_oracle=None,
                reasoning_module_labels={
                    "source": "seed_rule",
                    "cause": cause,
                    "base": "merged_patch (accepted revision)",
                },
                rationale_gold_cause=f"seeded {cause} injected into an accepted diff line",
                ambiguous_tail=False,
                natural_defect_control=False,
                decontamination=decontam,
                provenance={
                    "instance_id": inst,
                    "mutation_rule": cause,
                    "base_source": base.get("source_benchmark"),
                    "base_row_id": base.get("row_id"),
                },
            )
        )
        stats["code_seeded"] += 1

    return rows, stats


def main() -> int:
    rows, stats = mine()
    out = common.STAGING_DIR / "seeded.jsonl"
    n = common.write_jsonl(out, rows)
    print(f"[seed_mutations] wrote {n} rows -> {out}  stats={stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
