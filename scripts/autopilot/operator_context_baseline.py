#!/usr/bin/env python3
"""AP-ME-1 offline baseline: current per-operator context size, before any budget change.

AP-ME-1 asks for a per-operator context bound and, offline first, a measurement of
the current per-operator context size so the change has a baseline to beat. Today
the only journal-derived context is the AP-22 short-term memory (STM): one view,
rendered by ``stm_generated_view.render_generated_stm`` with a single shared bound
(``ShortTermMemory.refresh_from_journal`` defaults, 30 eligible entries / 2000
tokens) and injected into the controller prompt whatever operator is chosen next.

This census reconstructs, with the CURRENT renderer, the STM the controller would
see at each historical decision point (journal prefix strictly before the trial,
all rotated shards, supersessions folded) and attributes it to the operator that
was dispatched at that point. Per operator it reports the rendered size and the
card composition in the OpenMLE-Evo vocabulary the box seeds from:

* ``cards``: distinct prior trials cited (``[tN]`` tags), the related-card count;
* ``ancestor_cards``: cited trials on the dispatched trial's parent chain;
* ``sibling_cards``: cited trials sharing its parent;
* ``same_operator_cards``: cited trials dispatched with the same operator;
* ``failure_cards``: cited trials in the Failure Patterns section (debug analogue).

Operator = ``action_type``, refined by the ``mutation`` sub-type for prompt and code
mutations. OpenMLE class mapping (declared, not inferred): ``crossover`` and
``style_transfer`` are crossover; every other dispatched operator is improve; EPYC
dispatches no debug operator, so the debug seed has no current counterpart.

Prompt-author inputs beyond the STM (prompt file text, failure context) are not
journaled; only the controller-written ``description`` length is recoverable. The
default-off ``AUTOPILOT_AUTHOR_CONTEXT_CAPTURE`` sidecar is counted if present.

Stop rule: deterministic census of every decision point in the retained journal; no
sampling, one pass, complete when every trial entry has a reconstructed context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from context_budget import chars_for_tokens  # noqa: E402
from experiment_journal import ExperimentJournal  # noqa: E402
from journal_shards import journal_shards  # noqa: E402
from stm_generated_view import (  # noqa: E402
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_LAST_N,
    render_generated_stm,
)

SCHEMA = "epyc.autopilot.operator_context_baseline.v1"
TAG_RE = re.compile(r"\[t(\d+)\]")
TRUNCATION_MARKER = "...[truncated]"
SECTIONS = ("Running Hypotheses", "Optimization Directions", "Failure Patterns")
CROSSOVER_MUTATIONS = frozenset({"crossover", "style_transfer"})
OPENMLE_SEEDS = {
    "global": {"max_related_cards": 3},
    "improve": {"ancestor_k": 3, "sibling_k": 3},
    "crossover": {"ancestor_k": 2, "sibling_k": 2},
    "debug": {"max_related_cards": 8},
    "source": "OpenMLE-Evo tts_search/configs/search/airaevo.yaml (as quoted in AP-ME-1)",
}


def operator_label(entry: Any) -> str:
    snapshot = entry.config_snapshot if isinstance(entry.config_snapshot, dict) else {}
    mutation = snapshot.get("mutation")
    if entry.action_type in ("prompt_mutation", "code_mutation") and isinstance(mutation, str):
        return f"{entry.action_type}:{mutation}"
    return str(entry.action_type)


def openmle_class(label: str) -> str:
    return "crossover" if label.split(":")[-1] in CROSSOVER_MUTATIONS else "improve"


def section_tags(text: str) -> dict[str, set[int]]:
    """Trial tags cited per STM section."""
    out: dict[str, set[int]] = {name: set() for name in SECTIONS}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            continue
        if current in out:
            out[current].update(int(t) for t in TAG_RE.findall(line))
    return out


def ancestors(trial_id: int, parent_of: dict[int, int | None]) -> set[int]:
    seen: set[int] = set()
    node = parent_of.get(trial_id)
    while node is not None and node not in seen:
        seen.add(node)
        node = parent_of.get(node)
    return seen


def measure_decision(entry: Any, stm: str, *, parent_of: dict[int, int | None],
                     children_of: dict[int | None, list[int]],
                     operator_of: dict[int, str]) -> dict[str, Any]:
    tags = section_tags(stm)
    cards = set().union(*tags.values())
    tid = int(entry.trial_id)
    label = operator_label(entry)
    anc = ancestors(tid, parent_of)
    sibs = {c for c in children_of.get(entry.parent_trial, []) if c != tid and c < tid}
    snapshot = entry.config_snapshot if isinstance(entry.config_snapshot, dict) else {}
    description = snapshot.get("description")
    return {
        "trial_id": tid, "operator": label, "openmle_class": openmle_class(label),
        "chars": len(stm), "approx_tokens": len(stm) / chars_for_tokens(1),
        "lines": len(stm.splitlines()), "truncated": TRUNCATION_MARKER in stm,
        "cards": len(cards),
        "hypothesis_cards": len(tags["Running Hypotheses"]),
        "direction_cards": len(tags["Optimization Directions"]),
        "failure_cards": len(tags["Failure Patterns"]),
        "ancestor_cards": len(cards & anc), "ancestors_available": len(anc),
        "sibling_cards": len(cards & sibs), "siblings_available": len(sibs),
        "same_operator_cards": sum(1 for c in cards if operator_of.get(c) == label),
        "description_chars": len(description) if isinstance(description, str) else None,
    }


def _summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    return {"min": ordered[0], "median": statistics.median(ordered), "p90": p90,
            "max": ordered[-1], "mean": round(statistics.fmean(ordered), 4)}


def summarize(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    fields = ("approx_tokens", "lines", "cards", "hypothesis_cards", "direction_cards",
              "failure_cards", "ancestor_cards", "ancestors_available", "sibling_cards",
              "siblings_available", "same_operator_cards")
    out: dict[str, Any] = {}
    for name in sorted(groups):
        group = groups[name]
        out[name] = {"decisions": len(group),
                     "truncated_rate": sum(r["truncated"] for r in group) / len(group)}
        for f in fields:
            out[name][f] = _summary([r[f] for r in group])
        desc = [r["description_chars"] for r in group if r["description_chars"] is not None]
        out[name]["description_chars"] = _summary(desc)
    return out


def census(entries: Sequence[Any], *, last_n: int = DEFAULT_LAST_N,
           budget_tokens: int = DEFAULT_BUDGET_TOKENS) -> list[dict[str, Any]]:
    """Decision-time STM per trial, rendered from the strict prefix before it."""
    ordered = sorted(entries, key=lambda e: int(e.trial_id))
    parent_of = {int(e.trial_id): (int(e.parent_trial) if e.parent_trial is not None else None)
                 for e in ordered}
    children_of: dict[int | None, list[int]] = {}
    for e in ordered:
        children_of.setdefault(parent_of[int(e.trial_id)], []).append(int(e.trial_id))
    operator_of = {int(e.trial_id): operator_label(e) for e in ordered}
    rows = []
    for idx, entry in enumerate(ordered):
        stm = render_generated_stm(list(ordered[:idx]), last_n=last_n, budget_tokens=budget_tokens)
        rows.append(measure_decision(entry, stm, parent_of=parent_of,
                                     children_of=children_of, operator_of=operator_of))
    return rows


def stm_snapshot_sizes(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Sizes of persisted STM files (live view and checkpoint copies), read only."""
    out = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        out.append({"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "chars": len(text), "approx_tokens": len(text) / chars_for_tokens(1),
                    "lines": len(text.splitlines()), "cards": len(set(TAG_RE.findall(text)))})
    return out


def checkpoint_self_check(entries: Sequence[Any], checkpoint_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    """Replay check against STM files persisted by the live loop at checkpoints.

    Each checkpoint's ``checkpoint_meta.json`` names its trial; the current renderer
    over the journal prefix through that trial is compared with the persisted file
    section by section (byte identity of the section text before Failure Patterns,
    and identity of the cited-trial sets per section). Failure Patterns may differ
    where the renderer changed after the checkpoint was written (AP-ME-6 typed
    negative-evidence lines); that is the CURRENT-code baseline this census wants.
    """
    ordered = sorted(entries, key=lambda e: int(e.trial_id))
    out = []
    for directory in checkpoint_dirs:
        meta_path, stm_path = directory / "checkpoint_meta.json", directory / "autopilot_short_term_memory.md"
        if not (meta_path.is_file() and stm_path.is_file()):
            out.append({"checkpoint": str(directory), "status": "missing"})
            continue
        trial = int(json.loads(meta_path.read_text())["trial_id"])
        persisted = stm_path.read_text(errors="replace")
        rendered = render_generated_stm([e for e in ordered if int(e.trial_id) <= trial])
        head = lambda text: text.split("## Failure Patterns")[0]  # noqa: E731
        tp, tr = section_tags(persisted), section_tags(rendered)
        out.append({
            "checkpoint": str(directory), "trial_id": trial,
            "persisted_sha256": hashlib.sha256(persisted.encode()).hexdigest(),
            "persisted": {"chars": len(persisted), "lines": len(persisted.splitlines()),
                          "cards": len(set().union(*tp.values()))},
            "rendered": {"chars": len(rendered), "lines": len(rendered.splitlines()),
                         "cards": len(set().union(*tr.values()))},
            "pre_failure_sections_byte_identical": head(persisted) == head(rendered),
            "section_tag_sets_identical": {k: tp[k] == tr[k] for k in SECTIONS},
            "byte_identical": persisted == rendered,
        })
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(journal_dir: Path, stm_paths: Sequence[Path],
        checkpoint_dirs: Sequence[Path] = ()) -> dict[str, Any]:
    shards = journal_shards(journal_dir)
    journal = ExperimentJournal(journal_dir=journal_dir)
    entries = journal.entries_with_supersessions()
    rows = census(entries)
    capture = journal_dir / "mutation_author_context.v1.jsonl"
    return {
        "schema": SCHEMA,
        "inputs": {"journal_dir": str(journal_dir),
                   "shards": [{"name": p.name, "sha256": _sha256(p)} for p in shards],
                   "trial_entries": len(entries),
                   "supersession_events": len(journal.supersession_events())},
        "renderer": {"function": "stm_generated_view.render_generated_stm",
                     "last_n": DEFAULT_LAST_N, "budget_tokens": DEFAULT_BUDGET_TOKENS,
                     "token_estimator": "chars / 4 (context_budget.chars_for_tokens)",
                     "fold": "supersession events as of the retained ledger"},
        "current_bound": {"kind": "single shared bound for every operator",
                          "last_n_entries": DEFAULT_LAST_N, "budget_tokens": DEFAULT_BUDGET_TOKENS},
        "openmle_seeds": OPENMLE_SEEDS,
        "decisions": len(rows),
        "all_decisions": summarize([{**r, "_all": "all"} for r in rows], "_all")["all"],
        "by_operator": summarize(rows, "operator"),
        "by_openmle_class": summarize(rows, "openmle_class"),
        "author_context_capture_records": (sum(1 for _ in capture.open()) if capture.is_file() else 0),
        "persisted_stm_snapshots": stm_snapshot_sizes(stm_paths),
        "checkpoint_self_check": checkpoint_self_check(entries, checkpoint_dirs),
        "per_decision": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--journal-dir", type=Path, required=True,
                        help="directory holding every autopilot_journal*.jsonl shard (read only)")
    parser.add_argument("--stm-snapshot", type=Path, action="append", default=[],
                        help="persisted autopilot_short_term_memory.md to size (repeatable)")
    parser.add_argument("--checkpoint", type=Path, action="append", default=[],
                        help="autopilot checkpoint dir for the replay self-check (repeatable)")
    parser.add_argument("--out", type=Path, help="write the JSON record here")
    args = parser.parse_args(argv)
    if not args.journal_dir.is_dir():
        print(f"journal directory does not exist: {args.journal_dir}", file=sys.stderr)
        return 2
    record = run(args.journal_dir.resolve(), args.stm_snapshot, args.checkpoint)
    if args.out:
        args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    brief = {name: {"decisions": v["decisions"],
                    "tokens_median": (v["approx_tokens"] or {}).get("median"),
                    "cards_median": (v["cards"] or {}).get("median"),
                    "ancestor_cards_median": (v["ancestor_cards"] or {}).get("median"),
                    "sibling_cards_median": (v["sibling_cards"] or {}).get("median"),
                    "truncated_rate": round(v["truncated_rate"], 3)}
             for name, v in record["by_operator"].items()}
    checks = [{k: c.get(k) for k in ("trial_id", "pre_failure_sections_byte_identical",
                                      "section_tag_sets_identical")}
              for c in record["checkpoint_self_check"]]
    print(json.dumps({"decisions": record["decisions"], "by_operator": brief,
                      "checkpoint_self_check": checks}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
