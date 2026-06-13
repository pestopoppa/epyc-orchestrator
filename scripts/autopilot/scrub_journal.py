#!/usr/bin/env python3
"""One-shot tagger for bug-corrupted journal entries.

Use case: after fixing a bug that caused recent trials to fail (or to
produce misleading outcomes), tag those trials as `bug_corrupted_by=<sha>`
so the planner's hypothesis-chain reasoning doesn't learn wrong lessons
from them. Trustworthiness gate then sees the real signal count.

Filter semantics (all bounds combine with AND; absent bounds match
anything):
    --trial-id-min / --trial-id-max
    --since / --until   (ISO timestamps, e.g. 2026-05-20T00:00 or 2026-05-20)

Examples:

    # Tag everything from May 20-22 as corrupted by the chat-template fix
    python3 scripts/autopilot/scrub_journal.py \
        --commit-sha de34dd4 \
        --reason "chat-template-per-role bug + NLL-8B routing artifact" \
        --since 2026-05-20 --until 2026-05-22T23:59:59

    # Tag a single contiguous trial range
    python3 scripts/autopilot/scrub_journal.py \
        --commit-sha b3895aa \
        --reason "highlight.js CDN URL bug (no eval impact, scrub for tidiness)" \
        --trial-id-min 290 --trial-id-max 295

    # Preview without writing
    python3 scripts/autopilot/scrub_journal.py --dry-run --since 2026-05-20 \
        --commit-sha de34dd4 --reason "preview"

Notes:
    - The journal is rewritten in place — the JSONL files are completely
      regenerated from the in-memory entries (which now include the
      bug_corrupted_by tag). The TSV is also rewritten so quick-look
      consumers see the change.
    - A .bak-<timestamp> backup of every overwritten file is created
      alongside the original, so undoing the scrub is just a `mv` away.
    - The Pareto archive in autopilot_state.json is NOT touched. Tagging
      a journal entry as corrupted does NOT remove the corresponding
      Pareto point — the operator can do that separately if needed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Allow running as a script: add the autopilot dir to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import csv
import json

from experiment_journal import (  # noqa: E402
    ExperimentJournal,
    JournalEntry,
    TSV_COLUMNS,
)


def _autopilot_running_pids() -> list[int]:
    """Return PIDs of any live `autopilot.py start` processes (excludes this one).

    Used to guard against scrubbing while autopilot is mid-trial — writing
    new journal entries against a journal we're rewriting underneath it
    would either corrupt the file or silently lose the entries.
    """
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "autopilot.py start"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []  # no matches
    except FileNotFoundError:
        return []  # pgrep not available — skip the guard
    pids: list[int] = []
    me = os.getpid()
    for line in out.strip().splitlines():
        parts = line.split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == me:
            continue
        # Filter out our own scrub invocation showing up as a match (the
        # subprocess pgrep sees our argv if we're a child of bash with
        # the script name in there)
        if "scrub_journal.py" in line:
            continue
        pids.append(pid)
    return pids


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + f".bak-scrub-{int(time.time())}")
    shutil.copy2(path, bak)
    return bak


def _rewrite_all(journal: ExperimentJournal) -> list[tuple[Path, Path | None]]:
    """Rewrite all JSONL + TSV files from the journal's in-memory entries.

    Splits entries into per-batch files (matching the existing
    rotation semantics: MAX_TRIALS_PER_FILE per file). Returns a list of
    (rewritten_path, backup_path or None) tuples for reporting.
    """
    from experiment_journal import MAX_TRIALS_PER_FILE
    out: list[tuple[Path, Path | None]] = []

    # Group by batch (trial_id // MAX_TRIALS_PER_FILE)
    by_batch: dict[int, list[JournalEntry]] = {}
    for e in journal._entries:
        by_batch.setdefault(e.trial_id // MAX_TRIALS_PER_FILE, []).append(e)
    events_by_batch = getattr(journal, "_ledger_events_by_batch", {})

    for batch in sorted(set(by_batch) | set(events_by_batch)):
        entries = by_batch.get(batch, [])
        jsonl = journal._jsonl_path(batch)
        tsv = journal._tsv_path(batch)
        # Backups
        jsonl_bak = _backup(jsonl)
        tsv_bak = _backup(tsv)
        # Rewrite JSONL
        with open(jsonl, "w") as f:
            for e in entries:
                f.write(json.dumps(asdict(e), default=str) + "\n")
            for event in events_by_batch.get(batch, []):
                f.write(json.dumps(event, default=str) + "\n")
        # Rewrite TSV (header + rows)
        with open(tsv, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(TSV_COLUMNS)
            for e in entries:
                w.writerow([
                    e.trial_id, e.timestamp, e.species, e.action_type, e.tier,
                    e.quality, e.speed, e.cost, e.reliability, e.pareto_status,
                    e.git_tag, e.reasoning_hash,
                ])
        out.append((jsonl, jsonl_bak))
        out.append((tsv, tsv_bak))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--commit-sha", required=True,
                    help="short SHA of the bug-fix commit to tag entries with")
    ap.add_argument("--reason", required=True,
                    help="free-text operator note (truncated to 200c)")
    ap.add_argument("--trial-id-min", type=int, default=None,
                    help="minimum trial_id to tag (inclusive)")
    ap.add_argument("--trial-id-max", type=int, default=None,
                    help="maximum trial_id to tag (inclusive)")
    ap.add_argument("--since", default=None,
                    help="minimum timestamp (ISO 8601; entries lexicographically >= this)")
    ap.add_argument("--until", default=None,
                    help="maximum timestamp (ISO 8601; entries lexicographically <= this)")
    ap.add_argument("--journal-dir", type=Path, default=None,
                    help="override journal directory (default: orchestration/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be tagged without writing")
    ap.add_argument("--append-event", action="store_true",
                    help="append a supersession event instead of rewriting trial rows")
    ap.add_argument("--force-while-autopilot-alive", action="store_true",
                    help="skip the autopilot-running safety guard (rarely needed)")
    args = ap.parse_args()

    if not any([args.trial_id_min, args.trial_id_max, args.since, args.until]):
        print(
            "ERROR: at least one filter must be specified — refusing to tag every entry",
            file=sys.stderr,
        )
        return 2

    # Safety: scrubbing while autopilot is mid-trial = lost / corrupted
    # entries (autopilot would append to the file we just rewrote). Refuse
    # unless explicitly overridden. Dry-run is allowed regardless since it
    # doesn't write.
    if not args.dry_run and not args.force_while_autopilot_alive:
        live = _autopilot_running_pids()
        if live:
            print(
                "ERROR: autopilot is running (pids: "
                + ", ".join(str(p) for p in live)
                + "). Stop it before scrubbing so we don't race against new\n"
                "journal writes. Either kill the autopilot process or pass\n"
                "--force-while-autopilot-alive if you accept the risk of\n"
                "losing whatever trial is currently mid-evaluation.",
                file=sys.stderr,
            )
            return 3

    journal = ExperimentJournal(journal_dir=args.journal_dir)
    n_before_trustworthy = journal.trustworthiness_score()["trustworthy"]

    fields = {
        "bug_corrupted_by": args.commit_sha,
        "bug_corrupted_reason": (args.reason or "")[:200],
    }
    if args.append_event:
        tagged_ids = journal.matching_trial_ids(
            trial_id_min=args.trial_id_min,
            trial_id_max=args.trial_id_max,
            timestamp_min=args.since,
            timestamp_max=args.until,
        )
        n_tagged = len(tagged_ids)
    else:
        n_tagged, tagged_ids = journal.apply_scrub(
            commit_sha=args.commit_sha,
            reason=args.reason,
            trial_id_min=args.trial_id_min,
            trial_id_max=args.trial_id_max,
            timestamp_min=args.since,
            timestamp_max=args.until,
        )

    score = journal.trustworthiness_score()
    print(f"Filter:                  commit_sha={args.commit_sha!r} reason={args.reason!r}")
    print(f"  trial_id_min: {args.trial_id_min}  trial_id_max: {args.trial_id_max}")
    print(f"  since:        {args.since}  until: {args.until}")
    print()
    print(f"Entries that would be tagged: {n_tagged}")
    if n_tagged > 0:
        ids_preview = ", ".join(str(i) for i in tagged_ids[:30])
        if len(tagged_ids) > 30:
            ids_preview += f", ... ({len(tagged_ids) - 30} more)"
        print(f"  trial_ids: {ids_preview}")
    print()
    print(f"Before scrub: trustworthy={n_before_trustworthy}")
    if args.append_event:
        print(
            "After  scrub: unchanged (append-event mode does not rewrite trial rows)"
        )
    else:
        print(f"After  scrub: trustworthy={score['trustworthy']}  corrupted={score['corrupted']}  ratio={score['ratio']:.2%}")
    if score["corrupted_by"]:
        print(f"Corrupted-by breakdown: {score['corrupted_by']}")

    if args.dry_run:
        print()
        if args.append_event:
            print("DRY-RUN — no files written. Re-run without --dry-run to append event.")
        else:
            print("DRY-RUN — no files written. Re-run without --dry-run to apply.")
        return 0

    if n_tagged == 0:
        print("Nothing to tag — exiting without rewriting files.")
        return 0

    if args.append_event:
        event = journal.append_supersession_event(
            target_trial_ids=tagged_ids,
            fields=fields,
            reason=args.reason,
            policy_version="supersession-v1",
            actor="scrub_journal.py",
        )
        print()
        print("Appended supersession event:")
        print(f"  type: {event['type']}")
        print(f"  target_trial_ids: {len(tagged_ids)}")
        print(f"  fields: {fields}")
        print()
        print("Legacy trial rows were not rewritten.")
        return 0

    written = _rewrite_all(journal)
    print()
    print("Rewrote:")
    for path, bak in written:
        if bak:
            print(f"  {path}  (backed up to {bak.name})")
        else:
            print(f"  {path}")
    print()
    print(f"Done. {n_tagged} entries tagged as bug_corrupted_by={args.commit_sha}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
