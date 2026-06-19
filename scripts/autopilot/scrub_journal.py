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
    - Default mode is append-only: the journal gets a supersession event
      describing the override, and existing trial rows are not rewritten.
    - Legacy in-place JSONL/TSV rewriting is retired; use append-only
      supersession events so historical trial rows remain immutable.
    - The Pareto archive in autopilot_state.json is NOT touched. Tagging
      a journal entry as corrupted does NOT remove the corresponding
      Pareto point — the operator can do that separately if needed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Allow running as a script: add the autopilot dir to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_journal import (  # noqa: E402
    ExperimentJournal,
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
                    help="compatibility no-op; append-only supersession events are now the default")
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
    tagged_ids = journal.matching_trial_ids(
        trial_id_min=args.trial_id_min,
        trial_id_max=args.trial_id_max,
        timestamp_min=args.since,
        timestamp_max=args.until,
    )
    n_tagged = len(tagged_ids)

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
    print("After  scrub: unchanged (append-event mode does not rewrite trial rows)")
    if score["corrupted_by"]:
        print(f"Corrupted-by breakdown: {score['corrupted_by']}")

    if args.dry_run:
        print()
        print("DRY-RUN — no files written. Re-run without --dry-run to append event.")
        return 0

    if n_tagged == 0:
        print("Nothing to tag — exiting without writing files.")
        return 0

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


if __name__ == "__main__":
    sys.exit(main())
