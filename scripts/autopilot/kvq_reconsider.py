#!/usr/bin/env python3
"""Review archived KVQ reliance against a verified, explicitly supplied ledger snapshot.

Run on copies, after a synthetic-correction test, for example:
  python scripts/autopilot/kvq_reconsider.py --vidya-dir /workspace/scripts/vidya \
      --ledger /path/to/ledger-copy.jsonl --archive /path/to/archive-copy.jsonl \
      --as-of 2026-11-01T00:00:00Z

This prints a review report only. It does not change the archive, the ledger,
AutoPilot state, or promotion decisions. Unknown means the receipt could not
be replayed and must not be interpreted as an all-clear.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import sys
from pathlib import Path


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("missing timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone aware")
    return parsed.astimezone(timezone.utc)


def support_paths(belief):
    # Keep duplicate labels and the native grade on each path. No grading occurs here.
    return Counter((label, json.dumps(grade.as_dict(), sort_keys=True))
                   for label, grade in belief.pro_paths)


def source_changes(claim_id, frames, before, after):
    """Follow recorded dependency edges to changed native source beliefs."""
    # The fold exposes its authorized retracted-frame set on beliefs. Empty folds
    # provide no basis for traversing dependencies.
    if not after.beliefs:
        return []
    retracted = set(next(iter(after.beliefs.values())).retracted_support)
    formerly_retracted = set(next(iter(before.beliefs.values())).retracted_support) if before.beliefs else set()
    claim_source, dependencies, withdrawn_claims = {}, {}, {}
    def canonical(cid):
        seen = set()
        while cid in after.alias_map and cid not in seen:
            seen.add(cid)
            cid = after.alias_map[cid]
        return cid
    for frame in frames:
        fid = frame.get("frame_id")
        assertion = frame.get("assertion") or {}
        typ = frame.get("frame_type", "")
        if fid in retracted and fid not in formerly_retracted and typ.endswith("/claim_proposed/v1"):
            cid = assertion.get("claim_id")
            if isinstance(cid, str):
                withdrawn_claims[canonical(cid)] = fid
        # Include newly withdrawn links for this event, to retain its former source identity.
        if fid in formerly_retracted:
            continue
        if typ.endswith("/claim_proposed/v1"):
            cid, source = assertion.get("claim_id"), assertion.get("source_id")
            if isinstance(cid, str) and isinstance(source, str):
                claim_source[canonical(cid)] = source
        elif typ.endswith("/claim_depends_on/v1"):
            cid, source = assertion.get("claim_id"), assertion.get("depends_on_source")
            if isinstance(cid, str) and isinstance(source, str):
                dependencies.setdefault(canonical(cid), set()).add(source)
    reached, pending = set(), [canonical(claim_id)]
    while pending:
        cid = pending.pop()
        if cid in reached:
            continue
        reached.add(cid)
        sources = dependencies.get(cid, set())
        pending.extend(other for other, source in claim_source.items() if source in sources)
    changes = []
    for cid in sorted(reached - {canonical(claim_id)}):
        if cid in withdrawn_claims:
            changes.append({"source_claim_id": cid, "withdrawn_claim_frame_id": withdrawn_claims[cid]})
        old, new = before.beliefs.get(cid), after.beliefs.get(cid)
        if old is None or new is None:
            raise ValueError("dependency claim is not present in both folds")
        for correction in sorted(set(new.corrections) - set(old.corrections)):
            changes.append({"source_claim_id": cid, "correction_id": correction})
        for correction in sorted(set(old.corrections) - set(new.corrections)):
            changes.append({"source_claim_id": cid, "removed_correction_id": correction})
        removed = support_paths(old) - support_paths(new)
        for label, grade in sorted(removed.elements()):
            changes.append({"source_claim_id": cid, "removed_evidence_id": label,
                            "grade": json.loads(grade)})
        for alert in sorted(set(new.dependency_alerts) - set(old.dependency_alerts)):
            changes.append({"source_claim_id": cid, "dependency_alert": alert})
    return changes


def review(rows, frames, *, as_of, fold):
    """Replay each receipt at its actual frontier and detect later fold changes."""
    checked_at = timestamp(as_of)
    @lru_cache(maxsize=8)
    def snapshot(visible_indices, at):
        return fold([frames[i] for i in visible_indices], as_of=at)
    results = []
    for line, row in enumerate(rows, 1):
        if not isinstance(row, dict) or row.get("type") != "planner_coordinator":
            results.append({"archive_line": line, "status": "unknown",
                            "reason": "row outside planner_coordinator receipt coverage"})
            continue
        receipt = row.get("vidya_kvq")
        item = {"archive_line": line, "trial_id": row.get("trial_id")}
        if not isinstance(receipt, dict) or receipt.get("schema") != "epyc.vidya.kvq_planner_receipt.v1":
            results.append({**item, "status": "unknown", "reason": "missing receipt"})
            continue
        declared, exposed = receipt.get("declared_reliance_claim_ids"), receipt.get("exposed_claim_ids")
        if not all(isinstance(x, list) and all(isinstance(c, str) and c for c in x)
                   for x in (declared, exposed)):
            results.append({**item, "status": "unknown", "reason": "invalid claim lists"})
            continue
        relied = sorted(set(declared) & set(exposed))
        frontier, at, expected_hash = receipt.get("frontier"), receipt.get("as_of"), receipt.get("state_hash")
        if (type(frontier) is not int or not 0 <= frontier <= len(frames)
                or not isinstance(at, str) or not at or not isinstance(expected_hash, str) or not expected_hash):
            results.append({**item, "status": "unknown", "reason": "invalid or unavailable receipt frontier"})
            continue
        try:
            if timestamp(at) > checked_at:
                raise ValueError("checked time precedes receipt")
            original = snapshot(tuple(range(frontier)), at)
            if original.frontier != frontier or original.state_hash() != expected_hash:
                results.append({**item, "status": "unknown", "reason": "receipt fold does not replay"})
                continue
            if not relied:
                results.append({**item, "status": "no_declared_reliance",
                                "receipt_frontier": frontier, "receipt_as_of": at,
                                "receipt_state_hash": expected_hash})
                continue
            if any(cid not in original.beliefs for cid in relied):
                results.append({**item, "status": "unknown", "reason": "declared claim missing from receipt fold"})
                continue
            # KVQ receipt producer folds the complete recorded prefix, regardless
            # of publication timestamps; replay that exactly. Later frames require
            # explicit visibility at the requested review time.
            visible = list(range(frontier))
            for i in range(frontier, len(frames)):
                created = (frames[i].get("pubinfo") or {}).get("created_at")
                if timestamp(created) <= checked_at:
                    visible.append(i)
            event_positions = [i for i in visible if i >= frontier and
                               frames[i].get("frame_type", "").endswith(
                                   ("/correction_recorded/v1", "/retraction/v1"))]
            events = []
            unknown = False
            # Fold at every potential event, including corrections later reviewed, so
            # a cleared belief flag does not erase the decision's need for review.
            for i in event_positions:
                prior_indices = tuple(j for j in visible if j < i)
                current_indices = prior_indices + (i,)
                previous = snapshot(prior_indices, as_of)
                current = snapshot(current_indices, as_of)
                target = (frames[i].get("assertion") or {}).get("retracts")
                target_frame = next((f for f in frames[:i] if f.get("frame_id") == target), None) if target else None
                for cid in relied:
                    before, after = previous.beliefs.get(cid), current.beliefs.get(cid)
                    if before is None:
                        unknown = True
                        continue
                    if after is None:
                        unknown = True
                        continue
                    corrections = sorted(set(after.corrections) - set(before.corrections))
                    removed_corrections = sorted(set(before.corrections) - set(after.corrections))
                    dependency_alerts = sorted(set(after.dependency_alerts) - set(before.dependency_alerts))
                    dependency_changes = source_changes(cid, [frames[j] for j in current_indices], previous, current)
                    removed = list((support_paths(before) - support_paths(after)).elements())
                    claim_retracted = bool(target_frame and target in after.retracted_support
                        and target not in before.retracted_support
                        and target_frame.get("frame_type", "").endswith("/claim_proposed/v1")
                        and (target_frame.get("assertion") or {}).get("claim_id") == cid)
                    if corrections or removed or removed_corrections or claim_retracted or dependency_alerts or dependency_changes:
                        events.append({"claim_id": cid, "frontier": i + 1,
                                       "event_frame_id": frames[i].get("frame_id"),
                                       "correction_ids": corrections, "removed_correction_ids": removed_corrections,
                                       "dependency_alerts": dependency_alerts,
                                       "dependency_changes": dependency_changes,
                                       "removed_support": [{"evidence_id": label, "grade": json.loads(grade)}
                                                           for label, grade in sorted(removed)],
                                       "claim_retracted": claim_retracted})
            results.append({**item, "status": "review_required" if events else ("unknown" if unknown else "no_change_detected"),
                            "receipt_frontier": frontier, "receipt_as_of": at, "receipt_state_hash": expected_hash,
                            "checked_frontier": len(frames),
                            "claim_ids": relied, "events": events,
                            "coverage": "partial" if unknown else "complete",
                            "visible_frame_count": len(visible)})
        except Exception as exc:
            results.append({**item, "status": "unknown", "reason": f"fold failed: {type(exc).__name__}"})
    return {"schema": "epyc.autopilot.kvq_reconsideration.v1", "as_of": as_of,
            "frontier": len(frames), "authority": "human review only; no automatic reversal",
            "decisions": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vidya-dir", required=True, type=Path)
    p.add_argument("--ledger", required=True, type=Path)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--as-of", required=True)
    args = p.parse_args()
    if not args.ledger.is_file() or not args.archive.is_file():
        p.error("ledger and archive snapshots must exist")
    sys.path.insert(0, str(args.vidya_dir.resolve()))
    import ledger as ledger_module
    import fold as fold_module
    Ledger, fold = ledger_module.Ledger, fold_module.fold
    try:
        timestamp(args.as_of)
    except ValueError as exc:
        p.error(str(exc))
    ledger_bytes = args.ledger.read_bytes()
    ledger = Ledger(args.ledger)
    errors = ledger.verify()
    repair = []
    records = ledger.read_all(repair_report=repair)
    if errors or repair or args.ledger.read_bytes() != ledger_bytes:
        p.error("ledger integrity failed or snapshot changed during read")
    archive_bytes = args.archive.read_bytes()
    if archive_bytes and not archive_bytes.endswith(b"\n"):
        p.error("archive snapshot has an incomplete final line")
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj
    def reject_constant(value):
        raise ValueError("non-finite JSON value")
    try:
        rows = [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
                for line in archive_bytes.splitlines() if line.strip()]
        report = review(rows, [r.frame for r in records], as_of=args.as_of, fold=fold)
    except (ValueError, UnicodeError) as exc:
        p.error(f"invalid archive: {exc}")
    def source_pin(path):
        source = Path(path).resolve()
        return {"path": str(source), "on_disk_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    report["provenance"] = {
        "ledger_snapshot_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "archive_snapshot_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "source_hash_kind": "on-disk bytes; not loaded-byte attestation",
        "fold": source_pin(fold_module.__file__), "ledger": source_pin(ledger_module.__file__),
        "report_script": source_pin(__file__),
    }
    if args.archive.read_bytes() != archive_bytes or args.ledger.read_bytes() != ledger_bytes:
        p.error("snapshot changed during review")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
