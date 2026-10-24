# Pre-P2 journal snapshot rebuild

P2 changes the archive payload used by journal snapshots. The current writers emit
`journal-segment-snapshot-v2` from `ExperimentJournal` and
`journal-archive-snapshot-v2` from `journal_snapshot_create.py`; representative replay
state remains `representative-replay-state-v1`. A pre-P2 snapshot can still have a valid
payload hash, but exact comparison with current prefix replay reports
`archive_prefix_drift` and `prefix_invalidated`. `archive_payload_from_verified_snapshot`
then declines bounded authority and callers use full journal replay. Do not edit or
rehash an old snapshot in place.

The real 1,372-row `gate_frontier_replay.json` fixture reproduces the shape change: after
removing the new `exclusions.unmeasured_objectives` counter, its archive matches the
pinned pre-P2 golden digest. Current replay reports 85 unmeasured rows (maximum trial
1496). Removing that additive counter from the comparison is not part of production
verification; the supported migration is to append a current-format snapshot.

## Owner boundary and command

After this branch is integrated, the session that owns the live autopilot performs this
at its trial boundary, with autopilot stopped so the journal cannot change during the
snapshot. The writer's running-autopilot guard must remain enabled. `--force` is needed
when the existing snapshot already covers the current maximum trial; it permits an
append-only recut and does not bypass the live-process guard.

```bash
cd /workspace/repos/epyc-orchestrator
.venv/bin/python scripts/autopilot/journal_snapshot_create.py \
  --journal-dir /workspace/repos/epyc-orchestrator/orchestration \
  --append --force --json
```

The same owning session verifies the appended snapshot before resuming autopilot:

```bash
cd /workspace/repos/epyc-orchestrator
.venv/bin/python scripts/autopilot/journal_snapshot_replay.py \
  --journal-dir /workspace/repos/epyc-orchestrator/orchestration \
  --strict --allow-tail-fold --json
```

Require `status` to be `archive_prefix_match` and `strict_ready` to be true. If new trials
arrive after the snapshot, the command may report verified tail-fold readiness; otherwise
the snapshot should be `current`. Keep autopilot stopped and investigate if strict
verification refuses. Do not use `--force-while-autopilot-alive`, and do not infer
production readiness from this fixture review.
