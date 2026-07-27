# E8 Quality Baseline v5 Operator Candidates

These files are proposed human-owned boundary amendments. They are not invoked
by the evidence runner and do not alter the existing v4 operator artifacts.

- `collect_e8_quality_baseline_v5.sh` is the tokenless, evidence-only collection
  entry point. It requires external source, runner, and base-runner pins.
- `prepare_e8_quality_baseline_v5_candidate.sh` is the read-only sealed-bundle
  validator entry point. It requires external runner, base-runner, and Python
  validator pins.
- `apply_e8_quality_baseline_state_v5_candidate.py` is the minimal applier
  amendment: it imports the existing human-owned transaction implementation and
  changes only the accepted protocol identifier from v4 to v5.
- `ratify_and_apply_e8_quality_baseline_v5.sh` is the single final operator
  action. It validates the sealed evidence, checks reviewed state hashes, writes
  a no-replace protocol-ratification receipt outside the sealed bundle, and
  invokes the existing CAS state transaction through the v5 adapter.

All apply-time receipts, journals, backups, and attestations are written under
`/mnt/raid0/llm/epyc-root/artifacts/operator/`, never into the sealed evidence
directory.
