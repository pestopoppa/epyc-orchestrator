# Sequential rate era fence: verification

Final implementation commit: `ddb6391714483740ac1c87e6fca7d3b797ef491c`; original implementation `b336f50b7efb77644a492c707d4ffc503d69d744`. SHA-256 of the final three source/test files:

- `scripts/autopilot/autopilot.py`: `689fc3874695b94480a664f62613def06aa6e38692c3c45186a491f781b4939e`
- `scripts/autopilot/actions.py`: `c768e3f6e9464a1f34da71f97a9083795b6c2ece87ffbfd2790e6b0054597685`
- `tests/unit/test_autopilot_seq_rate_era_fence.py`: `9177f59f4e3a2864fd1cca3912ab94b5248a9f4a45239b3ed36e45fd0f4ce4cd`

The final [focused run](focused-49.txt) passed 49 tests. The first implementation's focused run passed 48, including the action-local to main cache control; the recorded supervisor result was child status 0, children drained, 72 samples, sampled peak RSS 186,101,760 bytes. Its tool output was not saved as a standalone log; the result was recorded before the follow-up in `../seq-rate-era-20261101/README.md`. All runs used existing R6 admission and supervisor on CPUs 122–123, two threads, no model or inference; sampled RSS is not a kernel-enforced peak.

The earlier broad selected run was 68 passed and 1 failed. The [final broader selection](broad-selection.txt), with additional tests, was 71 passed and 1 failed. The same unrelated numeric-action test failed on unchanged baseline source in the [baseline run](baseline-failure.txt); both changed source files were restored byte-for-byte afterward. The test expects `_action_numeric_trial` to finish the Optuna trial itself; current code defers that to `finalize_numeric_trial` after the main gate.

This applies an existing ratified speed cutoff and records cutoff provenance for the rate axis. It does not add an era, establish whether an unregistered host-speed event is covered, activate the default-off sequential gate, or complete a quality-history provenance fence beyond the existing timestamp policy. The protected DAR script and test retained SHA-256 `40385971b4abb302ada4fb9c6e1610e723882ab8321c680ec020cfd13659d129` and `1bb09bf74cfc7870801b1c2f8e1b1e9657203eed71e3acd12b0c8cd5b5be047d`.
