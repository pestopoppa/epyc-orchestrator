# Sequential rate era fence — validation

Implementation: `b336f50b7efb77644a492c707d4ffc503d69d744` (parent `58598acd1031a500c41eef9034d26c165eab0a4f`). The implementation uses the existing speed-era cutoff and captures both speed and quality cutoffs on newly computed rate evidence. It does not ratify a new era or establish whether the existing era covers any later host-speed change. Quality observation provenance beyond the existing timestamp fence remains outside this change.

## Verification

On 2026-11-01, the focused new rate-era tests, existing eval-quality-era tests, paired-rate tests, and two action-gate wiring controls passed: **48 passed**, one third-party deprecation warning. They ran with the existing R6 admission/supervisor on CPUs 122–123, two threads, no model or inference. The supervisor returned child status 0, `children_drained: true`, 72 samples, and sampled tree RSS peak 186,101,760 bytes; sampled RSS is not a kernel-enforced peak.

An earlier broader selection returned 68 passed and one failure in `test_numeric_trial_unmeasured_rate_fails_the_trial_instead_of_killing_it`. The same single test failed with the same assertion against exact parent HEAD versions of both changed source files, under the same guard; the draft files were then restored with their SHA-256 values matching their backups. The test expects completion inside `_action_numeric_trial`; the current implementation defers completion to `finalize_numeric_trial` after the main gate.

The protected DAR script and test remained unstaged and their SHA-256 values remained respectively `40385971b4abb302ada4fb9c6e1610e723882ab8321c680ec020cfd13659d129` and `1bb09bf74cfc7870801b1c2f8e1b1e9657203eed71e3acd12b0c8cd5b5be047d`.
