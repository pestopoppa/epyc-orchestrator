# AP-29 episodic control fixture

Base APP commit: `6fffd846369a4cfd7a1cff4a496a2b1816f88da1`.

`run_paired_arms` creates two isolated temporary StrategyStores, writes the same original raw episodes after explicit deletions to both, and runs identical native retrieval settings. Only the candidate store reaches the abstraction callback. The fixture runs the actual `KnowledgeDistiller` on the candidate, producing a pattern while the control retrieves original raw episodes; a no-op candidate verifies matched native retrieval. Invalid deletion IDs, duplicate IDs and synthetic entries fail before the candidate runs. The fixture uses a deterministic embedder; it is not a quality comparison. Adoption remains pending an authorized comparison rule, and this primitive does not wire or authorize a production caller. The existing tripwire remains in place.

`tests.log` captures targeted native tests and lint under the R4 supervisor with CPUs 28–29, one-thread numerical libraries and no model call. The recorded RSS is sampled. `readset.sha256` pins the principal source and test inputs. `caller-audit.log` is a read-only `git grep` over the base commit, rather than the incomplete projection's working tree; the only non-test mentions are the class definition and a context-budget docstring. GitNexus impact was LOW with zero upstream, but its index was stale and not used as caller evidence. Full-stack `make gates` was not run in this bounded projection.

The source motivating AP-29 reports a 19-problem ARC-AGI slice falling from 100% to 52.6% at R10, with two repeats per question and point estimates rather than formal error bars. This fixture does not reproduce or extend that result.
