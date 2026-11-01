# Paired edit instrument

This opt-in instrument presents the same initial Python file with either a fully specified or minimal instruction. Each fixture owns one shared executable oracle and includes first-party gold and negative code. It performs no inference and changes no existing benchmark pool or scoring gate. These fixtures demonstrate the instrument, not model quality or instruction-specificity effects.

Call `prompt(fixture, "descriptive")` or `prompt(fixture, "minimal")`; pass the complete returned code to `evaluate`. The prompt excludes the oracle and reference code. The result is `pass`, `fail`, `timeout`, or `unavailable`; an empty answer and a successful process exit without completion of the tests cannot pass. Execution forces the existing SEC-EXEC containment policy even when its feature flag is off. A host unable to enforce it refuses execution. The existing policy blocks networking and credential access; it is not a general filesystem-write sandbox or an adversarially secret oracle, since the oracle and attempt execute in the same process. Use only in an execution environment appropriate for that boundary.

Validation: native unit tests cover both first-party golds, mutants, original code, input mutation, timeout, missing attempts, early exit, exceptions, pair construction, and containment refusal. No model calls or benchmark measurements were made.
