"""Cross-role scheduling primitives (contention matrix + admission gate).

See `handoffs/active/cross-role-bw-aware-routing.md` for the design rationale.

Public surface:
- contention.ContentionMatrix, PairDecision, MatrixStatus
- contention.load_contention_matrix
- contention.contention_ratio
- contention.pair_policy
- contention.topology_fingerprint
- contention.matrix_status
- contention_gate.ContentionGate (Phase B)
- contention_gate.gate_decode (Phase B context manager)
"""

from __future__ import annotations
