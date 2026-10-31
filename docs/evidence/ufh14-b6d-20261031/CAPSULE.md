# UFH14-B6d — parked-role escalation routing

## Implemented source

APP implementation commit: `d0059590fc1eb8e42ef290c4bfc9fb81783dfc91`

Rebased source base: `c2407be32f1b4e958b0c1ff37c8edb0dfd75aefd`

Branch at implementation: `codex/td29-m4-cache-20261030`

The quality escalation helper now tries only distinct roles declared by the existing escalation chain. It skips a rung only when the primitive raises typed `RoleParkedError`; generic failures keep their in-band response. A force-pinned named consultant gets only its requested role, including when that role equals the incumbent. If the eligible chain is exhausted by parked roles, the typed refusal remains visible. The v1 step records the actual served role and skipped roles, and the serving record carries the route trace. Direct named-role and passthrough refusal paths retain their 503 behavior.

`LLMPrimitives.llm_call` adds opt-in `raise_parked_errors`; its default remains unchanged, and the opt-in propagates only `RoleParkedError`. The v1 test fake now implements the current `bind_request_usage` and `get_request_usage` protocol.

## Validation

All commands ran in the APP checkout with the canonical shared Python 3.11 interpreter. Test runs acquired `/mnt/raid0/llm/tmp/compute-flags/prep-locks/32-33.lock`, used CPUs 128–129 only, `OMP_NUM_THREADS=1`, an 8 GiB virtual-memory cap, and a 900 second timeout. No model calls, inference, live API changes, flag changes, reloads, downloads, or kernel operations were performed.

| Command | Result | Durable output |
|---|---:|---|
| `pytest -q tests/unit/test_gpu_window_parked_role.py tests/unit/test_v1_escalation.py` | 68 passed, 1 warning, 29.75s | [parked-and-v1-tests.log.gz](parked-and-v1-tests.log.gz) |
| `pytest -q tests/unit/test_stages.py tests/unit/test_llm_primitives.py` | 65 passed, 1 warning, 8.39s | [stages-and-primitives-tests.log.gz](stages-and-primitives-tests.log.gz) |
| `ruff check` on the five changed source/test files | passed | [ruff.log.gz](ruff.log.gz) |
| `git diff --check` | passed | recorded before source commit |
| `make gates` | passed (schema, shellcheck, shfmt, markdownlint, format/lint targets) | [make-gates.log.gz](make-gates.log.gz) |

Captured command output is gzip-compressed byte-for-byte to preserve logs containing trailing whitespace that `git diff --check` correctly rejects in ordinary text files; verify compressed artifact integrity with [SHA256SUMS](SHA256SUMS), and read with `gzip -dc <file>.gz`.

The final 68-test run followed the clean rebase onto the source base above. Its shell PID was 1703420 (start `2026-10-31 13:33:48 UTC`, initial RSS 3624 KiB); the process was confirmed absent after completion. The 65-test run shell PID was 1674717 (start `2026-10-31 13:32:54 UTC`, initial RSS 3628 KiB); it too was confirmed absent.

## Historical failures, qualified

These earlier runs are retained to make the correction sequence auditable; they are not the final result.

- The initial combined 131-test run had 33 failures and 98 passes because `FakePrimitives` did not implement the current route contract `bind_request_usage(chat_id)`. The captured traceback identifies the fixture `AttributeError`; the fake now implements `bind_request_usage` and `get_request_usage`. See [historical-33-failures.log.gz](historical-33-failures.log.gz).
- A later run had one unrelated failure in the existing `<100 ms` primitive refusal check: preempt notification file I/O made that measured section take about 253 ms. The test now stubs only the preempt notification, retains its temporary window and serving-record assertions, and keeps the `<100 ms` assertion unchanged. See [historical-preempt-timing-failure.log.gz](historical-preempt-timing-failure.log.gz). The final parked-role module passed.
- An intermediate broad error-propagation experiment was removed after review. The final implementation opts into propagation only for typed parked refusals, and the final 68-test run above passes.

## Original handoff draft

The owning ROOT session applies this draft after verifying the APP publication. See [HANDOFF-DRAFT.md](HANDOFF-DRAFT.md). This capsule does not edit ROOT handoffs or checkboxes.

Evidence class: source behavior and controlled tests only; no measurements or verified empirical findings. [no-belief: code-and-test verification only]
