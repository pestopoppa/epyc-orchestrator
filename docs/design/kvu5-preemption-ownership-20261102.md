# KVU-5: interfaces needed for safe shared-KV preemption

Source audit against orchestrator `f678231cb31c2898a1ff7ab8cb9fd183e50e7604`.
This is an implementation design, not an enabled feature or evidence that a
request was preempted. KVU-5 remains open.

## Existing boundary

`SharedKVPoolAdmission.acquire()` receives a URL, requested token reservation,
deadline and a **queued-request** `cancel_check`; it returns an integer ticket.
`_inflight[url][ticket]` contains only the reservation, not a logical request ID,
priority, output-commit state, cancellation capability or server task identity.
The condition, ticket counter and reservations are process-local. Its host-wide
flock controls long prefill, not ownership of every in-flight request across API
workers. `release()` removes the ticket and adjusts fork-dependent accounting.

The passthrough `_Call` owns an HTTP response and `threading.Event`, but the
event currently means **client disconnected**. `abort_upstream()` is a
best-effort socket shutdown/close; it does not produce a server-task terminal
receipt. `run_upstream()` emits response headers immediately and forwards body
chunks as received. Once the route has handed headers or data to the client, a
replacement generation cannot transparently overwrite that response.

In primitives, `inference_request_id` and `request_priority` exist outside the
KV admission API. Streaming cancellation is polled at chunk callbacks; batch
inference has no equivalent registered live cancellation handle. The
`_real_call_single` cleanup releases its ticket in a `finally`. The higher
`_real_call` already refuses overflow recovery or model fallback after a visible
stream chunk to avoid splicing two outputs. A KVU-5 path must preserve that rule.

## Implementation slices

1. **Attempt ownership, opt-in and visibility.** Extend the admission-to-caller
   contract with a unique logical request ID, unique attempt ID, priority,
   immutable original queue sequence, deadline and explicit retry eligibility.
   Each admitted attempt registers an owner-bound cancellation interface and
   monotonic output-commit state. Default is non-preemptible; no caller becomes
   eligible merely because it has a ticket. A missing identity, unknown output
   state, unsupported transport or callback owner must refuse preemption.
   The passthrough and primitives adapters must supply these facts themselves;
   do not infer them from a serving log or from `request_priority` alone.
2. **One-shot selection and cancellation.** Under the same admission mutex,
   atomically select and mark one eligible victim for a capacity-blocked waiter;
   execute owner cancellation outside the mutex. Use a documented deterministic
   priority/age ordering, do not preempt a request ahead of the waiter in its
   FCFS age class, and cap repeat preemptions per logical request. A victim that
   has crossed the output-commit boundary is ineligible. Cancellation has a
   distinct typed reason, separate from client disconnect, timeout and normal
   completion. Concurrent selection, normal completion and user disconnect must
   produce at most one terminal disposition.
3. **Prove drain before capacity credit.** The victim's owner closes its own
   upstream and returns a terminal acknowledgement tied to the attempt and
   server task/slot identity. A request to cancel or a closed local socket is not
   proof the server has released its KV. Keep its reservation until the owner
   terminal path and an appropriate server-side terminal/absence witness agree;
   missing, stale or contradictory identity means UNKNOWN and no preemption
   credit. Preserve fork-source credit reattribution and the long-prefill lease
   when the ticket really releases. A failed cancellation must not expose freed
   capacity or enqueue another copy.
4. **Idempotent retry and admission fairness.** Only a not-yet-committed attempt
   with a confirmed drain may re-enter admission using the same logical request
   ID and original age, with a new attempt ID, remaining deadline and retry
   budget. Enqueue exactly once despite duplicate acknowledgements; preserve the
   original request bytes and sampling inputs. Existing content, usage and
   serving records must distinguish attempts and charge real work without
   presenting a second answer. Do not replay already emitted headers/chunks or a
   client-cancelled request. Retain one terminal event per attempt.
5. **Cross-worker custody and deployment.** Define a single cross-worker owner
   and request routing/receipt protocol before enabling this on a multi-worker
   server; the current in-memory dictionaries and prefill flock do not supply
   one. A crashed owner or lost acknowledgement must fail closed and be
   recoverable without duplicate replay. Coordinate admission, fork index and
   RTG-58 owner boundaries; preserve default-off behavior and existing source
   and launch gates until deterministic controls and an authorized live test
   cover the complete path.

## Required no-inference controls

Use fake transports and a deterministic clock to exercise: priority/age ties;
blocked capacity; no eligible victim; missing identity; two concurrent selectors;
completion racing cancel; repeated cancel/ack; cancellation failure and delayed
server drain; client disconnect; headers or chunks already emitted; deadline
expiry during drain; retry budget exhaustion; duplicate retries; cross-worker
owner loss; and fork-source/lease release. Assert exact queue order, one
logical response and one terminal record, no early reservation credit, no
starvation under the declared bounded policy, and identical flag-off behavior.
Only a separate authorized integration run can establish actual server drain.

## Concrete owner boundary and native obstacle (2026-11-02)

The passthrough owner has an optional, default-off `OutputCommitObservation`.
It registers an attempt with no native task identity, cancellation or replay
capability, so selection always refuses it. If attached, `run_upstream` marks
that attempt committed before it queues the response headers. Its `finish`
marks only the local owner terminal; it does not certify native drain or release
capacity. Normal calls do not construct or attach this observation. This is an
integration seam, not an enabled preemption path.

The frozen v10 server's `SERVER_TASK_TYPE_SLOT_ERASE` implementation in
`tools/server/server-context.cpp` looks up the slot **when the erase task is
processed**, force-releases whichever task is then processing, and clears the
slot. The route `handle_slots_erase` accepts a slot ID without an expected
`id_task` condition. A `/slots` observation followed by erase therefore has a
slot-reuse race: it can cancel a successor. The OpenAI streaming response's
`id` is `oaicompat_cmpl_id`; it does not expose the native slot `id_task` for
an owner-bound compare. Disconnecting the upstream socket also supplies no
native task-bound terminal/drain receipt. The owner cannot safely select a
victim using these endpoints. It needs a native operation that atomically
checks server incarnation plus expected task identity, cancels only that task,
and returns or supports a verifiable terminal/drain witness; the admission owner
then needs to retain its reservation until that witness and local termination
agree. The frozen production kernel remains unchanged.
