# KPF-25 audit: byte-stable fan-out trunks (and the KPF-21 trunk request)

Date 2026-10-20. Scope: every orchestrator builder that emits N sibling prompts expected to share a
trunk. Method: read each builder, list what is placed before the junction, check it against
RTG-27 (`prompt-construction-determinism`: construction is deterministic, no timestamp/random in
the builder path; request ids use uuid but are not prompt text). Tests:
`tests/unit/test_kpf21_25_trunk.py` (shared trunk of the expected length, UFH14-B4
`prefix_ladder_for_text` rungs equal inside the trunk).

| Builder | What varied inside the trunk | Fix |
|---|---|---|
| `scout_stage.run_scout` (INF-78 scouting) | Nothing. Message 0 (`_SYSTEM`, formatted with `root`, command caps, `turns`, `words`) is a function of the stage config and is byte-identical across targets. Per-target text (`Target:`, evidence, `Replies left`) is in message 1, after the junction. Later turns append after it. | None needed. Pinned by `test_scout_messages_share_a_byte_identical_system_trunk`. |
| `parallel_step_executor.StepExecutor._build_step_prompt` (delegate waves, `/api/delegate`) | **The per-child `## Task` section came FIRST**, before the shared `## Overall Objective` and `## Prior Context`: siblings shared 24 leading characters. Input keys were emitted in the step's own order (a permuted list broke a shared context). | With INDEX and FORK on: shared-first order (objective, prior context with sorted input keys, persona, task, expected outputs). Flag-off: the original bytes (test). |
| `StepExecutor._execute_step` -> `llm_call(persona=...)` | `LLMPrimitives._apply_persona_prefix` **prepends** the persona system prompt; steps with different personas diverged at byte 0. | Flag-on: the persona text is resolved by the executor and placed after the trunk (`persona=None` to `llm_call`). If the persona cannot be resolved the original path is kept. |
| `routes/delegate.py` | Builds no prompt: it computes waves and hands the TaskIR to `ProactiveDelegator` -> `StepExecutor`. | Covered by the two rows above. |
| HS-19 scheduled dispatch (`v1_subagent_link`) | Stage 1 only RECORDS the harness subagent tree; the orchestrator does not build or reorder those prompts (the harness builds them). | Nothing to fix on the orchestrator side; stage 2 (scheduling) must inherit this rule when it lands. |
| `typed_decisions/fanout_policy.py` | Decision policy only; no prompt text. | None. |

Residual risks (not changed): (1) the role system-prompt suffix and content-cache sampling key are
per role/request and identical across same-role siblings; (2) chat-template junctions need the
children to tokenize the trunk identically (a token-boundary property, checked server-side by
`checkpoint_at` `{"message": k}`; a mismatch drops the checkpoint with a WARN); (3) wave siblings
with different `inputs` share only the objective, by construction.

## KPF-21 (what was added)

`src/inference/trunk_prefill.py` + `scout_stage.PoolGatedTransport`: behind
`ORCHESTRATOR_PREFIX_INDEX` + `ORCHESTRATOR_PREFIX_INDEX_FORK` (`1` or `auto`) and only when
`/props.slot_fork` advertises the fork and the trunk is at least `min_tokens`, the first admitted
sibling sends ONE non-streamed `max_tokens: 0` chat request (with
`checkpoint_at: [{"message": 0, "at": "end"}]` in `checkpoint` mode; none in `kv` mode) carrying its
own messages, then calls the gate's `prefill_done` so held siblings are released after the trunk
prefill completed. Siblings admitted meanwhile wait on, or skip, the same request (registry, 60 s
TTL, failures not retried). Any failure leaves the pre-existing path. The `messages`-form is used
because a bare system-only chat prompt is not renderable on some templates (Qwen).
Not wired: `llm_call` waves (delegate) have no per-site payload hook; they keep the gate's central
trunk-first ordering.

## Behaviour change under FORK

The KPF-25 reorder and persona relocation apply only with `ORCHESTRATOR_PREFIX_INDEX` on AND
`ORCHESTRATOR_PREFIX_INDEX_FORK` in {1, auto} (`auto` is accepted without reading `/props.slot_fork`,
which is not cheaply available in the executor path). INDEX on with FORK off (the KPF-27d shadow
window) leaves prompts byte-identical. Under FORK the persona moves from the system prompt that
`llm_call` prepends into the user prompt, after the trunk: a semantic change. KPF-26 must check task
outcome on persona-bearing waves.
