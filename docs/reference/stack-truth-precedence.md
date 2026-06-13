# Stack Truth Precedence

This document defines the source-of-truth order for model and serving facts used
by generated stack-prior artifacts and downstream routing/scoring consumers.

## Rule

Consumers must not infer live model, memory, speed, or serving facts from prose,
comments, or role names. They must consume generated artifacts or structured
APIs that record source evidence and precedence.

## Precedence Order

1. **Live serving topology**
   - `orchestration/model_registry.yaml` `server_mode.*`
   - `scripts/server/stack_manifest.py` `ROLE_LAUNCH_META` and computed server
     classification
   - Runtime attestation from the launcher when available
   - Owns endpoint, port, server role, tier, shared server binding, launch
     binary, acceleration launch knobs, and memory residency for deployed roles.

2. **Model identity descriptors**
   - `orchestration/model_descriptors.yaml`
   - Owns physical model identity, role bindings, suite vectors, speed evidence,
     acceleration compatibility, modality, context, and known gaps.
   - Descriptors are keyed by physical model identity, never role name.

3. **Role metadata**
   - `orchestration/model_registry.yaml` `roles.*`
   - Owns narrative role intent, fallback metadata, role-local defaults, and
     benchmark-only model records.
   - Role metadata may enrich descriptors but must not override live serving
     topology when both are present.

4. **Historical or benchmark-only records**
   - `source_registry.yaml`, progress logs, archived handoffs, benchmark reports,
     and old AutoPilot checkpoints.
   - Useful as provenance and candidate evidence only. They never override
     deployed topology without a new structured registry/descriptor update.

## Conflict Handling

- `server_mode.*.tier` overrides `roles.*.memory.residency` for deployed roles.
  Example: if `server_mode.architect_general.tier` is `hot` and
  `roles.architect_general.memory.residency` says `warm`, generated consumers
  must treat the role as HOT and record the role metadata as stale/conflicting.
- Shared mmap roles inherit the physical model and server binding of the live
  server. Example: `coder_escalation` shares the frontdoor Qwen3.6 Q8 server;
  consumers must not double-count the full model load as separate memory
  pressure.
- Retired roles are absent from generated live priors unless explicitly marked
  legacy, benchmark-only, or test-only. Example: `architect_coding` must not
  appear in active q_scorer priors, launch manifests, or active routing chains.
- Descriptor rows with role-server conflicts are preserved with `known_gaps` so
  validators can report them, but production consumers must not silently learn
  live cost or quality priors from the conflicted row.
- Benchmark-only roles may appear in descriptor or research artifacts, but they
  must carry a non-live status and cannot satisfy deployed routing requirements.

## Required Generated Consumer Evidence

Every generated stack-prior role record must include:

- role name and physical `model_id`
- server role and endpoint/ports where known
- tier and derived memory cost
- throughput prior and quality prior or explicit gap
- acceleration/serving requirements
- source paths, hashes, and precedence notes
- known gaps/conflicts copied from source descriptors

The first generated artifact for this contract is
`orchestration/derived/stack_priors.yaml`, compiled by
`scripts/registry/compile_stack_priors.py`.
