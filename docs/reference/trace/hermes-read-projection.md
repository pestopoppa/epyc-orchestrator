# Hermes read projection (ID-19)

`src/trace/hermes_reader.py` is a standard-library, read-only SQLite reader for one Hermes session. It emits the `session`/`messages` input envelope accepted by [the pinned Hermes adapter](https://github.com/letta-ai/trajectory/tree/b8c508fd1e05d1eeb1e2c7e13f41b8c2f3f09773/src/adapters/hermes), preserving native row-id order, active history, reasoning, tool-call JSON and untruncated tool text. It does not adopt or reimplement the complete normalizer. Unknown-role rows retain content-free ordering barriers so upstream call/result linking cannot accidentally cross an injected system row.

The upstream Python client at that revision still delegates to its bundled Node20+ runtime. This reader and exporter have no Node, subprocess, model, embedding or new package dependency. The existing `harvest_hermes_state` task-pair extractor remains unchanged; it serves a different purpose and omits tool/reasoning trajectories.

## Private data and fidelity boundary

The SQLite connection uses `mode=ro`, query-only mode and a read transaction. Only a fixed allowlist of conversation and session-display fields is returned. Billing data, user identity, session system prompts, token counts, latency, costs, outcomes, phase tags and resolved executor-build identity are not projected. Content itself remains local-private. This lossy field projection is for an agent to read; the original SQLite store remains the source of record. It is not a replayable or measurement-grade artifact and must never be written back as one or cited in a decision-gating number. No success/correctness, cost or measurement warrant is inferred.

The optional exporter creates a fresh file with0600 permissions in an existing owned0700 directory outside a Git tree. Parent symlinks, ambiguous paths, existing outputs and file symlinks refuse. Directory-relative writes and inode checks detect directory substitution; a failure leaves any owned partial for inspection and never unlinks or renames anything. These controls assume a cooperative owner; they are not a sandbox against hostile same-UID processes. Stdout contains aggregate counts only. Never put conversation exports, session IDs or private reference outputs in Git or public logs.

```
python scripts/trace/export_hermes.py PRIVATE_STATE_DB PRIVATE_SESSION_ID --out PRIVATE_0700_DIRECTORY/fresh-projection.json
```

## Reference parity and tests

[Sanitized parity summary](id19-hermes-reader-parity-20261031.json) pins the exact upstream commit and compiled reference bundle. Qualification ran that actual pinned Node bundle independently on raw native SQL envelopes and on the Python projections, with tool truncation disabled for the private comparison. Of five local sessions, three were eligible: all their normalized records were byte-equal. The other two yielded the same upstream `missing_user_records` refusal on both inputs; they are not successful eligible cases. All ten native rows were retained. Two public fixture pairs also matched both the real reference and its committed expected records; an actual reference negative detected deliberate removal of reasoning/tool calls. This is eligible-row recovery on this small available corpus, not the upstream project's1050-session validation claim.

Private text, IDs and raw reference outputs stay only in the protected qualification capsule. Public evidence contains aggregate counts, source identities and sanitized proof hashes. The source SQLite/WAL bytes stayed stable through qualification. Twelve unit controls cover public fixture recovery, optional columns, row ordering despite clock regression, bound SQL parameters, strict active-value types, long multibyte tool/reasoning retention, privacy-field omission, count-only stdout, missing schema, output freshness/permissions, Git/parent/file-symlink refusal, directory replacement and write-error refusal. Ruff passes. The existing staged privacy scanner passes the candidate (rc0), and an active credential-shape negative refuses (rc1); its exact byte identity is in the summary. No real private data is staged.
