# Cache-friendly injection: integration spec for REPL-embedding retrieval inlining

Status: spec only. As of origin/main a66e4f838 there is NO inlining path: REPL-EMB search returns
pointers `(source, line range, score, snippet)` and reads go through `get`/`peek` (handoff
`repl-embedding-retrieval.md`, invariants 1-2). The shared helper is
`src/prompt_builders/reusable_blocks.py` (flag `ORCHESTRATOR_CACHE_FRIENDLY_INJECTION`, default OFF).

If/when an inlining path is added (retrieved chunks placed in the prompt rather than read by the
model through `get`):

1. Chunk identity: item text = the chunk's source text only, byte-for-byte from the indexed file range
   (use the content hash the index already stores). Exclude score, rank, line numbers, timestamps and any
   per-request id from the item text; those belong in the pointer record / log.
2. Call `pack_blocks(chunk_texts)` (content-addressed; order-insensitive) and `render_blocks(...)`.
   Chunks are not split, so a chunk larger than `MAX_BLOCK_TOKENS` becomes a single oversize block, which
   is fine (it is already a reusable unit).
3. Neutralisation is applied inside `canonical_text` (`control_tokens`). Retrieved repo text is untrusted;
   keep the call-site `neutralise_control_tokens` on the final string as well (idempotent).
4. Where rank order carries meaning (reading order within one file), keep it INSIDE one chunk by
   retrieving contiguous ranges, not by ordering chunks.
5. Log `block_log_records(blocks, kind="doc_chunk")` into the B7 `blocks` field; never write the hash
   into the prompt.
6. Pointer invariants stay: inlining is a separate, flagged mode; `get`/`peek` accounting (OAB-12)
   must still count inlined bytes as pulled.
7. Gate: the quality A/B must pass before this is default; same flag.
