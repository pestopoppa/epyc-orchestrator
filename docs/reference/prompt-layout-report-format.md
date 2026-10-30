# Prompt layout reuse report (RTE-LOG-1)

Producer: `src/backends/prompt_layout.py` adds `blocks[]` (and `turn_index`, `source`,
`layout_match`, `layout_chars`, `unlocated_blocks`) to `dispatch.v1` and `serving_call.v1`
records, plus `prompt_tokens` / `cached_prompt_tokens` on `dispatch.v1`. Kill switch
`ORCHESTRATOR_PROMPT_LAYOUT_LOG=off`. Privacy: kind, character offset, length and a 12-hex
sha256 prefix only; chunks add a hash, rank and length. `source` is set by
`ORCHESTRATOR_REQUEST_SOURCE` (interactive, autopilot, benchmark, repl_experiment), never inferred.
Standalone direct calls now copy the existing serving record's layout projection into
`dispatch.v1` through an explicit field whitelist. They carry the serving record's
native prompt/cache counts, preserving `null` when the backend supplied no count.
Fleet calls retain their existing dispatch path; the direct path emits no duplicate.

Reader: `scripts/analysis/prompt_layout_reader.py [--format json|md] [--window N] [--role R] [--source S] FILES_OR_DIRS`.

## Record field

```json
{"blocks": [{"kind": "system", "start": 0, "length": 812, "sha256_12": "a1b2c3d4e5f6"},
            {"kind": "doc_chunk", "start": 1900, "length": 2400, "sha256_12": "...",
             "chunks": [{"sha256_12": "...", "rank": 1, "length": 1200, "start": 1900}]},
            {"kind": "skill", "start": 900, "length": 1100, "sha256_12": "...",
             "est_tokens": 275, "n_items": 6, "meets_min": true}],
 "turn_index": 2, "source": "autopilot", "layout_match": true, "layout_chars": 5400,
 "unlocated_blocks": 0}
```

Kinds: system, tools, rules, task, state, context, reference_code, skill, routing, doc_chunk, tool_output.
Offsets are characters of the rendered prompt (the unit of `prompt_chars`).

## Report (`epyc.orchestrator.prompt_layout_report.v1`)

* `inputs`: records read, records with blocks, requests analysed, window.
* `per_kind_role_source[]`: for each (kind, role, source) the occurrences and characters split into
  `first` (bytes not seen before for that role), `prefix_repeat` (seen before after an identical
  prefix: the server prefix cache already serves it), `moved_repeat` (seen before at a different
  offset and after a different predecessor chain: what a chunk-reuse path could add) and
  `pred_repeat` (same offset, different predecessor chain). Chunk entries replace their enclosing block.
* `per_request[]` per (role, source): requests, mean distinct chunks per request, chunk length
  median and mean, `plain_prefix_share_mean` (leading blocks in common with the best of the previous
  `window` prompts of the role) and `reconstructible_share_mean` (longest prefix of that earlier
  prompt whose blocks all occur, unused, in the incoming prompt: the KPF-61 reorder-planner idea at
  block granularity), and their mean difference. Shares are fractions of typed-block characters.

Priority event: document-QA over fixed 10-30K-token documents repeated per query shows up as a high
`moved_repeat` share on `doc_chunk` with a large `reconstructible_minus_plain_mean`.
