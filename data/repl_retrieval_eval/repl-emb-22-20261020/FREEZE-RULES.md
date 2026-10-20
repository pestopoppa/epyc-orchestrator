# REPL-EMB-2.2 launch rules (frozen 2026-10-20, before any arm ran)

## Operator rulings (2026-10-20, relayed by the coordinator; quoted)
1. Broaden the real-trace source (c): "Real-trace queries now also include Claude Code's own targeted lookups over large files: Grep hits, Read with an offset, and single sed ranges, over docs and source. Gold is the lines it actually read. Query is its stated intent: narration, or the Bash description with the answer-naming filter. Label it as a distinct distribution: agent code/docs navigation, not REPL spills. ... The rule's numbers (k, M, X, n >= 120 and >= 75 trace, CI) are UNCHANGED." Same exclusions (trivial-head, one span per lookup, leakage filter, PII scan, no raw transcripts committed); freeze the source rules and a per-session cap.
2. Supersedes live capture (b): operator: "can't we quickly simulate traces artificially for the purpose of the test? ... I don't see why we need to mess with the stack when all we're doing is testing embedder quality." Build a SYNTHETIC spill-follow stratum offline: realistic large tool outputs on disk, queries written from an information need tied to one gold span (writer A sees the span and must paraphrase; writer B sees only a one-line need description), writer prompts frozen before generation, leakage filter plus an n-gram filter (>= 3 consecutive tokens verbatim in the gold span => drop), gold fixed at generation, blind to every arm. The synthetic stratum counts toward the >= 75 trace floor by operator ruling, is labelled `synthetic-spill`, and is reported as its own stratum; the verdict is reported pooled, per stratum (probe45, agent_nav, synthetic) and with and without synthetic.
3. Known bias (stated): LLM-written queries tend to share vocabulary with their source span, which favours lexical arms; this makes the kill rule conservative toward dense arms.
4. FP32 EmbeddingGemma 2 is dropped as a recall arm (identical to BF16 at half the speed). Arms: grep, BM25, BGE-large f16, Granite-97M-R2, ColGREP (probe45 only), EmbeddingGemma 2 BF16 @768 and @256 (GGUF, experimental llama-server), potion-code-16M-v2, potion-base-8M; each dense arm dense-only and hybrid (shipped rrf_fuse with BM25).
5. Kill rule decided FIRST (hybrid recall@5 - L >= 0.10 and paired-bootstrap 95% lower bound > 0, n >= 120), selection second (cheapest within 0.05 of best by speed-block CPU-s per query, then p95 latency).

## Distribution caveats (recorded beside "no REPL spill-follow or context-bundle traces exist")
- No persisted REPL `_spill_output` traces or context-bundle traces exist (spill dir empty, no REPL turns in progress logs, tap holds code only). The spill_follow source mined from transcripts yielded n = 0 under the frozen rules (what-if variants V1/V2/V3 also 0).
- agent_nav rows are Claude Code tool lookups (mostly subagents), not REPL spills; intent text is terse (mostly Bash descriptions).
- Synthetic rows are written by LLM writers over on-disk logs, git history, listings and result files; two queries per span (A and B) are correlated; bootstrap is by query.
- probe45 is the 45 questions of intake-1357#03 (act-context@fd115351) over seaborn/pylint/pytest source at pinned commits; they were minted by the paper authors from test files.

## Source A: agent_nav (miner rules, verbatim)
# REPL-EMB-2.2 `agent_nav` query stratum: FROZEN MINING RULES (trace2)

Source decision (operator, relayed 2026-10-20): broaden the trace stratum with a second source, `agent_nav` = Claude Code's own
targeted lookups over LARGE TEXT FILES (docs, source, logs, spills) found in the same transcripts
(`/home/node/.claude/projects/**/*.jsonl`). `trace/` and `trace/whatif/` are untouched; this directory is separate.
No LLM labels anything. No raw transcript text, file content or secret is written to any output (queries are the assistant's own
prose, PII-scanned; bundles are path + sha256 + counts only). Run: `python3 -I /mnt/raid0/llm/tmp/repl-emb-22-20261020/trace2/mine_nav.py`.
`mine_nav.py` imports the frozen `trace/mine.py` (parsers, grep emulation, chunker loader, PII scanner, stopword loader); the
stopword list used is the one in `trace/RULES.md` (sha256 recorded in stats.json). Counts are recorded at every stage in `stats.json`.

## Part 1: the rules as given (verbatim)

LOOKUP TYPES (one span per lookup):
- Read with an explicit offset >= 2 and an explicit limit <= 200 lines. Gold = offset..offset+limit-1 clipped to the file length. Reads with an offset but no limit are excluded, because they read up to 2000 lines.
- Grep tool with a single FILE path (not a directory or glob). Gold = the file's lines matching the pattern with the same flags (-i, -n, -C ignored). Keep only if 1..40 matched lines and < 20% of the file; merge ranges.
- Bash `sed -n 'A,Bp' FILE`, or `grep ... FILE`, or a V1 pipeline whose first command is one of those and whose later stages are only from {cut, sort, uniq, awk print-only, tr, wc, head, tail, cat, nl}.
- Large file: >= 400 lines AND 20 KB <= size <= 1 MB, valid UTF-8.

STALENESS CHECK (mandatory): the file as it is on disk now must equal what the agent saw.
- For Read: the tool_result returns numbered lines. Require that every returned line equals the current file's same-numbered line (after stripping the number prefix and a trailing newline).
- For Grep/sed with returned content: do the same check on the returned lines.
- If no returned content exists to verify, exclude the lookup and count it. Never fake a pass.
- Count `stale_excluded`.

INTENT TEXT:
- Use the assistant text in the SAME message preceding the tool_use, or in the immediately preceding assistant message. Length is 25..600 characters.
- Also use the Bash tool_use `description`, subject to the same leakage filter.
- The leakage filter rejects any text that contains: the file path or basename, a line number or line range, or the grep pattern verbatim. Report how many it removed.
- Label `intent_source` as assistant_text | bash_description.
- Keep the flag `answer_named_verbatim` and `lexical_overlap` exactly as in the frozen rules.

SHARED EXCLUSIONS (unchanged): R0 same-day sessions excluded, R3 trivial-head (gold starts at line 1 or offset <= 1, written to a separate `queries-trivial-head.jsonl`), R4 chain collapse (final lookup on the same file in a chain), and R7 PII scan over the query texts AND every bundle file (counts only).

BUNDLE: bundle = the target file alone. Section name = the absolute path, must be <= 4096 chunks via the shipped chunker, and gold must lie in it. Record the file's sha256.

CAPS (frozen): at most 2 queries per file and at most 6 per session, where a session is the parent session id and subagent transcripts count toward the parent. Selection when over cap is deterministic: rank by sha256(f"agent_nav:{tool_use_id}") hex ascending and keep the first N. Report how many the caps removed, plus the max single-session share.

LABEL: every row carries `stratum: "agent_nav"` and `distribution: "agent code/docs navigation (Claude Code), not REPL _spill_output"`, plus `source` = {transcript, line, tool_use_id, lookup_kind}.

OUTPUTS: `queries-trace.jsonl`, `queries-trivial-head.jsonl`, `bundles.json`, `stats.json` (funnel at every stage), `digests.json`, `RULES.md`, the script, and a deterministic re-run check (two runs, identical digests).

Stop condition: if final scored n < 75, STOP and report the stage counts; no rule is loosened by the miner.

## Part 2: operationalisation (every place the rules left a choice; fixed BEFORE the first run)

Where a choice could go either way the stricter reading is taken. These are not new filters; they define terms the rules use.

P1 Universe. All non-R0 transcripts (any transcript, not only those that contain spill announcements). R0: transcript file mtime >= 2026-10-20T00:00Z is excluded. Session id = `<sid>` of `<project>/<sid>.jsonl` or `<project>/<sid>/subagents/agent-*.jsonl` (subagents count toward the parent). Duplicate tool_use ids inside a transcript are collapsed (first kept).
P2 Lookup candidate (the unit that forms chains) = a tool_use with a resolvable single target file:
  - `Read`: target = `file_path`.
  - `Grep` tool: target = `path` if it is an existing regular file (a directory/glob/absent path is not a lookup and breaks chains).
  - `Bash`: after optional leading `cd X` / `VAR=value` statements, ONE pipeline statement whose FIRST stage is `sed` or `grep` with exactly one positional file operand (no glob characters). Relative operands are resolved against the last absolute `cd` target in the command, else the transcript event's `cwd`; a wrong resolution is caught by the staleness check, never trusted.
  Any other tool_use (including Bash that is not of that shape, Edit, Write, ...) is "another tool call" and breaks chains.
P3 Chain (R4). In transcript line order, a chain is a maximal run of lookup candidates with the SAME resolved target path, broken by a real user turn (a `user` event with no tool_result block that is not `isMeta`, or a `queued_command` attachment whose prompt does not start with `<task-notification>`) or by any other tool_use. Only the FINAL candidate of each chain is kept; earlier ones are dropped and counted, whatever their kind. The kept final is then classified; if unsupported the chain yields no row.
P4 Line model. A file's lines are its UTF-8 text split on `\n` only; a trailing `\n` does not start another line (same as the shipped chunker). nlines = number of lines. The large-file test is done on the file as it is on disk now: it exists, is a regular file, 20480 <= bytes <= 1048576, strict UTF-8 decode, nlines >= 400. Failing files are dropped with the reason counted (missing, too_small_bytes, too_large_bytes, not_utf8, fewer_than_400_lines).
P5 Read. offset and limit must be integers. Scored iff offset >= 2 and 1 <= limit <= 200; gold = [offset, min(offset+limit-1, nlines)]; offset > nlines = gold_out_of_range drop. Trivial-head (R3) iff offset is absent or <= 1 AND an explicit 1 <= limit <= 200 is present (gold [1, min(limit, nlines)]). Everything else is unsupported: offset with no limit, limit > 200, no offset and no limit, `pages`, non-integer values.
P6 Grep tool. Supported only when `pattern` is present, `path` is a file, `multiline` is not true, no `glob`/`type` filter is set. The pattern is applied with Python `re` per line (ripgrep syntax is close to Python for the patterns seen; a pattern that does not compile is unsupported_grep_pattern); `-i` is honoured, `-n`/`-A`/`-B`/`-C`/`head_limit` are ignored for the gold. Caps: 1 <= matched <= 40 and matched < 20% of nlines; ranges merge only when adjacent.
P7 Bash sed/grep first stage: parsed with the frozen `mine.parse_sed` / `mine.parse_grep` / `mine.grep_matches` (same flag set: grep n i E F G w x v e H h s a o; sed -n with a single `A,Bp`, `Ap`, `A,$p` script). The same grep caps as P6. sed gold = [A, min(B, nlines)]; A > nlines = gold_out_of_range drop.
P8 V1 pipeline. Later stages must all be on this exact allowlist and nothing else: `cut sort uniq awk(print-only) tr wc head tail cat nl`. awk is allowed only as `awk [-F<sep> | -F <sep>] '{print ...}'` with one program argument matching `^\s*\{\s*print\b[^{};|<>()=~!]*\}\s*$`. Every other allowlisted stage must carry no operand beginning with `/`, `./`, `../`. Compounds (`;`/`&&` beyond leading cd/assignments), multi-line commands, redirects, `||`, subshells are unsupported. Gold = the first-stage span.
P9 Tool results. The tool_result for a tool_use is the user-event tool_result block with that `tool_use_id`; its content is the block string (or the joined text blocks). A trailing `<system-reminder>...</system-reminder>` block is removed before comparison. No tool_result, an `is_error` result, or empty content = no returned content.
P10 Staleness check (all strict equality; first mismatch = stale).
  - Read: every returned line must match `^(\d+)\t(.*)$`; the numbers must be exactly offset..offset+k-1; k must equal the gold length; and each text must equal the current file's same-numbered line. A result that cannot be parsed as numbered lines is "no returned content".
  - Bash sed: returned stdout lines (one trailing empty element from a final newline removed) must equal the current file lines A..B exactly, in order, and the counts must be equal.
  - Bash grep: expected output is rebuilt from the current file and the flags: matched lines in order, prefixed `N:` when `-n`, prefixed `<path>:` when `-H` (`-n -H` gives `<path>:N:`), `-o` is unverifiable (output is fragments). Returned lines must equal it exactly.
  - Grep tool: only `output_mode = content` returns content; each returned line `N:text` (match) must equal the current file line N and the set of returned match numbers must be a subset of the recomputed match set. `files_with_matches`/`count` modes return no content = unverifiable.
  - V1 pipeline: the first-stage expected lines are pushed through the later stages ONLY when those stages are `cat`, `head [-n N | -N]`, `tail [-n N | -N | -n +K]` (exactly simulated); the returned lines must equal the simulated output. If any later stage is cut/sort/uniq/awk/tr/wc/nl the output cannot be compared with the file, so the lookup is excluded and counted as `unverifiable_shaper` (never passed). Output of a Bash command that was truncated into a persisted-output file, or that carries any extra line, simply fails equality and is counted as stale mismatch (sub-counted as `persisted_output`).
  `stale_excluded` = lookups with returned content that does not match; `unverifiable_excluded` = lookups with no returned content to verify (reason-counted). Both are removals; neither is ever a pass.
P11 Intent text. Candidates in order: (1) assistant_text = the frozen R5 text (the text blocks that precede the tool_use inside the same assistant message, else all text blocks of the immediately preceding assistant message, joined by a blank line, stripped); (2) for Bash only, bash_description = the stripped `description` field. The first candidate that passes BOTH the length rule 25..600 AND the leakage filter is the query; if none passes the lookup is dropped (reason = the first failing reason of the first present candidate, empty if none). `intent_source` is the passing candidate's label. The leakage filter rejects a text containing: (a) the absolute file path or the file basename (case-sensitive substring); (b) a line number or range: `\b(lines?|offset|L)\s*\d+` (case-insensitive), `\bL\d+\b`, `<name>.<ext>:NN`, or a range `\d{1,7}\s*(-|en/em dash|..|to)\s*\d{1,7}` after ISO dates are removed; (c) the grep pattern verbatim: for lookups that have patterns, a case-insensitive substring if the pattern is >= 3 characters, else a whole-word match. Counts removed are reported per source and per reason (independent counts plus first-failing counts).
P12 Lexical leakage flag: exactly the frozen O13 (tokens `[a-z0-9_]{4,}` lowercased, Q = tokens(query) minus the trace/RULES.md stopword list, T = tokens(gold span text), overlap = |Q n T| / |Q| rounded to 4 decimals, `answer_named_verbatim` = Q non-empty and (|Q n T| >= 3 or overlap >= 0.5)). Diagnostic only (not a filter): the number of final rows whose text contains a standalone integer equal to a gold range bound is reported as `info_gold_bound_number_in_text`.
P13 Bundle. bundle = the target file alone; section name = the absolute path; total chunks from the shipped `chunk_sections([section], 800)` must be <= 4096; gold range must lie inside the file's lines. bundle_id = sha256 hex of `"{path}\t{sha256_of_file}"`. `bundles.json` entry = {session_id (first referencing session, sorted), sessions (all, sorted), files:[{path, sha256, bytes, chunks}], total_chunks}.
P14 PII (R7): the frozen `mine.pii_scan` (hook-equivalent secret and account_number patterns) on each kept query text and on each target file's text. A file with a hit is dropped (all its rows). Counts and file ids only are ever printed.
P15 Caps. Applied LAST, after every other filter, independently inside the scored set and inside the trivial-head set: (C1) per file: rows sorted by rank = sha256(f"agent_nav:{tool_use_id}").hexdigest() ascending, keep the first 2 per file; (C2) per session: among the survivors, sorted by the same rank, keep the first 6 per session. Removed counts are reported for C1 and C2 separately, plus the max single-session share of the final set.
P16 Order and ids. Final rows sort by (session_id, transcript line, transcript path); ids `agn-0001...` (scored) and `agt-0001...` (trivial-head). Both files carry `stratum: "agent_nav"`; trivial-head rows additionally carry `trivial_head: true`. Extra fields: `follow_kind` (read|sed|grep), `file_class` (`spill` if the path contains `/tool-results/`, else `other`), `intent_source`, `code_intent: null`.
P17 Funnel order: transcripts (R0) -> tool_uses -> lookup candidates -> chain collapse (R4) -> classification of finals (unsupported / large-file / gold range / grep caps) -> R3 split -> staleness (stale vs unverifiable) -> R5 intent text (leak filter, length) -> R6 bundle (chunk cap, gold in file) -> R7 PII -> caps -> final.
P18 Determinism. No timestamps, no randomness; all iteration is over sorted inputs. The only external state is the on-disk target files (their sha256 is recorded); two runs minutes apart see the same files unless a file is edited in between, which would also change its recorded sha256.
P19 Amendment recorded after the pilot run (before the final runs; made by the miner, flagged to main so it can veto). The first (pilot) run of `mine_nav.py` used strict list equality for Bash stdout. A diagnostic over the stale Bash-sed lookups showed the Bash tool strips leading and trailing WHITESPACE-ONLY lines from stdout, so 109 of 594 `sed` mismatches (58 trailing-blank, 51 leading+trailing-blank) were an artifact of the check, not file changes (no indentation or tab artifacts were found; every other mismatch is a genuine content difference). Amendment to P10 (Bash sed, grep, pipelines): the expected list is compared as-is OR after removing leading/trailing whitespace-only lines from the expected list (the tool's own trimming); nothing else is relaxed, every returned line must still equal its same-numbered file line. Read is unchanged (numbered lines, blank lines are returned as `N<TAB>`). The pilot figures (strict equality) were: 29 scored / 20 trivial-head rows from 6 sessions, recorded in the report for comparison. `stats.json` also gains `distinct_sessions_files_at_stage` (sessions and files surviving each stage) so the binding stage is visible.



## Source B: synthetic_spill (generation spec)
- corpus builder `synth/build_corpus.py` sha256 6039bcf9234054b39fe347d9e10a9b677e4b39e3ae48da5ef28f4eae4a560f1d; bundles.json 98e0acb55dacca7f9db3072a234dc4cbf6779f03b28e9aa3cdf8c07d7ae7468b; spans.jsonl 0fafd8c8b2a913cfedc0e933e9e3dffb14d6773ede926c08e88e9093ebcf4561; filters.py 0469acc342ac7cd14e62efa01b862d6b0b1dc44f71227a2e21f697e8daa80b37; assemble.py df24bba32d31a89f55906c085c2d3b6ccef110cebad0d0fdbc29c5101d79d0e1

### Frozen prompt `describer.md` (sha256 432750ce4949b4bce0d3c5ef8dc7c299355fd9e5b71977afd5d24cab39978abe)

# Describer (sees the span) — FROZEN PROMPT v1

You are given a batch file of items; each has an `id`, a `span_text` (3-15 lines of a large tool output) and a few lines of context. For EACH item write ONE neutral description of the NEED that the span satisfies, as a short noun phrase or sentence of at most 20 words — what kind of information lives there, e.g. "why the second compilation step stopped" or "which module was changed most in the oldest listed commit". 

HARD RULES: do not copy identifiers, file paths, function names, flags, hashes, numbers or line numbers from the span or context; no quotes, no code; do not mention a file, line, span or position; write only your output file and read only your batch file; run no other commands.

OUTPUT: JSONL, one object per line, same order: {"id": "<item id>", "need": "<description>"}

### Frozen prompt `writer_A.md` (sha256 4e92a1a9f2454bbda326383f3520f3b10e41dfcc51029bee1216c1281e717be4)

# Writer A (sees the span) — FROZEN PROMPT v1

You are generating SEARCH QUERIES for an offline retrieval benchmark. You are given a batch file of items. Each item has an `id`, a `span_text` (3-15 lines from a large tool output such as a build log, a git log -p, a grep listing, a directory listing, or a JSONL results file) and a few lines of `context_before` / `context_after`.

For EACH item write exactly ONE search query: the question or information need that a developer reading a huge output would have when the answer is exactly in `span_text`. The query will be used to find that span among thousands of other lines.

HARD RULES (a query that breaks one is discarded):
1. PARAPHRASE. Describe the need in your own words. Do NOT copy identifiers, file paths, function names, flags, hashes, commit ids, version strings, numbers, line numbers or any run of 3 or more consecutive words from the span or its context.
2. Do not mention that there is a file, a line, a span, an offset or a position. Do not say "in the log" with a name or path.
3. 8 to 30 words, one sentence, natural language, no quotes, no code, no backticks.
4. The query must be answerable from the span alone, and specific enough that the span (not the rest of the output) is the best answer — but expressed through concepts and behaviour, not through the span's own wording.
5. Use only the batch file you are told to read and write only your output file. Do not read any other file, do not run any command other than reading the batch file and writing the output, and do not look at other batches' outputs.

OUTPUT: a JSONL file (one JSON object per line, no other text): {"id": "<item id>", "query": "<your query>"} — one line per input item, in the same order.

### Frozen prompt `writer_B.md` (sha256 574f42d17e0ce873c26c116db63a0dfbffb6ef3103205c8c35dcd4ac8f35f0e7)

# Writer B (sees only a one-line need) — FROZEN PROMPT v1

You are generating SEARCH QUERIES for an offline retrieval benchmark. You are given a batch file of items; each has an `id` and a one-line `need` describing information that exists somewhere in a very large tool output (a build log, git history, grep listing, directory listing or results file). You do NOT see the output.

For EACH item write exactly ONE search query (8 to 30 words, one natural-language sentence, no quotes or code) that a developer would type to locate that information. Be concrete about the topic but do not invent specific identifiers, file names, numbers or versions that are not in the `need`.

Read only your batch file and write only your output file; run no other commands.

OUTPUT: JSONL, one object per line, same order: {"id": "<item id>", "query": "<your query>"}


### Leakage + n-gram filters (filters.py docstring)

"""Shared leakage filters for the REPL-EMB-2.2 synthetic spill-follow stratum.

Stdlib only.  Import it by path (``importlib``) or run ``python3 -I filters.py`` to execute
``selftest()``.

``leakage_reasons(query, span_text, doc_path, start_line, end_line)`` returns the list of
reason strings for which a writer-produced search query leaks the location or the wording of its
gold span.  An EMPTY list means the query passed every filter.

Reason strings (stable, machine-readable):

  length_out_of_range     len(query) not in 20..400
  doc_name_in_query       query names the document: full path, basename, stem (``turn_3``), or the
                          16-hex bundle id (any path component that is 16 lowercase hex chars)
  line_ref_pattern        ``\\b(lines?|L)\\s*\\d+`` or a ``\\d+ - \\d+`` range
  line_number_in_query    a digit run equal to a line number in [start_line-2, end_line+2]
  number_in_span          a digit run of length >= 3 in the query that is also a digit run of the span
  ngram3_verbatim         >= 3 CONSECUTIVE query tokens occur consecutively in the span's token
                          sequence (tokens: lowercase, split on non-alphanumeric/underscore)
  identifier_verbatim     an identifier-looking query token (>= 8 chars with ``_`` or a digit-letter
                          mix, or CamelCase) occurs verbatim (case-insensitive, whole token) in the span
"""

## Pre-run amendment (2026-10-20, coordinator, made BEFORE any arm ran): PRIMARY decision set excludes flagged rows
- Reason (coordinator): the pooled set is 81% synthetic (482 of 593 counted at first freeze), and 204 of those 482 synthetic queries (42%) are flagged `answer_named_verbatim`; left in the deciding set, that leakage would let the lexical arm dominate and could kill dense retrieval through a query-writing artefact.
- PRIMARY (the only deciding set): every stratum with `answer_named_verbatim` rows EXCLUDED: probe45 (45, none flagged) + agent_nav unflagged (23) + synthetic_spill unflagged (278; writer A 110, writer B 168) = 346 total, 301 trace rows. Kill rule, L, the n >= 120 and trace >= 75 floors, selection, the bootstrap tables and the extension-flip logic use this set only. If the floors do not hold, stop; do not run.
- SECONDARY (non-deciding, labelled as such): pooled including flagged rows (560); without synthetic (probe45 + agent_nav, flagged excluded and included); each stratum separately.
- Leakage measurement: recall on the FLAGGED subset reported separately from the unflagged subset, per writer and stratum, so the size of the leakage effect is measured. Flag rates at freeze: writer A 124/234 (53%), writer B 80/248 (32%), agent_nav 10/33 (30%); writer B never saw the gold text, so a 32% flag rate there means the flag (>= 3 distinct 4+ character query tokens, or >= 50% of them, present in the gold span) is broader than "names the answer" and also fires on shared domain vocabulary. The flag is kept as defined (not tuned after the fact); the leakage cut shows what it costs.

## Operator ruling on timing (2026-10-20, relayed by the coordinator; quoted verbatim)
"accuracy is not performance sensitive. You can do it now if other CPU tasks running aren't performance sensitive either. We know the EG 2 is significantly faster than BGE, we don't need to get precious about measuring the speedup properly on a quiet host."
- Consequences recorded: the recall legs run now on a region that is free (q2) or, if none is, co-run on q0's CPUs under `nice -n 19` pinned with `taskset` (a DEVIATION from the build-claim practice; q0's holder is correctness work, not timing). Never q1 (timing-sensitive bench claim held by another session); never q3 (operator session).
- The quiet-host speed block is DROPPED. The 2.1 selection rule's cost tie-break uses the pilot's measured speeds (REPL-EMB-2.7 pilot note, labelled contaminated) and any incidental per-query timing the recall run logs (also labelled contaminated). No exclusive bench claim is taken.
- Deadline: done or stopped cleanly at an arm boundary by 00:15Z; any remainder resumes after 06:30Z.
