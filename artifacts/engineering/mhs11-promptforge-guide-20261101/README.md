# MHS-11 PromptForge guide citation cleanup

Updated `docs/guides/meta-harness-operator-guide.md` to name current
`PromptForge` methods instead of stale `prompt_forge.py` line numbers. The guide
now describes the fourth validation layer as the static safety screen and
identifies `CODE_MUTATION_ALLOWLIST` and `screen_static_safety` in their
`src/autopilot_core/mutation_guards.py` source module.

Validation from the orchestrator checkout:

- `rg -n 'prompt_forge\.py(:[0-9]+|,\s*line\s+[0-9]+)|line\s+[0-9]+[^\n]*prompt_forge\.py' docs/guides/meta-harness-operator-guide.md` — no matches.
- Python AST verification — all seven cited `PromptForge` methods exist, and
  `CODE_MUTATION_ALLOWLIST` plus `screen_static_safety` exist in
  `src/autopilot_core/mutation_guards.py`.

This is a documentation-only repair; no behavior or measurement claim changed.
