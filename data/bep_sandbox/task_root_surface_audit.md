# Phase 0 — Model-Facing Filesystem Surface Audit (BEP-2 / DCP-6 task-root harness)

Source of truth: `handoffs/active/bep-dcp-falsification-harness.md` (reviewed 2026-05-26).
Goal: classify every model-facing filesystem surface as **task-root** (must point at the scratch
task repo during a BEP/DCP A/B) or **project-root** (orchestrator control plane, stays real). The
harness is invalid if writes go to scratch but reads/tests still hit the orchestrator repo.

`ORCHESTRATOR_EDIT_ROOT` (default unset = today's behavior) = the **model task-root**. A new
`src/repl_environment/task_root.py` will expose `get_task_root() -> Path` (returns
`$ORCHESTRATOR_EDIT_ROOT` if set+exists, else `get_config().paths.project_root`) and
`resolve_task_path(p) -> Path` (relative paths resolve against `get_task_root()`, not cwd).

## Classification (code-grounded)

| # | Surface | Touchpoint (file:line) | Current anchor | A/B anchor | Notes |
|---|---------|------------------------|----------------|------------|-------|
| 1 | **Relative-path resolution base** for all file tools | `environment.py:489` `_validate_file_path` (`os.path.realpath(path)` → cwd) | process cwd + `ALLOWED_FILE_PATHS` prefix check | **task-root** | Relative task paths (`cart.py`) must resolve under task-root. Add `resolve_task_path` before realpath; ensure task-root is inside `ALLOWED_FILE_PATHS`. |
| 2 | `ALLOWED_FILE_PATHS` | `environment.py:91` `_get_allowed_file_paths` (= `llm_root` + `/tmp/`) | `llm_root`, `/tmp/` | **append task-root** | Scratch repo under `/tmp/` already validates; if elsewhere, add task-root prefix when set. |
| 3 | `file_write_safe` write/read/backup | `file_mutation.py` `_file_write_safe` (validates via #1) | validated abs path | **task-root** | Baseline interleaved edits must mutate scratch. Inherits #1's resolution. |
| 4 | `peek` / `peek_grep` / `list_dir` / `file_info` | `combined_ops.py:224` `_peek_grep` (+ list/info), validate via #1 | validated path | **task-root** | Model must inspect the same scratch repo it edits. Inherits #1. |
| 5 | `run_shell` cwd | `external_access.py:207` (`cwd=_get_project_root()`) | project_root | **task-root** | Verifiers/tests the model runs must exercise scratch. Swap to `get_task_root()`. |
| 6 | `run_python_code` cwd | `external_access.py:273` (`cwd=tmp_dir`) | config tmp | **project-root OK** | Isolated tmp; only matters if a task imports repo-local files → prefer driver-side verifier, or pass task path. Leave default. |
| 7 | `code_search` / ColGREP root | `code_search.py:117` `_code_search` (ColGREP over project source) | project source root | **task-root** | Reactive discovery (interleaved arm) + DCP candidates must come from scratch. Needs ColGREP root param/env. |
| 8 | `_batch_edit_repo_root()` | `helpers.py:341` (`_get_project_root()`) | project_root | **task-root** | Treatment stages/promotes into scratch. Honor `get_task_root()`. |
| 9 | `apply_patchset_sandboxed(repo_root=)` / `promote_sandbox(repo_root=)` | `batch_edit_runner.py:161/188` | caller-provided | **task-root** | Caller (#8) passes task-root → already parametric. |
| 10 | DCP `_file_reader_fn` | `chat_delegation.py` `_maybe_dcp_seed_context` (`_get_project_root()`) | project_root | **task-root** | DCP bundle bodies must be the task repo. Honor `get_task_root()`. |
| 11 | DCP `code_search_fn` | same (`deleg_repl._code_search`) | = #7 | **task-root** | Inherits #7. |
| — | **project-root (UNCHANGED)** | | | | |
| 12 | procedures / checkpoints / registry / benchmarks reads | `procedure_tools.py:123/165/200/282` (`_get_project_root()`) | project_root | **project-root** | Control-plane reads, not task content. |
| 13 | `log_append` | `file_mutation.py:54` (`_get_project_root()/logs`) | project_root | **project-root** | Runtime/audit log. |
| 14 | patch ledgers | `file_mutation.py:144/191/241/298` (`/orchestration/patches`) | project_root | **project-root (or disabled)** | Orchestration metadata; prefer disabled during A/B. |
| 15 | registry / config / session / model_registry | `config/models.py` (project_root-derived) | project_root | **project-root** | Runtime control plane must stay real. |

## Key implementation findings

- **Single chokepoint for read/write/inspect**: surfaces #1–#4 all funnel through
  `_validate_file_path` (`environment.py:489`) which does `os.path.realpath(path)` (cwd-relative) +
  `ALLOWED_FILE_PATHS` prefix check. Redirecting relative-path resolution there + appending task-root
  to `ALLOWED_FILE_PATHS` covers write + peek + grep + list + file_info in one place. **Lowest-risk
  design.**
- **`run_shell` is a separate cwd** (`external_access.py:207`) — independent swap to `get_task_root()`.
- **`code_search`/ColGREP** (#7) needs its search root parameterized — verify whether `_code_search`
  passes a root to the ColGREP CLI or relies on cwd; this is the one surface that may need a CLI-arg
  or env change (flagged for Phase 1).
- **Batch + DCP** (#8/#10/#11) are already callback/param-based — they just need to source
  `get_task_root()` instead of `_get_project_root()`.
- **Two distinct `_get_project_root()` copies** exist (`file_mutation`, `external_access`,
  `procedure_tools`) — the override must NOT blanket-replace them; only the task-root surfaces
  (#3/#5/#7/#8) switch to `get_task_root()`; control-plane uses (#12–#15) keep `_get_project_root()`.

## Phase 0 exit-gate tests (must pass before Phase 1 build proceeds)

With `ORCHESTRATOR_EDIT_ROOT=<scratch>` set:
1. model-facing **write** to `cart.py` lands under `<scratch>/cart.py` (not orchestrator).
2. model-facing **read/grep/list** of `cart.py` reads `<scratch>` copy.
3. model-facing **run_shell** test runs with cwd `<scratch>`.
4. batch-edit treatment **promotes only into `<scratch>`**.
5. DCP bundle renders **`<scratch>` files**.
6. `code_search` returns candidates from **`<scratch>`**.
7. orchestrator registry/config/log paths remain **real** (`get_config().paths.project_root`).
8. `git -C <orchestrator> diff` is **unchanged** after a model-write safety test.
With env **unset**: every surface behaves exactly as today (default-off parity test).

## Status

Phase 0 audit COMPLETE — classification grounded in code. Next: Phase 1 implements `task_root.py` +
the #1–#11 redirects (default-off), with the 8 exit-gate tests above. The `code_search`/ColGREP root
(#7) is the one surface flagged as possibly needing a CLI-arg/env change — to confirm first in Phase 1.
