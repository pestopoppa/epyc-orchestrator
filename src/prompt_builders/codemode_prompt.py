"""Codemode reliability fix 3: the code-mode / tool-task root prompt, v2 (``ORCHESTRATOR_CODEMODE_PROMPT=v2``, DEFAULT off).

CODEMODE-AB found the shared ``MINIMAL`` root prompt hostile to code mode: rule 5 told the model to STOP after the first
CALL (anti-batching), a LeetCode few-shot and ``run_python_code`` ("ALWAYS test code!", blocked in tool tasks) were
advertised, and the AST sandbox refusals (``open``, ``dir``, ``getattr``, ...) were never stated, so the model re-sent the
same refused program each turn. v2 drops those, states the sandbox contract (generated from ``security.py`` and the REPL
globals so it cannot drift from what is enforced) and adds one batch-loop few-shot that ends in an explicit ``FINAL``.

Scope: only a builder created with ``PromptConfig(codemode=True)`` (the harness, and production only for a task_root-scoped
tool request) AND the flag set. The QA / coding prompt (everything else) is untouched, byte for byte.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from src.flag_env import flag_env

ENV_PROMPT = "ORCHESTRATOR_CODEMODE_PROMPT"
ENV_STUBS = "ORCHESTRATOR_CODEMODE_STUBS"


def prompt_v2_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = flag_env() if environ is None else environ
    return str(env.get(ENV_PROMPT, "")).strip().lower() == "v2"


def _stubs_on(environ: Mapping[str, str] | None = None) -> bool:
    env = flag_env() if environ is None else environ
    return str(env.get(ENV_STUBS, "")).strip().lower() in ("1", "true", "on", "yes")


def _security_names() -> tuple[list[str], list[str], list[str]]:
    """(blocked calls, blocked modules, allowed import modules), read from the enforcing code."""
    try:
        from src.repl_environment.security import ASTSecurityVisitor

        calls, mods = sorted(ASTSecurityVisitor.FORBIDDEN_CALLS), sorted(ASTSecurityVisitor.FORBIDDEN_MODULES)
    except Exception:  # noqa: BLE001 - the prompt must still build
        calls = ["compile", "dir", "eval", "exec", "getattr", "globals", "hasattr", "input", "locals", "open", "setattr", "vars"]
        mods = ["os", "pathlib", "shutil", "socket", "subprocess", "sys", "threading"]
    try:
        from src.repl_environment.environment import REPLEnvironment

        allowed = sorted(REPLEnvironment.SAFE_IMPORT_MODULES)
    except Exception:  # noqa: BLE001
        allowed = ["collections", "datetime", "itertools", "json", "math", "re", "statistics"]
    return calls, mods, allowed


V2_TOOLS = """\
CALL("web_research", query="...", max_pages=3) → web research (fetches pages, returns summaries)
CALL("web_search", query="...", max_results=5) → quick web search, JSON [{title, url, snippet}]
CALL("search_wikipedia", query="...") → search Wikipedia
context: str — full input text (use peek/grep to inspect, don't pass to llm_call)
artifacts: dict — store results between turns
read_file(path) → whole text of a file
peek(n, file_path) → first n chars of a file (file_path is REQUIRED for files)
grep(pattern, file_path=None) → regex matches in context/file
list_dir(path) → directory listing
file_info(path) → size / type of a path
file_write_safe(path, content) → write text to a file (creates parent dirs)
llm_call(prompt, role='worker') → sub-LM call (keep prompt short)
escalate(reason, target_role=None) → hand off to higher tier
FINAL(answer) → signal task completion (REQUIRED for every task)
CALL(name, **kw) → invoke any registered tool, returns JSON string; TOOL(name, **kw) → returns a Python object
list_tools() → [{name, description}] for ALL tools; it is long, so print only names: print([t["name"] for t in list_tools()])"""


V2_SYSTEM = """Simple questions: FINAL("answer") immediately. No code, no tools. Follow ALL user formatting constraints exactly.

Tool and file tasks: write Python in the sandboxed REPL. One snippet may call many tools and loop over many items. Finish the work, then call FINAL(answer) yourself.

Output ONLY valid Python — no prose, no markdown, no comment-only responses."""


def sandbox_contract() -> str:
    calls, mods, allowed = _security_names()
    return f"""## SANDBOX CONTRACT (a snippet that breaks it is refused BEFORE any line runs, so nothing in it executes)
- Blocked calls: {", ".join(f"{c}()" for c in calls)}.
- Blocked modules: {", ".join(mods)}.
- Allowed imports: {", ".join(allowed)}. `json`, `math`, `re`, `statistics`, `collections`, `itertools`, `datetime` are already loaded.
- Files: `read_file(path)` / `peek(n, file_path=path)` to read, `list_dir(path)` to list, `file_write_safe(path, content)` to write. Never `open()`, `os` or `pathlib`.
- Globals you can use: `context`, `artifacts`, `json`, the tool functions listed above, `FINAL`.
- `peek(n)` with no `file_path` reads `context`, not a file. Use `list_tools()` when a tool you need is not listed.
- A refused or failing snippet: change the named construct and resend the WHOLE snippet. Your earlier code is shown under Previous Turns when that is enabled."""


def _few_shot(names_form: bool) -> str:
    if names_form:
        listing = 'names = list_dir("data/inbox")                      # list of file names'
    else:
        listing = ('names = [e["name"] for e in list_dir("data/inbox")["files"]]'
                   "   # dict: {files: [{name, type, size}], total}")
    return f"""## EXAMPLE: batch task (read, loop, aggregate, write, finish) in ONE snippet
```python
{listing}
rows = [json.loads(read_file("data/inbox/" + n)) for n in names]
total = sum(r["qty"] for r in rows)
file_write_safe("out/summary.json", json.dumps({{"total_qty": total, "files": len(rows)}}))
print(total)
FINAL("wrote out/summary.json from %d files" % len(rows))
```
Do the whole loop in one snippet. Call FINAL explicitly, only after the work (the writes) is done."""


def v2_rules() -> str:
    return f"""## WHEN TO USE TOOLS vs DIRECT ANSWER
1. **Compute** (math, logic, data): write Python. Never web-search for computable answers.
2. **Answer directly** (facts you know): `FINAL("Paris")`. Do not search to confirm what you already know.
3. **Use tools for files and data you cannot know**: read it, process it in code, write results with `file_write_safe`.

{sandbox_contract()}

## CRITICAL RULES
1. **BATCH**: one snippet may call many tools, loop over many files and write many results. Do not stop after the first call.
2. **ALWAYS call FINAL(answer)** yourself, as the last statement, once the task is done. A tool call alone does not finish the task.
3. **Fix errors incrementally**: if a snippet fails, fix ONLY the broken part; do not repeat an approach that already failed.
4. Tool calls are code: write executable Python, not prose. Output ONLY valid Python, no explanations.

{_few_shot(_stubs_on())}

## EXAMPLES: Direct Answer (NO tools needed)
`FINAL("B")`  # multiple choice
`FINAL("42")`  # "What is 6 * 7?"

## OTHER RULES
- NEVER send full context to llm_call - use peek() or grep() first.
- Do NOT reason in Python comments. Think first, then write only executable statements ending with FINAL()."""
