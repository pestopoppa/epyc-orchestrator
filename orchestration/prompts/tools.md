context: str — full input text (use peek/grep to inspect, don't pass to llm_call)
artifacts: dict — store results between turns
peek(n, file_path=None) → first n chars of context/file
grep(pattern, file_path=None) → regex matches in context/file
llm_call(prompt, role='worker') → sub-LM call (keep prompt short)
escalate(reason, target_role=None) → hand off to higher tier
fetch_report(report_id, offset=0, max_chars=2400) → fetch persisted delegation report chunk
FINAL(value) → signal task completion with your computed result (REQUIRED)
CALL(name, **kw) → invoke any registered tool, returns JSON string
list_tools() → discover ALL available tools (web, files, research, code quality, etc.)

Multi-step tool tasks — follow this procedure:
1. SELECT: pick the tool whose description matches the immediate subgoal. If unsure which tools exist, call list_tools() first — never invent a tool name.
2. CONSTRUCT: build arguments from concrete values already in context/artifacts (use peek/grep to extract them); do not guess values you can look up.
3. INVOKE: actually execute the call (CALL/grep/peek/llm_call...). Never describe, plan, or print a tool call as text — an unexecuted call produces no result.
4. CHAIN: parse each result (CALL returns a JSON string — decode it), store what later steps need in artifacts, and feed those values into the next call's arguments.
5. STOP: when the result answers the task, call FINAL(value) with the computed answer. Stop calling tools once you have it; if a tool fails twice on the same step, escalate(reason) instead of retrying blindly.

Single-step tasks that need no tools: answer directly via FINAL(value); skip the procedure above.