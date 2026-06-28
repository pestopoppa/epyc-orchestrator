---
name: security-review
description: Use when reviewing orchestrator diffs, commits, dependencies, API routes, AutoPilot/controller changes, tool execution, model-stack config, launch scripts, generated artifacts, or runtime-state handling for exploitable security risk. Emits only exploit-path-gated findings with P0-P3 severity.
---

# Security Review

Use this skill for focused security review of the EPYC orchestrator. It complements normal code review: report only findings with a plausible exploit path, or say no exploitable issue was found.

## Inputs

- Scope: current diff, commit/range, file list, subsystem, dependency change, launch/config change, or generated artifact change.
- Trust model: local-only vs exposed API, authenticated operator vs untrusted caller, model/tool authority, filesystem scope, and whether runtime artifacts are trusted.
- Optional context: `AGENTS.md`, `CLAUDE.md`, `docs/ARCHITECTURE.md`, `orchestration/tool_registry.yaml`, `orchestration/model_registry.yaml`, relevant handoff, or threat-model notes.

If no scope is given, inspect the current diff first.

## Workflow

1. Scope the changed attack surface:
   - API and OpenAI-compatible routes under `src/api/routes/`.
   - AutoPilot, planner, StrategyStore, mutation, and eval paths under `scripts/autopilot/` and `src/autopilot_core/`.
   - Tool execution, REPL, MCP, filesystem, subprocess, and web/vision tools.
   - Stack/launch/config surfaces under `orchestration/`, `scripts/server/`, `scripts/registry/`, and `src/config.py`.
   - Security-sensitive generated or runtime artifacts, especially journals, strategies, short-term memory, attestation, and reports.
   - Dependency, setup, CI, Docker, and shell changes.

2. Discover candidate risks:
   - STRIDE: spoofing, tampering, repudiation, information disclosure, denial of service, elevation of privilege.
   - OWASP Web/API themes: access control, injection, SSRF, auth failures, unsafe deserialization, misconfiguration, vulnerable components, logging gaps.
   - OWASP LLM/agent themes: prompt injection, tool injection, sensitive information disclosure, excessive agency, system prompt leakage, vector/embedding poisoning, model output over-trust, unbounded resource use.
   - Supply chain: broad dependency ranges, lockfile drift, install scripts, vendored binaries, generated code, typosquatting, abandoned packages.

3. Validate exploitability before reporting:
   - A realistic attacker or compromised input source can reach the path.
   - The path crosses a trust boundary or weakens a security invariant.
   - A privileged sink is reachable: filesystem write/read, subprocess, network request, tool call, model-routing authority, persistent memory, journal/state mutation, secret/log exposure, or deployment/config change.
   - Existing guards, feature flags, allowlists, path validation, sandboxing, auth, and deployment constraints do not already block it.
   - Impact is concrete and the minimal fix is clear.
   - File/line evidence exists.

Do not promote checklist-only concerns. Put uncertain but plausible concerns under residual risk.

## Severity

- `P0 / Critical`: low-friction RCE, credential exfiltration, privileged auth bypass, broad data exposure, durable controller/tool compromise.
- `P1 / High`: authenticated privilege escalation, scoped secret disclosure, SSRF to sensitive systems, tool/subprocess injection, likely malicious dependency execution.
- `P2 / Medium`: narrowed security invariant break, realistic DoS, unsafe agent/tool behavior behind specific conditions, missing validation on sensitive paths.
- `P3 / Low`: defense-in-depth issue with a credible but limited path, audit/logging weakness, hardening gap without immediate sensitive impact.

Do not assign P0-P2 without a concrete exploit path.

## Output

Lead with findings ordered by severity:

```markdown
- [P1] Imperative title under 80 chars
  - Location: path/to/file.py:123
  - Problem: Security invariant that is broken.
  - Exploit path: Attacker input -> trust boundary -> sink -> impact.
  - Suggested fix: Minimal safe change.
```

Then include:

- **Residual risk**: candidates that did not pass exploit gates, uncertainty, or follow-up checks.
- **Checks run**: commands, files, and code paths inspected.

If no findings pass the gates, state that explicitly and name the highest-risk surfaces inspected.

## Guardrails

- Do not expose secrets found during review; name only the path and remediation class.
- Do not run exploit payloads against live services unless the operator explicitly asks and the target is isolated.
- Do not mutate journals, runtime state, generated artifacts, or indices while reviewing security.
- Prefer narrow fixes: validation, capability checks, path guards, permission checks, dependency pinning, sandboxing, output encoding, or fail-closed feature gates.
