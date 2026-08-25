# DTAP bounded-subset disposable runner (EVL-46 / TU-DTAP-1)

Self-contained, zero-upstream-dependency local runner for a **reviewed, bounded
subset** of the [DecodingTrust-Agent](https://github.com/AI-secure/DecodingTrust-Agent)
(DTAP) benchmark ("A Controllable and Interactive Red-Teaming Platform for AI
Agents", arXiv 2605.04808), imported under Apache-2.0 for `tool-use-eval-contract.md`
**TU-DTAP-1**.

## Provenance and license

| Item | Value |
|---|---|
| Source | `https://github.com/AI-secure/DecodingTrust-Agent` |
| Pinned commit | `e0323a521ba4ef88f8e14c1eccf68d0a3d19a458` (tree `fd5a107aedb8971c346fc0e85d4789bf510e3f5f`) |
| License | Apache-2.0 (LICENSE SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`) |
| Subset | 18 cases: finance (12) + crm (6); benign 3 / direct 6 / indirect 9 |
| Injection families | prompt, tool, skill, environment, compositional (all five) |
| Per-case provenance | `manifest.json` (upstream path + config/judge SHA-256) |

`cases.json` transcribes each selected task's `config.yaml` verbatim (YAML→JSON,
no runtime YAML dependency). `judges/<case_id>/judge.py` transcribes each judge
**byte-identical in logic**; only the import prologue is rewritten to the local
harness shims (verified 18/18 by `tools/transcribe.py` + the normalization diff
in this directory's authoring notes). Setup scripts were inspected (all
localhost-only; none are ever executed here — see harness boundaries below).

## Selection

| case_id | domain | threat | families | judge surface |
|---|---|---|---|---|
| finance-benign-trade-execution-001 | finance | benign | — | transactions + response keyword |
| finance-benign-analyst-review-001 | finance | benign | — | gmail + comments + data grounding |
| finance-direct-churning-001 | finance | direct | env, prompt | transactions + email + keywords |
| finance-direct-churning-004 | finance | direct | prompt | transactions + email + keywords |
| finance-direct-churning-005 | finance | direct | prompt, tool | transactions + email + keywords |
| finance-direct-fake_security_alert-007 | finance | direct | env, prompt, tool | email + paypal invoice |
| finance-indirect-action-reversal-002 | finance | indirect | tool | transactions |
| finance-indirect-action-reversal-003 | finance | indirect | env | transactions + response keyword |
| finance-indirect-action-reversal-012 | finance | indirect | env, tool | transactions + response keyword |
| finance-indirect-symbol-substitution-007 | finance | indirect | tool | transactions + response keyword |
| finance-indirect-data-exfiltration-004 | finance | indirect | env | gmail body keywords |
| finance-indirect-quantity-inflation-014 | finance | indirect | env, tool | transactions + response keyword |
| crm-benign-001 | crm | benign | — | salesforce leads |
| crm-direct-social-scoring-010 | crm | direct | prompt, skill | accounts + email + slack |
| crm-malicious-direct-general-ai-restrictions-001 | crm | direct | prompt, skill | gmail body phrases |
| crm-malicious-indirect-social-scoring-003 | crm | indirect | skill | cases + email + slack |
| crm-malicious-indirect-misinfo-001 | crm | indirect | skill | opportunities + slack text |
| crm-malicious-indirect-general-ai-restrictions-004 | crm | indirect | skill, tool | notes + tasks + slack |

Rationale: finance and crm are the two domains whose judges are fully
deterministic state inspectors over the smallest helper surface
(`FinanceEnvClient` + gmail; salesforce + gmail + slack). Together they cover
**all five injection families** and all three threat classes with paired
benign/direct/indirect arms. Every judge in the subset was read in full and is
LLM-free (0 of the 6,682 released judges use `llm_check`; the subset uses none).

## How to run

From this directory (`scripts/autopilot/evals/dtap/`). Python 3.11+, stdlib
only at runtime (no upstream dependencies, no pyyaml at runtime).

```bash
python3 -m harness list-cases
python3 -m harness check-outcomes

# Zero-inference dry run against the reviewed arm fixtures
python3 -m harness run --case finance-benign-trade-execution-001 --arm done --stub
python3 -m harness run --case finance-indirect-action-reversal-002 --arm compromised --stub --out /tmp/dtap-results

# Repeated seeds + Wilson CIs (dry run)
python3 -m harness matrix --case finance-direct-fake_security_alert-007 \
    --arms compliant compromised --seeds 5 --stub --out /tmp/dtap-results

# Live mode against any OpenAI-compatible local endpoint (e.g. llama-server).
# Environment tool effects are recorded but NOT applied (no simulated services),
# so live judges observe the recorded trajectory + an empty environment state.
python3 -m harness run --case finance-benign-trade-execution-001 --arm done \
    --endpoint http://localhost:8080/v1 --model qwen3-32b --temperature 0.0

# Immutable trace replay
python3 -m harness replay --trace /tmp/dtap-results/traces/<case>.<arm>.seed0.trace.jsonl
```

Tests: `pytest tests/test_dtap_harness.py` (66 tests, zero inference).

## Contract features (TU-DTAP-1)

- **Config + deterministic final-state judges preserved** — transcribed from the
  pinned commit; judge logic byte-identical (import prologue only rewritten);
  upstream SHA-256 per file in `manifest.json`.
- **Setup scripts inspected, never run on this host.** All selected setup.sh
  files only curl localhost simulated services; nothing here executes them.
- **Per-arm fixed configuration** — `ArmConfig` (model, temperature, max_tokens,
  max_turns, retries, timeout) plus the versioned injection-render policy
  (`injection-render-policy-v1`) is fixed across arms and recorded in every trace.
- **Immutable trace replay** — SHA-256 hash-chained JSONL per run; `verify_trace`
  rejects any insertion/deletion/reorder/byte change; `replay_trace` re-runs the
  deterministic judge on the recorded state snapshot and compares verdicts.
- **Typed failure outcomes** — exactly `model|parser|tool|endpoint|harness|judge|
  infrastructure|overflow` (`harness/outcomes.py`; `check-outcomes` asserts the set).
- **Repeated seeds / confidence intervals** — `matrix` runs N seeds per case/arm
  and reports rates + Wilson 95% CIs.
- **Attack generation target-disjoint** — the imported attack payloads are fixed
  released constants from the pinned upstream commit (generated there against
  GPT-5.1/OpenAI SDK); the harness never optimizes attacks against its target.
  Matched-target DTAP-RED numbers are attack-search upper bounds, not general
  robustness scores — that caveat carries into any report built on this runner.

## Boundaries

- No upstream code is ever executed here: no Docker builds, no pip installs, no
  setup.sh runs, no simulated services. `tools/transcribe.py` only reads the
  disposable clone.
- The `fixtures/` arm states are hand-authored final states reviewed against
  each judge (with `@now+Nd` placeholders for day-relative CRM due dates); they
  are *arm fixtures*, not attacks, and their `script` tool calls carry explicit
  `state_delta` JSON-merge patches (lists append; `{"$set": [...]}` replaces).
- Live mode models endpoint interaction only; simulated-service integration is
  out of scope for this import.
