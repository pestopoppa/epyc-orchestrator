# Operator hypothesis channel — `autopilot.py` integration patch

**Status:** APPLIED 2026-08-10. The former ownership note was stale: AutoPilot was stopped, the
file had advanced beyond the pinned hash, and the integration was still absent. The five seams
below are now wired. A separate read-only Vidya settled-ground block also grades recorded
resolution evidence and surfaces post-hoc invalidation without constraining hypothesis generation.

**Original patch anchors were verified against** `scripts/autopilot/autopilot.py` SHA-256
`fd2e2ce7d72e86b28bd27c0b6c11cf476dd9e00f0dc95e08b815a51b20ecaa1f` (9916 lines, 2026-08-03). Every
anchor below was checked to occur **exactly once**. The file is under active edit by another
session — if the hash no longer matches, re-locate by the quoted text rather than the line number;
the quoted text is the contract, the line numbers are a convenience.

**Module already landed (no `autopilot.py` change needed for these):**

| File | Role |
|---|---|
| `scripts/autopilot/operator_hypotheses.py` | the channel: load / validate / `still_open()` / resolution ledger / blacklist check / planner render |
| `orchestration/operator_hypotheses.yaml` | operator-authored statement store (hand-edited, never machine-written) |
| `orchestration/operator_hypothesis_resolutions.jsonl` | append-only resolution ledger (created on first `resolve`) |
| `tests/unit/test_autopilot_operator_hypotheses.py` | 25 tests, passing |

The module deliberately carries `build_planner_block()` and `record_resolution_from_rationale()`
so that each `autopilot.py` insertion below is one call, not a block of logic.

---

## What this changes, in one paragraph

AutoPilot's planner already carries hypotheses with a mandatory falsifier, keeps them still-open
until resolved, and re-surfaces the open set each planning round. The "Operator Outbox" runs
autopilot → operator. There is **no** inbound channel, so operator steering arrives out-of-band with
no falsifier and no resolution record. These five edits give the operator a way to state a
falsifiable claim that the planner *considers* and the loop can *refute*. It is a proposal source,
never an authority: an operator hypothesis enters with `seeded_by=operator` and
`evidence_trial_ids=[]` — the same provenance shape `seed_operator_strategies.py` already writes for
operator StrategyStore seeds — and it faces the critic, the failure blacklist and every gate
unchanged.

---

## Patch 1 of 5 — import the module

**Anchor** (`autopilot.py`, currently line 114, in the extracted-module import group):

```python
from blacklist_purge_plan import purge_scoped_target, retryable_reexploration_target
```

**Insert immediately AFTER that line:**

```python
from operator_hypotheses import (
    build_planner_block as _build_operator_hypotheses_block,
    record_resolution_from_rationale as _record_operator_hypothesis_resolution,
)
```

**Why here:** the group above it is the 2026-05-22 tranche-5 extracted-module import block, which is
where sibling `scripts/autopilot/` modules are imported (`controller_io`, `run_manifest`,
`planner_coordinator`, `state_store`, `blacklist_purge_plan`, `state_lock`, `actions`,
`paired_stats`). `sys.path` already contains `SCRIPT_DIR` at this point for exactly these imports.

---

## Patch 2 of 5 — a prompt section for the open operator set

**Anchor** (`autopilot.py`, currently lines 4809–4810, inside `CONTROLLER_PROMPT_TEMPLATE`):

```
### Hypotheses Under Test (last 3 trustworthy trials)
{hypotheses_under_test}
```

**Insert immediately AFTER those two lines** (keep the blank line that currently follows them):

```
### Operator Hypotheses (operator-stated priors, still open)
{operator_hypotheses_block}
```

**Why here and not next to "Still-open hypotheses":** the existing `### Still-open hypotheses` block
lives in `_EXPLORATION_RICH_TEMPLATE` (around line 5138), which is **stagnation-gated** — it only
renders when `_build_exploration_block` picks the rich fragment. Injecting the operator set *only*
there would make operator priors vanish exactly when the loop is healthy, which is most turns. The
always-rendered `CONTROLLER_PROMPT_TEMPLATE` slot above is the correct primary home; Patch 3 adds the
cross-reference into the rich fragment so the constrained-creativity protocol grades candidates
against operator priors too.

---

## Patch 3 of 5 — fold the operator set into the still-open list used by the creativity protocol

**Anchor** (`autopilot.py`, currently lines ~5565–5575, inside `_build_exploration_block`):

```python
    try:
        unfalsified = journal.unfalsified_hypotheses(n=5)
    except Exception:
        unfalsified = []
    if unfalsified:
        unfalsified_text = "\n".join(
            f"  #{tid}: {hyp[:160]}\n     falsifier: {fal[:160]}" for tid, hyp, fal in unfalsified
        )
    else:
        unfalsified_text = "  (no recent trials with explicit falsifiers yet)"
```

**Insert immediately AFTER that block** (before `block = _EXPLORATION_RICH_TEMPLATE.format(`):

```python
    # Operator-stated priors are still-open hypotheses too, and step 3's
    # "coherence" axis grades candidates against the still-open set. They are
    # PRIORS: rendered with their provenance, never as measured claims.
    try:
        from operator_hypotheses import still_open as _still_open_operator_hypotheses

        operator_open = _still_open_operator_hypotheses()
    except Exception:  # noqa: BLE001 — see Patch 5; the alarm path is the prompt block
        operator_open = []
    if operator_open:
        unfalsified_text += "\n" + "\n".join(
            f"  [operator prior: {item.id} | evidence_trial_ids=[]]: {item.hypothesis[:160]}\n"
            f"     falsifier: {item.falsifier[:160]}"
            for item in operator_open[:5]
        )
```

**Why:** `_EXPLORATION_RICH_TEMPLATE` step 3 scores candidates on *"consistency with … the still-open
hypotheses listed below"*. An operator prior that is invisible here gets no coherence credit and no
coherence penalty, so the creativity protocol would rank around it. The `[operator prior: … |
evidence_trial_ids=[]]` prefix is what keeps it from reading as a measured trial line.

*Note on the bare `except`:* this one is deliberate and narrow — the loud unreadable-store alarm is
delivered by Patch 4's block, which renders every turn. Duplicating the alarm here would double it in
the prompt. The channel is still fail-explicit; it just has one mouth.

---

## Patch 4 of 5 — populate the new placeholder

**Anchor** (`autopilot.py`, currently line ~7540, in the `CONTROLLER_PROMPT_TEMPLATE.format(...)`
call):

```python
                    blacklist_text=blacklist_text,
                    operator_outbox_feedback=_build_operator_outbox_feedback(),
```

**Insert immediately AFTER the `operator_outbox_feedback=` line:**

```python
                    operator_hypotheses_block=_build_operator_hypotheses_block(blacklist),
```

**Why `blacklist` is passed:** it flags an operator hypothesis whose `proposed_action` repeats a
recorded negative, inline in the prompt, with the wording *"Authorship is not new evidence."* The
local `blacklist` variable is in scope here (assigned at line 6866 and refreshed at 7667 / 8246 /
8379).

**`build_planner_block` never raises.** It has three states, not two:

| state | rendered |
|---|---|
| no open hypotheses | `  (none)` |
| open hypotheses | the list, with falsifiers, provenance and any negative-history flag |
| store unparseable | `!! OPERATOR HYPOTHESIS CHANNEL UNREADABLE: …` plus *"This does NOT mean the operator has no hypotheses"* |

The third state must never collapse into the second. The underlying `load_operator_hypotheses()`
raises `OperatorHypothesisError`; the wrapper converts that to an explicit in-prompt alarm plus
`log.error`, rather than aborting the trial or lying with `(none)`.

---

## Patch 5 of 5 — record resolutions from the trial that ran

### 5a — extend the rationale sidecar contract

**Anchor** (`autopilot.py`, currently lines ~4934–4939, in `CONTROLLER_PROMPT_TEMPLATE`):

````
```json:autopilot_rationale
{{"falsifier": "<one-line predicted outcome whose absence invalidates this hypothesis>",
 "rubric_scores": {{"info_gain": <1-5>, "coherence": <1-5>, "usefulness": <1-5>,
   "synthesis_note": "<optional one-line on fusion / cleaner model>"}}}}
```
````

**Replace the JSON body with** (note: doubled braces, this is an f-string-style `.format` template):

````
```json:autopilot_rationale
{{"falsifier": "<one-line predicted outcome whose absence invalidates this hypothesis>",
 "rubric_scores": {{"info_gain": <1-5>, "coherence": <1-5>, "usefulness": <1-5>,
   "synthesis_note": "<optional one-line on fusion / cleaner model>"}},
 "operator_hypothesis": {{"id": "<id from Operator Hypotheses, ONLY if this trial's
   outcome resolves it>", "status": "confirmed|refuted|inconclusive",
   "note": "<what the outcome showed>"}}}}
```
````

**And append one sentence to the paragraph directly above that fence**, which currently ends:

```
not abort the trial, but populating it lets future planner passes grade new
candidates against still-open hypotheses:
```

→ append after it, before the fence:

```
`operator_hypothesis` is OPTIONAL and omitted unless this trial's own outcome
resolves an operator prior. Do not supply trial ids — the trial that just ran is
the evidence, and it is attached for you. A refuted operator hypothesis is a
first-class result; record it.
```

### 5b — record it after the journal entry lands

**Anchor** (`autopilot.py`, currently line ~8926):

```python
        journal.record(journal_entry)
        if strategy_store is not None:
```

**Insert BETWEEN those two lines:**

```python
        # Operator hypothesis channel: the planner may name ONE operator prior this
        # trial resolved. Evidence is `trial_counter` — supplied here, never by the
        # planner — so nothing can be marked resolved without a trial behind it, and
        # an unknown/malformed claim is rejected, leaving the prior correctly open.
        _record_operator_hypothesis_resolution(rationale, trial_counter)
```

**Why here:** `rationale` and `trial_counter` are both in scope (`rationale.get("falsifier", "")` is
used in the `JournalEntry` construction immediately above), and placing it *after* `journal.record`
means a resolution can only ever cite a trial that is already durably journalled.

---

## Guards — what must NOT be added

These are the non-edits. They matter as much as the edits.

1. **No origin-based exemption anywhere.** An action derived from an operator hypothesis is an
   ordinary action: it passes `check_blacklist`, the pre-dispatch critic, `_critic_rejected_signature_skip`,
   the safety gate and the action-availability filter exactly as a planner-authored draft does. Do
   **not** add `if action.get("source") == "operator": ...` to any of them. The channel supplies a
   proposal, never authority.

2. **No exemption in `_build_action_availability` or the blacklist path.** If an operator hypothesis'
   `proposed_action` is blacklisted, the correct outcome is that the planner sees the
   `NEGATIVE HISTORY` line and does not propose it. Authorship is not new evidence.

3. **The block is prompt context, not an action source.** `_build_operator_hypotheses_block` returns
   a string. Nothing in it should ever be parsed back into an action, and it must not be added to
   `_first_unblacklisted_seed_action` / `_first_unblacklisted_numeric_trial_action` style fallback
   ladders — those are *substitutes* the loop picks when it must dispatch something, and an operator
   prior reaching them would be a queue jump.

4. **Resolution requires evidence, structurally.** `record_resolution()` refuses `confirmed` /
   `refuted` with zero evidence ids and refuses a second resolution for an already-resolved id;
   `record_resolution_from_rationale()` never lets the planner supply trial ids. Do not add a code
   path that writes the resolution ledger without going through these.

5. **Do not make the store fail-open.** `load_operator_hypotheses()` raising on a malformed store is
   the design, not a rough edge: an empty list would read as *"the operator has no hypotheses"*, and
   the planner would act on that. Only `build_planner_block()` may swallow the error, and only into a
   visible alarm.

---

## Verification after applying

```bash
cd /mnt/raid0/llm/epyc-orchestrator
.venv/bin/python -m pytest tests/unit/test_autopilot_operator_hypotheses.py -q
.venv/bin/python scripts/autopilot/operator_hypotheses.py validate
.venv/bin/python -c "import ast,sys; ast.parse(open('scripts/autopilot/autopilot.py').read())"
```

Then confirm the placeholder is wired — `CONTROLLER_PROMPT_TEMPLATE.format()` raises `KeyError` on a
placeholder with no keyword, so Patch 2 without Patch 4 fails loudly on the next planner turn rather
than silently. Apply them together.

---

## Operator-facing usage

```bash
# state one (falsifier is mandatory; a statement without one is refused at load)
$EDITOR orchestration/operator_hypotheses.yaml
uv run python scripts/autopilot/operator_hypotheses.py validate

# see what is still open
uv run python scripts/autopilot/operator_hypotheses.py list --open-only

# record an outcome — refutation is as recordable as confirmation
uv run python scripts/autopilot/operator_hypotheses.py resolve <id> \
    --status refuted --trial 1234 --note "what was run and what it showed"
```
