"""TRIPWIRE: `src/mutation_ledger.py` documents an integration that does not exist.

Third of the dead-machinery cluster (`sub_decision`, `binding_routing`, this), and the
worst of the three — because this one is not merely unreached, it is **self-describing
as reached**.

Its module docstring says, verbatim:

    the autopilot accept-path constructs MutationRecords and
    consults the ledger before composing a new mutation onto the live config.

Verified 2026-08-12 (`mainC`): `git grep` for `mutation_ledger` / `MutationLedger`
across `src/` and `scripts/`, excluding the module itself, returns **nothing**. No
production code imports it, so the accept-path consults no ledger and BSV-3's
conflict-aware acceptance is not in effect anywhere.

WHY THIS ONE IS WORSE THAN DEAD CODE. Dead code is inert. A docstring asserting its
own live integration is *actively misleading*: anyone reading the module — or grepping
for how conflict-aware acceptance works — concludes the feature is wired. The
surrounding audit found the same shape in a green test suite (`sub_decision`) and in a
feature flag (`binding_routing`); this is the same defect expressed in prose, which is
where a claim actually reaches a reader.

WHAT THIS IS. A tripwire on a known gap, not a behaviour test. It fails the moment
either half changes — the claim is softened, or the integration is built — so the two
can never silently drift apart again.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_MODULE = REPO / "src/mutation_ledger.py"

#: The exact clause under test. Kept as a literal so softening the docstring trips this
#: test rather than quietly resolving the contradiction in the wrong direction.
_CLAIM = "consults the ledger before composing a new mutation onto the live config"


def _production_references() -> list[str]:
    out = subprocess.run(
        ["git", "grep", "-n", "-E", r"mutation_ledger|MutationLedger", "--", "src", "scripts"],
        cwd=REPO, capture_output=True, text=True, check=False).stdout
    return [ln for ln in out.splitlines()
            if ln.strip() and not ln.startswith("src/mutation_ledger.py:")]


def test_the_docstring_still_claims_an_integration_that_does_not_exist() -> None:
    """FAILS when the claim and reality stop contradicting each other. Read the branches.

    * If the CLAIM is gone: someone softened the docstring. Good — confirm they did not
      also delete the module's purpose, and close the audit row.
    * If REFERENCES appeared: someone wired the accept-path. Also good — confirm the
      ledger is actually consulted on the accept path (an import is not a consultation),
      then close the row and delete this file.
    """
    src = _MODULE.read_text(encoding="utf-8")
    claim_present = _CLAIM in src
    refs = _production_references()

    assert claim_present and not refs, (
        f"The docstring claim and the code no longer contradict.\n"
        f"  claim still present : {claim_present}\n"
        f"  production refs     : {refs or 'none'}\n\n"
        "If the integration was BUILT: verify the accept path actually consults the "
        "ledger — an import is not a consultation — then close the BSV-3 row and delete "
        "this file. If the CLAIM was merely softened: make sure the row records that "
        "BSV-3 remains unimplemented, so a reader does not mistake a quieter docstring "
        "for a delivered feature."
    )


def test_the_module_still_exists_to_be_wired() -> None:
    """Guards the third outcome: silent deletion is also a resolution nobody recorded."""
    assert _MODULE.exists(), (
        "src/mutation_ledger.py was deleted. That may be the right call — BSV-3 was "
        "never wired — but it is a decision, and the audit row should say so rather "
        "than pointing at a file that no longer exists.")
