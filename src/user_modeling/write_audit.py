"""Profile-write precision audit (UTM-V7; rubric: Honcho deriver prompt, intake-2237#record).

Deterministic checks over stored ProfileStore facts / ``user_conclude`` writes:

  one_fact_per_entry          single sentence, no entry delimiter or newline
  about_target_user           no first-person speech or assistant-directed text
  no_conversational_mechanics no "the user asked/said", "in this conversation", ...
  no_paraphrase_duplicate     token-Jaccard >= 0.7 against an earlier entry
  references_dates_resolved   no relative dates, no dangling pronoun/deictic opener

Checks that cannot be decided by rule (is the fact true? about the *target* user
rather than a third party?) are left to a human-labelled sample: ``sample_for_labelling``.
Read-only; never writes to the store.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

CHECKS = (
    "one_fact_per_entry",
    "about_target_user",
    "no_conversational_mechanics",
    "no_paraphrase_duplicate",
    "references_dates_resolved",
)
DUP_JACCARD = 0.7

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_FIRST_PERSON_RE = re.compile(r"\b(I|I'm|I've|I'll|my|me|we|our)\b")
_ASSISTANT_RE = re.compile(r"\b(the assistant|you should|please|I will)\b", re.I)
_MECHANICS_RE = re.compile(
    r"\b(the user (?:asked|said|wrote|told|mentioned|replied|thanked)|asked (?:me|the assistant)|"
    r"in (?:this|the) (?:conversation|chat|session|thread)|as (?:mentioned|discussed)|"
    r"thank(?:s| you)|wanted me to|told me|earlier in)\b",
    re.I,
)
_REL_DATE_RE = re.compile(
    r"\b(yesterday|today|tonight|tomorrow|last (?:week|month|year|night)|next (?:week|month|year)|"
    r"this (?:morning|week|month|year)|\d+ (?:days?|weeks?|months?|years?) ago|recently|lately|soon)\b",
    re.I,
)
_DANGLING_OPENER_RE = re.compile(
    r"^(he|she|they|it|this|that|these|those|the above|the same|there)\b", re.I
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def check_entry(text: str, earlier: list[str] | None = None) -> list[str]:
    """Return the names of the checks this entry FAILS (empty = passes all)."""
    t = text.strip()
    fails: list[str] = []
    if "§" in t or "\n" in t or len(_SENT_SPLIT_RE.split(t)) > 1:
        fails.append("one_fact_per_entry")
    if _FIRST_PERSON_RE.search(t) or _ASSISTANT_RE.search(t):
        fails.append("about_target_user")
    if _MECHANICS_RE.search(t):
        fails.append("no_conversational_mechanics")
    tk = _tokens(t)
    for prev in earlier or ():
        pk = _tokens(prev)
        if tk and pk and len(tk & pk) / len(tk | pk) >= DUP_JACCARD:
            fails.append("no_paraphrase_duplicate")
            break
    if _REL_DATE_RE.search(t) or _DANGLING_OPENER_RE.search(t):
        fails.append("references_dates_resolved")
    return fails


@dataclass
class AuditReport:
    n_entries: int = 0
    n_clean: int = 0
    fail_counts: dict[str, int] = field(default_factory=lambda: {c: 0 for c in CHECKS})
    flagged: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def audit_facts(facts: list[str]) -> AuditReport:
    """Audit facts in order (oldest first, so later paraphrases are the ones flagged)."""
    rep = AuditReport(n_entries=len(facts))
    for i, f in enumerate(facts):
        fails = check_entry(f, facts[:i])
        if not fails:
            rep.n_clean += 1
            continue
        for c in fails:
            rep.fail_counts[c] += 1
        rep.flagged.append({"index": i, "fact": f, "fails": fails})
    return rep


def read_store_facts(db_path: str | Path, user_id: str | None = None) -> list[tuple[str, str]]:
    """Read (user_id, fact) rows oldest first through a read-only connection."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    try:
        q = "SELECT user_id, fact FROM user_facts"
        args: tuple = ()
        if user_id:
            q += " WHERE user_id = ?"
            args = (user_id,)
        return conn.execute(q + " ORDER BY created_at ASC, id ASC", args).fetchall()
    finally:
        conn.close()


def sample_for_labelling(facts: list[str], n: int = 25, seed: int = 0) -> list[dict]:
    """Seeded sample for the human-labelled half of the checklist."""
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(facts)), min(n, len(facts))))
    return [{"index": i, "fact": facts[i], "human_label": None} for i in idx]


def main(argv: list[str]) -> int:
    import os

    db = argv[1] if len(argv) > 1 else str(
        Path(os.environ.get("ORCHESTRATOR_DATA_DIR", os.path.expanduser("~/.epyc-orchestrator")))
        / "user_profiles.db"
    )
    if not Path(db).exists():
        print(json.dumps({"db": db, "error": "store_not_found"}))
        return 2
    rows = read_store_facts(db)
    out: dict = {"db": db, "users": {}}
    for uid in sorted({r[0] for r in rows}):
        facts = [f for u, f in rows if u == uid]
        out["users"][uid] = asdict(audit_facts(facts)) | {"label_sample": sample_for_labelling(facts)}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
