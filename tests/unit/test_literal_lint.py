"""M-24 literal-constant lint: unit tests (advisory lint; no gate semantics)."""

from orchestration.repl_memory.distillation.literal_lint import extract_literals, lint_skill


def test_flags_each_category():
    t = (
        "Open /root/data/sales.csv then use `store_id` as the key column. "
        "Ticket ABC-1234, id deadbeef01, port 8083, route to frontdoor on qwen3.6-35b. "
        "Total 123456 and 0.12345."
    )
    lits = extract_literals(t)
    assert "/root/data/sales.csv" in lits["PATH"]
    assert "sales.csv" in lits["FILE"]
    assert "store_id" in lits["COL"]
    assert {"abc-1234", "deadbeef01"} <= set(lits["ID"])
    assert "8083" in lits["PORT"]
    assert {"frontdoor"} <= set(lits["ROLE_MODEL"])
    assert {"123456", "0.12345"} <= set(lits["NUM"])


def test_generic_text_not_flagged_and_years_system_paths_ignored():
    t = "Prefer simple strategies. See /usr/bin/env in 2026 when stuck."
    assert extract_literals(t) == {}


def test_grounded_requires_literal_in_source():
    skill = "---\nname: x\n---\nRead /app/in.csv and column `price`."
    res = lint_skill(skill, "the task gives /app/in.csv only")
    assert res.flagged and res.flagged_grounded
    assert "PATH" in res.grounded and "COL" not in res.grounded
    assert not lint_skill(skill, "unrelated").flagged_grounded
    assert lint_skill(skill).grounded == {}
