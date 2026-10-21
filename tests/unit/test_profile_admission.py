"""HS-PROF-1 admission schema + UTM-V7 write-audit checks (no model, no network)."""

from src.user_modeling.admission import AdmissionPolicy, check_admission
from src.user_modeling.profile_store import ProfileStore, UserFact
from src.user_modeling.write_audit import audit_facts, check_entry, read_store_facts


def _adm(text, cat="domain", **kw):
    return check_admission(text, cat, **kw)


class TestAdmission:
    def test_allowed_kinds_admitted(self):
        for kind in ("identity", "attribute", "relationship", "instruction"):
            assert _adm("Works as a systems engineer", "general", kind=kind).admit

    def test_unclassified_general_rejected(self):
        d = _adm("Likes things", "general")
        assert not d.admit and d.reason.startswith("kind_not_allowed")

    def test_bad_kind_rejected(self):
        assert not _adm("x is y", "domain", kind="behaviour").admit

    def test_category_maps_to_kind(self):
        assert _adm("Prefers box-drawing tables", "format").kind == "instruction"

    def test_behavioural_rejected(self):
        for t in ("Tends to ask for tables", "Seems to prefer terse answers", "Usually runs benchmarks twice"):
            assert _adm(t, "style").reason == "behavioural_or_hedged"

    def test_inferred_flag_rejected(self):
        assert _adm("Is a senior engineer", "domain", inferred=True).reason == "inferred_flag"

    def test_transient_rejected(self):
        assert _adm("Is currently debugging a kernel", "domain").reason == "unstable_transient_marker"

    def test_six_month_span(self):
        assert _adm("Senior engineer", "domain", evidence_span_days=30).reason == "unstable_short_evidence_span"
        assert _adm("Senior engineer", "domain", evidence_span_days=200).admit

    def test_cap(self):
        p = AdmissionPolicy(max_entries=2)
        assert _adm("Senior engineer", "domain", current_entries=1, policy=p).admit
        assert _adm("Senior engineer", "domain", current_entries=2, policy=p).reason == "entry_cap"


class TestStoreIntegration:
    def test_off_by_default_is_legacy(self, tmp_path):
        s = ProfileStore(tmp_path / "p.db")
        assert s.add_fact("u", UserFact("Tends to ask for tables", "general"))

    def test_policy_rejects_and_caps(self, tmp_path):
        s = ProfileStore(tmp_path / "p.db", admission=AdmissionPolicy(max_entries=2))
        assert s.add_fact("u", UserFact("Senior systems engineer", "domain"))
        assert not s.add_fact("u", UserFact("Tends to ask for tables", "format"))
        assert s.last_rejection == "behavioural_or_hedged"
        assert s.add_fact("u", UserFact("Wants tables box-drawn", "format"))
        assert not s.add_fact("u", UserFact("Wants TPS shown after runs", "workflow"))
        assert s.last_rejection == "entry_cap"
        assert s.get_profile("u").entry_count == 2


class TestWriteAudit:
    def test_clean(self):
        assert check_entry("Senior systems engineer with NUMA expertise") == []

    def test_each_check_fires(self):
        assert "one_fact_per_entry" in check_entry("Likes tables. Hates emoji.")
        assert "about_target_user" in check_entry("I prefer terse output")
        assert "no_conversational_mechanics" in check_entry("The user asked for a table")
        assert "references_dates_resolved" in check_entry("Started a new job last week")
        assert "references_dates_resolved" in check_entry("He leads the kernel team")

    def test_paraphrase_duplicate_flags_later_only(self):
        rep = audit_facts(["Prefers box drawing tables", "Prefers box drawing tables always", "Uses fish shell"])
        assert rep.fail_counts["no_paraphrase_duplicate"] == 1
        assert rep.flagged[0]["index"] == 1 and rep.n_clean == 2

    def test_read_store_facts_readonly(self, tmp_path):
        s = ProfileStore(tmp_path / "p.db")
        s.add_fact("u", UserFact("Uses fish shell", "workflow"))
        assert read_store_facts(tmp_path / "p.db") == [("u", "Uses fish shell")]
