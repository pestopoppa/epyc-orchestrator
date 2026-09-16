"""RA-9 — dual-gold annotation schema, gold-sanity gate, negative-control axis.

Hermetic, pure, NO inference: every gate hook is a fake that records call order.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.proactive_delegation import gold_annotations as ga
from src.proactive_delegation import gold_sanity as gs

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR_PATH = REPO_ROOT / "orchestration" / "validate_ir.py"
H = "sha256:" + "a" * 64


# ── fakes ──────────────────────────────────────────────────────────────────


class FakeHarness:
    """Records every hook call so ordering can be asserted, not assumed."""

    def __init__(self, native_by_temp=None, judge_votes=(True, True, True), fail_stage=None):
        self.calls: list[tuple[str, object]] = []
        self.native_by_temp = native_by_temp or {}
        self.judge_votes = list(judge_votes)
        self.fail_stage = fail_stage
        self._current_temp = None

    def generate(self, t):
        self._current_temp = t
        self.calls.append(("generate", t))
        return f"def test_x():  # t={t}\n    assert True\n"

    def _stage(self, name, default=True):
        def hook(artifact):
            self.calls.append((name, self._current_temp))
            if self.fail_stage == name:
                raise RuntimeError(f"{name} crashed")
            if name == "run_native":
                return self.native_by_temp.get(self._current_temp, default)
            return True
        return hook

    def judge(self, artifact):
        self.calls.append(("judge", self._current_temp))
        return self.judge_votes.pop(0)

    def hooks(self):
        return gs.GateHooks(
            generate=self.generate,
            inject=self._stage("inject"),
            apply_gold=self._stage("apply_gold"),
            run_native=self._stage("run_native"),
            judge=self.judge,
            hash_artifact=lambda a: H,
        )

    def names(self):
        return [c[0] for c in self.calls]


# ── gate ordering: gate BEFORE judge ───────────────────────────────────────


def test_judge_never_consulted_when_native_fails_on_gold():
    """The source's judge endorsed all six invalid cases; it must never see an unrun test."""
    h = FakeHarness(native_by_temp={0.2: False, 0.6: False, 1.0: False})
    result = gs.run_gate(h.hooks())
    assert result.outcome == gs.DISCARDED_GOLD_FAILURE and not result.admitted
    assert "judge" not in h.names()
    assert [c[1] for c in h.calls if c[0] == "generate"] == [0.2, 0.6, 1.0]
    assert gs.validate_record(result.record) == []


def test_stage_order_is_inject_gold_native_then_judge():
    h = FakeHarness()
    result = gs.run_gate(h.hooks())
    assert result.admitted
    assert h.names() == ["generate", "inject", "apply_gold", "run_native", "judge", "judge", "judge"]
    assert gs.validate_record(result.record) == []


def test_gold_failure_retries_at_higher_temperature_then_admits():
    h = FakeHarness(native_by_temp={0.2: False})
    result = gs.run_gate(h.hooks())
    assert result.admitted
    assert [a["temperature"] for a in result.record["attempts"]] == [0.2, 0.6]
    judge_temps = {c[1] for c in h.calls if c[0] == "judge"}
    assert judge_temps == {0.6}
    assert gs.validate_record(result.record) == []


def test_judge_majority_rejection_discards_without_retry():
    h = FakeHarness(judge_votes=(True, False, False))
    result = gs.run_gate(h.hooks())
    assert result.outcome == gs.DISCARDED_JUDGE and result.artifact is None
    assert h.names().count("generate") == 1
    assert gs.validate_record(result.record) == []


@pytest.mark.parametrize("stage", ["inject", "apply_gold", "run_native"])
def test_crashed_stage_is_a_failure_never_a_pass(stage):
    h = FakeHarness(fail_stage=stage)
    result = gs.run_gate(h.hooks())
    assert result.outcome == gs.DISCARDED_GOLD_FAILURE
    assert "judge" not in h.names()
    assert gs.validate_record(result.record) == []


def test_errored_judge_sample_is_not_an_endorsement():
    h = FakeHarness()
    calls = iter([True, RuntimeError("boom"), False])

    def judge(_):
        v = next(calls)
        if isinstance(v, Exception):
            raise v
        return v

    hooks = gs.GateHooks(h.generate, h._stage("inject"), h._stage("apply_gold"), h._stage("run_native"), judge)
    assert gs.run_gate(hooks).outcome == gs.DISCARDED_JUDGE


# ── ablation preserved: retry presence mandatory, style not a parameter ────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperatures": (0.7,)},          # no retry
        {"temperatures": (0.6, 0.6)},      # retry without escalation
        {"temperatures": (1.0, 0.2)},      # de-escalation
        {"judge_samples": 1},
        {"judge_samples": 4},
    ],
)
def test_config_removing_a_load_bearing_property_is_refused(kwargs):
    with pytest.raises(gs.GoldSanityConfigError):
        gs.run_gate(FakeHarness().hooks(), **kwargs)


def test_generate_hook_takes_only_temperature():
    """Retry STYLE is not load-bearing (intake-983): there is no retry-prompt surface."""
    import inspect

    assert list(inspect.signature(gs.run_gate).parameters) == ["hooks", "temperatures", "judge_samples"]
    assert "prompt" not in " ".join(gs.GateHooks.__dataclass_fields__)


# ── validate_record refuses hand-assembled admissions ──────────────────────


def _admitted_record():
    return gs.run_gate(FakeHarness().hooks()).record


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda r: r["attempts"][0]["stages"].insert(0, r["attempts"][0]["stages"].pop(3)), "in-order prefix"),
        (lambda r: r["attempts"][0]["stages"].pop(2), "in-order prefix"),  # judge without native_run
        (lambda r: r["attempts"][0]["stages"][2].update(ok=False), "failed but later stages ran"),
        (lambda r: r["attempts"][0].update(judge_votes=[True]), "odd vote count"),
        (lambda r: r["attempts"][0].update(judge_votes=[False, False, True]), "disagrees with its votes"),
        (lambda r: r.update(outcome="discarded_judge"), "does not follow"),
        (lambda r: r["attempts"][0]["stages"].pop(3), "never consulted"),
        (lambda r: r.update(attempts=[]), "no attempts"),
    ],
)
def test_validate_record_catches_forged_admission(mutate, fragment):
    rec = _admitted_record()
    mutate(rec)
    assert any(fragment in e for e in gs.validate_record(rec)), gs.validate_record(rec)


def test_validate_record_catches_judge_on_non_final_attempt_and_non_escalation():
    rec = gs.run_gate(FakeHarness(native_by_temp={0.2: False}).hooks()).record
    bad = copy.deepcopy(rec)
    bad["attempts"][0] = copy.deepcopy(bad["attempts"][1]) | {"attempt": 1, "temperature": 0.2}
    assert any("non-final" in e for e in gs.validate_record(bad))
    bad2 = copy.deepcopy(rec)
    bad2["attempts"][1]["temperature"] = 0.1
    assert any("strictly increase" in e for e in gs.validate_record(bad2))


def test_single_attempt_gold_failure_discard_is_flagged():
    rec = {"procedure_version": "x", "outcome": gs.DISCARDED_GOLD_FAILURE,
           "attempts": [{"attempt": 1, "temperature": 0.2,
                         "stages": [{"stage": "inject", "ok": False}]}]}
    assert any("without any retry" in e for e in gs.validate_record(rec))


# ── dual-gold annotation schema ────────────────────────────────────────────


def _human(status="valid", **over):
    ann = {
        "schema_version": ga.GOLD_ANNOTATION_SCHEMA_VERSION,
        "annotation_id": f"ann-{status}",
        "corpus_id": "secreview-gold-v1",
        "subject": {"ref": "repo@abc123", "content_hash": H, "file": "src/app.py", "line_start": 10, "line_end": 12},
        "finding": {"title": "SQL built by string concatenation", "category": "OWASP-A03", "cwes": ["CWE-89"]},
        "status": status,
        "origin": "human",
        "annotated_by": [{"annotator": "operator", "kind": "human"}],
        "gold": {
            "executable_oracle": None,
            "reasoning_label": {"verdict": status, "rationale": "reviewed by hand"},
        },
        "needs_arbitration": False,
    }
    if status == "invalid":
        ann["invalid_reason"] = "input is a compile-time constant; no attacker-controlled data reaches the sink"
    else:
        ann["criticality"] = "must"
    ann.update(over)
    return ann


def _machine(status="invalid"):
    ann = _human(status)
    ann["annotation_id"] = f"ann-machine-{status}"
    ann["origin"] = "machine_generated"
    ann["annotated_by"] = [{"annotator": "annotator-bot", "kind": "model", "model_quant": "Qwen3.6-27B/Q4_K_M"}]
    ann["gold"]["executable_oracle"] = {
        "verdict": "pass", "instrument": "pytest", "instrument_version": "8.3.4",
        "witness_ref": "tests/test_witness.py::test_x", "witness_hash": H,
    }
    ann["gold_sanity"] = _admitted_record()
    return ann


def test_valid_and_decoy_human_annotations_are_admissible():
    assert ga.annotation_errors(_human("valid")) == []
    assert ga.annotation_errors(_human("invalid")) == []
    assert ga.expected_reviewer_action(_human("invalid")) == "reject"
    assert ga.corpus_gold_label(_human("valid")) == "accept"


def test_machine_annotation_with_admitted_gate_is_admissible():
    assert ga.annotation_errors(_machine()) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.pop("invalid_reason"),                                   # decoy without reason
        lambda a: a.update(criticality="must"),                              # tier on a decoy
        lambda a: a.update(status="bogus"),
        lambda a: a["gold"].update(executable_oracle=None, reasoning_label=None),  # no gold at all
        lambda a: a.update(annotated_by=[]),
        lambda a: a["finding"].update(cwes=["89"]),
        lambda a: a["subject"].update(content_hash="deadbeef"),
    ],
)
def test_schema_rejects_malformed_decoys(mutate):
    ann = _human("invalid")
    mutate(ann)
    assert ga.annotation_errors(ann)


def test_valid_finding_may_not_carry_invalid_reason():
    assert ga.annotation_errors(_human("valid", invalid_reason="x"))


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda a: a.pop("gold_sanity"), "gold_sanity"),
        (lambda a: a["gold"].update(executable_oracle=None), "executable_oracle"),
        (lambda a: a["gold_sanity"].update(outcome="discarded_judge"), "admitted"),
        (lambda a: a["gold_sanity"]["attempts"][0]["stages"].pop(3), "never consulted"),
    ],
)
def test_machine_annotation_without_earned_gate_is_refused(mutate, fragment):
    ann = _machine()
    mutate(ann)
    errs = ga.annotation_errors(ann)
    assert any(fragment in e for e in errs), errs


def test_model_annotator_cannot_pose_as_human_origin():
    ann = _human("invalid")
    ann["annotated_by"].append({"annotator": "bot", "kind": "model"})
    assert any("model annotator" in e for e in ga.annotation_errors(ann))


def test_conflicting_dual_gold_must_be_marked_for_arbitration():
    ann = _machine("invalid")
    ann["gold"]["executable_oracle"]["verdict"] = "fail"  # witness contradicts the decoy status
    assert any("needs_arbitration" in e for e in ga.annotation_errors(ann))
    ann["needs_arbitration"] = True
    assert ga.annotation_errors(ann) == []
    label_conflict = _human("invalid")
    label_conflict["gold"]["reasoning_label"]["verdict"] = "valid"
    assert ga.requires_arbitration(label_conflict)


def test_consensus_counts_must_be_coherent():
    ann = _human("valid", consensus={"n_annotators": 1, "n_agree": 2})
    assert any("n_agree" in e for e in ga.annotation_errors(ann))


# ── the negative-control axis ──────────────────────────────────────────────


def test_false_accept_rate_over_decoys_with_stated_denominator():
    decoys = [dict(_human("invalid"), annotation_id=f"d{i}") for i in range(4)]
    disputed = dict(_human("invalid"), annotation_id="d-arb", needs_arbitration=True)
    corpus = decoys + [disputed, _human("valid")]
    res = ga.false_accept_rate(corpus, {"d0": True, "d1": False, "d2": False, "d-arb": True, "ann-valid": True})
    assert (res.n_decoys, res.n_scored, res.n_endorsed) == (5, 3, 1)
    assert res.rate == pytest.approx(1 / 3)
    assert res.unscored == ("d3",) and res.excluded_for_arbitration == ("d-arb",)
    d = res.as_dict()
    assert d["direction"] == "lower_is_better" and d["denominator"] == 3


def test_false_accept_rate_is_none_without_scored_decoys():
    assert ga.false_accept_rate([_human("valid")], {}).rate is None


def test_endorsement_for_unknown_annotation_is_refused():
    with pytest.raises(ValueError, match="not in the corpus"):
        ga.false_accept_rate([_human("invalid")], {"ghost": True})


# ── validate_ir CLI wiring ─────────────────────────────────────────────────


def test_cli_validates_gold_annotation_kind():
    run = lambda doc: subprocess.run(  # noqa: E731
        [sys.executable, str(VALIDATOR_PATH), "gold_annotation", "-"],
        input=json.dumps(doc), capture_output=True, text=True,
    ).returncode
    assert run(_machine()) == 0
    bad = _human("invalid")
    del bad["invalid_reason"]
    assert run(bad) == 2
