"""SC83: the negative-control false-accept rate is persisted (write side).

Hermetic, NO inference. Pins: stale / unbound / tampered verdicts are dropped
BEFORE scoring and listed; the endorsement is read from the signed body only; a
mixed reviewer configuration is refused; the denominator is accounted for; a run
with no scored decoy writes no belief row; the file is append-only and write-once
per run id; the CLI refuses with exit 2.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.proactive_delegation import false_accept_record as far
from src.proactive_delegation import review_envelope as rev

REPO_ROOT = Path(__file__).resolve().parent.parent
H = "sha256:" + "a" * 64


def annotation(aid: str, status: str, **over) -> dict:
    ann = {
        "schema_version": "1.0.0",
        "annotation_id": aid,
        "corpus_id": "secreview-gold-v1",
        "subject": {"ref": "repo@abc123", "content_hash": H, "file": "src/app.py",
                    "line_start": 10, "line_end": 12},
        "finding": {"title": "SQL built by string concatenation", "category": "OWASP-A03",
                    "cwes": ["CWE-89"]},
        "status": status,
        "origin": "human",
        "annotated_by": [{"annotator": "operator", "kind": "human"}],
        "gold": {"executable_oracle": None,
                 "reasoning_label": {"verdict": status, "rationale": "reviewed by hand"}},
        "needs_arbitration": False,
    }
    if status == "invalid":
        ann["invalid_reason"] = "constant input; no attacker-controlled data reaches the sink"
    else:
        ann["criticality"] = "must"
    ann.update(over)
    return ann


def binding(aid: str, **over) -> rev.ReviewBinding:
    b = rev.ReviewBinding(
        source_hash=rev.content_hash(f"source-{aid}"), source_version="v1",
        candidate_hash=rev.content_hash(f"finding-{aid}"), candidate_version="v1",
        reviewer_model="Qwen3.6-27B", reviewer_quant="Q4_K_M",
        prompt_bundle_hash=rev.content_hash("prompt-bundle-1"),
        pipeline_version="review-pipeline-2026.09.16", review_schema_version="1.0.0",
    )
    return replace(b, **over)


def verdict(aid: str, endorsed, *, bound: rev.ReviewBinding | None = None) -> dict:
    b = bound or binding(aid)
    review = {"verdict": "endorse" if endorsed else "reject"} if endorsed is not None else {}
    body = rev.sign_review(b, review)
    env = rev.build_envelope(envelope_id=f"rev-{aid}", binding=b, signed_body=body,
                             validation_status="valid", created_at="2026-09-16T10:00:00+00:00")
    return {"annotation_id": aid, "envelope": env, "signed_body": body}


def corpus() -> list[dict]:
    return [
        annotation("v0", "valid"),
        annotation("d0", "invalid"), annotation("d1", "invalid"), annotation("d2", "invalid"),
        annotation("d3", "invalid"),
        annotation("d-arb", "invalid", needs_arbitration=True),
    ]


def current_bindings(ids) -> dict:
    return {aid: binding(aid).as_schema() for aid in ids}


def write_inputs(tmp: Path, annotations, verdicts, bindings) -> dict[str, Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    paths = {"corpus": tmp / "corpus.jsonl", "verdicts": tmp / "verdicts.jsonl",
             "bindings": tmp / "bindings.json"}
    paths["corpus"].write_text("".join(json.dumps(a) + "\n" for a in annotations))
    paths["verdicts"].write_text("".join(json.dumps(v) + "\n" for v in verdicts))
    paths["bindings"].write_text(json.dumps(bindings))
    return paths


def standard_run(tmp: Path) -> dict[str, Path]:
    """d0 endorsed (false accept), d1 rejected, d2 STALE (inputs changed), d3 never reviewed."""
    ids = ["v0", "d0", "d1", "d2", "d3", "d-arb"]
    stale_old = binding("d2", source_version="v0", source_hash=rev.content_hash("old"))
    verdicts = [verdict("v0", True), verdict("d0", True), verdict("d1", False),
                verdict("d2", True, bound=stale_old), verdict("d-arb", True)]
    return write_inputs(tmp, corpus(), verdicts, current_bindings(ids))


def test_stale_verdict_is_dropped_before_scoring_and_listed(tmp_path):
    p = standard_run(tmp_path)
    line = far.build_run_line(corpus_path=p["corpus"], verdicts_path=p["verdicts"],
                              bindings_path=p["bindings"], run_id="r1", out=tmp_path / "o.jsonl",
                              scored_at="2026-09-16T11:00:00+00:00")
    res = line["result"]
    assert (res["numerator"], res["denominator"], res["n_decoys"]) == (1, 2, 5)
    assert res["unscored"] == ["d2", "d3"] and res["excluded_for_arbitration"] == ["d-arb"]
    assert list(line["stale"]) == ["d2"]
    assert any("inputs changed" in r for r in line["stale"]["d2"])
    [row] = line["belief_measurements"]
    assert row["value"] == 0.5 and row["reps"] == 2 and row["protocol_id"] == ""
    assert row["metric_direction"] == "lower_better" and row["category"] == "BASELINE"
    assert row["extra"]["stale"] == ["d2"] and row["extra"]["objections_projected"] is False
    assert row["extra"]["row_sha256"] == far.row_digest(row)
    assert row["extra"]["corpus_sha256"] == far.sha256_bytes(p["corpus"].read_bytes())
    assert row["extra"]["reviewer_config"]["reviewer_model"] == "Qwen3.6-27B"
    assert row["attestation_locator"].endswith("#run=r1")


def test_stale_arbitration_decoy_is_listed_apart_from_unscored(tmp_path):
    """A stale verdict on a needs_arbitration decoy must not land in ``stale``."""
    ids = ["v0", "d0", "d1", "d2", "d3", "d-arb"]
    old = binding("d-arb", source_version="v0", source_hash=rev.content_hash("old"))
    verdicts = [verdict("d0", True), verdict("d1", False), verdict("d-arb", True, bound=old)]
    p = write_inputs(tmp_path, corpus(), verdicts, current_bindings(ids))
    line = far.build_run_line(corpus_path=p["corpus"], verdicts_path=p["verdicts"],
                              bindings_path=p["bindings"], run_id="arb1", out=tmp_path / "o")
    res = line["result"]
    assert line["stale"] == {} and list(line["stale_excluded"]) == ["d-arb"]
    assert set(line["stale"]) <= set(res["unscored"])
    assert set(line["stale_excluded"]) <= set(res["excluded_for_arbitration"])
    [row] = line["belief_measurements"]
    assert row["extra"]["stale"] == [] and row["extra"]["stale_excluded"] == ["d-arb"]


def test_endorsement_is_read_from_the_signed_body_only(tmp_path):
    v = verdict("d0", None)
    v["endorsed"] = True                       # unsigned side field is ignored
    p = write_inputs(tmp_path, corpus(), [v], current_bindings(["d0"]))
    line = far.build_run_line(corpus_path=p["corpus"], verdicts_path=p["verdicts"],
                              bindings_path=p["bindings"], run_id="r2", out=tmp_path / "o")
    assert line["result"]["denominator"] == 0 and line["belief_measurements"] == []
    assert "decides neither" in line["stale"]["d0"][0]


def test_tampered_body_and_missing_binding_are_stale(tmp_path):
    tampered = verdict("d0", False)
    tampered["signed_body"]["review"]["verdict"] = "endorse"
    p = write_inputs(tmp_path, corpus(), [tampered, verdict("d1", True)],
                     current_bindings(["d0"]))
    core = far.score_run(far.read_jsonl(p["corpus"]), far.read_jsonl(p["verdicts"]),
                         json.loads(p["bindings"].read_text()))
    assert set(core["stale"]) == {"d0", "d1"}
    assert core["result"]["rate"] is None


def test_mixed_reviewer_configuration_is_refused():
    other = binding("d1", reviewer_quant="Q8_0")
    with pytest.raises(far.FalseAcceptRecordError, match="more than one reviewer"):
        far.score_run(corpus(), [verdict("d0", True), verdict("d1", False, bound=other)],
                      {"d0": binding("d0").as_schema(), "d1": other.as_schema()})


def test_inadmissible_corpus_and_duplicate_verdicts_are_refused():
    bad = corpus() + [annotation("x", "invalid", needs_arbitration="no")]
    with pytest.raises(far.FalseAcceptRecordError, match="inadmissible"):
        far.score_run(bad, [], {})
    with pytest.raises(far.FalseAcceptRecordError, match="two verdicts"):
        far.score_run(corpus(), [verdict("d0", True), verdict("d0", True)],
                      current_bindings(["d0"]))
    with pytest.raises(far.FalseAcceptRecordError, match="one corpus"):
        far.score_run(corpus() + [annotation("y", "valid", corpus_id="other")], [], {})


def test_append_is_write_once_per_run_id(tmp_path):
    p = standard_run(tmp_path)
    out = tmp_path / "runs.jsonl"
    kw = dict(corpus_path=p["corpus"], verdicts_path=p["verdicts"],
              bindings_path=p["bindings"], out=out)
    far.append_line(out, far.build_run_line(run_id="r1", **kw))
    far.append_line(out, far.build_run_line(run_id="r2", **kw))
    with pytest.raises(far.FalseAcceptRecordError, match="already recorded"):
        far.append_line(out, far.build_run_line(run_id="r1", **kw))
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [x["run_id"] for x in lines] == ["r1", "r2"]
    for x in lines:
        body = dict(x)
        digest = body.pop("line_sha256")
        assert digest == far.sha256_bytes(far._canon(body).encode())


def test_cli_writes_and_refuses(tmp_path, capsys):
    from scripts.review import score_false_accept as cli

    p = standard_run(tmp_path)
    out = tmp_path / "runs.jsonl"
    args = ["--corpus", str(p["corpus"]), "--verdicts", str(p["verdicts"]),
            "--current-bindings", str(p["bindings"]), "--out", str(out)]
    assert cli.main([*args, "--run-id", "r1", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert not out.exists()
    assert cli.main([*args, "--run-id", "r1"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["belief_rows"] == 1 and list(summary["stale"]) == ["d2"]
    assert cli.main([*args, "--run-id", "r1"]) == 2
    assert "already recorded" in capsys.readouterr().err
    assert cli.main([*args, "--run-id", "bad id"]) == 2
