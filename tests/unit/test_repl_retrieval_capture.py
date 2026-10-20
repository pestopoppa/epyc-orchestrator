"""VB-UFH12-RETR / VB-UFH12-EMBSPEED: the retrieval eval and speed block belief writers (offline).

No model, server, timer or process is touched: the clock and the host sampler are injected.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark"))
import repl_retrieval_capture as cap  # noqa: E402

T_START = "2026-10-20T10:00:00+00:00"
T_END = "2026-10-20T10:30:00+00:00"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def clock(*stamps: str):
    queue = list(stamps)

    def tick() -> str:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return tick


def dense_arm(arm_id: str = "bge-large", role: str = "CANDIDATE", **over) -> dict:
    arm = {
        "arm_id": arm_id,
        "model": "BAAI/bge-large-en-v1.5",
        "revision": "d4aa6901d3a41ba39bb536a7c3e5ea6c7a0dc6a8",
        "dim": 1024,
        "dtype_or_quant": "f16",
        "runtime": "llama.cpp/b10303-ffc1bac82",
        "prompt_hash": sha("bge-prompt"),
        "index_identity_digest": sha(f"index-{arm_id}"),
        "role": role,
    }
    arm.update(over)
    return arm


def lexical_arm(arm_id: str = "bm25") -> dict:
    return {
        "arm_id": arm_id,
        "model": None,
        "revision": None,
        "dim": None,
        "dtype_or_quant": None,
        "runtime": "context_search.py@fd115351/rank_bm25",
        "prompt_hash": None,
        "index_identity_digest": sha(f"index-{arm_id}"),
        "role": "BASELINE",
    }


def host(**over) -> dict:
    ctx = {
        "hostname": "epyc-test",
        "cpu_list": "24-71",
        "cpus_allowed_observed": "24-71",
        "region_lock_claim": "q1+q2:build:sess-1",
        "loadavg": {"start": [1.0, 1.2, 1.5], "end": [1.1, 1.2, 1.5]},
        "frequency": {
            "governor": "performance",
            "mhz_min": 3500.0,
            "mhz_max": 4200.0,
            "throttled": False,
        },
    }
    ctx.update(over)
    return ctx


QUERIES = {"all": sha("q-all"), "probe45": sha("q-probe45"), "spill_follow": sha("q-spill")}
CORPUS = {"corpus": sha("corpus")}
LABELS = {"gold": sha("gold")}
MANIFEST = sha("arm-manifest")
CONFIG = {"chunk_chars": 800, "rrf_k": 60}


def retrieval_run(tmp_path, *, run_id="run-1", ctx=None, rule=None, **kw) -> cap.RetrievalRun:
    return cap.RetrievalRun(
        tmp_path,
        run_id,
        rule if rule is not None else cap.registered_rule(),
        CORPUS,
        QUERIES,
        LABELS,
        MANIFEST,
        ctx if ctx is not None else host(),
        clock=clock(T_START, T_END),
        **kw,
    )


def fill(run: cap.RetrievalRun, n: int = 120) -> None:
    """lexical L, dense and hybrid BGE on the pooled set at k=5, plus the paired diff."""
    run.add_recall(lexical_arm(), "all", "lexical", 5, "recall", 60 / n, n, 60, [0.3, 0.7], CONFIG)
    run.add_recall(dense_arm(), "all", "dense", 5, "recall", 66 / n, n, 66, [0.35, 0.75], CONFIG)
    run.add_recall(dense_arm(), "all", "hybrid", 5, "recall", 84 / n, n, 84, None, CONFIG)
    run.add_paired_diff(
        dense_arm(),
        lexical_arm(),
        "all",
        5,
        24 / n,
        [0.05, 0.35],
        n,
        10000,
        7,
        mode_a="hybrid",
        mode_b="lexical",
    )
    run.add_cost(dense_arm(), 0.42, 55.0, 80.0, 310.0, n, mode="hybrid")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# ------------------------------------------------------------------ the retrieval run


def test_good_retrieval_run_round_trips(tmp_path):
    run = retrieval_run(tmp_path)
    fill(run)
    sidecar = run.finalize()
    assert sidecar.name == "run-1.belief_measurements.jsonl"
    assert (tmp_path / "run-1.record.json").is_file()
    assert cap.sidecar_path(tmp_path / "run-1.record.json") == sidecar
    rows = read_rows(sidecar)
    assert len(rows) == 3 + 1 + 4  # recall x3, paired diff, cost x4
    assert cap.validate_rows(rows) == []
    assert {r["instrument"] for r in rows} == {"offline_recall"}
    assert {r["source_kind"] for r in rows} == {"repl-retrieval-eval"}
    assert all(r["protocol_id"] == "" for r in rows)  # no codified protocol -> an observation
    record_bytes = (tmp_path / "run-1.record.json").read_bytes()
    assert {r["record_sha256"] for r in rows} == {hashlib.sha256(record_bytes).hexdigest()}
    hybrid = next(r for r in rows if r["kind"] == "recall" and r["mode"] == "hybrid")
    assert hybrid["metric"] == "recall_at_5" and hybrid["value"] == pytest.approx(0.7)
    assert hybrid["category"] == "CANDIDATE" and hybrid["query_category"] == "all"
    assert hybrid["extra"]["provenance"]["rule"] == cap.REGISTERED_RULE
    assert "REPL-EMB-2.1" in hybrid["claim"] and "spill_pointer_follows" in hybrid["claim"]
    lex = next(r for r in rows if r["kind"] == "recall" and r["mode"] == "lexical")
    assert lex["category"] == "BASELINE" and lex["extra"]["arm"]["model"] is None
    diff = next(r for r in rows if r["kind"] == "paired_diff")
    assert diff["metric"] == "recall_at_5_paired_diff" and diff["samples"]["resamples"] == 10000
    assert diff["samples"]["against_arm_id"] == "bm25"
    # locator = run x arm x mode x category x k x metric: no two claims share one
    assert len({r["extra"]["locator"] for r in rows}) == len(rows)
    # the rows are a pure projection of the record
    record = json.loads(record_bytes)
    again = cap.build_rows(
        record,
        record_file="run-1.record.json",
        record_sha256=rows[0]["record_sha256"],
        emitted_at=rows[0]["emitted_at"],
    )
    assert again == rows


def test_finalize_is_all_or_nothing_and_immutable(tmp_path):
    run = retrieval_run(tmp_path)
    fill(run)
    run.finalize()
    with pytest.raises(cap.CaptureError, match="already finalized"):
        run.finalize()
    other = retrieval_run(tmp_path)
    fill(other)
    with pytest.raises(cap.CaptureError, match="immutable"):
        other.finalize()
    empty = retrieval_run(tmp_path, run_id="run-2")
    with pytest.raises(cap.CaptureError, match="at least one recall"):
        empty.finalize()
    assert not (tmp_path / "run-2.record.json").exists()


# ------------------------------------------------------------------ the pre-registered rule


@pytest.mark.parametrize(
    "key,value", [("k", 3), ("M", 0.05), ("X", 0.1), ("n_min", 45), ("ci_level", 0.9)]
)
def test_a_rule_with_different_values_is_refused(tmp_path, key, value):
    rule = cap.registered_rule()
    rule[key] = value
    with pytest.raises(cap.CaptureError, match="no dated amendment"):
        retrieval_run(tmp_path, rule=rule)


def test_rule_shape_is_strict(tmp_path):
    rule = cap.registered_rule()
    del rule["n_min"]
    with pytest.raises(cap.CaptureError, match="lacks"):
        retrieval_run(tmp_path, rule=rule)
    rule = {**cap.registered_rule(), "verdict": "pass"}
    with pytest.raises(cap.CaptureError, match="unknown keys"):
        retrieval_run(tmp_path, rule=rule)


def test_a_dated_amendment_is_accepted_and_flagged(tmp_path):
    rule = {**cap.registered_rule(), "n_min": 400, "M": 0.05}
    rule["amendment"] = {"approved_on": "2026-10-21", "approval_ref": "operator chat 2026-10-21"}
    run = retrieval_run(tmp_path, rule=rule)
    fill(run, n=400)
    rows = read_rows(run.finalize())
    assert all("AMENDED" in r["claim"] for r in rows if r["kind"] in ("recall", "paired_diff"))
    assert rows[0]["extra"]["provenance"]["rule"]["amendment"]["approved_on"] == "2026-10-21"


def test_an_amendment_must_be_dated_and_must_differ(tmp_path):
    rule = {
        **cap.registered_rule(),
        "amendment": {"approved_on": "2026-10-21", "approval_ref": "x"},
    }
    with pytest.raises(cap.CaptureError, match="every value is the registered one"):
        retrieval_run(tmp_path, rule=rule)
    rule = {
        **cap.registered_rule(),
        "M": 0.2,
        "amendment": {"approved_on": "soon", "approval_ref": "x"},
    }
    with pytest.raises(cap.CaptureError, match="approved_on"):
        retrieval_run(tmp_path, rule=rule)
    rule = {**cap.registered_rule(), "M": 0.2, "amendment": None}
    with pytest.raises(cap.CaptureError, match="amendment is null"):
        retrieval_run(tmp_path, rule=rule)


def test_a_row_that_carries_a_changed_rule_is_refused_by_the_shared_validator(tmp_path):
    run = retrieval_run(tmp_path)
    fill(run)
    rows = read_rows(run.finalize())
    row = copy.deepcopy(rows[0])
    row["extra"]["provenance"]["rule"]["M"] = 0.01
    row["extra"]["provenance_sha256"] = cap.content_hash(row["extra"]["provenance"])
    row["row_sha256"] = cap.row_digest(row)
    problems = cap.validate_row(row)
    assert any("rule differs from the REPL-EMB-2.1 registration" in p for p in problems)


# ------------------------------------------------------------------ instruments never pool


def test_a_foreign_or_online_instrument_is_refused(tmp_path):
    run = retrieval_run(tmp_path)
    fill(run)
    rows = read_rows(run.finalize())
    row = copy.deepcopy(rows[0])
    row["instrument"] = "online_spill_pointer_follows"
    row["row_sha256"] = cap.row_digest(row)
    problems = cap.validate_row(row)
    assert problems and "never pool" in problems[0]
    row = copy.deepcopy(rows[0])
    row["instrument"] = "offline_speed"  # a recall row labelled as the speed instrument
    row["row_sha256"] = cap.row_digest(row)
    assert any("do not pair" in p for p in cap.validate_row(row))
    record = json.loads((tmp_path / "run-1.record.json").read_text())
    record["instrument"] = "online_spill_pointer_follows"
    with pytest.raises(cap.CaptureError, match="instrument"):
        cap.build_rows(
            record, record_file="x.record.json", record_sha256="0" * 64, emitted_at=T_END
        )


def speed_run(tmp_path, *, run_id="run-1", ctx=None) -> cap.SpeedRun:
    return cap.SpeedRun(
        tmp_path, run_id, CORPUS, {"q200": sha("q200")}, MANIFEST,
        ctx if ctx is not None else host(), clock=clock(T_START, T_END),
    )  # fmt: skip


SPEED_CFG = {"threads": 4, "batch": 32, "leg": "in_process"}


def fill_speed(run: cap.SpeedRun, *, floors: bool = True) -> None:
    arm = dense_arm("bge-st-fp32", role="BASELINE", dtype_or_quant="fp32")
    run.add_speed(arm, "chunks_per_s", 12.5, "chunks/s", 3, 0.4, "chunks/s", SPEED_CFG)
    run.add_speed(arm, "query_p50_ms", 65.0, "ms", 600, 3.0, "ms", SPEED_CFG)
    run.add_speed(arm, "query_p95_ms", 84.0, "ms", 600, 5.0, "ms", SPEED_CFG)
    run.add_speed(arm, "rss_mb", 2100.0, "MB", 1, 0.0, "MB", SPEED_CFG)
    if floors:
        for metric, aa, unit in (
            ("chunks_per_s", 0.3, "chunks/s"),
            ("query_p50_ms", 2.0, "ms"),
            ("query_p95_ms", 4.0, "ms"),
            ("rss_mb", 5.0, "MB"),
        ):
            run.add_noise_floor(arm, metric, aa, unit, replication_unit="arm", n=3)


def test_good_speed_run_round_trips(tmp_path):
    run = speed_run(tmp_path)
    fill_speed(run)
    sidecar = run.finalize()
    assert sidecar.name == "run-1.speed.belief_measurements.jsonl"
    assert (tmp_path / "run-1.speed.record.json").is_file()
    rows = read_rows(sidecar)
    assert cap.validate_rows(rows) == []
    assert {r["instrument"] for r in rows} == {"offline_speed"}
    assert {r["source_kind"] for r in rows} == {"repl-embedder-speed"}
    chunks = next(r for r in rows if r["metric"] == "chunks_per_s")
    assert chunks["unit"] == "chunks/s" and chunks["metric_direction"] == "higher_better"
    assert chunks["samples"]["noise_floor"] == {
        "aa_spread": 0.3, "unit": "chunks/s", "replication_unit": "arm", "n": 2 + 1, "scope": "arm_metric",
    }  # fmt: skip
    ctx = chunks["extra"]["provenance"]["host_context"]
    assert ctx["cpu_list"] == "24-71" and ctx["region_lock_claim"] and ctx["loadavg"]["start"]
    assert ctx["frequency"]["throttled"] is False
    assert (
        "A/A noise floor 0.3 chunks/s" in chunks["claim"] and "region-lock claim" in chunks["claim"]
    )


# ------------------------------------------------------------------ refusals: noise floor and context


def test_a_speed_run_without_a_noise_floor_cannot_finalize(tmp_path):
    run = speed_run(tmp_path)
    fill_speed(run, floors=False)
    with pytest.raises(cap.CaptureError, match="lacks an A/A noise floor"):
        run.finalize()
    assert not list(tmp_path.glob("*.speed.*"))


def test_a_noise_floor_for_one_metric_does_not_cover_another(tmp_path):
    run = speed_run(tmp_path)
    arm = dense_arm("bge-st-fp32", role="BASELINE")
    run.add_speed(arm, "chunks_per_s", 12.5, "chunks/s", 3, 0.4, "chunks/s", SPEED_CFG)
    run.add_speed(arm, "tokens_per_s", 9000.0, "tokens/s", 3, 80.0, "tokens/s", SPEED_CFG)
    run.add_noise_floor(arm, "chunks_per_s", 0.3, "chunks/s", replication_unit="session", n=4)
    with pytest.raises(cap.CaptureError, match="tokens_per_s"):
        run.finalize()


def test_a_leg_bound_floor_binds_one_leg_only(tmp_path):
    run = speed_run(tmp_path)
    arm = dense_arm("bge-st-fp32", role="BASELINE")
    other = {**SPEED_CFG, "threads": 1}
    run.add_speed(arm, "chunks_per_s", 12.5, "chunks/s", 3, 0.4, "chunks/s", SPEED_CFG)
    run.add_speed(arm, "chunks_per_s", 4.0, "chunks/s", 3, 0.1, "chunks/s", other)
    run.add_noise_floor(
        arm, "chunks_per_s", 0.3, "chunks/s", replication_unit="arm", n=3, config_echo=SPEED_CFG
    )
    with pytest.raises(cap.CaptureError, match="lacks an A/A noise floor"):
        run.finalize()
    run.add_noise_floor(
        arm, "chunks_per_s", 0.1, "chunks/s", replication_unit="arm", n=3, config_echo=other
    )
    rows = read_rows(run.finalize())
    assert {r["samples"]["noise_floor"]["scope"] for r in rows} == {"leg"}
    assert len({r["extra"]["locator"] for r in rows}) == 2


def test_noise_floor_refusals(tmp_path):
    run = speed_run(tmp_path)
    arm = dense_arm("bge-st-fp32", role="BASELINE")
    with pytest.raises(cap.CaptureError, match="own unit|carried in"):
        run.add_noise_floor(arm, "chunks_per_s", 0.3, "texts/s", replication_unit="arm", n=3)
    with pytest.raises(cap.CaptureError, match="replication_unit"):
        run.add_noise_floor(arm, "chunks_per_s", 0.3, "chunks/s", replication_unit="run", n=3)
    with pytest.raises(cap.CaptureError, match="noise-floor n"):
        run.add_noise_floor(arm, "chunks_per_s", 0.3, "chunks/s", replication_unit="arm", n=1)
    with pytest.raises(cap.CaptureError, match="aa_spread"):
        run.add_noise_floor(
            arm, "chunks_per_s", float("nan"), "chunks/s", replication_unit="arm", n=3
        )
    with pytest.raises(cap.CaptureError, match="metric"):
        run.add_noise_floor(arm, "bogus", 0.3, "chunks/s", replication_unit="arm", n=3)
    run.add_noise_floor(arm, "chunks_per_s", 0.3, "chunks/s", replication_unit="arm", n=3)
    with pytest.raises(cap.CaptureError, match="duplicate"):
        run.add_noise_floor(arm, "chunks_per_s", 0.3, "chunks/s", replication_unit="arm", n=3)
    run.add_speed(arm, "rss_mb", 2100.0, "MB", 1, 0.0, "MB", SPEED_CFG)
    run.add_noise_floor(arm, "rss_mb", 5.0, "MB", replication_unit="arm", n=3)
    with pytest.raises(cap.CaptureError, match="match no speed row"):
        run.finalize()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: c.pop("region_lock_claim"), "region_lock_claim"),
        (lambda c: c.pop("loadavg"), "loadavg"),
        (lambda c: c.pop("frequency"), "frequency"),
        (lambda c: c.pop("cpus_allowed_observed"), "cpus_allowed_observed"),
        (lambda c: c.update(cpu_list="bogus"), "cpu_list"),
        (lambda c: c.update(cpus_allowed_observed="0-95"), "not inside the declared cpu_list"),
        (lambda c: c.update(loadavg={"start": [1, 2, 3]}), "loadavg"),
        (lambda c: c["frequency"].update(throttled="no"), "throttled"),
        (lambda c: c["frequency"].update(mhz_min=5000.0), "mhz_min"),
        (lambda c: c.update(region_lock_claim="  "), "region_lock_claim"),
    ],
)
def test_a_speed_run_needs_its_whole_timing_context(tmp_path, mutate, match):
    ctx = host()
    mutate(ctx)
    with pytest.raises(cap.CaptureError, match=match):
        speed_run(tmp_path, ctx=ctx)


def test_cost_rows_need_the_timing_context_but_recall_does_not(tmp_path):
    run = retrieval_run(tmp_path, ctx={"hostname": "epyc-test", "cpu_list": "0-95"})
    run.add_recall(lexical_arm(), "all", "lexical", 5, "recall", 0.5, 120, 60, None, CONFIG)
    with pytest.raises(cap.CaptureError, match="timing context"):
        run.add_cost(lexical_arm(), 0.1, 5.0, 9.0, 1.0, 120, mode="lexical")
    with pytest.raises(cap.CaptureError, match="hostname|cpu_list"):
        retrieval_run(tmp_path, ctx={"hostname": "epyc-test"})


# ------------------------------------------------------------------ refusals: values


def test_non_finite_values_are_refused(tmp_path):
    run = retrieval_run(tmp_path)
    for bad in (float("nan"), float("inf"), "0.5", None):
        with pytest.raises(cap.CaptureError, match="re-derive"):
            run.add_recall(lexical_arm(), "all", "lexical", 5, "recall", bad, 120, 60, None, CONFIG)
    with pytest.raises(cap.CaptureError, match="finite"):
        run.add_paired_diff(
            dense_arm(), lexical_arm(), "all", 5, float("nan"), [0.0, 0.1], 120, 1000, 1,
            mode_a="dense", mode_b="lexical",
        )  # fmt: skip
    with pytest.raises(cap.CaptureError, match="finite"):
        run.add_cost(dense_arm(), float("inf"), 5.0, 9.0, 1.0, 120, mode="dense")
    srun = speed_run(tmp_path, run_id="s")
    arm = dense_arm("bge-st-fp32", role="BASELINE")
    for bad in (float("nan"), float("inf"), -1.0, 0.0):
        with pytest.raises(cap.CaptureError, match="finite"):
            srun.add_speed(arm, "chunks_per_s", bad, "chunks/s", 3, 0.1, "chunks/s", SPEED_CFG)
    with pytest.raises(cap.CaptureError, match="finite"):
        srun.add_speed(arm, "chunks_per_s", 5.0, "chunks/s", 3, float("nan"), "chunks/s", SPEED_CFG)


def test_a_non_finite_value_in_a_row_is_refused_by_the_validator(tmp_path):
    run = retrieval_run(tmp_path)
    fill(run)
    row = read_rows(run.finalize())[0]
    row["value"] = float("nan")
    assert cap.validate_row(row) == ["value is not a finite number"]


def test_n_below_one_is_refused(tmp_path):
    run = retrieval_run(tmp_path)
    for n in (0, -3, 1.5, True, None):
        with pytest.raises(cap.CaptureError, match="n must be an int >= 1"):
            run.add_recall(lexical_arm(), "all", "lexical", 5, "recall", 0.5, n, 0, None, CONFIG)
    with pytest.raises(cap.CaptureError, match="n must be an int >= 1"):
        run.add_paired_diff(
            dense_arm(), lexical_arm(), "probe45", 5, 0.1, [0.0, 0.2], 0, 1000, 1,
            mode_a="dense", mode_b="lexical",
        )  # fmt: skip
    srun = speed_run(tmp_path, run_id="s")
    arm = dense_arm("bge-st-fp32", role="BASELINE")
    with pytest.raises(cap.CaptureError, match="n must be an int >= 1"):
        srun.add_speed(arm, "chunks_per_s", 5.0, "chunks/s", 0, 0.1, "chunks/s", SPEED_CFG)
    with pytest.raises(cap.CaptureError, match="n_queries"):
        run.add_cost(dense_arm(), 0.1, 5.0, 9.0, 1.0, 0, mode="dense")


def test_ci_lower_above_upper_is_refused(tmp_path):
    run = retrieval_run(tmp_path)
    with pytest.raises(cap.CaptureError, match="lower bound"):
        run.add_recall(
            lexical_arm(), "all", "lexical", 5, "recall", 0.5, 120, 60, [0.7, 0.3], CONFIG
        )
    with pytest.raises(cap.CaptureError, match="lower bound"):
        run.add_paired_diff(
            dense_arm(), lexical_arm(), "all", 5, 0.1, [0.3, 0.1], 120, 1000, 1,
            mode_a="dense", mode_b="lexical",
        )  # fmt: skip
    with pytest.raises(cap.CaptureError, match="two finite"):
        run.add_recall(
            lexical_arm(), "all", "lexical", 5, "recall", 0.5, 120, 60, [float("nan"), 0.7], CONFIG
        )
    with pytest.raises(cap.CaptureError, match="paired diff needs ci95"):
        run.add_paired_diff(
            dense_arm(), lexical_arm(), "all", 5, 0.1, None, 120, 1000, 1,
            mode_a="dense", mode_b="lexical",
        )  # fmt: skip


def test_a_tampered_ci_is_refused_by_the_validator(tmp_path):
    run = retrieval_run(tmp_path)
    fill(run)
    rows = read_rows(run.finalize())
    row = copy.deepcopy(next(r for r in rows if r["kind"] == "paired_diff"))
    row["samples"]["ci95"] = [0.4, 0.1]
    row["row_sha256"] = cap.row_digest(row)
    assert any("ci95" in p for p in cap.validate_row(row))


def test_recall_must_re_derive_from_hits_and_n(tmp_path):
    run = retrieval_run(tmp_path)
    with pytest.raises(cap.CaptureError, match="re-derive"):
        run.add_recall(lexical_arm(), "all", "lexical", 5, "recall", 0.9, 120, 60, None, CONFIG)
    with pytest.raises(cap.CaptureError, match="hits"):
        run.add_recall(lexical_arm(), "all", "lexical", 5, "recall", 1.0, 120, 121, None, CONFIG)
    with pytest.raises(cap.CaptureError, match="metric must be 'recall'"):
        run.add_recall(
            lexical_arm(), "all", "lexical", 5, "recall_at_5", 0.5, 120, 60, None, CONFIG
        )
    with pytest.raises(cap.CaptureError, match="k must be one of"):
        run.add_recall(lexical_arm(), "all", "lexical", 7, "recall", 0.5, 120, 60, None, CONFIG)
    with pytest.raises(cap.CaptureError, match="mode must be one of"):
        run.add_recall(lexical_arm(), "all", "colbert", 5, "recall", 0.5, 120, 60, None, CONFIG)
    with pytest.raises(cap.CaptureError, match="config_echo"):
        run.add_recall(lexical_arm(), "all", "lexical", 5, "recall", 0.5, 120, 60, None, {})


# ------------------------------------------------------------------ refusals: the claim tuple


def test_a_row_missing_a_claim_tuple_element_is_refused(tmp_path):
    run = retrieval_run(tmp_path)
    fill(run)
    rows = read_rows(run.finalize())
    for field in cap.CLAIM_TUPLE_FIELDS:
        row = copy.deepcopy(rows[0])
        del row[field]
        problems = cap.validate_row(row)
        assert problems == [f"claim tuple incomplete: ['{field}']"], field
    row = copy.deepcopy(rows[0])
    row["reps"] = 0
    row["row_sha256"] = cap.row_digest(row)
    assert any("reps must be a positive int" in p for p in cap.validate_row(row))
    row = copy.deepcopy(rows[0])
    row["protocol_id"] = "P-BENCH-1"
    row["row_sha256"] = cap.row_digest(row)
    assert any("protocol_id must be empty" in p for p in cap.validate_row(row))


def test_a_speed_row_missing_its_claim_tuple_or_floor_is_refused_by_the_validator(tmp_path):
    run = speed_run(tmp_path)
    fill_speed(run)
    rows = read_rows(run.finalize())
    row = copy.deepcopy(rows[0])
    del row["metric_direction"]
    assert cap.validate_row(row) == ["claim tuple incomplete: ['metric_direction']"]
    row = copy.deepcopy(rows[0])
    row["samples"]["noise_floor"] = None
    row["row_sha256"] = cap.row_digest(row)
    assert any("lacks its A/A noise floor" in p for p in cap.validate_row(row))
    row = copy.deepcopy(rows[0])
    row["samples"]["noise_floor"]["unit"] = "ms"
    row["row_sha256"] = cap.row_digest(row)
    assert any("metric's own unit" in p for p in cap.validate_row(row))
    row = copy.deepcopy(rows[0])
    del row["extra"]["provenance"]["host_context"]["region_lock_claim"]
    row["extra"]["provenance_sha256"] = cap.content_hash(row["extra"]["provenance"])
    row["row_sha256"] = cap.row_digest(row)
    assert any("region_lock_claim" in p for p in cap.validate_row(row))


def test_build_rows_refuses_a_doctored_record(tmp_path):
    """The projection validates what it builds: a record that cannot make valid rows makes none."""
    run = speed_run(tmp_path)
    fill_speed(run)
    run.finalize()
    record = json.loads((tmp_path / "run-1.speed.record.json").read_text())
    args = {
        "record_file": "run-1.speed.record.json",
        "record_sha256": "0" * 64,
        "emitted_at": T_END,
    }
    assert cap.validate_rows(cap.build_rows(record, **args)) == []
    for mutate, match in (
        (lambda r: r["results"][0].update(n=0), "reps must be a positive int|needs int n >= 1"),
        (lambda r: r["results"][0].update(value=float("nan")), "malformed|finite"),
        (lambda r: r["results"][0].pop("noise_floor"), "malformed"),
        (lambda r: r["host_context"].pop("loadavg"), "loadavg"),
        (lambda r: r["results"][0].update(metric="recall_at_5"), "unknown speed metric"),
    ):
        doctored = copy.deepcopy(record)
        mutate(doctored)
        with pytest.raises(cap.CaptureError, match=match):
            cap.build_rows(doctored, **args)


# ------------------------------------------------------------------ arm identity


@pytest.mark.parametrize("field", list(cap.ARM_FIELDS))
def test_every_arm_identity_field_is_required(tmp_path, field):
    run = retrieval_run(tmp_path)
    arm = dense_arm()
    del arm[field]
    with pytest.raises(cap.CaptureError, match="arm identity"):
        run.add_recall(arm, "all", "dense", 5, "recall", 0.5, 120, 60, None, CONFIG)


@pytest.mark.parametrize(
    "over,match",
    [
        ({"model": None}, "model must be non-empty"),
        ({"revision": ""}, "revision"),
        ({"dim": 0}, "dim"),
        ({"dim": "1024"}, "dim"),
        ({"prompt_hash": "abc"}, "prompt_hash"),
        ({"index_identity_digest": None}, "index_identity_digest"),
        ({"role": "OPTIMUM"}, "role"),
        ({"runtime": " "}, "runtime"),
    ],
)
def test_a_malformed_dense_identity_is_refused(tmp_path, over, match):
    run = retrieval_run(tmp_path)
    with pytest.raises(cap.CaptureError, match=match):
        run.add_recall(dense_arm(**over), "all", "dense", 5, "recall", 0.5, 120, 60, None, CONFIG)


def test_lexical_and_dense_identities_do_not_cross(tmp_path):
    run = retrieval_run(tmp_path)
    with pytest.raises(cap.CaptureError, match="must carry .* as null"):
        run.add_recall(dense_arm("bm25"), "all", "lexical", 5, "recall", 0.5, 120, 60, None, CONFIG)
    with pytest.raises(cap.CaptureError, match="model must be non-empty"):
        run.add_recall(lexical_arm(), "all", "hybrid", 5, "recall", 0.5, 120, 60, None, CONFIG)


def test_one_arm_id_keeps_one_identity(tmp_path):
    run = retrieval_run(tmp_path)
    run.add_recall(dense_arm(), "all", "dense", 5, "recall", 0.5, 120, 60, None, CONFIG)
    with pytest.raises(cap.CaptureError, match="different identity"):
        run.add_recall(
            dense_arm(revision="0" * 40), "all", "hybrid", 5, "recall", 0.5, 120, 60, None, CONFIG
        )
    with pytest.raises(cap.CaptureError, match="duplicate measurement"):
        run.add_recall(dense_arm(), "all", "dense", 5, "recall", 0.5, 120, 60, None, CONFIG)


# ------------------------------------------------------------------ categories, n, pairing


def test_the_pooled_decision_set_needs_the_registered_n(tmp_path):
    run = retrieval_run(tmp_path)
    with pytest.raises(cap.CaptureError, match="below the pre-registered n_min=120"):
        run.add_recall(
            lexical_arm(), "all", "lexical", 5, "recall", 59 / 119, 119, 59, None, CONFIG
        )
    # a stratum may be smaller: 45 queries is reported, not decided on
    run.add_recall(lexical_arm(), "probe45", "lexical", 5, "recall", 20 / 45, 45, 20, None, CONFIG)
    with pytest.raises(cap.CaptureError, match="not a key of query_digests"):
        run.add_recall(lexical_arm(), "mystery", "lexical", 5, "recall", 0.5, 120, 60, None, CONFIG)
    with pytest.raises(cap.CaptureError, match="below the pre-registered n_min"):
        run.add_paired_diff(
            dense_arm(), lexical_arm(), "all", 5, 0.1, [0.0, 0.2], 119, 1000, 1,
            mode_a="dense", mode_b="lexical",
        )  # fmt: skip


def test_arms_scored_on_different_n_in_one_category_cannot_finalize(tmp_path):
    run = retrieval_run(tmp_path)
    run.add_recall(lexical_arm(), "probe45", "lexical", 5, "recall", 20 / 45, 45, 20, None, CONFIG)
    run.add_recall(dense_arm(), "probe45", "dense", 5, "recall", 20 / 44, 44, 20, None, CONFIG)
    with pytest.raises(cap.CaptureError, match="never a dropped query"):
        run.finalize()


def test_a_paired_diff_must_equal_the_difference_of_its_recall_rows(tmp_path):
    run = retrieval_run(tmp_path)
    run.add_recall(lexical_arm(), "all", "lexical", 5, "recall", 0.5, 120, 60, None, CONFIG)
    run.add_recall(dense_arm(), "all", "hybrid", 5, "recall", 0.7, 120, 84, None, CONFIG)
    run.add_paired_diff(
        dense_arm(), lexical_arm(), "all", 5, 0.3, [0.1, 0.3], 120, 1000, 1,
        mode_a="hybrid", mode_b="lexical",
    )  # fmt: skip
    with pytest.raises(cap.CaptureError, match=r"is not recall\(A\) - recall\(B\)"):
        run.finalize()
    run2 = retrieval_run(tmp_path, run_id="run-2")
    run2.add_recall(lexical_arm(), "all", "lexical", 5, "recall", 0.5, 120, 60, None, CONFIG)
    run2.add_paired_diff(
        dense_arm(), lexical_arm(), "all", 5, 0.2, [0.1, 0.3], 120, 1000, 1,
        mode_a="hybrid", mode_b="lexical",
    )  # fmt: skip
    with pytest.raises(cap.CaptureError, match="no recall row for both sides"):
        run2.finalize()
    with pytest.raises(cap.CaptureError, match="against itself"):
        run2.add_paired_diff(
            lexical_arm(), lexical_arm(), "all", 5, 0.0, [-0.1, 0.1], 120, 1000, 1,
            mode_a="lexical", mode_b="lexical",
        )  # fmt: skip


def test_cost_refusals(tmp_path):
    run = retrieval_run(tmp_path)
    with pytest.raises(cap.CaptureError, match="p95_ms is below p50_ms"):
        run.add_cost(dense_arm(), 0.1, 90.0, 80.0, 1.0, 120, mode="dense")
    with pytest.raises(cap.CaptureError, match="mode must be one of"):
        run.add_cost(dense_arm(), 0.1, 80.0, 90.0, 1.0, 120, mode="colbert")
    with pytest.raises(cap.CaptureError, match="cpu_s_per_query"):
        run.add_cost(dense_arm(), 0.0, 80.0, 90.0, 1.0, 120, mode="dense")


# ------------------------------------------------------------------ pooling guards at the sidecar level


def test_a_sidecar_that_mixes_instruments_is_refused(tmp_path):
    rrun = retrieval_run(tmp_path)
    fill(rrun)
    rrows = read_rows(rrun.finalize())
    srun = speed_run(tmp_path)
    fill_speed(srun)
    srows = read_rows(srun.finalize())
    problems = cap.validate_rows([rrows[0], srows[0]])
    assert problems == [
        "sidecar mixes instruments or sources: offline_recall and offline_speed never pool"
    ]
    # two retrieval runs do not pool either
    other = retrieval_run(tmp_path, run_id="run-9")
    fill(other)
    orows = read_rows(other.finalize())
    assert any("mixes runs" in p for p in cap.validate_rows([rrows[0], orows[0]]))
    assert any("duplicate measurement" in p for p in cap.validate_rows([rrows[0], rrows[0]]))


def test_latency_percentiles_must_be_ordered(tmp_path):
    run = speed_run(tmp_path)
    arm = dense_arm("bge-st-fp32", role="BASELINE")
    run.add_speed(arm, "query_p50_ms", 90.0, "ms", 600, 3.0, "ms", SPEED_CFG)
    run.add_speed(arm, "query_p95_ms", 80.0, "ms", 600, 3.0, "ms", SPEED_CFG)
    for metric in ("query_p50_ms", "query_p95_ms"):
        run.add_noise_floor(arm, metric, 2.0, "ms", replication_unit="arm", n=3)
    with pytest.raises(cap.CaptureError, match="not ordered p50 <= p95 <= max"):
        run.finalize()


def test_speed_unit_must_be_the_declared_one(tmp_path):
    run = speed_run(tmp_path)
    arm = dense_arm("bge-st-fp32", role="BASELINE")
    with pytest.raises(cap.CaptureError, match="reported in 'chunks/s'"):
        run.add_speed(arm, "chunks_per_s", 12.0, "texts/s", 3, 0.1, "texts/s", SPEED_CFG)
    with pytest.raises(cap.CaptureError, match="metric must be one of"):
        run.add_speed(arm, "recall", 0.5, "fraction", 3, 0.1, "x", SPEED_CFG)
    with pytest.raises(cap.CaptureError, match="spread_unit"):
        run.add_speed(arm, "chunks_per_s", 12.0, "chunks/s", 3, 0.1, "", SPEED_CFG)
    with pytest.raises(cap.CaptureError, match="config_echo"):
        run.add_speed(arm, "chunks_per_s", 12.0, "chunks/s", 3, 0.1, "chunks/s", {})
    run.add_speed(arm, "chunks_per_s", 12.0, "chunks/s", 3, 0.1, "chunks/s", SPEED_CFG)
    with pytest.raises(cap.CaptureError, match="duplicate measurement"):
        run.add_speed(arm, "chunks_per_s", 12.1, "chunks/s", 3, 0.1, "chunks/s", SPEED_CFG)


# ------------------------------------------------------------------ tamper detection (the reader's checks)


def _mutations(rows):
    row = rows[0]

    def tamper(fn, rehash=True):
        out = copy.deepcopy(row)
        fn(out)
        if rehash:
            out["row_sha256"] = cap.row_digest(out)
        return out

    return {
        "value": (tamper(lambda r: r.update(value=r["value"] + 0.1)), "re-derive"),
        "rehash_missing": (
            tamper(lambda r: r.update(value=r["value"] + 0.1), rehash=False),
            "row_sha256",
        ),
        "category": (
            tamper(
                lambda r: r.update(
                    category="BASELINE" if r["category"] == "CANDIDATE" else "CANDIDATE"
                )
            ),
            "role",
        ),
        "locator": (tamper(lambda r: r["extra"].update(locator="replret:x")), "locator"),
        "unit": (tamper(lambda r: r.update(unit="furlongs")), "unit/direction"),
        "direction": (
            tamper(lambda r: r.update(metric_direction="lower_better")),
            "unit/direction",
        ),
        "record_file": (tamper(lambda r: r.update(record_file="../x.json")), "bare file name"),
        "foreign_schema": (tamper(lambda r: r.update(schema="other")), "schema is foreign"),
        "reps": (tamper(lambda r: r.update(reps=r["reps"] + 1)), "reps does not equal"),
        "extra_field": (tamper(lambda r: r.update(bonus=1)), "row fields differ"),
        "measurement_id": (
            tamper(lambda r: r.update(measurement_id="replret-x")),
            "measurement_id",
        ),
        "hook": (
            tamper(
                lambda r: (
                    r["extra"]["provenance"]["run"].update(started_at="2026-10-19T10:00:00+00:00"),
                    r["extra"].update(provenance_sha256=cap.content_hash(r["extra"]["provenance"])),
                )
            ),
            "retrospective",
        ),
    }


@pytest.mark.parametrize(
    "name",
    [
        "value",
        "rehash_missing",
        "category",
        "locator",
        "unit",
        "direction",
        "record_file",
        "foreign_schema",
        "reps",
        "extra_field",
        "measurement_id",
        "hook",
    ],
)
def test_tampered_rows_are_refused(tmp_path, name):
    run = retrieval_run(tmp_path)
    fill(run)
    rows = read_rows(run.finalize())
    row, match = _mutations(rows)[name]
    problems = cap.validate_row(row)
    assert any(match in p for p in problems), problems


def test_a_run_before_the_hook_cannot_be_written(tmp_path):
    with pytest.raises(cap.CaptureError, match="retrospective"):
        cap.RetrievalRun(
            tmp_path, "old", cap.registered_rule(), CORPUS, QUERIES, LABELS, MANIFEST, host(),
            clock=clock("2026-10-19T10:00:00+00:00"),
        )  # fmt: skip


# ------------------------------------------------------------------ digests and FROZEN-AT-LAUNCH


def test_digests_must_be_sha256(tmp_path):
    with pytest.raises(cap.CaptureError, match="corpus_digests"):
        cap.RetrievalRun(
            tmp_path,
            "r",
            cap.registered_rule(),
            {"corpus": "abc"},
            QUERIES,
            LABELS,
            MANIFEST,
            host(),
        )
    with pytest.raises(cap.CaptureError, match="query_digests"):
        cap.RetrievalRun(tmp_path, "r", cap.registered_rule(), CORPUS, {}, LABELS, MANIFEST, host())
    with pytest.raises(cap.CaptureError, match="label_digests"):
        cap.RetrievalRun(
            tmp_path, "r", cap.registered_rule(), CORPUS, QUERIES, None, MANIFEST, host()
        )
    with pytest.raises(cap.CaptureError, match="arm_manifest_sha256"):
        cap.RetrievalRun(
            tmp_path, "r", cap.registered_rule(), CORPUS, QUERIES, LABELS, "nope", host()
        )
    with pytest.raises(cap.CaptureError, match="run_id"):
        cap.RetrievalRun(
            tmp_path, "../r", cap.registered_rule(), CORPUS, QUERIES, LABELS, MANIFEST, host()
        )


def test_frozen_at_launch_must_contain_every_digest(tmp_path):
    lines = [
        f"{d}  {name}" for name, d in {**CORPUS, **LABELS, "manifest": MANIFEST, **QUERIES}.items()
    ]
    good = tmp_path / "FROZEN-AT-LAUNCH.sha256"
    good.write_text("\n".join(lines) + "\n")
    run = retrieval_run(tmp_path, frozen_at_launch_path=good)
    fill(run)
    rows = read_rows(run.finalize())
    assert rows[0]["extra"]["provenance"]["digests"]["frozen_at_launch_sha256"] == cap.file_sha256(
        good
    )
    short = tmp_path / "short.sha256"
    short.write_text("\n".join(lines[:2]) + "\n")
    with pytest.raises(cap.CaptureError, match="absent from short.sha256"):
        retrieval_run(tmp_path, run_id="r2", frozen_at_launch_path=short)
    with pytest.raises(cap.CaptureError, match="does not exist"):
        retrieval_run(tmp_path, run_id="r3", frozen_at_launch_path=tmp_path / "nope")


def test_harness_sha_is_recorded_when_given(tmp_path):
    harness = tmp_path / "harness.py"
    harness.write_text("print('x')\n")
    run = retrieval_run(tmp_path, harness_path=harness)
    fill(run)
    rows = read_rows(run.finalize())
    assert rows[0]["extra"]["provenance"]["instrument"]["harness"][
        "harness_sha256"
    ] == cap.file_sha256(harness)
    assert rows[0]["extra"]["provenance"]["instrument"]["capture_sha256"] == cap.file_sha256(
        cap.__file__
    )


# ------------------------------------------------------------------ the host window


def test_host_window_emits_a_valid_timing_context():
    samples = iter(
        [
            {"loadavg": [0.5, 0.6, 0.7], "cpus_allowed_observed": "24-71", "mhz": [3600.0, 3700.0],
             "governor": "performance", "capped": False},
            {"loadavg": [0.9, 0.7, 0.7], "cpus_allowed_observed": "24-71", "mhz": [3500.0, 4100.0],
             "governor": "performance", "capped": False},
        ]
    )  # fmt: skip
    window = cap.HostWindow("24-71", "q1+q2:sess", sample_fn=lambda cpus: next(samples))
    with pytest.raises(cap.CaptureError, match="both ends"):
        window.context()
    window.begin()
    window.finish()
    ctx = window.context()
    assert cap.check_host_context(ctx, timing=True) == []
    assert ctx["frequency"] == {
        "governor": "performance",
        "mhz_min": 3500.0,
        "mhz_max": 4100.0,
        "throttled": False,
    }
    assert ctx["loadavg"] == {"start": [0.5, 0.6, 0.7], "end": [0.9, 0.7, 0.7]}
    assert not any(math.isnan(x) for x in ctx["loadavg"]["start"])


def test_host_window_refuses_an_affinity_outside_the_declared_cpu_list():
    sample = {"loadavg": [0.5, 0.6, 0.7], "cpus_allowed_observed": "0-95", "mhz": [3600.0],
              "governor": "performance", "capped": False}  # fmt: skip
    window = cap.HostWindow("24-71", "q1+q2:sess", sample_fn=lambda cpus: dict(sample))
    window.begin()
    window.finish()
    with pytest.raises(cap.CaptureError, match="not inside the declared cpu_list"):
        window.context()


def test_real_host_sample_never_fabricates():
    """Reading /proc and /sys is light; where cpufreq is absent the sampler raises rather than guess."""
    import os

    allowed = "0"
    if hasattr(os, "sched_getaffinity"):
        allowed = str(sorted(os.sched_getaffinity(0))[0])
    try:
        got = cap.sample_host(allowed)
    except cap.CaptureError as exc:
        assert "unavailable" in str(exc)
        return
    assert len(got["loadavg"]) == 3 and got["mhz"] and got["governor"]
