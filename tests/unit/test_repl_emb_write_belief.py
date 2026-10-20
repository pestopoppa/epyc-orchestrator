"""VB-UFH12-RETR / REPL-EMB-2.4: the post-hoc projector of the 2.2 run into the belief write side.

A tiny synthetic run directory stands in for the real one: no model, server, timer or process is
touched and no real query text exists anywhere in the fixture.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark"))
import repl_emb_eval as ev  # noqa: E402
import repl_emb_write_belief as wb  # noqa: E402
import repl_retrieval_capture as cap  # noqa: E402

KS = (1, 3, 5, 10)
N_STRATUM = {"probe45": 10, "agent_nav": 20, "synthetic_spill": 100}
N_FLAGGED = {"agent_nav": 4, "synthetic_spill": 20}
VARIANTS = (
    "bge-large.dense",
    "bge-large.hybrid",
    "bm25",
    "colgrep",
    "grep",
    "potion-base-8m.dense",
    "potion-base-8m.hybrid",
    "potion-base-8m.whybrid",
)
BASE_RATE = {
    "grep": 0.50,
    "bm25": 0.42,
    "bge-large.dense": 0.45,
    "bge-large.hybrid": 0.47,
    "potion-base-8m.dense": 0.35,
    "potion-base-8m.hybrid": 0.44,
    "potion-base-8m.whybrid": 0.46,
    "colgrep": 0.62,
}
SPECS = ev.eval_specs("llama-test", "m2v-test")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def put_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, indent=1) + "\n")
    return path


def put_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def hits_for(variant: str, n: int, k: int) -> int:
    rate = BASE_RATE[variant] + 0.08 * KS.index(k)
    return min(n, round(n * rate))


def cell(hits: int, n: int) -> dict:
    r = hits / n
    return {"ci": "wilson-95", "ci_hi": min(1.0, r + 0.1), "ci_lo": max(0.0, r - 0.1),
            "hits": float(hits), "n": n, "recall": r}  # fmt: skip


def build_analysis() -> dict:
    n_all = sum(N_STRATUM.values())
    recall: dict = {}
    for v in VARIANTS:
        strata = ["probe45"] if v == "colgrep" else list(N_STRATUM)
        recall[v] = {
            s: {str(k): cell(hits_for(v, N_STRATUM[s], k), N_STRATUM[s]) for k in KS}
            for s in strata
        }
        if v == "colgrep":
            recall[v]["pooled"] = copy.deepcopy(recall[v]["probe45"])
        else:
            recall[v]["pooled"] = {
                str(k): cell(sum(hits_for(v, N_STRATUM[s], k) for s in N_STRATUM), n_all)
                for k in KS
            }
    boot: dict = {}
    for v in VARIANTS:
        if v == "grep":
            continue
        where = "probe45" if v == "colgrep" else "pooled"
        a, b = recall[v][where]["5"], recall["grep"][where]["5"]
        diff = (a["hits"] - b["hits"]) / a["n"]
        boot[v] = {
            "vs_L": {
                "L": "grep", "ci_hi": diff + 0.05, "ci_lo": diff - 0.05, "ci_level": 0.95,
                "mean_a": a["recall"], "mean_b": b["recall"], "mean_diff": diff, "n": a["n"],
                "method": "paired by-query percentile bootstrap", "resamples": 10000,
                "seed": 20261020,
            }
        }  # fmt: skip
    completeness = {v: {"n_failed": 0, "n": n_all} for v in VARIANTS}
    completeness["bge-large.dense"]["n_failed"] = 3
    return {
        "schema": "repl-emb-analysis/3",
        "bias_note": "Synthetic queries are LLM-written (fixture).",
        "variants": list(VARIANTS),
        "n_deciding": n_all,
        "n_flagged_excluded": sum(N_FLAGGED.values()),
        "failures_count_as_misses": True,
        "floor_check": {
            "ok": True,
            "n": n_all,
            "n_trace": n_all - N_STRATUM["probe45"],
            "trace_min": 75,
            "n_min": 120,
        },  # fmt: skip
        "lexical_baseline": {"L": "grep"},
        "recall": recall,
        "bootstrap_recall5": boot,
        "completeness": completeness,
        "decision_sets": {
            "primary": {"recall": copy.deepcopy(recall), "bootstrap_recall5": copy.deepcopy(boot)},
            "pooled_flagged_included": {
                "n_deciding": n_all + sum(N_FLAGGED.values()),
                "completeness": {**completeness, "bge-large.dense": {"n_failed": 5, "n": 144}},
            },
        },
    }


def build_config_echo(specs_digest: dict, harness_sha: str) -> dict:
    return {
        "git_sha": "1" * 40,
        "harness_files_dirty": False,
        "stable": {
            "arm_identities": {
                a: {"digest": d, "model_id": f"{a}#x"} for a, d in specs_digest.items()
            },
            "context_search_sha256": sha_text("context_search"),
            "freeze_sha256": sha_text("freeze"),
            "harness_sha256": {"repl_emb_eval.py": harness_sha},
            "predeclared": {
                "colgrep": {
                    "alpha": "0.95",
                    "bin": "/x/colgrep",
                    "env": {"NEXT_PLAID_FORCE_CPU": "1"},
                    "expected_sha256": sha_text("colgrep-bin"),
                },  # fmt: skip
                "context_search_constants": {
                    "BM25_B": 0.75,
                    "BM25_K1": 1.2,
                    "CHUNK_MAX_CHARS": 800,
                    "RRF_K": 60,
                    "CANDIDATES_PER_RANKER": 64,
                },  # fmt: skip
                "grep_stopwords_sha256": sha_text("stop"),
                "stats": {
                    "M": 0.1,
                    "X": 0.05,
                    "ci_level": 0.95,
                    "k_primary": 5,
                    "n_min": 120,
                    "resamples": 10000,
                    "seed": 20261020,
                    "trace_min": 75,
                },  # fmt: skip
                "top_k": 10,
                "weighted_hybrid": {"k": 60, "w_bm25": 0.25, "w_dense": 0.75},
            },
        },
    }


@pytest.fixture()
def world(tmp_path):
    """A run directory shaped like the real one, plus the out dir and the offered arm manifest."""
    run, inp = tmp_path / "run", tmp_path / "in"
    run.mkdir()
    inp.mkdir()
    manifest = put_json(
        inp / "arm_manifest.json",
        {"arms": [{"arm_id": "eg2-gguf-bf16-768", "runtime": "llama-test"},
                  {"arm_id": "potion-code-16m-v2", "runtime": "m2v-test"}]},
    )  # fmt: skip
    labelled, files = [], {}
    for group, n in N_STRATUM.items():
        rows = []
        for i in range(n + N_FLAGGED.get(group, 0)):
            flagged = i >= n
            row = {
                "id": f"{group}-{i}",
                "group": group,
                "scored": True,
                "bundle": ("probe45:repo/a" if group == "probe45" else f"{group[:2]}b{i % 3:060d}"
                           if group == "agent_nav" else f"{i % 4:016x}"),
                "text": "SECRET-QUERY-TEXT",
            }  # fmt: skip
            if group != "probe45":
                row["answer_named_verbatim"] = flagged
            rows.append(row)
        labelled += rows
        files[group] = rows
    labelled += [{"id": f"th-{i}", "group": "trivial_head", "scored": True, "bundle": "x",
                  "answer_named_verbatim": False, "text": "SECRET-QUERY-TEXT"} for i in range(3)]  # fmt: skip
    put_jsonl(run / "queries-labelled.jsonl", labelled)
    probe_q = put_jsonl(inp / "probe45.jsonl", files["probe45"])
    trace_q = put_jsonl(inp / "queries-trace.jsonl", files["agent_nav"])
    synth_q = put_jsonl(inp / "queries-synth.jsonl", files["synthetic_spill"])
    trivial_q = put_jsonl(inp / "queries-trivial-head.jsonl", [{"id": "th-0"}])
    trace_b = put_json(
        inp / "bundles.json", {r["bundle"]: {"files": []} for r in files["agent_nav"]}
    )
    synth_b = put_json(inp / "bundles-harness.json", {f"{i:016x}": {"files": []} for i in range(4)})
    (run / "corpus-digest-probe45.txt").write_text("probe45:repo/a\tx.py\t" + "0" * 64 + "\n")
    put_json(run / "gold-check.json", {"ok": True})
    for name in ("analysis.md", "FREEZE-RULES.md", "DEVIATIONS.md", "floor-check.json",
                 "predeclared-interpretations.json", "bundles-summary.json"):  # fmt: skip
        (run / name).write_text(f"fixture {name}\n")
    harness_sha = cap.file_sha256(Path(ev.__file__))
    digests = {a: SPECS[a].index_identity()["digest"] for a in ("bge-large", "potion-base-8m")}
    put_json(run / "config-echo.json", build_config_echo(digests, harness_sha))
    for arm in ("bge-large", "potion-base-8m"):
        put_json(run / f"armmeta-{arm}.json", {"arm_id": arm, "dim": SPECS[arm].dim,
                                               "identity_digest": digests[arm]})  # fmt: skip
    put_json(run / "armmeta-colgrep.json", {"arm_id": "colgrep"})
    put_json(run / "analysis.json", build_analysis())
    put_jsonl(run / "invocations.jsonl", [
        {"ts": "2026-10-20T19:46:52Z", "endpoints": {}},
        {"ts": "2026-10-20T19:47:09Z", "endpoints": {"bge-large": "http://127.0.0.1:18761"}},
        {"ts": "2026-10-20T20:14:36Z", "endpoints": {"bge-large": "http://127.0.0.1:18765"}},
    ])  # fmt: skip
    put_json(run / "inputs.json", {
        "arm_manifest": str(manifest), "probe45": str(probe_q),
        "bundles_json": [str(trace_b), str(synth_b)],
        "trace_queries": [str(trace_q), str(synth_q)], "trivial_queries": [str(trivial_q)],
    })  # fmt: skip
    frozen = [manifest, run / "queries-labelled.jsonl", probe_q, trace_q, synth_q, trace_b, synth_b,
              run / "corpus-digest-probe45.txt", run / "gold-check.json"]  # fmt: skip
    (run / "FROZEN-AT-LAUNCH.sha256").write_text(
        "".join(f"{cap.file_sha256(p)}  {p}\n" for p in frozen)
    )
    return {"run": run, "out": tmp_path / "out", "manifest": manifest, "tmp": tmp_path}


def write(w, **kw):
    return wb.write_run(w["run"], w["out"], arm_manifest=w["manifest"], run_id="fixture-run",
                        hostname="test-host", **kw)  # fmt: skip


def rows_of(summary) -> list[dict]:
    return [json.loads(x) for x in Path(summary["sidecar"]).read_text().splitlines()]


def edit_analysis(w, mutate) -> None:
    path = w["run"] / "analysis.json"
    data = json.loads(path.read_text())
    mutate(data)
    put_json(path, data)


# ------------------------------------------------------------------ the happy path


def test_the_sidecar_validates_and_holds_recall_and_paired_rows_only(world):
    summary = write(world)
    rows = rows_of(summary)
    assert cap.validate_rows(rows) == []
    # 7 non-ColGREP variants x (all + 3 strata) x 4 k, plus ColGREP on probe45 x 4 k
    assert summary["rows_by_kind"] == {"recall": 7 * 4 * 4 + 4, "paired_diff": 6}
    assert {r["kind"] for r in rows} == {"recall", "paired_diff"}  # no cost row, ever
    assert {r["instrument"] for r in rows} == {"offline_recall"}
    assert {r["protocol_id"] for r in rows} == {""}
    assert summary["amendment"] is None  # the registered rule is written unchanged
    assert Path(summary["record"]).is_file()


def test_pooled_all_is_the_primary_set_and_every_stratum_has_one_n(world):
    rows = [r for r in rows_of(write(world)) if r["kind"] == "recall"]
    ns: dict[str, set[int]] = {}
    for r in rows:
        ns.setdefault(r["query_category"], set()).add(r["samples"]["n"])
    assert ns == {"all": {130}, "probe45": {10}, "agent_nav": {20}, "synthetic_spill": {100}}
    assert {r["samples"]["hits"] / r["samples"]["n"] for r in rows} == {r["value"] for r in rows}


def test_colgrep_exists_on_probe45_only_and_never_in_all(world):
    rows = [r for r in rows_of(write(world)) if r["arm_id"] == "colgrep"]
    assert {(r["kind"], r["query_category"]) for r in rows} == {
        ("recall", "probe45"),
        ("paired_diff", "probe45"),
    }
    recall = next(r for r in rows if r["kind"] == "recall")
    assert recall["mode"] == "hybrid"
    assert recall["extra"]["arm"]["revision"] == "unrecorded-at-run"
    assert "RESOLVED POST HOC" in recall["extra"]["config_echo"]["identity_basis"]["status"]


def test_roles_follow_the_operator_ruling(world):
    roles = {r["arm_id"]: r["category"] for r in rows_of(write(world)) if r["kind"] == "recall"}
    assert roles == {
        "grep": "BASELINE",
        "bm25": "BASELINE",
        "bge-large": "BASELINE",
        "potion-base-8m": "CANDIDATE",
        "potion-base-8m.whybrid": "CANDIDATE",
        "colgrep": "CANDIDATE",
    }


def test_lexical_arms_carry_explicit_nulls_and_dense_arms_their_digest(world):
    rows = [r for r in rows_of(write(world)) if r["kind"] == "recall"]
    arms = {r["arm_id"]: r["extra"]["arm"] for r in rows}
    for lexical in ("grep", "bm25"):
        assert all(arms[lexical][f] is None for f in cap.DENSE_ONLY_FIELDS)
    bge = arms["bge-large"]
    assert bge["index_identity_digest"] == SPECS["bge-large"].index_identity()["digest"]
    assert bge["dim"] == 1024 and bge["model"] == "BAAI/bge-large-en-v1.5"
    weighted = arms["potion-base-8m.whybrid"]
    assert weighted["index_identity_digest"] == arms["potion-base-8m"]["index_identity_digest"]


def test_paired_diffs_are_recall_a_minus_recall_grep_on_all(world):
    rows = rows_of(write(world))
    recall = {
        (r["arm_id"], r["mode"], r["query_category"], r["k"]): r["value"]
        for r in rows
        if r["kind"] == "recall"
    }
    diffs = [r for r in rows if r["kind"] == "paired_diff"]
    assert {r["arm_id"] for r in diffs} == {
        "bge-large", "potion-base-8m", "potion-base-8m.whybrid", "colgrep",
    }  # fmt: skip
    for d in diffs:
        s = d["samples"]
        assert (s["against_arm_id"], s["against_mode"]) == ("grep", "lexical")
        assert d["k"] == 5 and d["metric"] == "recall_at_5_paired_diff"
        assert (s["resamples"], s["seed"]) == (10000, 20261020)
        a = recall[(d["arm_id"], d["mode"], d["query_category"], 5)]
        b = recall[("grep", "lexical", d["query_category"], 5)]
        assert d["value"] == pytest.approx(a - b, abs=1e-12)
    assert {d["query_category"] for d in diffs if d["arm_id"] != "colgrep"} == {"all"}
    assert {(d["arm_id"], d["mode"]) for d in diffs} >= {
        ("bge-large", "dense"), ("bge-large", "hybrid"), ("potion-base-8m.whybrid", "hybrid"),
    }  # fmt: skip


def test_the_registered_rule_rides_unchanged_and_the_floors_travel_in_the_echo(world):
    rows = rows_of(write(world))
    rules = {json.dumps(r["extra"]["provenance"]["rule"], sort_keys=True) for r in rows}
    assert rules == {json.dumps(cap.registered_rule(), sort_keys=True)}
    echo = next(r for r in rows if r["kind"] == "recall")["extra"]["config_echo"]
    assert echo["stats"]["trace_min"] == 75 and echo["stats"]["n_min"] == 120
    assert echo["decision_set"]["n"] == 130 and echo["decision_set"]["n_by_stratum"] == N_STRATUM
    assert echo["decision_set"]["flagged_excluded"] == 24
    assert echo["bge_failures"]["failed_in_primary_set"] == 3
    assert echo["bge_failures"]["failed_queries"] == 5
    assert "LLM-written" in echo["synthetic_stratum"]
    assert echo["failures_count_as_misses"] is True
    assert "contended" in echo["deviations"].lower() or "CONTAMINATED" in echo["deviations"]
    assert echo["endpoints"]["bge-large"] == ["http://127.0.0.1:18761", "http://127.0.0.1:18765"]


def test_the_run_window_is_the_real_start_and_the_digests_are_frozen_ones(world):
    summary = write(world)
    prov = rows_of(summary)[0]["extra"]["provenance"]
    assert prov["run"]["started_at"] == "2026-10-20T19:46:52+00:00"
    assert prov["run"]["finished_at"] >= prov["run"]["started_at"]
    assert prov["host_context"]["hostname"] == "test-host"
    assert "region_lock_claim" not in prov["host_context"]  # recall-only: no timing claim
    assert set(prov["digests"]["query"]) == {"all", "probe45", "agent_nav", "synthetic_spill"}
    labelled = cap.file_sha256(world["run"] / "queries-labelled.jsonl")
    assert prov["digests"]["query"]["all"] == labelled
    assert prov["digests"]["label"]["queries-labelled.jsonl"] == labelled
    frozen_copy = world["out"] / "FROZEN-AT-LAUNCH.sha256"
    assert prov["digests"]["frozen_at_launch_sha256"] == cap.file_sha256(frozen_copy)
    assert prov["digests"]["arm_manifest_sha256"] == cap.file_sha256(world["manifest"])


def test_evidence_is_copied_but_no_query_text_or_per_query_row_is(world):
    perquery = world["run"] / "perquery-grep.jsonl"
    perquery.write_text('{"qid": "x"}\n')
    summary = write(world)
    out = world["out"]
    names = {p.name for p in out.iterdir()}
    assert {"analysis.json", "FROZEN-AT-LAUNCH.sha256", "armmeta-colgrep.json",
            "armmeta-bge-large.json", "DEVIATIONS.md", "bundles-summary.json"} <= names  # fmt: skip
    assert "perquery-grep.jsonl" not in names and "queries-labelled.jsonl" not in names
    assert "SECRET-QUERY-TEXT" not in "".join(p.read_text() for p in out.iterdir())
    assert set(summary["evidence_copied"]) <= names


# ------------------------------------------------------------------ refusals


def test_a_second_write_is_refused_the_record_is_immutable(world):
    write(world)
    before = (world["out"] / "fixture-run.record.json").read_bytes()
    with pytest.raises(wb.WriteError, match="immutable"):
        write(world)
    assert (world["out"] / "fixture-run.record.json").read_bytes() == before


def test_a_bootstrap_mean_that_does_not_reconcile_with_the_hits_is_refused(world):
    edit_analysis(
        world, lambda a: a["bootstrap_recall5"]["bge-large.dense"]["vs_L"].update(mean_diff=0.2)
    )
    edit_analysis(
        world,
        lambda a: a["decision_sets"]["primary"].update(bootstrap_recall5=a["bootstrap_recall5"]),
    )
    with pytest.raises(wb.WriteError, match="mean_diff"):
        write(world)
    assert not (world["out"] / "fixture-run.record.json").exists()


def test_strata_that_do_not_sum_to_the_pooled_set_are_refused(world):
    def tamper(a):
        a["recall"]["grep"]["pooled"]["5"]["hits"] += 1.0
        a["recall"]["grep"]["pooled"]["5"]["recall"] = (
            a["recall"]["grep"]["pooled"]["5"]["hits"] / 130
        )
        a["decision_sets"]["primary"]["recall"] = a["recall"]

    edit_analysis(world, tamper)
    with pytest.raises(wb.WriteError, match="do not sum"):
        write(world)


def test_a_rebuilt_identity_that_differs_from_the_runs_is_refused(world):
    path = world["run"] / "config-echo.json"
    cfg = json.loads(path.read_text())
    cfg["stable"]["arm_identities"]["bge-large"]["digest"] = "0" * 64
    put_json(path, cfg)
    with pytest.raises(wb.WriteError, match="bge-large"):
        write(world)


def test_a_harness_that_is_not_the_frozen_one_is_refused(world, tmp_path):
    other = tmp_path / "other_harness.py"
    other.write_text("# not the frozen harness\n")
    with pytest.raises(wb.WriteError, match="not the harness"):
        write(world, harness_path=other)


def test_a_failed_floor_check_is_refused(world):
    edit_analysis(world, lambda a: a["floor_check"].update(ok=False))
    with pytest.raises(wb.WriteError, match="floor check"):
        write(world)


def test_a_digest_missing_from_the_frozen_file_is_refused_by_the_writer(world):
    frozen = world["run"] / "FROZEN-AT-LAUNCH.sha256"
    kept = [x for x in frozen.read_text().splitlines() if "gold-check" not in x]
    frozen.write_text("\n".join(kept) + "\n")
    with pytest.raises(cap.CaptureError, match="absent from FROZEN-AT-LAUNCH"):
        write(world)


# ------------------------------------------------------------------ helpers


def test_split_variant_maps_every_pipeline_to_a_writer_mode():
    assert wb.split_variant("grep") == ("grep", "lexical", "grep")
    assert wb.split_variant("bge-large.dense") == ("bge-large", "dense", "dense")
    assert wb.split_variant("bge-large.hybrid") == ("bge-large", "hybrid", "hybrid")
    assert wb.split_variant("potion-code-16m-v2.whybrid") == (
        "potion-code-16m-v2.whybrid", "hybrid", "whybrid",
    )  # fmt: skip
    assert wb.split_variant("colgrep") == ("colgrep", "hybrid", "colgrep")
    with pytest.raises(wb.WriteError):
        wb.split_variant("mystery.sparse")


def test_the_script_has_no_cost_or_timing_path():
    # the speed block was dropped by operator ruling and every run timing is contended
    assert ".add_cost(" not in Path(wb.__file__).read_text()
