"""REPL-EMB-2.2 offline eval harness: arms, accounting, resume, freeze, analysis, dry run.

Synthetic bundles and fake embedders only: no network, no process, no model, no benchmark.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark"))
import repl_emb_arms as arms  # noqa: E402
import repl_emb_eval as ev  # noqa: E402
import repl_emb_stats as st  # noqa: E402

cs = arms.load_context_search()
SPECS = ev.eval_specs("llama-test", "m2v-test")
FAKE_HEAD = lambda p: "0" * 40  # noqa: E731


# ───────────────────────────────────────────────────────────────────────────── fixtures


def make_world(tmp_path):
    inp = ev.make_synthetic_inputs(tmp_path / "in")
    root = inp["root"]
    rows, summary = ev.label_queries(
        ev.read_jsonl(root / "probe45.jsonl"), ev.read_jsonl(root / "queries-trace.jsonl")
    )
    store = ev.make_bundle_store(
        rows,
        bundles_json=root / "bundles.json",
        probe_checkouts=inp["checkouts"],
        head_fn=FAKE_HEAD,
    )
    return SimpleNamespace(rows=rows, store=store, root=root, inp=inp, summary=summary)


def fake(arm_id, **kw):
    kw.setdefault("concepts", ev.SYNTH_CONCEPTS)
    kw.setdefault("query_concepts", ev.SYNTH_QUERY_CONCEPTS)
    return ev.HashAdapter(SPECS[arm_id], **kw)


def run_dense(out, world, arm_id, adapter=None, **kw):
    adapter = adapter or fake(arm_id)
    meta = ev.ArmMeta(out, arm_id)
    rep = ev.run_dense_arm(
        ev.DENSE_ARMS[arm_id], adapter, world.rows, world.store, ev.ResultStore(out), meta, **kw
    )
    return adapter, rep


def vec(out, world, variant, group="spill_follow", k=1):
    qs = [r for r in world.rows if r["group"] == group]
    sc = ev._score_variant(ev.ResultStore(out).rows(variant), qs)
    return sc["vec"][k].tolist(), sc


# ───────────────────────────────────────────────────────────────────────────── chunk arithmetic


def _chunks(text, max_chars=800):
    return cs.chunk_sections([ev.Section("f.txt", text)], max_chars)


def test_chunk_line_range_arithmetic():
    assert [ev.chunk_pointer(c) for c in _chunks("a\nb\nc\n")] == [("f.txt", 1, 3)]
    assert [ev.chunk_pointer(c) for c in _chunks("a\nb")] == [
        ("f.txt", 1, 2)
    ]  # no trailing newline
    # exactly ONE trailing newline is removed: trailing blank lines stay counted
    assert [ev.chunk_pointer(c) for c in _chunks("a\n\n\n")] == [("f.txt", 1, 3)]
    assert [ev.chunk_pointer(c) for c in _chunks("one line only")] == [("f.txt", 1, 1)]
    # 24 lines of 100 chars -> three 800-char chunks of 8 lines
    text = "".join(("x" * 99) + "\n" for _ in range(24))
    assert [ev.chunk_pointer(c) for c in _chunks(text)] == [
        ("f.txt", 1, 8),
        ("f.txt", 9, 16),
        ("f.txt", 17, 24),
    ]


def test_chunk_line_ranges_tile_the_file_and_cover_cut_lines():
    # a 2000-char first line is cut at 800 chars (pieces stay on line 1); "x" is line 2
    ptrs = [ev.chunk_pointer(c) for c in _chunks(("y" * 1999) + "\nx\n")]
    assert ptrs == [("f.txt", 1, 1), ("f.txt", 1, 1), ("f.txt", 1, 2)]
    # ragged lines: consecutive ranges never leave a gap and end on the last line
    lines = [("z" * n) + "\n" for n in (5, 300, 700, 1, 90, 800, 40, 40, 40)]
    ptrs = [ev.chunk_pointer(c) for c in _chunks("".join(lines))]
    assert ptrs[0][1] == 1 and ptrs[-1][2] == len(lines)
    for (_, _, e1), (_, s2, _) in zip(ptrs, ptrs[1:]):
        assert s2 in (e1, e1 + 1)


# ───────────────────────────────────────────────────────────────────────────── labelling


def test_code_intent_regex_is_the_predeclared_one():
    assert ev.CODE_INTENT_REGEX == (
        r"`|\b[a-z]+_[a-z0-9_]+\b|\b[A-Z][a-z0-9]+[A-Z]\w*\b|\w+\(\)|\b\w+\.\w+\("
    )
    for yes in (
        "what does `x` return",
        "where is load_data defined",
        "how does PlotData align frames",
        "why does foo() fail",
        "what does np.mean(x) do",
    ):
        assert ev.code_intent_label(yes) is True, yes
    for no in (
        "How does the pipeline align the data in the resulting frame?",
        "What happens when the index is not unique?",
        "Plot Data and Another Thing",
    ):
        assert ev.code_intent_label(no) is False, no


def test_label_queries_fills_nulls_keeps_inputs_and_flags_unscored():
    probe = [
        {
            "id": "p1",
            "text": "what is `x`",
            "repo": "mwaskom/seaborn",
            "code_intent": None,
            "gold": {"path": "a.py", "start_line": 3, "end_line": 9},
        },
        {
            "id": "p2",
            "text": "plain words",
            "repo": "pylint-dev/pylint",
            "code_intent": True,
            "gold": {"path": "b.py", "start_line": 1, "end_line": 2},
        },
    ]
    trace = [
        {
            "id": "t1",
            "text": "a b c",
            "bundle_id": "bx",
            "code_intent": None,
            "stratum": "spill",
            "gold": {"path": "/s/1.txt", "ranges": [[1, 4], [20, 30]]},
        },
        {"id": "t2", "text": "no gold", "bundle": "bx", "code_intent": False, "gold": None},
    ]
    trivial = [
        {
            "id": "h1",
            "text": "head",
            "bundle": "bx",
            "gold": {"path": "/s/1.txt", "ranges": [[1, 2]]},
        }
    ]
    rows, summary = ev.label_queries(probe, trace, trivial)
    by = {r["id"]: r for r in rows}
    assert by["p1"]["code_intent"] is True and by["p1"]["code_intent_source"] == "regex"
    assert by["p2"]["code_intent"] is True and by["p2"]["code_intent_source"] == "input"
    assert by["p1"]["bundle"] == "probe45:mwaskom/seaborn" and by["p1"]["group"] == "probe45"
    assert by["t1"]["group"] == "spill_follow" and by["t1"]["bundle"] == "bx"
    assert by["t1"]["gold_spans"] == [["/s/1.txt", 1, 4], ["/s/1.txt", 20, 30]]
    assert by["t2"]["scored"] is False and by["t1"]["scored"] is True
    assert by["h1"]["group"] == "trivial_head"
    assert [r["group"] for r in rows] == [
        "probe45",
        "probe45",
        "spill_follow",
        "spill_follow",
        "trivial_head",
    ]
    assert (
        summary["n_labelled_by_regex"] == 3 and summary["code_intent_regex"] == ev.CODE_INTENT_REGEX
    )
    assert summary["by_group"]["spill_follow"]["unscored"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        ev.label_queries(probe, [{**trace[0], "id": "p1"}])
    with pytest.raises(ValueError, match="bundle"):
        ev.label_queries(
            [], [{"id": "z", "text": "t", "gold": {"path": "p", "start_line": 1, "end_line": 1}}]
        )
    with pytest.raises(ValueError, match="unknown repo"):
        ev.label_queries([{**probe[0], "repo": "nobody/nothing"}], [])


def test_gold_spans_both_formats():
    assert ev.gold_spans({"path": "a", "start_line": 2, "end_line": 5}) == [("a", 2, 5)]
    assert ev.gold_spans({"path": "a", "ranges": [[1, 2], [7, 9]]}) == [("a", 1, 2), ("a", 7, 9)]
    assert ev.gold_spans(None) == [] and ev.gold_spans({"path": "a", "ranges": []}) == []
    with pytest.raises(ValueError):
        ev.gold_spans({"path": "a", "start_line": 9, "end_line": 2})


def test_the_real_probe45_file_labels_without_touching_any_arm():
    p = ev.REPO_ROOT / "data/repl_retrieval_eval/arm-manifest-20261020/probe45-intake1357.jsonl"
    if not p.exists():
        pytest.skip("probe45 file not in this checkout")
    rows, summary = ev.label_queries(ev.read_jsonl(p), [])
    assert len(rows) == 45 and all(r["scored"] for r in rows)
    assert {r["bundle"] for r in rows} == {f"probe45:{r}" for r in ev.PROBE_REPOS}
    assert summary["n_labelled_by_regex"] == 45  # every input label is null
    again = sum(bool(re.search(ev.CODE_INTENT_REGEX, r["text"])) for r in rows)
    assert summary["n_code_intent_true"] == again


# ───────────────────────────────────────────────────────────────────────────── grep / weighted


def test_grep_terms_are_distinct_lowercase_len3_minus_stopwords():
    t = ev.grep_terms("How does PlotData align the data in the resulting frame? Align it.")
    assert t == ["plotdata", "align", "data", "resulting", "frame"]
    # identifier parts come from the shipped tokenizer; stopwords and short tokens are dropped
    assert ev.grep_terms("ggml_vec_dot is on a CPU") == [
        "ggml_vec_dot",
        "ggml",
        "vec",
        "dot",
        "cpu",
    ]
    assert ev.grep_terms("the of and") == []
    assert (
        hashlib.sha256("\n".join(sorted(ev.STOPWORDS)).encode()).hexdigest() == ev.STOPWORDS_SHA256
    )


def test_grep_rank_scores_distinct_terms_orders_by_chunk_and_drops_zero():
    lowered = ["alpha beta", "beta gamma delta", "zzz", "gamma", "beta gamma"]
    assert ev.grep_rank(lowered, ["beta", "gamma"]) == [1, 4, 0, 3]  # 2,2 (chunk order), then 1,1
    assert ev.grep_rank(lowered, ["beta", "gamma"], k=2) == [1, 4]
    assert ev.grep_rank(lowered, ["nothing"]) == []
    assert ev.grep_rank(lowered, []) == []
    assert ev.grep_rank(["gamma delta"], ["gam", "elt"]) == [0]  # substrings, not words


def test_weighted_rrf_known_answers_and_tie_break():
    # A: .75/61 + .25/62, B: .75/62 + .25/61 -> A first
    assert ev.weighted_rrf([(0.75, ["A", "B", "C"]), (0.25, ["B", "A"])]) == ["A", "B", "C"]
    # the bm25 side can win when its weight is large enough
    assert ev.weighted_rrf([(0.25, ["A", "B"]), (0.75, ["B", "A"])]) == ["B", "A"]
    # equal scores break on str(id), as rrf_fuse does ("10" < "9")
    assert ev.weighted_rrf([(0.5, [10, 9]), (0.5, [9, 10])]) == [10, 9]
    assert ev.weighted_rrf([(0.75, list(range(30))), (0.25, [])], limit=5) == [0, 1, 2, 3, 4]


# ───────────────────────────────────────────────────────────────────────────── exact recall vectors


def test_exact_recall_vectors_for_grep_bm25_dense_hybrid(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    store = ev.ResultStore(out)
    for arm in ev.LEXICAL_ARMS:
        ev.run_lexical_arm(arm, w.rows, w.store, store)
    run_dense(out, w, "granite-97m-r2")
    run_dense(out, w, "bge-large")
    for group in ("probe45", "spill_follow"):
        # q1..q6 of the synthetic world (see ev.synthetic_queries)
        assert vec(out, w, "grep", group)[0] == [1, 0, 1, 0, 1, 1]
        assert vec(out, w, "bm25", group)[0] == [1, 0, 1, 0, 1, 1]
        for a in ("granite-97m-r2", "bge-large"):
            assert vec(out, w, f"{a}.dense", group, 1)[0] == [1, 1, 0, 0, 1, 1]
            assert vec(out, w, f"{a}.hybrid", group, 1)[0] == [1, 1, 1, 0, 1, 1]
            # the 6-chunk bundle returns every chunk, so dense recall@10 only misses the impossible gold
            assert vec(out, w, f"{a}.dense", group, 10)[0] == [1, 1, 1, 0, 1, 1]
        assert vec(out, w, "bm25", group, 10)[0] == [1, 0, 1, 0, 1, 1]
    # nothing failed anywhere
    for v in store.variants():
        assert all(not r["failed"] for r in store.rows(v).values())


def test_pointers_carry_chunk_line_ranges_and_straddling_gold_hits_both_chunks(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    ev.run_lexical_arm("bm25", w.rows, w.store, ev.ResultStore(out))
    rows = ev.ResultStore(out).rows("bm25")
    assert rows["s-q1"]["pointers"][0][1:] == [1, 8]
    assert rows["s-q3"]["pointers"][0][1:] == [9, 16]
    # q5 gold = lines 8-9 straddles the boundary: the BM25 chunk (lines 1-8) hits through line 8,
    # and a pointer for lines 9-16 of the same file would hit through line 9
    assert st.pointer_hits(
        tuple(rows["s-q5"]["pointers"][0]), [tuple(g) for g in w.rows[4 + 6]["gold_spans"]]
    )
    p = rows["s-q5"]["pointers"][0][0]
    gold = [tuple(g) for g in w.rows[10]["gold_spans"]]
    assert st.pointer_hits((p, 9, 16), gold) and st.pointer_hits((p, 1, 8), gold)
    assert not st.pointer_hits((p, 10, 16), gold)
    assert not st.pointer_hits(("other.txt", 1, 16), gold)


def test_dense_only_reuses_the_search_query_vector_and_bge_prefixes_only_queries(tmp_path):
    w = make_world(tmp_path)
    q = next(r for r in w.rows if r["id"] == "s-q1")
    ad = fake("bge-large")
    idx = cs.ContextSearchIndex(w.store.get(q["bundle"]).sections)
    assert ev.build_dense(idx, ad)["ok"]
    doc_calls = [t for s, t in ad.seen if s == "document"]
    n_before = len(ad.seen)
    tap = ev.QueryTap(ad)
    res = ev.dense_query(idx, tap, ad, q, weighted=False)
    queries = [t for s, t in ad.seen[n_before:] if s == "query"]
    assert len(queries) == 1  # ONE query embedding feeds hybrid and dense-only
    assert queries[0][0] == ev.bge_query_prefix() + q["text"]
    assert all(not t.startswith("Represent this sentence") for texts in doc_calls for t in texts)
    ptrs, _, _ = res["dense"]
    ranking = [r["id"] for r in idx._dense_ranking(tap.last[0].vectors[0], None)]
    assert ptrs == [ev.chunk_pointer(idx.chunks[i]) for i in ranking[:10]]
    assert set(res) == {"dense", "hybrid"}  # no weighted variant for a non-static arm
    # granite sends nothing extra
    ad2 = fake("granite-97m-r2")
    idx2 = cs.ContextSearchIndex(w.store.get(q["bundle"]).sections)
    ev.build_dense(idx2, ad2)
    ev.dense_query(idx2, ev.QueryTap(ad2), ad2, q, weighted=False)
    assert [t for s, t in ad2.seen if s == "query"][0] == [q["text"]]


def test_code_intent_reaches_the_adapter_per_query(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    ad, _ = run_dense(out, w, "eg2-gguf-bf16-768")
    # 12 queries, only the two q6 (backticks) are code-intent; restored after each query
    assert sorted(ad.intents) == [False] * 10 + [True] * 2
    assert ad.code_intent is False
    seen_q = [t[0] for s, t in ad.seen if s == "query"]
    assert sum(t.startswith(arms.EG2_CODE_QUERY_PROMPT) for t in seen_q) == 2
    assert sum(t.startswith(arms.EG2_QUERY_PROMPT) for t in seen_q) == 10


def test_weighted_hybrid_exists_only_for_static_arms(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    run_dense(out, w, "potion-code-16m-v2")
    run_dense(out, w, "granite-97m-r2")
    names = ev.ResultStore(out).variants()
    assert "potion-code-16m-v2.whybrid" in names and "granite-97m-r2.whybrid" not in names
    assert vec(out, w, "potion-code-16m-v2.whybrid")[0] == [1, 1, 1, 0, 1, 1]
    assert vec(out, w, "potion-code-16m-v2.dense")[0] == [1, 1, 0, 0, 1, 1]


# ───────────────────────────────────────────────────────────────────────────── failures


def test_failing_embedder_counts_every_query_as_a_miss(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    ad, rep = run_dense(out, w, "bge-large", adapter=fake("bge-large", fail_docs=True))
    assert rep["n_failed_hybrid"] == len(w.rows)
    for variant in ("bge-large.dense", "bge-large.hybrid"):
        rows = ev.ResultStore(out).rows(variant)
        assert len(rows) == len(w.rows) and all(r["failed"] for r in rows.values())
        assert all(r["reason"].startswith("index_build:") for r in rows.values())
        v, sc = vec(out, w, variant, "spill_follow", 10)
        assert v == [0] * 6 and sc["n_failed"] == 6 and sc["n"] == 6
    meta = json.loads((out / "armmeta-bge-large.json").read_text())
    assert meta["bundles"]["spill-b1"]["ok"] is False
    # lexical arms are unaffected by the dense failure
    ev.run_lexical_arm("bm25", w.rows, w.store, ev.ResultStore(out))
    assert vec(out, w, "bm25")[0] == [1, 0, 1, 0, 1, 1]


def test_query_side_failure_is_a_miss_for_that_query_only(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    ad, rep = run_dense(
        out, w, "granite-97m-r2", adapter=fake("granite-97m-r2", fail_query_containing="slate")
    )
    rows = ev.ResultStore(out).rows("granite-97m-r2.hybrid")
    failed = sorted(q for q, r in rows.items() if r["failed"])
    assert failed == ["p-q4", "s-q4"]  # the only query mentioning 'slate'
    assert rows["s-q4"]["mode"].startswith("lexical_fallback:query_")
    d_rows = ev.ResultStore(out).rows("granite-97m-r2.dense")
    assert sorted(q for q, r in d_rows.items() if r["failed"]) == failed  # same MISS in dense-only
    # q4 is unanswerable anyway; everyone else is untouched
    assert vec(out, w, "granite-97m-r2.hybrid")[0] == [1, 1, 1, 0, 1, 1]
    assert vec(out, w, "granite-97m-r2.hybrid")[1]["n_failed"] == 1
    assert rep["n_failed_hybrid"] == 2


def test_nonfinite_vectors_are_invalid_not_silent_zeros(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    class NaNAdapter(ev.HashAdapter):
        def _encode(self, texts):
            v = super()._encode(texts)
            v[0, 1] = np.nan
            return v

    ad = NaNAdapter(SPECS["granite-97m-r2"])
    ev.run_dense_arm(
        ev.DENSE_ARMS["granite-97m-r2"],
        ad,
        w.rows,
        w.store,
        ev.ResultStore(out),
        ev.ArmMeta(out, "granite-97m-r2"),
    )
    assert ad.nonfinite_rows > 0
    assert all(r["failed"] for r in ev.ResultStore(out).rows("granite-97m-r2.hybrid").values())
    meta = json.loads((out / "armmeta-granite-97m-r2.json").read_text())
    assert meta["invocations"][0]["adapter_nonfinite_rows"] > 0
    assert meta["bundles"]["spill-b1"]["reason"] == "degenerate"


# ───────────────────────────────────────────────────────────────────────────── resume


def test_lexical_resume_is_a_noop_and_repairs_torn_rows(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    store = ev.ResultStore(out)
    assert ev.run_lexical_arm("grep", w.rows, w.store, store)["n_run"] == 12
    before = store.path("grep").read_bytes()
    assert ev.run_lexical_arm("grep", w.rows, w.store, store)["n_run"] == 0
    assert store.path("grep").read_bytes() == before
    # an interrupted write leaves a torn last line: it is ignored, re-run, never glued to
    lines = before.decode().splitlines(keepends=True)
    store.path("grep").write_text("".join(lines[:-1]) + lines[-1][:30])
    assert len(store.rows("grep")) == 11
    assert ev.run_lexical_arm("grep", w.rows, w.store, store)["n_run"] == 1
    assert len(store.rows("grep")) == 12
    # the torn fragment was dropped, not glued to: every line of the repaired file is valid JSON
    text = store.path("grep").read_text()
    assert text.endswith("\n") and len(text.splitlines()) == 12
    assert all(isinstance(json.loads(line), dict) for line in text.splitlines())


def test_dense_resume_embeds_nothing_and_fills_only_missing_variants(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    ad1, _ = run_dense(out, w, "potion-base-8m")
    assert any(s == "document" for s, _ in ad1.seen)
    files = {v: ev.ResultStore(out).path(v).read_bytes() for v in ev.ResultStore(out).variants()}
    ad2, rep = run_dense(out, w, "potion-base-8m")
    assert rep["n_run"] == 0 and ad2.seen == []  # no bundle rebuilt, no query embedded
    assert files == {
        v: ev.ResultStore(out).path(v).read_bytes() for v in ev.ResultStore(out).variants()
    }
    # drop the last DENSE row only: exactly that query is re-run, and only the dense row is written
    p = ev.ResultStore(out).path("potion-base-8m.dense")
    lines = p.read_text().splitlines(keepends=True)
    p.write_text("".join(lines[:-1]))
    ad3, rep = run_dense(out, w, "potion-base-8m")
    assert rep["n_run"] == 1
    assert len([1 for s, _ in ad3.seen if s == "query"]) == 1
    assert len(p.read_text().splitlines()) == 12
    assert (
        ev.ResultStore(out).path("potion-base-8m.hybrid").read_bytes()
        == files["potion-base-8m.hybrid"]
    )
    meta = json.loads((out / "armmeta-potion-base-8m.json").read_text())
    assert len(meta["invocations"]) == 3


def test_retry_failed_replaces_failures_and_default_resume_keeps_them(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    run_dense(out, w, "bge-large", adapter=fake("bge-large", fail_docs=True))
    assert vec(out, w, "bge-large.hybrid", "spill_follow", 10)[0] == [0] * 6
    run_dense(out, w, "bge-large")  # healthy adapter, but failures are "finished" by default
    assert vec(out, w, "bge-large.hybrid", "spill_follow", 10)[0] == [0] * 6
    run_dense(out, w, "bge-large", retry_failed=True)
    assert vec(out, w, "bge-large.hybrid", "spill_follow", 1)[0] == [1, 1, 1, 0, 1, 1]
    assert vec(out, w, "bge-large.hybrid")[1]["n_failed"] == 0  # last row per query wins


def test_offline_budget_is_applied_during_the_run_and_restored_after(tmp_path):
    before = (cs.INDEX_TIMEOUT_MIN_S, cs.INDEX_TIMEOUT_MAX_S)
    with ev.offline_index_budget():
        assert cs.index_build_timeout_s(10**7) == ev.OFFLINE_INDEX_TIMEOUT_S == 1e9
        assert cs.index_build_timeout_s(0) == 1e9
    assert (cs.INDEX_TIMEOUT_MIN_S, cs.INDEX_TIMEOUT_MAX_S) == before
    assert cs.index_build_timeout_s(0) == before[0]
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    seen = []

    class Spy(ev.HashAdapter):
        def encode_side(self, texts, side):
            seen.append(cs.index_build_timeout_s(1000))
            return super().encode_side(texts, side)

    run_dense(out, w, "granite-97m-r2", adapter=Spy(SPECS["granite-97m-r2"]))
    assert seen and set(seen) == {1e9}
    assert (cs.INDEX_TIMEOUT_MIN_S, cs.INDEX_TIMEOUT_MAX_S) == before


# ───────────────────────────────────────────────────────────────────────────── bundles


def test_probe_bundle_is_package_py_files_plus_outside_gold_with_commit_check(tmp_path):
    inp = ev.make_synthetic_inputs(tmp_path / "in")
    co = inp["checkouts"] / "seaborn"
    (co / "tests").mkdir()
    (co / "tests" / "extra.py").write_text("x = 1\n")
    (co / "seaborn" / "notpy.txt").write_text("ignored\n")
    (co / "seaborn" / "sub").mkdir()
    (co / "seaborn" / "sub" / "deep.py").write_text("y = 2\n")
    b = ev.build_probe_bundle(
        "mwaskom/seaborn",
        inp["checkouts"],
        ["tests/extra.py", "seaborn/alpha.py"],
        expected_commit="0" * 40,
        head_fn=FAKE_HEAD,
    )
    names = [s.name for s in b.sections]
    assert (
        names
        == sorted(names)
        == [
            "seaborn/alpha.py",
            "seaborn/beta.py",
            "seaborn/gamma.py",
            "seaborn/sub/deep.py",
            "tests/extra.py",
        ]
    )
    assert all(
        f["sha256"] == hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest() for f in b.files
    )
    with pytest.raises(RuntimeError, match="HEAD"):
        ev.build_probe_bundle(
            "mwaskom/seaborn", inp["checkouts"], [], expected_commit="1" * 40, head_fn=FAKE_HEAD
        )
    with pytest.raises(FileNotFoundError):
        ev.build_probe_bundle("mwaskom/seaborn", inp["checkouts"], ["nope.py"], head_fn=FAKE_HEAD)
    # package roots follow the predeclared table
    assert {k: v["package_root"] for k, v in ev.PROBE_REPOS.items()} == {
        "mwaskom/seaborn": "seaborn",
        "pylint-dev/pylint": "pylint",
        "pytest-dev/pytest": "src/_pytest",
    }


def test_bundle_store_asserts_the_chunk_ceiling_and_records_counts(tmp_path, monkeypatch):
    w = make_world(tmp_path)
    b = w.store.get("spill-b1")
    w.store.get("probe45:mwaskom/seaborn")
    assert w.store.summary["spill-b1"]["n_chunks"] == 6 and len(b.files) == 3
    assert w.store.summary["probe45:mwaskom/seaborn"]["n_chunks"] == 6
    monkeypatch.setattr(cs, "MAX_DENSE_CHUNKS", 5)
    fresh = ev.make_bundle_store(
        w.rows,
        bundles_json=w.root / "bundles.json",
        probe_checkouts=w.inp["checkouts"],
        head_fn=FAKE_HEAD,
    )
    with pytest.raises(AssertionError, match="MAX_DENSE_CHUNKS"):
        fresh.get("spill-b1")


def test_gold_check_errors_on_foreign_paths_and_warns_on_gold_past_eof(tmp_path):
    w = make_world(tmp_path)
    gc = ev.validate_gold(w.rows, w.store)
    assert gc["errors"] == []
    assert sorted(x.split(":")[0] for x in gc["warnings"]) == [
        "p-q4",
        "s-q4",
    ]  # gold lines 17-20 of a 16-line file
    bad = [dict(r) for r in w.rows]
    bad[0]["gold_spans"] = [["seaborn/not_in_bundle.py", 1, 2]]
    errs = ev.validate_gold(bad, w.store)["errors"]
    assert len(errs) == 1 and "not a section of bundle" in errs[0]
    bad[1]["scored"] = False
    bad[1]["gold_spans"] = [["nowhere.py", 1, 2]]
    assert len(ev.validate_gold(bad, w.store)["errors"]) == 1  # unscored rows are not checked


def test_freeze_refuses_gold_outside_its_bundle(tmp_path):
    w = make_world(tmp_path)
    rows = ev.read_jsonl(w.root / "queries-trace.jsonl")
    rows[0]["gold"] = {"path": "/no/such/spill.txt", "ranges": [[1, 2]]}
    (w.root / "queries-trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="gold check failed"):
        _freeze(w, tmp_path / "out")


def test_miner_chunk_counts_are_cross_checked_not_enforced(tmp_path):
    w = make_world(tmp_path)
    w.store.get("spill-b1")
    assert w.store.summary["spill-b1"]["chunk_count_mismatch_vs_miner"] == []
    data = json.loads((w.root / "bundles.json").read_text())
    data["spill-b1"]["files"][0]["chunks"] = 5
    (w.root / "bundles.json").write_text(json.dumps(data))
    store = ev.make_bundle_store(
        w.rows,
        bundles_json=w.root / "bundles.json",
        probe_checkouts=w.inp["checkouts"],
        head_fn=FAKE_HEAD,
    )
    store.get("spill-b1")
    assert store.summary["spill-b1"]["chunk_count_mismatch_vs_miner"] == [
        data["spill-b1"]["files"][0]["path"]
    ]


def test_spill_bundle_rejects_a_changed_file(tmp_path):
    w = make_world(tmp_path)
    entry = json.loads((w.root / "bundles.json").read_text())["spill-b1"]
    ev.build_spill_bundle("spill-b1", entry)
    Path(entry["files"][0]["path"]).write_text("tampered\n")
    with pytest.raises(RuntimeError, match="sha256"):
        ev.build_spill_bundle("spill-b1", entry)


def test_corpus_listing_is_sorted_pairs_and_digest_is_its_sha256(tmp_path):
    w = make_world(tmp_path)
    b = w.store.get("probe45:mwaskom/seaborn")
    text = ev.corpus_listing([b])
    lines = text.splitlines()
    assert lines == sorted(lines) and len(lines) == 3
    assert lines[0].split("\t")[:2] == ["probe45:mwaskom/seaborn", "seaborn/alpha.py"]
    assert (
        w.store.summary[b.bundle_id]["corpus_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    )


# ───────────────────────────────────────────────────────────────────────────── registry


def test_registry_has_the_six_arms_no_st_fp32_and_the_right_prefix_mechanics():
    assert (
        list(SPECS)
        == list(ev.DENSE_ARMS)
        == [
            "bge-large",
            "granite-97m-r2",
            "eg2-gguf-bf16-768",
            "eg2-gguf-bf16-256",
            "potion-code-16m-v2",
            "potion-base-8m",
        ]
    )
    assert not any("st-fp32" in a or "-st-" in a for a in SPECS)  # operator ruling: ST FP32 dropped
    ev.validate_registry(SPECS)
    bge = SPECS["bge-large"]
    assert "bge-large-en-v1.5" in bge.model_id().lower()
    assert (
        cs.query_text(bge.model_id(), "find x")
        == "Represent this sentence for searching relevant passages: find x"
    )
    assert (bge.dim, bge.native_dim, bge.dtype) == (1024, 1024, "f16")
    assert ev.BGE_GGUF_SHA256 in bge.revision
    assert SPECS["granite-97m-r2"].dim == 384
    assert SPECS["eg2-gguf-bf16-256"].dim == 256 and SPECS["eg2-gguf-bf16-256"].native_dim == 768
    assert {a: d.registered for a, d in ev.DENSE_ARMS.items()} == {
        "bge-large": True,
        "granite-97m-r2": True,
        "eg2-gguf-bf16-768": False,
        "eg2-gguf-bf16-256": False,
        "potion-code-16m-v2": False,
        "potion-base-8m": False,
    }
    assert [a for a, d in ev.DENSE_ARMS.items() if d.weighted] == [
        "potion-code-16m-v2",
        "potion-base-8m",
    ]
    # identities are distinct and sensitive to the runtime string
    assert len({s.model_id() for s in SPECS.values()}) == 6
    other = ev.eval_specs("llama-other", "m2v-test")
    assert other["bge-large"].model_id() != SPECS["bge-large"].model_id()


def test_manifest_cross_check_and_runtimes():
    manifest = {
        "arms": [
            {
                "arm_id": "eg2-gguf-bf16-768",
                "dense_embedder_model_id": SPECS["eg2-gguf-bf16-768"].model_id(),
                "runtime": "llama-test",
            },
            {
                "arm_id": "potion-code-16m-v2",
                "dense_embedder_model_id": "potion-code-16m-v2#deadbeef0000",
                "runtime": "m2v-test",
            },
        ]
    }
    assert ev.runtimes_from_manifest(manifest) == ("llama-test", "m2v-test")
    problems = ev.check_against_manifest(SPECS, manifest)
    assert len(problems) == 1 and "potion-code-16m-v2" in problems[0]


def test_llama_adapter_dimension_probe(monkeypatch):
    calls = []

    def fake_encode(self, texts):
        calls.append(self.base_url)
        return np.ones((len(texts), FAKE_WIDTH[0]), dtype=np.float32)

    monkeypatch.setattr(arms.LlamaServerAdapter, "_encode", fake_encode)
    FAKE_WIDTH = [512]
    ad = ev.resolve_llama_adapter(
        ev.DENSE_ARMS["granite-97m-r2"], SPECS["granite-97m-r2"], "http://127.0.0.1:9", timeout_s=1
    )
    assert (ad.spec.dim, ad.spec.native_dim) == (512, 512)  # Granite reads the real dimension
    assert ad.model_id != SPECS["granite-97m-r2"].model_id()  # ... and its identity follows
    FAKE_WIDTH[0] = 384
    assert (
        ev.resolve_llama_adapter(
            ev.DENSE_ARMS["granite-97m-r2"], SPECS["granite-97m-r2"], "http://x", timeout_s=1
        ).spec.dim
        == 384
    )
    FAKE_WIDTH[0] = 768
    with pytest.raises(RuntimeError, match="1024"):
        ev.resolve_llama_adapter(
            ev.DENSE_ARMS["bge-large"], SPECS["bge-large"], "http://x", timeout_s=1
        )
    ok = ev.resolve_llama_adapter(
        ev.DENSE_ARMS["eg2-gguf-bf16-256"], SPECS["eg2-gguf-bf16-256"], "http://x", timeout_s=1
    )
    assert ok.spec.dim == 256  # EG2@256: served 768-d, truncated client-side
    assert ev.semble_status().startswith(("not_run", "available_but_not_run"))


# ───────────────────────────────────────────────────────────────────────────── ColGREP


def test_colgrep_command_env_and_parsers(tmp_path):
    cmd = ev.colgrep_search_cmd(ev.COLGREP_BIN, "how does x work", "/w/seaborn")
    assert cmd == [
        "/mnt/raid0/llm/UTILS/bin/colgrep-1.2.0",
        "search",
        "how does x work",
        "-k",
        "10",
        "--alpha",
        "0.95",
        "--json",
        "-y",
        "/w/seaborn",
    ]
    assert ev.colgrep_init_cmd("b", "/w") == ["b", "init", "-y", "/w"]
    assert ev.colgrep_env({"A": "1"}) == {"A": "1", "NEXT_PLAID_FORCE_CPU": "1"}
    nested = json.dumps(
        [
            {
                "score": 1.0,
                "unit": {
                    "file": "/w/seaborn/_core/data.py",
                    "line": 15,
                    "end_line": 260,
                    "name": "PlotData",
                },
            }
        ]
    )
    flat = json.dumps(
        [
            {"path": "seaborn/_core/data.py", "start_line": 3, "end_line": 9},
            {"path": "./x.py", "start_line": 1, "end_line": 1},
        ]
    )
    assert ev.parse_colgrep_json(nested, "/w") == [("seaborn/_core/data.py", 15, 260)]
    assert ev.parse_colgrep_json(flat, "/w") == [("seaborn/_core/data.py", 3, 9), ("x.py", 1, 1)]
    assert ev.parse_colgrep_json("", "/w") == [] and ev.parse_colgrep_json("[{}]", "/w") == []
    assert (
        len(
            ev.parse_colgrep_json(
                json.dumps([{"path": "a", "start_line": 1, "end_line": 2}] * 30), "/w"
            )
        )
        == 10
    )


def test_colgrep_dry_run_prints_commands_and_runs_nothing(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    printed = []

    def boom(*a, **k):
        raise AssertionError("dry run must not execute anything")

    rep = ev.run_colgrep(
        w.rows,
        w.store,
        ev.ResultStore(out),
        ev.ArmMeta(out, "colgrep"),
        workdir_root=tmp_path / "work",
        dry_run=True,
        runner=boom,
        echo=printed.append,
    )
    assert rep["dry_run"] is True and rep["n_run"] == 0
    assert printed[0] == "# env NEXT_PLAID_FORCE_CPU=1" and "init -y" in printed[1]
    searches = [p for p in printed if " search " in p]
    assert len(searches) == 6  # probe45 queries only
    assert not (tmp_path / "work").exists() and not ev.ResultStore(out).variants()
    meta = json.loads((out / "armmeta-colgrep.json").read_text())
    assert meta["not_applicable"]["n_queries"] == 6 and meta["not_applicable"]["groups"] == [
        "spill_follow"
    ]
    assert ".txt" in meta["not_applicable"]["reason"]


def test_colgrep_real_path_with_a_fake_runner(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    workdir = tmp_path / "work"
    seen = []

    def runner(cmd, **kw):
        seen.append((cmd, kw))
        assert kw["env"]["NEXT_PLAID_FORCE_CPU"] == "1" and kw["stdin"] == subprocess.DEVNULL
        if cmd[1] == "init":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        q = cmd[2]
        if "slate" in q:
            return SimpleNamespace(returncode=3, stdout="", stderr="boom")
        if "granite" in q:
            raise subprocess.TimeoutExpired(cmd, 1)
        unit = {"file": str(Path(cmd[-1]) / "seaborn" / "alpha.py"), "line": 1, "end_line": 8}
        return SimpleNamespace(
            returncode=0, stdout=json.dumps([{"score": 0.9, "unit": unit}]), stderr=""
        )

    rep = ev.run_colgrep(
        w.rows,
        w.store,
        ev.ResultStore(out),
        ev.ArmMeta(out, "colgrep"),
        workdir_root=workdir,
        runner=runner,
    )
    assert rep["n_run"] == 6 and rep["n_not_applicable"] == 6
    rows = ev.ResultStore(out).rows("colgrep")
    assert set(rows) == {f"p-q{i}" for i in range(1, 7)}  # spill-follow queries get NO row
    assert rows["p-q1"]["pointers"] == [["seaborn/alpha.py", 1, 8]]
    assert rows["p-q4"]["failed"] and rows["p-q4"]["reason"] == "exit_3"
    assert rows["p-q5"]["failed"] and rows["p-q5"]["reason"] == "timeout"
    # the bundle's files were materialised under the per-repo workdir (hard link or copy)
    linked = workdir / "mwaskom__seaborn" / "seaborn" / "alpha.py"
    assert (
        linked.read_bytes()
        == (w.inp["checkouts"] / "seaborn" / "seaborn" / "alpha.py").read_bytes()
    )
    n_init = sum(1 for c, _ in seen if c[1] == "init")
    assert n_init == 1
    # resumable
    rep = ev.run_colgrep(
        w.rows,
        w.store,
        ev.ResultStore(out),
        ev.ArmMeta(out, "colgrep"),
        workdir_root=workdir,
        runner=runner,
    )
    assert rep["n_run"] == 0


def test_colgrep_init_failure_is_recorded_and_a_missing_binary_stops_the_run(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    def runner(cmd, **kw):
        if cmd[1] == "init":
            raise subprocess.TimeoutExpired(cmd, 1)
        return SimpleNamespace(returncode=2, stdout="", stderr="no index")

    ev.run_colgrep(
        w.rows,
        w.store,
        ev.ResultStore(out),
        ev.ArmMeta(out, "colgrep"),
        workdir_root=tmp_path / "work",
        runner=runner,
    )
    meta = json.loads((out / "armmeta-colgrep.json").read_text())
    assert meta["bundles"]["probe45:mwaskom/seaborn"]["ok"] is False
    assert meta["bundles"]["probe45:mwaskom/seaborn"]["init_error"] == "TimeoutExpired"
    assert all(r["failed"] for r in ev.ResultStore(out).rows("colgrep").values())
    with pytest.raises(SystemExit, match="binary not found"):
        ev.run_colgrep(
            w.rows,
            w.store,
            ev.ResultStore(tmp_path / "o2"),
            ev.ArmMeta(out, "colgrep2"),
            workdir_root=tmp_path / "w2",
            bin_path=str(tmp_path / "missing"),
        )


# ───────────────────────────────────────────────────────────────────────────── freeze


def _freeze(w, out, extra=()):
    ap = ev.build_parser()
    args = ap.parse_args(
        [
            "freeze",
            "--out",
            str(out),
            "--probe45",
            str(w.root / "probe45.jsonl"),
            "--trace-queries",
            str(w.root / "queries-trace.jsonl"),
            "--bundles-json",
            str(w.root / "bundles.json"),
            "--rules-md",
            str(w.root / "RULES.md"),
            "--arm-manifest",
            str(w.root / "arm_manifest.json"),
            "--probe-checkouts",
            str(w.inp["checkouts"]),
            "--colgrep-bin",
            str(w.root / "none"),
            *extra,
        ]
    )
    return ev.cmd_freeze(args, head_fn=FAKE_HEAD)


def test_freeze_writes_sha256sum_format_covering_every_required_artifact(tmp_path, capsys):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    assert _freeze(w, out) == 0
    printed = capsys.readouterr().out
    lines = (out / "FROZEN-AT-LAUNCH.sha256").read_text().splitlines()
    assert lines == sorted(lines, key=lambda s: s.split("  ", 1)[1])
    covered = {}
    for line in lines:
        m = re.fullmatch(r"([0-9a-f]{64})  (/.+)", line)
        assert m, line  # sha256sum format with an absolute path
        covered[Path(m.group(2)).name] = m.group(1)
        assert hashlib.sha256(Path(m.group(2)).read_bytes()).hexdigest() == m.group(1)
        assert line in printed  # every digest is printed
    for need in (
        "queries-labelled.jsonl",
        "queries-trace.jsonl",
        "bundles.json",
        "RULES.md",
        "probe45.jsonl",
        "arm_manifest.json",
        "repl_emb_eval.py",
        "repl_emb_stats.py",
        "repl_emb_arms.py",
        "context_search.py",
        "corpus-digest-probe45.txt",
        "predeclared-interpretations.json",
        "gold-check.json",
    ):
        assert need in covered, need
    # the corpus digest is the sha256 of the sorted (path, sha256) listing of the probe45 bundles
    listing = (out / "corpus-digest-probe45.txt").read_text()
    pairs = sorted(
        f"probe45:mwaskom/seaborn\tseaborn/{n}.py\t{hashlib.sha256(t.encode()).hexdigest()}\n"
        for n, t in ((k[:-3], v) for k, v in ev.synthetic_files("py").items())
    )
    assert listing == "".join(pairs)
    assert covered["corpus-digest-probe45.txt"] == hashlib.sha256(listing.encode()).hexdigest()
    assert f"queries-labelled.jsonl sha256 {covered['queries-labelled.jsonl']}" in printed
    assert ev.CODE_INTENT_REGEX in printed
    # the labelled file is what the harness wrote, with the regex source recorded per row
    lab = ev.read_jsonl(out / "queries-labelled.jsonl")
    assert len(lab) == 12 and {r["code_intent_source"] for r in lab} == {"regex"}
    assert ev.verify_freeze(out) == []


def test_verify_freeze_detects_any_change_and_run_refuses(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    _freeze(w, out)
    assert ev.verify_freeze(out) == []
    (w.root / "RULES.md").write_text("changed\n")
    problems = ev.verify_freeze(out)
    assert len(problems) == 1 and "RULES.md" in problems[0]
    args = ev.build_parser().parse_args(["run", "--out", str(out), "--arm", "grep"])
    with pytest.raises(SystemExit, match="freeze check failed"):
        ev.cmd_run(args, head_fn=FAKE_HEAD)
    (w.root / "RULES.md").write_text("synthetic dry-run rules\n")
    assert ev.verify_freeze(out) == []
    empty = tmp_path / "nofreeze"
    empty.mkdir()
    assert "missing" in ev.verify_freeze(empty)[0]


def test_freeze_refuses_to_overwrite_or_to_run_after_arms(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    _freeze(w, out)
    with pytest.raises(SystemExit, match="refreeze"):
        _freeze(w, out)
    assert _freeze(w, out, ["--refreeze"]) == 0
    (out / "perquery-grep.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="per-query results already exist"):
        _freeze(w, out, ["--refreeze"])


# ───────────────────────────────────────────────────────────────────────────── analysis


def _write_world(out, n_probe, n_spill, n_trivial=0, leak=lambda i: None):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, n in (("probe45", n_probe), ("spill_follow", n_spill), ("trivial_head", n_trivial)):
        for i in range(n):
            rows.append(
                {
                    "id": f"{group[0]}{i}",
                    "group": group,
                    "stratum": "intake1357_probe45" if group == "probe45" else group,
                    "scored": True,
                    "code_intent": False,
                    "code_intent_source": "input",
                    "bundle": "b",
                    "text": "t",
                    "gold_spans": [["f.py", 10, 20]],
                    "answer_named_verbatim": leak(len(rows)),
                }
            )
    (out / "queries-labelled.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def _write_variant(out, rows, name, hits, failed=(), skip=(), only_group=None):
    store = ev.ResultStore(out)
    for i, q in enumerate(rows):
        if i in skip or (only_group and q["group"] != only_group):
            continue
        is_failed = i in failed
        ptrs = [] if is_failed else [["f.py", 10, 20]] if i in hits else [["f.py", 100, 110]]
        store.append(
            name,
            {
                "variant": name,
                "qid": q["id"],
                "group": q["group"],
                "stratum": q["stratum"],
                "bundle": "b",
                "code_intent": False,
                "pointers": ptrs,
                "failed": is_failed,
                "latency_ms": 5.0 + i % 3,
            },
        )


def _pass_world(tmp_path, n_probe=45, n_spill=85, leak=lambda i: None):
    out = tmp_path / "a"
    rows = _write_world(out, n_probe, n_spill, leak=leak)
    n = len(rows)
    lex = set(range(0, 60))
    _write_variant(out, rows, "bm25", lex)
    _write_variant(out, rows, "grep", set(range(0, 40)))
    _write_variant(out, rows, "bge-large.hybrid", set(range(0, 50)) | set(range(60, 95)))  # 85 hits
    _write_variant(out, rows, "granite-97m-r2.hybrid", set(range(0, 55)))  # 55 hits: weaker than L
    _write_variant(out, rows, "bge-large.dense", set(range(0, 45)))
    return out, rows, n


def test_analyze_kill_rule_pass_selection_pending_without_a_speed_block(tmp_path):
    out, rows, n = _pass_world(tmp_path)
    res = ev.analyze(out)
    assert res["n_deciding"] == n == 130
    assert res["verdict_lines"] == [f"KILL RULE: PASS (n={n})", "SELECTION: pending speed block"]
    kill = res["kill_rule"]
    assert kill["registered_passers"] == ["bge-large.hybrid"]
    assert kill["registered"]["bge-large.hybrid"]["pass"] is True
    assert kill["registered"]["granite-97m-r2.hybrid"]["pass"] is False  # every result is reported
    assert kill["registered"]["bge-large.hybrid"]["mean_diff"] == pytest.approx(25 / 130)
    assert kill["lexical_baseline"] == "bm25" and res["lexical_baseline"]["L"] == "bm25"
    assert (
        res["selection"]["status"] == "pending speed block" and res["selection"]["selected"] is None
    )
    # bootstrap CIs vs L and vs BGE (recall@5)
    b = res["bootstrap_recall5"]
    assert b["granite-97m-r2.hybrid"]["vs_bge"]["ref"] == "bge-large.hybrid"
    assert b["bge-large.hybrid"]["vs_L"]["ci_lo"] > 0
    assert b["bge-large.dense"]["vs_L"]["L"] == "bm25"
    assert "vs_bge" not in b["bge-large.hybrid"]
    # per stratum recall with Wilson intervals; lexical vs dense vs hybrid; leakage cut
    assert res["recall"]["bm25"]["pooled"][5]["recall"] == pytest.approx(60 / 130)
    assert res["recall"]["bm25"]["probe45"][5]["n"] == 45
    assert res["lexical_vs_dense_vs_hybrid"]["bge-large"]["hybrid"][
        "pooled_recall5"
    ] == pytest.approx(85 / 130)
    assert res["lexical_vs_dense_vs_hybrid"]["bge-large"]["dense"][
        "pooled_recall5"
    ] == pytest.approx(45 / 130)
    assert res["leakage_cut"]["recall5"]["answer_named_verbatim=true"]["n"] == 0  # no flags here
    assert res["cost_flag"] == "contended_not_for_selection"
    cost = res["cost_contended"]["bm25"]
    assert len(cost["latencies_ms"]) == 130 and cost["p95_ms"] is not None
    assert cost["flag"] == "contended_not_for_selection"


def test_analyze_reports_index_build_cost_from_the_arm_meta(tmp_path):
    out, rows, n = _pass_world(tmp_path)
    (out / "armmeta-bge-large.json").write_text(
        json.dumps(
            {
                "arm_id": "bge-large",
                "bundles": {
                    "b1": {"index_build_wall_s": 3.5, "index_build_cpu_s": 2.0, "ok": True},
                    "b2": {"index_build_wall_s": 1.5, "index_build_cpu_s": 1.0, "ok": False},
                },
                "invocations": [{"adapter_nonfinite_rows": 2}, {"adapter_nonfinite_rows": 1}],
            }
        )
    )
    ib = ev.analyze(out)["index_build_contended"]["bge-large"]
    assert (
        ib["wall_s_total"],
        ib["cpu_s_total"],
        ib["n_failed_builds"],
        ib["adapter_nonfinite_rows"],
    ) == (5.0, 3.0, 1, 3)
    assert ib["flag"] == "contended_not_for_selection"


def test_analyze_selection_with_a_speed_block(tmp_path):
    out, rows, n = _pass_world(tmp_path)
    _write_variant(
        out, rows, "potion-code-16m-v2.hybrid", set(range(0, 45)) | set(range(60, 95))
    )  # 80 hits
    _write_variant(
        out, rows, "eg2-gguf-bf16-768.hybrid", set(range(0, 45)) | set(range(60, 85))
    )  # 70 hits
    speed = tmp_path / "speed.json"
    speed.write_text(
        json.dumps(
            {
                "arms": {
                    "bge-large": {
                        "cpu_s_per_query": 0.50,
                        "p95_ms": 400,
                    },  # bare id keys the hybrid
                    "granite-97m-r2.hybrid": {"cpu_s_per_query": 0.10, "p95_ms": 90},
                    "potion-code-16m-v2.hybrid": {"cpu_s_per_query": 0.02, "p95_ms": 8},
                    "eg2-gguf-bf16-768.hybrid": {"cpu_s_per_query": 0.30, "p95_ms": 200},
                }
            }
        )
    )
    res = ev.analyze(out, speed)
    sel = res["selection"]
    # best 85/130 = .654; threshold .604; potion (80/130 = .615) is within X and cheapest
    assert sel["status"] == "selected" and sel["selected"] == "potion-code-16m-v2.hybrid"
    assert (
        "bge-large.hybrid" in sel["candidates"]
        and "eg2-gguf-bf16-768.hybrid" not in sel["candidates"]
    )
    assert res["verdict_lines"][-1] == "SELECTION: potion-code-16m-v2.hybrid"
    # the speed-block capture names the latency metric query_p95_ms: accepted as p95_ms
    speed.write_text(
        json.dumps(
            {
                "bge-large": {"cpu_s_per_query": 0.5, "query_p95_ms": 400},
                "potion-code-16m-v2": {"cpu_s_per_query": 0.5, "query_p95_ms": 9},
                "granite-97m-r2": {"cpu_s_per_query": 0.5, "query_p95_ms": 90},
                "eg2-gguf-bf16-768": {"cpu_s_per_query": 0.5, "query_p95_ms": 200},
            }
        )
    )
    tie = ev.analyze(out, speed)["selection"]
    assert tie["selected"] == "potion-code-16m-v2.hybrid"  # equal CPU: the lowest p95 wins
    # a missing cost row for a candidate keeps the selection pending
    speed.write_text(json.dumps({"bge-large": {"cpu_s_per_query": 0.5, "p95_ms": 1}}))
    res = ev.analyze(out, speed)
    assert res["verdict_lines"][-1].startswith("SELECTION: pending speed block (no cost row for")


def test_analyze_fail_and_selection_not_applicable(tmp_path):
    out = tmp_path / "f"
    rows = _write_world(out, 45, 85)
    _write_variant(out, rows, "bm25", set(range(0, 60)))
    _write_variant(out, rows, "grep", set(range(0, 50)))
    _write_variant(out, rows, "bge-large.hybrid", set(range(0, 68)))  # +8 = 0.06 < M
    _write_variant(out, rows, "granite-97m-r2.hybrid", set(range(0, 62)))
    res = ev.analyze(out)
    assert res["verdict_lines"] == [
        "KILL RULE: FAIL (n=130)",
        "SELECTION: not applicable (kill rule FAIL)",
    ]
    assert res["selection"]["status"] == "not applicable"


def test_analyze_underpowered_below_120_is_a_kill_even_with_a_big_gain(tmp_path):
    out = tmp_path / "u"
    rows = _write_world(out, 45, 55)
    _write_variant(out, rows, "bm25", set(range(0, 30)))
    _write_variant(out, rows, "grep", set(range(0, 20)))
    _write_variant(out, rows, "bge-large.hybrid", set(range(0, 80)))
    _write_variant(out, rows, "granite-97m-r2.hybrid", set(range(0, 70)))
    res = ev.analyze(out)
    assert res["verdict_lines"][0] == "KILL RULE: UNDERPOWERED/KILL (n=100)"
    assert (
        res["kill_rule"]["registered"]["bge-large.hybrid"]["mean_diff"] > 0.10
    )  # evidence still shown


def test_analyze_never_decides_on_partial_data(tmp_path):
    out, rows, n = _pass_world(tmp_path)
    # drop the last 5 rows of the only passing hybrid -> incomplete, so it cannot pass
    p = ev.ResultStore(out).path("bge-large.hybrid")
    p.write_text("".join(p.read_text().splitlines(keepends=True)[:-5]))
    res = ev.analyze(out)
    assert res["completeness"]["bge-large.hybrid"]["n_missing"] == 5
    assert res["verdict_lines"][0] == f"KILL RULE: INCOMPLETE (n={n})"
    # a missing lexical leg is incomplete too
    ev.ResultStore(out).path("grep").unlink()
    res = ev.analyze(out)
    assert (
        res["verdict_lines"][0] == f"KILL RULE: INCOMPLETE (n={n})"
        and res["lexical_baseline"] is None
    )
    # but a pass from the registered arm that IS complete stands even if the other is absent
    out2, rows2, n2 = _pass_world(tmp_path / "again")
    ev.ResultStore(out2).path("granite-97m-r2.hybrid").unlink()
    assert ev.analyze(out2)["verdict_lines"][0] == f"KILL RULE: PASS (n={n2})"


def test_analyze_failures_are_misses_and_counted(tmp_path):
    out, rows, n = _pass_world(tmp_path)
    store = ev.ResultStore(out)
    store.path("bge-large.hybrid").unlink()
    _write_variant(
        out,
        rows,
        "bge-large.hybrid",
        set(range(0, 50)) | set(range(60, 95)),
        failed=set(range(60, 95)),
    )
    res = ev.analyze(out)
    assert res["completeness"]["bge-large.hybrid"]["n_failed"] == 35
    assert res["recall"]["bge-large.hybrid"]["pooled"][5]["recall"] == pytest.approx(50 / 130)
    assert res["verdict_lines"][0] == f"KILL RULE: FAIL (n={n})"


def test_analyze_extension_arm_that_flips_the_verdict_is_flagged_extension_only(tmp_path):
    out = tmp_path / "e"
    rows = _write_world(out, 45, 85)
    _write_variant(out, rows, "bm25", set(range(0, 60)))
    _write_variant(out, rows, "grep", set(range(0, 40)))
    _write_variant(out, rows, "bge-large.hybrid", set(range(0, 60)))
    _write_variant(out, rows, "granite-97m-r2.hybrid", set(range(0, 62)))
    _write_variant(out, rows, "eg2-gguf-bf16-768.hybrid", set(range(0, 50)) | set(range(60, 95)))
    res = ev.analyze(out)
    assert res["verdict_lines"][0] == "KILL RULE: FAIL (n=130)"
    assert (
        "KILL RULE (incl. extension arms): PASS [extension_only: eg2-gguf-bf16-768.hybrid]"
        in res["verdict_lines"][1]
    )
    assert res["kill_rule"]["extension_only"] == ["eg2-gguf-bf16-768.hybrid"]
    assert res["selection"]["status"] == "not applicable"  # the registered verdict governs


def test_analyze_colgrep_probe45_only_and_trivial_head_outside_the_decision(tmp_path):
    out = tmp_path / "c"
    rows = _write_world(out, 45, 85, n_trivial=10)
    # trivial_head rows come last (indices 130..139)
    for name, hits in (
        ("bm25", set(range(0, 60)) | set(range(130, 140))),
        ("grep", set(range(0, 40))),
    ):
        _write_variant(out, rows, name, hits)
    _write_variant(
        out, rows, "bge-large.hybrid", set(range(0, 50)) | set(range(60, 95)) | {131, 132}
    )
    _write_variant(out, rows, "granite-97m-r2.hybrid", set(range(0, 55)))
    _write_variant(out, rows, "colgrep", set(range(0, 30)), only_group="probe45")
    res = ev.analyze(out)
    assert res["n_deciding"] == 130 and res["n_by_group"]["trivial_head"] == 10
    assert res["verdict_lines"][0] == "KILL RULE: PASS (n=130)"  # the 10 trivial rows are not in n
    assert (
        res["completeness"]["colgrep"]["n_applicable"] == 45
        and res["completeness"]["colgrep"]["complete"]
    )
    assert set(res["recall"]["colgrep"]) == {"probe45", "pooled"}
    assert res["bootstrap_recall5"]["colgrep"]["vs_L"]["n"] == 45
    th = res["trivial_head"]
    assert th["bm25"]["n"] == 10 and th["bm25"]["recall"]["5"] == 1.0
    assert th["bge-large.hybrid"]["recall"]["5"] == pytest.approx(0.2)
    assert "colgrep" not in th
    assert "outside the decision" in res["trivial_head_note"]
    # the registered/extension split never includes colgrep
    assert "colgrep" not in json.dumps(res["kill_rule"]["registered"])


def test_cmd_analyze_writes_json_and_markdown(tmp_path, capsys):
    out, rows, n = _pass_world(tmp_path)
    args = ev.build_parser().parse_args(["analyze", "--out", str(out)])
    assert ev.cmd_analyze(args) == 0
    data = json.loads((out / "analysis.json").read_text())
    assert data["verdict_lines"][0] == f"KILL RULE: PASS (n={n})"
    md = (out / "analysis.md").read_text()
    assert f"KILL RULE: PASS (n={n})" in md and "## kill rule" in md and "Wilson" in md
    assert "SELECTION: pending speed block" in capsys.readouterr().out


# ───────────────────────────────────────────────────────────────────────────── end to end


def test_dry_run_end_to_end_without_network(tmp_path, capsys):
    assert ev.main(["--dry-run", "--out", str(tmp_path)]) == 0
    out = tmp_path / "out"
    printed = capsys.readouterr().out
    assert "KILL RULE: UNDERPOWERED/KILL (n=12)" in printed  # 12 < 120: small n kills by design
    assert "SELECTION: not applicable" in printed
    assert (
        "init -y" in printed and "NEXT_PLAID_FORCE_CPU=1" in printed
    )  # colgrep commands only printed
    for need in (
        "FROZEN-AT-LAUNCH.sha256",
        "config-echo.json",
        "invocations.jsonl",
        "analysis.json",
        "analysis.md",
        "queries-labelled.jsonl",
        "armmeta-bge-large.json",
        "perquery-bge-large.hybrid.jsonl",
    ):
        assert (out / need).exists(), need
    assert not (out / "perquery-colgrep.jsonl").exists()
    a = json.loads((out / "analysis.json").read_text())
    assert a["semble"].startswith("not_run (tool not available") or a["semble"].startswith(
        "available"
    )
    assert len(a["variants"]) == 16
    assert all(c["complete"] for c in a["completeness"].values())
    cfg = json.loads((out / "config-echo.json").read_text())
    stable = cfg["stable"]
    assert (
        stable["context_search_sha256"]
        == hashlib.sha256(
            (ev.REPO_ROOT / "src/repl_environment/context_search.py").read_bytes()
        ).hexdigest()
    )
    pre = stable["predeclared"]
    assert pre["offline_overrides"] == {"INDEX_TIMEOUT_MIN_S": 1e9, "INDEX_TIMEOUT_MAX_S": 1e9}
    assert (
        pre["stats"]["M"] == 0.10
        and pre["stats"]["n_min"] == 120
        and pre["stats"]["seed"] == 20261020
    )
    assert set(pre["interpretations"]) >= {
        "a_decision_set",
        "b_registered_hybrids",
        "c_selection",
        "i_dense_hybrid",
    }
    assert set(stable["arm_identities"]) == set(ev.DENSE_ARMS)
    inv = ev.read_jsonl(out / "invocations.jsonl")[0]
    assert (
        inv["index_timeout_override_s"] == 1e9
        and inv["cost_numbers"] == "contended_not_for_selection"
    )
    assert "semble" in inv and "env" in inv and "os_cpu_count" in inv


def test_run_twice_is_a_resume_and_a_changed_config_is_refused(tmp_path):
    out = tmp_path / "out"
    ev.run_synthetic(out, tmp_path / "inputs", echo=lambda s: None)
    files = sorted(out.glob("perquery-*.jsonl"))
    before = {p.name: p.read_bytes() for p in files}
    ap = ev.build_parser()

    def adapter_for(d, spec):
        return ev.HashAdapter(
            spec, concepts=ev.SYNTH_CONCEPTS, query_concepts=ev.SYNTH_QUERY_CONCEPTS
        ), spec

    args = ap.parse_args(
        [
            "run",
            "--out",
            str(out),
            "--arm",
            "bm25",
            "--arm",
            "bge-large",
            "--arm",
            "colgrep",
            "--colgrep-dry-run",
        ]
    )
    ev.cmd_run(args, adapter_for=adapter_for, head_fn=FAKE_HEAD)
    assert {p.name: p.read_bytes() for p in out.glob("perquery-*.jsonl")} == before
    assert len((out / "invocations.jsonl").read_text().splitlines()) == 2
    # a different runtime string changes the arm identities: the stable config no longer matches
    args = ap.parse_args(
        ["run", "--out", str(out), "--arm", "bm25", "--runtime-override", "bge-large=other-runtime"]
    )
    with pytest.raises(SystemExit):
        ev.cmd_run(args, adapter_for=adapter_for, head_fn=FAKE_HEAD)


def test_run_needs_urls_and_model_paths_for_real_arms(tmp_path):
    out = tmp_path / "out"
    ev.run_synthetic(out, tmp_path / "inputs", echo=lambda s: None)
    # a fresh out dir would need a freeze; reuse this one with an arm whose endpoint is missing
    args = ev.build_parser().parse_args(
        ["run", "--out", str(out), "--arm", "bge-large", "--groups", "probe45"]
    )
    with pytest.raises(SystemExit, match="--arm-url bge-large"):
        ev.cmd_run(args, head_fn=FAKE_HEAD)
    args = ev.build_parser().parse_args(["run", "--out", str(out), "--arm", "potion-base-8m"])
    with pytest.raises(SystemExit, match="--model-path potion-base-8m"):
        ev.cmd_run(args, head_fn=FAKE_HEAD)
    args = ev.build_parser().parse_args(["run", "--out", str(out), "--arm", "eg2-st-fp32-768"])
    with pytest.raises(SystemExit, match="unknown arm"):
        ev.cmd_run(args, head_fn=FAKE_HEAD)


def test_isolated_environment_imports_only_numpy_and_the_shipped_modules():
    src = Path(ev.__file__).read_text()
    imports = set(re.findall(r"^(?:import|from) ([A-Za-z_][\w]*)", src, flags=re.M))
    allowed = {
        "argparse",
        "collections",
        "contextlib",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "os",
        "platform",
        "re",
        "shutil",
        "subprocess",
        "sys",
        "tempfile",
        "time",
        "pathlib",
        "typing",
        "numpy",
        "repl_emb_arms",
        "repl_emb_stats",
        "__future__",
    }
    assert imports <= allowed, imports - allowed
    assert os.path.exists(ev.REPO_ROOT / "src" / "trace" / "navigation.py")


# ───────────────────────────────────────────────────────────────────────────── strata, multi-source freeze


def _spill_entry(tmp_path, name):
    return ev._write_spill_dir(tmp_path / name)


def test_trace_group_is_driven_by_the_row_stratum():
    assert ev.trace_group({"stratum": "agent_nav"}) == ("agent_nav", "stratum")
    assert ev.trace_group({"stratum": "synthetic_spill"}) == ("synthetic_spill", "stratum")
    assert ev.trace_group({"stratum": "spill_follow"}) == ("spill_follow", "stratum")
    assert ev.trace_group({"stratum": "trivial_head"}) == ("trivial_head", "stratum")
    # the legacy single-source layout: an unknown label (or none) is spill_follow, and says so
    assert ev.trace_group({"stratum": "spill"}) == ("spill_follow", "legacy_default")
    assert ev.trace_group({}) == ("spill_follow", "legacy_default")


def test_label_queries_keeps_the_three_trace_strata_apart():
    def row(i, stratum, **kw):
        return {
            "id": i,
            "text": "plain words",
            "stratum": stratum,
            "bundle_id": "b",
            "gold": {"path": "/s/1.txt", "ranges": [[1, 4]]},
            **kw,
        }

    probe = [
        {
            "id": "p1",
            "text": "t",
            "repo": "mwaskom/seaborn",
            "gold": {"path": "a.py", "start_line": 1, "end_line": 2},
        }
    ]
    trace = [
        row("n1", "agent_nav", intent_source="assistant_text"),
        row("n2", "agent_nav", intent_source="bash_description"),
        row("y1", "synthetic_spill", writer="A"),
        row("y2", "synthetic_spill", writer="B"),
        row("f1", "spill_follow"),
        row("u1", "something_else"),
    ]
    trivial = [row("h1", "agent_nav", intent_source="assistant_text")]
    rows, summary = ev.label_queries(probe, trace, trivial)
    group = {r["id"]: r["group"] for r in rows}
    assert group == {
        "p1": "probe45",
        "n1": "agent_nav",
        "n2": "agent_nav",
        "y1": "synthetic_spill",
        "y2": "synthetic_spill",
        "f1": "spill_follow",
        "u1": "spill_follow",
        "h1": "trivial_head",
    }
    src = {r["id"]: r["group_source"] for r in rows}
    assert (
        src["u1"] == "legacy_default"
        and src["n1"] == "stratum"
        and src["h1"] == "file"
        and src["p1"] == "file"
    )
    assert [r["id"] for r in rows] == [
        "p1",
        "n1",
        "n2",
        "y1",
        "y2",
        "f1",
        "u1",
        "h1",
    ]  # file order, trivial last
    assert (
        rows[3]["stratum"] == "synthetic_spill" and rows[3]["writer"] == "A"
    )  # pass-through fields survive
    assert rows[-1]["stratum"] == "agent_nav"  # the raw stratum is kept on a trivial-head row
    assert (
        summary["by_group"]["agent_nav"]["n"] == 2 and summary["by_group"]["trivial_head"]["n"] == 1
    )
    assert summary["by_writer"] == {"A": 1, "B": 1}
    assert summary["by_intent_source"] == {"assistant_text": 2, "bash_description": 1}
    assert summary["by_group_source"] == {"file": 2, "stratum": 5, "legacy_default": 1}


def test_load_bundles_json_merges_files_and_rejects_clashing_ids(tmp_path):
    e1, e2, e3 = (
        {
            "files": [{"path": f"/x/{n}", "sha256": "0" * 64, "bytes": 1, "chunks": 1}],
            "total_chunks": 1,
        }
        for n in "abc"
    )
    a, b, c, empty, wrapped = (
        tmp_path / n for n in ("a.json", "b.json", "c.json", "empty.json", "w.json")
    )
    a.write_text(json.dumps({"b1": e1}))
    b.write_text(json.dumps({"b2": e2, "b1": e1}))  # an identical repeat is fine
    c.write_text(json.dumps({"b1": e3}))
    empty.write_text("")  # the miner's empty trace/ source
    wrapped.write_text(json.dumps({"bundles": {"b4": e3}}))
    assert ev.load_bundles_json([a, b, empty, wrapped]) == {"b1": e1, "b2": e2, "b4": e3}
    assert ev.load_bundles_json(a) == {"b1": e1} and ev.load_bundles_json(str(a)) == {"b1": e1}
    assert (
        ev.load_bundles_json(None) == {}
        and ev.load_bundles_json([]) == {}
        and ev.load_bundles_json([empty]) == {}
    )
    with pytest.raises(ValueError, match="different content"):
        ev.load_bundles_json([a, c])


def _freeze_multi(inp, out, extra=()):
    argv = [
        "freeze",
        "--out",
        str(out),
        "--probe45",
        str(inp["root"] / "probe45.jsonl"),
        "--rules-md",
        str(inp["root"] / "RULES.md"),
        "--arm-manifest",
        str(inp["root"] / "arm_manifest.json"),
        "--probe-checkouts",
        str(inp["checkouts"]),
        "--colgrep-bin",
        str(inp["root"] / "none"),
        *extra,
    ]
    argv += [x for f in inp["trace_queries"] for x in ("--trace-queries", str(f))]
    argv += [x for f in inp["trivial_queries"] for x in ("--trivial-queries", str(f))]
    argv += [x for f in inp["bundles_json"] for x in ("--bundles-json", str(f))]
    return ev.cmd_freeze(ev.build_parser().parse_args(argv), head_fn=FAKE_HEAD)


def test_freeze_with_several_trace_sources_covers_every_file_given(tmp_path, capsys):
    inp = ev.make_synthetic_inputs(tmp_path / "in", multi=True)
    out = tmp_path / "out"
    assert _freeze_multi(inp, out) == 0
    covered = {
        Path(line.split("  ", 1)[1])
        for line in (out / "FROZEN-AT-LAUNCH.sha256").read_text().splitlines()
    }
    given = [*inp["trace_queries"], *inp["trivial_queries"], *inp["bundles_json"]]
    assert len(given) == 5
    for f in given:
        assert f.resolve() in covered, f
    assert (
        inp["root"] / "queries-trace.jsonl"
    ).resolve() not in covered  # the legacy file was not given
    lab = ev.read_jsonl(out / "queries-labelled.jsonl")
    assert _count_by(lab, "group") == {
        "probe45": 6,
        "agent_nav": 6,
        "synthetic_spill": 6,
        "trivial_head": 3,
    }
    assert {r["bundle"] for r in lab if r["group"] == "synthetic_spill"} == {"syn-b1"}
    assert {r["writer"] for r in lab if r["group"] == "synthetic_spill"} == {"A", "B"}
    inputs = json.loads((out / "inputs.json").read_text())
    assert (
        len(inputs["trace_queries"]) == 2
        and len(inputs["bundles_json"]) == 2
        and len(inputs["trivial_queries"]) == 1
    )
    assert ev.verify_freeze(out) == []
    # changing ANY of the files given breaks the freeze
    for f in given:
        original = f.read_text()
        f.write_text(original + " ")
        assert any(f.name in p for p in ev.verify_freeze(out)), f
        f.write_text(original)
    assert ev.verify_freeze(out) == []
    assert "by_intent_source" in capsys.readouterr().out


def _count_by(rows, key):
    out = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


def test_freeze_rejects_a_bundle_id_clash_with_different_content_and_allows_none_given(tmp_path):
    inp = ev.make_synthetic_inputs(tmp_path / "in", multi=True)
    clash = tmp_path / "clash.json"
    data = json.loads(inp["bundles_json"][0].read_text())
    ((bid, entry),) = data.items()
    entry["total_chunks"] = 99
    clash.write_text(json.dumps({bid: entry}))
    bad = dict(inp, bundles_json=[*inp["bundles_json"], clash])
    with pytest.raises(ValueError, match="different content"):
        _freeze_multi(bad, tmp_path / "out1")
    # an identical repeat of a file is accepted
    same = dict(inp, bundles_json=[*inp["bundles_json"], inp["bundles_json"][0]])
    assert _freeze_multi(same, tmp_path / "out2") == 0
    # only probe45 (no trace source at all) is a legal, tiny, UNDERPOWERED set
    only = dict(inp, trace_queries=[], trivial_queries=[], bundles_json=[])
    assert _freeze_multi(only, tmp_path / "out3") == 0
    assert {r["group"] for r in ev.read_jsonl(tmp_path / "out3" / "queries-labelled.jsonl")} == {
        "probe45"
    }


def test_multi_source_dry_run_runs_every_stage_and_prints_primary_and_secondary_sets(
    tmp_path, capsys
):
    assert ev.main(["--dry-run", "--multi-source", "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    # primary = probe45 (6, no flag) + agent_nav unflagged (3) + synthetic_spill unflagged (3)
    assert "KILL RULE: UNDERPOWERED/KILL (n=12)" in printed
    assert (
        "FLOOR CHECK FAILED (primary): n=12 < 120; trace rows=6 < 75 -> UNDERPOWERED/KILL"
        in printed
    )
    assert "SELECTION: not applicable (kill rule UNDERPOWERED/KILL)" in printed
    assert (
        "(non-deciding) KILL RULE (pooled, flagged included): UNDERPOWERED/KILL (n=18)" in printed
    )
    assert (
        "(non-deciding) KILL RULE (without synthetic, flagged excluded): UNDERPOWERED/KILL (n=9)"
        in printed
    )
    assert (
        "(non-deciding) KILL RULE (without synthetic, flagged included): UNDERPOWERED/KILL (n=12)"
        in printed
    )
    out = tmp_path / "out"
    a = json.loads((out / "analysis.json").read_text())
    assert a["synthetic_present"] and a["n_deciding"] == 12 and a["n_flagged_excluded"] == 6
    assert a["n_by_set"] == {
        "primary": 12,
        "pooled_flagged_included": 18,
        "without_synthetic_flagged_excluded": 9,
        "without_synthetic_flagged_included": 12,
    }
    assert a["n_by_group"] == {
        "probe45": 6,
        "agent_nav": 6,
        "synthetic_spill": 6,
        "trivial_head": 3,
    }
    assert a["floor_check"]["failures"] == ["n=12 < 120", "trace rows=6 < 75"]
    sets = a["decision_sets"]
    # recall@1 of the synthetic world on the pooled set that INCLUDES flagged rows: lexical 4/6,
    # dense 4/6, hybrid 5/6 in every stratum
    full = sets["pooled_flagged_included"]["recall"]
    for stratum in ("probe45", "agent_nav", "synthetic_spill"):
        assert full["bm25"][stratum]["1"]["recall"] == pytest.approx(4 / 6)
        assert full["bge-large.dense"][stratum]["1"]["recall"] == pytest.approx(4 / 6)
        assert full["bge-large.hybrid"][stratum]["1"]["recall"] == pytest.approx(5 / 6)
    # the primary drops the flagged queries q1, q3, q6 of the trace strata (they are easy for bm25)
    prim = a["recall"]
    assert prim["bm25"]["agent_nav"]["1"]["recall"] == pytest.approx(1 / 3)
    assert prim["bge-large.dense"]["agent_nav"]["1"]["recall"] == pytest.approx(2 / 3)
    assert prim["bge-large.hybrid"]["synthetic_spill"]["1"]["n"] == 3
    assert prim["bm25"]["probe45"]["1"]["n"] == 6  # probe45 is kept whole
    w = a["recall_by_writer"]["bge-large.hybrid"]  # primary: q5 (writer A); q2, q4 (writer B)
    assert w["writer=A"]["1"]["n"] == 1 and w["writer=B"]["1"]["n"] == 2
    assert w["writer=A"]["1"]["recall"] == 1.0 and w["writer=B"]["1"]["recall"] == pytest.approx(
        0.5
    )
    # the leakage effect is visible as a difference: bm25 aces the flagged agent_nav queries
    eff = a["leakage"]["effect_flagged_minus_unflagged"]
    assert eff["bm25"]["agent_nav"]["recall"]["1"]["diff"] == pytest.approx(1.0 - 1 / 3)
    assert eff["bge-large.dense"]["agent_nav"]["recall"]["1"]["diff"] == pytest.approx(0.0)
    assert eff["bge-large.hybrid"]["agent_nav"]["recall"]["1"]["diff"] == pytest.approx(1 / 3)
    rates = a["leakage"]["flag_rates"]
    assert rates["by_writer"]["A"]["flag_rate"] == pytest.approx(2 / 3)
    assert rates["by_writer"]["B"]["flag_rate"] == pytest.approx(1 / 3)
    assert rates["by_intent_source"]["assistant_text"]["flag_rate"] == pytest.approx(2 / 3)
    assert rates["by_stratum"]["probe45"]["flag_rate"] is None  # no flag on probe45
    assert set(a["leakage_cut_intent_source"]["values"]) == {"assistant_text", "bash_description"}
    assert a["trivial_head"]["bm25"]["n"] == 3
    assert "colgrep" not in a["variants"]  # the dry run only prints the ColGREP commands ...
    assert printed.count(" search ") == 6  # ... for the probe45 queries alone
    md = (out / "analysis.md").read_text()
    assert ev.BIAS_NOTE in md and ev.BIAS_NOTE_SCOPE in md
    assert (
        "# PRIMARY decision set" in md
        and "# SECONDARY (non-deciding): pooled, flagged included" in md
    )
    assert "!!! FLOOR CHECK FAILED (primary)" in md
    assert (out / "perquery-bge-large.hybrid.jsonl").exists()
    rows = ev.ResultStore(out).rows("bge-large.hybrid")
    assert {q for q in rows if q.startswith("y-")} == {
        f"y-q{i}" for i in range(1, 7)
    }  # flagged ran too
    assert {q for q in rows if q.startswith("th-")} == {"th-q1", "th-q2", "th-q3"}


# ───────────────────────────────────────────────────────────────────────────── two decision sets


def _write_strata(out, spec, leak=lambda i: None):
    """spec = [(group, n), ...] in row order; writer A/B on synthetic_spill, intent_source on agent_nav."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, n in spec:
        for i in range(n):
            r = {
                "id": f"{group}-{i}",
                "group": group,
                "scored": True,
                "code_intent": False,
                "code_intent_source": "input",
                "bundle": "b",
                "text": "t",
                "stratum": "intake1357_probe45" if group == "probe45" else group,
                "gold_spans": [["f.py", 10, 20]],
                "answer_named_verbatim": leak(len(rows)),
            }
            if group == "synthetic_spill":
                r["writer"] = "AB"[i % 2]
            if group == "agent_nav":
                r["intent_source"] = ("assistant_text", "bash_description")[i % 2]
            rows.append(r)
    (out / "queries-labelled.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def _lines(res):
    return res["verdict_lines"]


def test_primary_passes_while_the_secondary_without_synthetic_set_is_underpowered(tmp_path):
    out = tmp_path / "s1"
    rows = _write_strata(out, [("probe45", 45), ("agent_nav", 40), ("synthetic_spill", 60)])
    assert len(rows) == 145
    _write_variant(out, rows, "bm25", set(range(0, 60)))
    _write_variant(out, rows, "grep", set(range(0, 40)))
    _write_variant(out, rows, "bge-large.hybrid", set(range(0, 50)) | set(range(60, 95)))
    _write_variant(out, rows, "granite-97m-r2.hybrid", set(range(0, 55)))
    res = ev.analyze(out)
    # no flags anywhere: the primary set is the whole pool, and the duplicate secondary sets vanish
    assert _lines(res) == [
        "KILL RULE: PASS (n=145)",
        "SELECTION: pending speed block",
        "(non-deciding) KILL RULE (without synthetic, flagged excluded): UNDERPOWERED/KILL (n=85)",
    ]
    sets = res["decision_sets"]
    assert sets["pooled_flagged_included"] == {"identical_to": "primary"}
    assert sets["without_synthetic_flagged_included"] == {
        "identical_to": "without_synthetic_flagged_excluded"
    }
    prim, wo = sets["primary"], sets["without_synthetic_flagged_excluded"]
    assert (prim["n_deciding"], wo["n_deciding"]) == (145, 85)
    assert prim["deciding"] is True and wo["deciding"] is False
    assert wo["kill_rule"]["underpowered"] is True and wo["kill_rule"]["n_min"] == 120
    # the evidence for the small set is still reported (it just cannot pass)
    assert wo["kill_rule"]["registered"]["bge-large.hybrid"]["mean_diff"] == pytest.approx(15 / 85)
    assert prim["kill_rule"]["registered"]["bge-large.hybrid"]["mean_diff"] == pytest.approx(
        25 / 145
    )
    # top-level keys describe the PRIMARY set
    assert res["kill_rule"] == prim["kill_rule"] and res["n_deciding"] == 145
    assert (
        res["recall"]["bm25"]["pooled"][5]["n"] == 145
        and wo["recall"]["bm25"]["pooled"][5]["n"] == 85
    )
    assert set(res["recall"]["bm25"]) == {"probe45", "agent_nav", "synthetic_spill", "pooled"}
    assert set(wo["recall"]["bm25"]) == {"probe45", "agent_nav", "pooled"}
    assert res["n_by_group"] == {"probe45": 45, "agent_nav": 40, "synthetic_spill": 60}
    # a non-deciding set never selects
    assert wo["selection"]["status"] == "not applicable" and wo["selection_line"] is None


def test_a_secondary_pass_never_overrides_a_failing_primary(tmp_path):
    out = tmp_path / "s2"
    rows = _write_strata(out, [("probe45", 45), ("agent_nav", 90), ("synthetic_spill", 40)])
    assert len(rows) == 175
    synth = set(range(135, 175))
    _write_variant(out, rows, "bm25", set(range(0, 50)) | synth)  # lexical aces the synthetic rows
    _write_variant(out, rows, "grep", set(range(0, 30)))
    _write_variant(
        out, rows, "bge-large.hybrid", set(range(0, 40)) | set(range(50, 85)) | set(range(135, 155))
    )
    _write_variant(out, rows, "granite-97m-r2.hybrid", set(range(0, 45)))
    res = ev.analyze(out)
    assert _lines(res) == [
        "KILL RULE: FAIL (n=175)",
        "SELECTION: not applicable (kill rule FAIL)",
        "(non-deciding) KILL RULE (without synthetic, flagged excluded): PASS (n=135)",
    ]
    sets = res["decision_sets"]
    prim, wo = sets["primary"], sets["without_synthetic_flagged_excluded"]
    assert prim["kill_rule"]["registered"]["bge-large.hybrid"]["mean_diff"] == pytest.approx(
        5 / 175
    )
    assert wo["kill_rule"]["registered"]["bge-large.hybrid"]["mean_diff"] == pytest.approx(25 / 135)
    assert wo["kill_rule"]["registered"]["bge-large.hybrid"]["ci_lo"] > 0
    assert wo["selection"]["status"] == "not applicable"  # the secondary PASS selects nothing
    assert res["selection"]["reason"] == "kill rule FAIL"
    # per-stratum descriptive paired diff vs L: the synthetic stratum is where the hybrid loses
    ps = prim["per_stratum_paired_diff"]
    assert set(ps) == {"probe45", "agent_nav", "synthetic_spill"}
    assert set(wo["per_stratum_paired_diff"]) == {"probe45", "agent_nav"}
    d = ps["synthetic_spill"]["diffs"]["bge-large.hybrid"]
    assert ps["synthetic_spill"]["n"] == 40 and ps["synthetic_spill"]["vs"] == "bm25"
    assert d["mean_diff"] == pytest.approx(-0.5) and d["descriptive"] is True
    assert ps["agent_nav"]["n"] == 90 and "bm25" not in ps["agent_nav"]["diffs"]
    assert ps["probe45"]["diffs"]["bge-large.hybrid"]["mean_diff"] == pytest.approx(-5 / 45)
    assert prim["lexical_baseline"]["L"] == wo["lexical_baseline"]["L"] == "bm25"


def test_recall_by_writer_and_the_intent_source_cut(tmp_path):
    out = tmp_path / "s3"
    rows = _write_strata(out, [("probe45", 45), ("agent_nav", 40), ("synthetic_spill", 40)])
    index = {r["id"]: i for i, r in enumerate(rows)}
    a_ids = {index[f"synthetic_spill-{i}"] for i in range(0, 40, 2)}
    nav_text = {index[f"agent_nav-{i}"] for i in range(0, 40, 2)}
    # bm25 hits every writer-A synthetic row and no writer-B row; agent_nav: only assistant_text rows
    _write_variant(out, rows, "bm25", a_ids | nav_text)
    _write_variant(out, rows, "grep", a_ids)
    _write_variant(out, rows, "bge-large.hybrid", a_ids | nav_text | set(range(0, 20)))
    res = ev.analyze(out)
    w = res["recall_by_writer"]["bm25"]
    assert set(w) == {"writer=A", "writer=B", "pooled"}
    assert w["writer=A"][5]["recall"] == 1.0 and w["writer=B"][5]["recall"] == 0.0
    assert w["writer=A"][5]["n"] == w["writer=B"][5]["n"] == 20 and w["pooled"][5]["n"] == 40
    assert "probe45" not in json.dumps(w)  # the writer split is the synthetic stratum only
    ic = res["leakage_cut_intent_source"]
    assert (
        ic["field"] == "intent_source" and ic["n_unlabelled"] == 45 + 40
    )  # probe45 + synthetic carry none
    assert set(ic["values"]) == {"assistant_text", "bash_description"}
    assert ic["values"]["assistant_text"]["n"] == 20 and ic["values"]["bash_description"]["n"] == 20
    assert ic["values"]["assistant_text"]["recall5"]["bm25"] == 1.0
    assert ic["values"]["bash_description"]["recall5"]["bm25"] == 0.0
    assert (
        "kill_rule" in ic["values"]["assistant_text"]
    )  # evidence on the subset (UNDERPOWERED at n=20)
    assert ic["values"]["assistant_text"]["kill_rule"]["verdict"] == "UNDERPOWERED/KILL"
    # the answer_named_verbatim cut is still produced, per decision set
    assert "answer_named_verbatim=true" in res["leakage_cut"]["recall5"]
    assert "leakage_cut" in res["decision_sets"]["without_synthetic_flagged_excluded"]


def test_no_synthetic_rows_and_no_flags_mean_one_set_and_the_single_set_lines(tmp_path):
    out, rows, n = _pass_world(tmp_path)
    res = ev.analyze(out)
    assert res["synthetic_present"] is False and res["n_flagged_excluded"] == 0
    for key in (
        "pooled_flagged_included",
        "without_synthetic_flagged_excluded",
        "without_synthetic_flagged_included",
    ):
        assert res["decision_sets"][key] == {"identical_to": "primary"}
    assert res["verdict_lines"][0] == f"KILL RULE: PASS (n={n})"
    assert not any("non-deciding" in line for line in res["verdict_lines"])


def test_trivial_head_stays_outside_every_set(tmp_path):
    out = tmp_path / "s4"
    rows = _write_strata(
        out, [("probe45", 45), ("agent_nav", 40), ("synthetic_spill", 40), ("trivial_head", 12)]
    )
    _write_variant(out, rows, "bm25", set(range(0, 125)))
    _write_variant(out, rows, "grep", set(range(0, 20)))
    res = ev.analyze(out)
    assert res["n_deciding"] == 125 and res["n_by_set"]["without_synthetic_flagged_excluded"] == 85
    assert (
        res["trivial_head"]["bm25"]["n"] == 12 and res["trivial_head"]["bm25"]["recall"]["5"] == 0.0
    )
    for ds in res["decision_sets"].values():
        assert "trivial_head" not in ds.get("n_by_group", {})
    assert res["floor_check"]["n"] == 125 and res["floor_check"]["n_trace"] == 80


def test_bias_note_is_verbatim_in_the_header_and_the_json(tmp_path):
    verbatim = (
        "Synthetic queries are LLM-written; they tend to share vocabulary with their source span, "
        "which favours lexical arms. The synthetic stratum therefore makes the kill rule conservative "
        "toward dense arms."
    )
    assert ev.BIAS_NOTE == verbatim
    out = tmp_path / "s5"
    rows = _write_strata(out, [("probe45", 20), ("synthetic_spill", 20)])
    _write_variant(out, rows, "bm25", set(range(0, 10)))
    _write_variant(out, rows, "grep", set(range(0, 5)))
    assert ev.cmd_analyze(ev.build_parser().parse_args(["analyze", "--out", str(out)])) == 0
    md = (out / "analysis.md").read_text()
    head = md.splitlines()[:6]
    assert verbatim in head  # a line of its own in the header, before any table
    assert json.loads((out / "analysis.json").read_text())["bias_note"] == verbatim
    assert ev.PREDECLARED["o_synthetic_bias"] == "(o) " + verbatim
    # also present when there is no synthetic stratum at all (a fixed part of the header)
    out2, _, _ = _pass_world(tmp_path / "again")
    ev.cmd_analyze(ev.build_parser().parse_args(["analyze", "--out", str(out2)]))
    assert verbatim in (out2 / "analysis.md").read_text().splitlines()[:6]


def test_each_set_gets_its_own_incompleteness_and_only_the_primary_selects(tmp_path):
    out = tmp_path / "s6"
    rows = _write_strata(out, [("probe45", 45), ("agent_nav", 90), ("synthetic_spill", 40)])
    _write_variant(out, rows, "bm25", set(range(0, 50)))
    _write_variant(out, rows, "grep", set(range(0, 30)))
    _write_variant(out, rows, "bge-large.hybrid", set(range(0, 40)) | set(range(50, 85)))
    # the synthetic rows were never run for granite: incomplete on the primary (pooled) set,
    # complete on the secondary without-synthetic set
    _write_variant(
        out, rows, "granite-97m-r2.hybrid", set(range(0, 100)), skip=set(range(135, 175))
    )
    speed = tmp_path / "speed.json"
    speed.write_text(
        json.dumps(
            {
                "bge-large": {"cpu_s_per_query": 0.5, "p95_ms": 9},
                "granite-97m-r2": {"cpu_s_per_query": 0.1, "p95_ms": 5},
            }
        )
    )
    res = ev.analyze(out, speed)
    prim = res["decision_sets"]["primary"]
    wo = res["decision_sets"]["without_synthetic_flagged_excluded"]
    assert prim["completeness"]["granite-97m-r2.hybrid"]["n_missing"] == 40
    assert wo["completeness"]["granite-97m-r2.hybrid"]["complete"] is True
    assert "granite-97m-r2.hybrid" in wo["kill_rule"]["registered"]
    assert "granite-97m-r2.hybrid" not in prim["kill_rule"]["registered"]
    assert prim["kill_rule"]["missing_registered"] == ["granite-97m-r2.hybrid"]
    # granite (100 of the first 135 hits) would win the secondary set, but a non-deciding set selects
    # nothing: the single SELECTION line is the primary's, over the arms complete on the primary set
    assert wo["selection_line"] is None
    sel_lines = [ln for ln in _lines(res) if ln.startswith("SELECTION")]
    assert sel_lines == ["SELECTION: bge-large.hybrid"]
    assert res["selection"]["selected"] == "bge-large.hybrid"


# ───────────────────────────────────────────────────────────────────────────── primary set, floors, leakage


def _flagged_world(tmp_path, name="p"):
    """probe45 (45, no flag) + agent_nav (60) + synthetic_spill (300); every third trace row is
    flagged answer_named_verbatim. Lexical hits ALL flagged rows (the verbatim leak) but only a
    quarter of the unflagged ones; the hybrid hits half of each."""
    out = tmp_path / name
    rows = _write_strata(
        out,
        [("probe45", 45), ("agent_nav", 60), ("synthetic_spill", 300)],
        leak=lambda i: (i % 3 == 0) if i >= 45 else None,
    )
    flagged = [i for i, r in enumerate(rows) if r["answer_named_verbatim"] is True]
    unfl = [i for i, r in enumerate(rows) if r["group"] != "probe45" and i not in set(flagged)]
    assert (len(flagged), len(unfl)) == (120, 240)
    bm25 = set(flagged) | set(range(0, 20)) | set(unfl[:60])
    hybrid = set(flagged[:60]) | set(range(0, 35)) | set(unfl[:120])
    _write_variant(out, rows, "bm25", bm25)
    _write_variant(out, rows, "grep", set(list(bm25)[:30]))
    _write_variant(out, rows, "bge-large.hybrid", hybrid)
    _write_variant(out, rows, "granite-97m-r2.hybrid", set(range(0, 40)))
    return out, rows, flagged


def test_the_primary_set_excludes_flagged_rows_and_is_the_only_deciding_one(tmp_path):
    out, rows, flagged = _flagged_world(tmp_path)
    res = ev.analyze(out)
    # primary: lexical 80/285, hybrid 155/285 -> PASS. Pooled WITH the verbatim rows the lexical arm
    # catches up (215 vs 200 of 405) -> FAIL, but that set is non-deciding.
    assert _lines(res) == [
        "KILL RULE: PASS (n=285)",
        "SELECTION: pending speed block",
        "(non-deciding) KILL RULE (pooled, flagged included): FAIL (n=405)",
        "(non-deciding) KILL RULE (without synthetic, flagged excluded): UNDERPOWERED/KILL (n=85)",
        "(non-deciding) KILL RULE (without synthetic, flagged included): UNDERPOWERED/KILL (n=105)",
    ]
    assert res["n_deciding"] == 285 and res["n_flagged_excluded"] == 120
    assert res["n_by_set"] == {
        "primary": 285,
        "pooled_flagged_included": 405,
        "without_synthetic_flagged_excluded": 85,
        "without_synthetic_flagged_included": 105,
    }
    sets = res["decision_sets"]
    assert sets["primary"]["n_by_group"] == {"probe45": 45, "agent_nav": 40, "synthetic_spill": 200}
    assert res["n_by_group"] == {"probe45": 45, "agent_nav": 60, "synthetic_spill": 300}
    prim = sets["primary"]["kill_rule"]["registered"]["bge-large.hybrid"]
    pooled = sets["pooled_flagged_included"]["kill_rule"]["registered"]["bge-large.hybrid"]
    assert prim["mean_diff"] == pytest.approx(75 / 285) and prim["pass"] is True
    assert pooled["mean_diff"] == pytest.approx(15 / 405) and pooled["pass"] is False
    # top-level keys are the PRIMARY set's; L, the bootstrap tables and selection come from it alone
    assert (
        res["kill_rule"] == sets["primary"]["kill_rule"] and res["completeness"]["bm25"]["n"] == 285
    )
    assert res["bootstrap_recall5"]["bge-large.hybrid"]["vs_L"]["n"] == 285
    assert sets["primary"]["deciding"] is True
    for key in list(sets)[1:]:
        ds = sets[key]
        assert ds["deciding"] is False and ds["selection_line"] is None
        assert ds["selection"]["status"] == "not applicable"
        assert all(line.startswith("(non-deciding) ") for line in ds["kill_lines"])
    # a probe45 row is never excluded (it carries no flag); an unflagged-by-absence trace row is kept
    prim_rows = ev.primary_rows(ev.read_jsonl(out / "queries-labelled.jsonl"))
    assert sum(q["group"] == "probe45" for q in prim_rows) == 45
    assert all(q.get("answer_named_verbatim") is not True for q in prim_rows)
    # the extension-flip logic also lives in the primary set: nothing to flip here
    assert res["kill_rule"]["changed_by_extension"] is False


def test_floor_check_function_counts_primary_rows_only(tmp_path):
    def rows(spec):
        out = []
        for group, n, flagged in spec:
            for i in range(n):
                out.append(
                    {
                        "id": f"{group}{i}",
                        "group": group,
                        "scored": True,
                        "answer_named_verbatim": True
                        if i < flagged
                        else (None if group == "probe45" else False),
                    }
                )
        return out

    # the expected real set: 45 + 23 + 278 = 346 total, 301 trace (flagged rows do not count)
    ok = ev.floor_check(
        rows([("probe45", 45, 0), ("agent_nav", 43, 20), ("synthetic_spill", 478, 200)])
    )
    assert (ok["n"], ok["n_trace"], ok["ok"], ok["failures"]) == (346, 301, True, [])
    assert ok["n_flagged_excluded"] == 220 and ok["by_group"]["agent_nav"]["in_primary"] == 23
    assert ev.floor_banner(ok) == "floor check ok: primary n=346 (>= 120), trace rows=301 (>= 75)"
    # n >= 120 but too few trace rows
    low_trace = ev.floor_check(rows([("probe45", 100, 0), ("agent_nav", 60, 0)]))
    assert low_trace["failures"] == ["trace rows=60 < 75"] and low_trace["ok"] is False
    # lots of rows, but the flagged ones are excluded from both floors
    flagged_heavy = ev.floor_check(rows([("probe45", 45, 0), ("synthetic_spill", 200, 150)]))
    assert flagged_heavy["n"] == 95 and flagged_heavy["failures"] == [
        "n=95 < 120",
        "trace rows=50 < 75",
    ]
    # trivial_head and unscored rows count for nothing
    extra = rows([("probe45", 45, 0), ("agent_nav", 80, 0)]) + [
        {"id": "t", "group": "trivial_head", "scored": True, "answer_named_verbatim": False},
        {"id": "u", "group": "agent_nav", "scored": False, "answer_named_verbatim": False},
    ]
    assert ev.floor_check(extra)["n"] == 125
    assert "FLOOR CHECK FAILED (primary): n=95 < 120; trace rows=50 < 75" in ev.floor_banner(
        flagged_heavy
    )


def test_analyze_floor_failure_is_underpowered_even_when_the_evidence_passes(tmp_path):
    out = tmp_path / "f1"
    rows = _write_strata(
        out, [("probe45", 100), ("agent_nav", 60)]
    )  # n = 160 >= 120, trace 60 < 75
    _write_variant(out, rows, "bm25", set(range(0, 40)))
    _write_variant(out, rows, "grep", set(range(0, 20)))
    _write_variant(out, rows, "bge-large.hybrid", set(range(0, 100)))
    _write_variant(out, rows, "granite-97m-r2.hybrid", set(range(0, 90)))
    res = ev.analyze(out)
    assert res["verdict_lines"][0] == "KILL RULE: UNDERPOWERED/KILL (n=160)"
    assert (
        res["verdict_lines"][1]
        == "FLOOR CHECK FAILED (primary): trace rows=60 < 75 -> UNDERPOWERED/KILL"
    )
    assert res["verdict_lines"][2] == "SELECTION: not applicable (kill rule UNDERPOWERED/KILL)"
    kill = res["kill_rule"]
    assert kill["floor_failures"] == ["trace rows=60 < 75"] and kill["underpowered"] is True
    assert (
        kill["registered"]["bge-large.hybrid"]["pass"] is True
    )  # the evidence is shown, it cannot decide
    assert res["floor_check"]["ok"] is False
    ev.cmd_analyze(ev.build_parser().parse_args(["analyze", "--out", str(out)]))
    md = (out / "analysis.md").read_text()
    assert "**!!! FLOOR CHECK FAILED (primary): trace rows=60 < 75" in md.splitlines()[6]
    # UNDERPOWERED/KILL outranks INCOMPLETE: a set that misses a floor is killed whatever the arms did
    out2 = tmp_path / "f2"
    rows2 = _write_strata(out2, [("probe45", 45)])
    _write_variant(out2, rows2, "bm25", set(range(0, 10)))  # grep missing: L cannot be chosen
    assert ev.analyze(out2)["verdict_lines"][0] == "KILL RULE: UNDERPOWERED/KILL (n=45)"


def test_analyze_floors_pass_on_the_expected_real_shape(tmp_path):
    out = tmp_path / "f3"
    _write_strata(
        out,
        [("probe45", 45), ("agent_nav", 43), ("synthetic_spill", 478)],
        leak=lambda i: (i - 45) % 11 == 0 if i >= 45 else None,
    )
    res = ev.analyze(out)  # no arm results at all: INCOMPLETE, but the floors are the topic here
    fc = res["floor_check"]
    assert fc["ok"] is True and fc["failures"] == []
    assert fc["n"] == res["n_deciding"] and fc["n_trace"] == fc["n"] - 45
    assert not any(line.startswith("FLOOR CHECK") for line in res["verdict_lines"])
    assert res["verdict_lines"][0] == f"KILL RULE: INCOMPLETE (n={fc['n']})"


def test_freeze_reports_the_floor_check_loudly_and_hashes_it(tmp_path, capsys):
    inp = ev.make_synthetic_inputs(tmp_path / "in", multi=True)
    out = tmp_path / "out"
    assert _freeze_multi(inp, out) == 0  # a warning, not a refusal
    cap = capsys.readouterr()
    banner = "FLOOR CHECK FAILED (primary): n=12 < 120; trace rows=6 < 75 -> the 2.2 verdict reads UNDERPOWERED/KILL"
    assert banner in cap.out and banner in cap.err and "!!!" in cap.err
    fc = json.loads((out / "floor-check.json").read_text())
    assert (
        fc["n"] == 12 and fc["n_trace"] == 6 and fc["n_flagged_excluded"] == 6 and fc["ok"] is False
    )
    assert any(
        Path(line.split("  ", 1)[1]).name == "floor-check.json"
        for line in (out / "FROZEN-AT-LAUNCH.sha256").read_text().splitlines()
    )
    # --require-floors turns the warning into exit 3 (the freeze file is still written)
    inp2 = ev.make_synthetic_inputs(tmp_path / "in2", multi=True)
    assert _freeze_multi(inp2, tmp_path / "out2", ["--require-floors"]) == 3
    assert (tmp_path / "out2" / "FROZEN-AT-LAUNCH.sha256").exists()


def test_leakage_block_flag_rates_and_recall_by_flag_and_writer(tmp_path):
    out = tmp_path / "lk"
    rows = _write_strata(out, [("probe45", 45), ("agent_nav", 40), ("synthetic_spill", 40)])
    # writer A: 15 of 20 flagged; writer B: 5 of 20; assistant_text: 10 of 20; bash_description: 2 of 20
    seen = {"A": 0, "B": 0, "assistant_text": 0, "bash_description": 0}
    for r in rows:
        key = r.get("writer") or r.get("intent_source")
        if key:
            quota = {"A": 15, "B": 5, "assistant_text": 10, "bash_description": 2}[key]
            r["answer_named_verbatim"] = seen[key] < quota
            seen[key] += 1
    (out / "queries-labelled.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    flagged = {i for i, r in enumerate(rows) if r["answer_named_verbatim"] is True}
    # bm25 hits exactly the flagged rows; grep nothing
    _write_variant(out, rows, "bm25", flagged)
    _write_variant(out, rows, "grep", set())
    _write_variant(
        out, rows, "bge-large.hybrid", set(range(0, 45)) | {i for i in range(45, 125) if i % 2}
    )
    lk = ev.analyze(out)["leakage"]
    fr = lk["flag_rates"]
    assert fr["by_writer"]["A"] == {
        "n": 20,
        "flagged": 15,
        "unflagged": 5,
        "unlabelled": 0,
        "flag_rate": 0.75,
    }
    assert fr["by_writer"]["B"]["flag_rate"] == 0.25
    assert fr["by_intent_source"]["assistant_text"]["flag_rate"] == 0.5
    assert fr["by_intent_source"]["bash_description"]["flag_rate"] == 0.1
    assert fr["by_stratum"]["probe45"] == {
        "n": 45,
        "flagged": 0,
        "unflagged": 0,
        "unlabelled": 45,
        "flag_rate": None,
    }
    assert fr["by_stratum"]["agent_nav"]["flag_rate"] == pytest.approx(12 / 40)
    assert fr["trace_all"]["flag_rate"] == pytest.approx(32 / 80)
    rbf = lk["recall_by_flag"]["bm25"]
    for k in (1, 3, 5, 10):  # recall@k on a one-pointer hit is the same at every k
        assert rbf["synthetic_spill/writer=A:flagged"][k]["recall"] == 1.0
        assert rbf["synthetic_spill/writer=A:unflagged"][k]["recall"] == 0.0
    assert (
        rbf["synthetic_spill/writer=A:flagged"][5]["n"] == 15 and rbf["trace:flagged"][5]["n"] == 32
    )
    assert rbf["trace:unflagged"][5]["n"] == 48 and "ci_lo" in rbf["trace:flagged"][5]
    assert rbf["agent_nav/intent_source=assistant_text:flagged"][1]["n"] == 10
    eff = lk["effect_flagged_minus_unflagged"]
    assert eff["bm25"]["trace"]["recall"][5]["diff"] == pytest.approx(1.0)
    assert eff["bm25"]["synthetic_spill/writer=B"]["recall"][5] == {
        "flagged": 1.0,
        "unflagged": 0.0,
        "diff": 1.0,
    }
    assert eff["bge-large.hybrid"]["trace"]["n_flagged"] == 32
    # grep never hits: no flagged/unflagged difference
    assert eff["grep"]["trace"]["recall"][5]["diff"] == 0.0
    assert set(lk["vs_L_recall5"]) >= {"trace:flagged", "trace:unflagged", "agent_nav:flagged"}
    assert "probe45:unlabelled" not in lk["vs_L_recall5"]
    assert lk["vs"] in ("bm25", "grep")
    assert lk["vs_L_recall5"]["trace:flagged"]["bge-large.hybrid"]["descriptive"] is True


def test_prerun_amendment_q_is_recorded_in_the_interpretations_and_the_config(tmp_path):
    q = ev.PREDECLARED["q_prerun_amendment_primary_unflagged"]
    assert q.startswith("(q) PRE-RUN AMENDMENT 2026-10-20, before any arm ran")
    for needle in (
        "81% synthetic",
        "42%",
        "answer_named_verbatim",
        "n >= 120",
        "trace rows",
        ">= 75",
    ):
        assert needle in q
    assert "PRIMARY" in ev.PREDECLARED["a_decision_set"] and "SECONDARY" in ev.PREDECLARED["n_sets"]
    assert "PRIMARY set only" in ev.PREDECLARED["p_selection_primary_only"]
    assert q in (ev.__doc__ or "")  # echoed in the module docstring like the others
    ev.run_synthetic(tmp_path / "out", tmp_path / "inputs", multi=True)
    pre = json.loads((tmp_path / "out" / "config-echo.json").read_text())["stable"]["predeclared"]
    assert pre["interpretations"]["q_prerun_amendment_primary_unflagged"] == q
    assert pre["stats"]["trace_min"] == 75 and pre["stats"]["n_min"] == 120


def test_markdown_orders_primary_leakage_then_the_non_deciding_sets(tmp_path):
    out, _, _ = _flagged_world(tmp_path, "md")
    ev.cmd_analyze(ev.build_parser().parse_args(["analyze", "--out", str(out)]))
    md = (out / "analysis.md").read_text()
    head = md.splitlines()
    assert (
        head[2] == ev.BIAS_NOTE and head[4] == ev.BIAS_NOTE_SCOPE
    )  # header: the note, then its scope
    assert "!!! FLOOR CHECK" not in md  # floors hold: no alarm
    i_primary = md.index("# PRIMARY decision set")
    i_leak = md.index("## leakage measurement")
    i_sec = md.index("# SECONDARY (non-deciding): pooled, flagged included")
    assert i_primary < i_leak < i_sec
    assert "### flagged minus unflagged recall (the leakage effect)" in md
    assert "### flag rates" in md and "| writer A |" in md
    # a non-deciding set shows recall tables and kill-rule evidence, not selection or bootstrap tables
    secondary = md[i_sec:]
    assert "## kill rule (recall@5, hybrid vs L) - pooled, flagged included" in secondary
    assert "paired bootstrap" not in secondary.split("# SECONDARY (non-deciding): without")[0]


def test_real_trace_uses_existing_floors_and_preserves_legacy_cohort(tmp_path, monkeypatch):
    out, rows, n = _pass_world(tmp_path)
    current = ev.analyze(out)
    # Remove only the new recognized label, reproducing the legacy group configuration.
    monkeypatch.setattr(ev, "TRACE_STRATA", ("agent_nav", "synthetic_spill", "spill_follow"))
    monkeypatch.setattr(ev, "DECISION_GROUPS", ("probe45", *ev.TRACE_STRATA))
    assert ev.analyze(out) == current  # all denominators, numeric verdicts and report groups
    assert current["n_deciding"] == n == 130
    assert current["floor_check"]["n_trace"] == 85
    monkeypatch.undo()
    real = dict(rows[-1], stratum="real_trace", text="find the handler", gold={"path": "f.py", "ranges": [[10, 20]]})
    labelled, _ = ev.label_queries([], [real])
    assert labelled[0]["group"] == "real_trace" and labelled[0]["scored"]
    assert ev.primary_rows(labelled) == labelled
    assert ev.floor_check(labelled)["n_trace"] == 1
    assert ev.floor_check(labelled)["n_min"] == st.N_MIN == 120
    assert ev.floor_check(labelled)["trace_min"] == st.TRACE_N_MIN == 75
