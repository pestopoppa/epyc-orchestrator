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


def _pass_world(tmp_path, n_probe=45, n_spill=85):
    out = tmp_path / "a"
    rows = _write_world(out, n_probe, n_spill, leak=lambda i: i % 2 == 0)
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
    assert res["leakage_cut"]["recall5"]["answer_named_verbatim=true"]["n"] == 65
    assert "bge-large.hybrid" in res["leakage_cut"]["recall5"]["answer_named_verbatim=false"]
    assert res["leakage_cut"]["kill_rule_on_false_subset"]["n"] == 65
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
