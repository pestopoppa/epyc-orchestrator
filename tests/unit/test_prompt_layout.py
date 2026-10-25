"""RTE-LOG-1: prompt-structure logging and the offline reader (hashes and lengths only)."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.backends import dispatch_log, prompt_layout, serving_calls
from src.prompt_builders import reusable_blocks as rb
from src.prompt_builders.types import RootLMPrompt

_READER = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "prompt_layout_reader.py"
_spec = importlib.util.spec_from_file_location("prompt_layout_reader", _READER)
reader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reader)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(prompt_layout.LOG_ENV, raising=False)
    monkeypatch.delenv(prompt_layout.SOURCE_ENV, raising=False)
    prompt_layout.clear_layout()
    yield
    prompt_layout.clear_layout()


def _legacy_to_string(p: RootLMPrompt, prefix_stable: bool) -> str:
    """The pre-RTE-LOG-1 implementation, verbatim, as the byte-identity oracle."""
    parts = []
    if p.system:
        parts.append(p.system)
    if p.tools:
        parts.extend(["", "## Available Tools", p.tools])
    if p.rules:
        parts.extend(["", "## Rules", p.rules])
    if prefix_stable:
        if p.task:
            parts.extend(["", "## Task", p.task])
        if p.instruction:
            parts.extend(["", "## Your Code", p.instruction])
        if p.state:
            parts.extend(["", "## Current State", p.state])
        if p.context:
            parts.extend(["", p.context])
        if p.reference_code:
            parts.extend(["", "## Reference Code", p.reference_code])
    else:
        if p.state:
            parts.extend(["", "## Current State", p.state])
        if p.context:
            parts.extend(["", p.context])
        if p.reference_code:
            parts.extend(["", "## Reference Code", p.reference_code])
        if p.task:
            parts.extend(["", "## Task", p.task])
        if p.instruction:
            parts.extend(["", "## Your Code", p.instruction])
    return "\n".join(parts)


FIELDS = ("system", "tools", "rules", "state", "context", "reference_code", "task", "instruction")


def _prompts():
    for mask in itertools.product((0, 1), repeat=len(FIELDS)):
        yield RootLMPrompt(**{f: f"<{f} body>\nline2" for f, on in zip(FIELDS, mask) if on})


@pytest.mark.parametrize("stable", [False, True])
def test_to_string_is_byte_identical_to_legacy(stable):
    for p in _prompts():
        assert p.to_string(prefix_stable=stable) == _legacy_to_string(p, stable)


@pytest.mark.parametrize("stable", [False, True])
def test_section_spans_cover_exactly_their_section(stable):
    p = RootLMPrompt(**{f: f"<{f} body>" for f in FIELDS})
    text = p.to_string(prefix_stable=stable)
    spans = p.section_spans(stable)
    assert [k for k, _, _ in spans if k != "task"] == [
        "system",
        "tools",
        "rules",
        "state",
        "context",
        "reference_code",
    ]
    by_kind = {k: text[a:b] for k, a, b in spans}
    assert by_kind["system"] == "<system body>"
    assert by_kind["tools"] == "## Available Tools\n<tools body>"
    assert by_kind["state"] == "## Current State\n<state body>"
    assert by_kind["context"] == "<context body>"
    assert "<task body>" in text[spans[[k for k, _, _ in spans].index("task")][1] :]
    # spans are ordered and non-overlapping
    for (_, _, e1), (_, s2, _) in zip(spans, spans[1:]):
        assert e1 <= s2


def test_build_layout_hashes_lengths_and_no_text():
    prompt = "SYSTEM TEXT\n\n## Task\nsecret task text"
    lay = prompt_layout.build_layout(
        prompt,
        sections=[("system", 0, 11), ("task", 13, len(prompt))],
        turn_index=3,
        source="autopilot",
    )
    assert [b["kind"] for b in lay["blocks"]] == ["system", "task"]
    assert lay["blocks"][0]["length"] == 11 and lay["blocks"][0]["sha256_12"] == prompt_layout.h12(
        "SYSTEM TEXT"
    )
    assert lay["turn_index"] == 3 and lay["source"] == "autopilot"
    assert "secret" not in json.dumps(lay)


def test_chunks_carry_hash_rank_and_offset():
    prompt = "head\nCHUNK-A\nCHUNK-B\ntail"
    ref = "CHUNK-A\nCHUNK-B"
    lay = prompt_layout.build_layout(
        prompt, named=[("doc_chunk", ref, [("CHUNK-A", 1), ("CHUNK-B", 2)])]
    )
    ch = lay["blocks"][0]["chunks"]
    assert [(c["rank"], c["length"], c["start"]) for c in ch] == [(1, 7, 5), (2, 7, 13)]
    assert ch[0]["sha256_12"] == prompt_layout.h12("CHUNK-A")


def test_injected_blocks_located_by_bytes_and_unlocated_counted():
    blocks = rb.pack_blocks([f"item {i} " + "x" * 60 for i in range(30)])
    prompt = "intro\n" + rb.render_blocks(blocks, header="H") + "\ntail"
    absent = rb.pack_blocks(["not in the prompt"])
    lay = prompt_layout.build_layout(
        prompt, injected=[("skill", b) for b in blocks] + [("skill", absent[0])]
    )
    assert len(lay["blocks"]) == len(blocks) and lay["unlocated_blocks"] == 1
    assert all(b["kind"] == "skill" and "n_items" in b and "meets_min" in b for b in lay["blocks"])
    recs = rb.block_log_records(blocks, "skill")
    assert [
        (b["sha256_12"], b["est_tokens"], b["n_items"], b["meets_min"]) for b in lay["blocks"]
    ] == [(r["sha256_12"], r["est_tokens"], r["n_items"], r["meets_min"]) for r in recs]


def test_source_label_is_explicit_never_inferred(monkeypatch):
    assert prompt_layout.resolve_source({"client_class": "opencode"}) is None
    assert prompt_layout.resolve_source({"client_class": "Benchmark"}) == "benchmark"
    monkeypatch.setenv(prompt_layout.SOURCE_ENV, "autopilot")
    assert prompt_layout.resolve_source(None) == "autopilot"
    monkeypatch.setenv(prompt_layout.SOURCE_ENV, "whatever")
    assert prompt_layout.resolve_source(None) is None


def test_mismatched_prompt_attaches_nothing_and_kill_switch(monkeypatch):
    base = "alpha prompt " + "x" * 400
    prompt_layout.record_prompt(base, sections=[("system", 0, 5)], turn_index=0)
    assert prompt_layout.fields_for_request(SimpleNamespace(prompt=base + " tail"))["blocks"]
    assert (
        prompt_layout.fields_for_request(SimpleNamespace(prompt="other prompt " + "y" * 400)) == {}
    )
    chat = prompt_layout.fields_for_request(SimpleNamespace(prompt=None))
    assert chat["layout_match"] is None and chat["blocks"]
    monkeypatch.setenv(prompt_layout.LOG_ENV, "off")
    assert prompt_layout.fields_for_request(SimpleNamespace(prompt=base)) == {}


def test_serving_record_carries_blocks_and_flag_off_is_unchanged(monkeypatch):
    req = SimpleNamespace(
        prompt="alpha prompt",
        n_tokens=8,
        role="frontdoor",
        timeout=None,
        slot_id=None,
        chat_payload=None,
    )
    base = serving_calls.build_record(
        method="infer",
        role_config=None,
        request=req,
        base_url="http://x:8080",
        ts_start=1.0,
        ts_end=2.0,
    )
    assert "blocks" not in base
    prompt_layout.record_prompt(
        req.prompt, sections=[("system", 0, 5)], turn_index=2, source="benchmark"
    )
    rec = serving_calls.build_record(
        method="infer",
        role_config=None,
        request=req,
        base_url="http://x:8080",
        ts_start=1.0,
        ts_end=2.0,
    )
    assert (
        rec["blocks"][0]["kind"] == "system"
        and rec["turn_index"] == 2
        and rec["source"] == "benchmark"
    )
    for k in (
        "blocks",
        "turn_index",
        "source",
        "layout_match",
        "layout_chars",
        "unlocated_blocks",
        "layout_schema",
    ):
        rec.pop(k, None)
    base.pop("record_id"), rec.pop("record_id")
    assert rec == base


def test_dispatch_record_has_blocks_and_token_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "0")
    monkeypatch.setenv("ORCHESTRATOR_SERVING_CALLS_LOG", "off")
    path = tmp_path / "dispatch.jsonl"
    monkeypatch.setenv(dispatch_log.LOG_ENV, str(path))
    import src.backends.concurrency_aware as CA

    res = SimpleNamespace(success=True, prompt_tokens=11, cached_prompt_tokens=7)

    def stub(url):
        return SimpleNamespace(
            config=SimpleNamespace(base_url=url),
            infer=lambda rc, rq: res,
            infer_streaming=lambda rc, rq: "h",
            infer_stream_text=lambda rc, rq, on_chunk=None: res,
        )

    cab = CA.ConcurrencyAwareBackend(
        stub("http://localhost:8070"),
        [stub("http://localhost:8080")],
        role="frontdoor",
        full_port=8070,
    )
    prompt_layout.record_prompt(
        "hello world", sections=[("task", 0, 11)], turn_index=4, source="interactive"
    )
    cab.infer(
        SimpleNamespace(name="frontdoor"),
        SimpleNamespace(
            prompt="hello world", n_tokens=8, role="frontdoor", session_id="", task_id="t"
        ),
    )
    rec = [json.loads(x) for x in path.read_text().splitlines() if '"dispatch"' in x][-1]
    assert (
        rec["blocks"][0]["kind"] == "task"
        and rec["turn_index"] == 4
        and rec["source"] == "interactive"
    )
    assert rec["prompt_tokens"] == 11 and rec["cached_prompt_tokens"] == 7
    assert rec["record_sha256"] == serving_calls.record_digest(rec)
    assert "hello world" not in json.dumps(rec)
    serving_calls.clear_staged()


def test_builder_stages_layout_for_root_lm_prompt():
    from src.prompt_builders.builder import PromptBuilder

    text = PromptBuilder().build_root_lm_prompt(
        state="x = 1",
        original_prompt="do the thing",
        turn=2,
        routing_context="Role: frontdoor (Tier 1)",
    )
    lay = prompt_layout.peek_layout()
    kinds = [b["kind"] for b in lay["blocks"]]
    assert {"system", "tools", "rules", "state", "task", "routing"} <= set(kinds)
    assert lay["turn_index"] == 2 and lay["layout_chars"] == len(text)
    routing = [b for b in lay["blocks"] if b["kind"] == "routing"][0]
    assert text[routing["start"] : routing["start"] + routing["length"]].startswith(
        "## Routing Intelligence"
    )
    ends = [(b["start"], b["start"] + b["length"]) for b in lay["blocks"]]
    assert all(e1 <= s2 for (_, e1), (s2, _) in zip(ends, ends[1:]))


def test_builder_emits_tool_output_block_and_it_does_not_overlap_context():
    from src.prompt_builders.builder import PromptBuilder

    for kw, head in (
        ({"last_output": "hello out"}, "## Last Output"),
        ({"last_error": "Traceback boom"}, "## Last Error"),
    ):
        text = PromptBuilder().build_root_lm_prompt(
            state="x = 1",
            original_prompt="do it",
            turn=1,
            routing_context="Role: frontdoor (Tier 1)",
            **kw,
        )
        lay = prompt_layout.peek_layout()
        to = [b for b in lay["blocks"] if b["kind"] == "tool_output"]
        assert len(to) == 1
        seg = text[to[0]["start"] : to[0]["start"] + to[0]["length"]]
        assert seg.startswith(head) and seg.endswith("```")
        assert to[0]["sha256_12"] == prompt_layout.h12(seg)
        ends = [(b["start"], b["start"] + b["length"]) for b in lay["blocks"]]
        assert all(e1 <= s2 for (_, e1), (s2, _) in zip(ends, ends[1:]))
        assert {"routing", "tool_output"} <= {b["kind"] for b in lay["blocks"]}
    PromptBuilder().build_root_lm_prompt(state="x", original_prompt="t", turn=0)
    assert not [b for b in prompt_layout.peek_layout()["blocks"] if b["kind"] == "tool_output"]


def test_rebase_layout_cot_prefix_and_riders_then_compression():
    from src.graph.helpers import _rebase_prompt_layout
    from src.prompt_builders.builder import PromptBuilder

    base = PromptBuilder().build_root_lm_prompt(
        state="x = 1", original_prompt="do the thing", turn=3, last_output="out"
    )
    orig = {b["kind"]: b for b in prompt_layout.peek_layout()["blocks"]}
    final = "Think step by step.\n" + base + "\n\n[Auto Gathered Context]\nfoo\n\n** DEADLINE **"
    _rebase_prompt_layout(final, base)
    fields = prompt_layout.fields_for_request(SimpleNamespace(prompt=final))
    assert fields["layout_match"] is True and fields["rebased"] is True
    off = len("Think step by step.\n")
    assert fields["layout_chars"] == off + len(base) and fields["tail_chars"] == len(
        final
    ) - off - len(base)
    for b in fields["blocks"]:
        assert b["start"] == orig[b["kind"]]["start"] + off
        assert prompt_layout.h12(final[b["start"] : b["start"] + b["length"]]) == b["sha256_12"]
    assert {"tool_output", "task"} <= {b["kind"] for b in fields["blocks"]}
    # the old (unrebased) head would not have matched a cot_prefix prompt
    # compression rewrote the builder text: nothing is guessed
    prompt_layout.clear_layout()
    PromptBuilder().build_root_lm_prompt(state="x = 1", original_prompt="do the thing", turn=3)
    n = len(prompt_layout.peek_layout()["blocks"])
    squeezed = "[compressed] " + base[:100]
    _rebase_prompt_layout(
        squeezed,
        PromptBuilder().build_root_lm_prompt(state="x = 1", original_prompt="do the thing", turn=3),
    )
    f2 = prompt_layout.fields_for_request(SimpleNamespace(prompt=squeezed))
    assert f2["blocks"] == [] and f2["unlocated_blocks"] == n and f2["rebased"] is False


def test_rebase_ignores_foreign_layout_and_kill_switch(monkeypatch):
    prompt_layout.record_prompt("alpha " + "x" * 400, sections=[("system", 0, 5)])
    prompt_layout.rebase_layout("pre " + "beta " + "y" * 400, "beta " + "y" * 400)
    assert "tail_chars" not in prompt_layout.peek_layout()
    base = "alpha " + "x" * 400
    monkeypatch.setenv(prompt_layout.LOG_ENV, "off")
    prompt_layout.rebase_layout("pre " + base, base)
    assert "tail_chars" not in prompt_layout.peek_layout()


# -- reader --------------------------------------------------------------------


def _rec(ts, role, source, blocks):
    return {
        "kind": "dispatch",
        "ts_arrival": ts,
        "role": role,
        "source": source,
        "blocks": blocks,
        "prompt_tokens": 100,
        "cached_prompt_tokens": 40,
    }


def _b(kind, h, start, length, chunks=None):
    b = {"kind": kind, "sha256_12": h, "start": start, "length": length}
    if chunks:
        b["chunks"] = chunks
    return b


def test_reader_classifies_prefix_moved_and_pred_repeats():
    sysb = _b("system", "sys", 0, 100)
    recs = [
        _rec(
            "t1",
            "r",
            "autopilot",
            [
                sysb,
                _b(
                    "doc_chunk",
                    "D",
                    100,
                    50,
                    [
                        {"sha256_12": "c1", "rank": 1, "length": 25, "start": 100},
                        {"sha256_12": "c2", "rank": 2, "length": 25, "start": 125},
                    ],
                ),
            ],
        ),
        # same system prefix, same chunks again after the same predecessors: prefix repeats
        _rec(
            "t2",
            "r",
            "autopilot",
            [
                sysb,
                _b(
                    "doc_chunk",
                    "D",
                    100,
                    50,
                    [
                        {"sha256_12": "c1", "rank": 1, "length": 25, "start": 100},
                        {"sha256_12": "c2", "rank": 2, "length": 25, "start": 125},
                    ],
                ),
            ],
        ),
        # c2 now first and c1 later at a different offset: moved repeat for c1
        _rec(
            "t3",
            "r",
            "autopilot",
            [
                sysb,
                _b(
                    "doc_chunk",
                    "D",
                    100,
                    50,
                    [
                        {"sha256_12": "c2", "rank": 1, "length": 25, "start": 100},
                        {"sha256_12": "c1", "rank": 2, "length": 25, "start": 125},
                    ],
                ),
            ],
        ),
        # records without blocks and other-source records are counted/segregated
        {"kind": "dispatch", "ts_arrival": "t4", "role": "r"},
        _rec("t5", "r", "benchmark", [sysb]),
    ]
    rep = reader.analyse(recs)
    assert rep["inputs"]["records"] == 5 and rep["inputs"]["with_blocks"] == 4
    cell = {(c["kind"], c["role"], c["source"]): c for c in rep["per_kind_role_source"]}
    chunk = cell[("doc_chunk", "r", "autopilot")]
    assert chunk["n"] == {"first": 2, "prefix_repeat": 2, "moved_repeat": 2, "pred_repeat": 0}
    system = cell[("system", "r", "autopilot")]
    assert system["n"]["first"] == 1 and system["n"]["prefix_repeat"] == 2
    assert cell[("system", "r", "benchmark")]["n"]["prefix_repeat"] == 1
    pr = {(q["role"], q["source"]): q for q in rep["per_request"]}[("r", "autopilot")]
    assert (
        pr["requests"] == 3 and pr["distinct_chunks_mean"] == 2 and pr["chunk_length_median"] == 25
    )
    # t3 vs t2: plain prefix = system only (100 of 150); reordering rebuilds all of t2's prefix
    assert pr["reconstructible_share_mean"] > pr["plain_prefix_share_mean"]
    md = reader.render_markdown(rep)
    assert "| doc_chunk | r | autopilot |" in md and "moved_repeat" in md


def test_reader_cli_json_roundtrip(tmp_path, capsys):
    f = tmp_path / "d.jsonl"
    f.write_text(
        "\n".join(
            json.dumps(_rec(f"t{i}", "r", "interactive", [_b("system", "s", 0, 10)]))
            for i in range(3)
        )
        + "\nnot json\n"
    )
    assert reader.main(["--format", "json", str(f)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["schema"] == reader.REPORT_SCHEMA and rep["inputs"]["analysed_requests"] == 3


# -- live-record check (runs on the first real records) -------------------------


def _load_live_check():
    p = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "prompt_layout_live_check.py"
    spec = importlib.util.spec_from_file_location("prompt_layout_live_check", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_check_verdicts(tmp_path):
    lc = _load_live_check()
    blk = [
        {"kind": "state", "start": 0, "length": 5, "sha256_12": "a" * 12},
        {"kind": "tool_output", "start": 5, "length": 9, "sha256_12": "b" * 12},
    ]
    good = {
        "kind": "dispatch",
        "ts_arrival": "1",
        "blocks": blk,
        "layout_chars": 14,
        "prompt_tokens": 10,
        "cached_prompt_tokens": 0,
        "source": "benchmark",
        "rebased": True,
    }
    bad = {
        "kind": "serving_call",
        "ts_start": "2",
        "blocks": blk,
        "layout_chars": 14,
        "prompt_tokens": None,
        "cached_prompt_tokens": 3,
        "source": None,
    }
    nob = {"kind": "dispatch", "ts_arrival": "3"}
    f = tmp_path / "r.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in (good, nob)) + "\n")
    rep = lc.check([f])
    assert rep["verdict"] == "fail"
    assert rep["record_kind_shortfalls"] == {"serving_call": 1}
    serving_good = dict(good, kind="serving_call")
    f.write_text("\n".join(json.dumps(r) for r in (good, serving_good, nob)) + "\n")
    rep = lc.check([f])
    assert rep["verdict"] == "pass" and rep["record_kind_shortfalls"] == {}
    assert rep["repl_blocks"]["rebased"] == 2 and rep["repl_blocks"]["with_tool_output"] == 2
    f.write_text("\n".join(json.dumps(r) for r in (good, bad)) + "\n")
    rep = lc.check([f])
    assert (
        rep["verdict"] == "fail"
        and rep["problems"][0]["missing"] == ["prompt_tokens"]
        and rep["unlabelled_source"] == 1
    )
    f.write_text(json.dumps(nob) + "\n")
    assert lc.check([f])["verdict"] == "no_records" and lc.main([str(f)]) == 2


def test_live_check_requires_each_record_kind_minimum(tmp_path):
    lc = _load_live_check()
    good = {
        "kind": "serving_call",
        "blocks": [{"kind": "task", "start": 0, "length": 1, "sha256_12": "a" * 12}],
        "layout_chars": 1,
        "prompt_tokens": 1,
        "cached_prompt_tokens": 0,
    }
    f = tmp_path / "minimums.jsonl"
    # One family cannot stand in for the missing other family.
    f.write_text(json.dumps(good) + "\n")
    assert lc.check([f])["record_kind_shortfalls"] == {"dispatch": 1}
    assert lc.main([str(f)]) == 1
    # Summed count meets N, but serving has only one of the required two.
    dispatch = dict(good, kind="dispatch")
    f.write_text("\n".join(json.dumps(r) for r in (dispatch, dispatch, good)) + "\n")
    rep = lc.check([f], min_records=2)
    assert rep["verdict"] == "fail"
    assert rep["record_kind_shortfalls"] == {"serving_call": 1}
    f.write_text("\n".join(json.dumps(r) for r in (dispatch, dispatch, good, good)) + "\n")
    assert lc.check([f], min_records=2)["verdict"] == "pass"
