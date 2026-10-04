"""Coherence judge ``excerpt_mode="embed_drift"`` (2026-10-04, operator-approved option).

Covers the chunker, the alignment (identical chunks never embedded), drift selection
within the judged-token budget, unequal lengths (truncated / word-salad tails), the
recorded fail-closed fallback to ``head_tail_divergence``, the judge key per excerpt
mode, and the calibration runner's A/B plumbing.

Offline: the embedder is ``FakePooledEmbedder`` (bag-of-words vectors, no HTTP); token
counts use the 4-chars-per-token estimate.
"""

from __future__ import annotations

import json
import random
from contextlib import contextmanager
from typing import Any

import pytest

from src.embedding_pool.fake import FakePooledEmbedder
from src.runtime.measurement_windows import WindowHold
from src.typed_decisions import coherence_judge as cj
from src.typed_decisions import coherence_judge_calibration as cal
from src.typed_decisions import coherence_judge_drift as drift
from src.typed_decisions import coherence_judge_sidecar as sc
from src.typed_decisions.types import Decision, DecisionResult, QuestionKind

CAP = 384  # fixture paragraphs are ~90 tokens: head + both ends + 1-2 drift chunks
DIM = 1024  # sparse fake vectors: unrelated text ~0, a 2-word paraphrase ~0.9


def _para(seed: int, n_words: int = 60, prefix: str = "w") -> str:
    rng = random.Random(seed)
    words = [f"{prefix}{rng.randrange(10_000):04d}" for _ in range(n_words)]
    sentences = [" ".join(words[i : i + 12]).capitalize() + "." for i in range(0, n_words, 12)]
    return " ".join(sentences)


def _paraphrase(text: str, seed: int) -> str:
    """Same paragraph with two words replaced (high, but not identical, similarity)."""
    rng = random.Random(seed)
    words = text.split(" ")
    for k in rng.sample(range(len(words)), 2):
        words[k] = f"p{rng.randrange(10_000):04d}" + ("." if words[k].endswith(".") else "")
    return " ".join(words)


def _doc(paras: list[str]) -> str:
    return "\n\n".join(paras)


BASE_PARAS = [_para(i) for i in range(12)]
BASE = _doc(BASE_PARAS)


def _state(base: str, cand: str, *, embedder: Any = None, mode: str = cj.EMBED_DRIFT, cap: int = CAP):
    req = cj.JudgeRequest(prompt="Explain.", base_output=base, candidate_output=cand,
                          max_judged_tokens=cap, excerpt_mode=mode)

    def embed_fn(texts):
        if embedder is None:
            raise drift.DriftUnavailable("embedding_pool_disabled")
        return cj.embed_texts_sync(embedder, texts)

    return cj.build_state(req, embed_fn=embed_fn)


def _embedded(fake: FakePooledEmbedder) -> list[str]:
    return [t for call in fake.calls for t in call]


# ── chunker ──────────────────────────────────────────────────────────────────


def test_chunker_is_deterministic_covers_text_and_respects_max():
    text = BASE + "\n" + "x" * 3000 + " tail."
    chunks = drift.chunk_text(text, 4.0)
    assert chunks == drift.chunk_text(text, 4.0)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(text)
    assert all(a == prev_b for (a, _), (_, prev_b) in zip(chunks[1:], chunks[:-1]))  # contiguous
    assert max(b - a for a, b in chunks) <= drift.CHUNK_MAX_TOKENS * 4
    # each ~75-token paragraph is its own chunk (content-defined, so identical text re-syncs)
    assert [text[a:b].strip() for a, b in drift.chunk_text(BASE, 4.0)] == BASE_PARAS


# ── identical / under cap ────────────────────────────────────────────────────


def test_identical_outputs_embed_nothing():
    fake = FakePooledEmbedder(dim=DIM)
    _, _, ex = _state(BASE, BASE, embedder=fake)
    assert ex["mode_used"] == cj.EMBED_DRIFT and ex["fallback"] is None
    assert ex["embedded_chunks"] == 0 and fake.calls == []
    assert ex["drift"]["unit_kinds"] == {"equal": 12}
    assert ex["candidate_output"]["judged_tokens"] <= CAP


def test_nothing_before_the_divergence_is_embedded():
    cand_paras = BASE_PARAS[:7] + [_para(100 + i) for i in range(7, 12)]
    cand = _doc(cand_paras)
    fake = FakePooledEmbedder(dim=DIM)
    _, _, ex = _state(BASE, cand, embedder=fake)
    sent = _embedded(fake)
    assert sent and not any(p in t for t in sent for p in BASE_PARAS[:7])
    assert ex["embedded_chunks"] == len(set(sent)) == 10  # 5 base + 5 candidate chunks after it
    assert ex["drift"]["unit_kinds"]["equal"] == 7


def test_within_cap_is_whole_text_and_embeds_nothing():
    fake = FakePooledEmbedder(dim=DIM)
    state, truncated, ex = _state("Short base answer.", "Short candidate answer.", embedder=fake)
    assert fake.calls == [] and ex["excerpted"] is False and ex["mode_used"] == cj.EMBED_DRIFT
    assert ex["drift"]["skipped"] == "within_cap"
    assert "Short candidate answer." in state and truncated["candidate_output"] is False


# ── selection ────────────────────────────────────────────────────────────────


def test_single_paragraph_swap_is_selected():
    swapped = _para(999, prefix="z")
    cand = _doc(BASE_PARAS[:8] + [swapped] + BASE_PARAS[9:])
    fake = FakePooledEmbedder(dim=DIM)
    state, _, ex = _state(BASE, cand, embedder=fake)
    assert sorted(_embedded(fake)) == sorted([BASE_PARAS[8] + "\n\n", swapped + "\n\n"])
    assert ex["embedded_chunks"] == 2
    drifted = [s for s in ex["drift"]["selected"] if s["why"] == "drift"]
    assert drifted and all(s["similarity"] < 0.5 for s in drifted)
    cand_part = state.split("CANDIDATE OUTPUT (candidate kernel):")[1]
    ref_part = state.split("REFERENCE OUTPUT (base kernel):")[1].split("CANDIDATE OUTPUT")[0]
    assert swapped in cand_part and BASE_PARAS[8] in ref_part
    # identical paragraphs around the swap are elided, and the marker says they were identical
    assert BASE_PARAS[4] not in cand_part and "similarity ≥ 1.00" in cand_part
    # the head (first paragraph) and both ends are always shown
    assert BASE_PARAS[0] in cand_part and BASE_PARAS[11] in cand_part
    assert "passages where the two outputs differ most in meaning" in state


def test_truncated_candidate_tail_is_maximal_drift_and_its_end_is_shown():
    cut = len(_doc(BASE_PARAS[:7])) + 2 + len(BASE_PARAS[7]) // 2
    cand = BASE[:cut]
    fake = FakePooledEmbedder(dim=DIM)
    state, _, ex = _state(BASE, cand, embedder=fake)
    kinds = ex["drift"]["unit_kinds"]
    assert kinds.get("base_only") == 4 and "candidate_only" not in kinds
    # one replace block: the half paragraph vs base chunks 7..11 (the DP needs the base
    # vectors to place it; they are candidate-independent, so the pool LRU reuses them)
    assert ex["embedded_chunks"] == 6
    pair = [s for s in ex["drift"]["selected"] if s["kind"] == "pair"]
    assert len(pair) == 1 and 0.5 < pair[0]["similarity"] < 0.99
    cand_part = state.split("CANDIDATE OUTPUT (candidate kernel):\n<<<\n")[1].split("\n>>>")[0]
    assert cand_part.endswith(cand[-40:])  # the judge sees where the candidate stops
    base_only = [s for s in ex["drift"]["selected"] if s["kind"] == "base_only"]
    assert base_only and all(s["similarity"] == 0.0 for s in base_only)


def test_longer_candidate_extra_tail_is_unmatched():
    extra = [_para(500 + i, prefix="y") for i in range(3)]
    cand = _doc(BASE_PARAS + extra)
    fake = FakePooledEmbedder(dim=DIM)
    _, _, ex = _state(BASE, cand, embedder=fake)
    assert ex["drift"]["unit_kinds"].get("candidate_only", 0) >= 2
    assert ex["candidate_output"]["judged_tokens"] <= CAP


def test_word_salad_tail_is_selected():
    rng = random.Random(7)
    salad = ["\n".join(" ".join(f"q{rng.randrange(99999):05d}" for _ in range(60)) for _ in range(1))
             for _ in range(4)]
    cand = _doc(BASE_PARAS[:6] + salad)
    fake = FakePooledEmbedder(dim=DIM)
    state, _, ex = _state(BASE, cand, embedder=fake)
    cand_part = state.split("CANDIDATE OUTPUT (candidate kernel):")[1]
    assert salad[0] in cand_part and salad[-1] in cand_part
    assert BASE_PARAS[3] not in cand_part  # identical prefix body elided
    picked = [s for s in ex["drift"]["selected"] if s["why"] in ("drift", "tail")]
    assert picked and max(s["similarity"] for s in picked) < 0.5
    assert ex["candidate_output"]["judged_tokens"] <= CAP


def test_budget_cap_respected_and_lowest_similarity_first():
    cand_paras = BASE_PARAS[:2] + [_paraphrase(p, i) for i, p in enumerate(BASE_PARAS[2:], start=2)]
    cand_paras[9] = _para(4242, prefix="z")  # one real swap among paraphrases
    cand = _doc(cand_paras)
    fake = FakePooledEmbedder(dim=DIM)
    state, truncated, ex = _state(BASE, cand, embedder=fake)
    assert truncated["base_output"] and truncated["candidate_output"]
    assert ex["base_output"]["judged_tokens"] <= CAP and ex["candidate_output"]["judged_tokens"] <= CAP
    assert ex["judged_tokens"] == ex["base_output"]["judged_tokens"] + ex["candidate_output"]["judged_tokens"]
    drifted = [s for s in ex["drift"]["selected"] if s["why"] == "drift"]
    # lowest similarity first: nothing selected as drift is closer than anything elided
    assert drifted and max(s["similarity"] for s in drifted) <= ex["drift"]["min_elided_similarity"]
    assert min(s["similarity"] for s in drifted) == 0.0  # the swap (unmatched) went first
    assert cand_paras[9] in state
    # elided paraphrases are announced with their similarity floor
    assert "tokens elided, similarity ≥ 0." in state
    assert ex["drift"]["min_elided_similarity"] >= 0.5


def test_selection_is_deterministic():
    cand = _doc(BASE_PARAS[:3] + [_paraphrase(p, i) for i, p in enumerate(BASE_PARAS[3:])])
    a = _state(BASE, cand, embedder=FakePooledEmbedder(dim=DIM))
    b = _state(BASE, cand, embedder=FakePooledEmbedder(dim=DIM))
    assert a[0] == b[0]
    for ex in (a[2], b[2]):
        ex["embed_ms"] = ex["drift"]["embed_ms"] = 0
    assert a[2] == b[2]


# ── fail closed to the default mode ──────────────────────────────────────────


@pytest.mark.parametrize(
    "embedder, reason",
    [
        (None, "embedding_pool_disabled"),
        (FakePooledEmbedder(dim=DIM, available=False, reason="saturated"), "embedding_unavailable:saturated"),
        (FakePooledEmbedder(dim=DIM, available=False, reason="no_instances"), "embedding_unavailable:no_instances"),
    ],
)
def test_pool_down_falls_back_to_default_and_records_it(embedder, reason):
    cand = _doc(BASE_PARAS[:8] + [_para(999, prefix="z")] + BASE_PARAS[9:])
    state, truncated, ex = _state(BASE, cand, embedder=embedder)
    assert ex["mode_requested"] == cj.EMBED_DRIFT and ex["mode_used"] == cj.HEAD_TAIL_DIVERGENCE
    assert ex["fallback"]["reason"] == reason and ex["drift"] is None
    default_state, default_trunc, default_ex = _state(BASE, cand, mode=cj.HEAD_TAIL_DIVERGENCE)
    assert state == default_state and truncated == default_trunc  # judged exactly like the default
    assert ex["judged_tokens"] == default_ex["judged_tokens"] > 0
    assert default_ex["mode_used"] == cj.HEAD_TAIL_DIVERGENCE and default_ex["fallback"] is None


def test_divergence_at_byte_zero_shows_both_heads():
    cand = _doc([_para(300 + i, prefix="z") for i in range(12)])
    state, _, ex = _state(BASE, cand, embedder=FakePooledEmbedder(dim=DIM))
    heads = [s for s in ex["drift"]["selected"] if s["why"] == "head"]
    assert any(s["base"] for s in heads) and any(s["candidate"] for s in heads)
    assert BASE_PARAS[0] in state and _para(300, prefix="z") in state


def test_too_many_divergent_chunks_falls_back(monkeypatch):
    monkeypatch.setattr(drift, "MAX_EMBED_CHUNKS", 4)
    cand = _doc([_para(300 + i, prefix="z") for i in range(12)])
    fake = FakePooledEmbedder(dim=DIM)
    _, _, ex = _state(BASE, cand, embedder=fake)
    assert ex["mode_used"] == cj.HEAD_TAIL_DIVERGENCE and ex["fallback"]["reason"] == "too_many_chunks"
    assert fake.calls == []


def test_bad_vectors_fall_back():
    class Broken:
        def try_embed_many_sync(self, texts, **_):
            class O:
                is_dense = True
                vectors = [[float("nan")] * 4 for _ in texts]
            return O()

    cand = _doc(BASE_PARAS[:8] + [_para(999, prefix="z")] + BASE_PARAS[9:])
    _, _, ex = _state(BASE, cand, embedder=Broken())
    assert ex["mode_used"] == cj.HEAD_TAIL_DIVERGENCE and ex["fallback"]["reason"] == "embedding_invalid"


def test_async_only_embedder_is_refused_not_awaited():
    class AsyncOnly:
        async def try_embed_many(self, texts, **_):  # pragma: no cover - never awaited
            raise AssertionError

    with pytest.raises(drift.DriftUnavailable) as exc:
        cj.embed_texts_sync(AsyncOnly(), ["a"])
    assert exc.value.reason == "embedder_not_sync"


def test_default_head_is_unchanged_and_modes_have_distinct_fixed_heads():
    assert cj._STATE_HEAD == cj._state_head(cj.HEAD_TAIL_DIVERGENCE)
    assert "the region around the first divergence and its tail" in cj._STATE_HEAD
    assert cj._state_head(cj.EMBED_DRIFT) != cj._STATE_HEAD
    with pytest.raises(ValueError):
        cj.JudgeRequest(prompt="p", base_output="a", candidate_output="b", excerpt_mode="nope").validate()


# ── judge: keys, calibration, recorded fallback ──────────────────────────────

CHAMP_BUILD = "b10308-90c12df42"
SIDECAR_MODEL = "Qwen3.6-35B-A3B-MTP-Q8_0.gguf"


def _result(value):
    probs = {v: (0.7 if v == value else 0.1) for v in cj.VERDICTS}
    d = Decision(question_id=cj.QUESTION_ID, kind=QuestionKind.CHOICE, value=value, probabilities=probs,
                 confidence=0.6, mode="native", native_key="A")
    return DecisionResult(decisions=(d,), failures=(), raw_text="", mode="native", elapsed_ms=1.0,
                          prompt_sha256="0" * 64)


class _Run:
    def __init__(self):
        self.states: list[str] = []

    def __call__(self, primitives, **kwargs):
        self.states.append(kwargs.get("state", ""))
        return _result("COHERENT_EQUIVALENT")


class _Prims:
    server_urls: dict = {}

    @contextmanager
    def request_context(self, **_):
        yield

    def get_last_inference_meta(self):
        return {}


def _status():
    return sc.SidecarStatus(
        url="http://127.0.0.1:8199", reachable=True, champion=True, reason="ok", build_info=CHAMP_BUILD,
        expected_commit="90c12df42", served_model=SIDECAR_MODEL,
        state_path="/x/sidecar-8199.state.json", launch_command="launch",
    )


def _judge(tmp_path, *, embedder, holds=()):
    store = cj.CalibrationStore(tmp_path / "cal")
    run = _Run()
    judge = cj.CoherenceJudge(
        primitives_fn=lambda: None,
        holds_fn=lambda: list(holds),
        store=store,
        cloud_registry_fn=lambda: {},
        token_counter_fn=lambda p, role: None,
        sidecar_probe_fn=_status,
        sidecar_primitives_fn=lambda url: _Prims(),
        typed_run_fn=run,
        log_path=tmp_path / "calls.jsonl",
        embedder_fn=lambda: embedder,
    )
    return judge, store, run


def _seal(store, excerpt_mode):
    key = cj.judge_key(backend=cj.SIDECAR_BACKEND, model=sc.SIDECAR_ROLE, scoring_mode="native",
                       served_model=SIDECAR_MODEL, build=CHAMP_BUILD, excerpt_mode=excerpt_mode)
    record = cj.seal_calibration_record({"judge_key": key, "passed": True, "created_at": "2026-10-04T00:00:00+00:00"})
    store.write(record)
    return record["calibration_id"], key


def _req(**kw):
    cand = _doc(BASE_PARAS[:8] + [_para(999, prefix="z")] + BASE_PARAS[9:])
    return cj.JudgeRequest(prompt="Explain.", base_output=BASE, candidate_output=cand,
                           backend=cj.SIDECAR_BACKEND, max_judged_tokens=CAP, **kw)


def test_judge_key_carries_excerpt_mode():
    base = dict(backend=cj.SIDECAR_BACKEND, model="m", scoring_mode="native", served_model="s", build="b")
    assert cj.judge_key(**base) == cj.judge_key(**base, excerpt_mode=cj.HEAD_TAIL_DIVERGENCE)
    assert cj.judge_key(**base) != cj.judge_key(**base, excerpt_mode=cj.EMBED_DRIFT)


def test_embed_drift_verdict_is_keyed_and_calibrated_as_embed_drift(tmp_path):
    judge, store, run = _judge(tmp_path, embedder=FakePooledEmbedder(dim=DIM))
    _seal(store, cj.HEAD_TAIL_DIVERGENCE)
    cal_id, key = _seal(store, cj.EMBED_DRIFT)
    verdict = judge.judge(_req(excerpt_mode=cj.EMBED_DRIFT))
    assert verdict.judge_key == key and verdict.calibration_id == cal_id
    ex = verdict.excerpt
    assert ex["mode_used"] == cj.EMBED_DRIFT and ex["embedded_chunks"] == 2 and ex["drift"]["selected"]
    assert "differ most in meaning" in run.states[0]
    record = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()][-1]
    assert record["excerpt"]["mode_used"] == cj.EMBED_DRIFT and "judge_keys_fallback" in record


def test_default_calibration_does_not_validate_embed_drift(tmp_path):
    judge, store, _ = _judge(tmp_path, embedder=FakePooledEmbedder(dim=DIM))
    _seal(store, cj.HEAD_TAIL_DIVERGENCE)
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(_req(excerpt_mode=cj.EMBED_DRIFT))
    assert exc.value.kind == "judge_uncalibrated"


def test_pool_down_fallback_is_recorded_and_keyed_as_default(tmp_path):
    judge, store, run = _judge(tmp_path, embedder=FakePooledEmbedder(dim=DIM, available=False))
    cal_id, key = _seal(store, cj.HEAD_TAIL_DIVERGENCE)
    verdict = judge.judge(_req(excerpt_mode=cj.EMBED_DRIFT))
    ex = verdict.excerpt
    assert ex["mode_requested"] == cj.EMBED_DRIFT and ex["mode_used"] == cj.HEAD_TAIL_DIVERGENCE
    assert ex["fallback"]["reason"] == "embedding_unavailable:saturated"
    assert verdict.judge_key == key and verdict.calibration_id == cal_id
    assert "the region around the first divergence" in run.states[0]


def test_window_held_during_embed_falls_back(tmp_path):
    hold = WindowHold(window="cpu", path="/x/cpu-window.json", reason="ak cpu window open", retry_after_s=60)
    # A LOCAL call is refused before it gets here; a cloud call's embed step is guarded too.
    judge, _, _ = _judge(tmp_path, embedder=FakePooledEmbedder(dim=DIM), holds=(hold,))
    with pytest.raises(drift.DriftUnavailable) as exc:
        judge._embed_fn()(["x"])
    assert exc.value.reason == "measurement_window_held"
    free, _, _ = _judge(tmp_path, embedder=None)
    with pytest.raises(drift.DriftUnavailable) as exc2:
        free._embed_fn()(["x"])
    assert exc2.value.reason == "embedding_pool_disabled"


def test_default_mode_never_touches_the_embedder(tmp_path):
    def boom():
        raise AssertionError("embedder resolved in the default mode")

    store = cj.CalibrationStore(tmp_path / "cal")
    judge = cj.CoherenceJudge(
        primitives_fn=lambda: None, holds_fn=lambda: [], store=store, cloud_registry_fn=lambda: {},
        token_counter_fn=lambda p, role: None, sidecar_probe_fn=_status,
        sidecar_primitives_fn=lambda url: _Prims(), typed_run_fn=_Run(),
        log_path=tmp_path / "calls.jsonl", embedder_fn=boom,
    )
    verdict = judge.judge(_req(allow_uncalibrated=True))
    assert verdict.excerpt["mode_used"] == cj.HEAD_TAIL_DIVERGENCE and verdict.excerpt["embedded_chunks"] == 0


# ── calibration runner A/B plumbing ──────────────────────────────────────────


def _verdict(mode_used, *, fallback=None):
    return {"verdict": "INCOHERENT", "judge_key": f"cjk-{mode_used}", "backend": cj.SIDECAR_BACKEND,
            "model": sc.SIDECAR_ROLE, "served_model": SIDECAR_MODEL, "build": CHAMP_BUILD, "scoring_mode": "native",
            "excerpt": {"excerpted": True, "judged_tokens": 300, "mode_used": mode_used, "fallback": fallback,
                        "embedded_chunks": 4 if mode_used == cj.EMBED_DRIFT else 0, "embed_ms": 12.5}}


def test_calibration_runner_never_seals_a_fallback_row_under_embed_drift(tmp_path):
    items = [{"id": "loop:0", "category": "loop", "gold": "INCOHERENT"},
             {"id": "salad:0", "category": "salad", "gold": "INCOHERENT"}]
    bodies = iter([
        (200, _verdict(cj.EMBED_DRIFT)),
        (200, _verdict(cj.HEAD_TAIL_DIVERGENCE, fallback={"reason": "embedding_unavailable:saturated"})),
    ])
    store = cj.CalibrationStore(tmp_path / "cal")
    summary = cal.run_calibration(items, set_sha256="0" * 64, set_id="s", call=lambda item: next(bodies),
                                  out_dir=tmp_path / "run", store=store, backend=cj.SIDECAR_BACKEND,
                                  model=None, scoring="native", excerpt_mode=cj.EMBED_DRIFT)
    rows = [json.loads(x) for x in (tmp_path / "run" / "rows.jsonl").read_text().splitlines()]
    assert rows[1]["error"]["type"] == "identity_mismatch" and "fallback" in rows[1]["error"]["message"]
    assert [r["excerpt_mode"] for r in summary["records"]] == [cj.EMBED_DRIFT]
    assert summary["records"][0]["passed"] is False  # the unscored row counts against it
    rollup = summary["serving_rollup"]
    assert rollup["excerpt_fallbacks"] == {"embedding_unavailable:saturated": 1}
    assert rollup["excerpt_modes_used"] == {cj.EMBED_DRIFT: 1} and rollup["embedded_chunks_total"] == 4
    sealed = json.loads(open(summary["records"][0]["path"]).read())
    assert sealed["judge_identity"]["excerpt_mode"] == cj.EMBED_DRIFT


def test_calibration_cli_and_request_body_carry_excerpt_mode(capsys, tmp_path):
    body = cal._request_body({"prompt": "p", "base_output": "b", "candidate_output": "c"}, backend="local",
                             model=None, scoring="json", set_id="s", excerpt_mode=cj.EMBED_DRIFT,
                             max_judged_tokens=256)
    assert body["excerpt_mode"] == cj.EMBED_DRIFT and body["max_judged_tokens"] == 256
    cj.JudgeRequest(**body).validate()
    args = cal._parser().parse_args(["run", "--set", "x.jsonl", "--excerpt-mode", "embed_drift",
                                     "--max-judged-tokens", "256"])
    assert args.excerpt_mode == cj.EMBED_DRIFT and args.max_judged_tokens == 256


def test_route_body_and_client_carry_excerpt_mode():
    import importlib.util
    import sys

    from src.api.routes.typed_judge import CoherenceJudgeBody

    body = CoherenceJudgeBody(prompt="p", base_output="b", candidate_output="c", excerpt_mode="embed_drift")
    assert cj.JudgeRequest(**body.model_dump()).excerpt_mode == cj.EMBED_DRIFT
    assert CoherenceJudgeBody(prompt="p", base_output="b", candidate_output="c").excerpt_mode == cj.HEAD_TAIL_DIVERGENCE

    spec = importlib.util.spec_from_file_location("cjc_drift_test", "scripts/coherence_judge_client.py")
    cjc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cjc
    spec.loader.exec_module(cjc)
    sent = {}

    def post(url, body, timeout):
        sent.update(body)
        return 200, {"verdict": "COHERENT_EQUIVALENT", "calibration_id": "c", "backend": "b", "model": "m",
                     "judge_version": "v", "excerpt": {"excerpted": True, "mode_used": "embed_drift"}}

    verdict = cjc.make_judge_fn(post=post, excerpt_mode="embed_drift")({"prompt": "p", "base": "b", "candidate": "c"})
    assert sent["excerpt_mode"] == "embed_drift" and verdict.excerpt_mode == "embed_drift"
