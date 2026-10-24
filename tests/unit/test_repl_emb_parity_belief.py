"""VB-UFH12-PARITY: the parity gate's opt-in belief write side (G-P1 dry run on a synthetic vector set).

No model, server or timer is touched: a small seeded numpy vector set stands in for the encoded lanes, the gate's real
``compare`` computes ``parity.json``, and the epyc-root adapter writes the sidecar, projects and grades it. The adapter
is located exactly as the gate locates it (``--epyc-root`` / ``$EPYC_ROOT`` / pinned roots); the test skips when no
epyc-root checkout carrying ``repl_embedder_parity.py`` is reachable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark"))
import repl_emb_parity_gate as gate  # noqa: E402

ROOT_CANDIDATES = (
    os.environ.get("EPYC_ROOT"),
    "/mnt/raid0/llm/tmp/batch-20261024/wt-root",
    *gate.BELIEF_ROOTS,
)
EPYC_ROOT = next(
    (r for r in ROOT_CANDIDATES if r and (Path(r) / gate.ADAPTER_REL).is_file()), None
)
pytestmark = pytest.mark.skipif(EPYC_ROOT is None, reason="no epyc-root checkout with the parity adapter")

DIM, N_DOCS, N_Q = 768, 8, 4
NOISE = {"st-bf16": 0.002, "st-fp32-textonly": 0.0, "gguf-bf16": 0.001, "gguf-q8_0": 0.02}


def _items() -> dict:
    texts = [f"synthetic chunk {i}" for i in range(N_DOCS)]
    return {
        "files": [{"path": "/x/a.py", "sha256": "a" * 64, "kind": "code", "chars": 99}],
        "chunk_max_chars": 800,
        "dense_input_max_chars": 800,
        "n_chunks": N_DOCS,
        "chunk_kinds": {"code": N_DOCS},
        "chunks": [
            {"section": "/x/a.py", "start": i, "end": i + 5, "text_sha256": hashlib.sha256(t.encode()).hexdigest()}
            for i, t in enumerate(texts)
        ],
        "doc_texts": texts,
        "queries": [{"id": f"q{i}", "text": f"q {i}", "code_intent": False} for i in range(N_Q)],
        "query_set_kind": "proxy-first-line",
        "query_set_is_ufh07": False,
    }


def make_run(out: Path, *, nan_lane: str | None = None, seed: int = 7) -> None:
    """Reference + candidate vectors (reference plus lane-specific noise) and the per-lane records."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ref = {"docs": rng.normal(size=(N_DOCS, DIM)).astype("float32"), "queries": rng.normal(size=(N_Q, DIM)).astype("float32")}
    (out / "items.json").write_text(json.dumps(_items()))
    for lane in ("st-fp32-full", "st-fp32-textonly", "st-bf16", "gguf-bf16", "gguf-q8_0"):
        if lane == "st-fp32-full":
            vec = {k: v.copy() for k, v in ref.items()}
        else:
            vec = {k: v + NOISE[lane] * rng.normal(size=v.shape).astype("float32") * np.abs(v).mean() * 10 for k, v in ref.items()}
        if lane == nan_lane:
            vec["docs"][2, 5] = np.nan
        np.savez(out / f"vec-{lane}.npz", **vec)
        spec = gate._spec(lane)
        (out / f"lane-{lane}.json").write_text(
            json.dumps(
                {
                    "lane": lane,
                    "arm_id": spec.arm_id,
                    "index_identity": spec.index_identity(),
                    "threads": 4,
                    "n_docs": N_DOCS,
                    "n_queries": N_Q,
                    "raw_dim": DIM,
                }
            )
        )


def runtime_meta() -> dict:
    st = {
        "runtime": "sentence-transformers/transformers/torch-cpu",
        "argv": ["repl_emb_parity_gate.py", "encode", "--lane", "st"],
        "versions": {"torch": "2.4.0", "transformers": "5.20.1", "sentence_transformers": "6.1.0"},
    }
    gg = {
        "runtime": "llama-server production-consolidated-v10",
        "argv": ["llama-server", "-m", "embeddinggemma-2.gguf", "--embedding"],
        "binary_sha256": "c" * 64,
        "flash_attn_requested": "auto",
        "flash_attn_resolved": "off",
    }
    return {"st-fp32-full": st, "st-bf16": st, "gguf-bf16": gg, "gguf-q8_0": gg}


def _compare_cli(out: Path, tmp: Path, meta: dict | None = None, extra: list[str] | None = None) -> int:
    mp = tmp / "meta.json"
    mp.write_text(json.dumps(runtime_meta() if meta is None else meta))
    return gate.main(
        ["compare", "--out", str(out), "--belief-out", str(tmp / "belief"), "--runtime-meta", str(mp), "--epyc-root", EPYC_ROOT, *(extra or [])]
    )


def _adapter():
    rp = gate.load_belief_adapter(EPYC_ROOT)
    return rp


def test_gp1_dry_run_emits_a_record_the_read_side_projects_and_grades(tmp_path, capsys):
    out = tmp_path / "gp1"
    make_run(out)
    rc = _compare_cli(out, tmp_path)
    cap = capsys.readouterr()
    assert rc in (0, 3), cap.err  # verdicts may fail on noisy synthetic lanes; the capture itself must not
    side = tmp_path / "belief" / "repl_embedder_parity.beliefs.jsonl"
    assert side.is_file(), cap.err
    rp = _adapter()
    import claim_tuple as ct  # noqa: PLC0415

    natives = rp.native_rows(side)
    assert len(natives) == 9
    parity = json.loads((out / "parity.json").read_text())
    for n in natives:
        t = rp.project(n)
        e = n["row"]["extra"]
        assert ct.grade(t)[:2] == ("Judged", "Located")
        assert e["gate_verdict"] == parity["lanes"][e["lane"]]["verdict"]
        assert e["threshold"] == gate.GATES[e["lane"]] and e["reference"]["lane"] == gate.REFERENCE_LANE
        assert e["scope_complete"] is False and e["query_set_kind"] == "proxy-first-line"
        assert t.protocol_id == "" and t.attestation_present is True
    by = {(n["row"]["extra"]["lane"], n["row"]["metric"]): n["row"] for n in natives}
    q8 = by[("gguf-q8_0", "repl_embedder_parity_cos_min")]
    per_item = parity["lanes"]["gguf-q8_0"]["docs"]["per_item_cos"] + parity["lanes"]["gguf-q8_0"]["queries"]["per_item_cos"]
    assert q8["value"] == min(per_item)
    assert q8["extra"]["binary_sha256"] == "c" * 64 and q8["extra"]["flash_attn_resolved"] == "off"
    assert rp.frames_for_sidecar(side, as_of="2026-10-24T12:00:00Z")


def test_nonfinite_vector_is_recorded_as_a_failed_lane(tmp_path, capsys):
    out = tmp_path / "gp1"
    make_run(out, nan_lane="gguf-bf16")
    assert _compare_cli(out, tmp_path) == 3  # the gate's own FAIL exit; capture succeeded
    side = tmp_path / "belief" / "repl_embedder_parity.beliefs.jsonl"
    rows = {(n["row"]["extra"]["lane"], n["row"]["metric"]): n["row"] for n in _adapter().native_rows(side)}
    assert rows[("gguf-bf16", "repl_embedder_parity_nonfinite_vectors")]["value"] == 1
    assert rows[("gguf-bf16", "repl_embedder_parity_cos_min")]["extra"]["gate_verdict"] == "FAIL"


def test_missing_runtime_identity_exits_4_and_keeps_the_gate_record(tmp_path, capsys):
    out = tmp_path / "gp1"
    make_run(out)
    meta = runtime_meta()
    del meta["gguf-bf16"]["flash_attn_resolved"]
    assert _compare_cli(out, tmp_path, meta) == gate.EXIT_CAPTURE_FAILED
    assert "belief capture NOT written" in capsys.readouterr().err
    assert (out / "parity.json").is_file() and not (tmp_path / "belief" / "repl_embedder_parity.beliefs.jsonl").exists()


def test_belief_out_without_runtime_meta_is_refused(tmp_path, capsys):
    out = tmp_path / "gp1"
    make_run(out)
    rc = gate.main(["compare", "--out", str(out), "--belief-out", str(tmp_path / "belief"), "--epyc-root", EPYC_ROOT])
    assert rc == gate.EXIT_CAPTURE_FAILED and "--runtime-meta" in capsys.readouterr().err


def test_gate_runs_unchanged_without_the_flag(tmp_path, capsys):
    out = tmp_path / "gp1"
    make_run(out)
    rc = gate.main(["compare", "--out", str(out)])
    assert rc in (0, 3)
    assert (out / "parity.json").is_file() and not (tmp_path / "belief").exists()
    assert "belief" not in capsys.readouterr().out


def test_unreachable_root_exits_4(tmp_path, capsys, monkeypatch):
    out = tmp_path / "gp1"
    make_run(out)
    monkeypatch.delenv("EPYC_ROOT", raising=False)
    monkeypatch.setattr(gate, "BELIEF_ROOTS", ())
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(runtime_meta()))
    rc = gate.main(["compare", "--out", str(out), "--belief-out", str(tmp_path / "b"), "--runtime-meta", str(mp), "--epyc-root", str(tmp_path)])
    assert rc == gate.EXIT_CAPTURE_FAILED and "adapter not found" in capsys.readouterr().err
