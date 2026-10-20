"""REPL-EMB-2.5: arm specs, index identity and DenseEmbedder adapters (no model weights needed)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark"))
import repl_emb_arms as arms  # noqa: E402

cs = arms.load_context_search()


class FakeAdapter(arms._AdapterBase):
    """Records the prefixed strings and returns hash-seeded 768-d vectors."""

    def __init__(self, spec, nan=False):
        super().__init__(spec)
        self.seen = []
        self.nan = nan

    def _encode(self, texts):
        self.seen.append(list(texts))
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=self.spec.native_dim).astype("float32")
            out.append(v)
        out = np.asarray(out, dtype="float32")
        if self.nan:
            out[0, 3] = np.nan
        return out


def _spec(dim=768, dtype="fp32"):
    return next(
        s
        for s in arms.arm_specs("st", "ll", "m2v")
        if s.family == "st" and s.dim == dim and s.dtype == dtype
    )


def test_identity_is_per_arm_and_sensitive_to_every_field():
    a, b = _spec(768), _spec(256)
    assert a.model_id() != b.model_id()
    assert a.index_identity()["digest"] != b.index_identity()["digest"]
    ids = {s.model_id() for s in arms.arm_specs("st", "ll", "m2v")}
    assert len(ids) == len(arms.arm_specs("st", "ll", "m2v"))
    # none of the ids may trip the shipped BGE query-prefix table
    assert all(cs.query_text(i, "q") == "q" for i in ids)


def test_side_detection_uses_shipped_constants():
    assert arms.side_for(cs.QUERY_ADMISSION_WAIT_S) == "query"
    assert arms.side_for(cs.INDEX_ADMISSION_WAIT_S) == "document"


def test_truncate_renormalise_is_unit_norm():
    v = np.random.default_rng(0).normal(size=(5, 768)).astype("float32")
    t = arms.truncate_renormalise(v, 256)
    assert t.shape == (5, 256)
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-5)
    with pytest.raises(ValueError):
        arms.truncate_renormalise(v, 1024)


def test_document_and_query_prefixes_and_code_intent():
    ad = FakeAdapter(_spec(256))
    ad.embed(["def f(): pass"], timeout_s=1, admission_wait_s=cs.INDEX_ADMISSION_WAIT_S)
    assert ad.seen[-1] == ["title: none | text: def f(): pass"]
    ad.embed(["where is f"], timeout_s=1, admission_wait_s=cs.QUERY_ADMISSION_WAIT_S)
    assert ad.seen[-1] == ["task: search result | query: where is f"]
    with ad.query_intent(True):
        ad.embed(["where is f"], timeout_s=1, admission_wait_s=cs.QUERY_ADMISSION_WAIT_S)
    assert ad.seen[-1] == ["task: code retrieval | query: where is f"]
    assert ad.code_intent is False
    res = ad.embed(["x"], timeout_s=1, admission_wait_s=cs.INDEX_ADMISSION_WAIT_S)
    assert res.vectors.shape == (1, 256) and res.model_id == ad.model_id
    assert np.allclose(np.linalg.norm(res.vectors, axis=1), 1.0, atol=1e-5)


def test_nonfinite_is_an_invalid_result_not_a_vector():
    ad = FakeAdapter(_spec(768), nan=True)
    res = ad.embed(["a", "b"], timeout_s=1, admission_wait_s=cs.INDEX_ADMISSION_WAIT_S)
    assert res.vectors is None and res.reason == "degenerate"
    assert ad.nonfinite_rows == 1


def test_static_arm_applies_no_prefix():
    spec = next(s for s in arms.arm_specs("st", "ll", "m2v") if s.family == "static")
    ad = arms.Model2VecAdapter.__new__(arms.Model2VecAdapter)
    arms._AdapterBase.__init__(ad, spec)
    assert ad._prefixed(["hello"], "document") == ["hello"]
    assert ad._prefixed(["hello"], "query") == ["hello"]


def test_adapter_drives_the_shipped_index_end_to_end():
    pytest.importorskip("src.trace.navigation")
    sections = [
        SimpleNamespace(name="s", text="\n".join(f"line {i} alpha beta gamma" for i in range(200)))
    ]
    idx = cs.ContextSearchIndex(sections)
    ad = FakeAdapter(_spec(256))
    hits, meta = idx.search("alpha beta", 3, section=None, embedder=ad)
    assert meta["mode"] == cs.MODE_HYBRID and hits
    # chunks were embedded as documents, the query as a query, through the unchanged index
    assert all(t.startswith("title: none | text: ") for batch in ad.seen[:-1] for t in batch)
    assert ad.seen[-1][0].startswith("task: search result | query: ")
    assert idx.model_id == ad.model_id
    # a different arm never reuses these vectors (invariant 4)
    other = FakeAdapter(_spec(768))
    idx.search("alpha beta", 3, section=None, embedder=other)
    assert idx.model_id == other.model_id and idx.vectors.shape[1] == 768
