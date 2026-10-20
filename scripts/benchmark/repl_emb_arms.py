#!/usr/bin/env python3
from __future__ import annotations

"""REPL-EMB-2.5: candidate-arm specs, index identity and ``DenseEmbedder`` adapters.

One adapter per extension arm of the REPL-EMB-2.2 launch (EmbeddingGemma 2 @768 / @256 through
sentence-transformers or a harness-owned llama-server, and the two static Model2Vec arms). Every
adapter implements the shipped ``context_search.DenseEmbedder`` seam
(``model_id`` + ``embed(texts, *, timeout_s, admission_wait_s) -> DenseResult``); BM25, the chunker,
``rrf_fuse`` and ``ContextSearchIndex`` are the shipped code, loaded unchanged and never forked.

Why the adapter prefixes the document side: ``ContextSearchIndex`` only prefixes QUERIES, through
``query_text`` and the BGE-only ``_QUERY_PREFIXES`` table, and sends documents with no prefix.
EmbeddingGemma 2 is asymmetric (``task: search result | query: `` vs ``title: none | text: ``), so
the adapter applies BOTH prefixes itself. It tells the two calls apart by the ``admission_wait_s``
argument the shipped index passes (``INDEX_ADMISSION_WAIT_S`` for chunk batches,
``QUERY_ADMISSION_WAIT_S`` for the one query); ``side_for`` asserts the two constants differ, so a
future edit that makes them equal fails loudly instead of silently mis-prefixing.

No vector from one arm is ever scored against another's index: ``model_id`` is the arm's index
identity string, ``ContextSearchIndex.ensure_dense`` resets on a mismatch, and
``index_identity`` carries (model, revision, dim, dtype/quant, prompt hash, runtime).
"""

import hashlib
import importlib.util
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

# Prompt strings (config_sentence_transformers.json of google/embeddinggemma-2 @ 914f7f8).
EG2_QUERY_PROMPT = "task: search result | query: "
EG2_CODE_QUERY_PROMPT = "task: code retrieval | query: "
EG2_DOC_PROMPT = "title: none | text: "


def load_context_search() -> Any:
    """The shipped ``src/repl_environment/context_search.py``, loaded by file path.

    ``src.repl_environment`` imports the whole REPL stack in its ``__init__``; this harness runs
    in a slim isolated venv, so the module is loaded directly (it only needs numpy). Unchanged
    code, a different import path."""
    name = "epyc_context_search_shipped"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO_ROOT / "src" / "repl_environment" / "context_search.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def l2_normalise(mat: Any) -> Any:
    import numpy as np

    a = np.asarray(mat, dtype=np.float32)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    return a / n


def truncate_renormalise(mat: Any, dim: int) -> Any:
    """Matryoshka: keep the leading ``dim`` columns, then L2 re-normalise (the card requires it)."""
    import numpy as np

    a = np.asarray(mat, dtype=np.float32)
    if a.shape[-1] < dim:
        raise ValueError(f"cannot truncate {a.shape[-1]}-d vectors to {dim}")
    return l2_normalise(a[..., :dim])


# ───────────────────────────────────────────────────────────────────────────── arm specs


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    family: str  # "st" | "gguf" | "static"
    model_repo: str
    revision: str
    dim: int
    dtype: str  # "fp32" | "bf16" | "q8_0" | "static-fp32"
    runtime: str  # e.g. "sentence-transformers==6.1.0/transformers==5.20.1/torch-cpu"
    pooling: str
    query_prompt: str = ""
    code_query_prompt: str = ""
    doc_prompt: str = ""
    normalise: str = "l2-after-truncation"
    native_dim: int = 768
    notes: str = ""

    def prompt_hash(self) -> str:
        return sha256_text(
            json.dumps(
                {"q": self.query_prompt, "cq": self.code_query_prompt, "d": self.doc_prompt},
                sort_keys=True,
            )
        )

    def index_identity(self) -> dict[str, Any]:
        """The key under which this arm's vectors may be scored. Any field change is a new index."""
        ident = {
            "model": self.model_repo,
            "revision": self.revision,
            "dim": self.dim,
            "dtype": self.dtype,
            "prompt_hash": self.prompt_hash(),
            "runtime": self.runtime,
        }
        ident["digest"] = sha256_text(json.dumps(ident, sort_keys=True))
        return ident

    def model_id(self) -> str:
        """``DenseEmbedder.model_id``: arm id + identity digest prefix. Contains no BGE marker, so
        the shipped ``query_text`` leaves queries untouched (the adapter prefixes them)."""
        return f"{self.arm_id}#{self.index_identity()['digest'][:12]}"


EG2_REPO = "google/embeddinggemma-2"
EG2_REV = "914f7f89142e33e77833254d9c9b90c3cef7303b"
EG2_GGUF_REPO = "ggml-org/embeddinggemma-2-GGUF"
EG2_GGUF_REV = "58faf90940886df2d62be555c8765d00dac362c5"
POTION_CODE_REPO, POTION_CODE_REV = (
    "minishlab/potion-code-16M-v2",
    "e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b",
)
POTION_BASE_REPO, POTION_BASE_REV = (
    "minishlab/potion-base-8M",
    "bf8b056651a2c21b8d2565580b8569da283cab23",
)


def arm_specs(st_runtime: str, llama_runtime: str, m2v_runtime: str) -> list[ArmSpec]:
    """The extension arms of REPL-EMB-2.7, as fixed by REPL-EMB-2.5(c)."""
    eg2 = dict(
        pooling="mean",
        query_prompt=EG2_QUERY_PROMPT,
        code_query_prompt=EG2_CODE_QUERY_PROMPT,
        doc_prompt=EG2_DOC_PROMPT,
    )
    out: list[ArmSpec] = []
    for dim in (768, 256):
        out.append(
            ArmSpec(f"eg2-st-fp32-{dim}", "st", EG2_REPO, EG2_REV, dim, "fp32", st_runtime, **eg2)
        )
    out.append(
        ArmSpec(
            "eg2-st-bf16-768",
            "st",
            EG2_REPO,
            EG2_REV,
            768,
            "bf16",
            st_runtime,
            notes="parity reference arm only (BF16 vs FP32 gate); not a 2.7 recall arm",
            **eg2,
        )
    )
    for q, tag in (("bf16", "BF16"), ("q8_0", "Q8_0")):
        for dim in (768, 256):
            out.append(
                ArmSpec(
                    f"eg2-gguf-{q}-{dim}",
                    "gguf",
                    f"{EG2_GGUF_REPO}/embeddinggemma-2-{tag}.gguf",
                    EG2_GGUF_REV,
                    dim,
                    q,
                    llama_runtime,
                    notes="exists only if REPL-EMB-2.6 passes its parity gate",
                    **eg2,
                )
            )
    out.append(
        ArmSpec(
            "potion-code-16m-v2",
            "static",
            POTION_CODE_REPO,
            POTION_CODE_REV,
            256,
            "static-fp32",
            m2v_runtime,
            pooling="mean-of-token-lookup",
            normalise="l2",
            native_dim=256,
        )
    )
    out.append(
        ArmSpec(
            "potion-base-8m",
            "static",
            POTION_BASE_REPO,
            POTION_BASE_REV,
            256,
            "static-fp32",
            m2v_runtime,
            pooling="mean-of-token-lookup",
            normalise="l2",
            native_dim=256,
        )
    )
    return out


# ───────────────────────────────────────────────────────────────────────────── adapters


def side_for(admission_wait_s: float) -> str:
    """'query' or 'document' for one shipped ``embed`` call (see module docstring)."""
    cs = load_context_search()
    assert cs.QUERY_ADMISSION_WAIT_S != cs.INDEX_ADMISSION_WAIT_S, (
        "context_search no longer distinguishes query and index admission waits; "
        "the adapters' side detection is void — re-derive it before running any arm"
    )
    return "query" if admission_wait_s == cs.QUERY_ADMISSION_WAIT_S else "document"


class _AdapterBase:
    """Shared prefixing, truncation, finiteness accounting and the DenseEmbedder contract."""

    spec: ArmSpec

    def __init__(self, spec: ArmSpec) -> None:
        self.spec = spec
        self.model_id = spec.model_id()
        self.code_intent = False  # the harness sets this per query from the blind label
        self.calls: list[dict[str, Any]] = []
        self.nonfinite_rows = 0

    @contextmanager
    def query_intent(self, code_intent: bool) -> Iterator[None]:
        prev, self.code_intent = self.code_intent, bool(code_intent)
        try:
            yield
        finally:
            self.code_intent = prev

    def _prefixed(self, texts: Sequence[str], side: str) -> list[str]:
        s = self.spec
        if side == "query":
            p = (
                s.code_query_prompt
                if (self.code_intent and s.code_query_prompt)
                else s.query_prompt
            )
        else:
            p = s.doc_prompt
        return [p + t for t in texts]

    def _encode(self, texts: list[str]) -> Any:  # (N, native_dim) float32, unit or not
        raise NotImplementedError

    def encode_side(self, texts: Sequence[str], side: str) -> Any:
        raw = self._encode(self._prefixed(texts, side))
        import numpy as np

        raw = np.asarray(raw, dtype=np.float32)
        bad = ~np.isfinite(raw).all(axis=1)
        self.nonfinite_rows += int(bad.sum())
        return truncate_renormalise(raw, self.spec.dim)

    def embed(self, texts: Sequence[str], *, timeout_s: float, admission_wait_s: float) -> Any:
        cs = load_context_search()
        side = side_for(admission_wait_s)
        t0 = time.perf_counter()
        try:
            vec = self.encode_side(list(texts), side)
        except Exception as exc:  # noqa: BLE001 -- DenseEmbedder contract: a failure is a labelled miss
            return cs.DenseResult(None, self.model_id, type(exc).__name__)
        elapsed = time.perf_counter() - t0
        self.calls.append({"side": side, "n": len(texts), "s": round(elapsed, 4)})
        import numpy as np

        if not np.isfinite(vec).all():
            # A non-finite row is an invalid result, never a silent zero: the build records it
            # as `degenerate` and the query counts as a miss (2.7 counts invalid as a miss).
            return cs.DenseResult(None, self.model_id, "degenerate")
        return cs.DenseResult(vec, self.model_id)


class SentenceTransformersAdapter(_AdapterBase):
    """EmbeddingGemma 2 through sentence-transformers on CPU (FP32 default; BF16 reference arm)."""

    def __init__(
        self, spec: ArmSpec, model_path: str, *, text_only: bool = True, threads: int | None = None
    ):
        super().__init__(spec)
        import torch
        from sentence_transformers import SentenceTransformer

        if threads:
            torch.set_num_threads(int(threads))
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[spec.dtype]  # never fp16
        cfg = {"vision_config": None, "audio_config": None} if text_only else {}
        self.model = SentenceTransformer(
            model_path, device="cpu", model_kwargs={"torch_dtype": dtype}, config_kwargs=cfg
        )
        self.max_seq_length = getattr(self.model, "max_seq_length", None)

    def _encode(self, texts: list[str]) -> Any:
        # Full 768-d vector (Normalize module included); truncation + re-normalisation is ours.
        return self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype("float32")


class Model2VecAdapter(_AdapterBase):
    """Static-embedding arms: tokenise, look up, mean, L2. No prefixes, no forward pass."""

    def __init__(self, spec: ArmSpec, model_path: str):
        super().__init__(spec)
        from model2vec import StaticModel

        self.model = StaticModel.from_pretrained(model_path)

    def _prefixed(self, texts: Sequence[str], side: str) -> list[str]:
        return list(texts)

    def _encode(self, texts: list[str]) -> Any:
        return self.model.encode(texts, show_progress_bar=False)


class LlamaServerAdapter(_AdapterBase):
    """A harness-owned llama-server (embedding mode, mean pooling) over ``/v1/embeddings``.

    The server is started and stopped by the caller (PIDs captured there); this class only
    speaks HTTP. ``--embd-normalize`` is recorded by the caller in the run config; client-side
    truncation + L2 re-normalisation are applied here either way."""

    def __init__(self, spec: ArmSpec, base_url: str, *, timeout_s: float = 120.0):
        super().__init__(spec)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _encode(self, texts: list[str]) -> Any:
        import urllib.request

        import numpy as np

        body = json.dumps({"input": texts, "encoding_format": "float"}).encode()
        req = urllib.request.Request(
            self.base_url + "/v1/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:  # noqa: S310 -- local harness server
            payload = json.loads(r.read())
        rows = sorted(payload["data"], key=lambda d: d["index"])
        if len(rows) != len(texts):
            raise ValueError(f"server returned {len(rows)} rows for {len(texts)} texts")
        return np.asarray([d["embedding"] for d in rows], dtype=np.float32)


__all__ = [
    "ArmSpec",
    "EG2_CODE_QUERY_PROMPT",
    "EG2_DOC_PROMPT",
    "EG2_QUERY_PROMPT",
    "LlamaServerAdapter",
    "Model2VecAdapter",
    "SentenceTransformersAdapter",
    "arm_specs",
    "l2_normalise",
    "load_context_search",
    "side_for",
    "truncate_renormalise",
]
