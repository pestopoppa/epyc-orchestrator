"""Assert-bearing demo/tests for the GitNexus <-> KB federation tool.

Runs the two VERIFY demos from the handoff task. Degrades gracefully (skips
with a reason) when the live ColBERT index / encoder or the gitnexus CLI is not
reachable in this environment — never fabricates results.

Run:
    PYTHONPATH=/mnt/raid0/llm/venv/lib/python3.12/site-packages \
      /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python \
      -m pytest tests/retrieval/test_federation.py -v
or directly as a script:
    ... federation demo ... python tests/retrieval/test_federation.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_ORCH_ROOT = Path(__file__).resolve().parents[2]
if str(_ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCH_ROOT))

from src.retrieval import federation  # noqa: E402


def _gitnexus_available() -> bool:
    return shutil.which(federation._GITNEXUS_BIN) is not None


def test_extract_identifiers_pure_no_deps():
    """Identifier extraction is pure — always runs, no index/CLI needed."""
    ids = federation.extract_identifiers(
        "kb_rag.query enriches the maxsim score in colbert_encoder.py"
    )
    assert "kb_rag.query" in ids
    assert any("colbert_encoder" in i for i in ids)
    # bland english words are dropped
    assert "enriches" not in ids


def test_symbol_to_kb_kb_rag_query():
    """--symbol kb_rag.query -> KB chunks about KB-RAG / ColBERT / retrieval."""
    res = federation.symbol_to_kb("kb_rag.query", top_k=6)
    assert res["direction"] == "symbol_to_kb"
    assert res["query_text"], "expected a non-empty federated query text"

    enc = res["encoder_status"]
    if not enc["kb_queryable"]:
        import pytest
        pytest.skip(f"KB not queryable in this env: {enc}")

    hits = res["kb_hits"]
    assert hits, "expected non-empty KB hits for kb_rag.query"
    blob = " ".join(
        (h["file"] or "") + " " + h["heading"] + " " + h["snippet"] for h in hits
    ).lower()
    assert any(t in blob for t in ("kb-rag", "kb_rag", "colbert", "retriev")), (
        f"KB hits should discuss KB-RAG/ColBERT/retrieval; got {blob[:300]!r}"
    )


def test_doc_to_code_iqk_enablement():
    """--doc 'iqk IQ-quant enablement' -> code symbols/files, or clean empty."""
    if not _gitnexus_available():
        import pytest
        pytest.skip("gitnexus CLI not on PATH")

    res = federation.doc_to_code("iqk IQ-quant enablement", top_k=8)
    assert res["direction"] == "doc_to_code"
    # Either we get code hits, or a clean explanatory note (never a crash).
    if res["code_hits"]:
        for h in res["code_hits"]:
            assert h.get("file") and h.get("name")
    else:
        assert res["notes"], "empty result must carry a reason"


if __name__ == "__main__":
    # Script mode: run both demos and print a summary.
    print("=== encoder / index status ===")
    print(federation.encoder_status())

    print("\n=== DEMO 1: --symbol kb_rag.query ===")
    r1 = federation.symbol_to_kb("kb_rag.query", top_k=6)
    print(federation._format_symbol_to_kb(r1))

    print("\n=== DEMO 2: --doc 'iqk IQ-quant enablement' ===")
    r2 = federation.doc_to_code("iqk IQ-quant enablement", top_k=8)
    print(federation._format_doc_to_code(r2))

    # Lightweight assertions for the script path.
    assert r1["query_text"]
    assert r2["direction"] == "doc_to_code"
    print("\nOK: both demos ran (see notes for any graceful degradation).")
