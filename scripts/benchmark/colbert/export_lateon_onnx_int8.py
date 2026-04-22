"""Download LateOn INT8 ONNX + validate parity vs PyLate reference (NIB2-47).

LightOn ships LateOn (`lightonai/LateOn`) with both FP32 (`model.onnx`) and INT8
(`model_int8.onnx`) ONNX files on the HF Hub alongside tokenizer + config.
No local re-export is needed — we download the pre-quantized artifact and
validate that its embeddings match PyLate's reference (which loads the
safetensors backbone directly).

Run once to populate `/mnt/raid0/llm/models/lateon-onnx-int8/`, then flip the
orchestrator's `LATEON_MODEL_PATH` env var to activate LateOn in
`src/tools/web/colbert_reranker.py`.

Dependencies: huggingface_hub, onnxruntime, numpy (orchestrator .venv).
Parity test also needs: torch, transformers, pylate (colbert-export extras
or the pre-existing .venv-reranker site-packages).

Usage:
    # Download + parity validate
    python -m scripts.benchmark.colbert.export_lateon_onnx_int8 \
        --out /mnt/raid0/llm/models/lateon-onnx-int8

    # Skip parity (download only)
    python -m scripts.benchmark.colbert.export_lateon_onnx_int8 --no-parity
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("colbert.export_lateon")

REPO_ID = "lightonai/LateOn"
DEFAULT_OUT = Path("/mnt/raid0/llm/models/lateon-onnx-int8")

# Files required for inference (reranker consumes only model_int8.onnx + tokenizer.json + config).
# model.onnx (FP32) and safetensors kept for parity validation.
DOWNLOAD_FILES = [
    "model_int8.onnx",
    "model.onnx",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "config_sentence_transformers.json",
    "onnx_config.json",
    "modules.json",
    "sentence_bert_config.json",
    # Dense projection heads (ColBERT late-interaction layers).
    "1_Dense/config.json",
    "1_Dense/model.safetensors",
    "2_Dense/config.json",
    "2_Dense/model.safetensors",
    "3_Dense/config.json",
    "3_Dense/model.safetensors",
]

PARITY_SNIPPETS = [
    "The northern lights are caused by solar particles interacting with Earth's magnetosphere.",
    "Python list comprehensions offer a concise syntax for transforming iterables.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "The Schrödinger equation describes how the quantum state of a physical system evolves.",
    "Machine learning models generalize from training data to make predictions on unseen inputs.",
    "The French Revolution began in 1789 with the storming of the Bastille.",
    "DNA replication is semi-conservative, preserving one strand from the parent molecule.",
    "Plate tectonics explains continental drift and the formation of mountain ranges.",
    "Neural networks use backpropagation to adjust weights during training.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "Honeybees communicate the location of nectar sources via the waggle dance.",
    "Black holes have event horizons beyond which light cannot escape.",
    "The mitochondrion is the powerhouse of the eukaryotic cell.",
    "SQL joins combine rows from two or more tables based on a related column.",
    "The periodic table organizes elements by atomic number and chemical properties.",
    "Climate models project warming based on greenhouse gas concentrations.",
    "Cryptographic hash functions are deterministic and one-way.",
    "The human genome contains approximately 3 billion base pairs.",
    "Supernovas forge elements heavier than iron through nucleosynthesis.",
    "Protein folding determines a molecule's biological function.",
]

PARITY_TOLERANCE = 1e-2  # max |cosine(ref, int8) - 1| per snippet


def download(out_dir: Path) -> None:
    """Download LateOn weights + tokenizer + configs from HF Hub into out_dir."""
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s to %s", REPO_ID, out_dir)

    for rel_path in DOWNLOAD_FILES:
        log.info("  fetching %s", rel_path)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=rel_path,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )

    assert (out_dir / "model_int8.onnx").exists(), "model_int8.onnx missing after download"
    log.info("Download complete. Files in %s", out_dir)


def _pooled_vec(per_token: "np.ndarray") -> "np.ndarray":
    """Mean-pool per-token embeddings into a single vector for cosine comparison."""
    import numpy as np

    return per_token.mean(axis=0) / (np.linalg.norm(per_token.mean(axis=0)) + 1e-8)


def _encode_onnx(onnx_path: Path, tokenizer_path: Path, texts: list[str]) -> list:
    """Run ONNX model on each text, return list of per-token 128-dim arrays."""
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.enable_truncation(max_length=64)
    tok.enable_padding(length=64)

    out = []
    for text in texts:
        enc = tok.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        result = sess.run(None, feeds)[0][0]  # (seq, 128)
        mask = np.array(enc.attention_mask, dtype=bool)
        out.append(result[mask])
    return out


def _encode_pylate(model_dir: Path, texts: list[str]) -> list:
    """Run PyLate reference ColBERT encoder on texts, return per-token arrays."""
    from pylate import models

    model = models.ColBERT(model_name_or_path=str(model_dir))
    embeds = model.encode(texts, is_query=False, show_progress_bar=False)
    return [e.cpu().numpy() if hasattr(e, "cpu") else e for e in embeds]


def parity_test(model_dir: Path) -> bool:
    """Compare ONNX-INT8 embeddings vs PyLate reference on PARITY_SNIPPETS.

    Returns True if max per-snippet |cosine - 1| <= PARITY_TOLERANCE.
    """
    import numpy as np

    int8_embeds = _encode_onnx(
        model_dir / "model_int8.onnx",
        model_dir / "tokenizer.json",
        PARITY_SNIPPETS,
    )
    ref_embeds = _encode_pylate(model_dir, PARITY_SNIPPETS)

    divergences = []
    for i, (int8, ref) in enumerate(zip(int8_embeds, ref_embeds)):
        v_int8 = _pooled_vec(int8)
        v_ref = _pooled_vec(ref[: len(int8)])  # pylate may use different padding
        cos = float(np.dot(v_int8, v_ref))
        divergences.append(abs(1.0 - cos))

    max_div = max(divergences)
    mean_div = sum(divergences) / len(divergences)
    log.info("PARITY: max |cos - 1| = %.4e, mean = %.4e (N=%d)",
             max_div, mean_div, len(PARITY_SNIPPETS))

    ok = max_div <= PARITY_TOLERANCE
    marker = "✓" if ok else "✗"
    log.info("MAX_COS_DIV: %.4e (tol %.0e) %s", max_div, PARITY_TOLERANCE, marker)
    return ok


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    ap.add_argument("--no-download", action="store_true", help="Skip HF download")
    ap.add_argument("--no-parity", action="store_true", help="Skip parity validation")
    args = ap.parse_args()

    if not args.no_download:
        download(args.out)

    if args.no_parity:
        log.info("Parity skipped.")
        return 0

    try:
        ok = parity_test(args.out)
    except ImportError as e:
        log.error("Parity test requires torch + transformers + pylate: %s", e)
        log.error("Install via: pip install -e '.[colbert-export]' OR reuse .venv-reranker site-packages via PYTHONPATH.")
        return 2

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
