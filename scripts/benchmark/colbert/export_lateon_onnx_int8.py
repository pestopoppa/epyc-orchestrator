"""Prepare ColBERT ONNX INT8 artifacts + validate parity vs PyLate reference.

LightOn ships LateOn (`lightonai/LateOn`) with both FP32 (`model.onnx`) and INT8
(`model_int8.onnx`) ONNX files on the HF Hub alongside tokenizer + config.
No local re-export is needed — we download the pre-quantized artifact and
validate that its embeddings match PyLate's reference (which loads the
safetensors backbone directly).

Reason-mxbai (`DataScience-UIBK/Reason-mxbai-colbert-v0-32m`) does not ship
ONNX artifacts on the HF Hub. This helper can download and verify the source
artifact set, then fail fast with an actionable message until a local
Torch->ONNX->INT8 export step has populated `model_int8.onnx`.

Run once to populate `/mnt/raid0/llm/models/lateon-onnx-int8/`, then flip the
orchestrator's `LATEON_MODEL_PATH` env var to activate LateOn in
`src/tools/web/colbert_reranker.py`. For Reason-mxbai, populate
`/mnt/raid0/llm/models/reason-mxbai-colbert-v0-32m-onnx-int8/`, then set
`REASON_MXBAI_MODEL_PATH`.

Dependencies: huggingface_hub, onnxruntime, numpy (orchestrator .venv).
Parity test also needs: torch, transformers, pylate (colbert-export extras
or the pre-existing .venv-reranker site-packages).

Usage:
    # Download + parity validate
    python -m scripts.benchmark.colbert.export_lateon_onnx_int8 \
        --out /mnt/raid0/llm/models/lateon-onnx-int8

    # Download Reason-mxbai source artifacts only
    python -m scripts.benchmark.colbert.export_lateon_onnx_int8 \
        --profile reason-mxbai --no-parity

    # Print the artifact plan without network access
    python -m scripts.benchmark.colbert.export_lateon_onnx_int8 \
        --profile reason-mxbai --print-plan --json

    # Skip parity (download only)
    python -m scripts.benchmark.colbert.export_lateon_onnx_int8 --no-parity
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("colbert.export")

REPO_ID = "lightonai/LateOn"
DEFAULT_OUT = Path("/mnt/raid0/llm/models/lateon-onnx-int8")
REASON_MXBAI_REPO_ID = "DataScience-UIBK/Reason-mxbai-colbert-v0-32m"
REASON_MXBAI_DEFAULT_OUT = Path(
    "/mnt/raid0/llm/models/reason-mxbai-colbert-v0-32m-onnx-int8"
)

# Files required for inference (reranker consumes only model_int8.onnx + tokenizer.json + config).
# model.onnx (FP32) and safetensors kept for parity validation.
LATEON_DOWNLOAD_FILES = (
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
)

REASON_MXBAI_SOURCE_FILES = (
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Dense/config.json",
    "1_Dense/model.safetensors",
    "2_Dense/config.json",
    "2_Dense/model.safetensors",
    "3_Dense/config.json",
    "3_Dense/model.safetensors",
)

# Backward-compatible public names used by earlier handoffs and shell snippets.
DOWNLOAD_FILES = list(LATEON_DOWNLOAD_FILES)

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
REQUIRED_RUNTIME_FILES = ("model_int8.onnx", "tokenizer.json")


@dataclass(frozen=True)
class ModelArtifactProfile:
    """Artifact manifest for a ColBERT reranker candidate."""

    name: str
    repo_id: str
    default_out: Path
    download_files: tuple[str, ...]
    ships_prebuilt_onnx: bool
    model_slot_env: str
    notes: str


MODEL_PROFILES = {
    "lateon": ModelArtifactProfile(
        name="lateon",
        repo_id=REPO_ID,
        default_out=DEFAULT_OUT,
        download_files=LATEON_DOWNLOAD_FILES,
        ships_prebuilt_onnx=True,
        model_slot_env="LATEON_MODEL_PATH",
        notes="HF repo ships model_int8.onnx and model.onnx; parity can run after download.",
    ),
    "reason-mxbai": ModelArtifactProfile(
        name="reason-mxbai",
        repo_id=REASON_MXBAI_REPO_ID,
        default_out=REASON_MXBAI_DEFAULT_OUT,
        download_files=REASON_MXBAI_SOURCE_FILES,
        ships_prebuilt_onnx=False,
        model_slot_env="REASON_MXBAI_MODEL_PATH",
        notes=(
            "HF repo is source-only; download prepares safetensors/tokenizer/configs, "
            "then a separate local Torch->ONNX->INT8 export must create model_int8.onnx."
        ),
    ),
}

PROFILE_BY_REPO_ID = {profile.repo_id: profile for profile in MODEL_PROFILES.values()}


def resolve_profile(profile_name: str, model_id: str | None = None) -> ModelArtifactProfile:
    """Resolve a known profile, optionally overriding its HF repo id."""
    if profile_name not in MODEL_PROFILES:
        raise ValueError(f"unknown profile: {profile_name}")

    profile = MODEL_PROFILES[profile_name]
    if model_id is None or model_id == profile.repo_id:
        return profile

    known = PROFILE_BY_REPO_ID.get(model_id)
    if known is not None:
        return known

    return ModelArtifactProfile(
        name=f"custom:{model_id}",
        repo_id=model_id,
        default_out=profile.default_out,
        download_files=profile.download_files,
        ships_prebuilt_onnx=profile.ships_prebuilt_onnx,
        model_slot_env=profile.model_slot_env,
        notes=f"Custom repo id using the {profile.name} artifact manifest.",
    )


def artifact_plan(profile: ModelArtifactProfile, out_dir: Path) -> dict[str, object]:
    """Return a serializable plan for operators and tests."""
    return {
        "profile": profile.name,
        "repo_id": profile.repo_id,
        "out": str(out_dir),
        "ships_prebuilt_onnx": profile.ships_prebuilt_onnx,
        "model_slot_env": profile.model_slot_env,
        "runtime_files": list(REQUIRED_RUNTIME_FILES),
        "download_files": list(profile.download_files),
        "notes": profile.notes,
    }


def missing_files(out_dir: Path, rel_paths: tuple[str, ...] | list[str]) -> list[str]:
    """Return expected artifact paths that are absent from out_dir."""
    return [rel_path for rel_path in rel_paths if not (out_dir / rel_path).exists()]


class OnnxArtifactsMissingError(RuntimeError):
    """Raised when parity/runtime checks need ONNX files that are absent."""


def download(out_dir: Path, profile: ModelArtifactProfile | None = None) -> None:
    """Download profile weights + tokenizer + configs from HF Hub into out_dir."""
    from huggingface_hub import hf_hub_download

    profile = profile or MODEL_PROFILES["lateon"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s (%s) to %s", profile.name, profile.repo_id, out_dir)

    for rel_path in profile.download_files:
        log.info("  fetching %s", rel_path)
        hf_hub_download(
            repo_id=profile.repo_id,
            filename=rel_path,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )

    missing_downloads = missing_files(out_dir, profile.download_files)
    if missing_downloads:
        raise FileNotFoundError(f"Downloaded artifact set incomplete: {missing_downloads}")

    if profile.ships_prebuilt_onnx:
        ensure_onnx_artifacts(out_dir, profile)
    else:
        log.warning(
            "%s is source-only on HF; %s must be created by local export before parity/runtime.",
            profile.repo_id,
            out_dir / "model_int8.onnx",
        )
    log.info("Download complete. Files in %s", out_dir)


def ensure_onnx_artifacts(
    model_dir: Path,
    profile: ModelArtifactProfile | None = None,
) -> None:
    """Require files consumed by the ONNX reranker and parity test."""
    profile = profile or MODEL_PROFILES["lateon"]
    missing_runtime = missing_files(model_dir, REQUIRED_RUNTIME_FILES)
    if not missing_runtime:
        return

    if profile.ships_prebuilt_onnx:
        raise OnnxArtifactsMissingError(
            f"{profile.repo_id} should ship ONNX artifacts but {missing_runtime} "
            f"are missing from {model_dir}"
        )

    raise OnnxArtifactsMissingError(
        f"{profile.repo_id} does not ship prebuilt ONNX artifacts. Missing "
        f"{missing_runtime} in {model_dir}. Download source artifacts with "
        "--no-parity, then run the local Torch->ONNX->INT8 export step before "
        "parity or reranker latency benchmarking."
    )


def _pooled_vec(per_token):
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


def parity_test(
    model_dir: Path,
    profile: ModelArtifactProfile | None = None,
) -> bool:
    """Compare ONNX-INT8 embeddings vs PyLate reference on PARITY_SNIPPETS.

    Returns True if max per-snippet |cosine - 1| <= PARITY_TOLERANCE.
    """
    import numpy as np

    profile = profile or MODEL_PROFILES["lateon"]
    ensure_onnx_artifacts(model_dir, profile)
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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        choices=sorted(MODEL_PROFILES),
        default="lateon",
        help="Known artifact profile to use",
    )
    ap.add_argument(
        "--model-id",
        help="Override HF repo id; known repo ids auto-select their profile",
    )
    ap.add_argument("--out", type=Path, help="Output directory")
    ap.add_argument("--no-download", action="store_true", help="Skip HF download")
    ap.add_argument("--no-parity", action="store_true", help="Skip parity validation")
    ap.add_argument("--print-plan", action="store_true", help="Print artifact plan and exit")
    ap.add_argument("--json", action="store_true", help="Emit JSON for --print-plan")
    args = ap.parse_args(argv)

    profile = resolve_profile(args.profile, args.model_id)
    out_dir = args.out or profile.default_out

    if args.print_plan:
        plan = artifact_plan(profile, out_dir)
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True))
        else:
            for key, value in plan.items():
                print(f"{key}: {value}")
        return 0

    if not args.no_download:
        download(out_dir, profile)

    if args.no_parity:
        log.info("Parity skipped.")
        return 0

    try:
        ok = parity_test(out_dir, profile)
    except ImportError as e:
        log.error("Parity test requires torch + transformers + pylate: %s", e)
        log.error("Install via: pip install -e '.[colbert-export]' OR reuse .venv-reranker site-packages via PYTHONPATH.")
        return 2
    except OnnxArtifactsMissingError as e:
        log.error("%s", e)
        return 3

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
