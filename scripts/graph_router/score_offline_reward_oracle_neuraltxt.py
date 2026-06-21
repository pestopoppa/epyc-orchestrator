#!/usr/bin/env python3
"""Score offline reward-oracle rows with AVB's NeuralTxt reward checkpoint.

This is the scorer side of the A9 learned-routing offline quality-oracle lane.
It consumes JSONL rows with `reference` and `response`, adds `oracle_score`,
and leaves live routing untouched. The default checkpoint is the public
`paperbd/neuraltxt-reward-tiny` MiniLM reward model described in the AVB deep
dive. The model dependencies are optional on purpose; run this script with
`torch`, `transformers`, and `huggingface_hub` installed or supplied via `uv
run --with ...`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Protocol


DEFAULT_MODEL_ID = "paperbd/neuraltxt-reward-tiny"
ORACLE_SCORE_SOURCE = "neuraltxt_reward_tiny"


class PairScorer(Protocol):
    def score(self, reference: str, response: str) -> float:
        """Return a scalar score in [0, 1] for one reference/response pair."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(raw)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _text(row: dict[str, Any], field: str) -> str:
    return str(row.get(field) or "").strip()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class NeuralTxtRewardScorer:
    """Small wrapper around the published NeuralTxt reward model card recipe."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: Path | None = None,
        local_files_only: bool = False,
        device: str = "cpu",
    ) -> None:
        try:
            torch = importlib.import_module("torch")
            nn = importlib.import_module("torch.nn")
            transformers = importlib.import_module("transformers")
            hf_hub = importlib.import_module("huggingface_hub")
        except ModuleNotFoundError as exc:
            missing = exc.name or "optional dependency"
            raise RuntimeError(
                f"{missing} is required for NeuralTxt scoring. "
                "Run with optional deps, for example: "
                "uv run --with torch --with transformers --with safetensors "
                "--with huggingface-hub python scripts/graph_router/"
                "score_offline_reward_oracle_neuraltxt.py ..."
            ) from exc

        cache = str(cache_dir) if cache_dir else None
        self._torch = torch
        self._device = torch.device(device)
        self._encoder = transformers.AutoModel.from_pretrained(
            model_id,
            cache_dir=cache,
            local_files_only=local_files_only,
        ).to(self._device)
        self._encoder.eval()
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache,
            local_files_only=local_files_only,
        )
        head_path = hf_hub.hf_hub_download(
            model_id,
            filename="head_weights.pt",
            cache_dir=cache,
            local_files_only=local_files_only,
        )
        hidden_size = int(getattr(self._encoder.config, "hidden_size", 384))
        self._head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden_size * 2, 1)).to(
            self._device
        )
        try:
            state = torch.load(head_path, map_location=self._device, weights_only=True)
        except TypeError:
            state = torch.load(head_path, map_location=self._device)
        self._head.load_state_dict(state)
        self._head.eval()

    def _meanmax_pool(self, hidden: Any, mask: Any) -> Any:
        mask_f = mask.unsqueeze(-1).float()
        mean = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        masked = hidden.masked_fill(mask_f == 0, float("-inf"))
        max_pool = masked.max(1).values
        return self._torch.cat([mean, max_pool], dim=-1)

    def score(self, reference: str, response: str) -> float:
        text = f"{reference} [SEP] {response}"
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            output = self._encoder(**encoded)
            pooled = self._meanmax_pool(output.last_hidden_state, encoded["attention_mask"])
            raw_score = self._head(pooled).item()
        return _clamp01(raw_score)


def build_scorer(
    *,
    model_id: str,
    cache_dir: Path | None,
    local_files_only: bool,
    device: str,
) -> PairScorer:
    return NeuralTxtRewardScorer(
        model_id=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        device=device,
    )


def score_rows(
    rows: Iterable[dict[str, Any]],
    scorer: PairScorer,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    stats = Counter()
    values: list[float] = []

    for index, row in enumerate(rows, start=1):
        reference = _text(row, "reference")
        response = _text(row, "response")
        if not reference:
            raise ValueError(f"row {index}: reference is required")
        if not response:
            raise ValueError(f"row {index}: response is required")
        if "oracle_score" in row and not overwrite:
            raise ValueError(f"row {index}: oracle_score already present; pass --overwrite")

        out = dict(row)
        score = scorer.score(reference, response)
        out["oracle_score"] = _clamp01(score)
        out["oracle_score_source"] = ORACLE_SCORE_SOURCE
        out["oracle_model_id"] = model_id
        scored.append(out)
        values.append(out["oracle_score"])
        stats["rows"] += 1
        if out.get("variant_type"):
            stats[f"variant_type:{out['variant_type']}"] += 1

    summary = {
        "schema_version": "offline_reward_oracle_neuraltxt_scores.v1",
        "model_id": model_id,
        "oracle_score_source": ORACLE_SCORE_SOURCE,
        "rows": int(stats["rows"]),
        "score_min": min(values) if values else None,
        "score_max": max(values) if values else None,
        "score_mean": (sum(values) / len(values)) if values else None,
        "stats": {key: int(value) for key, value in sorted(stats.items())},
    }
    return scored, summary


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score offline reward-oracle rows with paperbd/neuraltxt-reward-tiny",
    )
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    rows = load_jsonl(args.input_jsonl)
    scorer = build_scorer(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    scored, summary = score_rows(
        rows,
        scorer,
        model_id=args.model_id,
        overwrite=args.overwrite,
    )
    write_jsonl(scored, args.output_jsonl)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
