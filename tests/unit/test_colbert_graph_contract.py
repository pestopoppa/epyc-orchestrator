"""K2 fail-loud: declared capability flags must agree with the loaded graph."""

import json
from types import SimpleNamespace

import pytest

from src.retrieval import colbert_encoder as encoder


def _session(inputs, width):
    return SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name=n) for n in inputs],
        get_outputs=lambda: [SimpleNamespace(name="out", shape=["batch", "seq", width])],
    )


def _dir(tmp_path, **cfg):
    (tmp_path / "onnx_config.json").write_text(json.dumps(cfg))
    return tmp_path


TWO = ("input_ids", "attention_mask")
THREE = TWO + ("token_type_ids",)


def test_absent_declarations_are_accepted(tmp_path):
    encoder._validate_graph_contract(_dir(tmp_path), _session(TWO, 128))


def test_matching_declarations_are_accepted(tmp_path):
    cfg = _dir(tmp_path, uses_token_type_ids=True, embedding_dim=128)
    encoder._validate_graph_contract(cfg, _session(THREE, 128))
    encoder._validate_graph_contract(
        _dir(tmp_path, uses_token_type_ids=False, embedding_dim=128), _session(TWO, 128)
    )


@pytest.mark.parametrize("declared,inputs", [(True, TWO), (False, THREE)])
def test_token_type_ids_mismatch_refused(tmp_path, declared, inputs):
    with pytest.raises(RuntimeError, match="token_type_ids"):
        encoder._validate_graph_contract(
            _dir(tmp_path, uses_token_type_ids=declared), _session(inputs, 128)
        )


def test_embedding_dim_mismatch_refused(tmp_path):
    with pytest.raises(RuntimeError, match="embedding_dim=96"):
        encoder._validate_graph_contract(_dir(tmp_path, embedding_dim=96), _session(TWO, 128))


def test_dynamic_output_width_is_not_a_mismatch(tmp_path):
    encoder._validate_graph_contract(_dir(tmp_path, embedding_dim=128), _session(TWO, "w"))


@pytest.mark.parametrize(
    "cfg", [{"uses_token_type_ids": "false"}, {"embedding_dim": "128"}, {"embedding_dim": 0}]
)
def test_malformed_declarations_raise(tmp_path, cfg):
    with pytest.raises(ValueError):
        encoder._validate_graph_contract(_dir(tmp_path, **cfg), _session(TWO, 128))
