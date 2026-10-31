"""Native document row semantics without loading a model or mutating an index."""

import json
import re
import string
from types import SimpleNamespace

import numpy as np
import pytest

from src.retrieval import colbert_encoder as encoder
from src.retrieval import kb_rag


class BodyTokenizer:
    """Minimal special-token tokenizer with a distinct case-sensitive marker."""

    def __init__(self):
        self.limit = None
        self.padding = None

    def enable_truncation(self, *, max_length):
        self.limit = max_length

    def no_truncation(self):
        self.limit = None

    def enable_padding(self, *, length):
        self.padding = length

    def no_padding(self):
        self.padding = None

    def encode(self, text, *, add_special_tokens=True):
        if text in ("[Q] ", "[D] "):
            ids = [50368 if text == "[Q] " else 50369]
        else:
            ids = [
                15 if word == "." else 10 if word == "word" else 20
                for word in re.findall(r"\w+|\.", text)
            ]
        if add_special_tokens:
            ids = [50281] + ids[: self.limit - 2 if self.limit else None] + [50282]
        mask = [1] * len(ids)
        if self.padding:
            mask += [0] * (self.padding - len(ids))
            ids += [0] * (self.padding - len(ids))
        return SimpleNamespace(ids=ids, attention_mask=mask)


@pytest.fixture
def native(monkeypatch):
    feeds = []

    def run(_, feed):
        feeds.append({key: value.copy() for key, value in feed.items()})
        return [np.ones((1, feed["input_ids"].shape[1], 128), dtype=np.float32)]

    for key, value in {
        "_tokenizer": BodyTokenizer(),
        "_count_tokenizer": BodyTokenizer(),
        "_session": SimpleNamespace(run=run),
        "_input_names": ("input_ids", "attention_mask"),
        "_prefix_tokens_ok": True,
        "_query_prefix": "[Q] ",
        "_document_prefix": "[D] ",
        "_do_lower_case": True,
        "_legacy_do_lower_case": False,
        "_document_skip_ids": (15,),
    }.items():
        monkeypatch.setattr(encoder, key, value)
    return feeds


@pytest.mark.parametrize("cap", [8, 32, 64])
@pytest.mark.parametrize("text", ["  WORD.  ", "word " * 100])
def test_native_document_marker_cap_and_rows(native, cap, text):
    result = encoder.encode(text, cap, role=encoder.ROLE_DOCUMENT)
    ids = native[-1]["input_ids"][0]
    assert ids[0] == 50281 and ids[1] == 50369 and ids[-1] == 50282
    assert np.all(native[-1]["attention_mask"] == 1)
    assert len(ids) <= cap
    expected_body = [10, 15] if text.strip() == "WORD." else [10] * (cap - 3)
    assert ids.tolist() == [50281, 50369] + expected_body + [50282]
    assert result.shape == (len(ids) - (15 in ids), 128)
    assert encoder.count_tokens(text, role=encoder.ROLE_DOCUMENT) >= len(ids)


def test_query_retains_punctuation(native):
    result = encoder.encode("WORD.", 32, role=encoder.ROLE_QUERY)
    assert native[-1]["input_ids"][0].tolist() == [50281, 50368, 10, 15, 50282]
    assert result.shape == (5, 128)


def test_legacy_none_preserves_case_padding_and_all_attended_rows(native):
    result = encoder.encode("WORD.", 32, role=encoder.ROLE_NONE)
    assert native[-1]["input_ids"][0].tolist() == [50281, 20, 15, 50282] + [0] * 28
    assert result.shape == (4, 128)
    assert encoder.count_tokens("WORD.", role=encoder.ROLE_NONE) == 4


@pytest.mark.parametrize("invalid", [0.0, float("nan"), float("inf")])
def test_undefined_cosine_directions_are_refused(native, monkeypatch, invalid):
    monkeypatch.setattr(
        encoder,
        "_session",
        SimpleNamespace(
            run=lambda _, feed: [np.full((1, feed["input_ids"].shape[1], 128), invalid)]
        ),
    )
    assert encoder.encode("word", 32, role=encoder.ROLE_DOCUMENT) is None


def test_effective_pylate14_contract_uses_transformer_lowercase_and_empty_skiplist_default(
    tmp_path,
):
    (tmp_path / "sentence_bert_config.json").write_text(json.dumps({"do_lower_case": True}))
    (tmp_path / "config_sentence_transformers.json").write_text(json.dumps({"skiplist_words": []}))
    config = encoder._effective_native_contract(tmp_path)
    assert config["do_lower_case"] is True
    assert config["skiplist_words"] == list(string.punctuation)


def test_explicit_export_contract_overrides_transformer_fallback(tmp_path):
    (tmp_path / "sentence_bert_config.json").write_text(json.dumps({"do_lower_case": True}))
    (tmp_path / "onnx_config.json").write_text(
        json.dumps({"do_lower_case": False, "skiplist_words": ["."]})
    )
    config = encoder._effective_native_contract(tmp_path)
    assert config["do_lower_case"] is False and config["skiplist_words"] == ["."]


def test_new_contract_refuses_old_prefixed_store_without_mutating_it():
    assert encoder.PREFIX_CONVENTION == "qd-v2"
    assert kb_rag._roles_for_convention("qd-v2") == ("query", "document")
    assert kb_rag._roles_for_convention("none") == ("none", "none")
    with pytest.raises(ValueError, match="qd-v1"):
        kb_rag._roles_for_convention("qd-v1")


def test_clear_drops_native_contract_state(native):
    encoder._clear_loaded_state()
    assert encoder._document_skip_ids == () and encoder._legacy_do_lower_case is False
