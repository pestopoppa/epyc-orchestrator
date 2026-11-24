"""`serving_shape.alternates` -- the registry slot for a champion's named serving configs.

PROMOTION-REGISTRY-SERVING-1: the slot must validate, survive `compile_lean` (master -> lean) intact,
and a malformed one must fail the compile rather than ride into a signed promotion.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.registry.registry_compiler import compile_lean
from src.registry.serving_alternates import (
    ServingAlternatesError,
    parse_alternates,
    validate_registry_alternates,
)

PROV = {"row_id": "ROW-1", "result_json": "x/RESULT.json", "result_sha256": "b" * 64}
GOOD = {
    "latency": {"argv_delta": {"--parallel": 1, "cache_ram": 8192}, "provenance": dict(PROV)},
    "low-temperature": {"argv_delta": {"temp": 0.6, "stop": ["a", "b"], "--no-warmup": True},
                        "provenance": dict(PROV)},
}


def test_undeclared_is_empty():
    assert parse_alternates(None) == {}


def test_valid_alternates_round_trip():
    assert parse_alternates(copy.deepcopy(GOOD)) == GOOD


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda a: a.update(bad="nope"), "must be a mapping"),
        (lambda a: a["latency"].update(extra=1), "unknown key"),
        (lambda a: a["latency"].update(argv_delta={}), "non-empty"),
        (lambda a: a["latency"].update(argv_delta={"k": {"nested": 1}}), "scalar"),
        (lambda a: a["latency"].update(argv_delta={"k": [{"x": 1}]}), "scalars only"),
        (lambda a: a["latency"].update(argv_delta={"-b": True}), "takes a value"),
        (lambda a: a["latency"].update(argv_delta={"-ub": None}), "takes a value"),
        (lambda a: a["latency"].update(argv_delta={"-ub": "1024|2048"}), "contains '|'"),
        (lambda a: a["latency"].update(argv_delta={"-ctk": "q8_0 q4_0"}), "whitespace"),
        (lambda a: a["latency"].update(argv_delta={"-ub": "big"}), "non-negative integer"),
        (lambda a: a["latency"].update(argv_delta={"-ub": 1.5}), "non-negative integer"),
        (lambda a: a["latency"].update(argv_delta={"-ub": [1, 2]}), "one value, not a list"),
        (lambda a: a["latency"].update(argv_delta={"--no-warmup": 1}), "bare switch"),
        (lambda a: a["latency"].update(argv_delta={"--new-flag": True}), "takes a value"),
        (lambda a: a["latency"].update(argv_delta={"cache_ram": "lots"}), "non-negative integer"),
        (lambda a: a["latency"].update(argv_delta={"temp": 1.5}), "no high-temperature testing"),
        (lambda a: a["latency"].update(argv_delta={"--temp": 1.5}), "no high-temperature testing"),
        (lambda a: a["latency"].update(argv_delta={"temp": -0.1}), "non-negative"),
        (lambda a: a["latency"].update(argv_delta={"k": None}), "must not be null"),
        (lambda a: a["latency"].update(argv_delta={"a|b": 1}), "'|'"),
        (lambda a: a["latency"].pop("provenance"), "provenance must be a mapping"),
        (lambda a: a["latency"]["provenance"].pop("row_id"), "missing"),
        (lambda a: a["latency"]["provenance"].update(result_sha256="short"), "64 lowercase hex"),
    ],
)
def test_malformed_alternates_are_rejected(mutate, needle):
    alts = copy.deepcopy(GOOD)
    mutate(alts)
    with pytest.raises(ServingAlternatesError, match=needle):
        parse_alternates(alts)


def test_typed_values_are_accepted():
    ok = {"a": {"argv_delta": {"-ub": 256, "-b": "1024", "--temp": 0.0, "-ctk": "q4_0", "--kv-unified": True,
                               "--ctx-checkpoints": 16, "--cache-ram": 32768, "-np": 6},
                "provenance": dict(PROV)}}
    assert parse_alternates(copy.deepcopy(ok)) == ok


def test_flag_tables_match_the_root_validator():
    """The orchestrator cannot import the root validator at runtime, so the two flag tables are kept in step here."""
    import importlib.util
    import os

    root = Path(os.environ.get("EPYC_ROOT", "/workspace"))
    path = root / "scripts/champion/serving_config/validate_serving_config.py"
    if not path.is_file():
        pytest.skip("epyc-root validator not on this host")
    spec = importlib.util.spec_from_file_location("_v", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "BARE_FLAGS"):
        pytest.skip("epyc-root validator predates the typed flag tables")
    from src.registry import serving_alternates as sa

    for name in ("BARE_FLAGS", "INT_FLAGS", "FLOAT_FLAGS", "TEMP_FLAGS", "MAX_TEMPERATURE"):
        assert getattr(sa, name) == getattr(mod, name), name


def test_alternates_must_be_a_mapping():
    with pytest.raises(ServingAlternatesError):
        parse_alternates(["latency"])


def _master(alternates):
    shape = {"n_ctx": 4096, "slots_by_shape": {"full": 2}}
    if alternates is not None:
        shape["alternates"] = alternates
    return {
        "server_mode": {"frontdoor": {"slots": 2, "serving_shape": shape}},
        "roles": {"frontdoor": {"tier": "hot"}},
    }


def test_validate_registry_reports_declaring_paths():
    assert validate_registry_alternates(_master(copy.deepcopy(GOOD))) == [
        "server_mode.frontdoor.serving_shape.alternates"
    ]
    assert validate_registry_alternates(_master(None)) == []


def _compile(tmp_path: Path, master: dict) -> dict:
    path = tmp_path / "master.yaml"
    path.write_text(yaml.safe_dump(master))
    topology = tmp_path / "topology.yaml"
    topology.write_text("{}\n")
    return compile_lean(path, {"frontdoor"}, topology)


def test_alternates_survive_compile_into_lean(tmp_path: Path):
    lean = _compile(tmp_path, _master(copy.deepcopy(GOOD)))
    assert lean["server_mode"]["frontdoor"]["serving_shape"]["alternates"] == GOOD
    # the optimum's own knobs are untouched by the slot
    assert lean["server_mode"]["frontdoor"]["serving_shape"]["n_ctx"] == 4096


def test_malformed_alternates_fail_the_compile(tmp_path: Path):
    bad = copy.deepcopy(GOOD)
    bad["latency"]["provenance"].pop("result_json")
    with pytest.raises(ServingAlternatesError, match="result_json"):
        _compile(tmp_path, _master(bad))
