"""HIL-16c: order/type-faithful tool payloads for replay, and the shared production registry construction."""
from __future__ import annotations

import datetime as dt
import logging
from types import SimpleNamespace

from src.registry import production
from src.trace.replay.payload import Opaque, decode, encode


# ------------------------------------------------------------------------------------------- payload
def test_dict_key_order_survives_unlike_the_canonical_text():
    import json

    rec = {"success": True, "timezone": "UTC", "datetime": "2026-10-22T12:00:00", "date": "2026-10-22"}
    canonical = json.loads(json.dumps(rec, sort_keys=True))
    assert repr(canonical) != repr(rec)                  # the hazard: what HIL-16b rebuilt from
    assert repr(decode(encode(rec))) == repr(rec)


def test_container_types_and_non_str_keys_round_trip():
    v = {"t": (1, 2, ("x",)), "l": [1.5, None, True], "fs": frozenset({3}), "n": {1: "a", (2, 3): "b"}, "s": {"k"}}
    out = decode(encode(v))
    assert out == v and repr(out) == repr(v)
    assert type(out["t"]) is tuple and type(out["fs"]) is frozenset and type(out["s"]) is set


def test_non_json_objects_keep_repr_and_str_only():
    when = dt.datetime(2026, 10, 22, 12, 0, 0)
    out = decode(encode({"when": when}))["when"]
    assert isinstance(out, Opaque) and repr(out) == repr(when) and str(out) == str(when)
    assert not hasattr(out, "year")                      # any other use fails loudly in replay


def test_reserved_tag_key_is_escaped():
    v = {"__replay_t": "tuple", "v": [1]}
    assert decode(encode(v)) == v and type(decode(encode(v))) is dict


def test_encode_never_raises():
    class Bad:
        def __repr__(self):
            raise RuntimeError("no repr")

    assert isinstance(decode(encode(Bad())), Opaque)


# ------------------------------------------------------------------------------------------- registries
def _f(tools=True, scripts=True):
    return SimpleNamespace(tools=tools, scripts=scripts)


def test_tool_registry_gates_match_the_lifespan():
    assert production.build_tool_registry(_f(tools=False)) is None
    assert production.build_tool_registry(_f(), registry_cls=None) is None   # optional import unavailable


def test_tool_registry_failure_logs_and_returns_none(caplog):
    class Boom:
        def __init__(self):
            raise RuntimeError("cannot build")

    with caplog.at_level(logging.INFO, logger="src.registry.production"):
        assert production.build_tool_registry(_f(), registry_cls=Boom) is None
    assert "Tool registry not available: cannot build" in caplog.text


def test_script_registry_requires_tools_and_its_flag():
    class SR:
        def load_from_directory(self, path):
            self.path = path

    assert production.build_script_registry(None, _f(), registry_cls=SR) is None
    assert production.build_script_registry(object(), _f(scripts=False), registry_cls=SR) is None
    assert production.build_script_registry(object(), _f(), registry_cls=None) is None
    sr = production.build_script_registry(object(), _f(), registry_cls=SR)
    assert isinstance(sr, SR) and sr.path


def test_real_tool_registry_loads_yaml_permissions_and_builtins():
    reg = production.build_tool_registry(_f())
    assert reg is not None and len(reg._tools) > 20
    assert "frontdoor" in reg._permissions and reg.list_tools(role="frontdoor")
    assert production.build_script_registry(reg, _f()) is not None


def test_lifespan_uses_the_shared_builders():
    import inspect

    import src.api as api

    src = inspect.getsource(api.lifespan)
    assert "build_tool_registry(" in src and "build_script_registry(" in src
    assert "load_tools_from_yaml" not in src           # no second copy of the construction to drift
