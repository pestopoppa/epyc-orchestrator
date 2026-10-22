"""Type- and order-faithful encoding of a tool's return value for replay (HIL-16c).

``tool_step.output_cas`` holds the CANONICAL text of a result (``json.dumps(sort_keys=True, default=str)``):
right for hashing and the L2 scan, wrong for handing the REPL back "the same object", because it sorts dict
keys and turns tuples into lists and other objects into strings. The REPL renders variables with ``repr()``
into the next prompt, so a dict rebuilt from the canonical text prints its keys in a different order than
the recorded run did (found by the first replay of a real registry tool, ``get_time``, whose result dict is
not alphabetical). ``encode`` keeps insertion order and container types; ``decode`` rebuilds them.

Objects JSON cannot carry (datetimes, dataclasses, ...) are recorded as ``Opaque``: replay gives back an
object whose ``repr()`` / ``str()`` are the recorded ones, which is what reaches a prompt; any other use of it
fails loudly in the REPL (and so shows up as a divergence), never silently as a look-alike value.
"""

from __future__ import annotations

import json
from typing import Any

_TAG = "__replay_t"


class Opaque:
    """Stand-in for a recorded object that is not JSON data: same ``repr`` and ``str``, nothing else."""

    __slots__ = ("type_name", "_repr", "_str")

    def __init__(self, type_name: str, r: str, s: str) -> None:
        self.type_name, self._repr, self._str = type_name, r, s

    def __repr__(self) -> str:
        return self._repr

    def __str__(self) -> str:
        return self._str


def _enc(x: Any, depth: int = 0) -> Any:
    if depth > 200:
        raise ValueError("payload nested too deeply")
    if x is None or type(x) in (bool, int, float, str):
        return x
    if type(x) is list:
        return [_enc(v, depth + 1) for v in x]
    if type(x) is dict and all(type(k) is str for k in x) and _TAG not in x:
        return {k: _enc(v, depth + 1) for k, v in x.items()}
    if type(x) is dict:
        return {_TAG: "dict", "v": [[_enc(k, depth + 1), _enc(v, depth + 1)] for k, v in x.items()]}
    if type(x) is tuple:
        return {_TAG: "tuple", "v": [_enc(v, depth + 1) for v in x]}
    if type(x) in (set, frozenset):
        return {_TAG: type(x).__name__, "v": [_enc(v, depth + 1) for v in x]}
    if isinstance(x, Opaque):
        return {_TAG: "opaque", "type": x.type_name, "repr": repr(x), "str": str(x)}
    return {_TAG: "opaque", "type": type(x).__name__, "repr": repr(x), "str": str(x)}


def _dec(x: Any) -> Any:
    if isinstance(x, list):
        return [_dec(v) for v in x]
    if not isinstance(x, dict):
        return x
    t = x.get(_TAG)
    if t is None:
        return {k: _dec(v) for k, v in x.items()}
    if t == "dict":
        return {_hashable(_dec(k)): _dec(v) for k, v in x["v"]}
    if t == "tuple":
        return tuple(_dec(v) for v in x["v"])
    if t == "set":
        return {_hashable(_dec(v)) for v in x["v"]}
    if t == "frozenset":
        return frozenset(_hashable(_dec(v)) for v in x["v"])
    if t == "opaque":
        return Opaque(x.get("type", "?"), x.get("repr", ""), x.get("str", ""))
    raise ValueError(f"unknown payload tag {t!r}")


def _hashable(v: Any) -> Any:
    return v if not isinstance(v, (list, dict, set)) else Opaque(type(v).__name__, repr(v), str(v))


def encode(obj: Any) -> str:
    """Never raises: an unencodable value becomes one ``Opaque`` at the top."""
    try:
        return json.dumps(_enc(obj), ensure_ascii=False, separators=(",", ":"))
    except Exception:
        try:
            r, s = repr(obj), str(obj)
        except Exception:
            r = s = f"<{type(obj).__name__}>"
        return json.dumps(
            {_TAG: "opaque", "type": type(obj).__name__, "repr": r, "str": s}, ensure_ascii=False
        )


def decode(text: str) -> Any:
    return _dec(json.loads(text))
