"""Shared structured-output repair helper for the typed-decision plane (TD-21).

See ``src.structured_output.repair`` for the full contract. Re-exported here
so consumers write ``from src.structured_output import parse_with_repair``.
"""

from src.structured_output.repair import (
    STRUCTURED_OUTPUT_REPAIR_COUNTS,
    CompleteFn,
    RepairResult,
    fish_json,
    http_chat_completer,
    parse_with_repair,
    primitives_completer,
)

__all__ = [
    "STRUCTURED_OUTPUT_REPAIR_COUNTS",
    "CompleteFn",
    "RepairResult",
    "fish_json",
    "http_chat_completer",
    "parse_with_repair",
    "primitives_completer",
]
