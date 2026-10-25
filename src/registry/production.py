"""The production tool / script registry construction, in ONE place (HIL-16c).

The API lifespan (``src/api/__init__.py``) builds the process-wide ``ToolRegistry`` and ``ScriptRegistry``
that every chat-pipeline REPL is constructed with. The replay harness
(``src/trace/replay/graph_harness.py:production_repl_factory``) must rebuild exactly the same registries, or
the tool catalogue a replayed REPL exposes (``list_tools()`` output, read-only / chainable tool sets, the
turn-1 tool grammar, permission answers) is not the one the recorded run had. Both call these two
functions, so they cannot drift apart. Behaviour is the lifespan's, unchanged: the same feature gates, the
same load order (YAML tools, then role permissions from the model registry, then programmatic builtins
whose duplicates are skipped), the same "log and fall back to None" failure handling, the same log lines.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT: Any = object()


def build_tool_registry(
    f: Any = None, *, registry_cls: Any = _DEFAULT, log: logging.Logger | None = None
) -> Any | None:
    """Production ``ToolRegistry`` (or ``None`` when ``f.tools`` is off, the class is unavailable, or loading fails).

    ``registry_cls`` defaults to ``src.registry.tool_registry.ToolRegistry``; the lifespan passes the class
    its optional-import step resolved (possibly ``None``), so its gating is preserved exactly.
    """
    log = log or logger
    if f is None:
        from src.features import features

        f = features()
    if registry_cls is _DEFAULT:
        from src.registry.tool_registry import ToolRegistry as registry_cls
    if not (f.tools and registry_cls):
        return None
    try:
        from src.tool_registry import load_from_yaml as load_tools_from_yaml

        registry = registry_cls()

        # Load tools from YAML registry (orchestration/tool_registry.yaml)
        from src.config import get_config as _get_config

        _paths = _get_config().paths
        loaded = load_tools_from_yaml(registry, str(_paths.tool_registry_path))
        log.info(f"Loaded {loaded} tools from YAML registry")

        # Load role permissions from model registry
        registry.load_permissions_from_registry(str(_paths.registry_path))

        # Keep YAML authoritative for overlapping lint_python/read_file handlers;
        # register programmatic compatibility tools when their names are absent.
        try:
            from src.builtin_tools import register_builtin_tools

            register_builtin_tools(registry)
        except ImportError:
            log.debug("No builtin_tools module found")
        except ValueError as e:
            # Tool already registered from YAML - this is fine
            log.debug(f"Some builtin tools skipped (already in YAML): {e}")
        return registry
    except Exception as e:
        log.info(f"Tool registry not available: {e}")
        return None


def build_script_registry(
    tool_registry: Any,
    f: Any = None,
    *,
    registry_cls: Any = _DEFAULT,
    log: logging.Logger | None = None,
) -> Any | None:
    """Production ``ScriptRegistry`` (requires a tool registry; ``None`` when gated off or loading fails)."""
    log = log or logger
    if f is None:
        from src.features import features

        f = features()
    if registry_cls is _DEFAULT:
        from src.registry.script_registry import ScriptRegistry as registry_cls
    if not (f.scripts and registry_cls and tool_registry):
        return None
    try:
        from src.config import get_config as _get_config

        registry = registry_cls()
        registry.load_from_directory(str(_get_config().paths.script_registry_dir))
        return registry
    except Exception as e:
        log.info(f"Script registry not available: {e}")
        return None
