"""Shared path + binary + health-timeout constants for the orchestrator stack.

Extracted from orchestrator_stack.py during the 2026-05-22 Tranche-7 refactor.
Lives below `stack_manifest.py` + `stack_commands.py` in the dependency graph
so those modules can import paths without creating a cycle with
`orchestrator_stack.py` (which itself imports back from the manifest).
"""

from __future__ import annotations

from pathlib import Path

from src.config import _registry_timeout


# Health check timeouts from registry (single source of truth)
_HEALTH_SERVER_STARTUP = int(_registry_timeout("health", "server_startup", 120))
_HEALTH_VISION_SERVER = int(_registry_timeout("health", "vision_server", 120))
_HEALTH_WORKER_SERVER = int(_registry_timeout("health", "worker_server", 90))


def _get_paths() -> dict[str, Path]:
    """Get paths from config with hardcoded fallbacks for robustness."""
    try:
        from src.config import get_config

        cfg = get_config()
        return {
            "llm_root": cfg.paths.llm_root,
            "project_root": cfg.paths.project_root,
            "models_dir": cfg.paths.models_dir,
            "model_base": cfg.paths.model_base,
            "llama_cpp_bin": cfg.paths.llama_cpp_bin,
            "log_dir": cfg.paths.log_dir,
            "cache_dir": cfg.paths.cache_dir,
            "tmp_dir": cfg.paths.tmp_dir,
        }
    except Exception:
        # Fallback to hardcoded defaults if config unavailable
        llm_root = Path("/mnt/raid0/llm")
        project_root = llm_root / "claude"
        return {
            "llm_root": llm_root,
            "project_root": project_root,
            "models_dir": llm_root / "models",
            "model_base": llm_root / "lmstudio/models",
            "llama_cpp_bin": llm_root / "llama.cpp/build/bin",
            "log_dir": project_root / "logs",
            "cache_dir": llm_root / "cache",
            "tmp_dir": llm_root / "tmp",
        }


_PATHS = _get_paths()


def _resolve_llama_cpp_binary(name: str, extra_candidates: tuple[Path, ...] = ()) -> Path:
    """Return the first known on-disk path for a llama.cpp helper binary."""
    primary = _PATHS["llama_cpp_bin"] / name
    for candidate in (primary, *extra_candidates):
        if candidate.exists():
            return candidate
    return primary


STATE_FILE = _PATHS["log_dir"] / "orchestrator_state.json"
LLAMA_SERVER = _resolve_llama_cpp_binary("llama-server")
LLAMA_MATH_TOOLS = _resolve_llama_cpp_binary(
    "llama-math-tools",
    (_PATHS["llm_root"] / "llama.cpp/tools/math-tools/build/llama-math-tools",),
)
# v2 binary retained for emergency fallback only. As of 2026-05-06 stack-swap,
# all hot-tier roles use the v5 binary (production-consolidated-v5). Previously
# coder_escalation needed v2 due to a Qwen2.5 spec-decode bug, but
# coder_escalation now runs Qwen3.6-35B-A3B Q8 (same model as frontdoor) which
# is v5-compatible.
LLAMA_SERVER_V2 = _PATHS["llama_cpp_bin"].parent / "build-v2" / "bin" / "llama-server"
_V2_ROLES: frozenset[str] = frozenset()  # was {"coder_escalation"}; empty since 2026-05-06
LOG_DIR = _PATHS["log_dir"]
# DS-3: KV state save/restore directory for dynamic stack management
SLOT_SAVE_DIR = _PATHS["cache_dir"] / "kv_slots"
