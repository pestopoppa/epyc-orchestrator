"""Resolve production kernel binaries by BACKEND, never by build path.

A backend is a capability ("run this on the GPU"), not a location. Naming a build
directory in a registry or a launcher couples the orchestration apparatus to how a
kernel happened to be built, which is why moving a role to the GPU by registry edit
alone would previously have launched it on the CPU-only build — silently, because a
missing ggml backend does not raise, it simply is not used.

The stable layer lives at ``/mnt/raid0/llm/kernels/production/<backend>/`` and is a
symlink per backend. Freezing a new kernel repoints the symlink and archives the old
one; nothing here or in either registry changes. See that directory's README.md.

The three trees run three different ggml generations, so every launcher must set
``LD_LIBRARY_PATH`` to the backend directory this module returns — inheriting another
tree's ggml runs silently wrong rather than failing.
"""

from __future__ import annotations

from pathlib import Path

KERNEL_ROOT = Path("/mnt/raid0/llm/kernels")
PRODUCTION_ROOT = KERNEL_ROOT / "production"

# backend -> the server binary that backend provides
BACKEND_BINARIES: dict[str, str] = {
    "cpu": "llama-server",
    "gpu": "llama-server",
    "stt": "whisper-server",
    "tts": "tts-server",
}

VALID_BACKENDS = frozenset(BACKEND_BINARIES)


class KernelPathError(RuntimeError):
    """Raised when a backend cannot be resolved.

    Deliberately fatal. Returning a fallback path here would reintroduce exactly the
    defect this module exists to prevent: a GPU request quietly satisfied by a CPU
    build. An unresolvable backend is a third outcome, not a success.
    """


def backend_dir(backend: str) -> Path:
    """Return the directory for ``backend``, or raise.

    Also the value a launcher must put on ``LD_LIBRARY_PATH``.
    """
    if backend not in VALID_BACKENDS:
        raise KernelPathError(
            f"unknown kernel backend {backend!r}; valid: {sorted(VALID_BACKENDS)}"
        )
    path = PRODUCTION_ROOT / backend
    if not path.is_dir():
        raise KernelPathError(
            f"kernel backend {backend!r} does not resolve: {path} is missing or dangling. "
            f"Repoint it with: ln -sfn <build dir> {path}"
        )
    return path.resolve()


def server_binary(backend: str) -> Path:
    """Return the executable server binary for ``backend``, or raise."""
    directory = backend_dir(backend)
    binary = directory / BACKEND_BINARIES[backend]
    if not binary.is_file():
        raise KernelPathError(
            f"kernel backend {backend!r} resolves to {directory}, but "
            f"{BACKEND_BINARIES[backend]!r} is not present there"
        )
    return binary


def describe() -> dict[str, dict[str, str]]:
    """Report every backend and where it currently points.

    Provenance, not decoration: a binary launched from the wrong backend is otherwise
    indistinguishable from one launched from the right one.
    """
    out: dict[str, dict[str, str]] = {}
    for backend in sorted(VALID_BACKENDS):
        entry: dict[str, str] = {"binary_name": BACKEND_BINARIES[backend]}
        try:
            resolved = backend_dir(backend)
            entry["resolves_to"] = str(resolved)
            entry["binary_present"] = str((resolved / BACKEND_BINARIES[backend]).is_file())
        except KernelPathError as exc:
            entry["error"] = str(exc)
        out[backend] = entry
    return out
