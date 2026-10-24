"""Thin Python bridge to the shared Bash llama-mtmd-cli probe."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess


_HELPER = Path(__file__).resolve().parents[2] / "scripts" / "lib" / "mtmd_probe.sh"


def run_mtmd_probe(path: Path) -> subprocess.CompletedProcess[str] | None:
    """Run the shared bounded, candidate-library-prefixed probe.

    The Bash helper uses a 20-second timeout plus a 2-second KILL grace and
    returns merged stdout/stderr with the actual `timeout`/binary status.
    Callers retain their own version-line parsing policy.
    """
    try:
        return subprocess.run(
            [str(_HELPER), str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=25,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def mtmd_subprocess_env(cli_path: str | Path, threads: int) -> dict[str, str]:
    """Launch env matching the probe: the binary's own dir leads LD_LIBRARY_PATH."""
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    cli_dir = str(Path(cli_path).resolve().parent)
    ld_parts = [part for part in env.get("LD_LIBRARY_PATH", "").split(":") if part]
    if cli_dir not in ld_parts:
        env["LD_LIBRARY_PATH"] = ":".join([cli_dir, *ld_parts])
    return env


__all__ = ["mtmd_subprocess_env", "run_mtmd_probe"]
