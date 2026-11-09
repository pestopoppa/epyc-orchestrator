"""WORKER_CPUS must stay inside the launching process's affinity (grant safety)."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_worker_cpus_subset_of_launch_affinity():
    code = (
        "import os, sys; sys.path.insert(0, %r);"
        "import scripts.kb_rag.parallel_reembed as m;"
        "a = set(os.sched_getaffinity(0));"
        "print(sorted(m.WORKER_CPUS) == sorted(a - m.GPU_HOST_LANE))" % str(REPO)
    )
    cpus = sorted(os.sched_getaffinity(0))[:2]
    pre = lambda: os.sched_setaffinity(0, set(cpus))  # noqa: E731
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        preexec_fn=pre, timeout=60, cwd=str(REPO),
    )
    assert out.stdout.strip().endswith("True"), out.stdout + out.stderr
