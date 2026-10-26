"""Importing seeding helpers preserves the calling process's signal ownership."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_seeding_import_preserves_callers_signal_handlers(tmp_path):
    probe = tmp_path / "import_probe.py"
    probe.write_text("""import importlib.util, signal, sys
def owned(signum, frame):
    pass
signal.signal(signal.SIGINT, owned)
signal.signal(signal.SIGTERM, owned)
spec = importlib.util.spec_from_file_location('seed_import_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert signal.getsignal(signal.SIGINT) is owned, 'SIGINT replaced at import'
assert signal.getsignal(signal.SIGTERM) is owned, 'SIGTERM replaced at import'
print('OWNED-HANDLERS-PRESERVED')
""")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CI="true", ORCHESTRATOR_MOCK_MODE="1")
    source = Path("scripts/benchmark/seed_specialist_routing.py")
    candidate = subprocess.run([sys.executable, str(probe), str(ROOT / source)],
                               cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert candidate.returncode == 0, candidate.stdout + candidate.stderr
    assert "OWNED-HANDLERS-PRESERVED" in candidate.stdout
