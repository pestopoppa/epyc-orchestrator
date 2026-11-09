"""own_port_proof.py: a real listener in this process is 'listening' and owned by os.getpid(), not by another pid."""
import importlib.util
import os
import socket
import sys
from pathlib import Path

_S = importlib.util.spec_from_file_location("opp", Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "own_port_proof.py")
M = importlib.util.module_from_spec(_S); sys.modules["opp"] = M; _S.loader.exec_module(M)


def test_listener_owner_and_foreign_pid():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(1)
    port = s.getsockname()[1]
    try:
        assert M.main(["x", "listening", str(port)]) == 0
        assert M.main(["x", "owner", str(port), str(os.getpid())]) == 0
        assert M.main(["x", "owner", str(port), str(os.getppid())]) == 1
    finally:
        s.close()
    assert M.main(["x", "listening", str(port)]) == 1
