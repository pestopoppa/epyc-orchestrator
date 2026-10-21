"""Content-addressed blob store for replay recording. Address = ``sha256:<hex>`` of the exact bytes."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

ENV_DIR = "ORCHESTRATOR_REPLAY_CAS_DIR"
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "logs" / "replay_cas"


def digest(data: bytes | str) -> str:
    b = data.encode("utf-8") if isinstance(data, str) else data
    return "sha256:" + hashlib.sha256(b).hexdigest()


class CASIntegrityError(Exception):
    pass


class CAS:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root or os.environ.get(ENV_DIR) or DEFAULT_DIR)

    def _path(self, addr: str) -> Path:
        algo, _, hexd = addr.partition(":")
        if algo != "sha256" or len(hexd) != 64 or any(c not in "0123456789abcdef" for c in hexd):
            raise ValueError(f"bad CAS address: {addr!r}")
        return self.root / hexd[:2] / hexd

    def put(self, data: bytes | str) -> str:
        b = data.encode("utf-8") if isinstance(data, str) else data
        addr = digest(b)
        p = self._path(addr)
        if p.exists():
            return addr
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        tmp.write_bytes(b)
        os.replace(tmp, p)
        return addr

    def has(self, addr: str) -> bool:
        return self._path(addr).exists()

    def get(self, addr: str) -> bytes:
        """Return the blob; the bytes are re-hashed, so a corrupted or swapped file is an error, not data."""
        b = self._path(addr).read_bytes()
        if digest(b) != addr:
            raise CASIntegrityError(f"blob does not match its address: {addr}")
        return b

    def get_text(self, addr: str) -> str:
        return self.get(addr).decode("utf-8")
