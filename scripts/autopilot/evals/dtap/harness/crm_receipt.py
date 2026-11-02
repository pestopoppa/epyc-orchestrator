"""Exclusive, fsynced partial-evidence journal for an admitted CRM pair."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Receipt:
    def __init__(self, path: Path, metadata: dict):
        # Do not follow symlinks or overwrite an earlier attempt.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.file = os.fdopen(fd, "wb")
        self.previous = "0" * 64
        try:
            self.append("start", metadata)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            self.file.close()
            raise

    def append(self, kind: str, payload: Any) -> None:
        record = {"kind": kind, "utc": datetime.now(timezone.utc).isoformat(),
                  "previous_sha256": self.previous, "payload": payload}
        data = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False,
                          separators=(",", ":")).encode("utf-8") + b"\n"
        self.file.write(data)
        self.file.flush()
        os.fsync(self.file.fileno())
        self.previous = hashlib.sha256(data).hexdigest()

    def close(self) -> None:
        self.file.close()
