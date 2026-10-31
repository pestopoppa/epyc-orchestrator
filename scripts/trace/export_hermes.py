#!/usr/bin/env python3
"""Export a private Hermes read projection; stdout contains counts only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.trace.hermes_reader import read_session


def _check_parent(path: Path, directory: int) -> None:
    actual = os.stat(path, follow_symlinks=False)
    expected = os.fstat(directory)
    if not stat.S_ISDIR(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ):
        raise PermissionError("private output directory namespace changed")


def write_private_projection(projection: dict, output: Path) -> None:
    """Create a fresh0600 file in an existing owned0700 directory outside Git.

    Relative dirfd I/O and namespace checks refuse symlink/directory swaps.
    This coordinates with a cooperative owner, not a hostile same-UID process.
    On failure any owned partial is retained; no cleanup unlink or rename occurs.
    """
    output = output.absolute()
    if ".." in output.parts or output.parent.resolve(strict=True) != output.parent:
        raise PermissionError("output parents must be real unambiguous directories")
    if any((parent / ".git").exists() for parent in output.parents):
        raise PermissionError("private conversation export into a Git tree is forbidden")
    directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(directory)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise PermissionError("output directory must be owned and0700")
        _check_parent(output.parent, directory)
        fd = os.open(
            output.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        try:
            _check_parent(output.parent, directory)
            data = (json.dumps(projection, ensure_ascii=False) + "\n").encode("utf-8")
            while data:
                written = os.write(fd, data)
                if written <= 0:
                    raise OSError("incomplete private export")
                data = data[written:]
            actual = os.stat(output.name, dir_fd=directory, follow_symlinks=False)
            expected = os.fstat(fd)
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                raise PermissionError("private output file namespace changed")
            _check_parent(output.parent, directory)
        finally:
            os.close(fd)
    finally:
        os.close(directory)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("session_id")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    projection = read_session(args.database, args.session_id)
    write_private_projection(projection, args.out)
    print(
        json.dumps(
            {
                "message_rows": len(projection["messages"]),
                "privacy_class": "local_private",
                "fidelity_tier": "agent_readable_projection",
                "measurement_grade": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
