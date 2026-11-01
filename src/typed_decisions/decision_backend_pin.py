"""Capture a local llama-server decision backend for a calibration receipt.

The opt-in direct transport checks listener, executable, model and process identity
before and after each request. The native executable reports its commit, which may
be abbreviated. This JSON /completion path does not assert decision-head scaling.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import urllib.request
import urllib.error
import json
from urllib.parse import urlsplit

_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
_FLAGS = {
    "-np": "np",
    "--parallel": "np",
    "-b": "batch",
    "--batch-size": "batch",
    "-ub": "ubatch",
    "--ubatch-size": "ubatch",
}


def _exact(f, n):
    value = f.read(n)
    if len(value) != n:
        raise ValueError("truncated GGUF metadata")
    return value


def _number(f, fmt):
    return struct.unpack("<" + fmt, _exact(f, struct.calcsize(fmt)))[0]


def _string(f):
    n = _number(f, "Q")
    if n > 16 * 1024 * 1024:
        raise ValueError("GGUF string exceeds metadata bound")
    return _exact(f, n).decode("utf-8")


def _value(f, kind, depth=0):
    if kind in _SCALARS:
        return _number(f, _SCALARS[kind])
    if kind == 8:
        return _string(f)
    if kind == 9 and depth == 0:
        element, count = _number(f, "I"), _number(f, "Q")
        if count > 100_000_000:
            raise ValueError("GGUF array exceeds metadata bound")
        if element in _SCALARS:
            size = struct.calcsize(_SCALARS[element]) * count
            f.seek(size, os.SEEK_CUR)
        else:
            for _ in range(count):
                _value(f, element, depth + 1)
        return None
    raise ValueError("unknown or nested GGUF metadata type")


def read_decision_temperatures(path):
    with open(path, "rb") as f:
        if _exact(f, 4) != b"GGUF" or _number(f, "I") not in (2, 3):
            raise ValueError("unsupported GGUF header")
        _number(f, "Q")  # tensor count
        count = _number(f, "Q")
        if count > 1_000_000:
            raise ValueError("GGUF metadata count exceeds bound")
        metadata = {}
        for _ in range(count):
            key = _string(f)
            if key in metadata:
                raise ValueError("duplicate GGUF metadata key")
            kind = _number(f, "I")
            metadata[key] = _value(f, kind)
        if f.tell() > os.fstat(f.fileno()).st_size:
            raise ValueError("truncated GGUF metadata")
    arch = metadata.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise ValueError("GGUF missing architecture")
    prefix = arch + ".decision.temperature."
    result = {}
    for key, value in metadata.items():
        marker = "decision.temperature."
        if key.startswith(marker) or "." + marker in key:
            if (
                not key.split(marker, 1)[1]
                or isinstance(value, bool)
                or not isinstance(value, (int, float, str))
            ):
                raise ValueError("invalid decision temperature metadata")
            try:
                temp = float(value)
            except ValueError as exc:
                raise ValueError("invalid decision temperature metadata") from exc
            if not math.isfinite(temp) or temp <= 0:
                raise ValueError("invalid decision temperature metadata")
            result[key] = value
    return arch, prefix, result


def parse_process_flags(argv):
    """Read explicit flags, including aliases and '='; reject ambiguous duplicates."""
    result = {}
    i = 1
    while i < len(argv):
        token = argv[i]
        key, sep, value = token.partition("=")
        if key in _FLAGS:
            name = _FLAGS[key]
            if name in result:
                raise ValueError("duplicate server flag: " + name)
            if not sep:
                i += 1
                if i >= len(argv):
                    raise ValueError("missing server flag value: " + key)
                value = argv[i]
            if not re.fullmatch(r"[0-9]+", value) or int(value) < 1:
                raise ValueError("invalid server flag: " + key)
            result[name] = int(value)
        i += 1
    if set(result) != {"np", "batch", "ubatch"}:
        raise ValueError("explicit np, batch and ubatch flags are required")
    return result


def capture_decision_backend_pin(model_path, argv, server_build_commit):
    """Capture from a file and argv. Caller must bind them to the serving process."""
    if not isinstance(server_build_commit, str) or not re.fullmatch(
        r"[0-9a-f]{7,40}", server_build_commit
    ):
        raise ValueError(
            "reported server build commit must be 7–40 lowercase hex characters"
        )
    flags = parse_process_flags(argv)
    path = Path(model_path)
    before = path.stat()
    architecture, native_prefix, temperatures = read_decision_temperatures(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
        opened = os.fstat(f.fileno())
    after = path.stat()

    def identity(s):
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    if identity(before) != identity(opened) or identity(before) != identity(after):
        raise ValueError("GGUF changed during capture")
    return {
        "schema": "typed_decisions.llama_decision_backend.v1",
        "model_identity": list(identity(before)),
        "gguf_sha256": digest.hexdigest(),
        "model_architecture": architecture,
        "native_temperature_prefix": native_prefix,
        "decision_temperatures": temperatures,
        "server_build_commit": server_build_commit,
        "server_argv_flags": flags,
        "server_build_commit_custody": "input",
    }


def listener_inode(pid, base_url):
    """Bind a localhost HTTP URL to the exact process holding its LISTEN socket."""
    url = urlsplit(base_url)
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "localhost")
        or not url.port
        or url.path not in ("", "/")
        or url.query
        or url.fragment
        or url.username
        or url.password
    ):
        raise ValueError("backend must be a direct localhost HTTP origin")
    port = url.port
    if (
        os.stat("/proc/self/ns/net").st_ino != os.stat(f"/proc/{pid}/ns/net").st_ino
        or os.stat("/proc/self/ns/mnt").st_ino != os.stat(f"/proc/{pid}/ns/mnt").st_ino
    ):
        raise ValueError("server namespaces do not match client")
    owned = set()
    for entry in (Path("/proc") / str(pid) / "fd").iterdir():
        try:
            link = os.readlink(entry)
        except FileNotFoundError:
            continue
        match = re.fullmatch(r"socket:\[(\d+)\]", link)
        if match:
            owned.add(match.group(1))
    matches = []
    for line in (Path("/proc") / str(pid) / "net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        addr, hex_port = fields[1].split(":")
        if (
            fields[3] == "0A"
            and addr in ("0100007F", "00000000")
            and int(hex_port, 16) == port
        ):
            matches.append(fields[9])
    if len(matches) != 1 or matches[0] not in owned:
        raise ValueError("process does not uniquely own localhost listener")
    return matches[0]


class PinnedCompletionClient:
    """Direct JSON /completion transport with listener checks for each attempt."""

    def __init__(self, pid, base_url, pin):
        self.pid, self.pin = pid, pin
        self.base_url = f"http://127.0.0.1:{urlsplit(base_url).port}"
        self.calls = 0
        self.requests = []

    def _check(self):
        proc = Path("/proc") / str(self.pid)
        stat = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        args = [
            os.fsdecode(x) for x in (proc / "cmdline").read_bytes().split(b"\0")[:-1]
        ]
        argv_sha256 = hashlib.sha256(
            json.dumps(args, ensure_ascii=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            int(stat[19]) != self.pin["process"]["start_ticks"]
            or argv_sha256 != self.pin["process"]["argv_sha256"]
        ):
            raise ValueError("server process changed during calibration")
        if listener_inode(self.pid, self.base_url) != self.pin["socket_inode"]:
            raise ValueError("server listener changed during calibration")
        exe = (proc / "exe").stat()
        if (exe.st_dev, exe.st_ino) != tuple(self.pin["process"]["exe_identity"]):
            raise ValueError("server executable changed during calibration")
        model = Path(self.pin["model_path"]).stat()
        if (
            model.st_dev,
            model.st_ino,
            model.st_size,
            model.st_mtime_ns,
            model.st_ctime_ns,
        ) != tuple(self.pin["model_identity"]):
            raise ValueError("server model changed during calibration")

    def llm_call(self, prompt, *, role, n_tokens, json_schema, temperature, seed):
        self._check()
        payload = json.dumps(
            {
                "prompt": prompt,
                "n_predict": n_tokens,
                "json_schema": json_schema,
                "temperature": temperature,
                "seed": seed,
                "stream": False,
            }
        ).encode()
        request = urllib.request.Request(
            self.base_url + "/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect()
        )
        with opener.open(request, timeout=300) as response:
            data = json.load(response)
        self._check()
        self.calls += 1
        self.requests.append(
            {
                "n_predict": n_tokens,
                "temperature": temperature,
                "seed": seed,
                "stream": False,
                "stop": "server_default_not_requested",
                "other_sampling": "server_default_not_requested",
                "json_schema_sha256": hashlib.sha256(
                    json.dumps(
                        json_schema, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
        )
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("content"), str)
            or data.get("truncated")
        ):
            raise ValueError("invalid or truncated llama completion response")
        return data["content"]


def capture_running_backend_pin(pid, model_path, base_url):
    """Bind live process, model, listener and executable-reported build commit."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid process id")
    proc = Path("/proc") / str(pid)
    pidfd = os.pidfd_open(pid)
    try:

        def snapshot():
            # starttime distinguishes PID reuse, even if /proc changes during capture.
            stat = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
            args = (proc / "cmdline").read_bytes().split(b"\0")
            if not args or args[-1] != b"":
                raise ValueError("unreadable process argv")
            return (
                stat[19],
                [os.fsdecode(x) for x in args[:-1]],
                os.readlink(proc / "exe"),
                os.readlink(proc / "cwd"),
            )

        before = snapshot()
        argv = before[1]
        models = []
        forbidden = {
            "--override-kv",
            "--lora",
            "--lora-scaled",
            "--control-vector",
            "--control-vector-scaled",
            "--models-dir",
            "--model-url",
            "--hf-repo",
        }
        if any(arg.split("=", 1)[0] in forbidden for arg in argv[1:]):
            raise ValueError(
                "server uses unsupported model override, adapter or router"
            )
        for i, arg in enumerate(argv[1:], 1):
            key, sep, val = arg.partition("=")
            if key in ("-m", "--model"):
                if not sep:
                    if i + 1 >= len(argv):
                        raise ValueError("missing model argument")
                    val = argv[i + 1]
                models.append(val)
        if len(models) != 1:
            raise ValueError("exactly one explicit model argument required")
        actual = Path(models[0])
        if not actual.is_absolute():
            actual = Path(before[3]) / actual
        if not os.path.samefile(actual, model_path):
            raise ValueError("process model differs from captured GGUF")
        socket_inode = listener_inode(pid, base_url)
        exe_path = Path(before[2])
        exe_stat = exe_path.stat()
        proc_exe_stat = (proc / "exe").stat()
        if (exe_stat.st_dev, exe_stat.st_ino) != (
            proc_exe_stat.st_dev,
            proc_exe_stat.st_ino,
        ):
            raise ValueError("running executable no longer matches its path")
        # The executable is queried directly, using its own library path, never
        # a git checkout whose source may not correspond to the serving binary.
        environ = (proc / "environ").read_bytes().split(b"\0")
        env = os.environ.copy()
        for item in environ:
            if item.startswith(b"LD_LIBRARY_PATH="):
                env["LD_LIBRARY_PATH"] = os.fsdecode(item.split(b"=", 1)[1])
        try:
            version = subprocess.run(
                [str(exe_path), "--version"],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except subprocess.SubprocessError as exc:
            raise ValueError("server executable --version failed") from exc
        version_text = (version.stdout + "\n" + version.stderr).strip()
        match = re.search(
            r"(?im)^version:.*?\bcommit\s+([0-9a-f]{7,40})\)", version_text
        ) or re.search(r"(?im)^version:\s*\d+\s*\(([0-9a-f]{7,40})\)", version_text)
        if not match:
            raise ValueError("executable did not report a commit in --version")
        result = capture_decision_backend_pin(model_path, argv, match.group(1))
        if before != snapshot() or socket_inode != listener_inode(pid, base_url):
            raise ValueError("process changed during capture")
        model_stat = Path(model_path).stat()
        if [
            model_stat.st_dev,
            model_stat.st_ino,
            model_stat.st_size,
            model_stat.st_mtime_ns,
            model_stat.st_ctime_ns,
        ] != result["model_identity"]:
            raise ValueError("GGUF changed after hashing")
        if (exe_path.stat().st_dev, exe_path.stat().st_ino) != (
            exe_stat.st_dev,
            exe_stat.st_ino,
        ):
            raise ValueError("executable changed after version query")
        result["model_path"] = str(Path(model_path).resolve())
        result["model_identity"] = [
            model_stat.st_dev,
            model_stat.st_ino,
            model_stat.st_size,
            model_stat.st_mtime_ns,
            model_stat.st_ctime_ns,
        ]
        argv_sha256 = hashlib.sha256(
            json.dumps(argv, ensure_ascii=True, separators=(",", ":")).encode()
        ).hexdigest()
        result["process"] = {
            "pid": pid,
            "start_ticks": int(before[0]),
            "exe": before[2],
            "argv_sha256": argv_sha256,
            "exe_identity": [exe_stat.st_dev, exe_stat.st_ino],
        }
        result["argv_custody"] = "proc_observed"
        result["socket_inode"] = socket_inode
        result["endpoint"] = base_url.rstrip("/") + "/completion"
        result["endpoint_kind"] = "llama_completion_json"
        result["transport_method"] = "direct_local_completion_v1"
        result["decision_temperature_scaling"] = "not_asserted_for_json"
        result["server_version"] = version_text
        result["server_build_commit_custody"] = "executable_version"
        return result
    finally:
        os.close(pidfd)
