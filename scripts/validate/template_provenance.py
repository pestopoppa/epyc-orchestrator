#!/usr/bin/env python3
"""SEC-ART-5: record upstream-vs-served chat-template provenance at onboarding.

For one role, compute three things and emit them as a registry-ready plan row:

  served_template_sha256    sha256 of the EFFECTIVE template (`--chat-template-file` bytes, else the GGUF's
                            embedded `tokenizer.chat_template`; same precedence as the launcher)
  upstream_template_sha256  sha256 of the template the FIRST-PARTY model repo ships at a pinned revision
                            (`chat_template.jinja`, else `tokenizer_config.json["chat_template"]`), or of a
                            local `--upstream-file`
  template_diff             a unified diff upstream -> served written to `--diff-dir/<role>.diff`
                            (always written, including "identical"; a reviewer reads it, then the registry
                            records `template_diff_reviewed`)

The row has `status: pin` and the keys `apply_registry_integrity_pins.py --template-plan` consumes, so the
master registry gets the fields without hand-editing. Network is OFF unless `--allow-network` (metadata and
one small template file; never weights). An unreachable upstream is reported as `upstream_unavailable` and the
row is NOT `pin`: a served pin without an upstream comparison does not satisfy SEC-ART-5.

    template_provenance.py --role frontdoor --gguf /mnt/raid0/llm/models/X.gguf \\
        [--chat-template-file F] --upstream-repo Qwen/Qwen3.6-35B-A3B [--upstream-revision main] \\
        --diff-dir DIR [--diff-rel-prefix data/template_reviews] --allow-network --out row.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.registry import artifact_integrity as ai  # noqa: E402

HF = "https://huggingface.co"


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "epyc-template-provenance/1"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed https host
        return r.read()


def fetch_upstream_template(repo: str, revision: str, get: Callable[[str], bytes] = http_get) -> tuple[bytes, str, str]:
    """(template bytes, resolved commit sha, file it came from). Raises LookupError if the repo ships none."""
    q = urllib.parse.quote
    sha = json.loads(get(f"{HF}/api/models/{repo}/revision/{q(revision, safe='')}"))["sha"]
    base = f"{HF}/{repo}/resolve/{sha}"
    try:
        return get(f"{base}/chat_template.jinja"), sha, "chat_template.jinja"
    except Exception:  # noqa: BLE001 - 404 means "try the next convention"
        pass
    try:
        cfg = json.loads(get(f"{base}/tokenizer_config.json"))
    except Exception as exc:  # noqa: BLE001
        raise LookupError(f"{repo}@{sha}: no chat_template.jinja and no tokenizer_config.json ({exc})") from exc
    tmpl = cfg.get("chat_template")
    if isinstance(tmpl, list):  # [{"name": "default", "template": ...}, ...]
        tmpl = next((t.get("template") for t in tmpl if isinstance(t, dict) and t.get("name") == "default"), None)
    if not isinstance(tmpl, str):
        raise LookupError(f"{repo}@{sha}: tokenizer_config.json has no string chat_template")
    return tmpl.encode(), sha, "tokenizer_config.json"


def build_row(
    role: str,
    served: bytes | None,
    served_source: str,
    upstream: bytes | None,
    upstream_info: dict[str, Any],
    diff_dir: Path,
    diff_rel_prefix: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {"role": role, "served_source": served_source, **upstream_info}
    if served is None:
        return {**row, "status": "no_served_template"}
    row["served_template_sha256"] = ai.template_sha256(served)
    row["served_template_bytes"] = len(served)
    if upstream is None:
        return {**row, "status": "upstream_unavailable"}
    row["upstream_template_sha256"] = ai.template_sha256(upstream)
    # Decode strictly: replacement characters can collapse distinct byte streams.
    try:
        upstream_text = upstream.decode("utf-8")
        served_text = served.decode("utf-8")
        rendering_note = ""
    except UnicodeDecodeError:
        # Use the same reversible rendering for both sides, with byte offsets.
        def hex_lines(data: bytes) -> str:
            return "".join(
                f"{offset:08x}: {data[offset:offset + 32].hex()}\n"
                for offset in range(0, len(data), 32)
            )

        upstream_text, served_text = hex_lines(upstream), hex_lines(served)
        rendering_note = "# Non-UTF-8 input: both sides rendered as offset: hexadecimal bytes\n"
    diff_dir.mkdir(parents=True, exist_ok=True)
    diff_lines = list(difflib.unified_diff(
        upstream_text.splitlines(keepends=True),
        served_text.splitlines(keepends=True),
        fromfile=f"upstream:{upstream_info.get('upstream_ref', '?')}@{upstream_info.get('upstream_revision', '?')[:12]}"
                 f" sha256={row['upstream_template_sha256'][:12]}",
        tofile=f"served:{served_source} sha256={row['served_template_sha256'][:12]}",
    ))
    if upstream == served:
        body = "# identical: served template bytes equal the upstream template\n"
    else:
        # Unified diff does not itself delimit unterminated source lines.
        body = rendering_note + "".join(
            line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            for line in diff_lines
        )
    (diff_dir / f"{role}.diff").write_text(body)
    row["template_diff"] = f"{diff_rel_prefix.rstrip('/')}/{role}.diff"
    row["identical"] = row["upstream_template_sha256"] == row["served_template_sha256"]
    row["status"] = "pin"
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--role", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--chat-template-file")
    ap.add_argument("--upstream-repo")
    ap.add_argument("--upstream-revision", default="main")
    ap.add_argument("--upstream-file", help="local copy of the upstream template (offline)")
    ap.add_argument("--diff-dir", required=True)
    ap.add_argument("--diff-rel-prefix", default="data/template_reviews")
    ap.add_argument("--allow-network", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    if a.chat_template_file:
        served, source = Path(a.chat_template_file).read_bytes(), f"file:{a.chat_template_file}"
    else:
        served, source = ai.read_gguf_chat_template(a.gguf), f"gguf:{a.gguf}"

    upstream: bytes | None = None
    info: dict[str, Any] = {}
    if a.upstream_file:
        upstream = Path(a.upstream_file).read_bytes()
        info = {"upstream_ref": a.upstream_repo or a.upstream_file, "upstream_revision": "local-file"}
    elif a.upstream_repo and a.allow_network:
        try:
            upstream, sha, src = fetch_upstream_template(a.upstream_repo, a.upstream_revision)
            info = {"upstream_ref": a.upstream_repo, "upstream_revision": sha, "upstream_file": src}
        except Exception as exc:  # noqa: BLE001 - reported on the row
            info = {"upstream_ref": a.upstream_repo, "upstream_error": f"{type(exc).__name__}: {exc}"}
    row = build_row(a.role, served, source, upstream, info, Path(a.diff_dir), a.diff_rel_prefix)
    doc = json.dumps({"rows": [row]}, indent=2)
    if a.out:
        Path(a.out).write_text(doc + "\n")
    else:
        print(doc)
    return 0 if row["status"] == "pin" else 1


if __name__ == "__main__":
    sys.exit(main())
