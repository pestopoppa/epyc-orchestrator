#!/usr/bin/env python3
"""HEAD-probe every CDN URL referenced from src/api/routes/dashboard.html.

Reports any URL that returns a non-2xx status, so we catch CDN drift before
users hit a silent-degradation banner. Runnable standalone (no external
dependencies beyond stdlib) or wired into CI.

Usage:
    python3 scripts/maintenance/check_cdn_urls.py             # exit 0 on all OK, 1 on any failure
    python3 scripts/maintenance/check_cdn_urls.py --json      # machine-readable
    python3 scripts/maintenance/check_cdn_urls.py --html PATH # check a different HTML file

Why this exists: the highlight.js CDN URL silently 404'd for an unknown
duration before the 2026-05-22 diagnostic-banner work surfaced it. The
warning banner was a runtime check; this is the build-time preflight that
catches it before it ever hits the browser.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HTML = REPO_ROOT / "src" / "api" / "routes" / "dashboard.html"

# Match `src="https://..."` and `href="https://..."` in <script> / <link> tags.
_URL_RE = re.compile(r'''(?:src|href)\s*=\s*["'](https?://[^"']+)["']''')


class Probe(NamedTuple):
    url: str
    status: int | None
    error: str | None
    bytes_len: int | None


def _extract_urls(html_path: Path) -> list[str]:
    """Pull every external src/href URL from <script> + <link> tags."""
    text = html_path.read_text()
    seen: list[str] = []
    for m in _URL_RE.finditer(text):
        url = m.group(1)
        if url not in seen:
            seen.append(url)
    return seen


def _probe(url: str, timeout: float = 5.0) -> Probe:
    """HTTP HEAD probe. Returns Probe with status code or error."""
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", "epyc-orchestrator-cdn-check/1.0")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            length = resp.headers.get("Content-Length")
            return Probe(
                url=url,
                status=status,
                error=None,
                bytes_len=int(length) if length and length.isdigit() else None,
            )
    except urllib.error.HTTPError as e:
        return Probe(url=url, status=e.code, error=f"HTTPError({e.code})", bytes_len=None)
    except Exception as e:
        return Probe(url=url, status=None, error=f"{type(e).__name__}: {e}", bytes_len=None)


def _format_probe(p: Probe) -> str:
    """Human-readable line for one probe."""
    if p.status is not None and 200 <= p.status < 300:
        size = f"{p.bytes_len:>10,} B" if p.bytes_len is not None else "         ? B"
        return f"  \033[32mOK\033[0m   {p.status} {size}  {p.url}"
    code = str(p.status) if p.status else "---"
    err = p.error or "unknown"
    return f"  \033[31mFAIL\033[0m {code:>3}             {p.url}   <- {err}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--html", type=Path, default=DEFAULT_HTML,
                    help=f"path to HTML file (default: {DEFAULT_HTML.relative_to(REPO_ROOT)})")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of text")
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="per-URL probe timeout in seconds (default: 5)")
    args = ap.parse_args()

    if not args.html.exists():
        print(f"ERROR: {args.html} not found", file=sys.stderr)
        return 2

    urls = _extract_urls(args.html)
    if not urls:
        print(f"WARNING: no external src=/href= URLs found in {args.html}", file=sys.stderr)
        return 0

    probes = [_probe(u, timeout=args.timeout) for u in urls]
    failures = [p for p in probes if not (p.status and 200 <= p.status < 300)]

    if args.json:
        out = {
            "html": str(args.html),
            "total": len(probes),
            "ok": len(probes) - len(failures),
            "failed": len(failures),
            "probes": [p._asdict() for p in probes],
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"CDN URL audit — {args.html.relative_to(REPO_ROOT) if args.html.is_relative_to(REPO_ROOT) else args.html}")
        print(f"  {len(probes)} URL(s), {len(probes) - len(failures)} OK, {len(failures)} failed")
        print()
        for p in probes:
            print(_format_probe(p))
        if failures:
            print()
            print(f"\033[31m{len(failures)} failure(s) — fix before deploying:\033[0m")
            for p in failures:
                print(f"  - {p.url}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
