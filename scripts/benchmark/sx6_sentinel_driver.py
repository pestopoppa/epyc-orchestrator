#!/usr/bin/env python3
"""SX-6 sentinel driver: issue web_research sentinel queries through ``web_search``.

Produces the telemetry ``scripts/analysis/searxng_health_report.py`` consumes, under a
driver-owned log root (default: never the repo ``logs/``):

* ``<root>/progress/<YYYY-MM-DD>.jsonl`` -- one ``tool_use`` record per search
  (``result.backend`` + ``result.elapsed_ms``), from ``web_search`` itself;
* ``<root>/sx6-driver.log`` -- the ``src.tools.web.search`` logger (SX-4 ``unresponsive_engines``
  and ``falling back to DDG`` lines), each line carrying its date;
* ``<root>/queries.jsonl`` -- what was asked, in order, with outcome.

``--ddg-baseline`` additionally times each query on DuckDuckGo directly (the p95 baseline the
2x latency gate needs). Queries come from a research-suite YAML (``questions[].prompt``); the
instruction wrapper ("Use web search to find:", the <answer> tag sentence) is stripped.

Precondition (NOT done here): a SearXNG service on :8888 started through the stack lifecycle.
The driver refuses to run when it is down unless ``--allow-searxng-down`` (then every query is a
fallback, which is itself a legitimate HOLD signal, so it is opt-in).

Decision rule (from the row): PROCEED iff bad-query <=5 %, fallback <=10 %, p95 <= 2x DDG p95,
n>=20; ``INSUFFICIENT_DATA`` is not a verdict. Run the report afterwards:

    python3 scripts/analysis/searxng_health_report.py --log-root <root> --from D --to D --json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_WRAP_PREFIX = re.compile(r"^\s*Use web search to find:\s*", re.I)
_WRAP_SUFFIX = re.compile(r"\s*Put your final answer inside <answer></answer> tags\.?\s*$", re.I)


def extract_query(prompt: str) -> str:
    return _WRAP_SUFFIX.sub("", _WRAP_PREFIX.sub("", prompt)).strip()


def load_queries(suite_yaml: Path, limit: int | None = None) -> list[dict[str, str]]:
    import yaml  # noqa: PLC0415

    data = yaml.safe_load(Path(suite_yaml).read_text(encoding="utf-8"))
    out = []
    for q in data.get("questions", []):
        text = extract_query(str(q.get("prompt", "")))
        if text:
            out.append({"id": str(q.get("id", "")), "query": text})
    return out[:limit] if limit else out


def searxng_up(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/healthz", timeout=timeout) as r:
            return 200 <= r.status < 500
    except Exception:  # noqa: BLE001
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return 200 <= r.status < 500
        except Exception:  # noqa: BLE001
            return False


def _tool_use(path_root: Path, now: datetime, query: str, result: dict[str, Any]) -> None:
    day = path_root / "progress" / f"{now.strftime('%Y-%m-%d')}.jsonl"
    day.parent.mkdir(parents=True, exist_ok=True)
    slim = {k: result.get(k) for k in ("success", "backend", "elapsed_ms", "result_count", "error")}
    with day.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "tool_use", "tool": "web_search", "ts": now.isoformat(),
                             "query": query, "result": slim}) + "\n")


def run(queries: list[dict[str, str]], log_root: Path, *,
        search: Callable[..., dict[str, Any]], ddg: Callable[..., Any] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> dict[str, Any]:
    log_root.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_root / "sx6-driver.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    lg = logging.getLogger("src.tools.web.search")
    lg.addHandler(handler)
    prev_level = lg.level
    lg.setLevel(logging.INFO)
    backends: dict[str, int] = {}
    try:
        with (log_root / "queries.jsonl").open("a", encoding="utf-8") as qf:
            for q in queries:
                res = search(q["query"])
                _tool_use(log_root, clock(), q["query"], res)
                backends[str(res.get("backend"))] = backends.get(str(res.get("backend")), 0) + 1
                if ddg is not None:
                    base = ddg(q["query"])
                    _tool_use(log_root, clock(), q["query"], base)
                qf.write(json.dumps({"id": q["id"], "query": q["query"], "backend": res.get("backend"),
                                     "success": res.get("success")}) + "\n")
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev_level)
        handler.close()
    return {"queries": len(queries), "backends": backends, "log_root": str(log_root)}


def _ddg_timed(query: str) -> dict[str, Any]:
    import time  # noqa: PLC0415

    from src.tools.web import search as s  # noqa: PLC0415

    t0 = time.monotonic()
    try:
        results = s._search_duckduckgo(query, max_results=5)
        return {"success": bool(results), "backend": "duckduckgo", "result_count": len(results),
                "elapsed_ms": (time.monotonic() - t0) * 1000.0}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "backend": "duckduckgo", "error": str(exc)[:200],
                "elapsed_ms": (time.monotonic() - t0) * 1000.0}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", type=Path, required=True, help="research suite YAML with questions[].prompt")
    p.add_argument("--log-root", type=Path, required=True, help="driver-owned log root (under /mnt/raid0/llm/tmp)")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--ddg-baseline", action="store_true")
    p.add_argument("--allow-searxng-down", action="store_true")
    args = p.parse_args(argv)
    from src.tools.web import search as s  # noqa: PLC0415

    if not s._SEARXNG_DEFAULT:
        print("sx6_sentinel_driver: REFUSING: ORCHESTRATOR_SEARXNG_DEFAULT is not 1", file=sys.stderr)
        return 2
    if not args.allow_searxng_down and not searxng_up(s._SEARXNG_URL):
        print(f"sx6_sentinel_driver: REFUSING: SearXNG not reachable at {s._SEARXNG_URL}", file=sys.stderr)
        return 3
    queries = load_queries(args.suite, args.limit)
    if len(queries) < 20:
        print(f"sx6_sentinel_driver: REFUSING: only {len(queries)} queries (need >=20)", file=sys.stderr)
        return 4
    out = run(queries, args.log_root, search=s.web_search, ddg=_ddg_timed if args.ddg_baseline else None)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
