"""Sandboxed tool world for the codemode A/B/C probe.

* Every tool is side-effect-free or confined to one per-unit scratch dir (``World.root``, always under
  ``/mnt/raid0/llm/tmp/``). Paths are sandbox-relative; absolute paths and ``..`` are refused.
* Registry = the REAL ``orchestration/tool_registry.yaml`` schemas (55 tools; only the seven pure/sandbox
  ones below have working handlers, the rest raise "unavailable in probe") + six probe tools carrying
  names the REPL advertises (list_dir, file_info, peek, grep, json_parse, file_write_safe) + SYNTHETIC
  tools, every one named ``syn_*`` and listed in ``World.synthetic`` (distractors with realistic schemas,
  plus the two the novel-tier tasks need: ``syn_sku_price``, ``syn_ledger_post``).
* ``AuditedRegistry`` is the real ``ToolRegistry`` whose ``invoke`` additionally logs EVERY attempt
  (including arg-validation failures) to ``World.audit``: the source of param accuracy, failed-call and
  duplicate-write counts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics as _stats
from pathlib import Path
from typing import Any

from src.registry.tool_registry import (
    Tool,
    ToolCategory,
    ToolPermissions,
    ToolRegistry,
    load_from_yaml,
)

SCRATCH_PREFIX = "/mnt/raid0/llm/tmp/"
PROBE_ROLE = "worker_general"
MUTATING = {"file_write_safe", "write_file", "syn_ledger_post"}
# real-registry tools given sandbox handlers (everything else real is a schema-only distractor)
IMPLEMENTED_REAL = {"read_file", "write_file", "list_directory", "calculate", "statistics", "csv_to_json", "get_current_date"}


class ToolError(RuntimeError):
    pass


def sku_price_cents(sku: str) -> int:
    """Deterministic synthetic price table (no state)."""
    return 100 + int(hashlib.sha256(sku.encode()).hexdigest()[:6], 16) % 9900


class World:
    def __init__(self, root: str | Path, orch_root: str | Path):
        root = Path(root).resolve()
        if not str(root).startswith(SCRATCH_PREFIX):
            raise ValueError(f"world root must be under {SCRATCH_PREFIX}: {root}")
        if root.exists() and any(root.iterdir()):  # a crashed/previous run's state must never leak into this unit: rename aside, never delete
            root.rename(root.with_name(f"{root.name}.stale-{int(__import__('time').time() * 1000)}"))
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.audit: list[dict[str, Any]] = []
        self.ledger_path = root / "state" / "ledger.jsonl"
        self.synthetic: set[str] = set()
        self.groups: dict[str, list[str]] = {}
        self.group_desc: dict[str, str] = {}
        self.registry = AuditedRegistry(self)
        self._build(Path(orch_root))

    # --- sandbox paths -------------------------------------------------------------------------------
    def path(self, rel: str) -> Path:
        if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in Path(rel).parts:
            raise ToolError(f"path must be relative to the sandbox root, got {rel!r}")
        p = (self.root / rel).resolve()
        if self.root not in p.parents and p != self.root:
            raise ToolError("path escapes the sandbox")
        return p

    # --- tool implementations (all sandboxed) --------------------------------------------------------
    def _read(self, path: str, encoding: str = "utf-8", max_bytes: int = 1_000_000, **_: Any) -> str:
        p = self.path(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        return p.read_bytes()[:max_bytes].decode(encoding, "replace")

    def _write(self, path: str, content: str, mode: str = "w", **_: Any) -> str:
        p = self.path(path)
        if p.is_dir():
            raise ToolError(f"is a directory: {path}")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a" if mode == "a" else "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"wrote {len(content)} chars to {path}"

    def _list(self, path: str, pattern: str = "*", recursive: bool = False, **_: Any) -> list[str]:
        p = self.path(path)
        if not p.is_dir():
            raise ToolError(f"not a directory: {path}")
        it = p.rglob(pattern) if recursive else p.glob(pattern)
        return sorted(str(q.relative_to(self.root)) for q in it if q.is_file())

    def _list_dir(self, path: str) -> str:
        return json.dumps([Path(x).name for x in self._list(path)])

    def _file_info(self, path: str) -> str:
        p = self.path(path)
        if not p.exists():
            raise ToolError(f"no such path: {path}")
        return json.dumps({"path": path, "size": p.stat().st_size, "type": "dir" if p.is_dir() else "file"})

    def _peek(self, n: int, file_path: str | None = None) -> str:
        if file_path is None:
            return ""
        return self._read(file_path)[: int(n)]

    def _grep(self, pattern: str, file_path: str | None = None) -> list[str]:
        if not file_path:
            return []
        return [ln for ln in self._read(file_path).splitlines() if re.search(pattern, ln)][:100]

    def _json_parse(self, content: str, extract_path: str | None = None) -> Any:
        obj = json.loads(content)
        for part in [x for x in (extract_path or "").split(".") if x]:
            obj = obj[int(part)] if isinstance(obj, list) else obj[part]
        return obj

    def _calculate(self, expression: str) -> Any:
        if not re.fullmatch(r"[0-9+\-*/(). %]+", expression):
            raise ToolError("expression may contain only digits and + - * / ( ) . %")
        return eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - charset-restricted arithmetic

    def _statistics(self, data: list, metrics: list | None = None) -> dict[str, float]:
        nums = [float(x) for x in data]
        if not nums:
            raise ToolError("data is empty")
        out = {"count": float(len(nums)), "mean": _stats.fmean(nums), "min": min(nums), "max": max(nums),
               "sum": sum(nums), "median": _stats.median(nums), "stdev": _stats.pstdev(nums)}
        return {m: out[m] for m in metrics if m in out} if metrics else out

    def _csv_to_json(self, csv_content: str, delimiter: str = ",", has_header: bool = True) -> list:
        rows = [ln.split(delimiter) for ln in csv_content.strip().splitlines()]
        if not has_header:
            return rows
        head, body = rows[0], rows[1:]
        return [dict(zip(head, r)) for r in body]

    def _sku_price(self, sku: str) -> dict[str, Any]:
        return {"sku": sku, "price_cents": sku_price_cents(sku), "currency": "USD"}

    def _ledger_post(self, account: str, amount_cents: int, memo: str) -> dict[str, Any]:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"account": account, "amount_cents": amount_cents, "memo": memo}) + "\n")
        return {"ok": True, "memo": memo}

    def ledger(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        return [json.loads(x) for x in self.ledger_path.read_text().splitlines() if x]

    # --- registry ------------------------------------------------------------------------------------
    def _build(self, orch_root: Path) -> None:
        reg = self.registry
        load_from_yaml(reg, orch_root / "orchestration" / "tool_registry.yaml")
        real = {
            "read_file": self._read, "write_file": self._write, "list_directory": self._list,
            "calculate": self._calculate, "statistics": self._statistics, "csv_to_json": self._csv_to_json,
            "get_current_date": lambda **_: "2026-01-01",
        }

        def unavailable(name: str):
            def h(**_: Any) -> Any:
                raise ToolError(f"{name}: unavailable in the offline probe sandbox")
            return h

        for name, tool in list(reg._tools.items()):  # noqa: SLF001
            tool.handler = real.get(name, unavailable(name))
        # real tools carry domain tags for the C2 groups
        for name in reg._tools:  # noqa: SLF001
            self._tag(name, _REAL_GROUP.get(name, f"real_{reg._tools[name].category.value}"))  # noqa: SLF001
        P = lambda t, d=None, r=True: {"type": t, "description": d or "", "required": r}  # noqa: E731
        probe = [
            ("list_dir", "List directory contents as a JSON list of file names.", ToolCategory.FILE,
             {"path": P("string", "directory, sandbox-relative")}, self._list_dir, "files"),
            ("file_info", "Get file metadata (size, type).", ToolCategory.FILE,
             {"path": P("string")}, self._file_info, "files"),
            ("peek", "Return first n characters of a file.", ToolCategory.FILE,
             {"n": P("integer"), "file_path": P("string", r=False)}, self._peek, "files"),
            ("grep", "Search a file with a regex; returns matching lines.", ToolCategory.FILE,
             {"pattern": P("string"), "file_path": P("string", r=False)}, self._grep, "files"),
            ("json_parse", "Parse/validate JSON text; optional dotted extract_path.", ToolCategory.DATA,
             {"content": P("string"), "extract_path": P("string", r=False)}, self._json_parse, "data_formats"),
            ("file_write_safe", "Write text content to a file (creates parent dirs).", ToolCategory.FILE,
             {"path": P("string"), "content": P("string")}, self._write, "files"),
        ]
        for name, desc, cat, params, handler, grp in probe:
            reg.register_tool(Tool(name=name, description=desc, category=cat, parameters=params, handler=handler))
            self._tag(name, grp)
        for name in ("file_write_safe",):
            reg._tools[name].side_effects = ["modifies_files"]  # noqa: SLF001
        # synthetic needed-by-tasks tools
        self._syn("syn_sku_price", "Look up the current unit price in cents for a product SKU.", ToolCategory.DATA,
                  {"sku": P("string", "product SKU such as SKU-0042")}, self._sku_price, "catalog")
        self._syn("syn_ledger_post", "Post one entry to the accounting ledger. NOT idempotent: each call appends an entry.",
                  ToolCategory.SYSTEM, {"account": P("string"), "amount_cents": P("integer"), "memo": P("string")},
                  self._ledger_post, "ledger", effects=["modifies_files"])
        for name, desc, cat, params, grp in _DISTRACTORS:
            self._syn(name, desc, cat, {k: P(*v) if isinstance(v, tuple) else P(v) for k, v in params.items()},
                      lambda **_: (_ for _ in ()).throw(ToolError("synthetic distractor: no implementation")), grp)
        all_names = list(reg._tools)  # noqa: SLF001
        reg.set_role_permissions(PROBE_ROLE, ToolPermissions(web_access=True, allowed_tools=all_names))

    def _tag(self, name: str, group: str) -> None:
        self.groups.setdefault(group, []).append(name)

    def _syn(self, name, desc, cat, params, handler, group, effects=None) -> None:
        self.registry.register_tool(Tool(name=name, description=desc, category=cat, parameters=params,
                                         handler=handler, side_effects=effects or []))
        self.synthetic.add(name)
        self._tag(name, group)

    def tool_dicts(self) -> list[dict[str, Any]]:
        return self.registry.list_tools(role=PROBE_ROLE)

    def call(self, name: str, **kwargs: Any) -> Any:
        """The path used by direct (arm A) calls and by REPL builtins: real registry.invoke."""
        return self.registry.invoke(name, PROBE_ROLE, **kwargs)


class AuditedRegistry(ToolRegistry):
    def __init__(self, world: World):
        super().__init__()
        self.world = world

    def invoke(self, tool_name: str, role: str, *a: Any, **kwargs: Any) -> Any:
        rec = {"tool": tool_name, "args": {k: v for k, v in kwargs.items() if k not in ("caller_type", "chain_id", "chain_index", "context")}}
        try:
            out = super().invoke(tool_name, role, *a, **kwargs)
            rec["ok"] = True
            self.world.audit.append(rec)
            return out
        except Exception as exc:  # noqa: BLE001 - recorded then re-raised unchanged
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
            self.world.audit.append(rec)
            raise


_REAL_GROUP = {
    "http_get": "web", "http_post": "web", "web_search": "web", "web_research": "web", "fetch_wikipedia": "reference_search",
    "search_arxiv": "reference_search", "search_papers": "reference_search", "search_wikipedia": "reference_search",
    "get_wikipedia_article": "reference_search", "search_books": "reference_search",
    "json_query": "data_formats", "csv_to_json": "data_formats", "json_to_csv": "data_formats", "sql_query": "data_formats",
    "read_file": "files", "write_file": "files", "list_directory": "files",
    "calculate": "math_stats", "statistics": "math_stats",
    "task_create": "tasks", "task_update": "tasks", "task_list": "tasks", "budget_override": "tasks",
    "user_profile": "user_memory", "user_search": "user_memory", "user_context": "user_memory", "user_conclude": "user_memory",
}

GROUP_DESC = {
    "files": "read, list, inspect and write files in the working directory",
    "data_formats": "parse and convert JSON, CSV and SQL-like structured data",
    "math_stats": "arithmetic, descriptive statistics and numeric analysis",
    "catalog": "product catalog, SKU and price lookups",
    "ledger": "accounting ledger entries, balances and invoices",
    "web": "HTTP requests and web search",
    "reference_search": "encyclopedia, papers and book search",
    "tasks": "task tracking and budgets",
    "user_memory": "user profile and memory",
    "calendar": "calendar events and scheduling",
    "email": "email drafting, sending and inbox search",
    "shipping": "shipping rates, labels and tracking",
    "hr": "employee records, leave and payroll",
    "devops": "deployments, services, logs and alerts",
    "geo": "geocoding, distances and routing",
    "text": "text transformation, translation and summarisation helpers",
    "units": "unit and currency conversion",
    "real_code": "run code, shell commands, git status and linting",
    "real_math": "numerical maths: ODEs, optimisation, Monte Carlo, matrices, symbolic solving",
    "real_data": "plotting, histograms and rendering of data",
    "real_llm": "text embedding, similarity search and classification",
    "real_specialized": "OCR and vision analysis of images",
    "real_system": "current date and time",
}

_S = "string"
_I = "integer"
_B = "boolean"
_W = ToolCategory.SPECIALIZED
_D = ToolCategory.DATA
_SYS = ToolCategory.SYSTEM
_M = ToolCategory.MATH
_DISTRACTORS: list[tuple[str, str, ToolCategory, dict[str, Any], str]] = [
    ("syn_sku_stock", "Get the units in stock for a product SKU at a warehouse.", _D, {"sku": _S, "warehouse": (_S, "", False)}, "catalog"),
    ("syn_sku_description", "Return the marketing description of a product SKU.", _D, {"sku": _S}, "catalog"),
    ("syn_sku_search", "Search the catalog by keyword and return matching SKUs.", _D, {"query": _S, "limit": (_I, "", False)}, "catalog"),
    ("syn_price_history", "Return the price history (cents) for a SKU over N days.", _D, {"sku": _S, "days": _I}, "catalog"),
    ("syn_price_set", "Set a new unit price for a SKU (requires approval).", _SYS, {"sku": _S, "price_cents": _I}, "catalog"),
    ("syn_sku_supplier", "Look up the supplier of a SKU.", _D, {"sku": _S}, "catalog"),
    ("syn_ledger_balance", "Get the current balance in cents of a ledger account.", _D, {"account": _S}, "ledger"),
    ("syn_ledger_void", "Void a posted ledger entry by entry id.", _SYS, {"entry_id": _S, "reason": _S}, "ledger"),
    ("syn_ledger_list", "List ledger entries for an account in a date range.", _D, {"account": _S, "since": (_S, "", False)}, "ledger"),
    ("syn_invoice_create", "Create an invoice for a customer from line items.", _SYS, {"customer": _S, "line_items": "array"}, "ledger"),
    ("syn_invoice_status", "Get the payment status of an invoice.", _D, {"invoice_id": _S}, "ledger"),
    ("syn_tax_rate", "Return the sales tax rate for a region code.", _D, {"region": _S}, "ledger"),
    ("syn_currency_convert", "Convert an amount between two currency codes at today's rate.", _M, {"amount": _I, "from_ccy": _S, "to_ccy": _S}, "units"),
    ("syn_unit_convert", "Convert a value between units (kg/lb, km/mi, C/F).", _M, {"value": _I, "from_unit": _S, "to_unit": _S}, "units"),
    ("syn_stats_zscore", "Compute z-scores for a list of numbers.", _M, {"data": "array"}, "math_stats"),
    ("syn_stats_percentile", "Compute a percentile of a numeric list.", _M, {"data": "array", "p": _I}, "math_stats"),
    ("syn_stats_histogram_bins", "Return histogram bin counts for numbers.", _M, {"data": "array", "bins": _I}, "math_stats"),
    ("syn_stats_correlation", "Pearson correlation of two numeric series.", _M, {"x": "array", "y": "array"}, "math_stats"),
    ("syn_csv_validate", "Validate that CSV text has a consistent column count.", _D, {"csv_content": _S}, "data_formats"),
    ("syn_csv_sort", "Sort CSV text by a column name.", _D, {"csv_content": _S, "column": _S}, "data_formats"),
    ("syn_json_diff", "Structural diff of two JSON documents.", _D, {"a": _S, "b": _S}, "data_formats"),
    ("syn_yaml_to_json", "Convert YAML text to JSON.", _D, {"yaml_content": _S}, "data_formats"),
    ("syn_file_checksum", "SHA-256 checksum of a file.", _SYS, {"path": _S}, "files"),
    ("syn_file_copy", "Copy a file to a new path.", _SYS, {"src": _S, "dst": _S}, "files"),
    ("syn_file_move", "Move or rename a file.", _SYS, {"src": _S, "dst": _S}, "files"),
    ("syn_file_delete", "Delete a file.", _SYS, {"path": _S}, "files"),
    ("syn_file_tail", "Return the last n lines of a text file.", _SYS, {"path": _S, "n": _I}, "files"),
    ("syn_dir_tree", "Render a directory tree up to a depth.", _SYS, {"path": _S, "depth": (_I, "", False)}, "files"),
    ("syn_calendar_create", "Create a calendar event.", _SYS, {"title": _S, "start": _S, "end": _S}, "calendar"),
    ("syn_calendar_list", "List calendar events in a date range.", _D, {"start": _S, "end": _S}, "calendar"),
    ("syn_calendar_free_slots", "Find free meeting slots for attendees.", _D, {"attendees": "array", "duration_min": _I}, "calendar"),
    ("syn_calendar_cancel", "Cancel a calendar event by id.", _SYS, {"event_id": _S}, "calendar"),
    ("syn_email_send", "Send an email message.", _SYS, {"to": _S, "subject": _S, "body": _S}, "email"),
    ("syn_email_search", "Search the inbox.", _D, {"query": _S, "limit": (_I, "", False)}, "email"),
    ("syn_email_draft", "Create an email draft.", _SYS, {"to": _S, "subject": _S, "body": _S}, "email"),
    ("syn_email_labels", "List or apply mailbox labels.", _SYS, {"message_id": _S, "label": (_S, "", False)}, "email"),
    ("syn_ship_rate", "Quote shipping rates for a parcel.", _D, {"weight_g": _I, "from_zip": _S, "to_zip": _S}, "shipping"),
    ("syn_ship_label", "Purchase a shipping label.", _SYS, {"order_id": _S, "service": _S}, "shipping"),
    ("syn_ship_track", "Track a parcel by tracking number.", _D, {"tracking_number": _S}, "shipping"),
    ("syn_ship_address_validate", "Validate and normalise a postal address.", _D, {"address": _S}, "shipping"),
    ("syn_hr_employee", "Look up an employee record by id.", _D, {"employee_id": _S}, "hr"),
    ("syn_hr_leave_balance", "Get remaining leave days for an employee.", _D, {"employee_id": _S}, "hr"),
    ("syn_hr_leave_request", "File a leave request.", _SYS, {"employee_id": _S, "start": _S, "end": _S}, "hr"),
    ("syn_hr_payroll_run", "Compute payroll for a month.", _SYS, {"month": _S}, "hr"),
    ("syn_deploy_status", "Get the deployment status of a service.", _D, {"service": _S}, "devops"),
    ("syn_deploy_rollback", "Roll back a service to the previous release.", _SYS, {"service": _S}, "devops"),
    ("syn_log_search", "Search service logs for a pattern.", _D, {"service": _S, "pattern": _S, "limit": (_I, "", False)}, "devops"),
    ("syn_alert_ack", "Acknowledge a monitoring alert.", _SYS, {"alert_id": _S}, "devops"),
    ("syn_metric_query", "Query a time-series metric.", _D, {"metric": _S, "window_min": _I}, "devops"),
    ("syn_geocode", "Geocode a street address to lat/lon.", _D, {"address": _S}, "geo"),
    ("syn_geo_distance", "Great-circle distance in km between two lat/lon points.", _M, {"lat1": _I, "lon1": _I, "lat2": _I, "lon2": _I}, "geo"),
    ("syn_route_plan", "Plan a driving route between two addresses.", _D, {"origin": _S, "destination": _S}, "geo"),
    ("syn_timezone_lookup", "Timezone for lat/lon.", _D, {"lat": _I, "lon": _I}, "geo"),
    ("syn_text_translate", "Translate text to a target language.", _W, {"text": _S, "target_lang": _S}, "text"),
    ("syn_text_summarize", "Summarise text to a target length.", _W, {"text": _S, "max_words": _I}, "text"),
    ("syn_text_regex_extract", "Extract regex matches from text.", _D, {"text": _S, "pattern": _S}, "text"),
    ("syn_text_dedupe_lines", "Remove duplicate lines from text.", _D, {"text": _S}, "text"),
    ("syn_text_slugify", "Slugify a title.", _D, {"text": _S}, "text"),
    ("syn_text_wordcount", "Count words, lines and characters.", _D, {"text": _S}, "text"),
    ("syn_text_diff", "Unified diff between two texts.", _D, {"a": _S, "b": _S}, "text"),
]
