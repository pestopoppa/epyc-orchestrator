"""Bridge pinned upstream CRM and Slack MCP modules to the paired runner.

The caller provisions isolated services and supplies reset/teardown callbacks;
this adapter never discovers or modifies host services. Import/configure both
modules with their isolated URLs and credentials before constructing it.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Callable

from .crm_provision import SEED_ACCOUNT_ID

UPSTREAM_COMMIT = "e0323a521ba4ef88f8e14c1eccf68d0a3d19a458"
SOURCE_SHA256 = {
    "crm": "9657a8dac3b3df63cf986c2f6684bc89aa28cf2f73cc12d061399887790173b3",
    "slack": "40d68afb3e9af09e5b0f05d93b7a86d8a3670ed5de30e41a3ef53e3338ff21be",
}


def load_pinned_module(path: Path, kind: str) -> Any:
    """Load the exact published MCP source, with URLs configured by caller first."""
    if kind not in SOURCE_SHA256:
        raise ValueError("unknown upstream module kind")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256[kind]:
        raise ValueError(f"upstream {kind} source does not match pinned bytes")
    spec = importlib.util.spec_from_file_location(f"dtap_pinned_{kind}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream {kind} module")
    module = importlib.util.module_from_spec(spec)
    # Execute the bytes that were checked, not a path that can be replaced in between.
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


class NativeServices:
    def __init__(self, crm: Any, slack: Any, *, reset: Callable[[], None],
                 teardown: Callable[[], None]):
        self.crm = crm
        self.slack = slack
        self._reset = reset
        self._teardown = teardown

    @staticmethod
    def _resolve(module: Any, name: str) -> Any:
        tool = getattr(module, name, None)
        fn = getattr(tool, "fn", None)
        if callable(fn):
            return fn
        if inspect.iscoroutinefunction(tool):
            return tool
        return None

    @staticmethod
    def _call(module: Any, name: str, arguments: dict) -> Any:
        fn = NativeServices._resolve(module, name)
        if not callable(fn):
            raise ValueError(f"native MCP callable unavailable: {name}")
        result = fn(**arguments)
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result

    def invoke(self, name: str, arguments: dict) -> Any:
        in_crm = self._resolve(self.crm, name) is not None
        in_slack = self._resolve(self.slack, name) is not None
        if in_crm == in_slack:
            raise ValueError(f"native MCP callable absent or ambiguous: {name}")
        return self._call(self.crm if in_crm else self.slack, name, arguments)

    def snapshot(self) -> dict:
        salesforce = {}
        account = self._call(self.crm, "get_entry", {"module_name": "Accounts", "id": SEED_ACCOUNT_ID})
        account = account.get("data") if isinstance(account, dict) else None
        if (not isinstance(account, dict) or account.get("id") != SEED_ACCOUNT_ID
                or (account.get("attributes") or {}).get("name") != "Meridian Group"):
            raise ValueError("deterministically seeded Meridian Group account is missing")
        salesforce["Accounts"] = [account]
        for module in ("Notes", "Tasks"):
            page = self._call(self.crm, "get_entry_list",
                              {"module_name": module, "page": 1, "page_size": 50})
            if (not isinstance(page, dict) or page.get("error") or page.get("errors")
                    or not isinstance(page.get("data"), list)):
                raise ValueError(f"invalid native CRM list for {module}")
            records = []
            for item in page["data"]:
                if not isinstance(item, dict) or not item.get("id") or not isinstance(item.get("attributes"), dict):
                    raise ValueError(f"invalid native CRM record for {module}")
                # The original judge reads the collection endpoint directly.
                # Some seeded rows have IDs rejected by the detail endpoint.
                records.append(item)
            salesforce[module] = records
        history = self._call(self.slack, "channels_history",
                             {"channel_name": "sales", "workspace_id": "W01"})
        if isinstance(history, dict):
            if history.get("error") or history.get("ok") is False:
                raise ValueError("native Slack history returned error")
            history = history.get("messages", history.get("data"))
        if not isinstance(history, list) or any(not isinstance(m, dict) for m in history):
            raise ValueError("invalid native Slack history")
        return {"salesforce": salesforce, "slack": {"channels": {"sales": history[:50]}}}

    def reset(self) -> dict:
        self._reset()
        return self.snapshot()

    def close(self) -> None:
        self._teardown()
