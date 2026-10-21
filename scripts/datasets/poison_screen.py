"""Per-sample poison screen for F3 W2 dataset builders (F3-W3-P1).

Deterministic, no-inference screen over raw trace rows BEFORE they can become
training examples.  Three signals, each reported by reason code only (no raw
text is ever copied into a report):

* ``outbound_url_tool_call:<host>`` -- a tool call whose arguments reach a host
  outside the run allowlist (URL, or a curl/wget/nc/ssh-style command target).
* ``credential_path_read:<id>`` -- a tool call whose arguments touch a
  credential-bearing path (ssh keys, cloud credentials, dotenv, netrc, ...).
* ``fetched_content_instruction:<id>`` -- content that originated from a fetch
  (web/browser/http tool result, or a row tagged external/fetched) containing
  instruction-shaped text aimed at the model.

External trace sources must also carry provenance (commit sha + sha256, the
SEC-ART-1 shape); ``assert_exportable`` refuses an export that contains an
unscreened row or an external row without it.

Threat model: an honest-but-careless data pipeline ingesting traces that may
contain planted exfiltration or injection behaviour (intake-2199#02,
intake-2210#03).  Not an adversarial classifier; false positives are
quarantined for review rather than trained on.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

SCREEN_VERSION = "poison_screen.v1"
REPORT_SCHEMA = "poison_screen_report.v1"

#: Hosts every run may talk to.  Callers extend this with the run allowlist.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

_URL_RE = re.compile(r"\b(?:https?|ftp|wss?)://(?:[^/\s@\"'\\]*@)?([A-Za-z0-9._\-]+|\[[0-9A-Fa-f:]+\])", re.IGNORECASE)
_CMD_HOST_RE = re.compile(
    r"\b(?:curl|wget|nc|ncat|netcat|scp|rsync|ssh|sftp|ftp|telnet)\b[^|;&\n]*?"
    r"(?:\s|@)((?:[A-Za-z0-9\-]+\.)+[A-Za-z]{2,}|\d{1,3}(?:\.\d{1,3}){3})(?::\d+)?\b",
    re.IGNORECASE,
)

# Bare "host" tokens after curl/wget/ssh... that are really file names.
_FILE_LIKE_TLDS = frozenset(
    "json jsonl txt md py sh yaml yml log csv tsv tar gz tgz zip xml html htm js ts c h cpp so bin conf cfg ini toml lock out pdf png jpg".split()
)

_CREDENTIAL_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ssh_key", re.compile(r"(?:^|[\s\"'=:/])(?:~|\$HOME|/home/[^/\s]+|/root)?/?\.ssh/(?:id_[a-z0-9]+|authorized_keys|config|known_hosts)\b", re.I)),
    ("ssh_private", re.compile(r"\bid_(?:rsa|ed25519|ecdsa|dsa)\b(?!\.pub)", re.I)),
    ("aws_credentials", re.compile(r"\.aws/(?:credentials|config)\b", re.I)),
    ("gcloud_config", re.compile(r"\.config/gcloud\b|application_default_credentials\.json", re.I)),
    ("kube_config", re.compile(r"\.kube/config\b", re.I)),
    ("docker_config", re.compile(r"\.docker/config\.json\b", re.I)),
    ("netrc", re.compile(r"(?:^|[\s\"'=:/])\.netrc\b", re.I)),
    ("git_credentials", re.compile(r"\.git-credentials\b|\.config/git/credentials\b", re.I)),
    ("npm_pypi_token", re.compile(r"(?:^|[\s\"'=:/])\.(?:npmrc|pypirc)\b", re.I)),
    ("dotenv", re.compile(r"(?:^|[\s\"'=:/])\.env(?:\.[a-z0-9_-]+)?(?=$|[\s\"'),;])", re.I)),
    ("etc_shadow", re.compile(r"/etc/(?:shadow|gshadow|sudoers)\b", re.I)),
    ("proc_environ", re.compile(r"/proc/(?:self|\d+)/environ\b", re.I)),
    ("gnupg", re.compile(r"\.gnupg\b", re.I)),
    ("claude_credentials", re.compile(r"\.claude/\.?credentials|\.claude\.json\b", re.I)),
    ("hf_token", re.compile(r"\.(?:cache/)?huggingface/token\b", re.I)),
)

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_prior", re.compile(r"\bignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|messages?|rules?)", re.I)),
    ("disregard_prior", re.compile(r"\bdisregard\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|your)\b", re.I)),
    ("role_override", re.compile(r"\b(?:you\s+are\s+now|from\s+now\s+on\s+you|new\s+instructions?\s*:|system\s+prompt\s*:|\[\s*system\s*\])", re.I)),
    ("run_command", re.compile(r"\b(?:run|execute|invoke)\s+(?:the\s+following|this)\s+(?:shell\s+)?(?:command|script|code)\b", re.I)),
    ("send_to_url", re.compile(r"\b(?:send|post|upload|forward|exfiltrate)\b[^.\n]{0,80}\b(?:to|at)\s+https?://", re.I)),
    ("conceal", re.compile(r"\b(?:do\s+not|don't|never)\s+(?:tell|inform|mention|reveal|show)\s+(?:this\s+to\s+)?the\s+user\b", re.I)),
    ("assistant_must", re.compile(r"\b(?:assistant|ai|model|agent)\s*[,:]?\s*(?:you\s+)?(?:must|should|need\s+to)\s+(?:now\s+)?(?:immediately\s+)?(?:run|execute|read|send|call|fetch|download)\b", re.I)),
)

_TOOL_NAME_KEYS = ("tool", "tool_name", "tool_call", "function", "tool_use", "name")
_TOOL_ARG_KEYS = ("arguments", "args", "input", "parameters", "params", "command", "cmd", "url", "path", "file_path", "code")
_FETCH_ORIGINS = frozenset(
    {"web", "web_fetch", "webfetch", "fetch", "fetched", "browser", "http", "https", "url", "external", "internet", "search", "web_search"}
)
_ORIGIN_KEYS = ("source", "origin", "provenance", "tool_name", "tool", "type", "source_kind", "trace_source_kind", "kind")

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class UnscreenedExternalTraceError(RuntimeError):
    """An export contained an unscreened or provenance-less external trace."""


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_dicts(child)


def _tool_call_texts(row: Any) -> list[str]:
    """Argument text of every tool-call-shaped dict anywhere in ``row``."""
    out: list[str] = []
    for node in _iter_dicts(row):
        has_name = any(isinstance(node.get(k), (str, dict)) for k in _TOOL_NAME_KEYS)
        args = [node[k] for k in _TOOL_ARG_KEYS if k in node]
        fn = node.get("function")
        if isinstance(fn, dict):
            args.extend(fn[k] for k in _TOOL_ARG_KEYS if k in fn)
        if has_name and args:
            out.append(" ".join(_text(a) for a in args))
    return out


def _host_allowed(host: str, allowed: frozenset[str]) -> bool:
    host = host.strip("[]").lower().rstrip(".")
    return any(host == a or host.endswith("." + a) for a in allowed)


def _is_fetch_origin(node: dict[str, Any]) -> bool:
    if node.get("external") is True or node.get("fetched") is True:
        return True
    for key in _ORIGIN_KEYS:
        val = node.get(key)
        if isinstance(val, str) and val.strip().lower() in _FETCH_ORIGINS:
            return True
    return False


def _fetched_texts(row: Any) -> list[str]:
    out: list[str] = []
    seen: set[int] = set()
    for node in _iter_dicts(row):
        if id(node) in seen or not _is_fetch_origin(node):
            continue
        for sub in _iter_dicts(node):
            seen.add(id(sub))
        out.append(" ".join(_text(v) for k, v in node.items() if k not in _ORIGIN_KEYS))
    return out


def screen_row(row: dict[str, Any], *, allowed_hosts: Iterable[str] = ()) -> list[str]:
    """Return sorted, de-duplicated reason codes; an empty list means pass."""
    allowed = frozenset(h.lower() for h in allowed_hosts) | LOCAL_HOSTS
    reasons: set[str] = set()
    for text in _tool_call_texts(row):
        for host in _URL_RE.findall(text) + _CMD_HOST_RE.findall(text):
            if host.rsplit(".", 1)[-1].lower() in _FILE_LIKE_TLDS:
                continue
            if not _host_allowed(host, allowed):
                reasons.add(f"outbound_url_tool_call:{host.strip('[]').lower()}")
        for ident, pattern in _CREDENTIAL_PATHS:
            if pattern.search(text):
                reasons.add(f"credential_path_read:{ident}")
    for text in _fetched_texts(row):
        for ident, pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                reasons.add(f"fetched_content_instruction:{ident}")
    return sorted(reasons)


def is_external(row: dict[str, Any]) -> bool:
    return (
        row.get("external") is True
        or str(row.get("trace_source_kind") or row.get("source_kind") or "").lower() == "external"
    )


def provenance_problems(row: dict[str, Any]) -> list[str]:
    """SEC-ART-1 shape: external sources carry a commit sha and a sha256."""
    problems = []
    if not _HEX40.match(str(row.get("source_commit_sha") or "").lower()):
        problems.append("source_commit_sha")
    if not _HEX64.match(str(row.get("source_sha256") or "").lower()):
        problems.append("source_sha256")
    return problems


def screen_stamp(reasons: list[str]) -> dict[str, Any]:
    return {
        "version": SCREEN_VERSION,
        "status": "quarantined" if reasons else "pass",
        "reasons": list(reasons),
    }


def assert_exportable(examples: Iterable[dict[str, Any]]) -> None:
    """Refuse an export holding an unscreened/quarantined row or a bare external trace."""
    unscreened = quarantined = no_prov = 0
    for ex in examples:
        stamp = ex.get("poison_screen")
        if not isinstance(stamp, dict) or stamp.get("version") != SCREEN_VERSION:
            unscreened += 1
        elif stamp.get("status") != "pass":
            quarantined += 1
        if ex.get("trace_origin") == "external" and provenance_problems(ex):
            no_prov += 1
    if unscreened or quarantined or no_prov:
        raise UnscreenedExternalTraceError(
            "refusing export: "
            f"{unscreened} unscreened, {quarantined} quarantined, "
            f"{no_prov} external without commit sha + sha256 provenance"
        )


def build_report(
    entries: list[dict[str, Any]], *, allowed_hosts: Iterable[str], external_total: int
) -> dict[str, Any]:
    """``entries``: one per screened row: {example_id, source_line, external, reasons}."""
    reason_counts: dict[str, int] = {}
    quarantined = []
    for e in entries:
        if e["reasons"]:
            quarantined.append(
                {"example_id": e["example_id"], "source_line": e["source_line"], "external": e["external"], "reasons": e["reasons"]}
            )
            for r in e["reasons"]:
                kind = r.split(":", 1)[0]
                reason_counts[kind] = reason_counts.get(kind, 0) + 1
    return {
        "schema_version": REPORT_SCHEMA,
        "screen_version": SCREEN_VERSION,
        "allowed_hosts": sorted(frozenset(h.lower() for h in allowed_hosts) | LOCAL_HOSTS),
        "screened": len(entries),
        "passed": len(entries) - len(quarantined),
        "quarantined": len(quarantined),
        "external_rows": external_total,
        "reason_counts": dict(sorted(reason_counts.items())),
        "quarantined_rows": quarantined,
    }
