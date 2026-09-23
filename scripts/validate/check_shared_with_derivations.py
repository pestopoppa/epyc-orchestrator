#!/usr/bin/env python3
"""Recompute every surface that restates ``shared_with`` and diff it against its source.

THE DEFECT THIS CLOSES (session friction audit 2026-09-22 §2, `restated derivation`):
a fact that is a function of a source is copied into a second place as a literal, and
the gate compares literal to literal -- or merely tests existence -- instead of
RECOMPUTING the fact from the source and diffing.

The concrete instance: five surfaces each independently restate
``server_mode.<host>.shared_with`` from the MASTER registry, so moving one role onto
another host took five hand edits, found one error at a time across nine pipeline runs
with the stack down.

    surface                                              what it restates
    -----------------------------------------------------------------------------
    launch_manifest.yaml  port_map                       which port a role answers on
    launch_manifest.yaml  role_launch_meta               whether a role launches at all
    stack_topology.yaml   numa_config.<role>             an alias must have none
    orchestration/procedures/*.yaml role enums           the live role set
    model_registry.yaml   roles.<alias>.model            must equal the host's artifact

The fourth is ALREADY solved correctly by ``scripts/registry/sync_procedure_role_enums.py``
(it regenerates from the compiled priors). This module does the other four, in the shape
that one solves them: derive from the source, diff, report the line and the expected value.

SOURCE OF TRUTH is the MASTER registry in epyc-inference-research -- the hand-edited file
a stack change touches FIRST -- not the lean copy. ``stack_manifest.validate_declaration_parity``
says "master" in its messages but reads the LEAN registry (stack_manifest.py:54-61 admits
this cost a debugging session), so it cannot fire until a lean compile has already succeeded.
This checker fires before anything is compiled at all.

WHAT IT REFUSES TO DO: ``--fix`` never writes either ``model_registry.yaml``. Registry
findings are reported as manual, with the exact expected value. The registry is the source;
a tool that "fixes" the source has inverted the derivation.

Exit codes: 0 clean, 1 findings, 2 usage / unreadable input.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER_REGISTRY = Path(
    "/mnt/raid0/llm/epyc-inference-research/orchestration/model_registry.yaml"
)
DEFAULT_LAUNCH_MANIFEST = REPO_ROOT / "orchestration" / "launch_manifest.yaml"
DEFAULT_STACK_TOPOLOGY = REPO_ROOT / "orchestration" / "stack_topology.yaml"

# Surfaces this tool may rewrite. The master registry is deliberately absent.
FIXABLE_FILES = {"launch_manifest.yaml", "stack_topology.yaml"}


class SourceError(RuntimeError):
    """A declared source could not be read or did not have the expected shape."""


# ---------------------------------------------------------------------------
# line locator -- "with the exact line" is half the value of this tool
# ---------------------------------------------------------------------------


class YamlLines:
    """1-based line numbers for key paths, from PyYAML's composed node tree.

    Returns the line of the deepest key on the path that exists, so a missing
    leaf still points at its parent block instead of at nothing.
    """

    def __init__(self, text: str, path: Path) -> None:
        self.path = path
        try:
            self._root = yaml.compose(text)
        except yaml.YAMLError as exc:  # pragma: no cover - malformed input
            raise SourceError(f"{path}: {exc}") from exc

    def line(self, *keys: str) -> int | None:
        node = self._root
        best: int | None = None
        for key in keys:
            if not isinstance(node, yaml.MappingNode):
                return best
            found = None
            for key_node, value_node in node.value:
                if getattr(key_node, "value", None) == key:
                    found = (key_node, value_node)
                    break
            if found is None:
                return best
            best = found[0].start_mark.line + 1
            node = found[1]
        return best


# ---------------------------------------------------------------------------
# findings
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    surface: str
    role: str
    file: str
    line: int | None
    found: str
    expected: str
    message: str
    fix: str | None = None  # None => not machine-fixable

    @property
    def fixable(self) -> bool:
        return self.fix is not None

    def render(self) -> str:
        where = f"{self.file}:{self.line}" if self.line else self.file
        tag = "FIXABLE" if self.fixable else "MANUAL "
        return (
            f"[{tag}] {self.surface}: {self.message}\n"
            f"          at {where}\n"
            f"          found:    {self.found}\n"
            f"          expected: {self.expected}"
        )


# ---------------------------------------------------------------------------
# sources + derivation
# ---------------------------------------------------------------------------


def _load(path: Path) -> tuple[dict[str, Any], YamlLines, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SourceError(f"cannot read {path}: {exc}") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SourceError(f"{path} did not parse to a mapping")
    return data, YamlLines(text, path), text


@dataclass
class Sources:
    master: dict[str, Any]
    master_lines: YamlLines
    master_path: Path
    manifest: dict[str, Any]
    manifest_lines: YamlLines
    manifest_path: Path
    topology: dict[str, Any]
    topology_lines: YamlLines
    topology_path: Path

    @property
    def server_mode(self) -> dict[str, dict[str, Any]]:
        raw = self.master.get("server_mode") or {}
        return {k: v for k, v in raw.items() if isinstance(v, dict)}

    @property
    def roles(self) -> dict[str, dict[str, Any]]:
        raw = self.master.get("roles") or {}
        return {k: v for k, v in raw.items() if isinstance(v, dict)}

    @property
    def port_map(self) -> dict[str, Any]:
        return self.manifest.get("port_map") or {}

    @property
    def role_launch_meta(self) -> dict[str, dict[str, Any]]:
        raw = self.manifest.get("role_launch_meta") or {}
        return {k: v for k, v in raw.items() if isinstance(v, dict)}

    @property
    def numa_config(self) -> dict[str, Any]:
        raw = self.topology.get("numa_config") or {}
        return {k: v for k, v in raw.items() if isinstance(v, dict)}


def load_sources(master: Path, manifest: Path, topology: Path) -> Sources:
    m, ml, _ = _load(master)
    lm, lml, _ = _load(manifest)
    st, stl, _ = _load(topology)
    return Sources(m, ml, master, lm, lml, manifest, st, stl, topology)


@dataclass
class Derivation:
    """Everything the four surfaces SHOULD say, computed from shared_with alone."""

    alias_to_host: dict[str, str] = field(default_factory=dict)
    host_to_aliases: dict[str, list[str]] = field(default_factory=dict)

    @property
    def hosts(self) -> set[str]:
        return set(self.host_to_aliases)

    @property
    def aliases(self) -> set[str]:
        return set(self.alias_to_host)


def derive(sources: Sources) -> Derivation:
    d = Derivation()
    for host, cfg in sources.server_mode.items():
        shared = cfg.get("shared_with")
        if not isinstance(shared, list):
            continue
        aliases = [str(a) for a in shared if isinstance(a, str)]
        if not aliases:
            continue
        d.host_to_aliases[host] = aliases
        for alias in aliases:
            d.alias_to_host[alias] = host
    return d


# ---------------------------------------------------------------------------
# artifact identity
# ---------------------------------------------------------------------------


def _stem(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    name = value.strip().rsplit("/", 1)[-1]
    if name.lower().endswith(".gguf"):
        name = name[: -len(".gguf")]
    return name.lower()


def artifact_stems(block: Any) -> set[str]:
    """Every acceptable spelling of the GGUF a registry block names."""
    stems: set[str] = set()
    if isinstance(block, str):
        s = _stem(block)
        if s:
            stems.add(s)
        return stems
    if isinstance(block, dict):
        for key in ("name", "path", "model", "model_path"):
            value = block.get(key)
            if isinstance(value, (str, dict)):
                stems |= artifact_stems(value)
    return stems


def server_artifact_stems(cfg: dict[str, Any]) -> set[str]:
    stems = artifact_stems(cfg.get("model"))
    stems |= artifact_stems(cfg.get("model_path"))
    return stems


def role_artifact_stems(role_cfg: dict[str, Any]) -> set[str]:
    return artifact_stems(role_cfg.get("model"))


def _describe(stems: Iterable[str]) -> str:
    values = sorted(stems)
    return ", ".join(values) if values else "<none declared>"


# ---------------------------------------------------------------------------
# the four checks
# ---------------------------------------------------------------------------


def check_alias_of_coherence(sources: Sources, d: Derivation) -> list[Finding]:
    """`alias_of` is documentation; `shared_with` is the binding. They must agree.

    This is the ROOT of the 2026-09-22 `ingest_long_context` and `worker` failures:
    both carried `alias_of`, neither was in the host's `shared_with`, so the
    registry validator could not resolve their fleet and called it a phantom --
    four surfaces downstream of where the mistake actually was.
    """
    findings: list[Finding] = []
    sm = sources.server_mode
    for role, cfg in sorted(sm.items()):
        host = cfg.get("alias_of")
        if not isinstance(host, str) or not host:
            continue
        host_cfg = sm.get(host)
        if not isinstance(host_cfg, dict):
            findings.append(
                Finding(
                    surface="master server_mode.alias_of",
                    role=role,
                    file=str(sources.master_path),
                    line=sources.master_lines.line("server_mode", role, "alias_of"),
                    found=f"alias_of: {host}",
                    expected=f"server_mode.{host} to exist",
                    message=f"role {role!r} is alias_of {host!r}, which has no server_mode row",
                )
            )
            continue
        shared = host_cfg.get("shared_with")
        listed = isinstance(shared, list) and role in shared
        if not listed:
            findings.append(
                Finding(
                    surface="master server_mode.shared_with",
                    role=role,
                    file=str(sources.master_path),
                    line=sources.master_lines.line("server_mode", host, "shared_with")
                    or sources.master_lines.line("server_mode", host),
                    found=f"server_mode.{host}.shared_with = {shared!r}",
                    expected=f"server_mode.{host}.shared_with to contain {role!r}",
                    message=(
                        f"role {role!r} declares alias_of: {host} but is absent from that host's "
                        f"shared_with. alias_of is DOCUMENTATION; shared_with is the load-bearing "
                        f"binding, so nothing resolves this role to a launching process "
                        f"('a fleet nothing launches is a phantom')"
                    ),
                )
            )
    return findings


def check_port_map(sources: Sources, d: Derivation) -> list[Finding]:
    """port_map restates `which port a role answers on` = its HOST's port."""
    findings: list[Finding] = []
    sm = sources.server_mode
    port_map = sources.port_map

    def declared_port(role: str) -> Any:
        cfg = sm.get(role) or {}
        return cfg.get("port")

    for alias in sorted(d.aliases):
        host = d.alias_to_host[alias]
        expected = declared_port(host)
        if expected is None:
            continue
        if alias not in port_map:
            findings.append(
                Finding(
                    surface="launch_manifest.port_map",
                    role=alias,
                    file=str(sources.manifest_path),
                    line=sources.manifest_lines.line("port_map"),
                    found="<no port_map entry>",
                    expected=f"{alias}: {expected}",
                    message=(
                        f"port for role {alias!r}: launcher declares nothing, master "
                        f"({host}/shared_with) declares {expected}"
                    ),
                    fix=f"port_map.{alias}={expected}",
                )
            )
            continue
        actual = port_map[alias]
        if str(actual) != str(expected):
            findings.append(
                Finding(
                    surface="launch_manifest.port_map",
                    role=alias,
                    file=str(sources.manifest_path),
                    line=sources.manifest_lines.line("port_map", alias),
                    found=str(actual),
                    expected=str(expected),
                    message=(
                        f"port for role {alias!r}: launcher declares {actual}, master "
                        f"({host}/shared_with) declares {expected}"
                    ),
                    fix=f"port_map.{alias}={expected}",
                )
            )

    for host in sorted(d.hosts):
        expected = declared_port(host)
        if expected is None:
            continue
        if host not in port_map:
            findings.append(
                Finding(
                    surface="launch_manifest.port_map",
                    role=host,
                    file=str(sources.manifest_path),
                    line=sources.manifest_lines.line("port_map"),
                    found="<no port_map entry>",
                    expected=f"{host}: {expected}",
                    message=f"host role {host!r} serves {len(d.host_to_aliases[host])} "
                    f"alias(es) but has no port_map entry",
                    fix=f"port_map.{host}={expected}",
                )
            )
        elif str(port_map[host]) != str(expected):
            findings.append(
                Finding(
                    surface="launch_manifest.port_map",
                    role=host,
                    file=str(sources.manifest_path),
                    line=sources.manifest_lines.line("port_map", host),
                    found=str(port_map[host]),
                    expected=str(expected),
                    message=(
                        f"port for host role {host!r}: launcher declares {port_map[host]}, "
                        f"master declares {expected}"
                    ),
                    fix=f"port_map.{host}={expected}",
                )
            )

    # An alias carrying its own server_mode row restates the host's port there too.
    for alias in sorted(d.aliases):
        cfg = sm.get(alias)
        if not isinstance(cfg, dict):
            continue
        host = d.alias_to_host[alias]
        expected = declared_port(host)
        own = cfg.get("port")
        if expected is not None and own is not None and str(own) != str(expected):
            findings.append(
                Finding(
                    surface="master server_mode.<alias>.port",
                    role=alias,
                    file=str(sources.master_path),
                    line=sources.master_lines.line("server_mode", alias, "port"),
                    found=str(own),
                    expected=str(expected),
                    message=(
                        f"alias {alias!r} keeps its own server_mode.port, which restates "
                        f"the port of its host {host!r}"
                    ),
                )
            )
    return findings


def check_role_launch_meta(sources: Sources, d: Derivation) -> list[Finding]:
    """role_launch_meta restates `whether a role launches at all` = whether it is an alias."""
    findings: list[Finding] = []
    meta = sources.role_launch_meta
    numa = sources.numa_config

    for alias in sorted(d.aliases):
        if alias in meta:
            host = d.alias_to_host[alias]
            findings.append(
                Finding(
                    surface="launch_manifest.role_launch_meta",
                    role=alias,
                    file=str(sources.manifest_path),
                    line=sources.manifest_lines.line("role_launch_meta", alias),
                    found=f"role_launch_meta[{alias!r}] present",
                    expected="<no entry>",
                    message=(
                        f"{alias!r} is an alias on {host!r}'s process (server_mode.{host}."
                        f"shared_with): it launches no server of its own, so it must carry "
                        f"no role_launch_meta entry"
                    ),
                    fix=f"delete role_launch_meta.{alias}",
                )
            )

    for host in sorted(d.hosts):
        if host not in meta:
            findings.append(
                Finding(
                    surface="launch_manifest.role_launch_meta",
                    role=host,
                    file=str(sources.manifest_path),
                    line=sources.manifest_lines.line("role_launch_meta"),
                    found="<no entry>",
                    expected=f"role_launch_meta.{host} with tier + mode",
                    message=(
                        f"{host!r} hosts aliases {d.host_to_aliases[host]} so it MUST launch, "
                        f"but has no role_launch_meta entry"
                    ),
                )
            )

    # The 2026-09-22 message, recomputed: a launching role needs NUMA wiring.
    for role, cfg in sorted(meta.items()):
        if cfg.get("no_numa") or cfg.get("launcher_only"):
            continue
        if role in numa:
            continue
        is_alias = role in d.alias_to_host
        findings.append(
            Finding(
                surface="launch_manifest.role_launch_meta x stack_topology.numa_config",
                role=role,
                file=str(sources.manifest_path),
                line=sources.manifest_lines.line("role_launch_meta", role),
                found=f"role_launch_meta[{role!r}] no_numa=False, numa_config[{role!r}] absent",
                expected=(
                    "<no role_launch_meta entry> (it is an alias)"
                    if is_alias
                    else f"numa_config.{role} in stack_topology.yaml, or no_numa: true"
                ),
                message=(
                    f"ROLE_LAUNCH_META[{role!r}] has no_numa=False but no NUMA_CONFIG entry"
                    + (
                        f" -- and {role!r} is an alias on {d.alias_to_host[role]!r}, so the "
                        f"entry itself is the defect"
                        if is_alias
                        else ""
                    )
                ),
                fix=f"delete role_launch_meta.{role}" if is_alias else None,
            )
        )

    # shared_with_first_n is derived; a declaration here is a restatement by definition.
    for role, cfg in sorted(meta.items()):
        if "shared_with_first_n" in cfg:
            findings.append(
                Finding(
                    surface="launch_manifest.role_launch_meta",
                    role=role,
                    file=str(sources.manifest_path),
                    line=sources.manifest_lines.line(
                        "role_launch_meta", role, "shared_with_first_n"
                    ),
                    found=f"shared_with_first_n: {cfg['shared_with_first_n']!r}",
                    expected=f"<derived from server_mode.{role}.shared_with>",
                    message=(
                        f"role_launch_meta[{role!r}] declares shared_with_first_n, which "
                        f"phase 2 DERIVES from server_mode.{role}.shared_with"
                    ),
                )
            )
    return findings


def check_numa_config(sources: Sources, d: Derivation) -> list[Finding]:
    """numa_config is keyed by HOST role only; an alias must have none."""
    findings: list[Finding] = []
    numa = sources.numa_config
    sm = sources.server_mode

    for alias in sorted(d.aliases):
        if alias in numa:
            host = d.alias_to_host[alias]
            findings.append(
                Finding(
                    surface="stack_topology.numa_config",
                    role=alias,
                    file=str(sources.topology_path),
                    line=sources.topology_lines.line("numa_config", alias),
                    found=f"numa_config[{alias!r}] present",
                    expected="<no entry>",
                    message=(
                        f"{alias!r} is an alias on {host!r}'s process: a role with no process "
                        f"must carry no NUMA wiring"
                    ),
                    fix=f"delete numa_config.{alias}",
                )
            )

    # Phantom fleet, recomputed: a declared fleet that resolves to no topology entry.
    for role, cfg in sorted(sm.items()):
        ports = cfg.get("numa_ports")
        count = cfg.get("numa_instances")
        if ports is None and count is None:
            continue
        host = d.alias_to_host.get(role)
        bound = role in numa or (host is not None and host in numa)
        if bound:
            continue
        findings.append(
            Finding(
                surface="master server_mode.numa_ports x stack_topology.numa_config",
                role=role,
                file=str(sources.master_path),
                line=sources.master_lines.line("server_mode", role, "numa_ports")
                or sources.master_lines.line("server_mode", role),
                found=f"numa_ports={ports!r} / numa_instances={count!r}",
                expected=(
                    f"<no declaration> (it is an alias on {host!r})"
                    if host
                    else f"numa_config.{role} in stack_topology.yaml"
                ),
                message=(
                    f"role {role!r} declares numa_ports={ports!r} / numa_instances={count!r} "
                    f"but the NUMA topology has NO entry for it, nor for its "
                    f"model_role/shared_with bindings. A fleet nothing launches is a phantom: "
                    f"either add the topology entry or delete the declaration"
                ),
            )
        )
    return findings


def check_role_models(sources: Sources, d: Derivation) -> list[Finding]:
    """`roles.<alias>.model` must name the artifact its HOST server actually loads."""
    findings: list[Finding] = []
    sm = sources.server_mode
    roles = sources.roles

    for alias in sorted(d.aliases):
        host = d.alias_to_host[alias]
        host_stems = server_artifact_stems(sm.get(host) or {})
        if not host_stems:
            continue
        role_cfg = roles.get(alias)
        if isinstance(role_cfg, dict) and role_cfg.get("model") is not None:
            stems = role_artifact_stems(role_cfg)
            if stems and not (stems & host_stems):
                findings.append(
                    Finding(
                        surface="master roles.<alias>.model",
                        role=alias,
                        file=str(sources.master_path),
                        line=sources.master_lines.line("roles", alias, "model"),
                        found=_describe(stems),
                        expected=_describe(host_stems),
                        message=(
                            f"Role-server conflict: role model metadata does not match the "
                            f"shared runtime server model (roles.{alias} vs "
                            f"server_mode.{host}, joined by shared_with)"
                        ),
                    )
                )
        # An alias keeping its own server_mode row restates the host's artifact there too.
        own = sm.get(alias)
        if isinstance(own, dict):
            own_stems = server_artifact_stems(own)
            if own_stems and not (own_stems & host_stems):
                findings.append(
                    Finding(
                        surface="master server_mode.<alias>.model",
                        role=alias,
                        file=str(sources.master_path),
                        line=sources.master_lines.line("server_mode", alias, "model"),
                        found=_describe(own_stems),
                        expected=_describe(host_stems),
                        message=(
                            f"alias {alias!r} keeps its own server_mode.model, which restates "
                            f"the artifact its host {host!r} loads"
                        ),
                    )
                )
            own_role = own.get("model_role")
            host_role = (sm.get(host) or {}).get("model_role")
            if own_role is not None and host_role is not None and own_role != host_role:
                findings.append(
                    Finding(
                        surface="master server_mode.<alias>.model_role",
                        role=alias,
                        file=str(sources.master_path),
                        line=sources.master_lines.line("server_mode", alias, "model_role"),
                        found=str(own_role),
                        expected=str(host_role),
                        message=(
                            f"alias {alias!r} points model_role at a different catalogue row "
                            f"than its host {host!r}"
                        ),
                    )
                )

    for host in sorted(d.hosts):
        host_stems = server_artifact_stems(sm.get(host) or {})
        role_cfg = roles.get(host)
        if not host_stems or not isinstance(role_cfg, dict):
            continue
        stems = role_artifact_stems(role_cfg)
        if stems and not (stems & host_stems):
            findings.append(
                Finding(
                    surface="master roles.<host>.model",
                    role=host,
                    file=str(sources.master_path),
                    line=sources.master_lines.line("roles", host, "model"),
                    found=_describe(stems),
                    expected=_describe(host_stems),
                    message=(
                        f"Role-server conflict: roles.{host}.model does not match "
                        f"server_mode.{host}, the server it launches"
                    ),
                )
            )
    return findings


CHECKS = (
    check_alias_of_coherence,
    check_port_map,
    check_role_launch_meta,
    check_numa_config,
    check_role_models,
)


def check_all(sources: Sources) -> list[Finding]:
    d = derive(sources)
    findings: list[Finding] = []
    for check in CHECKS:
        findings.extend(check(sources, d))
    return findings


# ---------------------------------------------------------------------------
# --fix : regenerate the surfaces that are safely derivable
# ---------------------------------------------------------------------------


def _block_span(lines: list[str], key_line_idx: int) -> tuple[int, int]:
    """[start, end) line indices of a mapping entry, trailing comments excluded.

    Leading comments belong to the FOLLOWING key, so the run of comment/blank
    lines that immediately precedes the next same-indent key is left in place.
    """
    indent = len(lines[key_line_idx]) - len(lines[key_line_idx].lstrip(" "))
    end = len(lines)
    for idx in range(key_line_idx + 1, len(lines)):
        line = lines[idx]
        if not line.strip():
            continue
        cur = len(line) - len(line.lstrip(" "))
        if cur <= indent and not line.lstrip().startswith("#"):
            end = idx
            break
    # walk back over the comment/blank run that introduces the next key
    while end - 1 > key_line_idx:
        prev = lines[end - 1]
        if prev.strip() and not prev.lstrip().startswith("#"):
            break
        end -= 1
    return key_line_idx, end


def _find_key_line(lines: list[str], section: str, key: str) -> int | None:
    in_section = False
    section_re = re.compile(rf"^{re.escape(section)}:\s*(#.*)?$")
    key_re = re.compile(rf"^(\s+){re.escape(key)}\s*:")
    for idx, line in enumerate(lines):
        if section_re.match(line):
            in_section = True
            continue
        if in_section:
            if line.strip() and not line.startswith((" ", "\t")) and not line.lstrip().startswith("#"):
                return None
            m = key_re.match(line)
            if m and len(m.group(1)) > 0:
                return idx
    return None


def apply_fixes(sources: Sources, findings: list[Finding]) -> tuple[list[str], list[Finding]]:
    """Rewrite the fixable surfaces. Never touches either model_registry.yaml."""
    applied: list[str] = []
    skipped: list[Finding] = []
    by_file: dict[str, list[Finding]] = {}
    for f in findings:
        if not f.fixable:
            skipped.append(f)
            continue
        if Path(f.file).name not in FIXABLE_FILES:
            skipped.append(f)
            continue
        by_file.setdefault(f.file, []).append(f)

    for file_path, items in by_file.items():
        path = Path(file_path)
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        deletions: list[tuple[int, int, str]] = []
        for f in items:
            action, _, arg = f.fix.partition(" ")
            if action == "delete":
                section, _, key = arg.partition(".")
                idx = _find_key_line(lines, section, key)
                if idx is None:
                    skipped.append(f)
                    continue
                start, end = _block_span(lines, idx)
                deletions.append((start, end, f.fix))
            else:  # port_map.<role>=<port>
                target, _, value = f.fix.partition("=")
                section, _, key = target.partition(".")
                idx = _find_key_line(lines, section, key)
                if idx is None:
                    # missing entry -> insert right after the section header
                    hdr = next(
                        (i for i, ln in enumerate(lines) if ln.startswith(f"{section}:")), None
                    )
                    if hdr is None:
                        skipped.append(f)
                        continue
                    lines.insert(hdr + 1, f"  {key}: {value}\n")
                    applied.append(f"{path.name}: inserted {section}.{key}: {value}")
                    continue
                lines[idx] = re.sub(
                    rf"^(\s+{re.escape(key)}\s*:\s*)\S+", rf"\g<1>{value}", lines[idx]
                )
                applied.append(f"{path.name}: set {section}.{key} = {value}")
        for start, end, label in sorted(deletions, key=lambda t: -t[0]):
            del lines[start:end]
            applied.append(f"{path.name}: {label} (lines {start + 1}-{end})")
        path.write_text("".join(lines), encoding="utf-8")
    return applied, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--master-registry", type=Path, default=DEFAULT_MASTER_REGISTRY)
    parser.add_argument("--launch-manifest", type=Path, default=DEFAULT_LAUNCH_MANIFEST)
    parser.add_argument("--stack-topology", type=Path, default=DEFAULT_STACK_TOPOLOGY)
    parser.add_argument("--check", action="store_true", help="report only (the default)")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="rewrite the derivable surfaces in launch_manifest.yaml / stack_topology.yaml",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        sources = load_sources(
            args.master_registry, args.launch_manifest, args.stack_topology
        )
        findings = check_all(sources)
    except SourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    applied: list[str] = []
    if args.fix:
        applied, _ = apply_fixes(sources, findings)
        sources = load_sources(
            args.master_registry, args.launch_manifest, args.stack_topology
        )
        findings = check_all(sources)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not findings,
                    "applied": applied,
                    "findings": [asdict(f) | {"fixable": f.fixable} for f in findings],
                },
                indent=2,
            )
        )
        return 1 if findings else 0

    for line in applied:
        print(f"FIXED  {line}")
    if not findings:
        print(
            "shared_with derivations: OK — port_map, role_launch_meta, numa_config and "
            "roles.<alias>.model all agree with server_mode.*.shared_with"
        )
        return 0
    print(
        f"shared_with derivations: {len(findings)} disagreement(s) with "
        f"server_mode.*.shared_with\n"
    )
    for f in findings:
        print(f.render())
        print()
    fixable = sum(1 for f in findings if f.fixable)
    print(
        f"{fixable} machine-fixable (--fix), {len(findings) - fixable} manual "
        f"(registry surfaces are the SOURCE; this tool never rewrites them)"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
