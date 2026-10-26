"""SEC-EXEC-3/5: kernel-enforced containment for model-exec children (run_shell, run_python_code).

Behind the ``exec_containment`` feature flag (default OFF). When on, a child is launched through
the standalone shim (``_fence_kernel_shim``) which applies, before exec and inherited by every
descendant:

* seccomp (SEC-EXEC-3): ``socket()`` is denied outright (EACCES); only ``socketpair(AF_UNIX)``
  survives. io_uring (which can open sockets without ``socket()``), ptrace, process_vm_*,
  pidfd_getfd, unshare and setns are denied too, so a child can neither open nor borrow a
  socket. Landlock TCP bind/connect handling (ABI >= 4, no rules == all denied) is layered
  underneath, and Landlock abstract-unix-socket and signal scoping (ABI >= 6) keeps the child
  from signalling processes outside its own domain.
* Landlock filesystem read-deny (SEC-EXEC-5): the credential files listed in
  ``credential_paths`` (HF token files, git/gh/ssh/cloud/agent-CLI credentials) are unreadable
  and unwritable; everything else keeps the access it had. Other processes' ``/proc/<pid>``
  files are unreadable too -- without that, ``/proc/<api-pid>/environ`` would hand the child
  the very secrets SEC-EXEC-1 removed from its environment.

Fail closed. If the host cannot enforce (probe fails), or the shim cannot apply a restriction
(it exits 126), the call is REFUSED with a logged reason; it is never run unrestricted.

Accepted residuals:

* Pathname AF_UNIX connect to an existing socket needs a ``socket()`` call, so it is denied
  here; an fd inherited from the parent would bypass it, and ``subprocess`` closes fds.
* The fenced set is computed from the directory tree at spec-build time (cached 300 s). A file
  created later INSIDE a directory that contains a denied path has no rule and stays denied;
  a new sibling directory of such a container is unreadable until the cache refreshes.
* Descendants forked by the child cannot read their own ``/proc/<pid>`` (only the exec'd
  process's pid directory is granted); ``/proc/self`` inside a grandchild fails. Landlock
  cannot express "any pid I own".
* Same-uid UNIX-socket services reachable by an inherited fd are not covered (none are passed).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)

SHARED_HF_HOME = "/mnt/raid0/llm/cache/huggingface"
DENY_EXTRA_ENV = "ORCHESTRATOR_EXEC_DENY_EXTRA"  # os.pathsep list of additional denied paths
_SPEC_TTL_S = 300.0
_NEG_PROBE_TTL_S = 60.0


class ContainmentUnavailable(RuntimeError):
    """The host (or the shim) cannot enforce the requested containment; the call must be refused."""


@dataclass(frozen=True)
class Policy:
    no_network: bool = True
    deny_credentials: bool = True
    hide_foreign_proc: bool = True
    scope: bool = True


def current_policy() -> Policy | None:
    """The active policy: ``Policy()`` when the ``exec_containment`` flag is on, else None."""
    from src.repl_environment.exec_containment import flag_enabled

    return Policy() if flag_enabled("exec_containment") else None


# ── credential paths ─────────────────────────────────────────────────────────

_HOME_RELATIVE = (
    ".git-credentials", ".config/git/credentials", ".config/gh", ".netrc", ".ssh", ".aws",
    ".config/gcloud", ".azure", ".docker/config.json", ".kube", ".npmrc", ".pypirc",
    ".config/wandb", ".config/huggingface", ".huggingface", ".cache/huggingface/token",
    ".cache/huggingface/stored_tokens",
    # agent / assistant CLIs
    ".claude", ".claude.json", ".config/claude", ".codex", ".config/opencode",
    ".local/share/opencode", ".gemini", ".anthropic",
)


def _home_dirs(env: dict[str, str]) -> list[str]:
    homes: list[str] = []
    try:
        import pwd

        homes.append(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError):
        pass
    for h in (os.path.expanduser("~"), env.get("HOME", "")):
        if h and h not in homes:
            homes.append(h)
    return [h for h in homes if h and h != "/"]


def _hf_roots(env: dict[str, str], homes: Iterable[str]) -> set[str]:
    roots = {SHARED_HF_HOME}
    if env.get("HF_HOME"):
        roots.add(env["HF_HOME"])
    if env.get("XDG_CACHE_HOME"):
        roots.add(os.path.join(env["XDG_CACHE_HOME"], "huggingface"))
    for h in homes:
        roots.add(os.path.join(h, ".cache", "huggingface"))
    return roots


_TOKEN_WARN_EVERY_S = 600.0
_token_warned: dict[str, float] = {}


def reappeared_hf_token_files(env: dict[str, str] | None = None, homes: Iterable[str] | None = None) -> list[str]:
    """Existing ``token`` / ``stored_tokens`` files under any HF root (existence only, never read).

    SEC-EXEC-6: the operator revoked and deleted the shared-cache token. Containment denies these
    paths to model children, but a standing credential reappearing (a `huggingface-cli login`, a
    copied cache) is itself the finding, so it is reported loudly.
    """
    env = dict(os.environ) if env is None else env
    homes = list(homes) if homes is not None else _home_dirs(env)
    found = []
    for root in sorted(_hf_roots(env, homes)):
        for name in ("token", "stored_tokens"):
            path = os.path.join(root, name)
            if os.path.lexists(path):
                found.append(path)
    if env.get("HF_TOKEN_PATH") and os.path.lexists(env["HF_TOKEN_PATH"]):
        found.append(env["HF_TOKEN_PATH"])
    return found


def warn_if_hf_token_files(env: dict[str, str] | None = None, homes: Iterable[str] | None = None) -> list[str]:
    """Log an ERROR per reappeared token file (rate-limited per path); returns the paths."""
    found = reappeared_hf_token_files(env, homes)
    now = time.monotonic()
    for path in found:
        last = _token_warned.get(path)
        if last is None or now - last >= _TOKEN_WARN_EVERY_S:
            _token_warned[path] = now
            logger.error(
                "SEC-EXEC-6: HF TOKEN FILE PRESENT at %s -- the shared-cache token was revoked and "
                "deleted (2026-10-21). Find out who recreated it; revoke it on huggingface.co if it "
                "is not intended. Model-exec children are denied this path, but nextplaid mounts and "
                "download wrappers can still see it.", path,
            )
    return found


def credential_paths(env: dict[str, str] | None = None, homes: Iterable[str] | None = None) -> list[str]:
    """Absolute credential locations to deny (existing or not; sorted, de-duplicated)."""
    env = dict(os.environ) if env is None else env
    homes = list(homes) if homes is not None else _home_dirs(env)
    out: set[str] = set()
    hf_roots = _hf_roots(env, homes)
    for h in homes:
        for rel in _HOME_RELATIVE:
            out.add(os.path.join(h, rel))
    for root in hf_roots:
        out.add(os.path.join(root, "token"))
        out.add(os.path.join(root, "stored_tokens"))
    if env.get("HF_TOKEN_PATH"):
        out.add(env["HF_TOKEN_PATH"])
    try:
        from src.repl_environment.external_access import _get_project_root

        root = _get_project_root()
        out.update({os.path.join(root, ".env"), os.path.join(root, ".env.local")})
    except Exception:  # noqa: BLE001 - the project .env entries are best effort additions
        pass
    for extra in (env.get(DENY_EXTRA_ENV) or "").split(os.pathsep):
        if extra.strip():
            out.add(os.path.abspath(extra.strip()))
    return sorted(p for p in out if os.path.isabs(p))


# ── host capability probe ────────────────────────────────────────────────────

_lock = threading.Lock()
_caps_cache: dict[str, Any] = {}
_spec_cache: dict[tuple, dict[str, Any]] = {}


def reset_caches() -> None:
    with _lock:
        _caps_cache.clear()
        _spec_cache.clear()


def capabilities() -> dict[str, Any]:
    """{'abi', 'landlock', 'netdeny'} proven by throwaway shim children (cached per process)."""
    from src.repl_environment import fence_kernel

    with _lock:
        cached = _caps_cache.get("caps")
        if cached is not None:
            ok = cached["landlock"] and cached["netdeny"]
            if ok or time.monotonic() - cached["ts"] < _NEG_PROBE_TTL_S:
                return cached
    abi = fence_kernel.landlock_abi()
    caps = {
        "abi": abi,
        "landlock": bool(abi) and fence_kernel._probe_child(fence_kernel.LANDLOCK),
        "netdeny": fence_kernel._probe_child("netdeny"),
        "ts": time.monotonic(),
    }
    with _lock:
        _caps_cache["caps"] = caps
    return caps


# ── spec + wrap ──────────────────────────────────────────────────────────────


def _spec_path(policy: Policy, deny_paths: list[str], fence_roots: Any) -> str:
    from src.repl_environment import fence_kernel

    key = (policy, tuple(sorted(deny_paths)))
    with _lock:
        hit = _spec_cache.get(key)
        if (
            hit is not None
            and hit["roots"] is fence_roots
            and time.monotonic() - hit["ts"] < _SPEC_TTL_S
            and os.path.exists(hit["path"])
        ):
            return hit["path"]
    fenced = set(fence_kernel.with_mount_aliases(deny_paths, fence_kernel.read_mounts()))
    if fence_roots is not None:
        fenced |= fence_kernel.kernel_fenced_paths(fence_roots)
    special = {"/proc": fence_kernel.KIND_PROC} if policy.hide_foreign_proc else {}
    fs_enforced = bool(fenced or special)
    abi = fence_kernel.landlock_abi() or 1
    spec: dict[str, Any] = {
        "mode": fence_kernel.LANDLOCK,
        "fenced": sorted(fenced),
        "netdeny": bool(policy.no_network),
        "scope": bool(policy.scope),
    }
    if fs_enforced:
        spec["rules"] = fence_kernel.build_landlock_rules(fenced, special=special)
        spec["masks"] = fence_kernel.landlock_access_masks(abi)
    else:
        spec["rules"] = []
        spec["masks"] = {"handled": 0, "full": 0, "file": 0}
    path = fence_kernel._write_cached(json.dumps(spec, sort_keys=True), "spec-contain")
    with _lock:
        _spec_cache[key] = {"roots": fence_roots, "path": path, "ts": time.monotonic()}
    return path


def wrap(
    argv: list[str],
    *,
    policy: Policy,
    extra_allow: Iterable[str] = (),
    fence_roots: Any = None,
    deny_paths: Iterable[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """(argv, extra env) running ``argv`` under the policy; raises ``ContainmentUnavailable``.

    ``fence_roots`` merges an armed AP-54 eval fence into the same restriction.
    ``deny_paths`` overrides the computed credential set (tests).
    """
    from src.repl_environment import fence_kernel

    warn_if_hf_token_files()
    caps = capabilities()
    if not caps["landlock"]:
        raise ContainmentUnavailable(
            f"Landlock is not usable on this host (abi={caps['abi']}); refusing to run unrestricted"
        )
    if policy.no_network and not caps["netdeny"]:
        raise ContainmentUnavailable("seccomp network deny is not usable on this host; refusing to run with network")
    if deny_paths is None:
        deny = credential_paths() if policy.deny_credentials else []
    else:
        deny = [str(p) for p in deny_paths]
    try:
        spec = _spec_path(policy, deny, fence_roots)
    except OSError as exc:
        raise ContainmentUnavailable(f"could not build the containment spec: {exc}") from exc
    extra = [str(p) for p in extra_allow if p]
    with open(spec, encoding="utf-8") as handle:
        extra_rules = fence_kernel.safe_extra_landlock_rules(extra, json.load(handle))
    wrapped = [sys.executable, "-c", fence_kernel.shim_source(), spec, "--", *argv]
    return wrapped, fence_kernel.extra_grant_environment(extra_rules)


def describe_refusal(exc: BaseException) -> str:
    return f"EXEC CONTAINMENT: model-exec child not started — {exc}"


def shim_failure_note(returncode: int, stderr: str) -> str | None:
    """A visible reason when the shim itself refused to start the command (exit 126)."""
    if returncode == 126 and "kernel enforcement failed" in (stderr or ""):
        return (stderr or "").strip().splitlines()[-1][:300]
    return None
