#!/usr/bin/python3
"""Fail-closed proof that GitNexus is inside a live region-lock owner tree."""
import json
import datetime
import hashlib
import math
import errno
import os
from pathlib import Path
import re
import shlex
import stat
import sys
import time

LOCK_ROOT = "/mnt/raid0/llm/tmp"
QUADRANTS = ("q0", "q1", "q2", "q3")
# Deliberately restricted to the primary-only route. This is REGION_CORE_RANGE
# from src/runtime/instance_topology.py at the reviewed base 49c1f749014d5290.
# SMT-containing, incomplete and cross-quarter resource sets are not admitted.
PRIMARY_QUARTER_CPUS = {
    "q0": frozenset(range(0, 24)), "q1": frozenset(range(24, 48)),
    "q2": frozenset(range(48, 72)), "q3": frozenset(range(72, 96)),
}
DENIED = 64
EXPECTED_ROOT = None
HERE = Path(__file__).absolute().parent
NODE = "/usr/local/bin/node"
GNX = Path("/usr/local/share/npm-global/lib/node_modules/gitnexus")
NATIVE_SHIM = HERE / "gitnexus-lbug-cap.so"
NATIVE_MANIFEST = HERE / "gitnexus-lbug-cap.manifest.json"
NATIVE_SOURCE_SHA = "4e64dcd9aa5ff1b2dff6b7dfb0cc9a79641e0333593c96792872fbc8c03ab29b"
NATIVE_ELF_SHA = "85b101251f07b052927443a994c69c23621f7c10159687145fa15b06decf7b49"
DB_BUFFER_BYTES = 512 * 1024 ** 2
DB_MAX_BYTES = 8 * 1024 ** 3
# Installed-source admission contract. An upgrade needs a new source audit;
# the existing whole-host route deliberately does not depend on these pins.
QUARTER_SOURCE = {
    "node_modules/@ladybugdb/core/lbugjs.node": NATIVE_ELF_SHA,
    "node_modules/@ladybugdb/core/lbug.d.ts": "faef724b45f7a333f4ea006369dd1e1830e72d8ef0dd67620d7e666443fb3996",
    "node_modules/@ladybugdb/core/lbug_native.js": "bf028ed690e7ad6c6f44cf01c6301eb402835de479335ed59d5eb337c2cebdf4",
    "dist/cli/analyze.js": "93792661dbe20190e0a684c321cbe11b203bafff20174d02f51513096493d02e",
    "dist/cli/index.js": "c4a575f85e7c40e707f4cb8bfbe8fcded4ffeefe6e1da1241545fad6c780078d",
    "dist/core/ingestion/workers/worker-pool.js": "0b74033466b07b211112c6fcd2e7df13fb3170409e46901eef88275d3faaf5fa",
    "node_modules/@ladybugdb/core/connection.js": "327c854139a89c9b0a3435026e5c253720e6d12ab97f49e6fbd5432a913dc54d",
    "node_modules/@ladybugdb/core/index.mjs": "b852efec8849de6dd2c9a7de22a337899547e20551d8d26a441fe845aea6b74c",
    "node_modules/@ladybugdb/core/index.js": "bee3af181fd739817df7ae0a41ad6b6c3be897d549450fec3bcd952d504bc8c7",
    "node_modules/@ladybugdb/core/database.js": "4c99c221f87b34ffa9efd87bcc70d1bcd14ecd4e8256e0ac8bff8c80485dcf3c",
    "node_modules/@ladybugdb/core/query_result.js": "efa7d34422a75980473f73112477121a5be11d7ee264cae0c4115fb217c9e345",
    "dist/core/ingestion/workers/parse-worker.js": "2ffd8a935e7d563ec819c497f4fdc23be049fee3785727a56e4fd2fa0d112e87",
    "dist/core/ingestion/pipeline-phases/parse-impl.js": "f2454cec7daac0f9d63fc81afc73ed6919f40b363f16b75bc096455eb2d29068",
    "dist/core/ingestion/pipeline.js": "6865799a39299184f0abee9f45c126cb59f507e43a280234230a2ac7042d3958",
    "dist/core/run-analyze.js": "d65195c8323e47dcdf1b7138c663423a198090b46011ceee11f4f3e06d051be7",
    "dist/core/lbug/lbug-config.js": "702d54600c7447d291fe358babbae6ac5ab3059b00ac5ef9c81bb06069765a36",
    "dist/core/lbug/pool-adapter.js": "65468b24968dcb940afebe531e9bcb3e26f4e97d9dda5f53878df06c457f117c",
    "dist/core/lbug/extension-loader.js": "08ea49e2ab763a596ab1a28afb48dd99224839d30dfe767c854ccf09b5de347e",
}

class Refusal(Exception):
    pass

def deny(message):
    raise Refusal(message)

def proc_stat(pid):
    with open(f"/proc/{pid}/stat", "r", encoding="ascii") as f:
        raw = f.read()
    end = raw.rfind(")")
    if end < 0:
        deny("malformed proc stat")
    fields = raw[end + 2:].split()  # starts at field 3
    if len(fields) < 20:
        deny("short proc stat")
    ppid = int(fields[1])
    starttime = int(fields[19])  # field 22
    with open(f"/proc/{pid}/status", "r", encoding="ascii") as f:
        status = f.read()
    uid_line = next((line for line in status.splitlines() if line.startswith("Uid:")), None)
    if uid_line is None:
        deny("missing process uid")
    uid = int(uid_line.split()[1])
    return ppid, starttime, uid

def ancestors():
    result = {}
    pid = os.getpid()
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        ppid, start, uid = proc_stat(pid)
        result[pid] = (start, uid)
        if ppid == pid:
            break
        pid = ppid
    return result

def proc_lock_owners():
    owners = {}
    with open("/proc/locks", "r", encoding="ascii") as f:
        for line in f:
            fields = line.split()
            if not fields or "->" in fields:
                continue  # a waiter is not an owner
            if len(fields) < 8 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
                continue
            if not fields[4].isdigit():
                continue
            major, minor, inode = fields[5].split(":")
            key = (int(major, 16), int(minor, 16), int(inode, 10))
            owners.setdefault(key, set()).add(int(fields[4]))
    return owners

def safe_root():
    for name in ("ORCHESTRATOR_TMP_DIR", "ORCHESTRATOR_PATHS_TMP_DIR", "TMP_DIR"):
        value = os.environ.get(name)
        if value and value != LOCK_ROOT:
            deny(f"noncanonical {name}")
    root_fd, root_stat = open_fixed_root()
    os.close(root_fd)
    return (root_stat.st_dev, root_stat.st_ino)

def open_fixed_root():
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open("/", flags)
    try:
        for part in LOCK_ROOT.strip("/").split("/"):
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd, os.fstat(fd)
    except BaseException:
        os.close(fd)
        raise

def open_identity(path):
    if os.path.dirname(path) != LOCK_ROOT:
        deny("lock path escaped fixed root")
    root_fd, root_stat = open_fixed_root()
    try:
        if EXPECTED_ROOT is None or (root_stat.st_dev, root_stat.st_ino) != EXPECTED_ROOT:
            deny("fixed lock-root identity changed")
        name = os.path.basename(path)
        before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            deny("lock path is not a regular nonsymlink")
        fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                deny("opened lock is not regular")
            if opened.st_nlink != 1:
                deny("opened lock is not single-link")
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                deny("lock path changed during open")
            if opened.st_uid != os.getuid():
                deny("lock file uid differs")
            data = os.read(fd, 65537)
            if len(data) > 65536:
                deny("oversized lock payload")
            after = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            final = os.fstat(fd)
            if not stat.S_ISREG(final.st_mode) or final.st_uid != os.getuid() or final.st_nlink != 1:
                deny("lock type or owner changed during read")
            if (after.st_dev, after.st_ino) != (final.st_dev, final.st_ino):
                deny("lock path changed during read")
            check_fd, check_stat = open_fixed_root()
            os.close(check_fd)
            if (check_stat.st_dev, check_stat.st_ino) != EXPECTED_ROOT:
                deny("fixed lock-root path changed during admission")
            return (final.st_dev, final.st_ino), data
        finally:
            os.close(fd)
    finally:
        os.close(root_fd)

def dev_inode(identity):
    dev, ino = identity
    return (os.major(dev), os.minor(dev), ino)

def valid_payload(data, owner, expected_region, regions=QUADRANTS):
    try:
        p = json.loads(data.decode("utf-8"))
    except Exception:
        deny("invalid build lock payload")
    if not isinstance(p, dict):
        deny("invalid build lock payload object")
    if (type(p.get("schema_version")) is not int or
        type(p.get("pid")) is not int):
        deny("build lock attribution field types mismatch")
    started = p.get("started_at")
    if (p.get("schema_version") != 1 or p.get("pid") != owner or
        p.get("role") != "build" or p.get("region") != expected_region or
        p.get("regions") != list(regions) or
        not isinstance(p.get("request_tag"), str) or not p["request_tag"].strip() or
        isinstance(started, bool) or not isinstance(started, (int, float)) or
        not math.isfinite(started) or started <= 0 or started > time.time() + 60):
        deny("build lock attribution mismatch")

def admit_whole_host():
    global EXPECTED_ROOT
    EXPECTED_ROOT = safe_root()
    if os.environ.get("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "").strip().lower() not in {"1", "true", "yes", "on"}:
        deny("cross-role global region locks are not enabled")
    anc = ancestors()
    current_uid = os.getuid()
    kernel = proc_lock_owners()
    identities = []
    for role in ("GLOBAL", "build"):
        for q in QUADRANTS:
            path = os.path.join(LOCK_ROOT, f"cpu_region.{role}.{q}.lock")
            ident, payload = open_identity(path)
            maj, min_, ino = dev_inode(ident)
            owners = kernel.get((maj, min_, ino), set())
            candidates = owners.intersection(anc)
            if len(candidates) != 1:
                deny(f"{role}/{q} lacks one granted ancestor write flock")
            owner = next(iter(candidates))
            start, uid = anc[owner]
            if uid != current_uid:
                deny("claim owner uid differs")
            if role == "build":
                valid_payload(payload, owner, q)
            identities.append((path, ident, owner, start))
    if len({entry[1] for entry in identities}) != 8:
        deny("CPU claims do not have eight distinct inode identities")
    owner_set = {entry[2] for entry in identities}
    if len(owner_set) != 1:
        deny("locks do not share one ancestor owner")
    owner = next(iter(owner_set))
    if owner == os.getpid():
        deny("guard itself cannot be the claim owner")
    # Re-open/re-read every pathname and revalidate process identity and kernel grants.
    ppid_start_uid = proc_stat(owner)
    anc2 = ancestors()
    if owner not in anc2 or ppid_start_uid[1] != anc[owner][0] or anc2[owner] != anc[owner] or ppid_start_uid[2] != current_uid:
        deny("claim owner changed")
    kernel2 = proc_lock_owners()
    for path, prior, expected_owner, start in identities:
        now, payload = open_identity(path)
        if now != prior:
            deny("lock identity changed during admission")
        maj, min_, ino = dev_inode(now)
        if expected_owner not in kernel2.get((maj, min_, ino), set()):
            deny("write flock disappeared during admission")
        if "cpu_region.build." in path:
            q = path.rsplit(".", 2)[-2]
            valid_payload(payload, owner, q)
    return owner


def cpu_set(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value):
        deny("invalid literal CPU list")
    result = set()
    for part in value.split(","):
        ends = [int(v) for v in part.split("-")]
        low, high = ends[0], ends[-1]
        if low > high or high > 65535:
            deny("invalid CPU interval")
        result.update(range(low, high + 1))
    return frozenset(result)


def option(argv, key):
    # Claimant options end at --; child argv is never attribution evidence.
    values = []
    for i, item in enumerate(argv):
        if item == "--":
            break
        if item == key:
            if i + 1 >= len(argv):
                deny("missing option value")
            values.append(argv[i + 1])
        elif item.startswith(key + "="):
            values.append(item[len(key) + 1:])
    if len(values) != 1:
        deny(f"expected one {key}")
    return values[0]


def proc_argv(pid):
    return Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")


def grant_cpu_set(owner, payload, expected_region):
    if payload.get("region") != expected_region or payload.get("regions") != [expected_region]:
        deny("resource attribution does not name the held quarter")
    filler_id = os.environ.get("FILLER_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", filler_id):
        deny("quarter admission needs a canonical filler id")
    if payload.get("request_tag") != "filler-" + filler_id:
        deny("build payload is not this filler")
    # Descriptor-relative, nonsymlink traversal to the canonical resource grant.
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open("/", flags | os.O_DIRECTORY)
    try:
        for part in (LOCK_ROOT + "/compute-flags/grants").strip("/").split("/"):
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
        name = filler_id + ".grant"
        grant_fd = os.open(name, flags | os.O_NONBLOCK, dir_fd=fd)
        try:
            before = os.fstat(grant_fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid():
                deny("unsafe canonical grant")
            raw = os.read(grant_fd, 4097)
            after = os.fstat(grant_fd)
            current = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if (len(raw) > 4096 or
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or
                (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)):
                deny("canonical grant changed during read")
        finally:
            os.close(grant_fd)
    finally:
        os.close(fd)
    words = shlex.split(raw.decode("ascii"))
    items = [word.split("=", 1) for word in words]
    if any(len(item) != 2 for item in items) or len({item[0] for item in items}) != len(items):
        deny("invalid canonical grant")
    grant = dict(items)
    if set(grant) != {"regions", "start", "end"}:
        deny("unexpected canonical grant fields")
    def epoch(value):
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            deny("grant timestamp lacks timezone")
        return parsed.timestamp()
    now = time.time()
    if not epoch(grant["start"]) <= now < epoch(grant["end"]):
        deny("canonical grant is not live")
    # Require the literal CPU resource payload, not qN's legacy SMT expansion.
    cpus = cpu_set(grant["regions"])
    if (cpus != PRIMARY_QUARTER_CPUS.get(expected_region) or
        cpu_set(os.environ.get("FILLER_CPUS", "")) != cpus):
        deny("canonical CPU resource is not the held quarter's exact primary CPU set")
    argv = proc_argv(owner)
    if (cpu_set(option(argv, "--cpu-list")) != cpus or
        option(argv, "--role") != "build" or option(argv, "--tag") != "filler-" + filler_id):
        deny("canonical resource and claim owner disagree")
    return cpus, hashlib.sha256(raw).hexdigest()


def quarter_claim():
    global EXPECTED_ROOT
    EXPECTED_ROOT = safe_root()
    if os.environ.get("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "").strip().lower() not in {"1", "true", "yes", "on"}:
        deny("cross-role global region locks are not enabled")
    anc, kernel = ancestors(), proc_lock_owners()
    held = []
    for role in ("GLOBAL", "build"):
        for q in QUADRANTS:
            path = os.path.join(LOCK_ROOT, f"cpu_region.{role}.{q}.lock")
            ident, payload = open_identity(path)
            owners = kernel.get(dev_inode(ident), set()).intersection(anc)
            if len(owners) > 1:
                deny("ambiguous claim ancestor")
            if owners:
                held.append((role, q, path, ident, next(iter(owners)), payload))
    if len(held) != 2 or {h[0] for h in held} != {"GLOBAL", "build"} or len({h[1] for h in held}) != 1:
        deny("quarter needs exactly one GLOBAL/build pair")
    owners = {h[4] for h in held}
    if len(owners) != 1:
        deny("quarter locks do not share one ancestor")
    owner = owners.pop()
    if owner == os.getpid() or anc[owner][1] != os.getuid() or held[0][3] == held[1][3]:
        deny("invalid quarter owner or inode identity")
    q = held[0][1]
    build = next(h for h in held if h[0] == "build")
    valid_payload(build[5], owner, q, (q,))
    cpus, grant_digest = grant_cpu_set(owner, json.loads(build[5]), q)
    # Retain the whole-host guard's second kernel/path/process observation.
    if ancestors().get(owner) != anc[owner] or proc_stat(owner)[1:] != anc[owner]:
        deny("quarter claim owner changed")
    kernel = proc_lock_owners()
    for role, region, path, prior, expected_owner, payload in held:
        ident, current = open_identity(path)
        if ident != prior or expected_owner not in kernel.get(dev_inode(ident), set()):
            deny("quarter write flock changed")
        if role == "build":
            valid_payload(current, owner, q, (q,))
            if grant_cpu_set(owner, json.loads(current), q) != (cpus, grant_digest):
                deny("quarter grant changed")
    return {"owner": owner, "start": anc[owner][0], "region": q, "cpus": cpus, "grant": grant_digest}


def claim():
    try:
        return {"owner": admit_whole_host(), "region": None}
    except (Refusal, OSError):
        return quarter_claim()


def installed_source():
    for relative, digest in QUARTER_SOURCE.items():
        if hashlib.sha256((GNX / relative).read_bytes()).hexdigest() != digest:
            deny(f"quarter source contract changed: {relative}")


def native_artifact():
    # This manifest is emitted only by the reviewed grant-only private build.
    # No compiler runs at admission, and the whole-host path never calls this.
    for path in (NATIVE_SHIM, NATIVE_MANIFEST, HERE / "gitnexus-lbug-cap.c"):
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_nlink != 1:
            deny("unsafe quarter native cap artifact")
    if hashlib.sha256((HERE / "gitnexus-lbug-cap.c").read_bytes()).hexdigest() != NATIVE_SOURCE_SHA:
        deny("quarter native cap source changed")
    manifest = json.loads(NATIVE_MANIFEST.read_text())
    if (not isinstance(manifest, dict) or set(manifest) != {
            "schema_version", "source_sha256", "native_sha256", "shim_sha256", "max_num_threads"} or
            type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or
            type(manifest["max_num_threads"]) is not int or manifest["max_num_threads"] != 1 or
            manifest["source_sha256"] != NATIVE_SOURCE_SHA or manifest["native_sha256"] != NATIVE_ELF_SHA or
            not re.fullmatch(r"[a-f0-9]{64}", manifest["shim_sha256"]) or
            hashlib.sha256(NATIVE_SHIM.read_bytes()).hexdigest() != manifest["shim_sha256"]):
        deny("quarter native cap build manifest mismatch")
    return str(NATIVE_SHIM)


def native_environment(pid, env, require_mapped=True):
    shim = native_artifact()
    if env.get("LD_PRELOAD") != shim:
        deny("quarter Node lacks its exact native constructor cap")
    if (env.get("GITNEXUS_LBUG_BUFFER_BYTES") != str(DB_BUFFER_BYTES) or
            env.get("GITNEXUS_LBUG_MAX_DB_SIZE") != str(DB_MAX_BYTES)):
        deny("quarter Node lacks its explicit database memory/file caps")
    metadata = NATIVE_SHIM.stat()
    matched = False
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or fields[5] != shim:
            continue
        major, minor = (int(value, 16) for value in fields[3].split(":"))
        if (os.makedev(major, minor), int(fields[4])) != (metadata.st_dev, metadata.st_ino):
            deny("quarter mapped native cap differs from checked artifact")
        matched = True
    if not matched and require_mapped:
        deny("quarter native constructor cap is not mapped")


def node_admission(current, pid, require_native_mapping=False):
    if current["region"] is None:
        return
    installed_source()
    start = proc_stat(pid)
    if start[2] != os.getuid() or not os.path.samefile(f"/proc/{pid}/exe", NODE):
        deny("quarter child is not the actual Node executable")
    argv = proc_argv(pid)
    expected = ["--v8-pool-size=1", str(HERE / "gitnexus-quarter.cjs"), "app", str(GNX / "dist/cli/index.js"), "analyze"]
    if argv[1:6] != expected:
        deny("quarter Node did not enter the fixed bootstrap")
    env = dict(part.split("=", 1) for part in Path(f"/proc/{pid}/environ").read_bytes().decode().rstrip("\0").split("\0") if "=" in part)
    # The supervisor can sample immediately after exec, while ld.so is still
    # mapping LD_PRELOAD. Bootstrap admission runs from JS and requires the
    # mapping before any native DB/CLI import can begin.
    native_environment(pid, env, require_mapped=require_native_mapping)
    for key in ("UV_THREADPOOL_SIZE", "GITNEXUS_PARSE_CHUNK_CONCURRENCY", "GITNEXUS_EMBEDDING_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if env.get(key) != "1":
            deny(f"quarter Node lacks {key}=1")
    if env.get("GITNEXUS_WORKER_POOL_SIZE") != "4" or not re.fullmatch(r"--max-old-space-size=[1-9][0-9]*", env.get("NODE_OPTIONS", "")):
        deny("quarter Node has uncapped worker or heap-respawn environment")
    if env.get("GITNEXUS_LBUG_EXTENSION_INSTALL") != "never":
        deny("quarter Node may spawn an unadmitted extension installer")
    if option(argv[6:], "--workers") != "4":
        deny("quarter CLI worker override differs")
    for item in argv[6:]:
        if item.startswith(("--embedding", "--skills")):
            deny("quarter analyze enables an unreviewed secondary workload")
    tids = list(Path(f"/proc/{pid}/task").iterdir())
    if not tids or len(tids) > len(current["cpus"]):
        deny("quarter Node exceeds the thread cap")
    for entry in tids:
        try:
            affinity = os.sched_getaffinity(int(entry.name))
        except OSError as exc:
            if exc.errno != errno.ESRCH:
                raise
            try:
                entry.stat()
            except FileNotFoundError:
                continue
            deny("Node task remains present after affinity ESRCH")
        if frozenset(affinity) != current["cpus"]:
            deny("actual Node thread affinity differs from granted CPUs")
    if proc_stat(pid) != start:
        deny("Node identity changed during admission")


def admit():
    current = claim()
    if current["region"] is not None:
        # Only the Node bootstrap calls this production CLI for a quarter.
        # A Python/shell ancestor's affinity can never authorize Node work.
        node_admission(current, os.getppid(), require_native_mapping=True)
        print(json.dumps({"gitnexus_admission": "quarter", "node_pid": os.getppid(),
                          "claim_owner": current["owner"], "region": current["region"],
                          "cpus_observed": sorted(current["cpus"]), "parse_workers": 4,
                          "uv_threads": 1, "v8_pool": 1, "db_query_cap_policy": 1,
                          "db_init_cap_policy": 1, "db_buffer_bytes": DB_BUFFER_BYTES,
                          "db_max_bytes": DB_MAX_BYTES}), file=sys.stderr)
    return current["owner"]

def main():
    try:
        if len(sys.argv) != 3 or sys.argv[1] != "--repo" or sys.argv[2] not in {"root", "app"}:
            deny("usage: gitnexus-guard.py --repo root|app")
        admit()
        return 0
    except Exception as exc:
        print(f"gitnexus guard: denied: {exc}", file=sys.stderr)
        return DENIED

if __name__ == "__main__":
    raise SystemExit(main())
