"""Offline admission fixtures. /proc mocks are policy tests, never Node proof.

All seams are import-time unittest patches; no production CLI/env test switch.
The separate granted real-Node fixture is required before owner acceptance.
"""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = load("gitnexus-guard")
W = load("gitnexus-writer")
UID = os.getuid()
OWNER = 31234
NODE_PID = 31235
CPUS = frozenset(range(24, 48))
CURRENT = {"owner": OWNER, "start": 100, "region": "q1", "cpus": CPUS, "grant": "digest"}


def payload(q, regions):
    return json.dumps({"schema_version": 1, "pid": OWNER, "role": "build", "region": q,
                       "regions": list(regions), "request_tag": "filler-case", "started_at": time.time() - 1}).encode()


class Claims(unittest.TestCase):
    def setUp(self):
        self.regions = ("q1",)
        self.bad_payload = False

        def identity(path):
            role, q = Path(path).name.split(".")[1:3]
            inode = 100 + (0 if role == "GLOBAL" else 4) + int(q[1])
            return (os.makedev(1, 2), inode), payload(q, ("q2",) if self.bad_payload else self.regions)

        def owners():
            return {(1, 2, 100 + offset + int(q[1])): {OWNER}
                    for offset in (0, 4) for q in self.regions}

        self.patches = [
            patch.object(G, "safe_root", return_value=(1, 2)),
            patch.object(G, "ancestors", return_value={OWNER: (100, UID)}),
            patch.object(G, "proc_stat", return_value=(1, 100, UID)),
            patch.object(G, "proc_lock_owners", side_effect=owners),
            patch.object(G, "open_identity", side_effect=identity),
            patch.object(G, "grant_cpu_set", return_value=(CPUS, "digest")),
            patch.dict(os.environ, {"ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT": "1"}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_single_quarter_claim_admits(self):
        self.assertEqual(G.quarter_claim(), CURRENT)

    def test_payload_single_region_must_match(self):
        self.bad_payload = True
        with self.assertRaises(G.Refusal):
            G.quarter_claim()

    def test_two_quarter_claims_refuse(self):
        self.regions = ("q1", "q2")
        with self.assertRaises(G.Refusal):
            G.quarter_claim()

    def test_whole_host_remains_valid(self):
        self.regions = G.QUADRANTS
        self.assertEqual(G.admit_whole_host(), OWNER)

    def test_flock_missing_refuses(self):
        with patch.object(G, "proc_lock_owners", return_value={}):
            with self.assertRaises(G.Refusal):
                G.quarter_claim()

    def test_cli_quarter_needs_actual_node_parent(self):
        with patch.object(G, "claim", return_value=CURRENT), patch.object(G, "node_admission", side_effect=G.Refusal("not Node")):
            with self.assertRaises(G.Refusal):
                G.admit()


class NodePlacement(unittest.TestCase):
    def setUp(self):
        self.env = {key: "1" for key in ("UV_THREADPOOL_SIZE", "GITNEXUS_PARSE_CHUNK_CONCURRENCY",
                    "GITNEXUS_EMBEDDING_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        self.env.update(GITNEXUS_WORKER_POOL_SIZE="4", NODE_OPTIONS="--max-old-space-size=8192", GITNEXUS_LBUG_EXTENSION_INSTALL="never")
        self.argv = [G.NODE, "--v8-pool-size=1", str(G.HERE / "gitnexus-quarter.cjs"),
                     "app", str(G.GNX / "dist/cli/index.js"), "analyze", "--workers", "4"]
        self.tids = [Path(f"/proc/{NODE_PID}/task/{NODE_PID}"), Path(f"/proc/{NODE_PID}/task/{NODE_PID + 1}")]
        self.affinity = CPUS
        self.patches = [
            patch.object(G, "installed_source"),
            patch.object(G, "native_environment"),
            patch.object(G, "proc_stat", return_value=(OWNER, 101, UID)),
            patch.object(G, "proc_argv", side_effect=lambda pid: self.argv),
            patch.object(G.os.path, "samefile", return_value=True),
            patch.object(G.Path, "read_bytes", side_effect=lambda: "\0".join(f"{k}={v}" for k, v in self.env.items()).encode()),
            patch.object(G.Path, "iterdir", side_effect=lambda: self.tids),
            patch.object(G.os, "sched_getaffinity", side_effect=lambda tid: self.affinity),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_actual_node_quarter_admits(self):
        G.node_admission(CURRENT, NODE_PID)

    def test_unpinned_node_refuses(self):
        self.affinity = frozenset(range(192))
        with self.assertRaises(G.Refusal):
            G.node_admission(CURRENT, NODE_PID)

    def test_one_worker_thread_off_quarter_refuses(self):
        with patch.object(G.os, "sched_getaffinity", side_effect=[CPUS, frozenset({0})]):
            with self.assertRaises(G.Refusal):
                G.node_admission(CURRENT, NODE_PID)

    def test_oversized_uv_refuses(self):
        self.env["UV_THREADPOOL_SIZE"] = "192"
        with self.assertRaises(G.Refusal):
            G.node_admission(CURRENT, NODE_PID)

    def test_oversized_gnx_env_refuses(self):
        self.env["GITNEXUS_WORKER_POOL_SIZE"] = "192"
        with self.assertRaises(G.Refusal):
            G.node_admission(CURRENT, NODE_PID)

    def test_oversized_cli_override_refuses(self):
        self.argv[-1] = "192"
        with self.assertRaises(G.Refusal):
            G.node_admission(CURRENT, NODE_PID)

    def test_extension_installer_node_child_refuses(self):
        self.env["GITNEXUS_LBUG_EXTENSION_INSTALL"] = "auto"
        with self.assertRaises(G.Refusal):
            G.node_admission(CURRENT, NODE_PID)

    def test_too_many_actual_threads_refuses(self):
        self.tids *= 13
        with self.assertRaises(G.Refusal):
            G.node_admission(CURRENT, NODE_PID)

    def test_actual_non_node_executable_refuses(self):
        with patch.object(G.os.path, "samefile", return_value=False):
            with self.assertRaises(G.Refusal):
                G.node_admission(CURRENT, NODE_PID)

    def test_whole_host_does_not_acquire_quarter_restrictions(self):
        G.node_admission({"owner": OWNER, "region": None}, NODE_PID)


class CanonicalResource(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        grants = self.root / "compute-flags/grants"
        grants.mkdir(parents=True)
        self.grant = grants / "case.grant"
        self.grant.write_text("regions=24-47 start=2000-01-01T00:00Z end=2100-01-01T00:00Z\n")
        for item in (patch.object(G, "LOCK_ROOT", str(self.root)),
                     patch.dict(os.environ, {"FILLER_ID": "case", "FILLER_CPUS": "24-47"}),
                     patch.object(G, "proc_argv", return_value=["python", "region_lock_cli.py", "run", "--cpu-list", "24-47", "--role", "build", "--tag", "filler-case", "--", "job"])):
            item.start()
            self.addCleanup(item.stop)
        self.payload = json.loads(payload("q1", ("q1",)))

    def test_literal_resource_cpus_are_authority(self):
        self.assertEqual(G.grant_cpu_set(OWNER, self.payload, "q1")[0], CPUS)

    def test_real_grant_cpus_must_belong_to_locked_region(self):
        # Real fd-relative grant read and parser: all other attribution agrees
        # on q0, but the grant/env/claim argv specify q1's CPUs 24-47.
        self.payload["region"] = "q0"
        self.payload["regions"] = ["q0"]
        with self.assertRaisesRegex(G.Refusal, "held quarter's exact primary CPU set"):
            G.grant_cpu_set(OWNER, self.payload, "q0")

    def test_mixed_24_cpus_do_not_constitute_one_quarter(self):
        self.grant.write_text("regions=12-35 start=2000-01-01T00:00Z end=2100-01-01T00:00Z\n")
        with patch.dict(os.environ, {"FILLER_CPUS": "12-35"}):
            with self.assertRaisesRegex(G.Refusal, "held quarter's exact primary CPU set"):
                G.grant_cpu_set(OWNER, self.payload, "q1")

    def test_q_shorthand_refuses_instead_of_inventing_smt_map(self):
        self.grant.write_text("regions=q1 start=2000-01-01T00:00Z end=2100-01-01T00:00Z\n")
        with self.assertRaises(G.Refusal):
            G.grant_cpu_set(OWNER, self.payload, "q1")

    def test_grant_payload_cpu_mismatch_refuses(self):
        with patch.dict(os.environ, {"FILLER_CPUS": "0-23"}):
            with self.assertRaises(G.Refusal):
                G.grant_cpu_set(OWNER, self.payload, "q1")

    def test_expired_grant_refuses(self):
        self.grant.write_text("regions=24-47 start=2000-01-01T00:00Z end=2001-01-01T00:00Z\n")
        with self.assertRaises(G.Refusal):
            G.grant_cpu_set(OWNER, self.payload, "q1")

    def test_claim_tag_mismatch_refuses(self):
        self.payload["request_tag"] = "another-task"
        with self.assertRaises(G.Refusal):
            G.grant_cpu_set(OWNER, self.payload, "q1")


class SupervisorSafety(unittest.TestCase):
    def test_parent_death_signal_is_kill_then_parent_race_check(self):
        events = []
        fake_prctl = SimpleNamespace()
        def prctl(*args):
            events.append(("prctl", args))
            return 0
        # Function attributes mirror ctypes' argtypes/restype assignment.
        fake_prctl.prctl = prctl
        with patch.object(W.ctypes, "CDLL", return_value=fake_prctl), \
             patch.object(W.os, "getppid", return_value=OWNER), \
             patch.object(W.signal, "pthread_sigmask", side_effect=lambda *args: events.append(("mask", args))):
            W.parent_death_guard(OWNER, set())
        self.assertEqual(events[0], ("prctl", (1, W.signal.SIGKILL, 0, 0, 0)))
        self.assertEqual(events[1], ("mask", (W.signal.SIG_SETMASK, set())))

    def test_parent_fork_race_exits_before_unblocking(self):
        class Exited(BaseException):
            pass
        fake = SimpleNamespace(prctl=lambda *args: 0)
        with patch.object(W.ctypes, "CDLL", return_value=fake), \
             patch.object(W.os, "getppid", return_value=1), \
             patch.object(W.os, "_exit", side_effect=Exited) as exited, \
             patch.object(W.signal, "pthread_sigmask") as unblocked:
            with self.assertRaises(Exited):
                W.parent_death_guard(OWNER, set())
            exited.assert_called_once_with(64)
            unblocked.assert_not_called()

    def test_term_int_hup_reap_captured_child_and_restore_handlers(self):
        for sig in W.QUARTER_SIGNALS:
            with self.subTest(signal=sig):
                class Child:
                    pid = NODE_PID
                    returncode = None
                    terminated = False
                    def poll(self):
                        return self.returncode
                    def terminate(self):
                        self.terminated = True
                        self.returncode = -W.signal.SIGTERM
                    def wait(self, timeout=None):
                        return self.returncode
                child = Child()
                admission = SimpleNamespace(NODE=G.NODE, GNX=G.GNX,
                    installed_source=lambda: None, claim=lambda: CURRENT,
                    native_artifact=lambda: "/fixture/cap.so",
                    DB_BUFFER_BYTES=G.DB_BUFFER_BYTES, DB_MAX_BYTES=G.DB_MAX_BYTES,
                    node_admission=lambda *args: W.terminate_quarter(sig, None))
                before = {s: W.signal.getsignal(s) for s in W.QUARTER_SIGNALS}
                with patch.object(W.os, "sched_getaffinity", return_value=CPUS), \
                     patch.object(W.Path, "glob", return_value=[]), \
                     patch.object(W.subprocess, "Popen", return_value=child) as launched, \
                     patch.dict(os.environ, {}, clear=True):
                    with self.assertRaises(W.Termination):
                        W.quarter_run("app", [], 9, admission, CURRENT)
                self.assertTrue(child.terminated)
                self.assertIsNotNone(child.poll())
                self.assertTrue(callable(launched.call_args.kwargs["preexec_fn"]))
                self.assertEqual(before, {s: W.signal.getsignal(s) for s in W.QUARTER_SIGNALS})


class WriterArguments(unittest.TestCase):
    def test_oversized_workers_refuse_before_node(self):
        for argv in (["--workers", "25"], ["--workers=192"], ["--workers", "0"], ["--workers", "-1"]):
            with self.subTest(argv=argv), self.assertRaises(W.Refusal):
                W.quarter_arguments(argv, CPUS)

    def test_valid_workers_normalize_to_bounded_cli(self):
        self.assertEqual(W.quarter_arguments(["/repo", "--workers", "8", "--index-only"], CPUS),
                         ["/repo", "--index-only", "--workers", "4"])

    def test_secondary_workload_refuses(self):
        for argv in (["--embeddings"], ["--embedding-threads", "192"], ["--skills"], ["--"]):
            with self.subTest(argv=argv), self.assertRaises(W.Refusal):
                W.quarter_arguments(argv, CPUS)


if __name__ == "__main__":
    unittest.main()
