"""Model-free artifact/mapping controls; never substitutes for native proof."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
spec = importlib.util.spec_from_file_location("native_guard", SCRIPTS / "gitnexus-guard.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


class NativeArtifact(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shim = self.root / "gitnexus-lbug-cap.so"
        self.source = self.root / "gitnexus-lbug-cap.c"
        self.manifest = self.root / "gitnexus-lbug-cap.manifest.json"
        self.shim.write_bytes(b"fake ELF for digest-only policy controls")
        self.source.write_bytes(b"fake source")
        self.record = {"schema_version": 1, "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
                       "native_sha256": G.NATIVE_ELF_SHA, "shim_sha256": hashlib.sha256(self.shim.read_bytes()).hexdigest(),
                       "max_num_threads": 1}
        self.write_manifest()
        for name, value in (("HERE", self.root), ("NATIVE_SHIM", self.shim),
                            ("NATIVE_MANIFEST", self.manifest), ("NATIVE_SOURCE_SHA", self.record["source_sha256"])):
            item = patch.object(G, name, value)
            item.start()
            self.addCleanup(item.stop)

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.record))

    def test_exact_artifact_admits(self):
        self.assertEqual(G.native_artifact(), str(self.shim))

    def test_changed_binary_refuses(self):
        self.shim.write_bytes(b"changed")
        with self.assertRaises(G.Refusal): G.native_artifact()

    def test_changed_native_pin_refuses(self):
        self.record["native_sha256"] = "0" * 64
        self.write_manifest()
        with self.assertRaises(G.Refusal): G.native_artifact()

    def test_changed_source_refuses(self):
        self.source.write_bytes(b"changed source")
        with self.assertRaises(G.Refusal): G.native_artifact()

    def test_oversized_or_boolean_native_cap_refuses(self):
        for value in (True, 24, 192):
            self.record["max_num_threads"] = value
            self.write_manifest()
            with self.assertRaises(G.Refusal): G.native_artifact()

    def test_symlink_refuses(self):
        target = self.root / "other.so"
        self.shim.rename(target)
        self.shim.symlink_to(target)
        with self.assertRaises(G.Refusal): G.native_artifact()

    def env(self):
        return {"LD_PRELOAD": str(self.shim), "GITNEXUS_LBUG_BUFFER_BYTES": str(G.DB_BUFFER_BYTES),
                "GITNEXUS_LBUG_MAX_DB_SIZE": str(G.DB_MAX_BYTES)}

    def maps(self, inode=None):
        st = self.shim.stat()
        return f"1000-2000 r-xp 0000 {os.major(st.st_dev):x}:{os.minor(st.st_dev):x} {inode or st.st_ino} {self.shim}\n"

    def test_exact_mapping_and_explicit_memory_admit(self):
        with patch.object(G.Path, "read_text", return_value=self.maps()), patch.object(G, "native_artifact", return_value=str(self.shim)):
            G.native_environment(4321, self.env())

    def test_missing_or_ambient_preload_refuses(self):
        for value in (None, "/other.so", f"{self.shim}:/other.so"):
            env = self.env()
            env["LD_PRELOAD"] = value
            with self.assertRaises(G.Refusal): G.native_environment(4321, env)

    def test_missing_or_zero_memory_refuses(self):
        for key in ("GITNEXUS_LBUG_BUFFER_BYTES", "GITNEXUS_LBUG_MAX_DB_SIZE"):
            env = self.env()
            env[key] = "0"
            with self.assertRaises(G.Refusal): G.native_environment(4321, env)

    def test_mapping_different_inode_refuses(self):
        with patch.object(G.Path, "read_text", return_value=self.maps(self.shim.stat().st_ino + 1)), patch.object(G, "native_artifact", return_value=str(self.shim)):
            with self.assertRaises(G.Refusal): G.native_environment(4321, self.env())

    def test_bootstrap_requires_actual_mapping(self):
        with patch.object(G.Path, "read_text", return_value=""), patch.object(G, "native_artifact", return_value=str(self.shim)):
            with self.assertRaises(G.Refusal): G.native_environment(4321, self.env())
            G.native_environment(4321, self.env(), require_mapped=False)


if __name__ == "__main__":
    unittest.main()
