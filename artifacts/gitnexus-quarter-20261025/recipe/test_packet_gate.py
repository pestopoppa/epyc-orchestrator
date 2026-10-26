import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch


class PacketGate(unittest.TestCase):
    def test_high_stops_even_with_main_unknown_approval(self):
        self.assertEqual(self.run_gate("HIGH", True), 69)

    def test_unknown_needs_exact_main_caller_review(self):
        self.assertEqual(self.run_gate(None, False), 69)

    def test_reviewed_unknown_allows_only_guard_fixture(self):
        self.assertEqual(self.run_gate(None, True), 0)

    def run_gate(self, risk, approved):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            callers = root / "callers.txt"
            callers.write_text("exact static caller review")
            (root / "private-status.txt").write_text("stale private snapshot")
            for label in ("guard-admit", "guard-valid-payload", "writer-guard", "writer-main"):
                value = {"error": "symbol unavailable"} if risk is None else {"target": {"id": "fixture"}, "risk": risk}
                (root / (label + ".json")).write_text(json.dumps(value))
            approval = root / "approval.json"
            approval.write_text(json.dumps({"unknown_graph_fallback_approved": approved,
                "static_callers_sha256": hashlib.sha256(callers.read_bytes()).hexdigest()}))
            script = Path(__file__).with_name("graph-gate.py")
            with patch.object(sys, "argv", [str(script), str(root), str(approval), str(callers)]):
                with self.assertRaises(SystemExit) as caught:
                    runpy.run_path(str(script), run_name="__main__")
            result = json.loads((root / "GRAPH-GATE.json").read_text())
            if caught.exception.code == 0:
                self.assertEqual(result["status"], "ALLOW_PRIVATE_GUARD_FIXTURE_ONLY")
                self.assertIn("no USAGE validation or index", result["scope"])
            return caught.exception.code


if __name__ == "__main__":
    unittest.main()
