import ast
from pathlib import Path
import pytest

class OriginalSource:
    def pytest_collection_modifyitems(self, items):
        module = items[0].module
        tree = ast.parse(Path("/mnt/raid0/llm/tmp/eval-tower-e5-codex3-20261101/old_autopilot_tui.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AutoPilotTUI")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "set_prompt")
        code = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        namespace = dict(module.autopilot_tui.__dict__)
        exec(compile(code, "old_autopilot_tui.py", "exec"), namespace)
        module.autopilot_tui.AutoPilotTUI.set_prompt = namespace["set_prompt"]

raise SystemExit(pytest.main(["-q", "tests/unit/test_eval_tower_e5_low_repairs.py::test_prompt_tap_writes_are_serialized"], plugins=[OriginalSource()]))
