"""Regression: _strip_import_lines must not drop imports the REPL cannot satisfy.

Origin: 2026-11-18 35B probe smoke -- `import csv`/`import io` were stripped while
csv/io were never pre-loaded, giving 17 consecutive NameErrors.
"""

from __future__ import annotations

from src.prompt_builders.code_utils import _strip_import_lines, extract_code_from_response
from src.repl_environment.environment import REPLEnvironment


def _run(code: str):
    return REPLEnvironment(context="x").execute(code)


def test_every_safe_module_is_preloaded_under_its_name():
    preloaded = REPLEnvironment.preloaded_modules()
    expected = REPLEnvironment.SAFE_IMPORT_MODULES - REPLEnvironment._LAZY_SAFE_MODULES
    for name in ("csv", "io", "string", "heapq", "uuid", "json", "math"):
        assert name in preloaded
    assert set(preloaded) <= expected
    g = REPLEnvironment(context="x")._build_globals()
    for name, mod in preloaded.items():
        assert g[name] is mod


def test_pathlib_not_allowed():
    assert "pathlib" not in REPLEnvironment.SAFE_IMPORT_MODULES


def test_csv_io_snippet_end_to_end():
    response = (
        "```python\n"
        "import csv\nimport io\n"
        "rows = list(csv.DictReader(io.StringIO('a,b\\n1,2\\n3,4')))\n"
        "FINAL(len(rows))\n"
        "```"
    )
    code = extract_code_from_response(response)
    res = _run(code)
    assert res.error in (None, ""), res.error
    assert res.is_final and str(res.final_answer) == "2"


def test_plain_preloaded_import_stripped():
    assert _strip_import_lines("import csv\nx = 1") == "x = 1"
    assert _strip_import_lines("import csv, io\nx = 1") == "x = 1"


def test_import_as_kept_and_works():
    code = _strip_import_lines("import collections as c\nFINAL(c.Counter('aab')['a'])")
    assert code.startswith("import collections as c")
    res = _run(code)
    assert res.is_final and str(res.final_answer) == "2"


def test_from_import_kept_and_works():
    code = _strip_import_lines("from collections import Counter\nFINAL(Counter('aab')['a'])")
    assert "from collections import Counter" in code
    res = _run(code)
    assert res.is_final and str(res.final_answer) == "2"


def test_non_preloaded_module_kept_and_clear_importerror():
    code = _strip_import_lines("import antigravity\nx = 1")
    assert "import antigravity" in code
    res = _run(code)
    assert res.error and "ImportError" in res.error
    assert "not allowed in the REPL sandbox" in res.error
    assert "Allowed:" in res.error


def test_forbidden_module_imports_still_stripped():
    assert _strip_import_lines("import os\nimport sys\nx = 1") == "x = 1"
    assert _strip_import_lines("from os import path\nx = 1") == "x = 1"
    assert _strip_import_lines("import os as o\nx = 1") == "x = 1"


def test_indented_import_kept():
    src = "def f():\n    import csv\n    return csv"
    assert _strip_import_lines(src) == src


def test_triple_quoted_imports_untouched():
    src = 'solution = """\nimport csv\nimport os\n"""\nimport io\nx = 1'
    out = _strip_import_lines(src)
    assert "import csv" in out and "import os" in out
    assert "import io" not in out.replace("import os", "")


def test_nameerror_hint_for_allowed_module():
    res = _run("rows = csv.reader([])")  # csv preloaded now; use a lazy module instead
    # csv is preloaded so no error; sympy is allowed but lazily imported
    assert not res.error
    res = _run("sympy.symbols('x')")
    assert res.error and "NameError" in res.error
    assert "allowed module" in res.error and "import sympy" in res.error


def test_nameerror_hint_for_forbidden_module():
    res = _run("os.getcwd()")
    assert res.error and "NameError" in res.error
    assert "not available in the sandbox" in res.error
