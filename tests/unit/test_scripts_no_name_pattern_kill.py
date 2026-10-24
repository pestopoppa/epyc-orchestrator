"""scripts/ must kill only PIDs it captured, never by name pattern or port.

Origin: INC-20260731-broad-process-pattern-kills (CLAUDE.md Process Management):
`pkill -f llama-server` also matches earlyoom's argv on the shared host.
"""

import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
NAME_PATTERN_KILL = re.compile(
    r"\bpkill\b|\bkillall\b|\bfuser\s+-k\b|\bpgrep\b[^\n|]*\|\s*(?:xargs\s+)?kill\b"
)


def test_scripts_never_kill_by_name_pattern():
    hits = [
        f"{path.relative_to(SCRIPTS.parent)}:{lineno}: {line.strip()}"
        for path in sorted(SCRIPTS.rglob("*"))
        if path.suffix in {".sh", ".py"} and path.is_file()
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
        if NAME_PATTERN_KILL.search(line)
    ]
    assert hits == []
