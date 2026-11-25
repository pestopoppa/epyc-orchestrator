"""Byte identity must survive human-review rendering; no models/network needed."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "byte_template_provenance", ROOT / "scripts/validate/template_provenance.py"
)
tp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tp)


def row(tmp_path, upstream, served):
    result = tp.build_row(
        "candidate",
        served,
        "fixture",
        upstream,
        {"upstream_ref": "fixture", "upstream_revision": "a" * 40},
        tmp_path,
        "reviews",
    )
    return result, (tmp_path / "candidate.diff").read_text()


@pytest.mark.parametrize(
    "upstream,served",
    [
        (b"\xff", b"\xfe"),
        (b"\xff", "\ufffd".encode()),
        (b"prefix\xff\n", b"prefix\xfe\n"),
        (b"", b"\xff"),
    ],
)
def test_distinct_invalid_bytes_remain_visible(tmp_path, upstream, served):
    result, diff = row(tmp_path, upstream, served)
    assert result["status"] == "pin" and result["identical"] is False
    assert result["upstream_template_sha256"] != result["served_template_sha256"]
    assert "# identical" not in diff
    assert "hexadecimal bytes" in diff
    if upstream:
        assert "-00000000: " + upstream.hex() in diff
    assert "+00000000: " + served.hex() in diff


def test_normal_unicode_diff_stays_readable(tmp_path):
    _, diff = row(tmp_path, "你好\nold\n".encode(), "你好\nnew\n".encode())
    assert " 你好\n-old\n+new\n" in diff
    assert "hexadecimal" not in diff


@pytest.mark.parametrize("data", [b"text\n", b"\xff", b""])
def test_identical_means_equal_bytes(tmp_path, data):
    result, diff = row(tmp_path, data, data)
    assert result["identical"] is True
    assert diff == "# identical: served template bytes equal the upstream template\n"


def test_unterminated_lines_are_delimited(tmp_path):
    result, diff = row(tmp_path, b"before", b"after")
    assert result["identical"] is False
    assert "-before\n\\ No newline at end of file\n+after\n" in diff
    assert diff.count("\\ No newline at end of file") == 2


def test_offline_cli_plan_references_truthful_diff(tmp_path):
    upstream, served = tmp_path / "upstream", tmp_path / "served"
    upstream.write_bytes(b"\xff")
    served.write_bytes(b"\xfe")
    output = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate/template_provenance.py"),
            "--role",
            "candidate",
            "--gguf",
            str(tmp_path / "not-read.gguf"),
            "--chat-template-file",
            str(served),
            "--upstream-file",
            str(upstream),
            "--diff-dir",
            str(tmp_path / "reviews"),
            "--out",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    result = json.loads(output.read_text())["rows"][0]
    assert result["template_diff"] == "data/template_reviews/candidate.diff"
    assert result["identical"] is False
    assert "-00000000: ff\n+00000000: fe\n" in (tmp_path / "reviews/candidate.diff").read_text()


def test_hex_review_recovers_every_byte_across_chunks(tmp_path):
    upstream, served = bytes(range(256)), bytes(reversed(range(256)))
    _, diff = row(tmp_path, upstream, served)
    for sign, original in [("-", upstream), ("+", served)]:
        lines = [
            line
            for line in diff.splitlines()
            if line.startswith(sign) and not line.startswith(sign * 3)
        ]
        recovered = bytearray()
        for line in lines:
            offset, encoded = line[1:].split(": ", 1)
            assert int(offset, 16) == len(recovered)
            recovered.extend(bytes.fromhex(encoded))
        assert recovered == original


# --- EV-DATE-PIN-1: template_reads_date + date pin warning ---------------------------------------------------

DATE_TMPL = b"{%- set d = strftime_now('%Y-%m-%d') -%}Today is {{ d }}.{{ messages }}"
PLAIN_TMPL = b"{% for m in messages %}{{ m.content }}{% endfor %}"


@pytest.mark.parametrize(
    "tmpl,expected",
    [
        (DATE_TMPL, True),
        (b"Date: {{ date_string }}", True),
        (b"{{ current_date }}", True),
        (PLAIN_TMPL, False),
        (b"", False),
        (None, False),
    ],
)
def test_template_reads_date(tmpl, expected):
    assert tp.ai.template_reads_date(tmpl) is expected


def test_plan_row_carries_template_reads_date(tmp_path):
    for tmpl, expected in [(DATE_TMPL, True), (PLAIN_TMPL, False)]:
        result, _ = row(tmp_path, tmpl, tmpl)
        assert result["template_reads_date"] is expected
    no_upstream = tp.build_row("r", DATE_TMPL, "fixture", None, {}, tmp_path, "reviews")
    assert no_upstream["template_reads_date"] is True


def test_date_pin_warning():
    reg = {"server_mode": {"r": {"chat_template_kwargs": {"enable_thinking": False}}}}
    assert "not pinned" in tp.ai.check_date_pin("r", DATE_TMPL, reg)[0]
    assert tp.ai.check_date_pin("r", PLAIN_TMPL, reg) == []
    pinned = {"server_mode": {"r": {"chat_template_kwargs": {"date_string": "2026-01-01"}}}}
    assert tp.ai.date_pinned(pinned, "r") and tp.ai.check_date_pin("r", DATE_TMPL, pinned) == []


def test_backfill_plan_rows_carry_flag(tmp_path):
    spec = importlib.util.spec_from_file_location("bf", ROOT / "scripts/validate/backfill_template_pins.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    t = tmp_path / "t.jinja"
    t.write_bytes(DATE_TMPL)
    g = tmp_path / "m.gguf"
    g.write_bytes(b"")
    reg = {"server_mode": {"a": {"model": str(g), "chat_template_file": str(t)}}}
    (r,) = bf.plan(reg)
    assert r["template_reads_date"] is True and r["date_pinned"] is False and "date_pin_warning" in r


def test_backfill_alias_row_inherits_host_flag(tmp_path):
    spec = importlib.util.spec_from_file_location("bf2", ROOT / "scripts/validate/backfill_template_pins.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    t = tmp_path / "t.jinja"
    t.write_bytes(DATE_TMPL)
    g = tmp_path / "m.gguf"
    g.write_bytes(b"")
    reg = {"server_mode": {"a": {"model": str(g), "chat_template_file": str(t)}, "b": {"alias_of": "a"}}}
    rows = {r["role"]: r for r in bf.plan(reg)}
    assert rows["b"]["status"] == "alias" and rows["b"]["template_reads_date"] is True
