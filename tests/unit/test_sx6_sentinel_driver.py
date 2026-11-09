import json
import logging
from datetime import date, datetime, timezone

from scripts.analysis import searxng_health_report as report
from scripts.benchmark import sx6_sentinel_driver as drv

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


def test_extract_query_strips_instruction_wrapper():
    p = ("Use web search to find: Who won X? Put your final answer inside <answer></answer> tags.")
    assert drv.extract_query(p) == "Who won X?"


def test_load_queries_from_real_suite_shape(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text("questions:\n  - id: a\n    prompt: 'Use web search to find: q1? Put your final answer inside <answer></answer> tags.'\n"
                 "  - id: b\n    prompt: ''\n", encoding="utf-8")
    assert drv.load_queries(f) == [{"id": "a", "query": "q1?"}]


def test_driver_output_is_consumable_by_the_health_report(tmp_path, monkeypatch):
    root = tmp_path / "root"
    log = logging.getLogger("src.tools.web.search")
    calls = []

    def fake_search(q):
        calls.append(q)
        if len(calls) == 3:
            log.warning("SearXNG failed (boom), falling back to DDG")
            return {"success": True, "backend": "duckduckgo", "elapsed_ms": 900.0}
        if len(calls) == 4:
            log.warning("searxng unresponsive_engines: brave, wikipedia (query='x')")
        return {"success": True, "backend": "searxng", "elapsed_ms": 300.0}

    qs = [{"id": str(i), "query": f"q{i}"} for i in range(25)]
    out = drv.run(qs, root, search=fake_search,
                  ddg=lambda q: {"success": True, "backend": "duckduckgo", "elapsed_ms": 500.0},
                  clock=lambda: NOW)
    assert out["queries"] == 25 and out["backends"] == {"searxng": 24, "duckduckgo": 1}
    assert len((root / "queries.jsonl").read_text().splitlines()) == 25
    assert not log.handlers or all(not isinstance(h, logging.FileHandler) for h in log.handlers)

    monkeypatch.setattr(report, "SERVER_LOG_DIR", root)
    monkeypatch.setattr(report, "LOG_DIR", root / "progress")
    # the logger lines carry the real wall-clock date, so score the day the driver ran
    today = date.today()
    tel = report.collect_telemetry(date(2026, 11, 9), date(2026, 11, 9))
    assert tel["searxng_queries"] == 24 and tel["ddg_queries"] == 26  # 25 baselines + 1 fallback
    if today != date(2026, 11, 9):  # log lines are dated today, not NOW
        tel2 = report.collect_telemetry(today, today)
        assert tel2["fallback_events"] == 1 and tel2["unresponsive_events"] == 1 and tel2["bad_queries"] == 1
