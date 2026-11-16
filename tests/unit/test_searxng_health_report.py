from datetime import date

from scripts.analysis import searxng_health_report as report


def test_configured_engines_match_live_settings_profile():
    assert report.CONFIGURED_ENGINES == {"bing", "yandex", "wikipedia"}


def test_single_brave_failure_is_not_bad_query(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "orchestrator.log").write_text(
        "2026-07-06 searxng unresponsive_engines: brave (query='x')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(report, "SERVER_LOG_DIR", log_dir)
    monkeypatch.setattr(report, "LOG_DIR", tmp_path / "progress")

    tel = report.collect_telemetry(date(2026, 7, 6), date(2026, 7, 6))

    assert tel["unresponsive_events"] == 1
    assert tel["bad_queries"] == 0
    assert tel["engine_failure_counts"]["brave"] == 1


def test_two_engine_failure_is_bad_query(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "orchestrator.log").write_text(
        "2026-07-06 searxng unresponsive_engines: yandex, wikipedia (query='x')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(report, "SERVER_LOG_DIR", log_dir)
    monkeypatch.setattr(report, "LOG_DIR", tmp_path / "progress")

    tel = report.collect_telemetry(date(2026, 7, 6), date(2026, 7, 6))

    assert tel["unresponsive_events"] == 1
    assert tel["bad_queries"] == 1


def test_human_report_names_configured_engines():
    tel = {
        "days_scanned": 1,
        "searxng_queries": 20,
        "ddg_queries": 0,
        "unresponsive_events": 1,
        "fallback_events": 0,
        "bad_queries": 0,
        "engine_failure_counts": {},
    }
    verdict = {
        "searxng_p50_ms": 0,
        "searxng_p95_ms": 0,
        "ddg_p50_ms": 0,
        "ddg_p95_ms": 0,
        "latency_ratio": 0,
        "bad_query_rate_pct": 0,
        "fallback_rate_pct": 0,
        "verdict": "PROCEED",
        "reasons": [],
    }

    text = report.format_human(tel, verdict)

    assert "configured engines      : bing, wikipedia, yandex" in text


def _write_query_log(root, rows):
    import json
    root.mkdir(parents=True, exist_ok=True)
    (root / "queries.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_bad_query_rate_uses_all_queries_as_denominator(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    lines = "".join(
        f"2026-07-06 searxng unresponsive_engines: brave, wikipedia (query='q{i}')\n"
        for i in range(36)
    )
    log_dir.mkdir()
    (log_dir / "orchestrator.log").write_text(lines, encoding="utf-8")
    rows = [{"id": i, "query": f"q{i}", "backend": "duckduckgo", "success": False} for i in range(47)]
    rows += [{"id": 90 + i, "query": "z", "backend": "searxng", "success": True} for i in range(3)]
    _write_query_log(log_dir, rows)
    monkeypatch.setattr(report, "SERVER_LOG_DIR", log_dir)
    monkeypatch.setattr(report, "LOG_DIR", tmp_path / "progress")

    tel = report.collect_telemetry(date(2026, 7, 6), date(2026, 7, 6))
    verdict = report.compute_verdict(tel)

    assert tel["total_queries"] == 50
    assert tel["bad_queries"] == 36
    assert verdict["bad_query_rate_pct"] == 72.0  # was 1200% over 3 served queries
    assert verdict["bad_query_rate_pct"] <= 100.0


def test_fallbacks_counted_from_query_log(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    rows = [{"query": "a", "backend": "duckduckgo"}] * 47 + [{"query": "b", "backend": "searxng"}] * 3
    _write_query_log(log_dir, rows)
    monkeypatch.setattr(report, "SERVER_LOG_DIR", log_dir)
    monkeypatch.setattr(report, "LOG_DIR", tmp_path / "progress")

    tel = report.collect_telemetry(date(2026, 7, 6), date(2026, 7, 6))
    verdict = report.compute_verdict(tel)

    assert tel["fallback_events"] == 47
    assert verdict["fallback_rate_pct"] == 94.0


def test_fallback_regex_covers_empty_and_low_relevance():
    for msg in (
        "SearXNG failed (boom), falling back to DDG",
        "SearXNG returned no results, falling back to DDG",
        "SearXNG low-relevance result set, falling back to DDG",
    ):
        assert report._FALLBACK_RE.search(msg), msg
