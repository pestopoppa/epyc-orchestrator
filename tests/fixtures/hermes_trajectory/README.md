# Public Hermes reference fixtures

Copied unchanged from `letta-ai/trajectory` at `b8c508fd1e05d1eeb1e2c7e13f41b8c2f3f09773`, `fixtures/hermes/{tool-calls,cleanup}/{input,expected}.json`, under the included Apache2.0 license. These are public synthetic examples, not exports from local sessions.

The Python reader emits the session/messages input envelope documented by that pinned Hermes adapter. The package itself is not adopted: its Python client still delegates to a bundled Node20+ subprocess. Private qualification runs that exact compiled bundle as an independent reference; shipped reader, exporter and unit tests use only Python's standard library plus pytest for tests.
