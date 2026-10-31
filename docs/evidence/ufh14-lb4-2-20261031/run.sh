#!/bin/bash
set -euo pipefail
PY=/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python
RUFF=/mnt/raid0/llm/epyc-orchestrator/.venv/bin/ruff
CAPSULE=docs/evidence/ufh14-lb4-2-20261031
"$PY" -m pytest -q -p no:cacheprovider tests/unit/test_openai_client_tool_mode.py -k 'native_tool_parser or native_tool_stream or native_tool_missing or streaming_chat_payload_emits' tests/unit/test_llama_server.py::TestChatCompletionsSchemaForwarding::test_streaming_chat_payload_emits_text_and_assembles_whole_tool_calls --disable-warnings > "$CAPSULE/pytest.stdout" 2> "$CAPSULE/pytest.stderr"
"$RUFF" check src/backends/native_tool_parse.py src/backends/llama_server.py src/llm_primitives/inference.py src/structured_output/parse_events.py tests/unit/test_openai_client_tool_mode.py tests/unit/test_llama_server.py > "$CAPSULE/ruff.stdout" 2> "$CAPSULE/ruff.stderr"
