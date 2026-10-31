# Planner local chat request source: regression and repair evidence

## Scope and provenance

This is the evidence for the three `test_local_chat_alias_*` failures on APP `origin/main` at `eb4561b0247f0a94b34da2851bc7f5a2be184efd`. Native implementation commit `63fd32d54d12f146172a317d00cc3326a3dd8d2f` adds `x_request_source` to the local chat planner wire payload, using `producer_source("autopilot")`. The production attribution is correct; the test wire fixtures had not been updated. The proposed fix changes only the three expected payloads and clears `ORCHESTRATOR_REQUEST_SOURCE` in each test so their default-source expectation is deterministic. Production source is unchanged.

## Pinned pre-fix reproduction

- Command: `python -m pytest -q tests/unit/test_autopilot_planner_providers.py -k test_local_chat_alias`
- Snapshot: selected `git archive origin/main` source at the commit above.
- Mutex: `/mnt/raid0/llm/tmp/compute-flags/prep-locks/34-35.lock` (32–33 was busy).
- Affinity: CPUs 34, 35, 130, 131; `OMP_NUM_THREADS=1`; timeout cap 900 seconds.
- PID 910367; `/proc` start ticks 157229076; sampled initial RSS 440 KiB; child max RSS 147960 KiB; process absent after wait.
- Result: 3 failed, all because the actual payload contains `"x_request_source": "autopilot"` and the expected payloads did not.
- Raw log SHA-256: `2a076d9dbc5901ca433e82c0c55c8e81542db4aef4e050fa3c67ee11da1b8b85`.

The readable excerpt below renders trailing spaces as `␠`. The exact raw stream is retained as base64 in the raw-log section.

```text
============================= test session starts ==============================
platform linux -- Python 3.11.17, pytest-9.0.3, pluggy-1.6.0
rootdir: /mnt/raid0/llm/tmp/ni-xsource-snapshot-full-20261031
configfile: pyproject.toml
plugins: asyncio-1.3.0, timeout-2.4.0, langsmith-0.7.30, cov-7.1.0, xdist-3.8.0, hypothesis-6.151.13, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 31 items / 28 deselected / 3 selected

tests/unit/test_autopilot_planner_providers.py FFF                       [100%]

=================================== FAILURES ===================================
________ test_local_chat_alias_posts_draft_chat_payload_with_force_role ________
tests/unit/test_autopilot_planner_providers.py:723: in test_local_chat_alias_posts_draft_chat_payload_with_force_role
    assert captured["payload"] == {
E   AssertionError: assert {'force_role'...': False, ...} == {'force_role'...': False, ...}
E␠␠␠␠␠
E     Omitting 9 identical items, use -vv to show
E     Left contains 1 more item:
E     {'x_request_source': 'autopilot'}
E␠␠␠␠␠
E     Full diff:
E       {
E           'force_role': 'ingest_long_context',
E           'max_tokens': 2048,
E           'max_turns': 1,
E           'mock_mode': False,
E           'prompt': 'planner prompt',
E           'real_mode': True,
E           'request_id': 'planner-local-chat-01234567',
E           'request_priority': 'background',
E           'workload_class': 'campaign',
E     +     'x_request_source': 'autopilot',
E       }
________ test_local_chat_alias_disabling_force_role_omits_payload_field ________
tests/unit/test_autopilot_planner_providers.py:776: in test_local_chat_alias_disabling_force_role_omits_payload_field
    assert captured["payload"] == {
E   AssertionError: assert {'max_tokens'... prompt', ...} == {'max_tokens'... prompt', ...}
E␠␠␠␠␠
E     Omitting 8 identical items, use -vv to show
E     Left contains 1 more item:
E     {'x_request_source': 'autopilot'}
E␠␠␠␠␠
E     Full diff:
E       {
E           'max_tokens': 2048,
E           'max_turns': 1,
E           'mock_mode': False,
E           'prompt': 'planner prompt',
E           'real_mode': True,
E           'request_id': 'planner-local-chat-01234567',
E           'request_priority': 'background',
E           'workload_class': 'campaign',
E     +     'x_request_source': 'autopilot',
E       }
_______ test_local_chat_alias_posts_critique_payload_without_force_role ________
tests/unit/test_autopilot_planner_providers.py:828: in test_local_chat_alias_posts_critique_payload_without_force_role
    assert captured["payload"] == {
E   AssertionError: assert {'max_tokens'... prompt', ...} == {'max_tokens'... prompt', ...}
E␠␠␠␠␠
E     Omitting 8 identical items, use -vv to show
E     Left contains 1 more item:
E     {'x_request_source': 'autopilot'}
E␠␠␠␠␠
E     Full diff:
E       {
E           'max_tokens': 2048,
E           'max_turns': 1,
E           'mock_mode': False,
E           'prompt': 'planner prompt',
E           'real_mode': True,
E           'request_id': 'planner-local-chat-01234567',
E           'request_priority': 'background',
E           'workload_class': 'campaign',
E     +     'x_request_source': 'autopilot',
E       }
=============================== warnings summary ===============================
../../epyc-orchestrator/.venv/lib/python3.11/site-packages/opentelemetry/util/_importlib_metadata.py:32
  /mnt/raid0/llm/epyc-orchestrator/.venv/lib/python3.11/site-packages/opentelemetry/util/_importlib_metadata.py:32: DeprecationWarning: SelectableGroups dict interface is deprecated. Use select.
    return EntryPoints(ep for group_eps in eps.values() for ep in group_eps)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/unit/test_autopilot_planner_providers.py::test_local_chat_alias_posts_draft_chat_payload_with_force_role - AssertionError: assert {'force_role'...': False, ...} == {'force_role'...': False, ...}
␠␠
  Omitting 9 identical items, use -vv to show
  Left contains 1 more item:
  {'x_request_source': 'autopilot'}
␠␠
  Full diff:
    {
        'force_role': 'ingest_long_context',
        'max_tokens': 2048,
        'max_turns': 1,
        'mock_mode': False,
        'prompt': 'planner prompt',
        'real_mode': True,
        'request_id': 'planner-local-chat-01234567',
        'request_priority': 'background',
        'workload_class': 'campaign',
  +     'x_request_source': 'autopilot',
    }
FAILED tests/unit/test_autopilot_planner_providers.py::test_local_chat_alias_disabling_force_role_omits_payload_field - AssertionError: assert {'max_tokens'... prompt', ...} == {'max_tokens'... prompt', ...}
␠␠
  Omitting 8 identical items, use -vv to show
  Left contains 1 more item:
  {'x_request_source': 'autopilot'}
␠␠
  Full diff:
    {
        'max_tokens': 2048,
        'max_turns': 1,
        'mock_mode': False,
        'prompt': 'planner prompt',
        'real_mode': True,
        'request_id': 'planner-local-chat-01234567',
        'request_priority': 'background',
        'workload_class': 'campaign',
  +     'x_request_source': 'autopilot',
    }
FAILED tests/unit/test_autopilot_planner_providers.py::test_local_chat_alias_posts_critique_payload_without_force_role - AssertionError: assert {'max_tokens'... prompt', ...} == {'max_tokens'... prompt', ...}
␠␠
  Omitting 8 identical items, use -vv to show
  Left contains 1 more item:
  {'x_request_source': 'autopilot'}
␠␠
  Full diff:
    {
        'max_tokens': 2048,
        'max_turns': 1,
        'mock_mode': False,
        'prompt': 'planner prompt',
        'real_mode': True,
        'request_id': 'planner-local-chat-01234567',
        'request_priority': 'background',
        'workload_class': 'campaign',
  +     'x_request_source': 'autopilot',
    }
================= 3 failed, 28 deselected, 1 warning in 0.20s ==================
```

## Pinned post-fix provider suite

- Command: `python -m pytest -q tests/unit/test_autopilot_planner_providers.py`
- Snapshot: same selected `git archive origin/main` tree with the proposed test-only patch applied.
- Mutex: `/mnt/raid0/llm/tmp/compute-flags/prep-locks/36-37.lock`.
- Affinity: CPUs 36, 37, 132, 133; `OMP_NUM_THREADS=1`; timeout cap 900 seconds.
- PID 913709; `/proc` start ticks 157233368; sampled initial RSS 424 KiB; child max RSS 147816 KiB; process absent after wait.
- Result: 31 passed, 1 existing deprecation warning.
- Raw log SHA-256: `77a7b355a213af9324a7d2f499b5e93a8d404e791a2b68d48357971141f389ce`.

```text
============================= test session starts ==============================
platform linux -- Python 3.11.17, pytest-9.0.3, pluggy-1.6.0
rootdir: /mnt/raid0/llm/tmp/ni-xsource-snapshot-full-20261031
configfile: pyproject.toml
plugins: asyncio-1.3.0, timeout-2.4.0, langsmith-0.7.30, cov-7.1.0, xdist-3.8.0, hypothesis-6.151.13, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 31 items

tests/unit/test_autopilot_planner_providers.py ......................... [ 80%]
......                                                                   [100%]

=============================== warnings summary ===============================
../../epyc-orchestrator/.venv/lib/python3.11/site-packages/opentelemetry/util/_importlib_metadata.py:32
  /mnt/raid0/llm/epyc-orchestrator/.venv/lib/python3.11/site-packages/opentelemetry/util/_importlib_metadata.py:32: DeprecationWarning: SelectableGroups dict interface is deprecated. Use select.
    return EntryPoints(ep for group_eps in eps.values() for ep in group_eps)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 31 passed, 1 warning in 0.19s =========================
```

## Exact raw logs

These base64 blocks decode to the original captured pytest stdout/stderr streams; their SHA-256 values are recorded above. No test result text was altered.

### Pre-fix raw stream (base64)

```text
PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0gdGVzdCBzZXNzaW9uIHN0YXJ0cyA9PT09PT09
PT09PT09PT09PT09PT09PT09PT09PT0KcGxhdGZvcm0gbGludXggLS0gUHl0aG9uIDMuMTEuMTcs
IHB5dGVzdC05LjAuMywgcGx1Z2d5LTEuNi4wCnJvb3RkaXI6IC9tbnQvcmFpZDAvbGxtL3RtcC9u
aS14c291cmNlLXNuYXBzaG90LWZ1bGwtMjAyNjEwMzEKY29uZmlnZmlsZTogcHlwcm9qZWN0LnRv
bWwKcGx1Z2luczogYXN5bmNpby0xLjMuMCwgdGltZW91dC0yLjQuMCwgbGFuZ3NtaXRoLTAuNy4z
MCwgY292LTcuMS4wLCB4ZGlzdC0zLjguMCwgaHlwb3RoZXNpcy02LjE1MS4xMywgYW55aW8tNC4x
My4wCmFzeW5jaW86IG1vZGU9TW9kZS5BVVRPLCBkZWJ1Zz1GYWxzZSwgYXN5bmNpb19kZWZhdWx0
X2ZpeHR1cmVfbG9vcF9zY29wZT1Ob25lLCBhc3luY2lvX2RlZmF1bHRfdGVzdF9sb29wX3Njb3Bl
PWZ1bmN0aW9uCmNvbGxlY3RlZCAzMSBpdGVtcyAvIDI4IGRlc2VsZWN0ZWQgLyAzIHNlbGVjdGVk
Cgp0ZXN0cy91bml0L3Rlc3RfYXV0b3BpbG90X3BsYW5uZXJfcHJvdmlkZXJzLnB5IEZGRiAgICAg
ICAgICAgICAgICAgICAgICAgWzEwMCVdCgo9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09
PT09PSBGQUlMVVJFUyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQpfX19fX19f
XyB0ZXN0X2xvY2FsX2NoYXRfYWxpYXNfcG9zdHNfZHJhZnRfY2hhdF9wYXlsb2FkX3dpdGhfZm9y
Y2Vfcm9sZSBfX19fX19fXwp0ZXN0cy91bml0L3Rlc3RfYXV0b3BpbG90X3BsYW5uZXJfcHJvdmlk
ZXJzLnB5OjcyMzogaW4gdGVzdF9sb2NhbF9jaGF0X2FsaWFzX3Bvc3RzX2RyYWZ0X2NoYXRfcGF5
bG9hZF93aXRoX2ZvcmNlX3JvbGUKICAgIGFzc2VydCBjYXB0dXJlZFsicGF5bG9hZCJdID09IHsK
RSAgIEFzc2VydGlvbkVycm9yOiBhc3NlcnQgeydmb3JjZV9yb2xlJy4uLic6IEZhbHNlLCAuLi59
ID09IHsnZm9yY2Vfcm9sZScuLi4nOiBGYWxzZSwgLi4ufQpFICAgICAKRSAgICAgT21pdHRpbmcg
OSBpZGVudGljYWwgaXRlbXMsIHVzZSAtdnYgdG8gc2hvdwpFICAgICBMZWZ0IGNvbnRhaW5zIDEg
bW9yZSBpdGVtOgpFICAgICB7J3hfcmVxdWVzdF9zb3VyY2UnOiAnYXV0b3BpbG90J30KRSAgICAg
CkUgICAgIEZ1bGwgZGlmZjoKRSAgICAgICB7CkUgICAgICAgICAgICdmb3JjZV9yb2xlJzogJ2lu
Z2VzdF9sb25nX2NvbnRleHQnLApFICAgICAgICAgICAnbWF4X3Rva2Vucyc6IDIwNDgsCkUgICAg
ICAgICAgICdtYXhfdHVybnMnOiAxLApFICAgICAgICAgICAnbW9ja19tb2RlJzogRmFsc2UsCkUg
ICAgICAgICAgICdwcm9tcHQnOiAncGxhbm5lciBwcm9tcHQnLApFICAgICAgICAgICAncmVhbF9t
b2RlJzogVHJ1ZSwKRSAgICAgICAgICAgJ3JlcXVlc3RfaWQnOiAncGxhbm5lci1sb2NhbC1jaGF0
LTAxMjM0NTY3JywKRSAgICAgICAgICAgJ3JlcXVlc3RfcHJpb3JpdHknOiAnYmFja2dyb3VuZCcs
CkUgICAgICAgICAgICd3b3JrbG9hZF9jbGFzcyc6ICdjYW1wYWlnbicsCkUgICAgICsgICAgICd4
X3JlcXVlc3Rfc291cmNlJzogJ2F1dG9waWxvdCcsCkUgICAgICAgfQpfX19fX19fXyB0ZXN0X2xv
Y2FsX2NoYXRfYWxpYXNfZGlzYWJsaW5nX2ZvcmNlX3JvbGVfb21pdHNfcGF5bG9hZF9maWVsZCBf
X19fX19fXwp0ZXN0cy91bml0L3Rlc3RfYXV0b3BpbG90X3BsYW5uZXJfcHJvdmlkZXJzLnB5Ojc3
NjogaW4gdGVzdF9sb2NhbF9jaGF0X2FsaWFzX2Rpc2FibGluZ19mb3JjZV9yb2xlX29taXRzX3Bh
eWxvYWRfZmllbGQKICAgIGFzc2VydCBjYXB0dXJlZFsicGF5bG9hZCJdID09IHsKRSAgIEFzc2Vy
dGlvbkVycm9yOiBhc3NlcnQgeydtYXhfdG9rZW5zJy4uLiBwcm9tcHQnLCAuLi59ID09IHsnbWF4
X3Rva2VucycuLi4gcHJvbXB0JywgLi4ufQpFICAgICAKRSAgICAgT21pdHRpbmcgOCBpZGVudGlj
YWwgaXRlbXMsIHVzZSAtdnYgdG8gc2hvdwpFICAgICBMZWZ0IGNvbnRhaW5zIDEgbW9yZSBpdGVt
OgpFICAgICB7J3hfcmVxdWVzdF9zb3VyY2UnOiAnYXV0b3BpbG90J30KRSAgICAgCkUgICAgIEZ1
bGwgZGlmZjoKRSAgICAgICB7CkUgICAgICAgICAgICdtYXhfdG9rZW5zJzogMjA0OCwKRSAgICAg
ICAgICAgJ21heF90dXJucyc6IDEsCkUgICAgICAgICAgICdtb2NrX21vZGUnOiBGYWxzZSwKRSAg
ICAgICAgICAgJ3Byb21wdCc6ICdwbGFubmVyIHByb21wdCcsCkUgICAgICAgICAgICdyZWFsX21v
ZGUnOiBUcnVlLApFICAgICAgICAgICAncmVxdWVzdF9pZCc6ICdwbGFubmVyLWxvY2FsLWNoYXQt
MDEyMzQ1NjcnLApFICAgICAgICAgICAncmVxdWVzdF9wcmlvcml0eSc6ICdiYWNrZ3JvdW5kJywK
RSAgICAgICAgICAgJ3dvcmtsb2FkX2NsYXNzJzogJ2NhbXBhaWduJywKRSAgICAgKyAgICAgJ3hf
cmVxdWVzdF9zb3VyY2UnOiAnYXV0b3BpbG90JywKRSAgICAgICB9Cl9fX19fX18gdGVzdF9sb2Nh
bF9jaGF0X2FsaWFzX3Bvc3RzX2NyaXRpcXVlX3BheWxvYWRfd2l0aG91dF9mb3JjZV9yb2xlIF9f
X19fX19fCnRlc3RzL3VuaXQvdGVzdF9hdXRvcGlsb3RfcGxhbm5lcl9wcm92aWRlcnMucHk6ODI4
OiBpbiB0ZXN0X2xvY2FsX2NoYXRfYWxpYXNfcG9zdHNfY3JpdGlxdWVfcGF5bG9hZF93aXRob3V0
X2ZvcmNlX3JvbGUKICAgIGFzc2VydCBjYXB0dXJlZFsicGF5bG9hZCJdID09IHsKRSAgIEFzc2Vy
dGlvbkVycm9yOiBhc3NlcnQgeydtYXhfdG9rZW5zJy4uLiBwcm9tcHQnLCAuLi59ID09IHsnbWF4
X3Rva2VucycuLi4gcHJvbXB0JywgLi4ufQpFICAgICAKRSAgICAgT21pdHRpbmcgOCBpZGVudGlj
YWwgaXRlbXMsIHVzZSAtdnYgdG8gc2hvdwpFICAgICBMZWZ0IGNvbnRhaW5zIDEgbW9yZSBpdGVt
OgpFICAgICB7J3hfcmVxdWVzdF9zb3VyY2UnOiAnYXV0b3BpbG90J30KRSAgICAgCkUgICAgIEZ1
bGwgZGlmZjoKRSAgICAgICB7CkUgICAgICAgICAgICdtYXhfdG9rZW5zJzogMjA0OCwKRSAgICAg
ICAgICAgJ21heF90dXJucyc6IDEsCkUgICAgICAgICAgICdtb2NrX21vZGUnOiBGYWxzZSwKRSAg
ICAgICAgICAgJ3Byb21wdCc6ICdwbGFubmVyIHByb21wdCcsCkUgICAgICAgICAgICdyZWFsX21v
ZGUnOiBUcnVlLApFICAgICAgICAgICAncmVxdWVzdF9pZCc6ICdwbGFubmVyLWxvY2FsLWNoYXQt
MDEyMzQ1NjcnLApFICAgICAgICAgICAncmVxdWVzdF9wcmlvcml0eSc6ICdiYWNrZ3JvdW5kJywK
RSAgICAgICAgICAgJ3dvcmtsb2FkX2NsYXNzJzogJ2NhbXBhaWduJywKRSAgICAgKyAgICAgJ3hf
cmVxdWVzdF9zb3VyY2UnOiAnYXV0b3BpbG90JywKRSAgICAgICB9Cj09PT09PT09PT09PT09PT09
PT09PT09PT09PT09PT0gd2FybmluZ3Mgc3VtbWFyeSA9PT09PT09PT09PT09PT09PT09PT09PT09
PT09PT09Ci4uLy4uL2VweWMtb3JjaGVzdHJhdG9yLy52ZW52L2xpYi9weXRob24zLjExL3NpdGUt
cGFja2FnZXMvb3BlbnRlbGVtZXRyeS91dGlsL19pbXBvcnRsaWJfbWV0YWRhdGEucHk6MzIKICAv
bW50L3JhaWQwL2xsbS9lcHljLW9yY2hlc3RyYXRvci8udmVudi9saWIvcHl0aG9uMy4xMS9zaXRl
LXBhY2thZ2VzL29wZW50ZWxlbWV0cnkvdXRpbC9faW1wb3J0bGliX21ldGFkYXRhLnB5OjMyOiBE
ZXByZWNhdGlvbldhcm5pbmc6IFNlbGVjdGFibGVHcm91cHMgZGljdCBpbnRlcmZhY2UgaXMgZGVw
cmVjYXRlZC4gVXNlIHNlbGVjdC4KICAgIHJldHVybiBFbnRyeVBvaW50cyhlcCBmb3IgZ3JvdXBf
ZXBzIGluIGVwcy52YWx1ZXMoKSBmb3IgZXAgaW4gZ3JvdXBfZXBzKQoKLS0gRG9jczogaHR0cHM6
Ly9kb2NzLnB5dGVzdC5vcmcvZW4vc3RhYmxlL2hvdy10by9jYXB0dXJlLXdhcm5pbmdzLmh0bWwK
PT09PT09PT09PT09PT09PT09PT09PT09PT09IHNob3J0IHRlc3Qgc3VtbWFyeSBpbmZvID09PT09
PT09PT09PT09PT09PT09PT09PT09PT0KRkFJTEVEIHRlc3RzL3VuaXQvdGVzdF9hdXRvcGlsb3Rf
cGxhbm5lcl9wcm92aWRlcnMucHk6OnRlc3RfbG9jYWxfY2hhdF9hbGlhc19wb3N0c19kcmFmdF9j
aGF0X3BheWxvYWRfd2l0aF9mb3JjZV9yb2xlIC0gQXNzZXJ0aW9uRXJyb3I6IGFzc2VydCB7J2Zv
cmNlX3JvbGUnLi4uJzogRmFsc2UsIC4uLn0gPT0geydmb3JjZV9yb2xlJy4uLic6IEZhbHNlLCAu
Li59CiAgCiAgT21pdHRpbmcgOSBpZGVudGljYWwgaXRlbXMsIHVzZSAtdnYgdG8gc2hvdwogIExl
ZnQgY29udGFpbnMgMSBtb3JlIGl0ZW06CiAgeyd4X3JlcXVlc3Rfc291cmNlJzogJ2F1dG9waWxv
dCd9CiAgCiAgRnVsbCBkaWZmOgogICAgewogICAgICAgICdmb3JjZV9yb2xlJzogJ2luZ2VzdF9s
b25nX2NvbnRleHQnLAogICAgICAgICdtYXhfdG9rZW5zJzogMjA0OCwKICAgICAgICAnbWF4X3R1
cm5zJzogMSwKICAgICAgICAnbW9ja19tb2RlJzogRmFsc2UsCiAgICAgICAgJ3Byb21wdCc6ICdw
bGFubmVyIHByb21wdCcsCiAgICAgICAgJ3JlYWxfbW9kZSc6IFRydWUsCiAgICAgICAgJ3JlcXVl
c3RfaWQnOiAncGxhbm5lci1sb2NhbC1jaGF0LTAxMjM0NTY3JywKICAgICAgICAncmVxdWVzdF9w
cmlvcml0eSc6ICdiYWNrZ3JvdW5kJywKICAgICAgICAnd29ya2xvYWRfY2xhc3MnOiAnY2FtcGFp
Z24nLAogICsgICAgICd4X3JlcXVlc3Rfc291cmNlJzogJ2F1dG9waWxvdCcsCiAgICB9CkZBSUxF
RCB0ZXN0cy91bml0L3Rlc3RfYXV0b3BpbG90X3BsYW5uZXJfcHJvdmlkZXJzLnB5Ojp0ZXN0X2xv
Y2FsX2NoYXRfYWxpYXNfZGlzYWJsaW5nX2ZvcmNlX3JvbGVfb21pdHNfcGF5bG9hZF9maWVsZCAt
IEFzc2VydGlvbkVycm9yOiBhc3NlcnQgeydtYXhfdG9rZW5zJy4uLiBwcm9tcHQnLCAuLi59ID09
IHsnbWF4X3Rva2VucycuLi4gcHJvbXB0JywgLi4ufQogIAogIE9taXR0aW5nIDggaWRlbnRpY2Fs
IGl0ZW1zLCB1c2UgLXZ2IHRvIHNob3cKICBMZWZ0IGNvbnRhaW5zIDEgbW9yZSBpdGVtOgogIHsn
eF9yZXF1ZXN0X3NvdXJjZSc6ICdhdXRvcGlsb3QnfQogIAogIEZ1bGwgZGlmZjoKICAgIHsKICAg
ICAgICAnbWF4X3Rva2Vucyc6IDIwNDgsCiAgICAgICAgJ21heF90dXJucyc6IDEsCiAgICAgICAg
J21vY2tfbW9kZSc6IEZhbHNlLAogICAgICAgICdwcm9tcHQnOiAncGxhbm5lciBwcm9tcHQnLAog
ICAgICAgICdyZWFsX21vZGUnOiBUcnVlLAogICAgICAgICdyZXF1ZXN0X2lkJzogJ3BsYW5uZXIt
bG9jYWwtY2hhdC0wMTIzNDU2NycsCiAgICAgICAgJ3JlcXVlc3RfcHJpb3JpdHknOiAnYmFja2dy
b3VuZCcsCiAgICAgICAgJ3dvcmtsb2FkX2NsYXNzJzogJ2NhbXBhaWduJywKICArICAgICAneF9y
ZXF1ZXN0X3NvdXJjZSc6ICdhdXRvcGlsb3QnLAogICAgfQpGQUlMRUQgdGVzdHMvdW5pdC90ZXN0
X2F1dG9waWxvdF9wbGFubmVyX3Byb3ZpZGVycy5weTo6dGVzdF9sb2NhbF9jaGF0X2FsaWFzX3Bv
c3RzX2NyaXRpcXVlX3BheWxvYWRfd2l0aG91dF9mb3JjZV9yb2xlIC0gQXNzZXJ0aW9uRXJyb3I6
IGFzc2VydCB7J21heF90b2tlbnMnLi4uIHByb21wdCcsIC4uLn0gPT0geydtYXhfdG9rZW5zJy4u
LiBwcm9tcHQnLCAuLi59CiAgCiAgT21pdHRpbmcgOCBpZGVudGljYWwgaXRlbXMsIHVzZSAtdnYg
dG8gc2hvdwogIExlZnQgY29udGFpbnMgMSBtb3JlIGl0ZW06CiAgeyd4X3JlcXVlc3Rfc291cmNl
JzogJ2F1dG9waWxvdCd9CiAgCiAgRnVsbCBkaWZmOgogICAgewogICAgICAgICdtYXhfdG9rZW5z
JzogMjA0OCwKICAgICAgICAnbWF4X3R1cm5zJzogMSwKICAgICAgICAnbW9ja19tb2RlJzogRmFs
c2UsCiAgICAgICAgJ3Byb21wdCc6ICdwbGFubmVyIHByb21wdCcsCiAgICAgICAgJ3JlYWxfbW9k
ZSc6IFRydWUsCiAgICAgICAgJ3JlcXVlc3RfaWQnOiAncGxhbm5lci1sb2NhbC1jaGF0LTAxMjM0
NTY3JywKICAgICAgICAncmVxdWVzdF9wcmlvcml0eSc6ICdiYWNrZ3JvdW5kJywKICAgICAgICAn
d29ya2xvYWRfY2xhc3MnOiAnY2FtcGFpZ24nLAogICsgICAgICd4X3JlcXVlc3Rfc291cmNlJzog
J2F1dG9waWxvdCcsCiAgICB9Cj09PT09PT09PT09PT09PT09IDMgZmFpbGVkLCAyOCBkZXNlbGVj
dGVkLCAxIHdhcm5pbmcgaW4gMC4yMHMgPT09PT09PT09PT09PT09PT09Cg==
```

### Post-fix raw stream (base64)

```text
PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0gdGVzdCBzZXNzaW9uIHN0YXJ0cyA9PT09PT09
PT09PT09PT09PT09PT09PT09PT09PT0KcGxhdGZvcm0gbGludXggLS0gUHl0aG9uIDMuMTEuMTcs
IHB5dGVzdC05LjAuMywgcGx1Z2d5LTEuNi4wCnJvb3RkaXI6IC9tbnQvcmFpZDAvbGxtL3RtcC9u
aS14c291cmNlLXNuYXBzaG90LWZ1bGwtMjAyNjEwMzEKY29uZmlnZmlsZTogcHlwcm9qZWN0LnRv
bWwKcGx1Z2luczogYXN5bmNpby0xLjMuMCwgdGltZW91dC0yLjQuMCwgbGFuZ3NtaXRoLTAuNy4z
MCwgY292LTcuMS4wLCB4ZGlzdC0zLjguMCwgaHlwb3RoZXNpcy02LjE1MS4xMywgYW55aW8tNC4x
My4wCmFzeW5jaW86IG1vZGU9TW9kZS5BVVRPLCBkZWJ1Zz1GYWxzZSwgYXN5bmNpb19kZWZhdWx0
X2ZpeHR1cmVfbG9vcF9zY29wZT1Ob25lLCBhc3luY2lvX2RlZmF1bHRfdGVzdF9sb29wX3Njb3Bl
PWZ1bmN0aW9uCmNvbGxlY3RlZCAzMSBpdGVtcwoKdGVzdHMvdW5pdC90ZXN0X2F1dG9waWxvdF9w
bGFubmVyX3Byb3ZpZGVycy5weSAuLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uIFsgODAlXQouLi4u
Li4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgWzEwMCVdCgo9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09IHdhcm5p
bmdzIHN1bW1hcnkgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQouLi8uLi9lcHljLW9y
Y2hlc3RyYXRvci8udmVudi9saWIvcHl0aG9uMy4xMS9zaXRlLXBhY2thZ2VzL29wZW50ZWxlbWV0
cnkvdXRpbC9faW1wb3J0bGliX21ldGFkYXRhLnB5OjMyCiAgL21udC9yYWlkMC9sbG0vZXB5Yy1v
cmNoZXN0cmF0b3IvLnZlbnYvbGliL3B5dGhvbjMuMTEvc2l0ZS1wYWNrYWdlcy9vcGVudGVsZW1l
dHJ5L3V0aWwvX2ltcG9ydGxpYl9tZXRhZGF0YS5weTozMjogRGVwcmVjYXRpb25XYXJuaW5nOiBT
ZWxlY3RhYmxlR3JvdXBzIGRpY3QgaW50ZXJmYWNlIGlzIGRlcHJlY2F0ZWQuIFVzZSBzZWxlY3Qu
CiAgICByZXR1cm4gRW50cnlQb2ludHMoZXAgZm9yIGdyb3VwX2VwcyBpbiBlcHMudmFsdWVzKCkg
Zm9yIGVwIGluIGdyb3VwX2VwcykKCi0tIERvY3M6IGh0dHBzOi8vZG9jcy5weXRlc3Qub3JnL2Vu
L3N0YWJsZS9ob3ctdG8vY2FwdHVyZS13YXJuaW5ncy5odG1sCj09PT09PT09PT09PT09PT09PT09
PT09PSAzMSBwYXNzZWQsIDEgd2FybmluZyBpbiAwLjE5cyA9PT09PT09PT09PT09PT09PT09PT09
PT09Cg==
```

## Review gate

`gitnexus status` for the APP MHS worktree reports “Repository not indexed”; the multi-repo index cannot resolve this test symbol (impact is `UNKNOWN`, context reports not found). This refusal was retained and no re-analysis or index bypass was performed. Manual trace confirms the change is confined to the three provider test fixtures; implementation and production request attribution are untouched.
