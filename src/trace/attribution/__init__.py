"""HIL-17 failure-attribution ladder (code half; zero inference in this package's tests).

- ``schema``      AgentDebugX 0.5.2 field-level mirror (``AgentEvent``, ``AgentTrajectory``)
- ``adapter``     orchestrator traces (``L2Trace``, replay recordings) -> AgentTrajectory, and back to L2
- ``judge``       injected judge callable; local OpenAI-compatible client; scripted fakes; token meter
- ``prompts``     trajectory rendering and the localize / bisect / adjudicate / multi-step prompts
- ``ladder``      rung 0 L2 rules, rung 1 whole trace, rung 2 bisection, rung 3 adjudicate-on-disagreement
- ``aggregate``   k-sample vote-share ranking, temperature-0 comparator, random baseline
- ``metrics``     nDCG@K (exp / linear, tie-expected), exact random expectation, single-pick accuracy
- ``whowhen_pro`` Who&When Pro text split -> cases with resolved gold units; stratified slice
- ``report``      per-rung accuracy / token table, aggregated nDCG table, markdown writer

Driver: ``scripts/trace/attribution_ladder.py`` (``--dry-run`` = fixtures + fake judge; a real run needs
an inference window and refuses without ``WINDOW_CONFIRMED=1``).
"""
