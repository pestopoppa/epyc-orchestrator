# ROOT completion preparation for MAIN

Original EPD-3-R7 task text, copied exactly from fresh ROOT `origin/main` handoff:

- [ ] **EPD-3-R7 — decide the fate of the 200-char objective truncation at the payload level**, i.e. the
  2026-07-27 audit's defect #2 (12.6% of rows at the cap, 96.9% objective collision) still live on the
  eval-injection path, where changing it changes what is *stored* and not only what is embedded
  (`scripts/benchmark/seeding_injection.py:189,287`) (found 2026-09-14, noninf sweep).

Prepared result details for MAIN to apply with no task-text changes:

- Implementing APP commit: `9c388042475d835f88e62355443e7036986b9a34`.
- Test/artifact: `tests/unit/test_seeding_injection_objective_storage.py`; `artifacts/audit/epd3-r7-objective-payload-20261031/verification.md`.
- Test result: 68 passed across the focused seeding and native embedding controls.
- Belief marker: `[no-belief: software source publication and regression verification]`.
