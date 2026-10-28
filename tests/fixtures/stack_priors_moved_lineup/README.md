# Moved-lineup compiler regression (SSU-F18)

The opt-in `tests.plugins.moved_lineup` plugin moves the declared worker aliases
to `architect_critic`, deriving its port and fleet from the current declarations.
These YAML files are test overlays, not serving configuration. The plugin
materializes coherent registry/manifest copies in pytest's temporary directory;
it never rewrites the repository's declarations or starts a server.

The original `/mnt/raid0/llm/tmp/ssuf5/` plugin and full fixtures no longer exist.
This fixture preserves the documented relocation intent, not their lost byte
identity. The historical suite had 35 tests; the current suite has 44. Its one
intentional pin requires the multi-instance `full:` wire prefix and must still
fail after moving onto a single-instance host. No test count is manufactured.

`test_moved_lineup_regression.py` runs the current compiler suite twice and
checks that the moved run has exactly that one failure, with all other cases
passing and no skips/errors. It also verifies that source declaration bytes
are unchanged. This keeps the mutation exercised by ordinary test runs.

For direct inspection:

```sh
python -m pytest -q -p tests.plugins.moved_lineup tests/unit/test_stack_priors_compiler.py
```

The pin is a strict expected failure in that command: an unexpected pass fails
the run. Add `--runxfail` to expose the native assertion failure. The standing
regression itself uses `--runxfail` and checks its exact identity.
