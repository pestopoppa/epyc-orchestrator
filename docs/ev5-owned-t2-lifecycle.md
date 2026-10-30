# EV-5 owned T2 model lifecycle

The ThinkPRM recipe is available through `thinkprm_eval_config()` in
`scripts/server/orchestrator_stack.py`. It resolves the canonical model and the
v10 CPU kernel store. It is absent from normal stack startup.

For an **isolated T2 eval whose launcher owns its generator process**, use the
same `subprocess.Popen` handle that launched the private generator:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "scripts/autopilot"))
from eval_tower import EvalTower
from scripts.server.orchestrator_stack import thinkprm_owned_t2_verifier

# The caller starts generation_proc with Popen and routes private_eval_api_url
# to that exact private model process. Do not pass a shared fleet PID or an
# API process whose model children the caller does not own.
verifier = thinkprm_owned_t2_verifier(
    generation_proc,
    log_path=Path(output_dir) / "thinkprm-server.log",
    receipt_path=Path(output_dir) / "generator-drained.json",
)
result = EvalTower(url=private_eval_api_url).eval_t2(
    n=500, seed=42, process_verifier=verifier, verification_n=10
)
```

`eval_t2` runs its original pool, scorer and confidence selection first. Its
whole-call finalizer drains the captured generator on every return or error,
including pool/admission failure and cross-family refusal. When eligible
cross-family rows exist, the verifier drains the generator **before** loading
ThinkPRM, writes an fsynced PID/starttime receipt, and drains its own private
server at exit. TERM/KILL goes through a pidfd bound to the direct child; it
never searches process names or signals a shared process. The caller retains
ownership of any private API process it launched separately.

The existing `eval_t2_process_verification.py` post-scoring CLI also uses the
stack recipe, but its `--generation-drained` flag is a caller assertion. That
CLI is not evidence of automatic unloading. The owned T2 hook above is the
automatic lifecycle path.

The isolated native acceptance at
`/mnt/raid0/llm/tmp/codex-ni-next-cpu-prep-20261030/native-t2-hook/`
exercises the actual `EvalTower.eval_t2` hook with one operational control and
proves CPU store load, one real ThinkPRM request, fsynced attribution and both
captured processes dead. The bounded post-work control at `native-postwork75/`
completed 2/4 control rows, returned rc75 with `partial=true`, and drained both
processes. Both used a fixed-confidence control and make **no quality claim**.
No shared production main model was unloaded in this acceptance; a serving
owner's actual boundary remains to be exercised separately.
