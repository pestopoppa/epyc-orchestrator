# AutoPilot Short-Term Memory
<!-- Journal-derived generated view from folded append-only ledger -->

## Running Hypotheses
- [t762] Toggle flags: {'session_scratchpad': False} -- observed (q=1.81)
- [t763] Toggle flags: {'generation_monitor': False} -- observed (q=1.74)
- [t764] Toggle flags: {'streaming': False} -- observed (q=1.88)
- [t765] Toggle flags: {'scripts': True} -- observed (q=1.81)
- [t766] Seed 10 questions across all -- observed (q=1.81)
- [t767] Toggle flags: {'structured_delimiters': False} -- observed (q=1.81)
- [t768] Seed 10 questions across all -- observed (q=1.81)
- [t769] Toggle flags: {'react_mode': False} -- observed (q=1.81)
- [t770] Toggle flags: {'approval_gates': True} -- observed (q=1.88)
- [t771] Toggle flags: {'task_token_budget': False} -- observed (q=1.81)
- [t772] Seed 10 questions across all -- observed (q=1.88)
- [t773] Toggle flags: {'langgraph_architect': False} -- observed (q=1.67)
- [t774] Seed 10 questions across all -- observed (q=1.81)
- [t775] Seed 10 questions across all -- confirmed (q=1.95)
- [t776] Seed 10 questions across all -- confirmed (q=1.88)
- [t778] Toggle flags: {'user_modeling': True} -- observed (q=1.58)
- [t779] Tighten the formalizer's output-format enforcement so constrained-format requests (instruction_precision suite: exact JSON/list/field-order/casing constraints) are satisfied verbatim. Add an explicit final-answer format-compliance check that re-emits strictly to the requested schema before returning. Do NOT lengthen general prose or alter non-format paths. Hypothesis: instruction_precision is stuck at 0.00 because the formalizer under-enforces strict schema adherence; targeted enforcement should move it off the floor without regressing already-passing suites. -- observed (q=1.74)
- [t780] Seed 10 questions across all -- confirmed (q=2.13)
- [t781] Seed 10 questions across all -- observed (q=2.13)
- [t782] Improve retention of answer-bearing details during context compaction for long_context requests: when folding/summarizing, explicitly preserve named entities, numeric values, and the specific spans needed to answer multi-part long-context questions, keeping the compaction free-zone near L3 (do not over-compress past the L3->L4 retention knee). Do NOT increase summary verbosity on short contexts or alter non-long-context paths. Hypothesis: long_context regressed 3.0->1.5 because compaction drops answer-bearing spans; retention-targeted compaction should recover it without regressing faithfulness or speed. -- observed (q=2.13)
- [t784] Sharpen tool-use instruction formatting for agentic/multi-step tool tasks: state tool-selection and argument-construction rules as explicit structured steps (when to call which tool, how to chain results, when to stop), and require the model to actually invoke the needed tool rather than describe it. Do NOT add verbosity to non-tool paths or change tool registration. Hypothesis: agentic is stuck at 1.50 because tool-use guidance is under-specified for multi-step orchestration; structured tool-use instructions should move agentic up without regressing the already-3.0 tool_use/coder suites. -- observed (q=2.13)
- [t802] Seed 10 questions across all -- confirmed (q=2.07)
- [t803] Seed 10 questions across all -- observed (q=2.07)
- [t805] Optimize memrl_retrieval surface -- confirmed (q=1.92)
- [t806] Seed 10 questions across all -- observed (q=1.86)
- [t815] Optimize memrl_retrieval surface -- observed (q=1.80)
- [t817] Optimize memrl_retrieval surface -- observed (q=1.86)
- [t833] Optimize memrl_retrieval surface -- observed (q=1.74)

## Optimization Directions
- [t770] Quality plateau — try a different optimization species or axis
- [t771] Investigate declining suites: long_context (1.50 < 3.00)
- [t772] Investigate declining suites: long_context (1.50 < 3.00)
- [t772] Seeding complete — evaluate if routing data is sufficient for training
- [t772] Quality plateau — try a different optimization species or axis
- [t773] Investigate declining suites: general (1.50 < 3.00), tool_use (2.40 < 3.00)
- [t774] Investigate declining suites: simpleqa (0.00 < 1.50)
- [t774] Seeding complete — evaluate if routing data is sufficient for training
- [t775] Seeding complete — evaluate if routing data is sufficient for training
- [t776] Investigate declining suites: hotpotqa (1.50 < 3.00)
- [t776] Seeding complete — evaluate if routing data is sufficient for training
- [t778] Investigate declining suites: gpqa (0.00 < 1.50)
- [t779] Investigate declining suites: general (1.50 < 3.00)
- [t780] Investigate declining suites: long_context (1.50 < 3.00)
- [t780] Seeding complete — evaluate if routing data is sufficient for training
- [t781] Seeding complete — evaluate if routing data is sufficient for training
- [t802] Investigate declining suites: hotpotqa (1.50 < 2.00)
- [t802] Seeding complete — evaluate if routing data is sufficient for training
- [t803] Seeding complete — evaluate if routing data is sufficient for training
- [t805] Numeric optimization working — continue exploring this surface
- [t806] Investigate declining suites: instruction_precision (1.50 < 3.00), vl (0.00 < 1.50), hotpotqa (1.50 < 2.00)
- [t806] Seeding complete — evaluate if routing data is sufficient for training
- [t815] Investigate declining suites: instruction_precision (0.00 < 3.00), math (2.00 < 3.00), vl (0.00 < 1.50), cruxeval (1.50 < 3.00)
- [t815] Numeric optimization working — continue exploring this surface
- [t817] Investigate declining suites: coder (1.50 < 3.00)
- [t817] Numeric optimization working — continue exploring this surface
- [t819] Investigate declining suites: debugbench (1.76 < 3.00), math (2.68 < 3.00), thinking (2.61 < 3.00), gpqa (0.69 < 1.50), coder (2.08 < 3.00), livecodebench (2.44 < 3.00), general (2.18 < 2.57), skill_transfer (1.38 < 1.50), long_context (2.88 < 3.00), instruction_precision (2.00 < 3.00)
- [t826] Investigate declining suites: math (2.57 < 2.68), hotpotqa (2.34 < 2.53), simpleqa (0.50 < 0.70), skill_transfer (1.27 < 1.38), long_context (2.54 < 2.88), agentic (2.54 < 2.65), cruxeval (1.78 < 1.89)
- [t833] Investigate declining suites: coder (1.50 < 3.00), bigcodebench (0.00 < 3.00)
- [t833] Numeric optimization working — continue exploring this surface

## Failure Patterns
- [t763] structural_lab/structural_experiment: VIOLATIONS: |   - Quality regression: 1.744 vs baseline 1.884 (-7.4%, threshold: -5%) |  | DEGRADED SUITES: |   - instruction_precision: 0.000 (floor: 1.0) |   
- [t773] structural_lab/structural_experiment: VIOLATIONS: |   - Quality regression: 1.674 vs baseline 1.884 (-11.1%, threshold: -5%) |  | DEGRADED SUITES: |   - instruction_precision: 0.000 (floor: 1.0) |  
- [t778] structural_lab/structural_experiment: VIOLATIONS: |   - Quality regression: 1.579 vs baseline 1.814 (-13.0%, threshold: -5%) |  | DEGRADED SUITES: |   - instruction_precision: 0.000 (floor: 1.0) |  
- [t819] seeder/deep_eval: VIOLATIONS: |   - Suite 'debugbench' regression: -0.203 (threshold: -0.103; n_result=29, n_baseline=None) |   - Suite 'gpqa' regression: -0.346 (threshold: -0.1
- [t826] seeder/deep_eval: VIOLATIONS: |   - Suite 'debugbench' regression: -0.203 (threshold: -0.103; n_result=29, n_baseline=None) |   - Suite 'gpqa' regression: -0.346 (threshold: -0.1

## Working Context
- Last trial: 833 (numeric_swarm/numeric_trial, q=1.74, revert)
- Best quality: 2.13
- Weak suites: bigcodebench=0.00, gpqa=0.69, instruction_precision=0.00, mode_advantage_hard=0.00, simpleqa=0.00, skill_transfer=1.27, usaco=0.00, vl=0.00
