# Fable5 Gate Report

Ready: false

## Blockers

- w4_w6_restart_cutover: W6 audit cutover readiness is blocked: W6 audit gaming alarm is triggered

## Next Actions

### continue_w4_w6_accrual

- Priority: `P0`
- Status: `blocked`
- Reason: Sequential authority and W6 audit cutover need more trusted rows before any flip.
- Evidence:
  - `trusted_vectors`: 379
  - `trusted_vectors_required`: 120
  - `trusted_vectors_remaining`: 0
  - `seq_shadow_rows`: 301
  - `seq_shadow_rows_required`: 30
  - `seq_shadow_rows_remaining`: 0
  - `w6_audited_rows`: 212
  - `w6_audited_rows_required`: 30
  - `w6_audited_rows_remaining`: 0
  - `w6_gaming_alarm`: true
  - `w6_core_inflation_warning`: false
  - `w6_era_excluded_gaming_event_count`: 0
  - `w6_fence_governance_status`: "no_excluded_gaming_events"
  - `w6_fence_governance_blockers`: []
  - `w6_alarm_clearance_clean_trials_required`: 19
  - `cutover_horizon_clean_trials_remaining`: 19
  - `cutover_horizon_blocker`: "w6_alarm_clearance"
  - `cutover_horizon_components`: {"seq_shadow_rows": 0, "seq_trusted_vectors": 0, "w6_alarm_clearance": 19, "w6_audited_trials": 0}
- Command: `cd /mnt/raid0/llm/epyc-orchestrator && uv run python scripts/autopilot/restart_readiness_report.py --json --strict --require-seq-cutover --require-w6-audit --require-current-code`
- Follow-up: `uv run python scripts/autopilot/fable5_gate_report.py --json --strict`

### activate_tool_use_sentinel_lane

- Priority: `P0`
- Status: `ready`
- Reason: StrategyStore already exposes tool-use hints to the planner; the remaining gap is activating the API and AutoPilot tool-sentinel telemetry lane so tool use is measured.
- Requires: coordinated API reload plus AutoPilot restart at a trial boundary; this changes the active eval mix
- Evidence:
  - `activation_gaps`: ["autopilot_env_missing_AUTOPILOT_TOOL_SENTINELS"]
  - `autopilot_tool_sentinels_enabled`: false
  - `api_tool_sentinels_enabled`: true
  - `api_tools_enabled`: true
  - `api_repl_enabled`: true
  - `latest_tool_metrics`: {"mean_tools_used": 0.06, "per_suite_tool_helpfulness": {}, "tool_helpfulness": NaN, "tool_name_counts": null, "tool_use_rate": 0.06, "total_tool_calls": 3, "trial_id": 1335}
  - `recent_tool_metrics`: {"evaluated_rows": 10, "latest_nonzero_tool_metrics": {"mean_tools_used": 0.06, "per_suite_tool_helpfulness": {}, "tool_helpfulness": NaN, "tool_name_counts": null, "tool_use_rate": 0.06, "total_tool_calls": 3, "trial_id": 1335}, "nonzero_rows": 10, "total_tool_calls": 43, "trial_ids": [1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335], "window": 10}
- Command: `At a controlled trial boundary, reload the orchestrator API with AUTOPILOT_TOOL_SENTINELS=1, restart AutoPilot with uv run python scripts/autopilot/start_fable_authority_daemon.py --max-trials 3000, then run AUTOPILOT_TOOL_SENTINELS=1 uv run python scripts/autopilot/gate3_tool_telemetry.py`
- Follow-up: `uv run python scripts/autopilot/fable5_gate_report.py --json --strict`

### collect_a9_audit_target_pairwise_preferences

- Priority: `P1`
- Status: `ready`
- Reason: The source-q-reward A9 training target is preregistered offline-only, but the broader audit-target-expanded pairwise ranker still has mixed independent-holdout signal. Remaining blockers are source_family:orchestrator_live_seed:insufficient_pairwise_signal, source_family:seeding_eval:insufficient_pairwise_signal, suite:general:insufficient_pairwise_signal.
- Requires: collect non-overlapping cross-action preference rows for the current audit-target direction gaps, then rebuild the score-ordered pairwise contract and rerun independent holdouts; do not retune the absolute verifier family
- Command: `cd /mnt/raid0/llm/epyc-orchestrator && orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/collect_offline_reward_pairwise_audit_target.sh`
- Follow-up: `uv run python scripts/autopilot/fable5_gate_report.py --json --strict --require-current-code`

## Sections

### phase_health

- Status: `ready`
- Summary: AutoPilot phase heartbeat is stopped at trial None / stopped.
- Details:
  - `ok`: true
  - `status`: "stopped"
  - `trial_id`: null
  - `phase`: "stopped"
  - `action_type`: null
  - `heartbeat_age_s`: 192316.2291560173
  - `pid`: 1039446
  - `pid_alive`: false
  - `process_started_at_s`: null
  - `require_current_code`: true
  - `code_stale`: false
  - `code_stale_paths`: []
  - `eval_label`: null
  - `eval_completed_questions`: null
  - `eval_total_questions`: null
  - `eval_correct_questions`: null
  - `eval_correct_pct`: null
  - `eval_concurrency`: null
  - `planner_hints_enabled`: null
  - `seq_verdict_enabled`: null
  - `w6_audit_accrual_enabled`: null
  - `w6_audit_shadow_only`: null
  - `w6_audit_n`: null
  - `w6_audit_every_n_trials`: null
  - `autopilot_planner_timeout`: null

### w4_w6_restart_cutover

- Status: `blocked`
- Summary: W4/W6 strict restart cutover remains blocked.
- Blockers:
  - W6 audit cutover readiness is blocked: W6 audit gaming alarm is triggered
- Details:
  - `restart_ready`: false
  - `archive_source_surface_ok`: true
  - `archive_source_surface_count`: 6
  - `archive_source_surface_failed_count`: 0
  - `seq_cutover_ready`: true
  - `seq_trusted_vector_trials`: 379
  - `seq_min_trusted_vector_trials`: 120
  - `seq_trusted_vector_trials_remaining`: 0
  - `seq_shadow_rows`: 301
  - `seq_min_shadow_rows`: 30
  - `seq_shadow_rows_remaining`: 0
  - `w8_promotion_status`: "blocked"
  - `w8_open_requirements`: ["last_blocked:stale-reference", "combined_E_below_required", "fresh_promotion_eval_required", "seq_confirmation_required"]
  - `w8_pending_candidate`: null
  - `w8_pending_source_trial_id`: null
  - `w8_pending_attempts`: null
  - `w8_last_finalized_trial_id`: null
  - `w8_last_finalized_candidate`: null
  - `w8_last_finalized_combined_E`: null
  - `w8_last_finalized_delta_excludes_regression`: null
  - `w8_last_blocked_trial_id`: 1305
  - `w8_last_blocked_candidate`: "3055f1e32fac0316"
  - `w8_last_blocked_reason`: "stale-reference"
  - `w8_latest_seq_trial_id`: 1335
  - `w8_latest_candidate`: "84a55fd82af865d8"
  - `w8_latest_combined_E`: 0.91
  - `w8_latest_required_E`: 100.0
  - `w8_latest_confirmed`: false
  - `w8_latest_seq_state`: "accumulating"
  - `w8_latest_baseline_reference_state`: "fresh"
  - `w8_latest_fresh_eval`: false
  - `w8_baseline_reference_last_forced_trial_id`: 1319
  - `w8_baseline_reference_last_forced_reason`: "10 trusted profile trials since baseline reference draw"
  - `w8_baseline_reference_last_forced_stale`: false
  - `w8_baseline_reference_blocked_trial_id`: 1336
  - `w8_baseline_reference_blocked_reason`: "Auto-blacklisted: 3 consecutive failures ending at trial 1320"
  - `w6_audit_cutover_ready`: false
  - `w6_audited_trial_count`: 212
  - `w6_min_audited_trials`: 30
  - `w6_audited_trial_count_remaining`: 0
  - `w6_alarm_clearance_clean_trials_required`: 19
  - `w6_raw_audited_trial_count`: 225
  - `w6_trusted_audited_trial_count`: 212
  - `w6_untrusted_audited_trial_count`: 13
  - `w6_untrusted_audited_trial_ids`: [1177, 1178, 1180, 1206, 1257, 1258, 1259, 1260, 1261, 1262, 1264, 1265, 1266]
  - `w6_gaming_alarm`: true
  - `w6_core_inflation_warning`: false
  - `w6_era_excluded_gaming_event_count`: 0
  - `w6_fence_governance_status`: "no_excluded_gaming_events"
  - `w6_fence_governance_blockers`: []
  - `w6_potential_overfit_divergences`: 2
  - `cutover_horizon_clean_trials_remaining`: 19
  - `cutover_horizon_blocker`: "w6_alarm_clearance"
  - `cutover_horizon_components`: {"seq_shadow_rows": 0, "seq_trusted_vectors": 0, "w6_alarm_clearance": 19, "w6_audited_trials": 0}
  - `baseline_seed_append_ready`: false
  - `baseline_seed_append_required`: false
  - `baseline_seed_append_expect_trial_counter`: null
  - `baseline_seed_append_expect_journal_max_trial_id`: null
  - `durable_journal_max_trial_id`: 1336
  - `state_trial_counter`: 1337
  - `snapshot_restart_readiness`: "full_replay_ready"
  - `snapshot_payload_journal_max_trial_id`: 1336

### w8_promotion_trajectory

- Status: `ready`
- Summary: W8 replay trajectory has no concentration warning.
- Details:
  - `status`: "progressing"
  - `ok`: true
  - `latest_trial_id`: 1335
  - `snapshot_count`: 312
  - `candidate_count`: 108
  - `status_counts`: {"active_recent_replay": 4, "refuted": 4, "reverted": 79, "single_observation": 19, "stale_accumulating": 2}
  - `terminal_reason_counts`: {"Quality floor violation: 0.000 < 1.0 (tier 1)": 2, "Quality floor violation: 0.055 < 1.0 (tier 1)": 1, "Quality floor violation: 0.109 < 1.0 (tier 1)": 1, "Quality floor violation: 0.218 < 1.0 (tier 1)": 3, "Quality floor violation: 0.240 < 1.0 (tier 1)": 1, "Quality floor violation: 0.273 < 1.0 (tier 1)": 3, "Quality floor violation: 0.327 < 1.0 (tier 1)": 1, "Quality floor violation: 0.382 < 1.0 (tier 1)": 1, "Quality floor violation: 0.390 < 1.0 (tier 1)": 1, "Quality floor violation: 0.900 < 1.0 (tier 1)": 1, "Quality regression: 1.036 vs baseline 1.814 (-42.9%, threshold: -5%)": 1, "Quality regression: 1.081 vs baseline 1.524 (-29.1%, threshold: -5%)": 1, "Quality regression: 1.091 vs baseline 1.814 (-39.9%, threshold: -5%)": 1, "Quality regression: 1.364 vs baseline 1.814 (-24.8%, threshold: -5%)": 1, "Quality regression: 1.440 vs baseline 1.814 (-20.6%, threshold: -5%)": 1, "Quality regression: 1.473 vs baseline 1.814 (-18.8%, threshold: -5%)": 1, "Quality regression: 1.691 vs baseline 1.814 (-6.8%, threshold: -5%)": 1, "Suite 'debugbench' regression: -2.250 (threshold: -1.500; n_result=4, n_baseline=2)": 1, "Suite 'general' regression: -1.800 (threshold: -1.500; n_result=5, n_baseline=2)": 26, "Suite 'hotpotqa' regression: -2.000 (threshold: -1.500; n_result=3, n_baseline=2)": 7, "Suite 'tool_use' regression: -1.200 (threshold: -0.600; n_result=5, n_baseline=5)": 7, "Suite 'tool_use' regression: -1.800 (threshold: -0.600; n_result=5, n_baseline=5)": 11, "Suite 'tool_use' regression: -2.400 (threshold: -0.600; n_result=5, n_baseline=5)": 1, "Suite 'tool_use' regression: -3.000 (threshold: -0.600; n_result=5, n_baseline=5)": 4, "candidate status refuted": 4}
  - `dominant_terminal_reason`: {"baseline_sample_warning": true, "candidate": "7c5026202c4aa284", "count": 26, "details": {"delta": -1.8, "kind": "suite_regression", "n_baseline": 2, "n_result": 5, "suite": "general", "threshold": -1.5}, "latest_trial_id": 1131, "reason": "Suite 'general' regression: -1.800 (threshold: -1.500; n_result=5, n_baseline=2)", "status": "reverted"}
  - `open_requirements`: ["combined_E_below_required", "fresh_promotion_eval_required", "stale_accumulating_candidates_present", "seq_confirmation_required"]
  - `candidate_generation_required`: false
  - `recent_active_candidates`: ["5c7d629d24291a05", "28c8732694945a90", "5dea51f15dcb5f75", "d522d0f44587142f"]
  - `replay_eligible_candidates`: ["5c7d629d24291a05", "5dea51f15dcb5f75", "d522d0f44587142f", "289c4fc0fb5a334d", "d3f28243801548b2"]
  - `recent_replay_eligible_candidates`: ["5c7d629d24291a05", "5dea51f15dcb5f75", "d522d0f44587142f"]
  - `stale_accumulating_candidate_count`: 2
  - `replay_concentration`: {"active_recent_attempts": {"5c7d629d24291a05": 2, "5dea51f15dcb5f75": 2, "d522d0f44587142f": 2}, "active_recent_candidate_count": 3, "single_observation_count": 19, "stale_accumulating_count": 2, "top_active_attempt_share": 0.333333, "top_active_attempts": 2, "top_active_candidate": "d522d0f44587142f", "total_active_recent_attempts": 6, "warning": false, "warning_reason": null}

### ds_e1_dynamic_stack

- Status: `ready`
- Summary: DS-E1 dynamic-stack packet is decision-ready.
- Details:
  - `ready_for_profile_decision`: true
  - `generated_at`: "2026-07-14T11:16:48Z"
  - `section_statuses`: {"contention_matrix": "ready", "ds5_roster_manifest": "ready", "kv_size_measurements": "ready", "ri10_canary": "ready", "stack_roster": "ready"}
  - `kv_required_measurements`: {"architect_general": [2048, 8192], "frontdoor": [2048, 8192, 32768], "ingest_long_context": [2048, 8192, 32768], "worker_general": [2048, 8192]}
  - `kv_observed_measurements`: {"architect_general": [2048, 8192], "frontdoor": [2048, 8192, 32768], "ingest_long_context": [2048, 8192, 32768], "worker_general": [2048, 8192]}
  - `kv_missing_measurements`: null
  - `kv_expected_csv_columns`: null
  - `kv_candidate_paths`: ["/mnt/raid0/llm/epyc-inference-research/data/dynamic_stack/ds_e1_kv_measurements_20260704T163852Z/kv_measurements.csv"]
  - `kv_searched_globs`: ["orchestration/reports/ds_e1*kv*", "orchestration/reports/dynamic_stack*kv*", "../epyc-inference-research/data/dynamic_stack/**/kv*", "../epyc-inference-research/data/kv_measurements/**"]
  - `clean_window_ready`: false
  - `clean_window_blockers`: ["live llama-server process(es): 1524670 /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-27B-MTP-Q8_0.gguf --spec-type draft-mtp -ngl 99 --device ROCm0 --spec-draft-ngl 99 --spec-draft-device ROCm0 --spec-draft-n-max 3 -c 262144 -ub 8192 -fa on -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 --jinja --host 127.0.0.1 --port 8600 -np 1 --slot-save-path /mnt/raid0/llm/cache/kv_slots/mi210_qwen36_27b; 2127073 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 8070 -np 1 -c 32768 -t 96 -ub 8192 --flash-attn on --jinja -ctk q8_0 -ctv q8_0 --mlock --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/frontdoor; 2127392 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16 -ub 512 --no-mmap --reasoning off --jinja --host 127.0.0.1 --port 8072 -np 1 -c 16384 -t 96 -ctk q8_0 -ctv q8_0 --flash-attn on; 2127693 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf --host 127.0.0.1 --port 8083 -np 2 -c 16384 -t 96 -ub 8192 --flash-attn on --jinja -ctk q4_0 -ctv f16 --mlock --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/architect_general; ... +25 more"]
  - `ri10_telemetry_collection_blocker`: "decision_ready"
  - `ri10_telemetry_collection_reason`: "current high-risk telemetry has decision-grade canary arm coverage"
  - `ri10_canary_role_sample_deficit`: 0
  - `ri10_canary_arm_volume_deficit`: 0
  - `ri10_canary_arm_balance_deficits`: {"enforce_high_risk": 0, "shadow_high_risk": 0}
  - `ri10_high_risk_by_role_current`: {"frontdoor": 42, "worker_general": 57, "worker_vision": 46}
  - `ri10_canary_role_high_risk_by_role_current`: {"frontdoor": 42, "worker_general": 57, "worker_vision": 46}
  - `ri10_canary_arm_counts_current`: {"enforce_high_risk": 63, "shadow_high_risk": 82}
  - `ri10_canary_arm_counts_by_role_current`: {"frontdoor": {"enforce_high_risk": 21, "shadow_high_risk": 21}, "worker_general": {"enforce_high_risk": 22, "shadow_high_risk": 35}, "worker_vision": {"enforce_high_risk": 20, "shadow_high_risk": 26}}

### a9_pairwise_collection

- Status: `ready`
- Summary: A9 pairwise source-acquisition window is closed with 0 batch(es); candidate-only contract decision is insufficient_contrast; source-reward diagnostic is contract_ready; source-reward ranker is pairwise_ranker_signal; source-reward target contract is preregistered_offline_training_target; audit-target ranker holdout is mixed_holdout_signal (13/16 passing); audit-target collection window is ready with 4 batch(es).
- Details:
  - `ready`: false
  - `status`: "no_runnable_batches"
  - `manifest_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_expanded_gap_collection_manifest.json"
  - `manifest_schema_version`: "offline_reward_pairwise_collection_window.v1"
  - `source_plan_decision`: {"recommended_next": "score_selected_candidates_and_rebuild_pairwise_contract", "runtime_gate_change_allowed": false, "status": "expansion_plan_ready"}
  - `batch_count`: 0
  - `post_collection_step_count`: 7
  - `candidate_contract_summary_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_preference_contract_candidate_only_expanded_gap_summary.json"
  - `candidate_contract_decision`: {"min_cross_action_pairs": 50, "min_pairs": 100, "recommended_next": "collect_more_within_task_positive_negative_contrasts", "runtime_gate_change_allowed": false, "status": "insufficient_contrast"}
  - `candidate_contract_coverage`: {"action_pair_counts": {"architect_general>coder_escalation": 14, "coder_escalation>architect_general": 2}, "contrastive_groups": 16, "cross_action_pair_rows": 16, "pair_rows": 16, "same_action_pair_rows": 0, "skipped_no_contrast_groups": 222, "source_family_pair_counts": {"seeding_eval": 16}, "source_record_groups": 238, "suite_pair_counts": {"general": 4, "simpleqa": 12}, "unique_action_pairs": 2}
  - `candidate_contract_exhausted`: true
  - `source_reward_diagnostic_summary_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_source_reward_diagnostic_summary.json"
  - `source_reward_diagnostic_decision`: {"min_cross_action_pairs": 50, "min_pairs": 100, "recommended_next": "decide whether A9 should train on source-q-reward pairwise labels or build a new independent scorer/source contract before ranker use", "runtime_gate_change_allowed": false, "status": "contract_ready"}
  - `source_reward_diagnostic_coverage`: {"action_pair_counts": {"architect_general>coder_escalation": 14, "architect_general>frontdoor": 8, "coder_escalation>architect_general": 2, "frontdoor>architect_general": 134, "frontdoor>coder_escalation": 22}, "contrastive_groups": 158, "cross_action_pair_rows": 180, "pair_rows": 180, "same_action_pair_rows": 0, "skipped_no_contrast_groups": 144, "source_family_pair_counts": {"orchestrator_live_seed": 20, "seeding_eval": 160}, "source_record_groups": 302, "suite_pair_counts": {"coder": 8, "debugbench": 8, "general": 8, "gpqa": 8, "hotpotqa": 48, "instruction_precision": 44, "livecodebench": 8, "long_context": 8, "math": 8, "simpleqa": 18, "thinking": 14}, "unique_action_pairs": 5}
  - `source_reward_diagnostic`: {"binary_threshold": 0.5, "diagnostic_only": true, "independent_oracle": false, "interpretation": "Use this to test whether the candidate set contains enough within-task source-reward contrast. It is not an adopted independent reward oracle.", "label_status": "diagnostic_source_reward_passthrough", "recommended_next": "decide whether A9 should train on source-q-reward pairwise labels or build a new independent scorer/source contract before ranker use", "runtime_gate_change_allowed": false, "score_source": "source_q_reward_passthrough", "score_value_counts": {"-0.3": 60, "-0.5": 8, "0.0": 154, "0.1": 56, "0.363842889264226": 2, "0.5": 30, "1.0": 316}, "source_reward_passthrough": true}
  - `source_reward_ranker_summary_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_ranker_source_reward_diagnostic_summary.json"
  - `source_reward_ranker_input`: {"action_pair_counts": {"architect_general>coder_escalation": 14, "architect_general>frontdoor": 8, "coder_escalation>architect_general": 2, "frontdoor>architect_general": 134, "frontdoor>coder_escalation": 22}, "cross_action_pair_rows": 180, "group_count": 158, "pair_rows": 180, "pairing_mode_counts": {"score_ordered": 180}, "same_action_pair_rows": 0, "source_family_pair_counts": {"orchestrator_live_seed": 20, "seeding_eval": 160}, "suite_pair_counts": {"coder": 8, "debugbench": 8, "general": 8, "gpqa": 8, "hotpotqa": 48, "instruction_precision": 44, "livecodebench": 8, "long_context": 8, "math": 8, "simpleqa": 18, "thinking": 14}}
  - `source_reward_ranker_aggregate_decision`: {"best_family": "hist_gradient_boosting", "blockers": [], "recommended_next": "cross_validate_pairwise_ranker_on_expanded_contract", "runtime_gate_change_allowed": false, "status": "pairwise_ranker_signal"}
  - `source_reward_ranker_cv_decision`: {"best_family": "hist_gradient_boosting", "blockers": [], "recommended_next": "resolve_independent_holdout_blockers_before_runtime_use", "runtime_gate_change_allowed": false, "status": "pairwise_ranker_signal"}
  - `source_reward_ranker_holdout_decision`: {"blockers": [], "eligible_holdouts": 3, "passing_holdouts": 3, "recommended_next": "preregister_downstream_pairwise_reward_use", "runtime_gate_change_allowed": false, "status": "holdout_signal_consistent"}
  - `source_reward_ranker_ready`: true
  - `source_reward_target_contract_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_source_reward_pairwise_target_contract.json"
  - `source_reward_target_contract`: {"acceptance_constraints": {"min_cross_action_pair_rows": 50, "min_pair_rows": 100, "required_aggregate_status": "pairwise_ranker_signal", "required_cv_status": "pairwise_ranker_signal", "required_holdout_status": "holdout_signal_consistent", "runtime_gate_change_allowed": false}, "allowed_use": ["offline ranker/reward-model training experiments", "offline feature-target validation and ablation", "handoff planning for a future independent oracle/source contract"], "evidence": {"aggregate_decision": {"best_family": "hist_gradient_boosting", "blockers": [], "recommended_next": "cross_validate_pairwise_ranker_on_expanded_contract", "runtime_gate_change_allowed": false, "status": "pairwise_ranker_signal"}, "coverage": {"action_pair_counts": {"architect_general>coder_escalation": 14, "architect_general>frontdoor": 8, "coder_escalation>architect_general": 2, "frontdoor>architect_general": 134, "frontdoor>coder_escalation": 22}, "contrastive_groups": 158, "cross_action_pair_rows": 180, "pair_rows": 180, "same_action_pair_rows": 0, "skipped_no_contrast_groups": 144, "source_family_pair_counts": {"orchestrator_live_seed": 20, "seeding_eval": 160}, "source_record_groups": 302, "suite_pair_counts": {"coder": 8, "debugbench": 8, "general": 8, "gpqa": 8, "hotpotqa": 48, "instruction_precision": 44, "livecodebench": 8, "long_context": 8, "math": 8, "simpleqa": 18, "thinking": 14}, "unique_action_pairs": 5}, "cross_validation_decision": {"best_family": "hist_gradient_boosting", "blockers": [], "recommended_next": "resolve_independent_holdout_blockers_before_runtime_use", "runtime_gate_change_allowed": false, "status": "pairwise_ranker_signal"}, "cv_folds": 5, "diagnostic_boundary": {"binary_threshold": 0.5, "diagnostic_only": true, "independent_oracle": false, "interpretation": "Use this to test whether the candidate set contains enough within-task source-reward contrast. It is not an adopted independent reward oracle.", "label_status": "diagnostic_source_reward_passthrough", "recommended_next": "decide whether A9 should train on source-q-reward pairwise labels or build a new independent scorer/source contract before ranker use", "runtime_gate_change_allowed": false, "score_source": "source_q_reward_passthrough", "score_value_counts": {"-0.3": 60, "-0.5": 8, "0.0": 154, "0.1": 56, "0.363842889264226": 2, "0.5": 30, "1.0": 316}, "source_reward_passthrough": true}, "diagnostic_decision": {"min_cross_action_pairs": 50, "min_pairs": 100, "recommended_next": "decide whether A9 should train on source-q-reward pairwise labels or build a new independent scorer/source contract before ranker use", "runtime_gate_change_allowed": false, "status": "contract_ready"}, "families_requested": ["logistic_l2", "hist_gradient_boosting", "random_forest"], "holdout_decision": {"blockers": [], "eligible_holdouts": 3, "passing_holdouts": 3, "recommended_next": "preregister_downstream_pairwise_reward_use", "runtime_gate_change_allowed": false, "status": "holdout_signal_consistent"}, "ranker_input": {"action_pair_counts": {"architect_general>coder_escalation": 14, "architect_general>frontdoor": 8, "coder_escalation>architect_general": 2, "frontdoor>architect_general": 134, "frontdoor>coder_escalation": 22}, "cross_action_pair_rows": 180, "group_count": 158, "pair_rows": 180, "pairing_mode_counts": {"score_ordered": 180}, "same_action_pair_rows": 0, "source_family_pair_counts": {"orchestrator_live_seed": 20, "seeding_eval": 160}, "suite_pair_counts": {"coder": 8, "debugbench": 8, "general": 8, "gpqa": 8, "hotpotqa": 48, "instruction_precision": 44, "livecodebench": 8, "long_context": 8, "math": 8, "simpleqa": 18, "thinking": 14}}, "seeds": [42, 7, 13, 101, 2026], "source_reward_diagnostic_summary": "orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_source_reward_diagnostic_summary.json", "source_reward_ranker_summary": "orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_ranker_source_reward_diagnostic_summary.json", "split": {"group_disjoint": true, "test_split": 0.25}}, "forbidden_use": ["live routing decisions", "serve-time request gating", "online reward updates", "claiming independent oracle evidence", "production promotion without a separate deployment gate"], "generated_at": "2026-07-04T18:19:55.911523+00:00", "next_step": "Use this only as an offline A9 training target candidate; any live use requires a separate independent deployment gate.", "schema_version": "offline_reward_source_reward_pairwise_target_contract.v1", "status": "preregistered_offline_training_target", "target": {"diagnostic_only": false, "independent_oracle": false, "learning_target": "within_source_record_score_ordered_preference_from_source_q_reward", "name": "a9_source_q_reward_pairwise_training_target_v1", "pairwise_jsonl": "orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_source_reward_diagnostic.jsonl", "prompt_free": true, "runtime_gate_change_allowed": false, "score_source": "source_q_reward_passthrough", "source_reward_passthrough": true}}
  - `source_reward_target_preregistered`: true
  - `audit_target_ranker_summary_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_ranker_score_ordered_audit_target_expanded_summary.json"
  - `audit_target_ranker_input`: {"action_pair_counts": {"architect_general>architect_general": 735, "architect_general>coder_escalation": 30, "architect_general>frontdoor": 1523, "coder_escalation>architect_general": 9, "coder_escalation>coder_escalation": 35, "coder_escalation>frontdoor": 25, "frontdoor>architect_general": 2740, "frontdoor>coder_escalation": 21, "frontdoor>frontdoor": 1126}, "cross_action_pair_rows": 4348, "group_count": 1981, "pair_rows": 6244, "pairing_mode_counts": {"score_ordered": 6244}, "same_action_pair_rows": 1896, "source_family_pair_counts": {"orchestrator_live_seed": 104, "seeding_eval": 205, "three_way_eval": 5935}, "suite_pair_counts": {"agentic": 10, "coder": 543, "debugbench": 658, "general": 535, "gpqa": 961, "hotpotqa": 1114, "instruction_precision": 64, "livecodebench": 616, "long_context": 197, "math": 23, "mode_advantage": 134, "mode_advantage_hard": 501, "simpleqa": 470, "thinking": 418}}
  - `audit_target_ranker_aggregate_decision`: {"best_family": "hist_gradient_boosting", "blockers": [], "recommended_next": "cross_validate_pairwise_ranker_on_expanded_contract", "runtime_gate_change_allowed": false, "status": "pairwise_ranker_signal"}
  - `audit_target_ranker_holdout_decision`: {"blockers": ["source_family:orchestrator_live_seed:insufficient_pairwise_signal", "source_family:seeding_eval:insufficient_pairwise_signal", "suite:general:insufficient_pairwise_signal"], "eligible_holdouts": 16, "passing_holdouts": 13, "recommended_next": "collect_more_non_overlapping_cross_action_preferences", "runtime_gate_change_allowed": false, "status": "mixed_holdout_signal"}
  - `audit_target_direction_audit_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_audit_target_direction_audit.json"
  - `audit_target_direction_decision`: {"recommended_next": "collect non-overlapping cross-action preference rows for the listed collection_targets (balance both directions); re-run evaluate_offline_reward_pairwise_ranker.py holdouts after collection. Do NOT retune the absolute MLP/calibrator/pairwise family.", "runtime_gate_change_allowed": false, "status": "preference_coverage_gaps_found", "weak_strata": ["source_family:seeding_eval", "suite:agentic", "suite:general", "suite:hotpotqa", "suite:simpleqa"]}
  - `audit_target_collection_targets`: [{"action_pair": "coder_escalation>frontdoor", "current_direction_balance": 0.0, "current_rows": 2, "needs_direction": ["prefer coder_escalation"], "prefer_hi": 2, "prefer_lo": 0, "stratum_field": "source_family", "stratum_value": "seeding_eval", "suggested_min_rows": 20}, {"action_pair": "architect_general>coder_escalation", "current_direction_balance": 0.0, "current_rows": 1, "needs_direction": ["prefer architect_general"], "prefer_hi": 1, "prefer_lo": 0, "stratum_field": "suite", "stratum_value": "general", "suggested_min_rows": 20}, {"action_pair": "architect_general>frontdoor", "current_direction_balance": 0.0577, "current_rows": 918, "needs_direction": ["balance both directions"], "prefer_hi": 865, "prefer_lo": 53, "stratum_field": "suite", "stratum_value": "hotpotqa", "suggested_min_rows": 20}, {"action_pair": "architect_general>coder_escalation", "current_direction_balance": 0.1429, "current_rows": 14, "needs_direction": ["balance both directions"], "prefer_hi": 2, "prefer_lo": 12, "stratum_field": "suite", "stratum_value": "simpleqa", "suggested_min_rows": 20}]
  - `audit_target_collection_manifest_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_audit_target_collection_manifest.json"
  - `audit_target_collection_status`: {"autopilot_guard": {"active_processes": [], "process_pattern": "scripts/autopilot/autopilot.py start", "refusal_exit_code": 75}, "batch_count": 4, "blockers": [], "manifest_path": "/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_audit_target_collection_manifest.json", "manifest_schema_version": "offline_reward_pairwise_collection_window.v1", "post_collection_step_count": 7, "ready": true, "schema_version": "offline_reward_pairwise_collection_status.v1", "source_plan_decision": {"recommended_next": "add_more_source_records_for_failed_pairwise_holdout_strata", "runtime_gate_change_allowed": false, "status": "insufficient_non_overlapping_cross_action_candidates"}, "status": "ready", "warnings": []}
  - `autopilot_guard`: {"active_processes": [], "process_pattern": "scripts/autopilot/autopilot.py start", "refusal_exit_code": 75}
  - `blockers`: []
  - `warnings`: ["manifest has no runnable collection batches"]

### xmas_production_path

- Status: `ready`
- Summary: X-MAS production routing is enforce-ready.
- Details:
  - `config_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/classifier_config.yaml"
  - `candidate_table_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/xmas_winner_table.yaml"
  - `quiet_window_ready`: true
  - `quiet_window_blockers`: []
  - `mode`: "enforce"
  - `winner_table_path`: "orchestration/xmas_winner_table.yaml"
  - `require_complete_table`: true
  - `config_validation_errors`: []
  - `candidate_table_errors`: []
  - `candidate_table_ready`: true
  - `latest_ab_summary_path`: "/mnt/raid0/llm/epyc-orchestrator/benchmarks/results/runs/xmas_live_ab/20260703T213541Z-constrained-policy-v2/summary.json"
  - `latest_ab_results_path`: "/mnt/raid0/llm/epyc-orchestrator/benchmarks/results/runs/xmas_live_ab/20260703T213541Z-constrained-policy-v2/results.jsonl"
  - `latest_ab_decision_status`: "promote_candidate"
  - `latest_ab_score_delta`: 0.09999999999999998
  - `latest_ab_latency_ratio`: 0.9383678904318052
  - `latest_ab_blockers`: []
  - `latest_ab_policy`: "incumbent_constrained_cheapfirst_v2"
  - `required_ab_policy`: "incumbent_constrained_cheapfirst_v2"
  - `latest_ab_ready`: true

### eval_task_coverage

- Status: `attention`
- Summary: Eval-task coverage low_coverage: 3206/52210 stable qids (6.1406%), repeat_factor=10.2823x.
- Details:
  - `coverage_status`: "low_coverage"
  - `question_result_rows`: 32965
  - `distinct_journal_question_keys`: 3206
  - `pool_stable_question_keys`: 52210
  - `distinct_vs_pool_stable_upper_bound_pct`: 6.1406
  - `matched_pool_stable_pct`: 6.0927
  - `repeat_factor`: 10.2823
  - `interpretation`: "Fixed authority-core repetition is acceptable for paired safety evidence; planner-learning coverage is narrow if this is the dominant optimization signal."
  - `eval_bearing_trials`: 400
  - `trial_id_min`: 0
  - `trial_id_max`: 1336
  - `tier_coverage`: {"0": {"distinct_journal_question_keys": 10, "distinct_vs_pool_pct": 0.0, "eval_bearing_trials": 1, "pool_question_keys": 0, "question_result_rows": 10}, "1": {"distinct_journal_question_keys": 2560, "distinct_vs_pool_pct": 12.1138, "eval_bearing_trials": 377, "pool_question_keys": 21133, "question_result_rows": 22965}, "2": {"distinct_journal_question_keys": 843, "distinct_vs_pool_pct": 3.1612, "eval_bearing_trials": 19, "pool_question_keys": 26667, "question_result_rows": 9510}, "3": {"distinct_journal_question_keys": 160, "distinct_vs_pool_pct": 2.9461, "eval_bearing_trials": 3, "pool_question_keys": 5431, "question_result_rows": 480}}
  - `least_covered_non_sentinel_suites`: [{"distinct_qids": 15, "suite": "tool_use"}, {"distinct_qids": 36, "suite": "agentic"}, {"distinct_qids": 36, "suite": "skill_transfer"}, {"distinct_qids": 44, "suite": "long_context"}, {"distinct_qids": 49, "suite": "real_suite_v1"}, {"distinct_qids": 57, "suite": "mode_advantage_hard"}, {"distinct_qids": 85, "suite": "mode_advantage"}, {"distinct_qids": 165, "suite": "coder"}, {"distinct_qids": 184, "suite": "gpqa"}, {"distinct_qids": 188, "suite": "bigcodebench"}]
  - `pool_path`: "/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl"
  - `recommendation`: {"do_not_change_mid_w8": true, "lane_split": ["authority_core: fixed paired core for W4/W6/W8 promotion evidence", "exploration_coverage: rotating/advisory pool for planner learning", "promotion_holdout: fresh held-out T2 acceptance evidence"], "next_step": "Use this report as a guardrail before changing sampling policy; introduce any rotation behind a new instrument-era label."}

### p0_2_amendment_bundle_inputs

- Status: `attention`
- Summary: P0.2 amendment-bundle evidence inputs are attention: rate_axis=below_required, control_attestation=missing, eval_discriminability=low_coverage, t3_hard_lane=visible, ri10_canary=ready. Operator signing remains outside this report.
- Details:
  - `observe_only`: true
  - `operator_signing_required`: true
  - `rate_axis_status`: "below_required"
  - `rate_axis_latest_combined_E`: 0.91
  - `rate_axis_required_E`: 100.0
  - `rate_axis_seq_state`: "accumulating"
  - `rate_axis_fresh_eval`: false
  - `w8_promotion_status`: "blocked"
  - `control_attestation_status`: "missing"
  - `control_attestation_trial_id`: null
  - `control_attestation_enabled`: null
  - `control_attestation_eligible_for_evidence`: null
  - `control_attestation_controls_seen`: null
  - `control_attestation_suites`: null
  - `control_attestation_reason`: "no journaled control attestation found"
  - `eval_discriminability_status`: "low_coverage"
  - `eval_task_coverage_pct`: 6.1406
  - `eval_task_repeat_factor`: 10.2823
  - `t3_hard_lane_status`: "visible"
  - `t3_eval_bearing_trials`: 3
  - `t3_question_result_rows`: 480
  - `t3_distinct_journal_question_keys`: 160
  - `t3_pool_question_keys`: 5431
  - `t3_distinct_vs_pool_pct`: 2.9461
  - `ri10_canary_status`: "ready"
  - `ri10_telemetry_collection_blocker`: "decision_ready"
  - `ri10_telemetry_collection_reason`: "current high-risk telemetry has decision-grade canary arm coverage"
  - `evidence_gaps`: ["rate_axis_below_required", "control_attestation_missing", "eval_discriminability_low_coverage"]

### tool_use_activation

- Status: `attention`
- Summary: Tool-use planner hints are visible, but the live sentinel/telemetry lane is not fully active.
- Details:
  - `autopilot_pid`: 1039446
  - `api_pid`: 2352519
  - `autopilot_tool_sentinels_enabled`: false
  - `api_tool_sentinels_enabled`: true
  - `api_tools_enabled`: true
  - `api_repl_enabled`: true
  - `api_structured_tool_output_enabled`: true
  - `activation_gaps`: ["autopilot_env_missing_AUTOPILOT_TOOL_SENTINELS"]
  - `latest_tool_metrics`: {"mean_tools_used": 0.06, "per_suite_tool_helpfulness": {}, "tool_helpfulness": NaN, "tool_name_counts": null, "tool_use_rate": 0.06, "total_tool_calls": 3, "trial_id": 1335}
  - `recent_tool_metrics`: {"evaluated_rows": 10, "latest_nonzero_tool_metrics": {"mean_tools_used": 0.06, "per_suite_tool_helpfulness": {}, "tool_helpfulness": NaN, "tool_name_counts": null, "tool_use_rate": 0.06, "total_tool_calls": 3, "trial_id": 1335}, "nonzero_rows": 10, "total_tool_calls": 43, "trial_ids": [1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335], "window": 10}
  - `config_attest_error`: null
