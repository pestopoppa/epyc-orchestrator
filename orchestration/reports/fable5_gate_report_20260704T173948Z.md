# Fable5 Gate Report

Ready: true

## Next Actions

### collect_w8_promotion_eval_evidence

- Priority: `P0`
- Status: `active`
- Reason: W4/W6 authority is restart-ready; W8 needs a new keepable candidate before replay/promotion evidence can accrue.
- Evidence:
  - `w8_promotion_status`: "none"
  - `open_requirements`: ["combined_E_below_required", "fresh_promotion_eval_required", "seq_confirmation_required"]
  - `pending_candidate`: null
  - `pending_source_trial_id`: null
  - `pending_attempts`: null
  - `last_blocked_trial_id`: null
  - `last_blocked_candidate`: null
  - `last_blocked_reason`: null
  - `latest_seq_trial_id`: 1131
  - `latest_candidate`: "7c5026202c4aa284"
  - `latest_combined_E`: 0.97769
  - `latest_required_E`: 100.0
  - `latest_confirmed`: false
  - `latest_fresh_eval`: false
  - `latest_seq_state`: "accumulating"
  - `latest_baseline_reference_state`: "fresh"
  - `baseline_reference_last_forced_trial_id`: 1107
  - `baseline_reference_last_forced_reason`: "10 trusted profile trials since baseline reference draw"
  - `baseline_reference_blocked_reason`: "Auto-blacklisted: 3 consecutive failures ending at trial 1117"
  - `candidate_generation_required`: true
  - `candidate_status_counts`: {"excluded": 6, "refuted": 3, "reverted": 43}
  - `recent_active_candidates`: []
  - `replay_concentration_warning`: false
  - `replay_concentration_reason`: null
  - `replay_top_active_candidate`: null
  - `replay_top_active_attempt_share`: null
  - `replay_stale_accumulating_count`: 0
- Command: `cd /mnt/raid0/llm/epyc-orchestrator && uv run python scripts/autopilot/w8_promotion_trajectory_report.py --journal orchestration`
- Follow-up: `uv run python scripts/autopilot/fable5_gate_report.py --json --strict`

### revise_a9_reward_oracle_or_reference_source

- Priority: `P1`
- Status: `active`
- Reason: A9 clean-window acquisition is exhausted for the current reference_token_coverage oracle; remaining instruction-precision targets have no reference text for that scorer.
- Requires: materially different scorer/feature design or a reference-bearing instruction-following source
- Follow-up: `Do not rerun the current collection script; regenerate A9 only after the oracle/source contract changes.`

## Sections

### phase_health

- Status: `ready`
- Summary: AutoPilot phase heartbeat is active at trial 1132 / paused.
- Details:
  - `ok`: true
  - `status`: "active"
  - `trial_id`: 1132
  - `phase`: "paused"
  - `action_type`: null
  - `heartbeat_age_s`: 0.8719620704650879
  - `pid`: 1466899
  - `pid_alive`: true
  - `process_started_at_s`: 1783185056.0
  - `require_current_code`: true
  - `code_stale`: false
  - `code_stale_paths`: []
  - `eval_label`: null
  - `eval_completed_questions`: null
  - `eval_total_questions`: null
  - `eval_correct_questions`: null
  - `eval_correct_pct`: null
  - `eval_concurrency`: null
  - `planner_hints_enabled`: true
  - `seq_verdict_enabled`: true
  - `w6_audit_accrual_enabled`: true
  - `w6_audit_shadow_only`: true
  - `w6_audit_n`: "10"
  - `w6_audit_every_n_trials`: "1"
  - `autopilot_planner_timeout`: "600"

### w4_w6_restart_cutover

- Status: `ready`
- Summary: W4/W6 strict restart cutover is ready.
- Details:
  - `restart_ready`: true
  - `archive_source_surface_ok`: true
  - `archive_source_surface_count`: 6
  - `archive_source_surface_failed_count`: 0
  - `seq_cutover_ready`: true
  - `seq_trusted_vector_trials`: 255
  - `seq_min_trusted_vector_trials`: 120
  - `seq_trusted_vector_trials_remaining`: 0
  - `seq_shadow_rows`: 178
  - `seq_min_shadow_rows`: 30
  - `seq_shadow_rows_remaining`: 0
  - `w8_promotion_status`: "none"
  - `w8_open_requirements`: ["combined_E_below_required", "fresh_promotion_eval_required", "seq_confirmation_required"]
  - `w8_pending_candidate`: null
  - `w8_pending_source_trial_id`: null
  - `w8_pending_attempts`: null
  - `w8_last_finalized_trial_id`: null
  - `w8_last_finalized_candidate`: null
  - `w8_last_finalized_combined_E`: null
  - `w8_last_finalized_delta_excludes_regression`: null
  - `w8_last_blocked_trial_id`: null
  - `w8_last_blocked_candidate`: null
  - `w8_last_blocked_reason`: null
  - `w8_latest_seq_trial_id`: 1131
  - `w8_latest_candidate`: "7c5026202c4aa284"
  - `w8_latest_combined_E`: 0.97769
  - `w8_latest_required_E`: 100.0
  - `w8_latest_confirmed`: false
  - `w8_latest_seq_state`: "accumulating"
  - `w8_latest_baseline_reference_state`: "fresh"
  - `w8_latest_fresh_eval`: false
  - `w8_baseline_reference_last_forced_trial_id`: 1107
  - `w8_baseline_reference_last_forced_reason`: "10 trusted profile trials since baseline reference draw"
  - `w8_baseline_reference_last_forced_stale`: false
  - `w8_baseline_reference_blocked_trial_id`: 1124
  - `w8_baseline_reference_blocked_reason`: "Auto-blacklisted: 3 consecutive failures ending at trial 1117"
  - `w6_audit_cutover_ready`: true
  - `w6_audited_trial_count`: 92
  - `w6_min_audited_trials`: 30
  - `w6_audited_trial_count_remaining`: 0
  - `w6_alarm_clearance_clean_trials_required`: 0
  - `w6_raw_audited_trial_count`: 92
  - `w6_trusted_audited_trial_count`: 92
  - `w6_untrusted_audited_trial_count`: 0
  - `w6_untrusted_audited_trial_ids`: []
  - `w6_gaming_alarm`: false
  - `w6_core_inflation_warning`: false
  - `w6_era_excluded_gaming_event_count`: 0
  - `w6_fence_governance_status`: "no_excluded_gaming_events"
  - `w6_fence_governance_blockers`: []
  - `w6_potential_overfit_divergences`: 0
  - `cutover_horizon_clean_trials_remaining`: 0
  - `cutover_horizon_blocker`: null
  - `cutover_horizon_components`: {"seq_shadow_rows": 0, "seq_trusted_vectors": 0, "w6_alarm_clearance": 0, "w6_audited_trials": 0}
  - `baseline_seed_append_ready`: false
  - `baseline_seed_append_required`: false
  - `baseline_seed_append_expect_trial_counter`: null
  - `baseline_seed_append_expect_journal_max_trial_id`: null
  - `durable_journal_max_trial_id`: 1131
  - `state_trial_counter`: 1132
  - `snapshot_restart_readiness`: "tail_fold_ready"
  - `snapshot_payload_journal_max_trial_id`: 1131

### w8_promotion_trajectory

- Status: `ready`
- Summary: W8 replay trajectory has no concentration warning.
- Details:
  - `status`: "evidence_bound"
  - `ok`: false
  - `latest_trial_id`: 1131
  - `snapshot_count`: 176
  - `candidate_count`: 52
  - `status_counts`: {"excluded": 6, "refuted": 3, "reverted": 43}
  - `open_requirements`: ["combined_E_below_required", "fresh_promotion_eval_required", "no_recent_multi_observation_accumulating_candidate", "seq_confirmation_required"]
  - `candidate_generation_required`: true
  - `recent_active_candidates`: []
  - `stale_accumulating_candidate_count`: 0
  - `replay_concentration`: {"active_recent_attempts": {}, "active_recent_candidate_count": 0, "single_observation_count": 0, "stale_accumulating_count": 0, "top_active_attempt_share": null, "top_active_attempts": 0, "top_active_candidate": null, "total_active_recent_attempts": 0, "warning": false, "warning_reason": null}

### ds_e1_dynamic_stack

- Status: `ready`
- Summary: DS-E1 dynamic-stack packet is decision-ready.
- Details:
  - `ready_for_profile_decision`: true
  - `generated_at`: "2026-07-04T17:39:54Z"
  - `section_statuses`: {"contention_matrix": "ready", "ds5_roster_manifest": "ready", "kv_size_measurements": "ready", "ri10_canary": "ready", "stack_roster": "ready"}
  - `kv_required_measurements`: {"architect_general": [2048, 8192], "frontdoor": [2048, 8192, 32768], "ingest_long_context": [2048, 8192, 32768], "worker_general": [2048, 8192]}
  - `kv_observed_measurements`: {"architect_general": [2048, 8192], "frontdoor": [2048, 8192, 32768], "ingest_long_context": [2048, 8192, 32768], "worker_general": [2048, 8192]}
  - `kv_missing_measurements`: null
  - `kv_expected_csv_columns`: null
  - `kv_candidate_paths`: ["/mnt/raid0/llm/epyc-inference-research/data/dynamic_stack/ds_e1_kv_measurements_20260704T163852Z/kv_measurements.csv"]
  - `kv_searched_globs`: ["orchestration/reports/ds_e1*kv*", "orchestration/reports/dynamic_stack*kv*", "../epyc-inference-research/data/dynamic_stack/**/kv*", "../epyc-inference-research/data/kv_measurements/**"]
  - `clean_window_ready`: false
  - `clean_window_blockers`: ["active AutoPilot process(es): 1466899 .venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000", "live llama-server process(es): 1420537 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 8070 -np 1 -c 32768 -t 96 -ub 8192 --flash-attn on --jinja -ctk q8_0 -ctv q8_0 --mlock --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/frontdoor; 1420883 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16 -ub 512 --no-mmap --reasoning off --jinja --host 127.0.0.1 --port 8072 -np 1 -c 16384 -t 96 -ctk q8_0 -ctv q8_0 --flash-attn on; 1421205 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf --host 127.0.0.1 --port 8083 -np 2 -c 16384 -t 96 -ub 8192 --flash-attn on --jinja -ctk q4_0 -ctv f16 --mlock --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/architect_general; 1421589 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf --host 127.0.0.1 --port 8085 -np 1 -c 32768 -t 96 -ub 8192 --flash-attn on --jinja -ctk q4_0 -ctv q4_0 --mlock --override-kv qwen3next.expert_used_count=int:4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/ingest_long_context; ... +8 more"]
  - `ri10_telemetry_collection_blocker`: "decision_ready"
  - `ri10_telemetry_collection_reason`: "current high-risk telemetry has decision-grade canary arm coverage"
  - `ri10_canary_role_sample_deficit`: 0
  - `ri10_canary_arm_volume_deficit`: 0
  - `ri10_canary_arm_balance_deficits`: {"enforce_high_risk": 0, "shadow_high_risk": 0}
  - `ri10_high_risk_by_role_current`: {"frontdoor": 20, "worker_general": 32, "worker_vision": 25}
  - `ri10_canary_role_high_risk_by_role_current`: {"frontdoor": 20, "worker_general": 32, "worker_vision": 25}
  - `ri10_canary_arm_counts_current`: {"enforce_high_risk": 29, "shadow_high_risk": 48}
  - `ri10_canary_arm_counts_by_role_current`: {"frontdoor": {"enforce_high_risk": 9, "shadow_high_risk": 11}, "worker_general": {"enforce_high_risk": 10, "shadow_high_risk": 22}, "worker_vision": {"enforce_high_risk": 10, "shadow_high_risk": 15}}

### a9_pairwise_collection

- Status: `attention`
- Summary: A9 pairwise source-acquisition window is no_runnable_batches with 0 batch(es).
- Details:
  - `ready`: false
  - `status`: "no_runnable_batches"
  - `manifest_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_expanded_gap_collection_manifest.json"
  - `manifest_schema_version`: "offline_reward_pairwise_collection_window.v1"
  - `source_plan_decision`: {"recommended_next": "score_selected_candidates_and_rebuild_pairwise_contract", "runtime_gate_change_allowed": false, "status": "expansion_plan_ready"}
  - `batch_count`: 0
  - `post_collection_step_count`: 7
  - `autopilot_guard`: {"active_processes": ["1466899 .venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000"], "process_pattern": "scripts/autopilot/autopilot.py start", "refusal_exit_code": 75}
  - `blockers`: []
  - `warnings`: ["manifest has no runnable collection batches"]

### xmas_production_path

- Status: `ready`
- Summary: X-MAS production routing is enforce-ready.
- Details:
  - `config_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/classifier_config.yaml"
  - `candidate_table_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/xmas_winner_table.yaml"
  - `quiet_window_ready`: false
  - `quiet_window_blockers`: ["active AutoPilot process(es): 1466899 .venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000"]
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

### tool_use_activation

- Status: `ready`
- Summary: Tool-use planner hints are backed by active API/AutoPilot sentinel telemetry.
- Details:
  - `autopilot_pid`: 1466899
  - `api_pid`: 313881
  - `autopilot_tool_sentinels_enabled`: true
  - `api_tool_sentinels_enabled`: true
  - `api_tools_enabled`: true
  - `api_repl_enabled`: true
  - `api_structured_tool_output_enabled`: true
  - `activation_gaps`: []
  - `latest_tool_metrics`: {"mean_tools_used": 0.10909090909090909, "per_suite_tool_helpfulness": {}, "tool_helpfulness": NaN, "tool_name_counts": null, "tool_use_rate": 0.09090909090909091, "total_tool_calls": 6, "trial_id": 1131}
  - `recent_tool_metrics`: {"evaluated_rows": 10, "latest_nonzero_tool_metrics": {"mean_tools_used": 0.10909090909090909, "per_suite_tool_helpfulness": {}, "tool_helpfulness": NaN, "tool_name_counts": null, "tool_use_rate": 0.09090909090909091, "total_tool_calls": 6, "trial_id": 1131}, "nonzero_rows": 10, "total_tool_calls": 67, "trial_ids": [1117, 1118, 1119, 1120, 1123, 1124, 1125, 1126, 1128, 1131], "window": 10}
  - `config_attest_error`: null
