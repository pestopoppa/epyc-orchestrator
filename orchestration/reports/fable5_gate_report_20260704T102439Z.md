# Fable5 Gate Report

Ready: false

## Blockers

- ds_e1_dynamic_stack: ri10_canary: RI-10 raw high-risk sample-count coverage exists, but configured canary_roles have insufficient current high-risk rows (only 20 current high-risk row(s) matched configured canary_roles; gate requires 50).
- ds_e1_dynamic_stack: kv_size_measurements: No direct DS-E1 production KV-size measurement series was found.
- a9_pairwise_collection: active AutoPilot process(es): 3220616 uv run python scripts/autopilot/autopilot.py start --max-trials 2000; 3220621 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000

## Next Actions

### collect_w8_promotion_eval_evidence

- Priority: `P0`
- Status: `active`
- Reason: W4/W6 authority is restart-ready; W8 still needs live promotion-eval finalization evidence before closing the tail.
- Evidence:
  - `w8_promotion_status`: "none"
  - `open_requirements`: ["combined_E_below_required", "fresh_promotion_eval_required", "seq_confirmation_required"]
  - `pending_candidate`: null
  - `pending_source_trial_id`: null
  - `pending_attempts`: null
  - `last_blocked_trial_id`: null
  - `last_blocked_candidate`: null
  - `last_blocked_reason`: null
  - `latest_seq_trial_id`: 1116
  - `latest_candidate`: "f159d9ca6b1ff453"
  - `latest_combined_E`: 0.673803
  - `latest_required_E`: 100.0
  - `latest_confirmed`: false
  - `latest_fresh_eval`: false
  - `latest_seq_state`: "accumulating"
  - `latest_baseline_reference_state`: "fresh"
  - `baseline_reference_last_forced_trial_id`: 1107
  - `baseline_reference_last_forced_reason`: "10 trusted profile trials since baseline reference draw"
  - `baseline_reference_blocked_reason`: null
  - `replay_concentration_warning`: false
  - `replay_concentration_reason`: null
  - `replay_top_active_candidate`: null
  - `replay_top_active_attempt_share`: null
  - `replay_stale_accumulating_count`: 0
- Command: `cd /mnt/raid0/llm/epyc-orchestrator && uv run python scripts/autopilot/w8_promotion_trajectory_report.py --journal orchestration`
- Follow-up: `uv run python scripts/autopilot/fable5_gate_report.py --json --strict`

### run_ds_e1_kv_measurements

- Priority: `P0`
- Status: `blocked`
- Reason: DS-E1 cannot decide DS-7/DS-6 profiles until direct production KV-size rows exist.
- Requires: attested clean window with AutoPilot and live llama-server processes stopped/coordinated
- Blocked by:
  - active AutoPilot process(es): 3220616 uv run python scripts/autopilot/autopilot.py start --max-trials 2000; 3220621 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000
  - live llama-server process(es): 2073983 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 8070 -np 1 -c 32768 -t 96 -ub 8192 --flash-attn on --jinja -ctk q8_0 -ctv q8_0 --mlock --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/frontdoor; 2074290 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 8080 -np 1 -c 32768 -t 48 -ub 8192 --flash-attn on --jinja -ctk q8_0 -ctv q8_0 --mlock --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/frontdoor; 2074558 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 8180 -np 1 -c 32768 -t 48 -ub 8192 --flash-attn on --jinja -ctk q8_0 -ctv q8_0 --mlock --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/frontdoor; 2074818 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 8280 -np 1 -c 32768 -t 48 -ub 8192 --flash-attn on --jinja -ctk q8_0 -ctv q8_0 --mlock --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/frontdoor; ... +24 more
- Command: `cd /mnt/raid0/llm/epyc-inference-research && scripts/benchmark/ds_e1_kv_measurements.sh --execute`
- Follow-up: `cd /mnt/raid0/llm/epyc-orchestrator && uv run python scripts/server/dynamic_stack_evidence_packet.py --output orchestration/reports/ds_e1_evidence_packet_$(date -u +%Y%m%dT%H%M%SZ).md --strict`

### collect_ri10_canary_arm_telemetry

- Priority: `P0`
- Status: `active`
- Reason: RI-10 has raw high-risk samples, but arm-attributed canary telemetry is not yet decision-grade.
- Command: `uv run python scripts/analysis/ri10_canary_sample_report.py`

### run_a9_pairwise_collection_window

- Priority: `P1`
- Status: `blocked`
- Reason: A9 offline reward-oracle pairwise holdouts need the guarded priority-0/1 collection window before another pairwise contract rebuild.
- Requires: coordinated clean window; collection script refuses active AutoPilot
- Blocked by:
  - active AutoPilot process(es): 3220616 uv run python scripts/autopilot/autopilot.py start --max-trials 2000; 3220621 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000
- Command: `cd /mnt/raid0/llm/epyc-orchestrator && orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/collect_offline_reward_pairwise_expanded_gap.sh`
- Follow-up: `cd /mnt/raid0/llm/epyc-orchestrator && uv run python scripts/graph_router/offline_reward_pairwise_collection_status.py`

## Sections

### phase_health

- Status: `ready`
- Summary: AutoPilot phase heartbeat is active at trial 1117 / dispatch_action.
- Details:
  - `ok`: true
  - `status`: "active"
  - `trial_id`: 1117
  - `phase`: "dispatch_action"
  - `action_type`: "seed_batch"
  - `heartbeat_age_s`: 2.770852565765381
  - `pid`: 3220621
  - `pid_alive`: true
  - `process_started_at_s`: 1783158819.1
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
  - `seq_trusted_vector_trials`: 245
  - `seq_min_trusted_vector_trials`: 120
  - `seq_trusted_vector_trials_remaining`: 0
  - `seq_shadow_rows`: 168
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
  - `w8_latest_seq_trial_id`: 1116
  - `w8_latest_candidate`: "f159d9ca6b1ff453"
  - `w8_latest_combined_E`: 0.673803
  - `w8_latest_required_E`: 100.0
  - `w8_latest_confirmed`: false
  - `w8_latest_seq_state`: "accumulating"
  - `w8_latest_baseline_reference_state`: "fresh"
  - `w8_latest_fresh_eval`: false
  - `w8_baseline_reference_last_forced_trial_id`: 1107
  - `w8_baseline_reference_last_forced_reason`: "10 trusted profile trials since baseline reference draw"
  - `w8_baseline_reference_last_forced_stale`: false
  - `w8_baseline_reference_blocked_trial_id`: null
  - `w8_baseline_reference_blocked_reason`: null
  - `w6_audit_cutover_ready`: true
  - `w6_audited_trial_count`: 82
  - `w6_min_audited_trials`: 30
  - `w6_audited_trial_count_remaining`: 0
  - `w6_alarm_clearance_clean_trials_required`: 0
  - `w6_raw_audited_trial_count`: 82
  - `w6_trusted_audited_trial_count`: 82
  - `w6_untrusted_audited_trial_count`: 0
  - `w6_untrusted_audited_trial_ids`: []
  - `w6_gaming_alarm`: false
  - `w6_potential_overfit_divergences`: 0
  - `cutover_horizon_clean_trials_remaining`: 0
  - `cutover_horizon_blocker`: null
  - `cutover_horizon_components`: {"seq_shadow_rows": 0, "seq_trusted_vectors": 0, "w6_alarm_clearance": 0, "w6_audited_trials": 0}
  - `baseline_seed_append_ready`: false
  - `baseline_seed_append_required`: false
  - `baseline_seed_append_expect_trial_counter`: null
  - `baseline_seed_append_expect_journal_max_trial_id`: null
  - `durable_journal_max_trial_id`: 1116
  - `state_trial_counter`: 1117
  - `snapshot_restart_readiness`: "tail_fold_ready"
  - `snapshot_payload_journal_max_trial_id`: 1116

### w8_promotion_trajectory

- Status: `ready`
- Summary: W8 replay trajectory has no concentration warning.
- Details:
  - `status`: "evidence_bound"
  - `ok`: false
  - `latest_trial_id`: 1116
  - `snapshot_count`: 166
  - `candidate_count`: 46
  - `status_counts`: {"excluded": 6, "refuted": 3, "reverted": 37}
  - `open_requirements`: ["combined_E_below_required", "fresh_promotion_eval_required", "no_recent_multi_observation_accumulating_candidate", "seq_confirmation_required"]
  - `recent_active_candidates`: []
  - `stale_accumulating_candidate_count`: 0
  - `replay_concentration`: {"active_recent_attempts": {}, "active_recent_candidate_count": 0, "single_observation_count": 0, "stale_accumulating_count": 0, "top_active_attempt_share": null, "top_active_attempts": 0, "top_active_candidate": null, "total_active_recent_attempts": 0, "warning": false, "warning_reason": null}

### ds_e1_dynamic_stack

- Status: `blocked`
- Summary: DS-E1 dynamic-stack packet is not decision-ready.
- Blockers:
  - ri10_canary: RI-10 raw high-risk sample-count coverage exists, but configured canary_roles have insufficient current high-risk rows (only 20 current high-risk row(s) matched configured canary_roles; gate requires 50).
  - kv_size_measurements: No direct DS-E1 production KV-size measurement series was found.
- Details:
  - `ready_for_profile_decision`: false
  - `generated_at`: "2026-07-04T10:24:40Z"
  - `section_statuses`: {"contention_matrix": "ready", "ds5_roster_manifest": "ready", "kv_size_measurements": "missing", "ri10_canary": "insufficient_data", "stack_roster": "ready"}
  - `kv_required_measurements`: {"architect_general": [2048, 8192], "frontdoor": [2048, 8192, 32768], "ingest_long_context": [2048, 8192, 32768], "worker_general": [2048, 8192]}
  - `kv_observed_measurements`: null
  - `kv_missing_measurements`: null
  - `kv_expected_csv_columns`: ["role", "model_id", "model_path", "context_length", "max_context", "status", "rss_load_mb", "rss_after_prefill_mb", "server_kv_size_mb", "prompt_tokens", "prompt_tps", "log_file", "notes"]
  - `kv_candidate_paths`: null
  - `kv_searched_globs`: ["orchestration/reports/ds_e1*kv*", "orchestration/reports/dynamic_stack*kv*", "../epyc-inference-research/data/dynamic_stack/**/kv*", "../epyc-inference-research/data/kv_measurements/**"]
  - `clean_window_ready`: false
  - `clean_window_blockers`: ["active AutoPilot process(es): 3220616 uv run python scripts/autopilot/autopilot.py start --max-trials 2000; 3220621 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000", "live llama-server process(es): 2073983 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 8070 -np 1 -c 32768 -t 96 -ub 8192 --flash-attn on --jinja -ctk q8_0 -ctv q8_0 --mlock --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/frontdoor; 2074290 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 8080 -np 1 -c 32768 -t 48 -ub 8192 --flash-attn on --jinja -ctk q8_0 -ctv q8_0 --mlock --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/frontdoor; 2074558 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 8180 -np 1 -c 32768 -t 48 -ub 8192 --flash-attn on --jinja -ctk q8_0 -ctv q8_0 --mlock --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/frontdoor; 2074818 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 8280 -np 1 -c 32768 -t 48 -ub 8192 --flash-attn on --jinja -ctk q8_0 -ctv q8_0 --mlock --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/frontdoor; ... +24 more"]

### a9_pairwise_collection

- Status: `blocked`
- Summary: A9 pairwise source-acquisition window is blocked with 1 batch(es).
- Blockers:
  - active AutoPilot process(es): 3220616 uv run python scripts/autopilot/autopilot.py start --max-trials 2000; 3220621 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000
- Details:
  - `ready`: false
  - `status`: "blocked"
  - `manifest_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_expanded_gap_collection_manifest.json"
  - `manifest_schema_version`: "offline_reward_pairwise_collection_window.v1"
  - `source_plan_decision`: {"recommended_next": "score_selected_candidates_and_rebuild_pairwise_contract", "runtime_gate_change_allowed": false, "status": "expansion_plan_ready"}
  - `batch_count`: 1
  - `post_collection_step_count`: 7
  - `autopilot_guard`: {"active_processes": ["3220616 uv run python scripts/autopilot/autopilot.py start --max-trials 2000", "3220621 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000"], "process_pattern": "scripts/autopilot/autopilot.py start", "refusal_exit_code": 75}
  - `blockers`: ["active AutoPilot process(es): 3220616 uv run python scripts/autopilot/autopilot.py start --max-trials 2000; 3220621 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000"]
  - `warnings`: []

### xmas_production_path

- Status: `ready`
- Summary: X-MAS production routing is enforce-ready.
- Details:
  - `config_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/classifier_config.yaml"
  - `candidate_table_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/xmas_winner_table.yaml"
  - `quiet_window_ready`: false
  - `quiet_window_blockers`: ["active AutoPilot process(es): 3220616 uv run python scripts/autopilot/autopilot.py start --max-trials 2000; 3220621 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000"]
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
  - `autopilot_pid`: 3220621
  - `api_pid`: 3255216
  - `autopilot_tool_sentinels_enabled`: true
  - `api_tool_sentinels_enabled`: true
  - `api_tools_enabled`: true
  - `api_repl_enabled`: true
  - `api_structured_tool_output_enabled`: true
  - `activation_gaps`: []
  - `latest_tool_metrics`: {"mean_tools_used": 0.0, "per_suite_tool_helpfulness": {}, "tool_helpfulness": NaN, "tool_name_counts": null, "tool_use_rate": 0.0, "total_tool_calls": 0, "trial_id": 1116}
  - `recent_tool_metrics`: {"evaluated_rows": 10, "latest_nonzero_tool_metrics": {"mean_tools_used": 0.14814814814814814, "per_suite_tool_helpfulness": {}, "tool_helpfulness": NaN, "tool_name_counts": null, "tool_use_rate": 0.07407407407407407, "total_tool_calls": 8, "trial_id": 1115}, "nonzero_rows": 4, "total_tool_calls": 26, "trial_ids": [1103, 1104, 1105, 1106, 1107, 1111, 1112, 1114, 1115, 1116], "window": 10}
  - `config_attest_error`: null
