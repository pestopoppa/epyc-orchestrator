# Lab Review Queue Report

- generated_at: `2026-07-06T13:14:53+00:00`
- status: `attention`
- pending_reviews: `8`
- pending_active_safe: `8`
- pending_review_candidates: `0`
- queue_dir: `/mnt/raid0/llm/epyc-orchestrator/orchestration/lab_review_queue`

## Pending Items

| job_id | run_id | class | stage | next_reviewer | output |
|---|---|---|---|---|---|
| autopilot_authority_watch | `autopilot_authority_watch-20260706T115634Z-52d1a015` | active_safe_deterministic | shadow | operator | `autopilot_authority_watch/autopilot_authority_watch-20260706T115634Z-52d1a015/output.json` |
| autopilot_restart_advisor | `autopilot_restart_advisor-20260706T115635Z-2a913962` | active_safe_deterministic | shadow | operator | `autopilot_restart_advisor/autopilot_restart_advisor-20260706T115635Z-2a913962/output.json` |
| autopilot_outcome_progress_watch | `autopilot_outcome_progress_watch-20260706T115635Z-37413a28` | active_safe_deterministic | shadow | operator | `autopilot_outcome_progress_watch/autopilot_outcome_progress_watch-20260706T115635Z-37413a28/output.json` |
| quiet_window_queue_watch | `quiet_window_queue_watch-20260706T115636Z-05ddb4d7` | active_safe_deterministic | shadow | operator | `quiet_window_queue_watch/quiet_window_queue_watch-20260706T115636Z-05ddb4d7/output.json` |
| lab_review_queue_watch | `lab_review_queue_watch-20260706T115636Z-6b509d96` | active_safe_deterministic | shadow | operator | `lab_review_queue_watch/lab_review_queue_watch-20260706T115636Z-6b509d96/output.json` |
| quiet_window_lab_plan_watch | `quiet_window_lab_plan_watch-20260706T115636Z-5d733b97` | active_safe_deterministic | shadow | operator | `quiet_window_lab_plan_watch/quiet_window_lab_plan_watch-20260706T115636Z-5d733b97/output.json` |
| autopilot_planner_provider_watch | `autopilot_planner_provider_watch-20260706T115823Z-ab6d7cb7` | active_safe_deterministic | shadow | operator | `autopilot_planner_provider_watch/autopilot_planner_provider_watch-20260706T115823Z-ab6d7cb7/output.json` |
| lab_review_queue_watch | `lab_review_queue_watch-20260706T121233Z-37bc5b88` | active_safe_deterministic | shadow | operator | `lab_review_queue_watch/lab_review_queue_watch-20260706T121233Z-37bc5b88/output.json` |

## Review Batch Template

Edit `verdict`, `confidence`, and `notes`, then pass the JSONL to `scripts/lab/apply_review_batch.py`.

```jsonl
{"confidence": null, "job_id": "autopilot_authority_watch", "local_output": "autopilot_authority_watch/autopilot_authority_watch-20260706T115634Z-52d1a015/output.json", "notes": "", "reference_output": null, "reviewer": "operator", "run_id": "autopilot_authority_watch-20260706T115634Z-52d1a015", "schema_version": "lab_review_batch.v1", "stage": "shadow", "verdict": "<accept|reject>"}
{"confidence": null, "job_id": "autopilot_restart_advisor", "local_output": "autopilot_restart_advisor/autopilot_restart_advisor-20260706T115635Z-2a913962/output.json", "notes": "", "reference_output": null, "reviewer": "operator", "run_id": "autopilot_restart_advisor-20260706T115635Z-2a913962", "schema_version": "lab_review_batch.v1", "stage": "shadow", "verdict": "<accept|reject>"}
{"confidence": null, "job_id": "autopilot_outcome_progress_watch", "local_output": "autopilot_outcome_progress_watch/autopilot_outcome_progress_watch-20260706T115635Z-37413a28/output.json", "notes": "", "reference_output": null, "reviewer": "operator", "run_id": "autopilot_outcome_progress_watch-20260706T115635Z-37413a28", "schema_version": "lab_review_batch.v1", "stage": "shadow", "verdict": "<accept|reject>"}
{"confidence": null, "job_id": "quiet_window_queue_watch", "local_output": "quiet_window_queue_watch/quiet_window_queue_watch-20260706T115636Z-05ddb4d7/output.json", "notes": "", "reference_output": null, "reviewer": "operator", "run_id": "quiet_window_queue_watch-20260706T115636Z-05ddb4d7", "schema_version": "lab_review_batch.v1", "stage": "shadow", "verdict": "<accept|reject>"}
{"confidence": null, "job_id": "lab_review_queue_watch", "local_output": "lab_review_queue_watch/lab_review_queue_watch-20260706T115636Z-6b509d96/output.json", "notes": "", "reference_output": null, "reviewer": "operator", "run_id": "lab_review_queue_watch-20260706T115636Z-6b509d96", "schema_version": "lab_review_batch.v1", "stage": "shadow", "verdict": "<accept|reject>"}
{"confidence": null, "job_id": "quiet_window_lab_plan_watch", "local_output": "quiet_window_lab_plan_watch/quiet_window_lab_plan_watch-20260706T115636Z-5d733b97/output.json", "notes": "", "reference_output": null, "reviewer": "operator", "run_id": "quiet_window_lab_plan_watch-20260706T115636Z-5d733b97", "schema_version": "lab_review_batch.v1", "stage": "shadow", "verdict": "<accept|reject>"}
{"confidence": null, "job_id": "autopilot_planner_provider_watch", "local_output": "autopilot_planner_provider_watch/autopilot_planner_provider_watch-20260706T115823Z-ab6d7cb7/output.json", "notes": "", "reference_output": null, "reviewer": "operator", "run_id": "autopilot_planner_provider_watch-20260706T115823Z-ab6d7cb7", "schema_version": "lab_review_batch.v1", "stage": "shadow", "verdict": "<accept|reject>"}
{"confidence": null, "job_id": "lab_review_queue_watch", "local_output": "lab_review_queue_watch/lab_review_queue_watch-20260706T121233Z-37bc5b88/output.json", "notes": "", "reference_output": null, "reviewer": "operator", "run_id": "lab_review_queue_watch-20260706T121233Z-37bc5b88", "schema_version": "lab_review_batch.v1", "stage": "shadow", "verdict": "<accept|reject>"}
```
