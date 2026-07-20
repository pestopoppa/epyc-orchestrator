# RM-3 C-CRAB P-REV-1 Screening Dry-Run

Date: 2026-07-19

No inference ran. Queue is bound to the same 48-row C-CRAB P-REV-1 row-id allowlist used by RM-2.fast/RM-2.next.

Jobs: `64`; pairings considered: `6525`; priority-truncated: `6461`; per-pairing n: `12`.

The full pool-generation JSON is omitted from the committed artifact to avoid
large generated churn; regenerate it with `scripts/analysis/reviewer_pool_gen.py`
against the registry hash recorded in `resolved_queue_code_top64.json`.

Corpus slice: `{'content_sha256': '0391166cf53225a1c54d95cc48c9672121cb5feb009039a32221dc35d0fd22c3', 'corpus_id': 'nearmiss-v1', 'domain': 'code', 'gate_worthy_multi_oracle': 7026, 'n_rows': 4796, 'row_id_filter_n': 48, 'row_id_filter_path': '/mnt/raid0/llm/epyc-inference-research/docs/data/rm2_reviewer_slate_ccrab_p_rev1_matched_row_ids_20260719.txt', 'row_id_filter_sha256': '3233a350b20a76f9e7f70c676158dc257166f660fca6c1bcfd9026a2d71ec57a', 'schema_version': 'nearmiss_corpus_row.v1'}`

Transport: `{'request_priority': 'background', 'transport': 'placement_queue', 'uses_chat_endpoint': False, 'workload_class': 'eval_batch'}`

| rank | pairing | architect | reviewer | grader | anchor | cross-family | staged | fits |
|---:|---|---|---|---|---|---|---|---|
| 0 | `architect_general__glm_52_ud_iq2m__toolrunner` | `architect_general` | `glm_52_ud_iq2m` | `toolrunner` | `A4` | `True` | `True` | `True` |
| 1 | `architect_general__qwen35_122b_q4km__toolrunner` | `architect_general` | `qwen35_122b_q4km` | `toolrunner` | `A3` | `False` | `False` | `True` |
| 2 | `architect_general__architect_general__toolrunner` | `architect_general` | `architect_general` | `toolrunner` | `A1` | `False` | `False` | `True` |
| 3 | `architect_general__deepseek_v4_flash_local_q4kexperts__toolrunner` | `architect_general` | `deepseek_v4_flash_local_q4kexperts` | `toolrunner` | `None` | `True` | `True` | `True` |
| 4 | `architect_general__gemma4_31b_q4km_mtp__toolrunner` | `architect_general` | `gemma4_31b_q4km_mtp` | `toolrunner` | `None` | `True` | `True` | `True` |
| 5 | `architect_general__hy3_angelslim_iq1m_mtp__toolrunner` | `architect_general` | `hy3_angelslim_iq1m_mtp` | `toolrunner` | `None` | `True` | `True` | `True` |
| 6 | `architect_general__minimax_m27_q8__toolrunner` | `architect_general` | `minimax_m27_q8` | `toolrunner` | `None` | `True` | `True` | `True` |
| 7 | `architect_hermes_4_70b__deepseek_v4_flash_local_q4kexperts__toolrunner` | `architect_hermes_4_70b` | `deepseek_v4_flash_local_q4kexperts` | `toolrunner` | `None` | `True` | `True` | `True` |
| 8 | `architect_hermes_4_70b__gemma4_31b_q4km_mtp__toolrunner` | `architect_hermes_4_70b` | `gemma4_31b_q4km_mtp` | `toolrunner` | `None` | `True` | `True` | `True` |
| 9 | `architect_hermes_4_70b__glm_52_ud_iq2m__toolrunner` | `architect_hermes_4_70b` | `glm_52_ud_iq2m` | `toolrunner` | `None` | `True` | `True` | `True` |
| 10 | `architect_hermes_4_70b__hy3_angelslim_iq1m_mtp__toolrunner` | `architect_hermes_4_70b` | `hy3_angelslim_iq1m_mtp` | `toolrunner` | `None` | `True` | `True` | `True` |
| 11 | `architect_hermes_4_70b__ingest_long_context__toolrunner` | `architect_hermes_4_70b` | `ingest_long_context` | `toolrunner` | `None` | `True` | `True` | `True` |
| 12 | `architect_hermes_4_70b__minimax_m27_q8__toolrunner` | `architect_hermes_4_70b` | `minimax_m27_q8` | `toolrunner` | `None` | `True` | `True` | `True` |
| 13 | `architect_hermes_4_70b__qwen36_27b_q8__toolrunner` | `architect_hermes_4_70b` | `qwen36_27b_q8` | `toolrunner` | `None` | `True` | `True` | `True` |
| 14 | `architect_qwen2_5_72b__deepseek_v4_flash_local_q4kexperts__toolrunner` | `architect_qwen2_5_72b` | `deepseek_v4_flash_local_q4kexperts` | `toolrunner` | `None` | `True` | `True` | `True` |
| 15 | `architect_qwen2_5_72b__gemma4_31b_q4km_mtp__toolrunner` | `architect_qwen2_5_72b` | `gemma4_31b_q4km_mtp` | `toolrunner` | `None` | `True` | `True` | `True` |
| 16 | `architect_qwen2_5_72b__glm_52_ud_iq2m__toolrunner` | `architect_qwen2_5_72b` | `glm_52_ud_iq2m` | `toolrunner` | `None` | `True` | `True` | `True` |
| 17 | `architect_qwen2_5_72b__hy3_angelslim_iq1m_mtp__toolrunner` | `architect_qwen2_5_72b` | `hy3_angelslim_iq1m_mtp` | `toolrunner` | `None` | `True` | `True` | `True` |
| 18 | `architect_qwen2_5_72b__minimax_m27_q8__toolrunner` | `architect_qwen2_5_72b` | `minimax_m27_q8` | `toolrunner` | `None` | `True` | `True` | `True` |
| 19 | `architect_qwen2_5_72b_q4_k_m__deepseek_v4_flash_local_q4kexperts__toolrunner` | `architect_qwen2_5_72b_q4_k_m` | `deepseek_v4_flash_local_q4kexperts` | `toolrunner` | `None` | `True` | `True` | `True` |

## Live Execution Status

not run; standalone runner is row-id-capable but live production RM-3 still needs grammar/schema reviewer path wiring and actions.py live-execution gap closure
