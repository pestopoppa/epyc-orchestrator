| scorer | source | category | n | FPR (fp/neg) | FNR (fn/pos) | acceptance | coverage |
|---|---|---|---|---|---|---|---|
| exact_match | gsm8k | clean_correct | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | gsm8k | clean_wrong | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | contradictory_finals | 400 | 0.000 (0/400) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | format_variants | 300 | - (0/0) | 0.000 (0/300) | 1.000 | 1.000 |
| exact_match | gsm8k | loose_float | 400 | 0.000 (0/300) | 0.000 (0/100) | 0.250 | 1.000 |
| exact_match | gsm8k | missing_marker | 200 | 0.500 (100/200) | - (0/0) | 0.500 | 1.000 |
| exact_match | gsm8k | partial_answer | 200 | 0.000 (0/200) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | wrong_then_correct | 200 | - (0/0) | 0.000 (0/200) | 1.000 | 1.000 |
| exact_match | math500 | clean_correct | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | math500 | clean_wrong | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | contradictory_finals | 400 | 0.000 (0/400) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | format_variants | 302 | - (0/0) | 0.007 (2/302) | 0.993 | 1.000 |
| exact_match | math500 | gold_prefix_collision | 6 | 0.000 (0/6) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | loose_float | 264 | 0.000 (0/198) | 0.000 (0/66) | 0.250 | 1.000 |
| exact_match | math500 | missing_marker | 200 | 0.500 (100/200) | - (0/0) | 0.500 | 1.000 |
| exact_match | math500 | partial_answer | 200 | 0.000 (0/200) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | wrong_then_correct | 200 | - (0/0) | 0.000 (0/200) | 1.000 | 1.000 |
| math_verify | gsm8k | clean_correct | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | gsm8k | clean_wrong | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | contradictory_finals | 400 | 0.000 (0/400) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | format_variants | 300 | - (0/0) | 0.000 (0/300) | 1.000 | 1.000 |
| math_verify | gsm8k | loose_float | 400 | 0.000 (0/300) | 0.000 (0/100) | 0.250 | 1.000 |
| math_verify | gsm8k | missing_marker | 200 | 0.000 (0/200) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | partial_answer | 200 | 0.000 (0/200) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | wrong_then_correct | 200 | - (0/0) | 0.000 (0/200) | 1.000 | 1.000 |
| math_verify | math500 | clean_correct | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | math500 | clean_wrong | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | contradictory_finals | 400 | 0.000 (0/400) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | format_variants | 302 | - (0/0) | 0.000 (0/302) | 1.000 | 1.000 |
| math_verify | math500 | gold_prefix_collision | 6 | 0.000 (0/6) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | loose_float | 264 | 0.000 (0/198) | 0.000 (0/66) | 0.250 | 1.000 |
| math_verify | math500 | missing_marker | 200 | 0.000 (0/200) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | partial_answer | 200 | 0.000 (0/200) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | wrong_then_correct | 200 | - (0/0) | 0.000 (0/200) | 1.000 | 1.000 |

## By template

| scorer | source | category | template | n | FPR (fp/neg) | FNR (fn/pos) | acceptance | coverage |
|---|---|---|---|---|---|---|---|---|
| exact_match | gsm8k | clean_correct | boxed | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | gsm8k | clean_wrong | boxed | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | contradictory_finals | gold_then_wrong_answer_tags | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | contradictory_finals | gold_then_wrong_boxed | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | contradictory_finals | wrong_then_gold_boxed | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | contradictory_finals | wrong_then_gold_hashes | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | format_variants | boxed_bare | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | gsm8k | format_variants | boxed_bold | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | gsm8k | format_variants | boxed_spaced | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | gsm8k | loose_float | drift_1e-3 | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | loose_float | drift_1e-5 | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | loose_float | drift_1e-7 | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | loose_float | trailing_zero_equal | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | gsm8k | missing_marker | bare_last_line | 100 | 1.000 (100/100) | - (0/0) | 1.000 | 1.000 |
| exact_match | gsm8k | missing_marker | sentence | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | partial_answer | considered | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | partial_answer | intermediate | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | gsm8k | wrong_then_correct | retracted | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | gsm8k | wrong_then_correct | retracted_tags | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | math500 | clean_correct | boxed | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | math500 | clean_wrong | boxed | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | contradictory_finals | gold_then_wrong_answer_tags | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | contradictory_finals | gold_then_wrong_boxed | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | contradictory_finals | wrong_then_gold_boxed | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | contradictory_finals | wrong_then_gold_hashes | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | format_variants | boxed_bare | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | math500 | format_variants | boxed_bold | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | math500 | format_variants | boxed_spaced | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | math500 | format_variants | suffix_dropped | 2 | - (0/0) | 1.000 (2/2) | 0.000 | 1.000 |
| exact_match | math500 | gold_prefix_collision | leading_number | 6 | 0.000 (0/6) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | loose_float | drift_1e-3 | 66 | 0.000 (0/66) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | loose_float | drift_1e-5 | 66 | 0.000 (0/66) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | loose_float | drift_1e-7 | 66 | 0.000 (0/66) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | loose_float | trailing_zero_equal | 66 | - (0/0) | 0.000 (0/66) | 1.000 | 1.000 |
| exact_match | math500 | missing_marker | bare_last_line | 100 | 1.000 (100/100) | - (0/0) | 1.000 | 1.000 |
| exact_match | math500 | missing_marker | sentence | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | partial_answer | considered | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | partial_answer | intermediate | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| exact_match | math500 | wrong_then_correct | retracted | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| exact_match | math500 | wrong_then_correct | retracted_tags | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | gsm8k | clean_correct | boxed | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | gsm8k | clean_wrong | boxed | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | contradictory_finals | gold_then_wrong_answer_tags | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | contradictory_finals | gold_then_wrong_boxed | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | contradictory_finals | wrong_then_gold_boxed | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | contradictory_finals | wrong_then_gold_hashes | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | format_variants | boxed_bare | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | gsm8k | format_variants | boxed_bold | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | gsm8k | format_variants | boxed_spaced | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | gsm8k | loose_float | drift_1e-3 | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | loose_float | drift_1e-5 | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | loose_float | drift_1e-7 | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | loose_float | trailing_zero_equal | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | gsm8k | missing_marker | bare_last_line | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | missing_marker | sentence | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | partial_answer | considered | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | partial_answer | intermediate | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | gsm8k | wrong_then_correct | retracted | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | gsm8k | wrong_then_correct | retracted_tags | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | math500 | clean_correct | boxed | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | math500 | clean_wrong | boxed | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | contradictory_finals | gold_then_wrong_answer_tags | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | contradictory_finals | gold_then_wrong_boxed | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | contradictory_finals | wrong_then_gold_boxed | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | contradictory_finals | wrong_then_gold_hashes | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | format_variants | boxed_bare | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | math500 | format_variants | boxed_bold | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | math500 | format_variants | boxed_spaced | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | math500 | format_variants | suffix_dropped | 2 | - (0/0) | 0.000 (0/2) | 1.000 | 1.000 |
| math_verify | math500 | gold_prefix_collision | leading_number | 6 | 0.000 (0/6) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | loose_float | drift_1e-3 | 66 | 0.000 (0/66) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | loose_float | drift_1e-5 | 66 | 0.000 (0/66) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | loose_float | drift_1e-7 | 66 | 0.000 (0/66) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | loose_float | trailing_zero_equal | 66 | - (0/0) | 0.000 (0/66) | 1.000 | 1.000 |
| math_verify | math500 | missing_marker | bare_last_line | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | missing_marker | sentence | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | partial_answer | considered | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | partial_answer | intermediate | 100 | 0.000 (0/100) | - (0/0) | 0.000 | 1.000 |
| math_verify | math500 | wrong_then_correct | retracted | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
| math_verify | math500 | wrong_then_correct | retracted_tags | 100 | - (0/0) | 0.000 (0/100) | 1.000 | 1.000 |
