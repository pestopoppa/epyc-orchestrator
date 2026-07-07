# PDF Fast-Path Probe
- corpus_name: `hybrid_sidecar`
- corpus_kind: `structural_table_heavy`
- manifest_path: `orchestration/reports/pdf_structural_candidates_20260706T145900Z.json`
- pdf_count: `27`
- backend_count: `1`
- success_count: `0`
- failure_count: `27`
- structural_signal_totals: `{"liteparse_bboxes": 0, "liteparse_page_images": 0, "structured_figures": 0, "structured_headings": 0, "structured_tables": 0, "table_like_lines": 0}`
- structural_signal_pdf_count: `0` (0.0%)

| Backend | Attempts | Successes | Failures | Median latency ms | Median quality | Table-like lines | Structured h/t/f | BBoxes | Page images | Failure reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| opendataloader_hybrid | 27 | 0 | 27 | n/a | n/a | 0 | 0/0/0 | 0 | 0 | `{"missing_dependency": 27}` |
