# PDF Fast-Path Probe
- corpus_name: `structural_table_heavy`
- corpus_kind: `hybrid_sidecar`
- manifest_path: `orchestration/reports/pdf_structural_candidates_20260706T145900Z.json`
- pdf_count: `27`
- backend_count: `1`
- success_count: `27`
- failure_count: `0`
- structural_signal_totals: `{"liteparse_bboxes": 0, "liteparse_page_images": 0, "structured_figures": 0, "structured_headings": 0, "structured_tables": 0, "table_like_lines": 3307}`
- structural_signal_pdf_count: `27` (100.0%)

| Backend | Attempts | Successes | Failures | Median latency ms | Median quality | Table-like lines | Structured h/t/f | BBoxes | Page images | Failure reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| opendataloader_hybrid | 27 | 27 | 0 | 1510.743 | 1.000 | 3307 | 0/0/0 | 0 | 0 | `{}` |
