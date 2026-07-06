# PDF Fast-Path Probe
- corpus_name: `structural-table-heavy-candidates`
- corpus_kind: `structural_table_heavy`
- manifest_path: `orchestration/reports/pdf_structural_candidates_20260706T145900Z.json`
- pdf_count: `27`
- backend_count: `3`
- success_count: `81`
- failure_count: `0`
- structural_signal_totals: `{"liteparse_bboxes": 0, "liteparse_page_images": 0, "structured_figures": 0, "structured_headings": 0, "structured_tables": 0, "table_like_lines": 19725}`
- structural_signal_pdf_count: `27` (100.0%)

| Backend | Attempts | Successes | Failures | Median latency ms | Median quality | Table-like lines | Structured h/t/f | BBoxes | Page images | Failure reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pdftotext | 27 | 27 | 0 | 121.591 | 0.928 | 4870 | 0/0/0 | 0 | 0 | `{}` |
| opendataloader_structured | 27 | 27 | 0 | 2877.039 | 1.000 | 3341 | 0/0/0 | 0 | 0 | `{}` |
| liteparse | 27 | 27 | 0 | 91.897 | 0.959 | 11514 | 0/0/0 | 0 | 0 | `{}` |
