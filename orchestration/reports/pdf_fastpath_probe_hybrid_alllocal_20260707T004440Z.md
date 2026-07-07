# PDF Fast-Path Probe
- corpus_name: `structural-table-heavy-alllocal-n51`
- corpus_kind: `hybrid_sidecar_alllocal`
- manifest_path: `orchestration/reports/pdf_structural_candidates_20260707T004421Z_alllocal_n200.json`
- pdf_count: `51`
- backend_count: `4`
- success_count: `153`
- failure_count: `51`
- structural_signal_totals: `{"liteparse_bboxes": 0, "liteparse_page_images": 0, "structured_figures": 0, "structured_headings": 0, "structured_tables": 0, "table_like_lines": 13784}`
- structural_signal_pdf_count: `51` (100.0%)

| Backend | Attempts | Successes | Failures | Median latency ms | Median quality | Table-like lines | Structured h/t/f | BBoxes | Page images | Failure reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pdftotext | 51 | 51 | 0 | 41.132 | 0.898 | 5851 | 0/0/0 | 0 | 0 | `{}` |
| opendataloader_structured | 51 | 51 | 0 | 1230.890 | 0.991 | 3984 | 0/0/0 | 0 | 0 | `{}` |
| opendataloader_hybrid | 51 | 51 | 0 | 1109.577 | 0.991 | 3949 | 0/0/0 | 0 | 0 | `{}` |
| liteparse | 51 | 0 | 51 | n/a | n/a | 0 | 0/0/0 | 0 | 0 | `{"missing_dependency": 51}` |
