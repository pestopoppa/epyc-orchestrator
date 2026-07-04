# PDF Fast-Path Probe
- corpus_name: `local-valid-pdf-8`
- corpus_kind: `born_digital_fastpath`
- pdf_count: `8`
- backend_count: `4`
- success_count: `28`
- failure_count: `4`
- structural_signal_totals: `{"liteparse_bboxes": 0, "liteparse_page_images": 0, "structured_figures": 0, "structured_headings": 0, "structured_tables": 0, "table_like_lines": 1195}`

| Backend | Attempts | Successes | Failures | Median latency ms | Median quality | Table-like lines | Structured h/t/f | BBoxes | Page images | Failure reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pdftotext | 8 | 7 | 1 | 20.819 | 0.822 | 209 | 0/0/0 | 0 | 0 | `{"empty_output": 1}` |
| opendataloader | 8 | 7 | 1 | 645.011 | 0.987 | 110 | 0/0/0 | 0 | 0 | `{"empty_output": 1}` |
| opendataloader_structured | 8 | 7 | 1 | 702.757 | 0.987 | 110 | 0/0/0 | 0 | 0 | `{"empty_output": 1}` |
| liteparse | 8 | 7 | 1 | 16.180 | 0.935 | 766 | 0/0/0 | 0 | 0 | `{"empty_output": 1}` |
