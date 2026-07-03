# PDF Fast-Path Probe
- pdf_count: `1`
- backend_count: `4`
- success_count: `1`
- failure_count: `3`

| Backend | Attempts | Successes | Failures | Median latency ms | Median quality | Failure reasons |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| pdftotext | 1 | 1 | 0 | 8.864 | 0.822 | `{}` |
| opendataloader | 1 | 0 | 1 | n/a | n/a | `{"missing_dependency": 1}` |
| opendataloader_structured | 1 | 0 | 1 | n/a | n/a | `{"missing_dependency": 1}` |
| liteparse | 1 | 0 | 1 | n/a | n/a | `{"missing_dependency": 1}` |
