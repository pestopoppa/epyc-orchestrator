# PDF Fast-Path Probe
- pdf_count: `12`
- backend_count: `4`
- success_count: `16`
- failure_count: `32`

| Backend | Attempts | Successes | Failures | Median latency ms | Median quality | Failure reasons |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| pdftotext | 12 | 4 | 8 | 24.122 | 0.942 | `{"empty_output": 8}` |
| opendataloader | 12 | 4 | 8 | 766.554 | 0.994 | `{"empty_output": 8}` |
| opendataloader_structured | 12 | 4 | 8 | 990.345 | 0.994 | `{"empty_output": 8}` |
| liteparse | 12 | 4 | 8 | 44.964 | 0.955 | `{"exception": 8}` |
