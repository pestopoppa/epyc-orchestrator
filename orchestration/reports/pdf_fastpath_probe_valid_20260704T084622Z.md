# PDF Fast-Path Probe
- pdf_count: `8`
- backend_count: `4`
- success_count: `28`
- failure_count: `4`

| Backend | Attempts | Successes | Failures | Median latency ms | Median quality | Failure reasons |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| pdftotext | 8 | 7 | 1 | 22.439 | 0.822 | `{"empty_output": 1}` |
| opendataloader | 8 | 7 | 1 | 644.072 | 0.987 | `{"empty_output": 1}` |
| opendataloader_structured | 8 | 7 | 1 | 685.962 | 0.987 | `{"empty_output": 1}` |
| liteparse | 8 | 7 | 1 | 18.384 | 0.935 | `{"empty_output": 1}` |
