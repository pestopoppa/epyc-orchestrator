# CPU Embedded NEXTN Self-Draft A/B

- Generated: `2026-07-03T18:01:18+00:00`
- Decision: `embedded_self_draft_slower`
- Speedup ratio, median t/s embedded/no-`-md` over same-file `-md`: `0.9585510183105003`
- PSS delta MiB, embedded minus same-file `-md`: `-4372.9619140625`

## Arms

| Arm | Same-file `-md` | Runs | Median t/s | Mean t/s | Load PSS MiB | Acceptance lines | Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| same_file_md | yes | 3 | 39.21556612878024 | 39.306899389687594 | 11454.548828125 | 8 | - |
| embedded_self_draft | no | 3 | 37.59012084636506 | 37.598534080366726 | 7081.5869140625 | 8 | - |

## Notes

- This harness uses `/completion`, not `/chat`, and launches throwaway local servers.
- The default quiet-window guard refuses to run while AutoPilot appears active.
- Decision-grade publication still depends on the measurement protocol in `/workspace/MEASUREMENT.md`.
