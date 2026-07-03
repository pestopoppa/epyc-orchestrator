# CPU Embedded NEXTN Self-Draft A/B

- Generated: `2026-07-03T18:03:58+00:00`
- Decision: `embedded_self_draft_slower`
- Speedup ratio, median t/s embedded/no-`-md` over same-file `-md`: `0.9618967786066821`
- PSS delta MiB, embedded minus same-file `-md`: `-4368.3056640625`

## Arms

| Arm | Same-file `-md` | Runs | Median t/s | Mean t/s | Load PSS MiB | Acceptance lines | Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| same_file_md | yes | 8 | 36.19389889835907 | 36.440642140487874 | 11453.521484375 | 20 | - |
| embedded_self_draft | no | 8 | 34.81479475554753 | 34.806036929658674 | 7085.2158203125 | 20 | - |

## Notes

- This harness uses `/completion`, not `/chat`, and launches throwaway local servers.
- The default quiet-window guard refuses to run while AutoPilot appears active.
- Decision-grade publication still depends on the measurement protocol in `/workspace/MEASUREMENT.md`.
