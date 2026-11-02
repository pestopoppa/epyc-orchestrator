# CRM + Slack model-free control — 2026-11-02

The attached native event journal is the original q1-claimed model-free control:
`receipt.jsonl` SHA-256 `f4c5e96a6d764721f21fa5904318ea1ee1c50b2185797aecefb3bb539b173f8c`.
The isolated CRM and Slack services were started from recorded immutable image
digests, reset, and seeded with the pinned upstream case seed after replacing
its nondeterministic UUID/time expressions with fixed values. Pinned upstream
MCP clients created a synthetic CRM note and posted a synthetic Slack message;
the service snapshots contained both. A second reset reproduced the initial
judge-facing state and the exact owned containers and network were removed.
The original source/control hashes at the execution were:

- `crm_provision.py`: `54d10ca3b88e320302b006be4a9e0b348257668b22581c03b453a865838df584`
- `crm_services.py`: `864a713c11a3245298d411f0570befcee59d11845f65440e01d515bc6808d86f`
- `crm_service_control.py`: `c29a97712ecd60b93fed4a0dffa04b8e1e8650f98ae47c4652c69f9280103e3f`
- upstream seed SQL: `f8da9246199906e77c95e6b9afd4857604967a393ff829a9268d5f515cd2916f`

Slack image `sha256:7e864c1706d77a2932ce488ea76f4ab9057687a7ed379b707289f2c2442a0f4d`
was the publisher's legacy API image resolved on the test date. It is not a
build proven from the pinned upstream MCP commit. An earlier current-image
attempt failed because that image requires a newer admin-auth setup; its
captured containers and network were subsequently removed. This control ran no
model and proves no attack delivery, model quality, security result, benchmark
qualification, inference grant, or belief-substrate wiring.
