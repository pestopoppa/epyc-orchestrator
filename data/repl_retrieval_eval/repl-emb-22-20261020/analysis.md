# REPL-EMB-2.2 offline retrieval eval

Synthetic queries are LLM-written; they tend to share vocabulary with their source span, which favours lexical arms. The synthetic stratum therefore makes the kill rule conservative toward dense arms.

The deciding (primary) set excludes the answer_named_verbatim rows (interpretation (q)): the note above applies in full to the secondary, non-deciding sets that include flagged rows, and only in part to the unflagged synthetic rows that remain in the primary set.

- KILL RULE: FAIL (n=346)
- KILL RULE (incl. extension arms): PASS [extension_only: eg2-gguf-bf16-256.hybrid, eg2-gguf-bf16-768.hybrid] (not a registered-arm result)
- SELECTION: not applicable (kill rule FAIL)
- (non-deciding) KILL RULE (pooled, flagged included): FAIL (n=560)
- (non-deciding) KILL RULE (pooled, flagged included; incl. extension arms): PASS [extension_only: eg2-gguf-bf16-256.hybrid, eg2-gguf-bf16-768.hybrid] (not a registered-arm result)
- (non-deciding) KILL RULE (without synthetic, flagged excluded): UNDERPOWERED/KILL (n=68)
- (non-deciding) KILL RULE (without synthetic, flagged included): UNDERPOWERED/KILL (n=78)

deciding (primary) n = 346 after excluding 214 answer_named_verbatim rows; floor check ok: primary n=346 (>= 120), trace rows=301 (>= 75)
all scored rows by group: {'probe45': 45, 'agent_nav': 33, 'synthetic_spill': 482, 'trivial_head': 22}; unscored 0
costs here are `contended_not_for_selection`; failures count as misses.

# PRIMARY decision set (the only deciding set; n=346)

**L** = grep (grep 0.538, bm25 0.451, tie=False)

## recall@k per arm and stratum (Wilson 95%)

| arm | stratum | n | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| bge-large.dense | agent_nav | 23 | 0.304 [0.156, 0.509] | 0.522 [0.330, 0.708] | 0.609 [0.408, 0.778] | 0.652 [0.449, 0.812] |
| bge-large.dense | probe45 | 45 | 0.556 [0.412, 0.691] | 0.667 [0.521, 0.786] | 0.733 [0.590, 0.840] | 0.778 [0.637, 0.875] |
| bge-large.dense | synthetic_spill | 278 | 0.227 [0.181, 0.279] | 0.360 [0.306, 0.418] | 0.410 [0.354, 0.469] | 0.489 [0.431, 0.548] |
| bge-large.dense | pooled | 346 | 0.275 [0.230, 0.324] | 0.410 [0.360, 0.463] | 0.465 [0.413, 0.518] | 0.538 [0.485, 0.589] |
| bge-large.hybrid | agent_nav | 23 | 0.348 [0.188, 0.551] | 0.565 [0.368, 0.744] | 0.652 [0.449, 0.812] | 0.783 [0.581, 0.903] |
| bge-large.hybrid | probe45 | 45 | 0.511 [0.370, 0.650] | 0.667 [0.521, 0.786] | 0.756 [0.613, 0.858] | 0.800 [0.662, 0.891] |
| bge-large.hybrid | synthetic_spill | 278 | 0.237 [0.191, 0.291] | 0.349 [0.295, 0.407] | 0.421 [0.364, 0.480] | 0.518 [0.459, 0.576] |
| bge-large.hybrid | pooled | 346 | 0.280 [0.236, 0.330] | 0.405 [0.354, 0.457] | 0.480 [0.428, 0.532] | 0.572 [0.520, 0.623] |
| bm25 | agent_nav | 23 | 0.174 [0.070, 0.371] | 0.565 [0.368, 0.744] | 0.696 [0.491, 0.844] | 0.826 [0.629, 0.930] |
| bm25 | probe45 | 45 | 0.333 [0.214, 0.479] | 0.533 [0.391, 0.671] | 0.578 [0.433, 0.710] | 0.689 [0.543, 0.805] |
| bm25 | synthetic_spill | 278 | 0.216 [0.172, 0.268] | 0.356 [0.302, 0.414] | 0.410 [0.354, 0.469] | 0.500 [0.442, 0.558] |
| bm25 | pooled | 346 | 0.228 [0.187, 0.275] | 0.393 [0.343, 0.445] | 0.451 [0.399, 0.504] | 0.546 [0.494, 0.598] |
| colgrep | probe45 | 45 | 0.378 [0.251, 0.524] | 0.689 [0.543, 0.805] | 0.756 [0.613, 0.858] | 0.822 [0.687, 0.907] |
| colgrep | pooled | 45 | 0.378 [0.251, 0.524] | 0.689 [0.543, 0.805] | 0.756 [0.613, 0.858] | 0.822 [0.687, 0.907] |
| eg2-gguf-bf16-256.dense | agent_nav | 23 | 0.261 [0.125, 0.465] | 0.522 [0.330, 0.708] | 0.565 [0.368, 0.744] | 0.652 [0.449, 0.812] |
| eg2-gguf-bf16-256.dense | probe45 | 45 | 0.378 [0.251, 0.524] | 0.667 [0.521, 0.786] | 0.778 [0.637, 0.875] | 0.867 [0.738, 0.937] |
| eg2-gguf-bf16-256.dense | synthetic_spill | 278 | 0.378 [0.323, 0.436] | 0.604 [0.546, 0.660] | 0.687 [0.630, 0.739] | 0.799 [0.747, 0.841] |
| eg2-gguf-bf16-256.dense | pooled | 346 | 0.370 [0.321, 0.422] | 0.607 [0.555, 0.657] | 0.691 [0.640, 0.737] | 0.798 [0.752, 0.837] |
| eg2-gguf-bf16-256.hybrid | agent_nav | 23 | 0.304 [0.156, 0.509] | 0.652 [0.449, 0.812] | 0.739 [0.535, 0.875] | 0.870 [0.679, 0.955] |
| eg2-gguf-bf16-256.hybrid | probe45 | 45 | 0.444 [0.309, 0.588] | 0.689 [0.543, 0.805] | 0.689 [0.543, 0.805] | 0.822 [0.687, 0.907] |
| eg2-gguf-bf16-256.hybrid | synthetic_spill | 278 | 0.392 [0.337, 0.451] | 0.572 [0.513, 0.629] | 0.669 [0.612, 0.722] | 0.773 [0.721, 0.819] |
| eg2-gguf-bf16-256.hybrid | pooled | 346 | 0.393 [0.343, 0.445] | 0.592 [0.540, 0.643] | 0.676 [0.625, 0.723] | 0.786 [0.740, 0.826] |
| eg2-gguf-bf16-768.dense | agent_nav | 23 | 0.348 [0.188, 0.551] | 0.522 [0.330, 0.708] | 0.609 [0.408, 0.778] | 0.696 [0.491, 0.844] |
| eg2-gguf-bf16-768.dense | probe45 | 45 | 0.400 [0.270, 0.545] | 0.644 [0.498, 0.768] | 0.756 [0.613, 0.858] | 0.844 [0.712, 0.923] |
| eg2-gguf-bf16-768.dense | synthetic_spill | 278 | 0.414 [0.357, 0.472] | 0.640 [0.582, 0.694] | 0.716 [0.660, 0.766] | 0.809 [0.759, 0.851] |
| eg2-gguf-bf16-768.dense | pooled | 346 | 0.408 [0.357, 0.460] | 0.633 [0.581, 0.682] | 0.714 [0.664, 0.759] | 0.806 [0.761, 0.845] |
| eg2-gguf-bf16-768.hybrid | agent_nav | 23 | 0.261 [0.125, 0.465] | 0.522 [0.330, 0.708] | 0.652 [0.449, 0.812] | 0.783 [0.581, 0.903] |
| eg2-gguf-bf16-768.hybrid | probe45 | 45 | 0.422 [0.290, 0.567] | 0.667 [0.521, 0.786] | 0.711 [0.566, 0.823] | 0.778 [0.637, 0.875] |
| eg2-gguf-bf16-768.hybrid | synthetic_spill | 278 | 0.399 [0.343, 0.458] | 0.579 [0.520, 0.636] | 0.655 [0.597, 0.708] | 0.766 [0.713, 0.812] |
| eg2-gguf-bf16-768.hybrid | pooled | 346 | 0.393 [0.343, 0.445] | 0.587 [0.534, 0.637] | 0.662 [0.610, 0.710] | 0.769 [0.722, 0.810] |
| granite-97m-r2.dense | agent_nav | 23 | 0.348 [0.188, 0.551] | 0.522 [0.330, 0.708] | 0.652 [0.449, 0.812] | 0.739 [0.535, 0.875] |
| granite-97m-r2.dense | probe45 | 45 | 0.533 [0.391, 0.671] | 0.733 [0.590, 0.840] | 0.756 [0.613, 0.858] | 0.778 [0.637, 0.875] |
| granite-97m-r2.dense | synthetic_spill | 278 | 0.219 [0.175, 0.272] | 0.421 [0.364, 0.480] | 0.504 [0.445, 0.562] | 0.608 [0.549, 0.663] |
| granite-97m-r2.dense | pooled | 346 | 0.269 [0.225, 0.318] | 0.468 [0.416, 0.521] | 0.546 [0.494, 0.598] | 0.639 [0.587, 0.688] |
| granite-97m-r2.hybrid | agent_nav | 23 | 0.391 [0.222, 0.592] | 0.609 [0.408, 0.778] | 0.696 [0.491, 0.844] | 0.870 [0.679, 0.955] |
| granite-97m-r2.hybrid | probe45 | 45 | 0.556 [0.412, 0.691] | 0.711 [0.566, 0.823] | 0.756 [0.613, 0.858] | 0.800 [0.662, 0.891] |
| granite-97m-r2.hybrid | synthetic_spill | 278 | 0.320 [0.268, 0.377] | 0.478 [0.420, 0.537] | 0.558 [0.499, 0.615] | 0.647 [0.590, 0.701] |
| granite-97m-r2.hybrid | pooled | 346 | 0.355 [0.307, 0.407] | 0.517 [0.465, 0.570] | 0.592 [0.540, 0.643] | 0.682 [0.631, 0.729] |
| grep | agent_nav | 23 | 0.130 [0.045, 0.321] | 0.261 [0.125, 0.465] | 0.435 [0.256, 0.632] | 0.565 [0.368, 0.744] |
| grep | probe45 | 45 | 0.422 [0.290, 0.567] | 0.600 [0.455, 0.730] | 0.644 [0.498, 0.768] | 0.689 [0.543, 0.805] |
| grep | synthetic_spill | 278 | 0.281 [0.231, 0.336] | 0.450 [0.392, 0.508] | 0.529 [0.470, 0.587] | 0.597 [0.539, 0.653] |
| grep | pooled | 346 | 0.289 [0.244, 0.339] | 0.457 [0.405, 0.509] | 0.538 [0.485, 0.589] | 0.607 [0.555, 0.657] |
| potion-base-8m.dense | agent_nav | 23 | 0.261 [0.125, 0.465] | 0.391 [0.222, 0.592] | 0.478 [0.292, 0.670] | 0.565 [0.368, 0.744] |
| potion-base-8m.dense | probe45 | 45 | 0.267 [0.160, 0.410] | 0.444 [0.309, 0.588] | 0.533 [0.391, 0.671] | 0.711 [0.566, 0.823] |
| potion-base-8m.dense | synthetic_spill | 278 | 0.129 [0.095, 0.174] | 0.259 [0.211, 0.314] | 0.338 [0.285, 0.396] | 0.446 [0.389, 0.505] |
| potion-base-8m.dense | pooled | 346 | 0.156 [0.122, 0.198] | 0.292 [0.246, 0.342] | 0.373 [0.324, 0.425] | 0.488 [0.436, 0.541] |
| potion-base-8m.hybrid | agent_nav | 23 | 0.217 [0.097, 0.419] | 0.522 [0.330, 0.708] | 0.652 [0.449, 0.812] | 0.739 [0.535, 0.875] |
| potion-base-8m.hybrid | probe45 | 45 | 0.467 [0.329, 0.609] | 0.600 [0.455, 0.730] | 0.600 [0.455, 0.730] | 0.667 [0.521, 0.786] |
| potion-base-8m.hybrid | synthetic_spill | 278 | 0.201 [0.159, 0.253] | 0.317 [0.265, 0.373] | 0.421 [0.364, 0.480] | 0.522 [0.463, 0.580] |
| potion-base-8m.hybrid | pooled | 346 | 0.237 [0.195, 0.285] | 0.367 [0.318, 0.419] | 0.460 [0.408, 0.512] | 0.555 [0.502, 0.606] |
| potion-base-8m.whybrid | agent_nav | 23 | 0.304 [0.156, 0.509] | 0.478 [0.292, 0.670] | 0.609 [0.408, 0.778] | 0.783 [0.581, 0.903] |
| potion-base-8m.whybrid | probe45 | 45 | 0.444 [0.309, 0.588] | 0.600 [0.455, 0.730] | 0.600 [0.455, 0.730] | 0.733 [0.590, 0.840] |
| potion-base-8m.whybrid | synthetic_spill | 278 | 0.183 [0.142, 0.233] | 0.281 [0.231, 0.336] | 0.399 [0.343, 0.458] | 0.514 [0.456, 0.573] |
| potion-base-8m.whybrid | pooled | 346 | 0.225 [0.185, 0.272] | 0.335 [0.288, 0.387] | 0.439 [0.388, 0.492] | 0.561 [0.508, 0.612] |
| potion-code-16m-v2.dense | agent_nav | 23 | 0.217 [0.097, 0.419] | 0.435 [0.256, 0.632] | 0.522 [0.330, 0.708] | 0.652 [0.449, 0.812] |
| potion-code-16m-v2.dense | probe45 | 45 | 0.400 [0.270, 0.545] | 0.600 [0.455, 0.730] | 0.667 [0.521, 0.786] | 0.778 [0.637, 0.875] |
| potion-code-16m-v2.dense | synthetic_spill | 278 | 0.223 [0.178, 0.276] | 0.349 [0.295, 0.407] | 0.410 [0.354, 0.469] | 0.514 [0.456, 0.573] |
| potion-code-16m-v2.dense | pooled | 346 | 0.246 [0.203, 0.294] | 0.387 [0.337, 0.440] | 0.451 [0.399, 0.504] | 0.558 [0.505, 0.609] |
| potion-code-16m-v2.hybrid | agent_nav | 23 | 0.130 [0.045, 0.321] | 0.522 [0.330, 0.708] | 0.652 [0.449, 0.812] | 0.783 [0.581, 0.903] |
| potion-code-16m-v2.hybrid | probe45 | 45 | 0.444 [0.309, 0.588] | 0.644 [0.498, 0.768] | 0.689 [0.543, 0.805] | 0.756 [0.613, 0.858] |
| potion-code-16m-v2.hybrid | synthetic_spill | 278 | 0.230 [0.185, 0.283] | 0.360 [0.306, 0.418] | 0.450 [0.392, 0.508] | 0.586 [0.528, 0.643] |
| potion-code-16m-v2.hybrid | pooled | 346 | 0.251 [0.209, 0.300] | 0.408 [0.357, 0.460] | 0.494 [0.442, 0.547] | 0.621 [0.569, 0.671] |
| potion-code-16m-v2.whybrid | agent_nav | 23 | 0.217 [0.097, 0.419] | 0.478 [0.292, 0.670] | 0.565 [0.368, 0.744] | 0.739 [0.535, 0.875] |
| potion-code-16m-v2.whybrid | probe45 | 45 | 0.467 [0.329, 0.609] | 0.689 [0.543, 0.805] | 0.733 [0.590, 0.840] | 0.756 [0.613, 0.858] |
| potion-code-16m-v2.whybrid | synthetic_spill | 278 | 0.216 [0.172, 0.268] | 0.345 [0.292, 0.403] | 0.460 [0.403, 0.519] | 0.561 [0.502, 0.618] |
| potion-code-16m-v2.whybrid | pooled | 346 | 0.249 [0.206, 0.297] | 0.399 [0.349, 0.451] | 0.503 [0.450, 0.555] | 0.598 [0.546, 0.649] |

## synthetic_spill recall split by writer

| arm | writer | n | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| bge-large.dense | writer=A | 110 | 0.255 | 0.391 | 0.436 | 0.518 |
| bge-large.dense | writer=B | 168 | 0.208 | 0.339 | 0.393 | 0.470 |
| bge-large.dense | pooled | 278 | 0.227 | 0.360 | 0.410 | 0.489 |
| bge-large.hybrid | writer=A | 110 | 0.291 | 0.400 | 0.464 | 0.564 |
| bge-large.hybrid | writer=B | 168 | 0.202 | 0.315 | 0.393 | 0.488 |
| bge-large.hybrid | pooled | 278 | 0.237 | 0.349 | 0.421 | 0.518 |
| bm25 | writer=A | 110 | 0.282 | 0.445 | 0.500 | 0.564 |
| bm25 | writer=B | 168 | 0.173 | 0.298 | 0.351 | 0.458 |
| bm25 | pooled | 278 | 0.216 | 0.356 | 0.410 | 0.500 |
| eg2-gguf-bf16-256.dense | writer=A | 110 | 0.509 | 0.727 | 0.791 | 0.873 |
| eg2-gguf-bf16-256.dense | writer=B | 168 | 0.292 | 0.524 | 0.619 | 0.750 |
| eg2-gguf-bf16-256.dense | pooled | 278 | 0.378 | 0.604 | 0.687 | 0.799 |
| eg2-gguf-bf16-256.hybrid | writer=A | 110 | 0.473 | 0.682 | 0.791 | 0.864 |
| eg2-gguf-bf16-256.hybrid | writer=B | 168 | 0.339 | 0.500 | 0.589 | 0.714 |
| eg2-gguf-bf16-256.hybrid | pooled | 278 | 0.392 | 0.572 | 0.669 | 0.773 |
| eg2-gguf-bf16-768.dense | writer=A | 110 | 0.527 | 0.755 | 0.809 | 0.891 |
| eg2-gguf-bf16-768.dense | writer=B | 168 | 0.339 | 0.565 | 0.655 | 0.756 |
| eg2-gguf-bf16-768.dense | pooled | 278 | 0.414 | 0.640 | 0.716 | 0.809 |
| eg2-gguf-bf16-768.hybrid | writer=A | 110 | 0.500 | 0.655 | 0.727 | 0.873 |
| eg2-gguf-bf16-768.hybrid | writer=B | 168 | 0.333 | 0.530 | 0.607 | 0.696 |
| eg2-gguf-bf16-768.hybrid | pooled | 278 | 0.399 | 0.579 | 0.655 | 0.766 |
| granite-97m-r2.dense | writer=A | 110 | 0.291 | 0.536 | 0.600 | 0.664 |
| granite-97m-r2.dense | writer=B | 168 | 0.173 | 0.345 | 0.440 | 0.571 |
| granite-97m-r2.dense | pooled | 278 | 0.219 | 0.421 | 0.504 | 0.608 |
| granite-97m-r2.hybrid | writer=A | 110 | 0.382 | 0.564 | 0.664 | 0.736 |
| granite-97m-r2.hybrid | writer=B | 168 | 0.280 | 0.423 | 0.488 | 0.589 |
| granite-97m-r2.hybrid | pooled | 278 | 0.320 | 0.478 | 0.558 | 0.647 |
| grep | writer=A | 110 | 0.382 | 0.573 | 0.645 | 0.700 |
| grep | writer=B | 168 | 0.214 | 0.369 | 0.452 | 0.530 |
| grep | pooled | 278 | 0.281 | 0.450 | 0.529 | 0.597 |
| potion-base-8m.dense | writer=A | 110 | 0.127 | 0.273 | 0.327 | 0.455 |
| potion-base-8m.dense | writer=B | 168 | 0.131 | 0.250 | 0.345 | 0.440 |
| potion-base-8m.dense | pooled | 278 | 0.129 | 0.259 | 0.338 | 0.446 |
| potion-base-8m.hybrid | writer=A | 110 | 0.227 | 0.355 | 0.455 | 0.527 |
| potion-base-8m.hybrid | writer=B | 168 | 0.185 | 0.292 | 0.399 | 0.518 |
| potion-base-8m.hybrid | pooled | 278 | 0.201 | 0.317 | 0.421 | 0.522 |
| potion-base-8m.whybrid | writer=A | 110 | 0.191 | 0.318 | 0.418 | 0.545 |
| potion-base-8m.whybrid | writer=B | 168 | 0.179 | 0.256 | 0.387 | 0.494 |
| potion-base-8m.whybrid | pooled | 278 | 0.183 | 0.281 | 0.399 | 0.514 |
| potion-code-16m-v2.dense | writer=A | 110 | 0.245 | 0.373 | 0.427 | 0.545 |
| potion-code-16m-v2.dense | writer=B | 168 | 0.208 | 0.333 | 0.399 | 0.494 |
| potion-code-16m-v2.dense | pooled | 278 | 0.223 | 0.349 | 0.410 | 0.514 |
| potion-code-16m-v2.hybrid | writer=A | 110 | 0.245 | 0.400 | 0.464 | 0.600 |
| potion-code-16m-v2.hybrid | writer=B | 168 | 0.220 | 0.333 | 0.440 | 0.577 |
| potion-code-16m-v2.hybrid | pooled | 278 | 0.230 | 0.360 | 0.450 | 0.586 |
| potion-code-16m-v2.whybrid | writer=A | 110 | 0.218 | 0.345 | 0.491 | 0.600 |
| potion-code-16m-v2.whybrid | writer=B | 168 | 0.214 | 0.345 | 0.440 | 0.536 |
| potion-code-16m-v2.whybrid | pooled | 278 | 0.216 | 0.345 | 0.460 | 0.561 |

## kill rule (recall@5, hybrid vs L)

| hybrid | kind | R@5 | L R@5 | diff | CI lo | CI hi | margin | CI>0 | pass |
|---|---|---|---|---|---|---|---|---|---|
| bge-large.hybrid | registered | 0.480 | 0.538 | -0.058 | -0.116 | -0.003 | False | False | False |
| granite-97m-r2.hybrid | registered | 0.592 | 0.538 | +0.055 | +0.003 | +0.107 | False | True | False |
| eg2-gguf-bf16-256.hybrid | extension (extension_only if it flips) | 0.676 | 0.538 | +0.139 | +0.087 | +0.191 | True | True | True |
| eg2-gguf-bf16-768.hybrid | extension (extension_only if it flips) | 0.662 | 0.538 | +0.124 | +0.072 | +0.176 | True | True | True |
| potion-base-8m.hybrid | extension (extension_only if it flips) | 0.460 | 0.538 | -0.078 | -0.136 | -0.020 | False | False | False |
| potion-code-16m-v2.hybrid | extension (extension_only if it flips) | 0.494 | 0.538 | -0.043 | -0.101 | +0.014 | False | False | False |

n = 346, M = 0.1, n_min = 120, bootstrap resamples/seed = 10000/20261020

## lexical vs dense vs hybrid (pooled recall@5)

| dense arm | registered | L | dense | hybrid | weighted hybrid |
|---|---|---|---|---|---|
| bge-large | True | 0.538 | 0.465 | 0.480 | n/a |
| eg2-gguf-bf16-256 | False | 0.538 | 0.691 | 0.676 | n/a |
| eg2-gguf-bf16-768 | False | 0.538 | 0.714 | 0.662 | n/a |
| granite-97m-r2 | True | 0.538 | 0.546 | 0.592 | n/a |
| potion-base-8m | False | 0.538 | 0.373 | 0.460 | 0.439 |
| potion-code-16m-v2 | False | 0.538 | 0.451 | 0.494 | 0.503 |

## paired bootstrap, recall@5 (diff, 95% CI)

| arm | vs | diff | CI |
|---|---|---|---|
| bge-large.dense | grep | -0.072 | [-0.133, -0.012] |
| bge-large.hybrid | grep | -0.058 | [-0.116, -0.003] |
| bm25 | grep | -0.087 | [-0.139, -0.035] |
| bm25 | bge-large.hybrid | -0.029 | [-0.084, +0.023] |
| colgrep | grep | +0.111 | [-0.022, +0.244] |
| colgrep | bge-large.hybrid | +0.000 | [-0.133, +0.133] |
| eg2-gguf-bf16-256.dense | grep | +0.153 | [+0.095, +0.211] |
| eg2-gguf-bf16-256.dense | bge-large.dense | +0.225 | [+0.165, +0.286] |
| eg2-gguf-bf16-256.hybrid | grep | +0.139 | [+0.087, +0.191] |
| eg2-gguf-bf16-256.hybrid | bge-large.hybrid | +0.197 | [+0.145, +0.249] |
| eg2-gguf-bf16-768.dense | grep | +0.176 | [+0.118, +0.234] |
| eg2-gguf-bf16-768.dense | bge-large.dense | +0.249 | [+0.191, +0.306] |
| eg2-gguf-bf16-768.hybrid | grep | +0.124 | [+0.072, +0.176] |
| eg2-gguf-bf16-768.hybrid | bge-large.hybrid | +0.182 | [+0.133, +0.234] |
| granite-97m-r2.dense | grep | +0.009 | [-0.049, +0.064] |
| granite-97m-r2.dense | bge-large.dense | +0.081 | [+0.026, +0.139] |
| granite-97m-r2.hybrid | grep | +0.055 | [+0.003, +0.107] |
| granite-97m-r2.hybrid | bge-large.hybrid | +0.113 | [+0.064, +0.165] |
| grep | bge-large.hybrid | +0.058 | [+0.003, +0.116] |
| potion-base-8m.dense | grep | -0.165 | [-0.228, -0.104] |
| potion-base-8m.dense | bge-large.dense | -0.092 | [-0.150, -0.035] |
| potion-base-8m.hybrid | grep | -0.078 | [-0.136, -0.020] |
| potion-base-8m.hybrid | bge-large.hybrid | -0.020 | [-0.075, +0.035] |
| potion-base-8m.whybrid | grep | -0.098 | [-0.159, -0.038] |
| potion-code-16m-v2.dense | grep | -0.087 | [-0.147, -0.026] |
| potion-code-16m-v2.dense | bge-large.dense | -0.014 | [-0.072, +0.040] |
| potion-code-16m-v2.hybrid | grep | -0.043 | [-0.101, +0.014] |
| potion-code-16m-v2.hybrid | bge-large.hybrid | +0.014 | [-0.035, +0.064] |
| potion-code-16m-v2.whybrid | grep | -0.035 | [-0.095, +0.023] |

## per-stratum paired diff vs L, recall@5 (descriptive, no verdict)

| stratum | n | arm | diff | CI |
|---|---|---|---|---|
| probe45 | 45 | bge-large.dense | +0.089 | [-0.022, +0.200] |
| probe45 | 45 | bge-large.hybrid | +0.111 | [+0.022, +0.200] |
| probe45 | 45 | bm25 | -0.067 | [-0.178, +0.044] |
| probe45 | 45 | colgrep | +0.111 | [-0.022, +0.244] |
| probe45 | 45 | eg2-gguf-bf16-256.dense | +0.133 | [+0.000, +0.267] |
| probe45 | 45 | eg2-gguf-bf16-256.hybrid | +0.044 | [-0.067, +0.156] |
| probe45 | 45 | eg2-gguf-bf16-768.dense | +0.111 | [-0.022, +0.244] |
| probe45 | 45 | eg2-gguf-bf16-768.hybrid | +0.067 | [-0.067, +0.200] |
| probe45 | 45 | granite-97m-r2.dense | +0.111 | [+0.000, +0.222] |
| probe45 | 45 | granite-97m-r2.hybrid | +0.111 | [+0.022, +0.200] |
| probe45 | 45 | potion-base-8m.dense | -0.111 | [-0.244, +0.022] |
| probe45 | 45 | potion-base-8m.hybrid | -0.044 | [-0.156, +0.067] |
| probe45 | 45 | potion-base-8m.whybrid | -0.044 | [-0.156, +0.067] |
| probe45 | 45 | potion-code-16m-v2.dense | +0.022 | [-0.067, +0.111] |
| probe45 | 45 | potion-code-16m-v2.hybrid | +0.044 | [-0.067, +0.156] |
| probe45 | 45 | potion-code-16m-v2.whybrid | +0.089 | [+0.000, +0.200] |
| agent_nav | 23 | bge-large.dense | +0.174 | [+0.000, +0.391] |
| agent_nav | 23 | bge-large.hybrid | +0.217 | [+0.000, +0.435] |
| agent_nav | 23 | bm25 | +0.261 | [+0.043, +0.478] |
| agent_nav | 23 | eg2-gguf-bf16-256.dense | +0.130 | [-0.130, +0.391] |
| agent_nav | 23 | eg2-gguf-bf16-256.hybrid | +0.304 | [+0.087, +0.522] |
| agent_nav | 23 | eg2-gguf-bf16-768.dense | +0.174 | [-0.043, +0.391] |
| agent_nav | 23 | eg2-gguf-bf16-768.hybrid | +0.217 | [+0.000, +0.435] |
| agent_nav | 23 | granite-97m-r2.dense | +0.217 | [+0.000, +0.435] |
| agent_nav | 23 | granite-97m-r2.hybrid | +0.261 | [+0.043, +0.478] |
| agent_nav | 23 | potion-base-8m.dense | +0.043 | [-0.174, +0.261] |
| agent_nav | 23 | potion-base-8m.hybrid | +0.217 | [-0.043, +0.435] |
| agent_nav | 23 | potion-base-8m.whybrid | +0.174 | [-0.043, +0.391] |
| agent_nav | 23 | potion-code-16m-v2.dense | +0.087 | [-0.130, +0.304] |
| agent_nav | 23 | potion-code-16m-v2.hybrid | +0.217 | [-0.043, +0.478] |
| agent_nav | 23 | potion-code-16m-v2.whybrid | +0.130 | [-0.130, +0.391] |
| synthetic_spill | 278 | bge-large.dense | -0.119 | [-0.187, -0.050] |
| synthetic_spill | 278 | bge-large.hybrid | -0.108 | [-0.173, -0.043] |
| synthetic_spill | 278 | bm25 | -0.119 | [-0.176, -0.061] |
| synthetic_spill | 278 | eg2-gguf-bf16-256.dense | +0.158 | [+0.090, +0.223] |
| synthetic_spill | 278 | eg2-gguf-bf16-256.hybrid | +0.140 | [+0.083, +0.201] |
| synthetic_spill | 278 | eg2-gguf-bf16-768.dense | +0.187 | [+0.122, +0.252] |
| synthetic_spill | 278 | eg2-gguf-bf16-768.hybrid | +0.126 | [+0.068, +0.187] |
| synthetic_spill | 278 | granite-97m-r2.dense | -0.025 | [-0.090, +0.040] |
| synthetic_spill | 278 | granite-97m-r2.hybrid | +0.029 | [-0.029, +0.086] |
| synthetic_spill | 278 | potion-base-8m.dense | -0.191 | [-0.259, -0.122] |
| synthetic_spill | 278 | potion-base-8m.hybrid | -0.108 | [-0.173, -0.043] |
| synthetic_spill | 278 | potion-base-8m.whybrid | -0.129 | [-0.198, -0.061] |
| synthetic_spill | 278 | potion-code-16m-v2.dense | -0.119 | [-0.191, -0.047] |
| synthetic_spill | 278 | potion-code-16m-v2.hybrid | -0.079 | [-0.147, -0.014] |
| synthetic_spill | 278 | potion-code-16m-v2.whybrid | -0.068 | [-0.137, +0.000] |

## leakage cut: answer_named_verbatim (reported variant)

unlabelled rows: 45
- answer_named_verbatim=true: n=0; 
- answer_named_verbatim=false: n=301; bge-large.dense 0.425, bge-large.hybrid 0.439, bm25 0.432, eg2-gguf-bf16-256.dense 0.678, eg2-gguf-bf16-256.hybrid 0.674, eg2-gguf-bf16-768.dense 0.708, eg2-gguf-bf16-768.hybrid 0.654, granite-97m-r2.dense 0.515, granite-97m-r2.hybrid 0.568, grep 0.522, potion-base-8m.dense 0.349, potion-base-8m.hybrid 0.439, potion-base-8m.whybrid 0.415, potion-code-16m-v2.dense 0.419, potion-code-16m-v2.hybrid 0.465, potion-code-16m-v2.whybrid 0.468

## leakage cut: intent_source (reported variant)

rows without the field: 323
- assistant_text: n=5; bge-large.dense 0.600, bge-large.hybrid 0.600, bm25 0.600, eg2-gguf-bf16-256.dense 0.600, eg2-gguf-bf16-256.hybrid 0.600, eg2-gguf-bf16-768.dense 0.600, eg2-gguf-bf16-768.hybrid 0.600, granite-97m-r2.dense 0.600, granite-97m-r2.hybrid 0.600, grep 0.600, potion-base-8m.dense 0.600, potion-base-8m.hybrid 0.600, potion-base-8m.whybrid 0.600, potion-code-16m-v2.dense 0.400, potion-code-16m-v2.hybrid 0.600, potion-code-16m-v2.whybrid 0.400; kill rule on this subset: UNDERPOWERED/KILL (n=5)
- bash_description: n=18; bge-large.dense 0.611, bge-large.hybrid 0.667, bm25 0.722, eg2-gguf-bf16-256.dense 0.556, eg2-gguf-bf16-256.hybrid 0.778, eg2-gguf-bf16-768.dense 0.611, eg2-gguf-bf16-768.hybrid 0.667, granite-97m-r2.dense 0.667, granite-97m-r2.hybrid 0.722, grep 0.389, potion-base-8m.dense 0.444, potion-base-8m.hybrid 0.667, potion-base-8m.whybrid 0.611, potion-code-16m-v2.dense 0.556, potion-code-16m-v2.hybrid 0.667, potion-code-16m-v2.whybrid 0.611; kill rule on this subset: UNDERPOWERED/KILL (n=18)

## selection

status: not applicable

## leakage measurement (answer_named_verbatim; descriptive, never gates)

### flag rates

| scope | n | flagged | unflagged | unlabelled | flag rate |
|---|---|---|---|---|---|
| stratum probe45 | 45 | 0 | 0 | 45 | n/a |
| stratum agent_nav | 33 | 10 | 23 | 0 | 0.303 |
| stratum synthetic_spill | 482 | 204 | 278 | 0 | 0.423 |
| all trace strata | 515 | 214 | 301 | 0 | 0.416 |
| writer A | 234 | 124 | 110 | 0 | 0.530 |
| writer B | 248 | 80 | 168 | 0 | 0.323 |
| intent_source assistant_text | 7 | 2 | 5 | 0 | 0.286 |
| intent_source bash_description | 26 | 8 | 18 | 0 | 0.308 |

### recall@k on the flagged vs the unflagged subset

| arm | subset | n | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| bge-large.dense | agent_nav:flagged | 10 | 0.300 | 0.900 | 0.900 | 1.000 |
| bge-large.dense | agent_nav:unflagged | 23 | 0.304 | 0.522 | 0.609 | 0.652 |
| bge-large.dense | synthetic_spill/writer=A:flagged | 124 | 0.508 | 0.653 | 0.726 | 0.758 |
| bge-large.dense | synthetic_spill/writer=A:unflagged | 110 | 0.255 | 0.391 | 0.436 | 0.518 |
| bge-large.dense | synthetic_spill/writer=B:flagged | 80 | 0.362 | 0.537 | 0.575 | 0.688 |
| bge-large.dense | synthetic_spill/writer=B:unflagged | 168 | 0.208 | 0.339 | 0.393 | 0.470 |
| bge-large.dense | synthetic_spill:flagged | 204 | 0.451 | 0.608 | 0.667 | 0.730 |
| bge-large.dense | synthetic_spill:unflagged | 278 | 0.227 | 0.360 | 0.410 | 0.489 |
| bge-large.dense | trace:flagged | 214 | 0.444 | 0.621 | 0.678 | 0.743 |
| bge-large.dense | trace:unflagged | 301 | 0.233 | 0.372 | 0.425 | 0.502 |
| bge-large.hybrid | agent_nav:flagged | 10 | 0.400 | 0.500 | 0.900 | 1.000 |
| bge-large.hybrid | agent_nav:unflagged | 23 | 0.348 | 0.565 | 0.652 | 0.783 |
| bge-large.hybrid | synthetic_spill/writer=A:flagged | 124 | 0.556 | 0.677 | 0.742 | 0.815 |
| bge-large.hybrid | synthetic_spill/writer=A:unflagged | 110 | 0.291 | 0.400 | 0.464 | 0.564 |
| bge-large.hybrid | synthetic_spill/writer=B:flagged | 80 | 0.425 | 0.575 | 0.637 | 0.688 |
| bge-large.hybrid | synthetic_spill/writer=B:unflagged | 168 | 0.202 | 0.315 | 0.393 | 0.488 |
| bge-large.hybrid | synthetic_spill:flagged | 204 | 0.505 | 0.637 | 0.701 | 0.765 |
| bge-large.hybrid | synthetic_spill:unflagged | 278 | 0.237 | 0.349 | 0.421 | 0.518 |
| bge-large.hybrid | trace:flagged | 214 | 0.500 | 0.631 | 0.710 | 0.776 |
| bge-large.hybrid | trace:unflagged | 301 | 0.246 | 0.365 | 0.439 | 0.538 |
| bm25 | agent_nav:flagged | 10 | 0.200 | 0.400 | 0.700 | 0.900 |
| bm25 | agent_nav:unflagged | 23 | 0.174 | 0.565 | 0.696 | 0.826 |
| bm25 | synthetic_spill/writer=A:flagged | 124 | 0.500 | 0.718 | 0.750 | 0.815 |
| bm25 | synthetic_spill/writer=A:unflagged | 110 | 0.282 | 0.445 | 0.500 | 0.564 |
| bm25 | synthetic_spill/writer=B:flagged | 80 | 0.338 | 0.500 | 0.575 | 0.637 |
| bm25 | synthetic_spill/writer=B:unflagged | 168 | 0.173 | 0.298 | 0.351 | 0.458 |
| bm25 | synthetic_spill:flagged | 204 | 0.436 | 0.632 | 0.681 | 0.745 |
| bm25 | synthetic_spill:unflagged | 278 | 0.216 | 0.356 | 0.410 | 0.500 |
| bm25 | trace:flagged | 214 | 0.425 | 0.621 | 0.682 | 0.752 |
| bm25 | trace:unflagged | 301 | 0.213 | 0.372 | 0.432 | 0.525 |
| eg2-gguf-bf16-256.dense | agent_nav:flagged | 10 | 0.000 | 0.600 | 0.700 | 0.700 |
| eg2-gguf-bf16-256.dense | agent_nav:unflagged | 23 | 0.261 | 0.522 | 0.565 | 0.652 |
| eg2-gguf-bf16-256.dense | synthetic_spill/writer=A:flagged | 124 | 0.500 | 0.774 | 0.823 | 0.879 |
| eg2-gguf-bf16-256.dense | synthetic_spill/writer=A:unflagged | 110 | 0.509 | 0.727 | 0.791 | 0.873 |
| eg2-gguf-bf16-256.dense | synthetic_spill/writer=B:flagged | 80 | 0.388 | 0.600 | 0.688 | 0.750 |
| eg2-gguf-bf16-256.dense | synthetic_spill/writer=B:unflagged | 168 | 0.292 | 0.524 | 0.619 | 0.750 |
| eg2-gguf-bf16-256.dense | synthetic_spill:flagged | 204 | 0.456 | 0.706 | 0.770 | 0.828 |
| eg2-gguf-bf16-256.dense | synthetic_spill:unflagged | 278 | 0.378 | 0.604 | 0.687 | 0.799 |
| eg2-gguf-bf16-256.dense | trace:flagged | 214 | 0.435 | 0.701 | 0.766 | 0.822 |
| eg2-gguf-bf16-256.dense | trace:unflagged | 301 | 0.369 | 0.598 | 0.678 | 0.787 |
| eg2-gguf-bf16-256.hybrid | agent_nav:flagged | 10 | 0.300 | 0.700 | 0.700 | 0.900 |
| eg2-gguf-bf16-256.hybrid | agent_nav:unflagged | 23 | 0.304 | 0.652 | 0.739 | 0.870 |
| eg2-gguf-bf16-256.hybrid | synthetic_spill/writer=A:flagged | 124 | 0.589 | 0.790 | 0.871 | 0.903 |
| eg2-gguf-bf16-256.hybrid | synthetic_spill/writer=A:unflagged | 110 | 0.473 | 0.682 | 0.791 | 0.864 |
| eg2-gguf-bf16-256.hybrid | synthetic_spill/writer=B:flagged | 80 | 0.412 | 0.600 | 0.713 | 0.800 |
| eg2-gguf-bf16-256.hybrid | synthetic_spill/writer=B:unflagged | 168 | 0.339 | 0.500 | 0.589 | 0.714 |
| eg2-gguf-bf16-256.hybrid | synthetic_spill:flagged | 204 | 0.520 | 0.716 | 0.809 | 0.863 |
| eg2-gguf-bf16-256.hybrid | synthetic_spill:unflagged | 278 | 0.392 | 0.572 | 0.669 | 0.773 |
| eg2-gguf-bf16-256.hybrid | trace:flagged | 214 | 0.509 | 0.715 | 0.804 | 0.864 |
| eg2-gguf-bf16-256.hybrid | trace:unflagged | 301 | 0.385 | 0.578 | 0.674 | 0.781 |
| eg2-gguf-bf16-768.dense | agent_nav:flagged | 10 | 0.100 | 0.700 | 0.700 | 0.800 |
| eg2-gguf-bf16-768.dense | agent_nav:unflagged | 23 | 0.348 | 0.522 | 0.609 | 0.696 |
| eg2-gguf-bf16-768.dense | synthetic_spill/writer=A:flagged | 124 | 0.589 | 0.798 | 0.831 | 0.887 |
| eg2-gguf-bf16-768.dense | synthetic_spill/writer=A:unflagged | 110 | 0.527 | 0.755 | 0.809 | 0.891 |
| eg2-gguf-bf16-768.dense | synthetic_spill/writer=B:flagged | 80 | 0.412 | 0.600 | 0.688 | 0.787 |
| eg2-gguf-bf16-768.dense | synthetic_spill/writer=B:unflagged | 168 | 0.339 | 0.565 | 0.655 | 0.756 |
| eg2-gguf-bf16-768.dense | synthetic_spill:flagged | 204 | 0.520 | 0.721 | 0.775 | 0.848 |
| eg2-gguf-bf16-768.dense | synthetic_spill:unflagged | 278 | 0.414 | 0.640 | 0.716 | 0.809 |
| eg2-gguf-bf16-768.dense | trace:flagged | 214 | 0.500 | 0.720 | 0.771 | 0.846 |
| eg2-gguf-bf16-768.dense | trace:unflagged | 301 | 0.409 | 0.631 | 0.708 | 0.801 |
| eg2-gguf-bf16-768.hybrid | agent_nav:flagged | 10 | 0.300 | 0.600 | 0.700 | 0.900 |
| eg2-gguf-bf16-768.hybrid | agent_nav:unflagged | 23 | 0.261 | 0.522 | 0.652 | 0.783 |
| eg2-gguf-bf16-768.hybrid | synthetic_spill/writer=A:flagged | 124 | 0.629 | 0.798 | 0.871 | 0.927 |
| eg2-gguf-bf16-768.hybrid | synthetic_spill/writer=A:unflagged | 110 | 0.500 | 0.655 | 0.727 | 0.873 |
| eg2-gguf-bf16-768.hybrid | synthetic_spill/writer=B:flagged | 80 | 0.487 | 0.637 | 0.700 | 0.787 |
| eg2-gguf-bf16-768.hybrid | synthetic_spill/writer=B:unflagged | 168 | 0.333 | 0.530 | 0.607 | 0.696 |
| eg2-gguf-bf16-768.hybrid | synthetic_spill:flagged | 204 | 0.574 | 0.735 | 0.804 | 0.873 |
| eg2-gguf-bf16-768.hybrid | synthetic_spill:unflagged | 278 | 0.399 | 0.579 | 0.655 | 0.766 |
| eg2-gguf-bf16-768.hybrid | trace:flagged | 214 | 0.561 | 0.729 | 0.799 | 0.874 |
| eg2-gguf-bf16-768.hybrid | trace:unflagged | 301 | 0.389 | 0.575 | 0.654 | 0.767 |
| granite-97m-r2.dense | agent_nav:flagged | 10 | 0.100 | 0.600 | 0.800 | 1.000 |
| granite-97m-r2.dense | agent_nav:unflagged | 23 | 0.348 | 0.522 | 0.652 | 0.739 |
| granite-97m-r2.dense | synthetic_spill/writer=A:flagged | 124 | 0.395 | 0.677 | 0.766 | 0.831 |
| granite-97m-r2.dense | synthetic_spill/writer=A:unflagged | 110 | 0.291 | 0.536 | 0.600 | 0.664 |
| granite-97m-r2.dense | synthetic_spill/writer=B:flagged | 80 | 0.325 | 0.512 | 0.588 | 0.700 |
| granite-97m-r2.dense | synthetic_spill/writer=B:unflagged | 168 | 0.173 | 0.345 | 0.440 | 0.571 |
| granite-97m-r2.dense | synthetic_spill:flagged | 204 | 0.368 | 0.613 | 0.696 | 0.779 |
| granite-97m-r2.dense | synthetic_spill:unflagged | 278 | 0.219 | 0.421 | 0.504 | 0.608 |
| granite-97m-r2.dense | trace:flagged | 214 | 0.355 | 0.612 | 0.701 | 0.790 |
| granite-97m-r2.dense | trace:unflagged | 301 | 0.229 | 0.429 | 0.515 | 0.618 |
| granite-97m-r2.hybrid | agent_nav:flagged | 10 | 0.300 | 0.600 | 1.000 | 1.000 |
| granite-97m-r2.hybrid | agent_nav:unflagged | 23 | 0.391 | 0.609 | 0.696 | 0.870 |
| granite-97m-r2.hybrid | synthetic_spill/writer=A:flagged | 124 | 0.565 | 0.742 | 0.806 | 0.879 |
| granite-97m-r2.hybrid | synthetic_spill/writer=A:unflagged | 110 | 0.382 | 0.564 | 0.664 | 0.736 |
| granite-97m-r2.hybrid | synthetic_spill/writer=B:flagged | 80 | 0.412 | 0.625 | 0.662 | 0.750 |
| granite-97m-r2.hybrid | synthetic_spill/writer=B:unflagged | 168 | 0.280 | 0.423 | 0.488 | 0.589 |
| granite-97m-r2.hybrid | synthetic_spill:flagged | 204 | 0.505 | 0.696 | 0.750 | 0.828 |
| granite-97m-r2.hybrid | synthetic_spill:unflagged | 278 | 0.320 | 0.478 | 0.558 | 0.647 |
| granite-97m-r2.hybrid | trace:flagged | 214 | 0.495 | 0.692 | 0.762 | 0.836 |
| granite-97m-r2.hybrid | trace:unflagged | 301 | 0.326 | 0.488 | 0.568 | 0.664 |
| grep | agent_nav:flagged | 10 | 0.200 | 0.300 | 0.500 | 0.700 |
| grep | agent_nav:unflagged | 23 | 0.130 | 0.261 | 0.435 | 0.565 |
| grep | synthetic_spill/writer=A:flagged | 124 | 0.532 | 0.710 | 0.782 | 0.863 |
| grep | synthetic_spill/writer=A:unflagged | 110 | 0.382 | 0.573 | 0.645 | 0.700 |
| grep | synthetic_spill/writer=B:flagged | 80 | 0.400 | 0.575 | 0.675 | 0.787 |
| grep | synthetic_spill/writer=B:unflagged | 168 | 0.214 | 0.369 | 0.452 | 0.530 |
| grep | synthetic_spill:flagged | 204 | 0.480 | 0.657 | 0.740 | 0.833 |
| grep | synthetic_spill:unflagged | 278 | 0.281 | 0.450 | 0.529 | 0.597 |
| grep | trace:flagged | 214 | 0.467 | 0.640 | 0.729 | 0.827 |
| grep | trace:unflagged | 301 | 0.269 | 0.435 | 0.522 | 0.595 |
| potion-base-8m.dense | agent_nav:flagged | 10 | 0.400 | 0.600 | 0.700 | 0.700 |
| potion-base-8m.dense | agent_nav:unflagged | 23 | 0.261 | 0.391 | 0.478 | 0.565 |
| potion-base-8m.dense | synthetic_spill/writer=A:flagged | 124 | 0.258 | 0.508 | 0.597 | 0.637 |
| potion-base-8m.dense | synthetic_spill/writer=A:unflagged | 110 | 0.127 | 0.273 | 0.327 | 0.455 |
| potion-base-8m.dense | synthetic_spill/writer=B:flagged | 80 | 0.175 | 0.438 | 0.512 | 0.562 |
| potion-base-8m.dense | synthetic_spill/writer=B:unflagged | 168 | 0.131 | 0.250 | 0.345 | 0.440 |
| potion-base-8m.dense | synthetic_spill:flagged | 204 | 0.225 | 0.480 | 0.564 | 0.608 |
| potion-base-8m.dense | synthetic_spill:unflagged | 278 | 0.129 | 0.259 | 0.338 | 0.446 |
| potion-base-8m.dense | trace:flagged | 214 | 0.234 | 0.486 | 0.570 | 0.612 |
| potion-base-8m.dense | trace:unflagged | 301 | 0.140 | 0.269 | 0.349 | 0.455 |
| potion-base-8m.hybrid | agent_nav:flagged | 10 | 0.400 | 0.400 | 0.600 | 0.800 |
| potion-base-8m.hybrid | agent_nav:unflagged | 23 | 0.217 | 0.522 | 0.652 | 0.739 |
| potion-base-8m.hybrid | synthetic_spill/writer=A:flagged | 124 | 0.419 | 0.629 | 0.702 | 0.758 |
| potion-base-8m.hybrid | synthetic_spill/writer=A:unflagged | 110 | 0.227 | 0.355 | 0.455 | 0.527 |
| potion-base-8m.hybrid | synthetic_spill/writer=B:flagged | 80 | 0.300 | 0.525 | 0.600 | 0.637 |
| potion-base-8m.hybrid | synthetic_spill/writer=B:unflagged | 168 | 0.185 | 0.292 | 0.399 | 0.518 |
| potion-base-8m.hybrid | synthetic_spill:flagged | 204 | 0.373 | 0.588 | 0.662 | 0.711 |
| potion-base-8m.hybrid | synthetic_spill:unflagged | 278 | 0.201 | 0.317 | 0.421 | 0.522 |
| potion-base-8m.hybrid | trace:flagged | 214 | 0.374 | 0.579 | 0.659 | 0.715 |
| potion-base-8m.hybrid | trace:unflagged | 301 | 0.203 | 0.332 | 0.439 | 0.538 |
| potion-base-8m.whybrid | agent_nav:flagged | 10 | 0.300 | 0.500 | 0.700 | 0.800 |
| potion-base-8m.whybrid | agent_nav:unflagged | 23 | 0.304 | 0.478 | 0.609 | 0.783 |
| potion-base-8m.whybrid | synthetic_spill/writer=A:flagged | 124 | 0.371 | 0.597 | 0.661 | 0.710 |
| potion-base-8m.whybrid | synthetic_spill/writer=A:unflagged | 110 | 0.191 | 0.318 | 0.418 | 0.545 |
| potion-base-8m.whybrid | synthetic_spill/writer=B:flagged | 80 | 0.237 | 0.537 | 0.575 | 0.637 |
| potion-base-8m.whybrid | synthetic_spill/writer=B:unflagged | 168 | 0.179 | 0.256 | 0.387 | 0.494 |
| potion-base-8m.whybrid | synthetic_spill:flagged | 204 | 0.319 | 0.574 | 0.627 | 0.681 |
| potion-base-8m.whybrid | synthetic_spill:unflagged | 278 | 0.183 | 0.281 | 0.399 | 0.514 |
| potion-base-8m.whybrid | trace:flagged | 214 | 0.318 | 0.570 | 0.631 | 0.687 |
| potion-base-8m.whybrid | trace:unflagged | 301 | 0.193 | 0.296 | 0.415 | 0.535 |
| potion-code-16m-v2.dense | agent_nav:flagged | 10 | 0.200 | 0.800 | 0.900 | 0.900 |
| potion-code-16m-v2.dense | agent_nav:unflagged | 23 | 0.217 | 0.435 | 0.522 | 0.652 |
| potion-code-16m-v2.dense | synthetic_spill/writer=A:flagged | 124 | 0.444 | 0.621 | 0.685 | 0.790 |
| potion-code-16m-v2.dense | synthetic_spill/writer=A:unflagged | 110 | 0.245 | 0.373 | 0.427 | 0.545 |
| potion-code-16m-v2.dense | synthetic_spill/writer=B:flagged | 80 | 0.300 | 0.475 | 0.537 | 0.637 |
| potion-code-16m-v2.dense | synthetic_spill/writer=B:unflagged | 168 | 0.208 | 0.333 | 0.399 | 0.494 |
| potion-code-16m-v2.dense | synthetic_spill:flagged | 204 | 0.387 | 0.564 | 0.627 | 0.730 |
| potion-code-16m-v2.dense | synthetic_spill:unflagged | 278 | 0.223 | 0.349 | 0.410 | 0.514 |
| potion-code-16m-v2.dense | trace:flagged | 214 | 0.379 | 0.575 | 0.640 | 0.738 |
| potion-code-16m-v2.dense | trace:unflagged | 301 | 0.223 | 0.355 | 0.419 | 0.525 |
| potion-code-16m-v2.hybrid | agent_nav:flagged | 10 | 0.300 | 0.400 | 0.800 | 1.000 |
| potion-code-16m-v2.hybrid | agent_nav:unflagged | 23 | 0.130 | 0.522 | 0.652 | 0.783 |
| potion-code-16m-v2.hybrid | synthetic_spill/writer=A:flagged | 124 | 0.492 | 0.734 | 0.774 | 0.847 |
| potion-code-16m-v2.hybrid | synthetic_spill/writer=A:unflagged | 110 | 0.245 | 0.400 | 0.464 | 0.600 |
| potion-code-16m-v2.hybrid | synthetic_spill/writer=B:flagged | 80 | 0.325 | 0.487 | 0.613 | 0.675 |
| potion-code-16m-v2.hybrid | synthetic_spill/writer=B:unflagged | 168 | 0.220 | 0.333 | 0.440 | 0.577 |
| potion-code-16m-v2.hybrid | synthetic_spill:flagged | 204 | 0.426 | 0.637 | 0.711 | 0.779 |
| potion-code-16m-v2.hybrid | synthetic_spill:unflagged | 278 | 0.230 | 0.360 | 0.450 | 0.586 |
| potion-code-16m-v2.hybrid | trace:flagged | 214 | 0.421 | 0.626 | 0.715 | 0.790 |
| potion-code-16m-v2.hybrid | trace:unflagged | 301 | 0.223 | 0.372 | 0.465 | 0.601 |
| potion-code-16m-v2.whybrid | agent_nav:flagged | 10 | 0.400 | 0.600 | 0.900 | 1.000 |
| potion-code-16m-v2.whybrid | agent_nav:unflagged | 23 | 0.217 | 0.478 | 0.565 | 0.739 |
| potion-code-16m-v2.whybrid | synthetic_spill/writer=A:flagged | 124 | 0.452 | 0.677 | 0.742 | 0.823 |
| potion-code-16m-v2.whybrid | synthetic_spill/writer=A:unflagged | 110 | 0.218 | 0.345 | 0.491 | 0.600 |
| potion-code-16m-v2.whybrid | synthetic_spill/writer=B:flagged | 80 | 0.325 | 0.500 | 0.575 | 0.662 |
| potion-code-16m-v2.whybrid | synthetic_spill/writer=B:unflagged | 168 | 0.214 | 0.345 | 0.440 | 0.536 |
| potion-code-16m-v2.whybrid | synthetic_spill:flagged | 204 | 0.402 | 0.608 | 0.676 | 0.760 |
| potion-code-16m-v2.whybrid | synthetic_spill:unflagged | 278 | 0.216 | 0.345 | 0.460 | 0.561 |
| potion-code-16m-v2.whybrid | trace:flagged | 214 | 0.402 | 0.607 | 0.687 | 0.771 |
| potion-code-16m-v2.whybrid | trace:unflagged | 301 | 0.216 | 0.355 | 0.468 | 0.575 |

### flagged minus unflagged recall (the leakage effect)

| arm | scope | n flagged | n unflagged | dR@1 | dR@3 | dR@5 | dR@10 |
|---|---|---|---|---|---|---|---|
| bge-large.dense | agent_nav | 10 | 23 | -0.004 | +0.378 | +0.291 | +0.348 |
| bge-large.dense | trace | 214 | 301 | +0.211 | +0.249 | +0.252 | +0.241 |
| bge-large.dense | agent_nav/intent_source=bash_description | 8 | 18 | +0.097 | +0.500 | +0.389 | +0.333 |
| bge-large.dense | agent_nav/intent_source=assistant_text | 2 | 5 | -0.400 | -0.100 | -0.100 | +0.400 |
| bge-large.dense | synthetic_spill | 204 | 278 | +0.224 | +0.248 | +0.257 | +0.241 |
| bge-large.dense | synthetic_spill/writer=A | 124 | 110 | +0.254 | +0.262 | +0.289 | +0.240 |
| bge-large.dense | synthetic_spill/writer=B | 80 | 168 | +0.154 | +0.198 | +0.182 | +0.217 |
| bge-large.hybrid | agent_nav | 10 | 23 | +0.052 | -0.065 | +0.248 | +0.217 |
| bge-large.hybrid | trace | 214 | 301 | +0.254 | +0.265 | +0.272 | +0.237 |
| bge-large.hybrid | agent_nav/intent_source=bash_description | 8 | 18 | +0.167 | +0.014 | +0.333 | +0.222 |
| bge-large.hybrid | agent_nav/intent_source=assistant_text | 2 | 5 | -0.400 | -0.400 | -0.100 | +0.200 |
| bge-large.hybrid | synthetic_spill | 204 | 278 | +0.267 | +0.288 | +0.280 | +0.247 |
| bge-large.hybrid | synthetic_spill/writer=A | 124 | 110 | +0.266 | +0.277 | +0.278 | +0.251 |
| bge-large.hybrid | synthetic_spill/writer=B | 80 | 168 | +0.223 | +0.260 | +0.245 | +0.199 |
| bm25 | agent_nav | 10 | 23 | +0.026 | -0.165 | +0.004 | +0.074 |
| bm25 | trace | 214 | 301 | +0.213 | +0.249 | +0.250 | +0.227 |
| bm25 | agent_nav/intent_source=bash_description | 8 | 18 | +0.083 | -0.236 | -0.097 | +0.042 |
| bm25 | agent_nav/intent_source=assistant_text | 2 | 5 | -0.200 | +0.100 | +0.400 | +0.200 |
| bm25 | synthetic_spill | 204 | 278 | +0.220 | +0.276 | +0.271 | +0.245 |
| bm25 | synthetic_spill/writer=A | 124 | 110 | +0.218 | +0.272 | +0.250 | +0.251 |
| bm25 | synthetic_spill/writer=B | 80 | 168 | +0.165 | +0.202 | +0.224 | +0.179 |
| eg2-gguf-bf16-256.dense | agent_nav | 10 | 23 | -0.261 | +0.078 | +0.135 | +0.048 |
| eg2-gguf-bf16-256.dense | trace | 214 | 301 | +0.066 | +0.103 | +0.089 | +0.035 |
| eg2-gguf-bf16-256.dense | agent_nav/intent_source=bash_description | 8 | 18 | -0.222 | -0.056 | +0.069 | -0.042 |
| eg2-gguf-bf16-256.dense | agent_nav/intent_source=assistant_text | 2 | 5 | -0.400 | +0.600 | +0.400 | +0.400 |
| eg2-gguf-bf16-256.dense | synthetic_spill | 204 | 278 | +0.078 | +0.102 | +0.083 | +0.030 |
| eg2-gguf-bf16-256.dense | synthetic_spill/writer=A | 124 | 110 | -0.009 | +0.047 | +0.032 | +0.006 |
| eg2-gguf-bf16-256.dense | synthetic_spill/writer=B | 80 | 168 | +0.096 | +0.076 | +0.068 | +0.000 |
| eg2-gguf-bf16-256.hybrid | agent_nav | 10 | 23 | -0.004 | +0.048 | -0.039 | +0.030 |
| eg2-gguf-bf16-256.hybrid | trace | 214 | 301 | +0.124 | +0.137 | +0.129 | +0.084 |
| eg2-gguf-bf16-256.hybrid | agent_nav/intent_source=bash_description | 8 | 18 | -0.028 | -0.042 | -0.153 | -0.014 |
| eg2-gguf-bf16-256.hybrid | agent_nav/intent_source=assistant_text | 2 | 5 | +0.100 | +0.400 | +0.400 | +0.200 |
| eg2-gguf-bf16-256.hybrid | synthetic_spill | 204 | 278 | +0.128 | +0.144 | +0.140 | +0.089 |
| eg2-gguf-bf16-256.hybrid | synthetic_spill/writer=A | 124 | 110 | +0.116 | +0.109 | +0.080 | +0.040 |
| eg2-gguf-bf16-256.hybrid | synthetic_spill/writer=B | 80 | 168 | +0.073 | +0.100 | +0.123 | +0.086 |
| eg2-gguf-bf16-768.dense | agent_nav | 10 | 23 | -0.248 | +0.178 | +0.091 | +0.104 |
| eg2-gguf-bf16-768.dense | trace | 214 | 301 | +0.091 | +0.088 | +0.063 | +0.045 |
| eg2-gguf-bf16-768.dense | agent_nav/intent_source=bash_description | 8 | 18 | -0.389 | +0.014 | +0.014 | +0.028 |
| eg2-gguf-bf16-768.dense | agent_nav/intent_source=assistant_text | 2 | 5 | +0.300 | +0.800 | +0.400 | +0.400 |
| eg2-gguf-bf16-768.dense | synthetic_spill | 204 | 278 | +0.106 | +0.080 | +0.059 | +0.039 |
| eg2-gguf-bf16-768.dense | synthetic_spill/writer=A | 124 | 110 | +0.061 | +0.044 | +0.022 | -0.004 |
| eg2-gguf-bf16-768.dense | synthetic_spill/writer=B | 80 | 168 | +0.073 | +0.035 | +0.033 | +0.032 |
| eg2-gguf-bf16-768.hybrid | agent_nav | 10 | 23 | +0.039 | +0.078 | +0.048 | +0.117 |
| eg2-gguf-bf16-768.hybrid | trace | 214 | 301 | +0.172 | +0.154 | +0.145 | +0.106 |
| eg2-gguf-bf16-768.hybrid | agent_nav/intent_source=bash_description | 8 | 18 | -0.028 | +0.000 | -0.042 | +0.042 |
| eg2-gguf-bf16-768.hybrid | agent_nav/intent_source=assistant_text | 2 | 5 | +0.300 | +0.400 | +0.400 | +0.400 |
| eg2-gguf-bf16-768.hybrid | synthetic_spill | 204 | 278 | +0.174 | +0.156 | +0.149 | +0.106 |
| eg2-gguf-bf16-768.hybrid | synthetic_spill/writer=A | 124 | 110 | +0.129 | +0.144 | +0.144 | +0.055 |
| eg2-gguf-bf16-768.hybrid | synthetic_spill/writer=B | 80 | 168 | +0.154 | +0.108 | +0.093 | +0.091 |
| granite-97m-r2.dense | agent_nav | 10 | 23 | -0.248 | +0.078 | +0.148 | +0.261 |
| granite-97m-r2.dense | trace | 214 | 301 | +0.126 | +0.184 | +0.186 | +0.172 |
| granite-97m-r2.dense | agent_nav/intent_source=bash_description | 8 | 18 | -0.208 | -0.056 | +0.083 | +0.222 |
| granite-97m-r2.dense | agent_nav/intent_source=assistant_text | 2 | 5 | -0.400 | +0.600 | +0.400 | +0.400 |
| granite-97m-r2.dense | synthetic_spill | 204 | 278 | +0.148 | +0.192 | +0.192 | +0.171 |
| granite-97m-r2.dense | synthetic_spill/writer=A | 124 | 110 | +0.104 | +0.141 | +0.166 | +0.167 |
| granite-97m-r2.dense | synthetic_spill/writer=B | 80 | 168 | +0.152 | +0.167 | +0.147 | +0.129 |
| granite-97m-r2.hybrid | agent_nav | 10 | 23 | -0.091 | -0.009 | +0.304 | +0.130 |
| granite-97m-r2.hybrid | trace | 214 | 301 | +0.170 | +0.203 | +0.194 | +0.172 |
| granite-97m-r2.hybrid | agent_nav/intent_source=bash_description | 8 | 18 | -0.069 | +0.139 | +0.278 | +0.111 |
| granite-97m-r2.hybrid | agent_nav/intent_source=assistant_text | 2 | 5 | -0.200 | -0.600 | +0.400 | +0.200 |
| granite-97m-r2.hybrid | synthetic_spill | 204 | 278 | +0.185 | +0.218 | +0.192 | +0.181 |
| granite-97m-r2.hybrid | synthetic_spill/writer=A | 124 | 110 | +0.183 | +0.178 | +0.143 | +0.143 |
| granite-97m-r2.hybrid | synthetic_spill/writer=B | 80 | 168 | +0.133 | +0.202 | +0.174 | +0.161 |
| grep | agent_nav | 10 | 23 | +0.070 | +0.039 | +0.065 | +0.135 |
| grep | trace | 214 | 301 | +0.198 | +0.205 | +0.207 | +0.232 |
| grep | agent_nav/intent_source=bash_description | 8 | 18 | +0.083 | +0.042 | +0.111 | +0.069 |
| grep | agent_nav/intent_source=assistant_text | 2 | 5 | +0.000 | +0.000 | -0.100 | +0.400 |
| grep | synthetic_spill | 204 | 278 | +0.200 | +0.207 | +0.211 | +0.236 |
| grep | synthetic_spill/writer=A | 124 | 110 | +0.150 | +0.137 | +0.137 | +0.163 |
| grep | synthetic_spill/writer=B | 80 | 168 | +0.186 | +0.206 | +0.223 | +0.258 |
| potion-base-8m.dense | agent_nav | 10 | 23 | +0.139 | +0.209 | +0.222 | +0.135 |
| potion-base-8m.dense | trace | 214 | 301 | +0.094 | +0.217 | +0.221 | +0.157 |
| potion-base-8m.dense | agent_nav/intent_source=bash_description | 8 | 18 | +0.222 | +0.292 | +0.306 | +0.194 |
| potion-base-8m.dense | agent_nav/intent_source=assistant_text | 2 | 5 | -0.200 | -0.100 | -0.100 | -0.100 |
| potion-base-8m.dense | synthetic_spill | 204 | 278 | +0.096 | +0.221 | +0.226 | +0.162 |
| potion-base-8m.dense | synthetic_spill/writer=A | 124 | 110 | +0.131 | +0.235 | +0.270 | +0.183 |
| potion-base-8m.dense | synthetic_spill/writer=B | 80 | 168 | +0.044 | +0.188 | +0.167 | +0.122 |
| potion-base-8m.hybrid | agent_nav | 10 | 23 | +0.183 | -0.122 | -0.052 | +0.061 |
| potion-base-8m.hybrid | trace | 214 | 301 | +0.171 | +0.247 | +0.220 | +0.177 |
| potion-base-8m.hybrid | agent_nav/intent_source=bash_description | 8 | 18 | +0.278 | -0.056 | -0.167 | +0.028 |
| potion-base-8m.hybrid | agent_nav/intent_source=assistant_text | 2 | 5 | -0.200 | -0.400 | +0.400 | +0.200 |
| potion-base-8m.hybrid | synthetic_spill | 204 | 278 | +0.171 | +0.272 | +0.241 | +0.189 |
| potion-base-8m.hybrid | synthetic_spill/writer=A | 124 | 110 | +0.192 | +0.274 | +0.247 | +0.231 |
| potion-base-8m.hybrid | synthetic_spill/writer=B | 80 | 168 | +0.115 | +0.233 | +0.201 | +0.120 |
| potion-base-8m.whybrid | agent_nav | 10 | 23 | -0.004 | +0.022 | +0.091 | +0.017 |
| potion-base-8m.whybrid | trace | 214 | 301 | +0.125 | +0.274 | +0.216 | +0.152 |
| potion-base-8m.whybrid | agent_nav/intent_source=bash_description | 8 | 18 | +0.042 | +0.181 | +0.139 | -0.028 |
| potion-base-8m.whybrid | agent_nav/intent_source=assistant_text | 2 | 5 | -0.200 | -0.600 | -0.100 | +0.200 |
| potion-base-8m.whybrid | synthetic_spill | 204 | 278 | +0.135 | +0.293 | +0.228 | +0.167 |
| potion-base-8m.whybrid | synthetic_spill/writer=A | 124 | 110 | +0.180 | +0.279 | +0.243 | +0.164 |
| potion-base-8m.whybrid | synthetic_spill/writer=B | 80 | 168 | +0.059 | +0.282 | +0.188 | +0.143 |
| potion-code-16m-v2.dense | agent_nav | 10 | 23 | -0.017 | +0.365 | +0.378 | +0.248 |
| potion-code-16m-v2.dense | trace | 214 | 301 | +0.156 | +0.219 | +0.222 | +0.213 |
| potion-code-16m-v2.dense | agent_nav/intent_source=bash_description | 8 | 18 | +0.028 | +0.431 | +0.319 | +0.208 |
| potion-code-16m-v2.dense | agent_nav/intent_source=assistant_text | 2 | 5 | -0.200 | +0.100 | +0.600 | +0.400 |
| potion-code-16m-v2.dense | synthetic_spill | 204 | 278 | +0.164 | +0.215 | +0.217 | +0.216 |
| potion-code-16m-v2.dense | synthetic_spill/writer=A | 124 | 110 | +0.198 | +0.248 | +0.258 | +0.245 |
| potion-code-16m-v2.dense | synthetic_spill/writer=B | 80 | 168 | +0.092 | +0.142 | +0.139 | +0.143 |
| potion-code-16m-v2.hybrid | agent_nav | 10 | 23 | +0.170 | -0.122 | +0.148 | +0.217 |
| potion-code-16m-v2.hybrid | trace | 214 | 301 | +0.198 | +0.254 | +0.250 | +0.188 |
| potion-code-16m-v2.hybrid | agent_nav/intent_source=bash_description | 8 | 18 | +0.264 | -0.111 | +0.083 | +0.222 |
| potion-code-16m-v2.hybrid | agent_nav/intent_source=assistant_text | 2 | 5 | -0.200 | -0.200 | +0.400 | +0.200 |
| potion-code-16m-v2.hybrid | synthetic_spill | 204 | 278 | +0.196 | +0.278 | +0.261 | +0.193 |
| potion-code-16m-v2.hybrid | synthetic_spill/writer=A | 124 | 110 | +0.246 | +0.334 | +0.311 | +0.247 |
| potion-code-16m-v2.hybrid | synthetic_spill/writer=B | 80 | 168 | +0.105 | +0.154 | +0.172 | +0.098 |
| potion-code-16m-v2.whybrid | agent_nav | 10 | 23 | +0.183 | +0.122 | +0.335 | +0.261 |
| potion-code-16m-v2.whybrid | trace | 214 | 301 | +0.186 | +0.252 | +0.218 | +0.196 |
| potion-code-16m-v2.whybrid | agent_nav/intent_source=bash_description | 8 | 18 | +0.278 | +0.069 | +0.264 | +0.278 |
| potion-code-16m-v2.whybrid | agent_nav/intent_source=assistant_text | 2 | 5 | -0.200 | +0.300 | +0.600 | +0.200 |
| potion-code-16m-v2.whybrid | synthetic_spill | 204 | 278 | +0.186 | +0.263 | +0.216 | +0.199 |
| potion-code-16m-v2.whybrid | synthetic_spill/writer=A | 124 | 110 | +0.233 | +0.332 | +0.251 | +0.223 |
| potion-code-16m-v2.whybrid | synthetic_spill/writer=B | 80 | 168 | +0.111 | +0.155 | +0.135 | +0.127 |

### hybrid minus L (grep) on each subset, recall@5

| subset | n | arm | diff | CI |
|---|---|---|---|---|
| agent_nav:flagged | 10 | bge-large.hybrid | +0.400 | [+0.100, +0.700] |
| agent_nav:flagged | 10 | eg2-gguf-bf16-256.hybrid | +0.200 | [-0.200, +0.600] |
| agent_nav:flagged | 10 | eg2-gguf-bf16-768.hybrid | +0.200 | [-0.200, +0.600] |
| agent_nav:flagged | 10 | granite-97m-r2.hybrid | +0.500 | [+0.200, +0.800] |
| agent_nav:flagged | 10 | potion-base-8m.hybrid | +0.100 | [-0.200, +0.400] |
| agent_nav:flagged | 10 | potion-code-16m-v2.hybrid | +0.300 | [+0.000, +0.600] |
| trace:flagged | 214 | bge-large.hybrid | -0.019 | [-0.079, +0.042] |
| trace:flagged | 214 | eg2-gguf-bf16-256.hybrid | +0.075 | [+0.019, +0.136] |
| trace:flagged | 214 | eg2-gguf-bf16-768.hybrid | +0.070 | [+0.014, +0.131] |
| trace:flagged | 214 | granite-97m-r2.hybrid | +0.033 | [-0.028, +0.093] |
| trace:flagged | 214 | potion-base-8m.hybrid | -0.070 | [-0.140, +0.000] |
| trace:flagged | 214 | potion-code-16m-v2.hybrid | -0.014 | [-0.079, +0.051] |
| agent_nav/intent_source=bash_description:flagged | 8 | bge-large.hybrid | +0.500 | [+0.125, +0.875] |
| agent_nav/intent_source=bash_description:flagged | 8 | eg2-gguf-bf16-256.hybrid | +0.125 | [-0.250, +0.500] |
| agent_nav/intent_source=bash_description:flagged | 8 | eg2-gguf-bf16-768.hybrid | +0.125 | [-0.250, +0.500] |
| agent_nav/intent_source=bash_description:flagged | 8 | granite-97m-r2.hybrid | +0.500 | [+0.125, +0.875] |
| agent_nav/intent_source=bash_description:flagged | 8 | potion-base-8m.hybrid | +0.000 | [-0.375, +0.375] |
| agent_nav/intent_source=bash_description:flagged | 8 | potion-code-16m-v2.hybrid | +0.250 | [+0.000, +0.625] |
| agent_nav:unflagged | 23 | bge-large.hybrid | +0.217 | [+0.000, +0.435] |
| agent_nav:unflagged | 23 | eg2-gguf-bf16-256.hybrid | +0.304 | [+0.087, +0.522] |
| agent_nav:unflagged | 23 | eg2-gguf-bf16-768.hybrid | +0.217 | [+0.000, +0.435] |
| agent_nav:unflagged | 23 | granite-97m-r2.hybrid | +0.261 | [+0.043, +0.478] |
| agent_nav:unflagged | 23 | potion-base-8m.hybrid | +0.217 | [-0.043, +0.435] |
| agent_nav:unflagged | 23 | potion-code-16m-v2.hybrid | +0.217 | [-0.043, +0.478] |
| trace:unflagged | 301 | bge-large.hybrid | -0.083 | [-0.143, -0.020] |
| trace:unflagged | 301 | eg2-gguf-bf16-256.hybrid | +0.153 | [+0.096, +0.209] |
| trace:unflagged | 301 | eg2-gguf-bf16-768.hybrid | +0.133 | [+0.076, +0.193] |
| trace:unflagged | 301 | granite-97m-r2.hybrid | +0.047 | [-0.010, +0.106] |
| trace:unflagged | 301 | potion-base-8m.hybrid | -0.083 | [-0.146, -0.020] |
| trace:unflagged | 301 | potion-code-16m-v2.hybrid | -0.056 | [-0.120, +0.007] |
| agent_nav/intent_source=bash_description:unflagged | 18 | bge-large.hybrid | +0.278 | [+0.000, +0.556] |
| agent_nav/intent_source=bash_description:unflagged | 18 | eg2-gguf-bf16-256.hybrid | +0.389 | [+0.167, +0.611] |
| agent_nav/intent_source=bash_description:unflagged | 18 | eg2-gguf-bf16-768.hybrid | +0.278 | [+0.111, +0.500] |
| agent_nav/intent_source=bash_description:unflagged | 18 | granite-97m-r2.hybrid | +0.333 | [+0.111, +0.556] |
| agent_nav/intent_source=bash_description:unflagged | 18 | potion-base-8m.hybrid | +0.278 | [+0.000, +0.556] |
| agent_nav/intent_source=bash_description:unflagged | 18 | potion-code-16m-v2.hybrid | +0.278 | [+0.000, +0.556] |
| agent_nav/intent_source=assistant_text:unflagged | 5 | bge-large.hybrid | +0.000 | [+0.000, +0.000] |
| agent_nav/intent_source=assistant_text:unflagged | 5 | eg2-gguf-bf16-256.hybrid | +0.000 | [-0.600, +0.600] |
| agent_nav/intent_source=assistant_text:unflagged | 5 | eg2-gguf-bf16-768.hybrid | +0.000 | [-0.600, +0.600] |
| agent_nav/intent_source=assistant_text:unflagged | 5 | granite-97m-r2.hybrid | +0.000 | [-0.600, +0.600] |
| agent_nav/intent_source=assistant_text:unflagged | 5 | potion-base-8m.hybrid | +0.000 | [-0.600, +0.600] |
| agent_nav/intent_source=assistant_text:unflagged | 5 | potion-code-16m-v2.hybrid | +0.000 | [-0.600, +0.600] |
| agent_nav/intent_source=assistant_text:flagged | 2 | bge-large.hybrid | +0.000 | [+0.000, +0.000] |
| agent_nav/intent_source=assistant_text:flagged | 2 | eg2-gguf-bf16-256.hybrid | +0.500 | [+0.000, +1.000] |
| agent_nav/intent_source=assistant_text:flagged | 2 | eg2-gguf-bf16-768.hybrid | +0.500 | [+0.000, +1.000] |
| agent_nav/intent_source=assistant_text:flagged | 2 | granite-97m-r2.hybrid | +0.500 | [+0.000, +1.000] |
| agent_nav/intent_source=assistant_text:flagged | 2 | potion-base-8m.hybrid | +0.500 | [+0.000, +1.000] |
| agent_nav/intent_source=assistant_text:flagged | 2 | potion-code-16m-v2.hybrid | +0.500 | [+0.000, +1.000] |
| synthetic_spill:unflagged | 278 | bge-large.hybrid | -0.108 | [-0.173, -0.043] |
| synthetic_spill:unflagged | 278 | eg2-gguf-bf16-256.hybrid | +0.140 | [+0.083, +0.201] |
| synthetic_spill:unflagged | 278 | eg2-gguf-bf16-768.hybrid | +0.126 | [+0.068, +0.187] |
| synthetic_spill:unflagged | 278 | granite-97m-r2.hybrid | +0.029 | [-0.029, +0.086] |
| synthetic_spill:unflagged | 278 | potion-base-8m.hybrid | -0.108 | [-0.173, -0.043] |
| synthetic_spill:unflagged | 278 | potion-code-16m-v2.hybrid | -0.079 | [-0.147, -0.014] |
| synthetic_spill/writer=A:unflagged | 110 | bge-large.hybrid | -0.182 | [-0.282, -0.082] |
| synthetic_spill/writer=A:unflagged | 110 | eg2-gguf-bf16-256.hybrid | +0.145 | [+0.055, +0.236] |
| synthetic_spill/writer=A:unflagged | 110 | eg2-gguf-bf16-768.hybrid | +0.082 | [-0.009, +0.173] |
| synthetic_spill/writer=A:unflagged | 110 | granite-97m-r2.hybrid | +0.018 | [-0.073, +0.109] |
| synthetic_spill/writer=A:unflagged | 110 | potion-base-8m.hybrid | -0.191 | [-0.291, -0.091] |
| synthetic_spill/writer=A:unflagged | 110 | potion-code-16m-v2.hybrid | -0.182 | [-0.282, -0.082] |
| synthetic_spill:flagged | 204 | bge-large.hybrid | -0.039 | [-0.098, +0.020] |
| synthetic_spill:flagged | 204 | eg2-gguf-bf16-256.hybrid | +0.069 | [+0.010, +0.127] |
| synthetic_spill:flagged | 204 | eg2-gguf-bf16-768.hybrid | +0.064 | [+0.005, +0.123] |
| synthetic_spill:flagged | 204 | granite-97m-r2.hybrid | +0.010 | [-0.049, +0.069] |
| synthetic_spill:flagged | 204 | potion-base-8m.hybrid | -0.078 | [-0.147, -0.005] |
| synthetic_spill:flagged | 204 | potion-code-16m-v2.hybrid | -0.029 | [-0.093, +0.034] |
| synthetic_spill/writer=A:flagged | 124 | bge-large.hybrid | -0.040 | [-0.113, +0.032] |
| synthetic_spill/writer=A:flagged | 124 | eg2-gguf-bf16-256.hybrid | +0.089 | [+0.016, +0.161] |
| synthetic_spill/writer=A:flagged | 124 | eg2-gguf-bf16-768.hybrid | +0.089 | [+0.024, +0.153] |
| synthetic_spill/writer=A:flagged | 124 | granite-97m-r2.hybrid | +0.024 | [-0.048, +0.089] |
| synthetic_spill/writer=A:flagged | 124 | potion-base-8m.hybrid | -0.081 | [-0.177, +0.008] |
| synthetic_spill/writer=A:flagged | 124 | potion-code-16m-v2.hybrid | -0.008 | [-0.097, +0.073] |
| synthetic_spill/writer=B:flagged | 80 | bge-large.hybrid | -0.037 | [-0.138, +0.062] |
| synthetic_spill/writer=B:flagged | 80 | eg2-gguf-bf16-256.hybrid | +0.037 | [-0.062, +0.138] |
| synthetic_spill/writer=B:flagged | 80 | eg2-gguf-bf16-768.hybrid | +0.025 | [-0.075, +0.125] |
| synthetic_spill/writer=B:flagged | 80 | granite-97m-r2.hybrid | -0.013 | [-0.113, +0.087] |
| synthetic_spill/writer=B:flagged | 80 | potion-base-8m.hybrid | -0.075 | [-0.188, +0.037] |
| synthetic_spill/writer=B:flagged | 80 | potion-code-16m-v2.hybrid | -0.062 | [-0.163, +0.037] |
| synthetic_spill/writer=B:unflagged | 168 | bge-large.hybrid | -0.060 | [-0.143, +0.024] |
| synthetic_spill/writer=B:unflagged | 168 | eg2-gguf-bf16-256.hybrid | +0.137 | [+0.060, +0.214] |
| synthetic_spill/writer=B:unflagged | 168 | eg2-gguf-bf16-768.hybrid | +0.155 | [+0.077, +0.232] |
| synthetic_spill/writer=B:unflagged | 168 | granite-97m-r2.hybrid | +0.036 | [-0.048, +0.113] |
| synthetic_spill/writer=B:unflagged | 168 | potion-base-8m.hybrid | -0.054 | [-0.137, +0.030] |
| synthetic_spill/writer=B:unflagged | 168 | potion-code-16m-v2.hybrid | -0.012 | [-0.095, +0.071] |

# SECONDARY (non-deciding): pooled, flagged included (n=560)

**L - pooled, flagged included** = grep (grep 0.611, bm25 0.539, tie=False)

## recall@k per arm and stratum (Wilson 95%) - pooled, flagged included

| arm | stratum | n | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| bge-large.dense | agent_nav | 33 | 0.303 [0.174, 0.473] | 0.636 [0.466, 0.778] | 0.697 [0.527, 0.826] | 0.758 [0.590, 0.872] |
| bge-large.dense | probe45 | 45 | 0.556 [0.412, 0.691] | 0.667 [0.521, 0.786] | 0.733 [0.590, 0.840] | 0.778 [0.637, 0.875] |
| bge-large.dense | synthetic_spill | 482 | 0.322 [0.281, 0.365] | 0.465 [0.421, 0.509] | 0.519 [0.474, 0.563] | 0.591 [0.547, 0.634] |
| bge-large.dense | pooled | 560 | 0.339 [0.301, 0.379] | 0.491 [0.450, 0.532] | 0.546 [0.505, 0.587] | 0.616 [0.575, 0.655] |
| bge-large.hybrid | agent_nav | 33 | 0.364 [0.222, 0.534] | 0.545 [0.380, 0.702] | 0.727 [0.558, 0.849] | 0.848 [0.691, 0.933] |
| bge-large.hybrid | probe45 | 45 | 0.511 [0.370, 0.650] | 0.667 [0.521, 0.786] | 0.756 [0.613, 0.858] | 0.800 [0.662, 0.891] |
| bge-large.hybrid | synthetic_spill | 482 | 0.351 [0.309, 0.394] | 0.471 [0.427, 0.516] | 0.539 [0.495, 0.583] | 0.622 [0.578, 0.665] |
| bge-large.hybrid | pooled | 560 | 0.364 [0.325, 0.405] | 0.491 [0.450, 0.532] | 0.568 [0.527, 0.608] | 0.650 [0.610, 0.688] |
| bm25 | agent_nav | 33 | 0.182 [0.086, 0.344] | 0.515 [0.352, 0.675] | 0.697 [0.527, 0.826] | 0.848 [0.691, 0.933] |
| bm25 | probe45 | 45 | 0.333 [0.214, 0.479] | 0.533 [0.391, 0.671] | 0.578 [0.433, 0.710] | 0.689 [0.543, 0.805] |
| bm25 | synthetic_spill | 482 | 0.309 [0.270, 0.352] | 0.473 [0.429, 0.518] | 0.525 [0.480, 0.569] | 0.604 [0.559, 0.646] |
| bm25 | pooled | 560 | 0.304 [0.267, 0.343] | 0.480 [0.439, 0.522] | 0.539 [0.498, 0.580] | 0.625 [0.584, 0.664] |
| colgrep | probe45 | 45 | 0.378 [0.251, 0.524] | 0.689 [0.543, 0.805] | 0.756 [0.613, 0.858] | 0.822 [0.687, 0.907] |
| colgrep | pooled | 45 | 0.378 [0.251, 0.524] | 0.689 [0.543, 0.805] | 0.756 [0.613, 0.858] | 0.822 [0.687, 0.907] |
| eg2-gguf-bf16-256.dense | agent_nav | 33 | 0.182 [0.086, 0.344] | 0.545 [0.380, 0.702] | 0.606 [0.437, 0.753] | 0.667 [0.496, 0.802] |
| eg2-gguf-bf16-256.dense | probe45 | 45 | 0.378 [0.251, 0.524] | 0.667 [0.521, 0.786] | 0.778 [0.637, 0.875] | 0.867 [0.738, 0.937] |
| eg2-gguf-bf16-256.dense | synthetic_spill | 482 | 0.411 [0.368, 0.455] | 0.647 [0.604, 0.689] | 0.722 [0.680, 0.760] | 0.811 [0.774, 0.844] |
| eg2-gguf-bf16-256.dense | pooled | 560 | 0.395 [0.355, 0.436] | 0.643 [0.602, 0.681] | 0.720 [0.681, 0.755] | 0.807 [0.772, 0.838] |
| eg2-gguf-bf16-256.hybrid | agent_nav | 33 | 0.303 [0.174, 0.473] | 0.667 [0.496, 0.802] | 0.727 [0.558, 0.849] | 0.879 [0.727, 0.952] |
| eg2-gguf-bf16-256.hybrid | probe45 | 45 | 0.444 [0.309, 0.588] | 0.689 [0.543, 0.805] | 0.689 [0.543, 0.805] | 0.822 [0.687, 0.907] |
| eg2-gguf-bf16-256.hybrid | synthetic_spill | 482 | 0.446 [0.402, 0.491] | 0.633 [0.589, 0.675] | 0.728 [0.687, 0.766] | 0.811 [0.774, 0.844] |
| eg2-gguf-bf16-256.hybrid | pooled | 560 | 0.438 [0.397, 0.479] | 0.639 [0.599, 0.678] | 0.725 [0.687, 0.760] | 0.816 [0.782, 0.846] |
| eg2-gguf-bf16-768.dense | agent_nav | 33 | 0.273 [0.151, 0.442] | 0.576 [0.408, 0.728] | 0.636 [0.466, 0.778] | 0.727 [0.558, 0.849] |
| eg2-gguf-bf16-768.dense | probe45 | 45 | 0.400 [0.270, 0.545] | 0.644 [0.498, 0.768] | 0.756 [0.613, 0.858] | 0.844 [0.712, 0.923] |
| eg2-gguf-bf16-768.dense | synthetic_spill | 482 | 0.459 [0.415, 0.503] | 0.674 [0.631, 0.715] | 0.741 [0.700, 0.778] | 0.826 [0.789, 0.857] |
| eg2-gguf-bf16-768.dense | pooled | 560 | 0.443 [0.402, 0.484] | 0.666 [0.626, 0.704] | 0.736 [0.698, 0.771] | 0.821 [0.788, 0.851] |
| eg2-gguf-bf16-768.hybrid | agent_nav | 33 | 0.273 [0.151, 0.442] | 0.545 [0.380, 0.702] | 0.667 [0.496, 0.802] | 0.818 [0.656, 0.914] |
| eg2-gguf-bf16-768.hybrid | probe45 | 45 | 0.422 [0.290, 0.567] | 0.667 [0.521, 0.786] | 0.711 [0.566, 0.823] | 0.778 [0.637, 0.875] |
| eg2-gguf-bf16-768.hybrid | synthetic_spill | 482 | 0.473 [0.429, 0.518] | 0.645 [0.602, 0.687] | 0.718 [0.676, 0.756] | 0.811 [0.774, 0.844] |
| eg2-gguf-bf16-768.hybrid | pooled | 560 | 0.457 [0.416, 0.499] | 0.641 [0.601, 0.680] | 0.714 [0.676, 0.750] | 0.809 [0.774, 0.839] |
| granite-97m-r2.dense | agent_nav | 33 | 0.273 [0.151, 0.442] | 0.545 [0.380, 0.702] | 0.697 [0.527, 0.826] | 0.818 [0.656, 0.914] |
| granite-97m-r2.dense | probe45 | 45 | 0.533 [0.391, 0.671] | 0.733 [0.590, 0.840] | 0.756 [0.613, 0.858] | 0.778 [0.637, 0.875] |
| granite-97m-r2.dense | synthetic_spill | 482 | 0.282 [0.244, 0.324] | 0.502 [0.458, 0.547] | 0.585 [0.541, 0.628] | 0.680 [0.638, 0.721] |
| granite-97m-r2.dense | pooled | 560 | 0.302 [0.265, 0.341] | 0.523 [0.482, 0.564] | 0.605 [0.564, 0.645] | 0.696 [0.657, 0.733] |
| granite-97m-r2.hybrid | agent_nav | 33 | 0.364 [0.222, 0.534] | 0.606 [0.437, 0.753] | 0.788 [0.622, 0.893] | 0.909 [0.764, 0.969] |
| granite-97m-r2.hybrid | probe45 | 45 | 0.556 [0.412, 0.691] | 0.711 [0.566, 0.823] | 0.756 [0.613, 0.858] | 0.800 [0.662, 0.891] |
| granite-97m-r2.hybrid | synthetic_spill | 482 | 0.398 [0.356, 0.443] | 0.571 [0.526, 0.614] | 0.639 [0.595, 0.681] | 0.724 [0.683, 0.762] |
| granite-97m-r2.hybrid | pooled | 560 | 0.409 [0.369, 0.450] | 0.584 [0.543, 0.624] | 0.657 [0.617, 0.695] | 0.741 [0.703, 0.776] |
| grep | agent_nav | 33 | 0.152 [0.067, 0.309] | 0.273 [0.151, 0.442] | 0.455 [0.298, 0.620] | 0.606 [0.437, 0.753] |
| grep | probe45 | 45 | 0.422 [0.290, 0.567] | 0.600 [0.455, 0.730] | 0.644 [0.498, 0.768] | 0.689 [0.543, 0.805] |
| grep | synthetic_spill | 482 | 0.365 [0.323, 0.409] | 0.537 [0.493, 0.581] | 0.618 [0.574, 0.661] | 0.697 [0.655, 0.736] |
| grep | pooled | 560 | 0.357 [0.319, 0.398] | 0.527 [0.485, 0.568] | 0.611 [0.570, 0.650] | 0.691 [0.652, 0.728] |
| potion-base-8m.dense | agent_nav | 33 | 0.303 [0.174, 0.473] | 0.455 [0.298, 0.620] | 0.545 [0.380, 0.702] | 0.606 [0.437, 0.753] |
| potion-base-8m.dense | probe45 | 45 | 0.267 [0.160, 0.410] | 0.444 [0.309, 0.588] | 0.533 [0.391, 0.671] | 0.711 [0.566, 0.823] |
| potion-base-8m.dense | synthetic_spill | 482 | 0.170 [0.139, 0.206] | 0.353 [0.311, 0.396] | 0.434 [0.390, 0.478] | 0.515 [0.470, 0.559] |
| potion-base-8m.dense | pooled | 560 | 0.186 [0.156, 0.220] | 0.366 [0.327, 0.407] | 0.448 [0.408, 0.490] | 0.536 [0.494, 0.577] |
| potion-base-8m.hybrid | agent_nav | 33 | 0.273 [0.151, 0.442] | 0.485 [0.325, 0.648] | 0.636 [0.466, 0.778] | 0.758 [0.590, 0.872] |
| potion-base-8m.hybrid | probe45 | 45 | 0.467 [0.329, 0.609] | 0.600 [0.455, 0.730] | 0.600 [0.455, 0.730] | 0.667 [0.521, 0.786] |
| potion-base-8m.hybrid | synthetic_spill | 482 | 0.274 [0.236, 0.315] | 0.432 [0.388, 0.476] | 0.523 [0.478, 0.567] | 0.602 [0.557, 0.644] |
| potion-base-8m.hybrid | pooled | 560 | 0.289 [0.253, 0.328] | 0.448 [0.408, 0.490] | 0.536 [0.494, 0.577] | 0.616 [0.575, 0.655] |
| potion-base-8m.whybrid | agent_nav | 33 | 0.303 [0.174, 0.473] | 0.485 [0.325, 0.648] | 0.636 [0.466, 0.778] | 0.788 [0.622, 0.893] |
| potion-base-8m.whybrid | probe45 | 45 | 0.444 [0.309, 0.588] | 0.600 [0.455, 0.730] | 0.600 [0.455, 0.730] | 0.733 [0.590, 0.840] |
| potion-base-8m.whybrid | synthetic_spill | 482 | 0.241 [0.205, 0.281] | 0.405 [0.362, 0.449] | 0.496 [0.451, 0.540] | 0.585 [0.541, 0.628] |
| potion-base-8m.whybrid | pooled | 560 | 0.261 [0.226, 0.299] | 0.425 [0.385, 0.466] | 0.512 [0.471, 0.554] | 0.609 [0.568, 0.648] |
| potion-code-16m-v2.dense | agent_nav | 33 | 0.212 [0.107, 0.378] | 0.545 [0.380, 0.702] | 0.636 [0.466, 0.778] | 0.727 [0.558, 0.849] |
| potion-code-16m-v2.dense | probe45 | 45 | 0.400 [0.270, 0.545] | 0.600 [0.455, 0.730] | 0.667 [0.521, 0.786] | 0.778 [0.637, 0.875] |
| potion-code-16m-v2.dense | synthetic_spill | 482 | 0.293 [0.254, 0.335] | 0.440 [0.396, 0.484] | 0.502 [0.458, 0.547] | 0.606 [0.562, 0.648] |
| potion-code-16m-v2.dense | pooled | 560 | 0.296 [0.260, 0.336] | 0.459 [0.418, 0.500] | 0.523 [0.482, 0.564] | 0.627 [0.586, 0.666] |
| potion-code-16m-v2.hybrid | agent_nav | 33 | 0.182 [0.086, 0.344] | 0.485 [0.325, 0.648] | 0.697 [0.527, 0.826] | 0.848 [0.691, 0.933] |
| potion-code-16m-v2.hybrid | probe45 | 45 | 0.444 [0.309, 0.588] | 0.644 [0.498, 0.768] | 0.689 [0.543, 0.805] | 0.756 [0.613, 0.858] |
| potion-code-16m-v2.hybrid | synthetic_spill | 482 | 0.313 [0.273, 0.356] | 0.477 [0.433, 0.522] | 0.560 [0.516, 0.604] | 0.668 [0.625, 0.709] |
| potion-code-16m-v2.hybrid | pooled | 560 | 0.316 [0.279, 0.356] | 0.491 [0.450, 0.532] | 0.579 [0.537, 0.619] | 0.686 [0.646, 0.723] |
| potion-code-16m-v2.whybrid | agent_nav | 33 | 0.273 [0.151, 0.442] | 0.515 [0.352, 0.675] | 0.667 [0.496, 0.802] | 0.818 [0.656, 0.914] |
| potion-code-16m-v2.whybrid | probe45 | 45 | 0.467 [0.329, 0.609] | 0.689 [0.543, 0.805] | 0.733 [0.590, 0.840] | 0.756 [0.613, 0.858] |
| potion-code-16m-v2.whybrid | synthetic_spill | 482 | 0.295 [0.256, 0.337] | 0.456 [0.412, 0.501] | 0.552 [0.507, 0.596] | 0.645 [0.602, 0.687] |
| potion-code-16m-v2.whybrid | pooled | 560 | 0.307 [0.270, 0.347] | 0.479 [0.437, 0.520] | 0.573 [0.532, 0.614] | 0.664 [0.624, 0.702] |

## synthetic_spill recall split by writer - pooled, flagged included

| arm | writer | n | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| bge-large.dense | writer=A | 234 | 0.389 | 0.530 | 0.590 | 0.645 |
| bge-large.dense | writer=B | 248 | 0.258 | 0.403 | 0.452 | 0.540 |
| bge-large.dense | pooled | 482 | 0.322 | 0.465 | 0.519 | 0.591 |
| bge-large.hybrid | writer=A | 234 | 0.432 | 0.547 | 0.611 | 0.697 |
| bge-large.hybrid | writer=B | 248 | 0.274 | 0.399 | 0.472 | 0.552 |
| bge-large.hybrid | pooled | 482 | 0.351 | 0.471 | 0.539 | 0.622 |
| bm25 | writer=A | 234 | 0.397 | 0.590 | 0.632 | 0.697 |
| bm25 | writer=B | 248 | 0.226 | 0.363 | 0.423 | 0.516 |
| bm25 | pooled | 482 | 0.309 | 0.473 | 0.525 | 0.604 |
| eg2-gguf-bf16-256.dense | writer=A | 234 | 0.504 | 0.752 | 0.808 | 0.876 |
| eg2-gguf-bf16-256.dense | writer=B | 248 | 0.323 | 0.548 | 0.641 | 0.750 |
| eg2-gguf-bf16-256.dense | pooled | 482 | 0.411 | 0.647 | 0.722 | 0.811 |
| eg2-gguf-bf16-256.hybrid | writer=A | 234 | 0.534 | 0.739 | 0.833 | 0.885 |
| eg2-gguf-bf16-256.hybrid | writer=B | 248 | 0.363 | 0.532 | 0.629 | 0.742 |
| eg2-gguf-bf16-256.hybrid | pooled | 482 | 0.446 | 0.633 | 0.728 | 0.811 |
| eg2-gguf-bf16-768.dense | writer=A | 234 | 0.560 | 0.778 | 0.821 | 0.889 |
| eg2-gguf-bf16-768.dense | writer=B | 248 | 0.363 | 0.577 | 0.665 | 0.766 |
| eg2-gguf-bf16-768.dense | pooled | 482 | 0.459 | 0.674 | 0.741 | 0.826 |
| eg2-gguf-bf16-768.hybrid | writer=A | 234 | 0.568 | 0.731 | 0.803 | 0.902 |
| eg2-gguf-bf16-768.hybrid | writer=B | 248 | 0.383 | 0.565 | 0.637 | 0.726 |
| eg2-gguf-bf16-768.hybrid | pooled | 482 | 0.473 | 0.645 | 0.718 | 0.811 |
| granite-97m-r2.dense | writer=A | 234 | 0.346 | 0.611 | 0.688 | 0.752 |
| granite-97m-r2.dense | writer=B | 248 | 0.222 | 0.399 | 0.488 | 0.613 |
| granite-97m-r2.dense | pooled | 482 | 0.282 | 0.502 | 0.585 | 0.680 |
| granite-97m-r2.hybrid | writer=A | 234 | 0.479 | 0.658 | 0.739 | 0.812 |
| granite-97m-r2.hybrid | writer=B | 248 | 0.323 | 0.488 | 0.544 | 0.641 |
| granite-97m-r2.hybrid | pooled | 482 | 0.398 | 0.571 | 0.639 | 0.724 |
| grep | writer=A | 234 | 0.462 | 0.645 | 0.718 | 0.786 |
| grep | writer=B | 248 | 0.274 | 0.435 | 0.524 | 0.613 |
| grep | pooled | 482 | 0.365 | 0.537 | 0.618 | 0.697 |
| potion-base-8m.dense | writer=A | 234 | 0.197 | 0.397 | 0.470 | 0.551 |
| potion-base-8m.dense | writer=B | 248 | 0.145 | 0.310 | 0.399 | 0.480 |
| potion-base-8m.dense | pooled | 482 | 0.170 | 0.353 | 0.434 | 0.515 |
| potion-base-8m.hybrid | writer=A | 234 | 0.329 | 0.500 | 0.585 | 0.650 |
| potion-base-8m.hybrid | writer=B | 248 | 0.222 | 0.367 | 0.464 | 0.556 |
| potion-base-8m.hybrid | pooled | 482 | 0.274 | 0.432 | 0.523 | 0.602 |
| potion-base-8m.whybrid | writer=A | 234 | 0.286 | 0.466 | 0.547 | 0.632 |
| potion-base-8m.whybrid | writer=B | 248 | 0.198 | 0.347 | 0.448 | 0.540 |
| potion-base-8m.whybrid | pooled | 482 | 0.241 | 0.405 | 0.496 | 0.585 |
| potion-code-16m-v2.dense | writer=A | 234 | 0.350 | 0.504 | 0.564 | 0.675 |
| potion-code-16m-v2.dense | writer=B | 248 | 0.238 | 0.379 | 0.444 | 0.540 |
| potion-code-16m-v2.dense | pooled | 482 | 0.293 | 0.440 | 0.502 | 0.606 |
| potion-code-16m-v2.hybrid | writer=A | 234 | 0.376 | 0.577 | 0.628 | 0.731 |
| potion-code-16m-v2.hybrid | writer=B | 248 | 0.254 | 0.383 | 0.496 | 0.609 |
| potion-code-16m-v2.hybrid | pooled | 482 | 0.313 | 0.477 | 0.560 | 0.668 |
| potion-code-16m-v2.whybrid | writer=A | 234 | 0.342 | 0.521 | 0.624 | 0.718 |
| potion-code-16m-v2.whybrid | writer=B | 248 | 0.250 | 0.395 | 0.484 | 0.577 |
| potion-code-16m-v2.whybrid | pooled | 482 | 0.295 | 0.456 | 0.552 | 0.645 |

## kill rule (recall@5, hybrid vs L) - pooled, flagged included

| hybrid | kind | R@5 | L R@5 | diff | CI lo | CI hi | margin | CI>0 | pass |
|---|---|---|---|---|---|---|---|---|---|
| bge-large.hybrid | registered | 0.568 | 0.611 | -0.043 | -0.084 | -0.002 | False | False | False |
| granite-97m-r2.hybrid | registered | 0.657 | 0.611 | +0.046 | +0.007 | +0.087 | False | True | False |
| eg2-gguf-bf16-256.hybrid | extension (extension_only if it flips) | 0.725 | 0.611 | +0.114 | +0.075 | +0.154 | True | True | True |
| eg2-gguf-bf16-768.hybrid | extension (extension_only if it flips) | 0.714 | 0.611 | +0.104 | +0.064 | +0.143 | True | True | True |
| potion-base-8m.hybrid | extension (extension_only if it flips) | 0.536 | 0.611 | -0.075 | -0.120 | -0.030 | False | False | False |
| potion-code-16m-v2.hybrid | extension (extension_only if it flips) | 0.579 | 0.611 | -0.032 | -0.075 | +0.013 | False | False | False |

n = 560, M = 0.1, n_min = 120, bootstrap resamples/seed = 10000/20261020

# SECONDARY (non-deciding): without synthetic, flagged excluded (n=68)

**L - without synthetic, flagged excluded** = bm25 (grep 0.574, bm25 0.618, tie=False)

## recall@k per arm and stratum (Wilson 95%) - without synthetic, flagged excluded

| arm | stratum | n | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| bge-large.dense | agent_nav | 23 | 0.304 [0.156, 0.509] | 0.522 [0.330, 0.708] | 0.609 [0.408, 0.778] | 0.652 [0.449, 0.812] |
| bge-large.dense | probe45 | 45 | 0.556 [0.412, 0.691] | 0.667 [0.521, 0.786] | 0.733 [0.590, 0.840] | 0.778 [0.637, 0.875] |
| bge-large.dense | pooled | 68 | 0.471 [0.357, 0.588] | 0.618 [0.499, 0.724] | 0.691 [0.574, 0.788] | 0.735 [0.620, 0.826] |
| bge-large.hybrid | agent_nav | 23 | 0.348 [0.188, 0.551] | 0.565 [0.368, 0.744] | 0.652 [0.449, 0.812] | 0.783 [0.581, 0.903] |
| bge-large.hybrid | probe45 | 45 | 0.511 [0.370, 0.650] | 0.667 [0.521, 0.786] | 0.756 [0.613, 0.858] | 0.800 [0.662, 0.891] |
| bge-large.hybrid | pooled | 68 | 0.456 [0.343, 0.573] | 0.632 [0.514, 0.737] | 0.721 [0.604, 0.813] | 0.794 [0.684, 0.873] |
| bm25 | agent_nav | 23 | 0.174 [0.070, 0.371] | 0.565 [0.368, 0.744] | 0.696 [0.491, 0.844] | 0.826 [0.629, 0.930] |
| bm25 | probe45 | 45 | 0.333 [0.214, 0.479] | 0.533 [0.391, 0.671] | 0.578 [0.433, 0.710] | 0.689 [0.543, 0.805] |
| bm25 | pooled | 68 | 0.279 [0.187, 0.396] | 0.544 [0.427, 0.657] | 0.618 [0.499, 0.724] | 0.735 [0.620, 0.826] |
| colgrep | probe45 | 45 | 0.378 [0.251, 0.524] | 0.689 [0.543, 0.805] | 0.756 [0.613, 0.858] | 0.822 [0.687, 0.907] |
| colgrep | pooled | 45 | 0.378 [0.251, 0.524] | 0.689 [0.543, 0.805] | 0.756 [0.613, 0.858] | 0.822 [0.687, 0.907] |
| eg2-gguf-bf16-256.dense | agent_nav | 23 | 0.261 [0.125, 0.465] | 0.522 [0.330, 0.708] | 0.565 [0.368, 0.744] | 0.652 [0.449, 0.812] |
| eg2-gguf-bf16-256.dense | probe45 | 45 | 0.378 [0.251, 0.524] | 0.667 [0.521, 0.786] | 0.778 [0.637, 0.875] | 0.867 [0.738, 0.937] |
| eg2-gguf-bf16-256.dense | pooled | 68 | 0.338 [0.237, 0.457] | 0.618 [0.499, 0.724] | 0.706 [0.589, 0.801] | 0.794 [0.684, 0.873] |
| eg2-gguf-bf16-256.hybrid | agent_nav | 23 | 0.304 [0.156, 0.509] | 0.652 [0.449, 0.812] | 0.739 [0.535, 0.875] | 0.870 [0.679, 0.955] |
| eg2-gguf-bf16-256.hybrid | probe45 | 45 | 0.444 [0.309, 0.588] | 0.689 [0.543, 0.805] | 0.689 [0.543, 0.805] | 0.822 [0.687, 0.907] |
| eg2-gguf-bf16-256.hybrid | pooled | 68 | 0.397 [0.289, 0.516] | 0.676 [0.558, 0.776] | 0.706 [0.589, 0.801] | 0.838 [0.733, 0.907] |
| eg2-gguf-bf16-768.dense | agent_nav | 23 | 0.348 [0.188, 0.551] | 0.522 [0.330, 0.708] | 0.609 [0.408, 0.778] | 0.696 [0.491, 0.844] |
| eg2-gguf-bf16-768.dense | probe45 | 45 | 0.400 [0.270, 0.545] | 0.644 [0.498, 0.768] | 0.756 [0.613, 0.858] | 0.844 [0.712, 0.923] |
| eg2-gguf-bf16-768.dense | pooled | 68 | 0.382 [0.276, 0.501] | 0.603 [0.484, 0.711] | 0.706 [0.589, 0.801] | 0.794 [0.684, 0.873] |
| eg2-gguf-bf16-768.hybrid | agent_nav | 23 | 0.261 [0.125, 0.465] | 0.522 [0.330, 0.708] | 0.652 [0.449, 0.812] | 0.783 [0.581, 0.903] |
| eg2-gguf-bf16-768.hybrid | probe45 | 45 | 0.422 [0.290, 0.567] | 0.667 [0.521, 0.786] | 0.711 [0.566, 0.823] | 0.778 [0.637, 0.875] |
| eg2-gguf-bf16-768.hybrid | pooled | 68 | 0.368 [0.263, 0.486] | 0.618 [0.499, 0.724] | 0.691 [0.574, 0.788] | 0.779 [0.667, 0.862] |
| granite-97m-r2.dense | agent_nav | 23 | 0.348 [0.188, 0.551] | 0.522 [0.330, 0.708] | 0.652 [0.449, 0.812] | 0.739 [0.535, 0.875] |
| granite-97m-r2.dense | probe45 | 45 | 0.533 [0.391, 0.671] | 0.733 [0.590, 0.840] | 0.756 [0.613, 0.858] | 0.778 [0.637, 0.875] |
| granite-97m-r2.dense | pooled | 68 | 0.471 [0.357, 0.588] | 0.662 [0.543, 0.763] | 0.721 [0.604, 0.813] | 0.765 [0.651, 0.850] |
| granite-97m-r2.hybrid | agent_nav | 23 | 0.391 [0.222, 0.592] | 0.609 [0.408, 0.778] | 0.696 [0.491, 0.844] | 0.870 [0.679, 0.955] |
| granite-97m-r2.hybrid | probe45 | 45 | 0.556 [0.412, 0.691] | 0.711 [0.566, 0.823] | 0.756 [0.613, 0.858] | 0.800 [0.662, 0.891] |
| granite-97m-r2.hybrid | pooled | 68 | 0.500 [0.384, 0.616] | 0.676 [0.558, 0.776] | 0.735 [0.620, 0.826] | 0.824 [0.716, 0.896] |
| grep | agent_nav | 23 | 0.130 [0.045, 0.321] | 0.261 [0.125, 0.465] | 0.435 [0.256, 0.632] | 0.565 [0.368, 0.744] |
| grep | probe45 | 45 | 0.422 [0.290, 0.567] | 0.600 [0.455, 0.730] | 0.644 [0.498, 0.768] | 0.689 [0.543, 0.805] |
| grep | pooled | 68 | 0.324 [0.224, 0.442] | 0.485 [0.371, 0.602] | 0.574 [0.455, 0.684] | 0.647 [0.528, 0.750] |
| potion-base-8m.dense | agent_nav | 23 | 0.261 [0.125, 0.465] | 0.391 [0.222, 0.592] | 0.478 [0.292, 0.670] | 0.565 [0.368, 0.744] |
| potion-base-8m.dense | probe45 | 45 | 0.267 [0.160, 0.410] | 0.444 [0.309, 0.588] | 0.533 [0.391, 0.671] | 0.711 [0.566, 0.823] |
| potion-base-8m.dense | pooled | 68 | 0.265 [0.174, 0.380] | 0.426 [0.316, 0.545] | 0.515 [0.398, 0.629] | 0.662 [0.543, 0.763] |
| potion-base-8m.hybrid | agent_nav | 23 | 0.217 [0.097, 0.419] | 0.522 [0.330, 0.708] | 0.652 [0.449, 0.812] | 0.739 [0.535, 0.875] |
| potion-base-8m.hybrid | probe45 | 45 | 0.467 [0.329, 0.609] | 0.600 [0.455, 0.730] | 0.600 [0.455, 0.730] | 0.667 [0.521, 0.786] |
| potion-base-8m.hybrid | pooled | 68 | 0.382 [0.276, 0.501] | 0.574 [0.455, 0.684] | 0.618 [0.499, 0.724] | 0.691 [0.574, 0.788] |
| potion-base-8m.whybrid | agent_nav | 23 | 0.304 [0.156, 0.509] | 0.478 [0.292, 0.670] | 0.609 [0.408, 0.778] | 0.783 [0.581, 0.903] |
| potion-base-8m.whybrid | probe45 | 45 | 0.444 [0.309, 0.588] | 0.600 [0.455, 0.730] | 0.600 [0.455, 0.730] | 0.733 [0.590, 0.840] |
| potion-base-8m.whybrid | pooled | 68 | 0.397 [0.289, 0.516] | 0.559 [0.441, 0.671] | 0.603 [0.484, 0.711] | 0.750 [0.636, 0.838] |
| potion-code-16m-v2.dense | agent_nav | 23 | 0.217 [0.097, 0.419] | 0.435 [0.256, 0.632] | 0.522 [0.330, 0.708] | 0.652 [0.449, 0.812] |
| potion-code-16m-v2.dense | probe45 | 45 | 0.400 [0.270, 0.545] | 0.600 [0.455, 0.730] | 0.667 [0.521, 0.786] | 0.778 [0.637, 0.875] |
| potion-code-16m-v2.dense | pooled | 68 | 0.338 [0.237, 0.457] | 0.544 [0.427, 0.657] | 0.618 [0.499, 0.724] | 0.735 [0.620, 0.826] |
| potion-code-16m-v2.hybrid | agent_nav | 23 | 0.130 [0.045, 0.321] | 0.522 [0.330, 0.708] | 0.652 [0.449, 0.812] | 0.783 [0.581, 0.903] |
| potion-code-16m-v2.hybrid | probe45 | 45 | 0.444 [0.309, 0.588] | 0.644 [0.498, 0.768] | 0.689 [0.543, 0.805] | 0.756 [0.613, 0.858] |
| potion-code-16m-v2.hybrid | pooled | 68 | 0.338 [0.237, 0.457] | 0.603 [0.484, 0.711] | 0.676 [0.558, 0.776] | 0.765 [0.651, 0.850] |
| potion-code-16m-v2.whybrid | agent_nav | 23 | 0.217 [0.097, 0.419] | 0.478 [0.292, 0.670] | 0.565 [0.368, 0.744] | 0.739 [0.535, 0.875] |
| potion-code-16m-v2.whybrid | probe45 | 45 | 0.467 [0.329, 0.609] | 0.689 [0.543, 0.805] | 0.733 [0.590, 0.840] | 0.756 [0.613, 0.858] |
| potion-code-16m-v2.whybrid | pooled | 68 | 0.382 [0.276, 0.501] | 0.618 [0.499, 0.724] | 0.676 [0.558, 0.776] | 0.750 [0.636, 0.838] |

## kill rule (recall@5, hybrid vs L) - without synthetic, flagged excluded

| hybrid | kind | R@5 | L R@5 | diff | CI lo | CI hi | margin | CI>0 | pass |
|---|---|---|---|---|---|---|---|---|---|
| bge-large.hybrid | registered | 0.721 | 0.618 | +0.103 | +0.000 | +0.206 | True | False | False |
| granite-97m-r2.hybrid | registered | 0.735 | 0.618 | +0.118 | +0.015 | +0.221 | True | True | True |
| eg2-gguf-bf16-256.hybrid | extension (extension_only if it flips) | 0.706 | 0.618 | +0.088 | -0.015 | +0.191 | False | False | False |
| eg2-gguf-bf16-768.hybrid | extension (extension_only if it flips) | 0.691 | 0.618 | +0.074 | -0.029 | +0.191 | False | False | False |
| potion-base-8m.hybrid | extension (extension_only if it flips) | 0.618 | 0.618 | +0.000 | -0.088 | +0.088 | False | False | False |
| potion-code-16m-v2.hybrid | extension (extension_only if it flips) | 0.676 | 0.618 | +0.059 | -0.044 | +0.162 | False | False | False |

n = 68, M = 0.1, n_min = 120, bootstrap resamples/seed = 10000/20261020

# SECONDARY (non-deciding): without synthetic, flagged included (n=78)

**L - without synthetic, flagged included** = bm25 (grep 0.564, bm25 0.628, tie=False)

## recall@k per arm and stratum (Wilson 95%) - without synthetic, flagged included

| arm | stratum | n | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| bge-large.dense | agent_nav | 33 | 0.303 [0.174, 0.473] | 0.636 [0.466, 0.778] | 0.697 [0.527, 0.826] | 0.758 [0.590, 0.872] |
| bge-large.dense | probe45 | 45 | 0.556 [0.412, 0.691] | 0.667 [0.521, 0.786] | 0.733 [0.590, 0.840] | 0.778 [0.637, 0.875] |
| bge-large.dense | pooled | 78 | 0.449 [0.343, 0.559] | 0.654 [0.543, 0.750] | 0.718 [0.610, 0.806] | 0.769 [0.664, 0.849] |
| bge-large.hybrid | agent_nav | 33 | 0.364 [0.222, 0.534] | 0.545 [0.380, 0.702] | 0.727 [0.558, 0.849] | 0.848 [0.691, 0.933] |
| bge-large.hybrid | probe45 | 45 | 0.511 [0.370, 0.650] | 0.667 [0.521, 0.786] | 0.756 [0.613, 0.858] | 0.800 [0.662, 0.891] |
| bge-large.hybrid | pooled | 78 | 0.449 [0.343, 0.559] | 0.615 [0.504, 0.716] | 0.744 [0.637, 0.827] | 0.821 [0.721, 0.890] |
| bm25 | agent_nav | 33 | 0.182 [0.086, 0.344] | 0.515 [0.352, 0.675] | 0.697 [0.527, 0.826] | 0.848 [0.691, 0.933] |
| bm25 | probe45 | 45 | 0.333 [0.214, 0.479] | 0.533 [0.391, 0.671] | 0.578 [0.433, 0.710] | 0.689 [0.543, 0.805] |
| bm25 | pooled | 78 | 0.269 [0.183, 0.377] | 0.526 [0.416, 0.633] | 0.628 [0.517, 0.727] | 0.756 [0.651, 0.838] |
| colgrep | probe45 | 45 | 0.378 [0.251, 0.524] | 0.689 [0.543, 0.805] | 0.756 [0.613, 0.858] | 0.822 [0.687, 0.907] |
| colgrep | pooled | 45 | 0.378 [0.251, 0.524] | 0.689 [0.543, 0.805] | 0.756 [0.613, 0.858] | 0.822 [0.687, 0.907] |
| eg2-gguf-bf16-256.dense | agent_nav | 33 | 0.182 [0.086, 0.344] | 0.545 [0.380, 0.702] | 0.606 [0.437, 0.753] | 0.667 [0.496, 0.802] |
| eg2-gguf-bf16-256.dense | probe45 | 45 | 0.378 [0.251, 0.524] | 0.667 [0.521, 0.786] | 0.778 [0.637, 0.875] | 0.867 [0.738, 0.937] |
| eg2-gguf-bf16-256.dense | pooled | 78 | 0.295 [0.205, 0.404] | 0.615 [0.504, 0.716] | 0.705 [0.596, 0.795] | 0.782 [0.678, 0.859] |
| eg2-gguf-bf16-256.hybrid | agent_nav | 33 | 0.303 [0.174, 0.473] | 0.667 [0.496, 0.802] | 0.727 [0.558, 0.849] | 0.879 [0.727, 0.952] |
| eg2-gguf-bf16-256.hybrid | probe45 | 45 | 0.444 [0.309, 0.588] | 0.689 [0.543, 0.805] | 0.689 [0.543, 0.805] | 0.822 [0.687, 0.907] |
| eg2-gguf-bf16-256.hybrid | pooled | 78 | 0.385 [0.284, 0.496] | 0.679 [0.570, 0.773] | 0.705 [0.596, 0.795] | 0.846 [0.750, 0.910] |
| eg2-gguf-bf16-768.dense | agent_nav | 33 | 0.273 [0.151, 0.442] | 0.576 [0.408, 0.728] | 0.636 [0.466, 0.778] | 0.727 [0.558, 0.849] |
| eg2-gguf-bf16-768.dense | probe45 | 45 | 0.400 [0.270, 0.545] | 0.644 [0.498, 0.768] | 0.756 [0.613, 0.858] | 0.844 [0.712, 0.923] |
| eg2-gguf-bf16-768.dense | pooled | 78 | 0.346 [0.250, 0.457] | 0.615 [0.504, 0.716] | 0.705 [0.596, 0.795] | 0.795 [0.692, 0.870] |
| eg2-gguf-bf16-768.hybrid | agent_nav | 33 | 0.273 [0.151, 0.442] | 0.545 [0.380, 0.702] | 0.667 [0.496, 0.802] | 0.818 [0.656, 0.914] |
| eg2-gguf-bf16-768.hybrid | probe45 | 45 | 0.422 [0.290, 0.567] | 0.667 [0.521, 0.786] | 0.711 [0.566, 0.823] | 0.778 [0.637, 0.875] |
| eg2-gguf-bf16-768.hybrid | pooled | 78 | 0.359 [0.261, 0.470] | 0.615 [0.504, 0.716] | 0.692 [0.583, 0.784] | 0.795 [0.692, 0.870] |
| granite-97m-r2.dense | agent_nav | 33 | 0.273 [0.151, 0.442] | 0.545 [0.380, 0.702] | 0.697 [0.527, 0.826] | 0.818 [0.656, 0.914] |
| granite-97m-r2.dense | probe45 | 45 | 0.533 [0.391, 0.671] | 0.733 [0.590, 0.840] | 0.756 [0.613, 0.858] | 0.778 [0.637, 0.875] |
| granite-97m-r2.dense | pooled | 78 | 0.423 [0.320, 0.534] | 0.654 [0.543, 0.750] | 0.731 [0.623, 0.817] | 0.795 [0.692, 0.870] |
| granite-97m-r2.hybrid | agent_nav | 33 | 0.364 [0.222, 0.534] | 0.606 [0.437, 0.753] | 0.788 [0.622, 0.893] | 0.909 [0.764, 0.969] |
| granite-97m-r2.hybrid | probe45 | 45 | 0.556 [0.412, 0.691] | 0.711 [0.566, 0.823] | 0.756 [0.613, 0.858] | 0.800 [0.662, 0.891] |
| granite-97m-r2.hybrid | pooled | 78 | 0.474 [0.367, 0.584] | 0.667 [0.556, 0.761] | 0.769 [0.664, 0.849] | 0.846 [0.750, 0.910] |
| grep | agent_nav | 33 | 0.152 [0.067, 0.309] | 0.273 [0.151, 0.442] | 0.455 [0.298, 0.620] | 0.606 [0.437, 0.753] |
| grep | probe45 | 45 | 0.422 [0.290, 0.567] | 0.600 [0.455, 0.730] | 0.644 [0.498, 0.768] | 0.689 [0.543, 0.805] |
| grep | pooled | 78 | 0.308 [0.216, 0.417] | 0.462 [0.355, 0.571] | 0.564 [0.454, 0.669] | 0.654 [0.543, 0.750] |
| potion-base-8m.dense | agent_nav | 33 | 0.303 [0.174, 0.473] | 0.455 [0.298, 0.620] | 0.545 [0.380, 0.702] | 0.606 [0.437, 0.753] |
| potion-base-8m.dense | probe45 | 45 | 0.267 [0.160, 0.410] | 0.444 [0.309, 0.588] | 0.533 [0.391, 0.671] | 0.711 [0.566, 0.823] |
| potion-base-8m.dense | pooled | 78 | 0.282 [0.194, 0.390] | 0.449 [0.343, 0.559] | 0.538 [0.429, 0.645] | 0.667 [0.556, 0.761] |
| potion-base-8m.hybrid | agent_nav | 33 | 0.273 [0.151, 0.442] | 0.485 [0.325, 0.648] | 0.636 [0.466, 0.778] | 0.758 [0.590, 0.872] |
| potion-base-8m.hybrid | probe45 | 45 | 0.467 [0.329, 0.609] | 0.600 [0.455, 0.730] | 0.600 [0.455, 0.730] | 0.667 [0.521, 0.786] |
| potion-base-8m.hybrid | pooled | 78 | 0.385 [0.284, 0.496] | 0.551 [0.441, 0.657] | 0.615 [0.504, 0.716] | 0.705 [0.596, 0.795] |
| potion-base-8m.whybrid | agent_nav | 33 | 0.303 [0.174, 0.473] | 0.485 [0.325, 0.648] | 0.636 [0.466, 0.778] | 0.788 [0.622, 0.893] |
| potion-base-8m.whybrid | probe45 | 45 | 0.444 [0.309, 0.588] | 0.600 [0.455, 0.730] | 0.600 [0.455, 0.730] | 0.733 [0.590, 0.840] |
| potion-base-8m.whybrid | pooled | 78 | 0.385 [0.284, 0.496] | 0.551 [0.441, 0.657] | 0.615 [0.504, 0.716] | 0.756 [0.651, 0.838] |
| potion-code-16m-v2.dense | agent_nav | 33 | 0.212 [0.107, 0.378] | 0.545 [0.380, 0.702] | 0.636 [0.466, 0.778] | 0.727 [0.558, 0.849] |
| potion-code-16m-v2.dense | probe45 | 45 | 0.400 [0.270, 0.545] | 0.600 [0.455, 0.730] | 0.667 [0.521, 0.786] | 0.778 [0.637, 0.875] |
| potion-code-16m-v2.dense | pooled | 78 | 0.321 [0.227, 0.430] | 0.577 [0.466, 0.680] | 0.654 [0.543, 0.750] | 0.756 [0.651, 0.838] |
| potion-code-16m-v2.hybrid | agent_nav | 33 | 0.182 [0.086, 0.344] | 0.485 [0.325, 0.648] | 0.697 [0.527, 0.826] | 0.848 [0.691, 0.933] |
| potion-code-16m-v2.hybrid | probe45 | 45 | 0.444 [0.309, 0.588] | 0.644 [0.498, 0.768] | 0.689 [0.543, 0.805] | 0.756 [0.613, 0.858] |
| potion-code-16m-v2.hybrid | pooled | 78 | 0.333 [0.239, 0.444] | 0.577 [0.466, 0.680] | 0.692 [0.583, 0.784] | 0.795 [0.692, 0.870] |
| potion-code-16m-v2.whybrid | agent_nav | 33 | 0.273 [0.151, 0.442] | 0.515 [0.352, 0.675] | 0.667 [0.496, 0.802] | 0.818 [0.656, 0.914] |
| potion-code-16m-v2.whybrid | probe45 | 45 | 0.467 [0.329, 0.609] | 0.689 [0.543, 0.805] | 0.733 [0.590, 0.840] | 0.756 [0.613, 0.858] |
| potion-code-16m-v2.whybrid | pooled | 78 | 0.385 [0.284, 0.496] | 0.615 [0.504, 0.716] | 0.705 [0.596, 0.795] | 0.782 [0.678, 0.859] |

## kill rule (recall@5, hybrid vs L) - without synthetic, flagged included

| hybrid | kind | R@5 | L R@5 | diff | CI lo | CI hi | margin | CI>0 | pass |
|---|---|---|---|---|---|---|---|---|---|
| bge-large.hybrid | registered | 0.744 | 0.628 | +0.115 | +0.013 | +0.218 | True | True | True |
| granite-97m-r2.hybrid | registered | 0.769 | 0.628 | +0.141 | +0.038 | +0.244 | True | True | True |
| eg2-gguf-bf16-256.hybrid | extension (extension_only if it flips) | 0.705 | 0.628 | +0.077 | -0.013 | +0.167 | False | False | False |
| eg2-gguf-bf16-768.hybrid | extension (extension_only if it flips) | 0.692 | 0.628 | +0.064 | -0.026 | +0.154 | False | False | False |
| potion-base-8m.hybrid | extension (extension_only if it flips) | 0.615 | 0.628 | -0.013 | -0.103 | +0.064 | False | False | False |
| potion-code-16m-v2.hybrid | extension (extension_only if it flips) | 0.692 | 0.628 | +0.064 | -0.038 | +0.167 | False | False | False |

n = 78, M = 0.1, n_min = 120, bootstrap resamples/seed = 10000/20261020

## trivial-head stratum (outside the decision)

| arm | n | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|
| bge-large.dense | 22 | 0.500 | 0.909 | 0.955 | 1.000 |
| bge-large.hybrid | 22 | 0.591 | 0.818 | 0.909 | 1.000 |
| bm25 | 22 | 0.364 | 0.591 | 0.727 | 0.773 |
| eg2-gguf-bf16-256.dense | 22 | 0.727 | 0.955 | 0.955 | 1.000 |
| eg2-gguf-bf16-256.hybrid | 22 | 0.591 | 0.909 | 0.909 | 1.000 |
| eg2-gguf-bf16-768.dense | 22 | 0.773 | 1.000 | 1.000 | 1.000 |
| eg2-gguf-bf16-768.hybrid | 22 | 0.682 | 0.909 | 0.909 | 1.000 |
| granite-97m-r2.dense | 22 | 0.591 | 0.864 | 0.909 | 1.000 |
| granite-97m-r2.hybrid | 22 | 0.591 | 0.818 | 1.000 | 1.000 |
| grep | 22 | 0.682 | 0.818 | 0.955 | 1.000 |
| potion-base-8m.dense | 22 | 0.500 | 0.727 | 0.818 | 0.955 |
| potion-base-8m.hybrid | 22 | 0.545 | 0.773 | 0.818 | 0.955 |
| potion-base-8m.whybrid | 22 | 0.455 | 0.773 | 0.864 | 0.955 |
| potion-code-16m-v2.dense | 22 | 0.364 | 0.773 | 0.909 | 0.955 |
| potion-code-16m-v2.hybrid | 22 | 0.545 | 0.773 | 0.909 | 0.955 |
| potion-code-16m-v2.whybrid | 22 | 0.455 | 0.864 | 0.909 | 0.955 |

## failures (counted as misses) and contended cost

| arm | n | failed | missing | p50 ms | p95 ms |
|---|---|---|---|---|---|
| bge-large.dense | 346 | 47 | 0 | 127.555 | 179.837 |
| bge-large.hybrid | 346 | 47 | 0 | 133.136 | 184.916 |
| bm25 | 346 | 0 | 0 | 0.659 | 1.798 |
| colgrep | 45 | 0 | 0 | 320.290 | 399.860 |
| eg2-gguf-bf16-256.dense | 346 | 0 | 0 | 13.853 | 25.567 |
| eg2-gguf-bf16-256.hybrid | 346 | 0 | 0 | 16.214 | 26.945 |
| eg2-gguf-bf16-768.dense | 346 | 0 | 0 | 113.497 | 145.649 |
| eg2-gguf-bf16-768.hybrid | 346 | 0 | 0 | 119.865 | 151.210 |
| granite-97m-r2.dense | 346 | 0 | 0 | 3.649 | 102.435 |
| granite-97m-r2.hybrid | 346 | 0 | 0 | 5.127 | 111.235 |
| grep | 346 | 0 | 0 | 4.117 | 8.271 |
| potion-base-8m.dense | 346 | 0 | 0 | 0.276 | 0.487 |
| potion-base-8m.hybrid | 346 | 0 | 0 | 1.374 | 3.169 |
| potion-base-8m.whybrid | 346 | 0 | 0 | 1.039 | 2.736 |
| potion-code-16m-v2.dense | 346 | 0 | 0 | 0.236 | 0.494 |
| potion-code-16m-v2.hybrid | 346 | 0 | 0 | 1.063 | 3.027 |
| potion-code-16m-v2.whybrid | 346 | 0 | 0 | 0.897 | 2.794 |

semble: not_run (tool not available in the isolated venv)
