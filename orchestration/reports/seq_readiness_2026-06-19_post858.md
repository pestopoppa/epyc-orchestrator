# AutoPilot Sequential Verdict Readiness Report

- Status: blocked
- Recommendation: continue shadow collection; cutover blockers remain
- Vector trials: trusted=45, raw=52, untrusted=7
- Sequential shadow rows: 4 (disagreements=4, flip_rate=100.0%)
- Thresholds: trusted_vectors>=120, seq_shadow>=30, flip_rate>=30%, shared_qids>=35

## Candidate Clusters

- fp=sha1:b31e5a675b02ebd8 trials=[856,857] n=2 median_q=0.510 latest_q=0.660 median_questions=60 learning_exclusions={'none': 2} seq_rows=0 latest_seq=none
- fp=sha1:322c76ae095b3c4f trials=[835,855] n=2 median_q=1.380 latest_q=0.840 median_questions=55 learning_exclusions={'none': 1, 'seq_accumulating': 1} seq_rows=1 latest_seq=accumulating
- fp=sha1:d3b587bf67689cda trials=[850] n=1 median_q=0.360 latest_q=0.360 median_questions=60 learning_exclusions={'none': 1} seq_rows=0 latest_seq=none
- fp=sha1:ab7c66164ffd5325 trials=[845,848] n=2 median_q=0.630 latest_q=0.720 median_questions=60 learning_exclusions={'none': 2} seq_rows=0 latest_seq=none
- fp=sha1:d9082afb27a4ed94 trials=[819,826,836,842,843,847] n=6 median_q=0.918 latest_q=0.252 median_questions=500 learning_exclusions={'none': 6} seq_rows=1 latest_seq=accumulating
- fp=sha1:9bc91f314e72c47d trials=[846] n=1 median_q=0.480 latest_q=0.480 median_questions=60 learning_exclusions={'none': 1} seq_rows=0 latest_seq=none
- fp=sha1:2b2127b2b11ff944 trials=[844] n=1 median_q=0.840 latest_q=0.840 median_questions=60 learning_exclusions={'none': 1} seq_rows=0 latest_seq=none
- fp=sha1:b62aeb57e9097af4 trials=[815,816,817,...,834,838] n=11 median_q=1.860 latest_q=1.260 median_questions=60 learning_exclusions={'mad_noise': 1, 'none': 5, 'reproduction_confirmed': 3, 'seq_accumulating': 2} seq_rows=2 latest_seq=accumulating
- fp=sha1:c4edd26b3db58732 trials=[837] n=1 median_q=1.080 latest_q=1.080 median_questions=60 learning_exclusions={'none': 1} seq_rows=0 latest_seq=none
- fp=sha1:63c7a0a0fa36943b trials=[824] n=1 median_q=1.860 latest_q=1.860 median_questions=60 learning_exclusions={'mad_noise': 1} seq_rows=0 latest_seq=none
- fp=sha1:13599b7287e18a4c trials=[822] n=1 median_q=1.920 latest_q=1.920 median_questions=60 learning_exclusions={'reproduction_confirmed': 1} seq_rows=0 latest_seq=none
- fp=sha1:3052bc8a4b602afe trials=[818] n=1 median_q=1.964 latest_q=1.964 median_questions=55 learning_exclusions={'reproduction_confirmed': 1} seq_rows=0 latest_seq=none
- fp=sha1:fcadbcc2165b811b trials=[799,800,801,...,812,813] n=13 median_q=2.018 latest_q=1.980 median_questions=50 learning_exclusions={'mad_noise': 3, 'none': 3, 'reproduction_confirmed': 7} seq_rows=0 latest_seq=none
- fp=sha1:c3fa97e069022c64 trials=[805,810] n=2 median_q=1.980 latest_q=2.040 median_questions=50 learning_exclusions={'mad_noise': 1, 'none': 1} seq_rows=0 latest_seq=none

## Blockers

- trusted vector history too small: 45 < 120
- sequential shadow history too small: 4 < 30

## Pairwise Replay

- sha1:fcadbcc2165b811b -> sha1:b62aeb57e9097af4: shared=55 discordant=6 flip_rate=10.9% delta_b_minus_a=-0.036 p=0.6875
- sha1:fcadbcc2165b811b -> sha1:b31e5a675b02ebd8: shared=45 discordant=26 flip_rate=57.8% delta_b_minus_a=-0.489 p=1.049e-05
- sha1:fcadbcc2165b811b -> sha1:ab7c66164ffd5325: shared=43 discordant=23 flip_rate=53.5% delta_b_minus_a=-0.442 p=6.604e-05
- sha1:fcadbcc2165b811b -> sha1:c3fa97e069022c64: shared=46 discordant=0 flip_rate=0.0% delta_b_minus_a=+0.000 p=1
- sha1:fcadbcc2165b811b -> sha1:d3b587bf67689cda: shared=50 discordant=31 flip_rate=62.0% delta_b_minus_a=-0.540 p=4.6e-07
- sha1:fcadbcc2165b811b -> sha1:9bc91f314e72c47d: shared=50 discordant=29 flip_rate=58.0% delta_b_minus_a=-0.500 p=1.62e-06
- sha1:fcadbcc2165b811b -> sha1:2b2127b2b11ff944: shared=50 discordant=25 flip_rate=50.0% delta_b_minus_a=-0.380 p=0.0001565
- sha1:fcadbcc2165b811b -> sha1:c4edd26b3db58732: shared=50 discordant=15 flip_rate=30.0% delta_b_minus_a=-0.300 p=6.104e-05
- sha1:fcadbcc2165b811b -> sha1:63c7a0a0fa36943b: shared=50 discordant=8 flip_rate=16.0% delta_b_minus_a=-0.040 p=0.7266
- sha1:fcadbcc2165b811b -> sha1:13599b7287e18a4c: shared=50 discordant=11 flip_rate=22.0% delta_b_minus_a=-0.020 p=1
- sha1:fcadbcc2165b811b -> sha1:3052bc8a4b602afe: shared=55 discordant=8 flip_rate=14.5% delta_b_minus_a=-0.036 p=0.7266
- sha1:b62aeb57e9097af4 -> sha1:b31e5a675b02ebd8: shared=45 discordant=25 flip_rate=55.6% delta_b_minus_a=-0.467 p=1.943e-05
- sha1:b62aeb57e9097af4 -> sha1:ab7c66164ffd5325: shared=43 discordant=18 flip_rate=41.9% delta_b_minus_a=-0.419 p=7.63e-06
- sha1:b62aeb57e9097af4 -> sha1:c3fa97e069022c64: shared=46 discordant=4 flip_rate=8.7% delta_b_minus_a=+0.000 p=1
- sha1:b62aeb57e9097af4 -> sha1:d3b587bf67689cda: shared=50 discordant=25 flip_rate=50.0% delta_b_minus_a=-0.500 p=6e-08
- sha1:b62aeb57e9097af4 -> sha1:9bc91f314e72c47d: shared=50 discordant=25 flip_rate=50.0% delta_b_minus_a=-0.460 p=1.55e-06
- sha1:b62aeb57e9097af4 -> sha1:2b2127b2b11ff944: shared=50 discordant=21 flip_rate=42.0% delta_b_minus_a=-0.340 p=0.0002213
- sha1:b62aeb57e9097af4 -> sha1:c4edd26b3db58732: shared=50 discordant=15 flip_rate=30.0% delta_b_minus_a=-0.260 p=0.0009766
- sha1:b62aeb57e9097af4 -> sha1:63c7a0a0fa36943b: shared=50 discordant=2 flip_rate=4.0% delta_b_minus_a=+0.000 p=1
- sha1:b62aeb57e9097af4 -> sha1:13599b7287e18a4c: shared=50 discordant=5 flip_rate=10.0% delta_b_minus_a=+0.020 p=1
- sha1:b62aeb57e9097af4 -> sha1:3052bc8a4b602afe: shared=55 discordant=2 flip_rate=3.6% delta_b_minus_a=+0.000 p=1
- sha1:b31e5a675b02ebd8 -> sha1:ab7c66164ffd5325: shared=40 discordant=6 flip_rate=15.0% delta_b_minus_a=+0.050 p=0.6875
- sha1:b31e5a675b02ebd8 -> sha1:c3fa97e069022c64: shared=42 discordant=26 flip_rate=61.9% delta_b_minus_a=+0.524 p=1.049e-05
- sha1:b31e5a675b02ebd8 -> sha1:d3b587bf67689cda: shared=45 discordant=6 flip_rate=13.3% delta_b_minus_a=+0.000 p=1
- sha1:b31e5a675b02ebd8 -> sha1:9bc91f314e72c47d: shared=45 discordant=6 flip_rate=13.3% delta_b_minus_a=+0.044 p=0.6875
- sha1:b31e5a675b02ebd8 -> sha1:2b2127b2b11ff944: shared=45 discordant=16 flip_rate=35.6% delta_b_minus_a=+0.133 p=0.2101
- sha1:b31e5a675b02ebd8 -> sha1:c4edd26b3db58732: shared=45 discordant=15 flip_rate=33.3% delta_b_minus_a=+0.200 p=0.03516
- sha1:b31e5a675b02ebd8 -> sha1:63c7a0a0fa36943b: shared=45 discordant=26 flip_rate=57.8% delta_b_minus_a=+0.489 p=1.049e-05
- sha1:b31e5a675b02ebd8 -> sha1:13599b7287e18a4c: shared=45 discordant=29 flip_rate=64.4% delta_b_minus_a=+0.511 p=1.524e-05
- sha1:b31e5a675b02ebd8 -> sha1:3052bc8a4b602afe: shared=45 discordant=26 flip_rate=57.8% delta_b_minus_a=+0.489 p=1.049e-05
- sha1:ab7c66164ffd5325 -> sha1:c3fa97e069022c64: shared=39 discordant=21 flip_rate=53.8% delta_b_minus_a=+0.436 p=0.0002213
- sha1:ab7c66164ffd5325 -> sha1:d3b587bf67689cda: shared=43 discordant=3 flip_rate=7.0% delta_b_minus_a=-0.070 p=0.25
- sha1:ab7c66164ffd5325 -> sha1:9bc91f314e72c47d: shared=44 discordant=4 flip_rate=9.1% delta_b_minus_a=+0.000 p=1
- sha1:ab7c66164ffd5325 -> sha1:2b2127b2b11ff944: shared=43 discordant=11 flip_rate=25.6% delta_b_minus_a=+0.116 p=0.2266
- sha1:ab7c66164ffd5325 -> sha1:c4edd26b3db58732: shared=43 discordant=15 flip_rate=34.9% delta_b_minus_a=+0.163 p=0.1185
- sha1:ab7c66164ffd5325 -> sha1:63c7a0a0fa36943b: shared=43 discordant=19 flip_rate=44.2% delta_b_minus_a=+0.442 p=3.81e-06
- sha1:ab7c66164ffd5325 -> sha1:13599b7287e18a4c: shared=43 discordant=22 flip_rate=51.2% delta_b_minus_a=+0.465 p=1.097e-05
- sha1:ab7c66164ffd5325 -> sha1:3052bc8a4b602afe: shared=43 discordant=19 flip_rate=44.2% delta_b_minus_a=+0.442 p=3.81e-06
- sha1:c3fa97e069022c64 -> sha1:d3b587bf67689cda: shared=46 discordant=29 flip_rate=63.0% delta_b_minus_a=-0.543 p=1.62e-06
- sha1:c3fa97e069022c64 -> sha1:9bc91f314e72c47d: shared=46 discordant=27 flip_rate=58.7% delta_b_minus_a=-0.500 p=5.65e-06
- sha1:c3fa97e069022c64 -> sha1:2b2127b2b11ff944: shared=46 discordant=22 flip_rate=47.8% delta_b_minus_a=-0.391 p=0.0001211
- sha1:c3fa97e069022c64 -> sha1:c4edd26b3db58732: shared=46 discordant=13 flip_rate=28.3% delta_b_minus_a=-0.283 p=0.0002441
- sha1:c3fa97e069022c64 -> sha1:63c7a0a0fa36943b: shared=46 discordant=5 flip_rate=10.9% delta_b_minus_a=-0.022 p=1
- sha1:c3fa97e069022c64 -> sha1:13599b7287e18a4c: shared=46 discordant=7 flip_rate=15.2% delta_b_minus_a=-0.022 p=1
- sha1:c3fa97e069022c64 -> sha1:3052bc8a4b602afe: shared=46 discordant=5 flip_rate=10.9% delta_b_minus_a=-0.022 p=1
- sha1:d3b587bf67689cda -> sha1:9bc91f314e72c47d: shared=50 discordant=4 flip_rate=8.0% delta_b_minus_a=+0.040 p=0.625
- sha1:d3b587bf67689cda -> sha1:2b2127b2b11ff944: shared=50 discordant=16 flip_rate=32.0% delta_b_minus_a=+0.160 p=0.07681
- sha1:d3b587bf67689cda -> sha1:c4edd26b3db58732: shared=50 discordant=18 flip_rate=36.0% delta_b_minus_a=+0.240 p=0.007538
- sha1:d3b587bf67689cda -> sha1:63c7a0a0fa36943b: shared=50 discordant=25 flip_rate=50.0% delta_b_minus_a=+0.500 p=6e-08
- sha1:d3b587bf67689cda -> sha1:13599b7287e18a4c: shared=50 discordant=28 flip_rate=56.0% delta_b_minus_a=+0.520 p=2.2e-07
- sha1:d3b587bf67689cda -> sha1:3052bc8a4b602afe: shared=50 discordant=25 flip_rate=50.0% delta_b_minus_a=+0.500 p=6e-08
- sha1:9bc91f314e72c47d -> sha1:2b2127b2b11ff944: shared=50 discordant=16 flip_rate=32.0% delta_b_minus_a=+0.120 p=0.2101
- sha1:9bc91f314e72c47d -> sha1:c4edd26b3db58732: shared=50 discordant=16 flip_rate=32.0% delta_b_minus_a=+0.200 p=0.02127
- sha1:9bc91f314e72c47d -> sha1:63c7a0a0fa36943b: shared=50 discordant=25 flip_rate=50.0% delta_b_minus_a=+0.460 p=1.55e-06
- sha1:9bc91f314e72c47d -> sha1:13599b7287e18a4c: shared=50 discordant=28 flip_rate=56.0% delta_b_minus_a=+0.480 p=3.03e-06
- sha1:9bc91f314e72c47d -> sha1:3052bc8a4b602afe: shared=50 discordant=25 flip_rate=50.0% delta_b_minus_a=+0.460 p=1.55e-06
- sha1:2b2127b2b11ff944 -> sha1:c4edd26b3db58732: shared=50 discordant=20 flip_rate=40.0% delta_b_minus_a=+0.080 p=0.5034
- sha1:2b2127b2b11ff944 -> sha1:63c7a0a0fa36943b: shared=50 discordant=21 flip_rate=42.0% delta_b_minus_a=+0.340 p=0.0002213
- sha1:2b2127b2b11ff944 -> sha1:13599b7287e18a4c: shared=50 discordant=24 flip_rate=48.0% delta_b_minus_a=+0.360 p=0.0002772
- sha1:2b2127b2b11ff944 -> sha1:3052bc8a4b602afe: shared=50 discordant=21 flip_rate=42.0% delta_b_minus_a=+0.340 p=0.0002213
- sha1:c4edd26b3db58732 -> sha1:63c7a0a0fa36943b: shared=50 discordant=17 flip_rate=34.0% delta_b_minus_a=+0.260 p=0.00235
- sha1:c4edd26b3db58732 -> sha1:13599b7287e18a4c: shared=51 discordant=20 flip_rate=39.2% delta_b_minus_a=+0.275 p=0.002577
- sha1:c4edd26b3db58732 -> sha1:3052bc8a4b602afe: shared=50 discordant=17 flip_rate=34.0% delta_b_minus_a=+0.260 p=0.00235
- sha1:63c7a0a0fa36943b -> sha1:13599b7287e18a4c: shared=50 discordant=3 flip_rate=6.0% delta_b_minus_a=+0.020 p=1
- sha1:63c7a0a0fa36943b -> sha1:3052bc8a4b602afe: shared=50 discordant=0 flip_rate=0.0% delta_b_minus_a=+0.000 p=1
- sha1:13599b7287e18a4c -> sha1:3052bc8a4b602afe: shared=50 discordant=3 flip_rate=6.0% delta_b_minus_a=-0.020 p=1
