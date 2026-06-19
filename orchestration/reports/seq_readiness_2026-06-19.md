# AutoPilot Sequential Verdict Readiness Report

- Status: blocked
- Recommendation: continue shadow collection; cutover blockers remain
- Vector trials: trusted=36, raw=43, untrusted=7
- Sequential shadow rows: 4 (disagreements=4, flip_rate=100.0%)
- Thresholds: trusted_vectors>=120, seq_shadow>=30, flip_rate>=30%, shared_qids>=35

## Candidate Clusters

- fp=sha1:d9082afb27a4ed94 trials=[819,826,836,842,843] n=5 median_q=1.572 latest_q=0.264 median_questions=500 learning_exclusions={'none': 5} seq_rows=1 latest_seq=accumulating
- fp=sha1:b62aeb57e9097af4 trials=[815,816,817,...,834,838] n=11 median_q=1.860 latest_q=1.260 median_questions=60 learning_exclusions={'mad_noise': 1, 'none': 5, 'reproduction_confirmed': 3, 'seq_accumulating': 2} seq_rows=2 latest_seq=accumulating
- fp=sha1:c4edd26b3db58732 trials=[837] n=1 median_q=1.080 latest_q=1.080 median_questions=60 learning_exclusions={'none': 1} seq_rows=0 latest_seq=none
- fp=sha1:322c76ae095b3c4f trials=[835] n=1 median_q=1.920 latest_q=1.920 median_questions=50 learning_exclusions={'seq_accumulating': 1} seq_rows=1 latest_seq=accumulating
- fp=sha1:63c7a0a0fa36943b trials=[824] n=1 median_q=1.860 latest_q=1.860 median_questions=60 learning_exclusions={'mad_noise': 1} seq_rows=0 latest_seq=none
- fp=sha1:13599b7287e18a4c trials=[822] n=1 median_q=1.920 latest_q=1.920 median_questions=60 learning_exclusions={'reproduction_confirmed': 1} seq_rows=0 latest_seq=none
- fp=sha1:3052bc8a4b602afe trials=[818] n=1 median_q=1.964 latest_q=1.964 median_questions=55 learning_exclusions={'reproduction_confirmed': 1} seq_rows=0 latest_seq=none
- fp=sha1:fcadbcc2165b811b trials=[799,800,801,...,812,813] n=13 median_q=2.018 latest_q=1.980 median_questions=50 learning_exclusions={'mad_noise': 3, 'none': 3, 'reproduction_confirmed': 7} seq_rows=0 latest_seq=none
- fp=sha1:c3fa97e069022c64 trials=[805,810] n=2 median_q=1.980 latest_q=2.040 median_questions=50 learning_exclusions={'mad_noise': 1, 'none': 1} seq_rows=0 latest_seq=none

## Blockers

- trusted vector history too small: 36 < 120
- sequential shadow history too small: 4 < 30

## Pairwise Replay

- sha1:fcadbcc2165b811b -> sha1:b62aeb57e9097af4: shared=55 discordant=6 flip_rate=10.9% delta_b_minus_a=-0.036 p=0.6875
- sha1:fcadbcc2165b811b -> sha1:c3fa97e069022c64: shared=46 discordant=0 flip_rate=0.0% delta_b_minus_a=+0.000 p=1
- sha1:fcadbcc2165b811b -> sha1:c4edd26b3db58732: shared=50 discordant=15 flip_rate=30.0% delta_b_minus_a=-0.300 p=6.104e-05
- sha1:fcadbcc2165b811b -> sha1:322c76ae095b3c4f: shared=50 discordant=3 flip_rate=6.0% delta_b_minus_a=-0.020 p=1
- sha1:fcadbcc2165b811b -> sha1:63c7a0a0fa36943b: shared=50 discordant=8 flip_rate=16.0% delta_b_minus_a=-0.040 p=0.7266
- sha1:fcadbcc2165b811b -> sha1:13599b7287e18a4c: shared=50 discordant=11 flip_rate=22.0% delta_b_minus_a=-0.020 p=1
- sha1:fcadbcc2165b811b -> sha1:3052bc8a4b602afe: shared=55 discordant=8 flip_rate=14.5% delta_b_minus_a=-0.036 p=0.7266
- sha1:b62aeb57e9097af4 -> sha1:c3fa97e069022c64: shared=46 discordant=4 flip_rate=8.7% delta_b_minus_a=+0.000 p=1
- sha1:b62aeb57e9097af4 -> sha1:c4edd26b3db58732: shared=50 discordant=15 flip_rate=30.0% delta_b_minus_a=-0.260 p=0.0009766
- sha1:b62aeb57e9097af4 -> sha1:322c76ae095b3c4f: shared=50 discordant=5 flip_rate=10.0% delta_b_minus_a=+0.020 p=1
- sha1:b62aeb57e9097af4 -> sha1:63c7a0a0fa36943b: shared=50 discordant=2 flip_rate=4.0% delta_b_minus_a=+0.000 p=1
- sha1:b62aeb57e9097af4 -> sha1:13599b7287e18a4c: shared=50 discordant=5 flip_rate=10.0% delta_b_minus_a=+0.020 p=1
- sha1:b62aeb57e9097af4 -> sha1:3052bc8a4b602afe: shared=55 discordant=2 flip_rate=3.6% delta_b_minus_a=+0.000 p=1
- sha1:c3fa97e069022c64 -> sha1:c4edd26b3db58732: shared=46 discordant=13 flip_rate=28.3% delta_b_minus_a=-0.283 p=0.0002441
- sha1:c3fa97e069022c64 -> sha1:322c76ae095b3c4f: shared=46 discordant=3 flip_rate=6.5% delta_b_minus_a=-0.022 p=1
- sha1:c3fa97e069022c64 -> sha1:63c7a0a0fa36943b: shared=46 discordant=5 flip_rate=10.9% delta_b_minus_a=-0.022 p=1
- sha1:c3fa97e069022c64 -> sha1:13599b7287e18a4c: shared=46 discordant=7 flip_rate=15.2% delta_b_minus_a=-0.022 p=1
- sha1:c3fa97e069022c64 -> sha1:3052bc8a4b602afe: shared=46 discordant=5 flip_rate=10.9% delta_b_minus_a=-0.022 p=1
- sha1:c4edd26b3db58732 -> sha1:322c76ae095b3c4f: shared=50 discordant=16 flip_rate=32.0% delta_b_minus_a=+0.280 p=0.0005188
- sha1:c4edd26b3db58732 -> sha1:63c7a0a0fa36943b: shared=50 discordant=17 flip_rate=34.0% delta_b_minus_a=+0.260 p=0.00235
- sha1:c4edd26b3db58732 -> sha1:13599b7287e18a4c: shared=51 discordant=20 flip_rate=39.2% delta_b_minus_a=+0.275 p=0.002577
- sha1:c4edd26b3db58732 -> sha1:3052bc8a4b602afe: shared=50 discordant=17 flip_rate=34.0% delta_b_minus_a=+0.260 p=0.00235
- sha1:322c76ae095b3c4f -> sha1:63c7a0a0fa36943b: shared=50 discordant=7 flip_rate=14.0% delta_b_minus_a=-0.020 p=1
- sha1:322c76ae095b3c4f -> sha1:13599b7287e18a4c: shared=50 discordant=10 flip_rate=20.0% delta_b_minus_a=+0.000 p=1
- sha1:322c76ae095b3c4f -> sha1:3052bc8a4b602afe: shared=50 discordant=7 flip_rate=14.0% delta_b_minus_a=-0.020 p=1
- sha1:63c7a0a0fa36943b -> sha1:13599b7287e18a4c: shared=50 discordant=3 flip_rate=6.0% delta_b_minus_a=+0.020 p=1
- sha1:63c7a0a0fa36943b -> sha1:3052bc8a4b602afe: shared=50 discordant=0 flip_rate=0.0% delta_b_minus_a=+0.000 p=1
- sha1:13599b7287e18a4c -> sha1:3052bc8a4b602afe: shared=50 discordant=3 flip_rate=6.0% delta_b_minus_a=-0.020 p=1
