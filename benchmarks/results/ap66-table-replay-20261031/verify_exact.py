"""Verify table membership, initialization, missing values and source identity."""
import csv, glob, hashlib, json
from pathlib import Path

root = Path(__file__).resolve().parent
rows = {r['ID']: r for r in csv.DictReader((root / 'sglang.csv').open())}
source_hash = hashlib.sha256((root / 'sglang.csv').read_bytes()).hexdigest()
writer_hash = hashlib.sha256((root / 'replay_exact.py').read_bytes()).hexdigest()
files = sorted(glob.glob(str(root / 'exact' / '*.json')))
assert len(files) == 40
initial = {}
for path in files:
    record = json.loads(Path(path).read_text())
    assert record['source_sha256'] == source_hash and record['script_sha256'] == writer_hash
    assert [x['n'] for x in record['snapshots']] == [50, 100]
    assert len(record['trials']) == 100
    initialization = tuple(t['table_row_id'] for t in record['trials'][:20])
    assert initial.setdefault(record['seed'], initialization) == initialization
    for t in record['trials']:
        row = rows[t['table_row_id']]
        assert all(row[k] == value for k, value in t['evaluated_params'].items())
        assert t['context_rule_infeasible'] == (int(row['cfg-context_length']) >= 9216)
        for target, source in [('native_request_throughput_per_s', 'obj-request_throughput+'),
                               ('native_p99_e2e_ms', 'obj-p99_e2e_latency_ms-')]:
            expected = float(row[source]) if row[source] else None
            assert t[target] == expected
print('40 runs, 4,000 stored/evaluated vectors, native/unknown values, and shared Sobol initialization verified')
