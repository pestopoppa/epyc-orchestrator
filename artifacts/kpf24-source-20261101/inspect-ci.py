from pathlib import Path
from collections import Counter
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

root = Path(sys.argv[1])
repo = Path(__file__).resolve().parents[2]
record = next((root / 'extracted').iterdir())
sha = (record / 'source-commit.txt').read_text().strip()
checks = []
for line in (record / 'source-files.sha256').read_text().splitlines():
    expected, path = line.split(None, 1)
    path = path.lstrip('* ')
    blob = subprocess.check_output(['git', 'show', f'{sha}:{path}'], cwd=repo)
    actual = hashlib.sha256(blob).hexdigest()
    if actual != expected:
        raise RuntimeError(f'source mismatch: {path}')
    checks.append({'path': path, 'sha256': actual})
api = json.loads((root / 'run-api.json').read_text())
assert api['head_sha'] == sha
artifact_api = json.loads((root / 'artifacts-api.json').read_text())
zip_digest = hashlib.sha256((root / 'original-artifact.zip').read_bytes()).hexdigest()
assert artifact_api['artifacts'][0]['digest'] == 'sha256:' + zip_digest
summaries = {}
for junit in record.glob('*.xml'):
    cases = list(ET.parse(junit).getroot().iter('testcase'))
    counts = Counter()
    failures = []
    skipped = []
    for case in cases:
        name = case.attrib['classname'] + '::' + case.attrib['name']
        kind = next((kind for kind in ('failure', 'error', 'skipped') if case.find(kind) is not None), 'passed')
        counts[kind] += 1
        if kind in ('failure', 'error'):
            failures.append(name)
        if kind == 'skipped':
            skipped.append({'test': name, 'reason': case.find('skipped').attrib.get('message')})
    summaries[junit.name] = {'total': len(cases), 'counts': dict(counts), 'failures': failures, 'skipped': skipped}
receipt = {'software_verification_only': True, 'run_id': api['id'], 'source_commit': sha,
           'github_conclusion': api['conclusion'], 'artifact_sha256': zip_digest,
           'verified_source_files': checks, 'junit': summaries,
           'exit_codes': {p.name: p.read_text().strip() for p in record.glob('*.exit-code')}}
(root / 'inspection.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
