#!/usr/bin/env python3
"""Rehash prepared model in a separately claimed post-capture job."""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def hash_stable(path, should_stop=None):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            if should_stop is not None and should_stop():
                raise InterruptedError('STOP/deadline during post-hash')
            h.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError('model changed during post-hash')
    return {'path': str(path), 'sha256': h.hexdigest(), 'device': after.st_dev,
            'inode': after.st_ino, 'size': after.st_size, 'mtime_ns': after.st_mtime_ns,
            'ctime_ns': after.st_ctime_ns}


def main():
    if os.environ.get('FILLER_DRYRUN') == '1':
        root = Path(os.environ['FILLER_DRYRUN_DIR']).resolve(strict=True)
        out = Path(os.environ['FILLER_DRYRUN_RESULT_PATH'])
        if not out.parent.resolve(strict=True).is_relative_to(root):
            raise ValueError('dry-run result escapes fixture root')
        model = out.parent / 'fixture-model'
        model.write_bytes(b'fixture only, no model')
        fact = hash_stable(model)
        record = {'schema': 'dtap.crm_posthash_fixture.v1', 'capture_kind': 'fixture', 'fact': fact}
    else:
        cfg = json.loads(Path(os.environ['CRM_POSTHASH_CONFIG']).read_bytes())
        prep = json.loads(Path(cfg['input_manifest']).read_bytes())
        if prep.get('schema') != 'dtap.crm_input_preparation.v1':
            raise ValueError('invalid preparation')
        close = Path(cfg['claim_close'])
        journal = Path(cfg['pair_journal'])
        if not close.is_file() or not journal.is_file():
            raise ValueError('post-hash requires completed provider close and pair journal')
        expected = prep['model']
        started = datetime.now(timezone.utc).isoformat()
        deadline = int(os.environ['FILLER_DEADLINE'])
        stop = Path(os.environ['FILLER_STOPFILE'])
        def should_stop():
            return stop.exists() or time.time() >= deadline
        fact = hash_stable(expected['path'], should_stop)
        if any(fact[k] != expected[k] for k in ('sha256', 'device', 'inode', 'size')):
            raise ValueError('post-hash differs from prepared model identity')
        record = {'schema': 'dtap.crm_posthash.v1', 'capture_kind': 'live', 'started_utc': started,
                  'completed_utc': datetime.now(timezone.utc).isoformat(), 'fact': fact,
                  'input_manifest_path': str(Path(cfg['input_manifest']).resolve()),
                  'input_manifest_sha256': hashlib.sha256(Path(cfg['input_manifest']).read_bytes()).hexdigest(),
                  'claim_close_path': str(close.resolve()), 'claim_close_sha256': hashlib.sha256(close.read_bytes()).hexdigest(),
                  'pair_journal_path': str(journal.resolve()), 'pair_journal_sha256': hashlib.sha256(journal.read_bytes()).hexdigest(),
                  'limitation': 'matching pre/post hashes do not exclude same-UID in-place mutation during capture'}
        out = Path(cfg['posthash_result'])
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == '__main__':
    sys.exit(main())
