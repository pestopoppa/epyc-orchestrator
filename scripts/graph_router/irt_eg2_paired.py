#!/usr/bin/env python3
"""Prospective paired IRT projector experiment; never writes production models.

Targets are historical aggregate role correctness, not independent response trials.
The held-out items' targets use action abilities estimated from training items only.
Both arms use the same fixed item IDs and targets. Higher Spearman rho is better.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from orchestration.repl_memory.embedder import EmbeddingConfig, TaskEmbedder
from scripts.graph_router.irt_scorer import estimate_irt_scores, fit_embedding_projector, IRTScores


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_frozen(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') != 'epyc.irt.p45_frozen_input.v1':
        raise ValueError('unexpected input schema')
    for name in ('dataset.npz', 'prompts.jsonl'):
        if sha(root / name) != manifest['artifacts_sha256'][name]:
            raise ValueError('input hash mismatch: ' + name)
    data = np.load(root / 'dataset.npz', allow_pickle=False)
    required = {'qids', 'bge', 'correctness', 'observed_mask', 'roles', 'train_indices', 'heldout_indices'}
    if not required.issubset(data.files):
        raise ValueError('missing arrays including explicit observed_mask')
    qids = data['qids'].tolist(); n = len(qids)
    prompts = [json.loads(x) for x in (root / 'prompts.jsonl').read_text().splitlines()]
    if len(set(qids)) != n or [r['qid'] for r in prompts] != qids:
        raise ValueError('missing, duplicated or misaligned question ids')
    train, held = data['train_indices'], data['heldout_indices']
    if train.ndim != 1 or held.ndim != 1 or sorted(np.r_[train, held].tolist()) != list(range(n)):
        raise ValueError('split overlaps or omits rows')
    if [qids[i] for i in train] != manifest['train_qids'] or [qids[i] for i in held] != manifest['heldout_qids']:
        raise ValueError('split IDs mismatch')
    train_text = {prompts[i]['prompt'] for i in train}
    if any(prompts[i]['prompt'] in train_text for i in held):
        raise ValueError('identical prompt text crosses split')
    y, mask, bge = data['correctness'], data['observed_mask'], data['bge']
    if y.ndim != 2 or mask.dtype != bool or y.shape != mask.shape or y.shape[0] != n or bge.shape != (n, 1024):
        raise ValueError('invalid array dimensions or mask')
    if not np.all(mask.any(axis=1)) or not np.all(np.isfinite(y[mask])) or np.any((y[mask] < 0) | (y[mask] > 1)) or not np.all(np.isfinite(bge)):
        raise ValueError('invalid observed correctness or BGE')
    return manifest, data, prompts


def heldout_targets(y, mask, abilities):
    """Per-item statistics with action abilities frozen from training rows."""
    weights = mask.astype(float); count = weights.sum(axis=1)
    success = np.where(mask, y, 0).sum(axis=1) / count
    difficulty = -np.log(np.clip(success, 1e-4, 1-1e-4) / (1-np.clip(success, 1e-4, 1-1e-4)))
    mean = (weights * abilities).sum(axis=1) / count
    delta = np.where(mask, abilities - mean[:, None], 0)
    cov = (delta * np.where(mask, y - success[:, None], 0)).sum(axis=1) / count
    var = (delta * delta).sum(axis=1) / count
    return np.column_stack((difficulty, np.abs(cov / np.maximum(var, 1e-8))))


def ranks(v):
    order = np.argsort(v, kind='stable'); result = np.empty(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i + 1
        while j < len(v) and v[order[j]] == v[order[i]]: j += 1
        result[order[i:j]] = (i + j - 1) / 2
        i = j
    return result


def spearman(a, b):
    a, b = ranks(np.asarray(a)), ranks(np.asarray(b))
    a -= a.mean(); b -= b.mean(); denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0: raise ValueError('undefined rank correlation: constant target or prediction')
    return float(np.dot(a, b) / denom)


def embed(prompts, url):
    if url != 'http://127.0.0.1:19113': raise ValueError('only private port 19113 permitted')
    client = TaskEmbedder(EmbeddingConfig(server_url=url, use_parallel=False, use_fallback=False,
                                         allow_subprocess=False, truncate_dim=256))
    try:
        vectors = []
        for record in prompts:
            deadline = os.environ.get('FILLER_DEADLINE')
            stop = os.environ.get('FILLER_STOPFILE')
            if (deadline and time.time() >= int(deadline)) or (stop and Path(stop).exists()):
                raise RuntimeError('grant expired or revoked before next prompt')
            vectors.append(client.embed_text(record['prompt'], side='document'))
        vectors = np.stack(vectors)
    finally:
        if client._http_client is not None: client._http_client.close()
    if vectors.shape != (len(prompts), 256) or not np.all(np.isfinite(vectors)) or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError('invalid EG2 embeddings')
    return vectors


def evaluate(d, eg2, *, artifacts=None):
    train, held = d['train_indices'], d['heldout_indices']; y=d['correctness']; mask=d['observed_mask']
    if eg2.shape != (len(y), 256) or not np.all(np.isfinite(eg2)): raise ValueError('expected 256-dimensional EG2')
    scores = estimate_irt_scores(y[train], mask=mask[train])
    truth = heldout_targets(y[held], mask[held], scores.action_abilities)
    result = {}
    for name, x, dim in (('bge',d['bge'],1024), ('eg2',eg2,256)):
        p = fit_embedding_projector(x[train], scores, embedding_dim=dim)
        pred = np.column_stack(p.predict(x[held]))
        if artifacts is not None:
            artifacts[name + '_prediction'] = pred
            for key in ('feature_mean', 'feature_scale', 'weights', 'target_mean', 'target_scale'):
                artifacts[name + '_' + key] = getattr(p, key)
        result[name] = {'difficulty_spearman': spearman(truth[:,0], pred[:,0]),
                        'discrimination_spearman': spearman(truth[:,1], pred[:,1])}
    if artifacts is not None:
        artifacts['heldout_targets'] = truth
        artifacts['action_abilities_train'] = scores.action_abilities
        artifacts['train_difficulty'] = scores.latent_difficulty
        artifacts['train_discrimination'] = scores.latent_discrimination
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--frozen', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--url', default='http://127.0.0.1:19113')
    ap.add_argument('--runtime', type=Path, required=True, help='prospective runtime identity from launcher')
    args=ap.parse_args()
    if args.output.exists(): raise ValueError('output already exists')
    manifest,d,prompts=load_frozen(args.frozen)
    runtime=json.loads(args.runtime.read_text())
    if runtime.get('schema') != 'epyc.graph_router.irt_runtime.v1': raise ValueError('invalid runtime provenance')
    artifacts={}; started=time.time(); eg2=embed(prompts,args.url); metrics=evaluate(d,eg2,artifacts=artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_path=args.output.with_suffix('.npz')
    if artifact_path.exists(): raise ValueError('artifact already exists')
    with artifact_path.open('xb') as f: np.savez_compressed(f, eg2=eg2, **artifacts)
    report={'schema':'epyc.graph_router.irt_eg2_paired.v1', 'source_scope':manifest['scope'],
            'manifest_sha256':sha(args.frozen/'manifest.json'), 'input_artifacts':manifest['artifacts_sha256'],
            'train_count':len(d['train_indices']), 'heldout_count':len(d['heldout_indices']),
            'metrics':metrics, 'metric_direction':'higher_is_better', 'criterion':'both EG2 rho >= BGE rho',
            'passed':all(metrics['eg2'][k]>=metrics['bge'][k] for k in metrics['bge']),
            'baseline':'paired BGE refit on same split; historical BGE vectors used truncated prompts',
            'eg2':'production TaskEmbedder document side; first 256 then L2', 'elapsed_s':time.time()-started,
            'runtime':runtime, 'runtime_sha256':sha(args.runtime), 'artifact_sha256':sha(artifact_path)}
    with args.output.open('x') as f: json.dump(report,f,indent=2,sort_keys=True); f.write('\n')

if __name__=='__main__': main()
