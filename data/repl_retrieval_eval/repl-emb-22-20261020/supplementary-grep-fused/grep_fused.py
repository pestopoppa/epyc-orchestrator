import json, sys
sys.path.insert(0, '/mnt/raid0/llm/worktrees/repl-emb-25-orch-20261020/scripts/benchmark')
import repl_emb_stats as S
R = '/mnt/raid0/llm/tmp/repl-emb-22-20261020/run/'
Q = {}
for l in open(R + 'queries-labelled.jsonl'):
    r = json.loads(l); Q[r['id']] = r
def ranges(g):
    return [tuple(x) for x in g['ranges']] if 'ranges' in g else [(g['start_line'], g['end_line'])]
def hit(ptrs, qid, k=5):
    g = Q[qid]['gold']
    for s, a, b in ptrs[:k]:
        if s == g['path'] and any(a <= y and b >= x for x, y in ranges(g)):
            return 1
    return 0
def load(v):
    return {json.loads(l)['qid']: json.loads(l) for l in open(R + f'perquery-{v}.jsonl')}
grep, bm25 = load('grep'), load('bm25')
dense, hyb = load('eg2-gguf-bf16-256.dense'), load('eg2-gguf-bf16-256.hybrid')
def rrf(lists, k=60, limit=10):
    sc = {}
    for lst in lists:
        for rank, p in enumerate(lst, 1):
            key = tuple(p); sc[key] = sc.get(key, 0.0) + 1.0 / (k + rank)
    order = sorted(sc, key=lambda x: (-sc[x], str(x)))
    return [list(x) for x in order[:limit]]
def fused(lex, d, qid):
    if d[qid]['failed']:
        return []
    return rrf([lex[qid]['pointers'], d[qid]['pointers']])
prim = [i for i, q in Q.items() if q.get('group') != 'trivial_head' and not q.get('answer_named_verbatim')]
strata = {'primary(all)': prim,
          'probe45': [i for i in prim if Q[i]['group'] == 'probe45'],
          'agent_nav': [i for i in prim if Q[i]['group'] == 'agent_nav'],
          'synthetic': [i for i in prim if Q[i]['group'] == 'synthetic_spill']}
vec = {
    'grep (L)': {i: hit(grep[i]['pointers'], i) for i in prim},
    'bm25': {i: hit(bm25[i]['pointers'], i) for i in prim},
    'eg2-256 dense': {i: (0 if dense[i]['failed'] else hit(dense[i]['pointers'], i)) for i in prim},
    'bm25+eg2-256 hybrid (REAL, frozen run)': {i: (0 if hyb[i]['failed'] else hit(hyb[i]['pointers'], i)) for i in prim},
    'bm25+eg2-256 (top-10 lists refused, CALIBRATION)': {i: hit(fused(bm25, dense, i), i) for i in prim},
    'grep+eg2-256 fused (SUPPLEMENTARY, top-10 lists)': {i: hit(fused(grep, dense, i), i) for i in prim},
}
agree = sum(1 for i in prim if vec['bm25+eg2-256 hybrid (REAL, frozen run)'][i] == vec['bm25+eg2-256 (top-10 lists refused, CALIBRATION)'][i])
print('calibration: real vs top-10-reconstructed bm25 hybrid agree on', agree, 'of', len(prim), 'queries (recall@5 hit/miss)')
base = vec['grep (L)']
import numpy as np
for sname, ids in strata.items():
    print('\n==', sname, 'n=', len(ids))
    for name, v in vec.items():
        a = np.array([v[i] for i in ids], float); b = np.array([base[i] for i in ids], float)
        line = f'  {name:52s} recall@5 {a.mean():.3f}'
        if 'hybrid' in name or 'fused' in name or 'dense' in name or 'refused' in name:
            if sname in ('primary(all)',):
                bs = S.paired_bootstrap(a, b)
                line += f'  diff vs grep {bs["mean_diff"]:+.3f} CI [{bs["ci_lo"]:.3f}, {bs["ci_hi"]:.3f}]'
            else:
                line += f'  diff vs grep {a.mean() - b.mean():+.3f}'
        print(line)
