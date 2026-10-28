"""Synthetic-only paired IRT checks; no models or live fleet."""
import hashlib
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import pytest
from scripts.graph_router import irt_eg2_paired as m


def fixture(tmp_path):
    n=20; rng=np.random.default_rng(8)
    ids=[f'q{i}' for i in range(n)]; train=np.arange(16); held=np.arange(16,20)
    y=rng.random((n,3)); mask=np.ones((n,3),dtype=bool); y[0,0]=0.0
    np.savez_compressed(tmp_path/'dataset.npz',qids=ids,bge=rng.normal(size=(n,1024)).astype('f'),correctness=y,observed_mask=mask,roles=['a','b','c'],train_indices=train,heldout_indices=held)
    (tmp_path/'prompts.jsonl').write_text(''.join(json.dumps({'qid':q,'prompt':q})+'\n' for q in ids))
    manifest={'schema':'epyc.irt.p45_frozen_input.v1','artifacts_sha256':{name:m.sha(tmp_path/name) for name in ('dataset.npz','prompts.jsonl')},'train_qids':ids[:16],'heldout_qids':ids[16:]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return m.load_frozen(tmp_path)


def test_paired_pipeline_and_no_heldout_training_leak(tmp_path):
    _,d,_=fixture(tmp_path); rng=np.random.default_rng(7); eg2=rng.normal(size=(20,256))
    a=m.evaluate(d,eg2)
    assert set(a)=={'bge','eg2'} and all(np.isfinite(x) for z in a.values() for x in z.values())
    train=d['train_indices']; y=d['correctness'].copy(); mask=d['observed_mask']
    s=m.estimate_irt_scores(y[train],mask=mask[train]); y[d['heldout_indices']]=0
    s2=m.estimate_irt_scores(y[train],mask=mask[train]); np.testing.assert_array_equal(s.action_abilities,s2.action_abilities)
    assert mask[0,0] and d['correctness'][0,0]==0
    with pytest.raises(ValueError): m.evaluate(d,eg2[:,:255])
    with pytest.raises(ValueError): m.spearman([1,1],[1,2])


def test_hash_and_missing_mask_refused(tmp_path):
    fixture(tmp_path); p=tmp_path/'dataset.npz'; d=np.load(p); arrays={k:d[k] for k in d.files if k!='observed_mask'}; np.savez_compressed(p,**arrays)
    with pytest.raises(ValueError,match='hash mismatch'): m.load_frozen(tmp_path)
    manifest=json.loads((tmp_path/'manifest.json').read_text()); manifest['artifacts_sha256']['dataset.npz']=m.sha(p); (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='observed_mask'): m.load_frozen(tmp_path)


def test_owned_fake_http_client_prefix_truncate_and_cleanup():
    received=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self): self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length'])))['content'])
            body=json.dumps({'embedding':list(range(1,769))}).encode(); self.send_response(200); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    server=HTTPServer(('127.0.0.1',19113),Handler); thread=threading.Thread(target=server.serve_forever); thread.start()
    try:
        v=m.embed([{'prompt':'synthetic'}], 'http://127.0.0.1:19113')
        assert received==['title: none | text: synthetic'] and v.shape==(1,256)
        expected=np.arange(1,257,dtype=float); expected/=np.linalg.norm(expected); np.testing.assert_allclose(v[0],expected,rtol=1e-6)
        with pytest.raises(ValueError): m.embed([], 'http://127.0.0.1:8090')
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)
    assert not thread.is_alive()


def test_end_to_end_http_and_write(tmp_path, monkeypatch):
    fixture(tmp_path)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self): self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
        def do_POST(self):
            content=json.loads(self.rfile.read(int(self.headers['Content-Length'])))['content']
            seed=int(hashlib.sha256(content.encode()).hexdigest()[:8],16)
            body=json.dumps({'embedding':np.random.default_rng(seed).normal(size=768).tolist()}).encode()
            self.send_response(200); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    server=HTTPServer(('127.0.0.1',19113),Handler); thread=threading.Thread(target=server.serve_forever); thread.start()
    try:
        out=tmp_path/'new-output'/'report.json'
        runtime=tmp_path/'runtime.json'; runtime.write_text(json.dumps({'schema':'epyc.graph_router.irt_runtime.v1','synthetic':True}))
        monkeypatch.setattr('sys.argv',['paired','--frozen',str(tmp_path),'--output',str(out),'--runtime',str(runtime)])
        # Production inputs include an explicit provenance scope.
        manifest=json.loads((tmp_path/'manifest.json').read_text()); manifest['scope']='SYNTHETIC fixture; no measurement'; (tmp_path/'manifest.json').write_text(json.dumps(manifest))
        m.main(); shutil.copyfile(runtime,out.parent/'runtime.json'); report=json.loads(out.read_text())
        assert report['metric_direction']=='higher_is_better' and report['train_count']==16 and report['heldout_count']==4
        import importlib.util, os
        if os.environ.get('IRT_CAPTURE_PATH'):
            spec=importlib.util.spec_from_file_location('synthetic_irt_capture',os.environ['IRT_CAPTURE_PATH']); capture=importlib.util.module_from_spec(spec); spec.loader.exec_module(capture)
            side=capture.emit(out,tmp_path,True)
            rows=capture.c.load_rows(side,capture.validate)
            assert len(rows)==4 and all(r['extra']['synthetic'] for r in rows)
        with pytest.raises(ValueError,match='already exists'): m.main()
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)
    assert not thread.is_alive()


def test_http_error_does_not_fall_back_or_leave_client_open(monkeypatch):
    closed=[]
    class BadClient:
        def close(self): closed.append(True)
    class BadEmbedder:
        def __init__(self, config): self._http_client=BadClient()
        def embed_text(self, text, side): raise RuntimeError('synthetic HTTP failure')
    monkeypatch.setattr(m,'TaskEmbedder',BadEmbedder)
    with pytest.raises(RuntimeError,match='HTTP failure'): m.embed([{'prompt':'x'}],'http://127.0.0.1:19113')
    assert closed==[True]


def test_cross_split_duplicate_prompt_refused(tmp_path):
    fixture(tmp_path)
    path=tmp_path/'prompts.jsonl'; rows=[json.loads(x) for x in path.read_text().splitlines()]
    rows[16]['prompt']=rows[0]['prompt']; path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    manifest=json.loads((tmp_path/'manifest.json').read_text()); manifest['artifacts_sha256']['prompts.jsonl']=m.sha(path); (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='crosses split'): m.load_frozen(tmp_path)


def test_existing_projector_still_refuses_256():
    from scripts.graph_router.irt_scorer import EmbeddingIRTProjector
    p=EmbeddingIRTProjector(1024,np.zeros(1024),np.ones(1024),np.zeros((1025,2)),np.zeros(2),np.ones(2))
    with pytest.raises(ValueError,match='1024'): p.predict(np.ones(256))


def test_real_client_http_error_refuses_fallback():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self): self.send_response(200); self.end_headers()
        def do_POST(self): self.send_response(500); self.end_headers()
    server=HTTPServer(('127.0.0.1',19113),Handler); thread=threading.Thread(target=server.serve_forever); thread.start()
    try:
        with pytest.raises(RuntimeError,match='unavailable'): m.embed([{'prompt':'synthetic'}],'http://127.0.0.1:19113')
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)
    assert not thread.is_alive()
