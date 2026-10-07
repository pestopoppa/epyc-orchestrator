"""Capture bounded native outcomes for the complete ET13/W6 related test modules."""
from __future__ import annotations
import ast, hashlib, importlib.metadata, importlib.util, json, os, platform, re, stat, subprocess, sys, tomllib
from pathlib import Path
import xml.etree.ElementTree as ET

W6_PIN = "ce48ed2c386c2f69d23ad87289e384d7942ee7a5"
BASE_PIN = W6_PIN  # fresh candidate source is W6 plus the reviewed ET13 two-file patch
SOURCE_PIN = "73a3f746d152ea983768f938522a944053a5762c"
ROOT_PIN = "72a0d06a251667fe6dc1cdf05ab19313e03f9736"
ROOT_CONTEXT_FILES = ["scripts/ci/native_conformance.py", "scripts/vidya/adapters/__init__.py", "scripts/vidya/adapters/ci_conformance.py", "scripts/vidya/claim_tuple.py", "scripts/vidya/ingest_sources.py", "scripts/vidya/adapters/README.md", "handoffs/active/vidya-belief-substrate-program.md", "tests/vidya/test_ci_conformance.py", "scripts/vidya/lattice.py", "scripts/vidya/frames.py", "scripts/vidya/canonical.py", "handoffs/active/eval-tower-architecture-audit-2026-07-20.md"]
PY_PIN = "3.13.15"
REQUIREMENT_SEEDS = ["httpx", "math-verify", "pytest", "pyyaml"]
ENV = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1", "PYTHONDONTWRITEBYTECODE":"1", "PYTHONHASHSEED":"0",
       "PYTHONUNBUFFERED":"1", "PYTEST_ADDOPTS":"", "PYTEST_PLUGINS":"", "ORCHESTRATOR_MOCK_MODE":"1",
       "ORCHESTRATOR_IGNORE_RUNTIME_STACK_FACTS":"1"}

def sha(p: Path) -> str:
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        a=os.fstat(fd)
        if not stat.S_ISREG(a.st_mode) or a.st_nlink != 1: raise RuntimeError(f"not a single-link regular file: {p}")
        h=hashlib.sha256()
        with os.fdopen(fd,'rb',closefd=False) as f:
            for b in iter(lambda:f.read(1<<20),b''): h.update(b)
        z=os.fstat(fd)
        if (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)!=(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns): raise RuntimeError(f"changed while hashing: {p}")
        named=os.stat(p,follow_symlinks=False)
        if (z.st_dev,z.st_ino)!=(named.st_dev,named.st_ino): raise RuntimeError(f"path rebound while hashing: {p}")
        return h.hexdigest()
    finally: os.close(fd)

def git(repo,*args): return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
def snapshot(root:Path):
    out={'.':'directory'}; stack=[(root,'')]
    while stack:
        d,prefix=stack.pop()
        for e in os.scandir(d):
            rel=f'{prefix}/{e.name}' if prefix else e.name; m=e.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(m): out[rel]='symlink:'+os.readlink(e.path)
            elif stat.S_ISDIR(m): out[rel+'/']='directory'; stack.append((Path(e.path),rel))
            elif stat.S_ISREG(m) and e.stat(follow_symlinks=False).st_nlink==1: out[rel]=sha(Path(e.path))
            else: raise RuntimeError(f'unsupported result object: {rel}')
    return dict(sorted(out.items()))
def tracked(repo,rel):
    p=repo/rel; info=p.lstat(); ent=git(repo,'ls-tree','HEAD','--',rel).split()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or len(ent)<3 or ent[1]!='blob' or ent[0] not in ('100644','100755'): raise RuntimeError(f'untracked/nonregular input {rel}')
    return p,ent[0],ent[2]
def _marker(node, env):
    if isinstance(node,ast.Expression): return _marker(node.body,env)
    if isinstance(node,ast.Name): return env[node.id]
    if isinstance(node,ast.Constant): return node.value
    if isinstance(node,ast.BoolOp): return all(_marker(x,env) for x in node.values) if isinstance(node.op,ast.And) else any(_marker(x,env) for x in node.values)
    if isinstance(node,ast.Compare):
        left=_marker(node.left,env)
        for op,right_node in zip(node.ops,node.comparators):
            right=_marker(right_node,env)
            ok=(left==right if isinstance(op,ast.Eq) else left!=right if isinstance(op,ast.NotEq) else left<right if isinstance(op,ast.Lt) else left<=right if isinstance(op,ast.LtE) else left>right if isinstance(op,ast.Gt) else left>=right if isinstance(op,ast.GtE) else False)
            if not ok:return False
            left=right
        return True
    raise RuntimeError('unsupported marker grammar')
def closure(lock, seeds):
    packages={x['name'].lower().replace('_','-'):x for x in lock['package']}
    env={'implementation_name':'cpython','platform_python_implementation':'CPython','platform_machine':'x86_64','sys_platform':'linux','python_full_version':'3.13.15','python_version':'3.13'}
    start=[x.lower().replace('_','-') for x in seeds]
    seen=set(); todo=list(start)
    while todo:
        n=todo.pop()
        if n in seen: continue
        if n not in packages: raise RuntimeError('lock missing dependency '+n)
        seen.add(n)
        for d in packages[n].get('dependencies',[]):
            if isinstance(d,str): todo.append(re.split(r'[<>=!~; ]',d,1)[0].lower().replace('_','-'))
            elif isinstance(d,dict) and d.get('name') and (not d.get('marker') or _marker(ast.parse(d['marker'],mode='eval'),env)):
                todo.append(d['name'].lower().replace('_','-'))
    return packages,seen

def main():
    app=Path(os.environ['GITHUB_WORKSPACE'])/'app'; carrier=Path(os.environ['GITHUB_WORKSPACE'])/'carrier'
    result=Path(os.environ['RUNNER_TEMP'])/'ni08-et13-w6'/'result'; status_path=result/'status.json'
    status={'state':'preparing','fixture_execution_conformant':None,'native_outcome_kind':'not_captured','shared_grade_state':'not_started','promotion_or_release_acceptance':None}
    receipt=None; stage='setup'
    status_path.write_text(json.dumps(status,sort_keys=True)+'\n')
    try:
        if platform.python_version()!=PY_PIN or sys.platform!='linux' or platform.machine()!='x86_64' or sys.prefix==sys.base_prefix: raise RuntimeError('runner/Python/venv differs from declared hosted context')
        for k,v in ENV.items():
            if os.environ.get(k)!=v: raise RuntimeError('environment mismatch '+k)
        if any(k in os.environ for k in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONHOME','PYTHONPATH')): raise RuntimeError('ambient loader/Python path is present')
        if os.environ.get('EPYC_ORCH_ROOT')!=str(app) or os.environ.get('VIDYA_ORCH_WRITER_ROOT')!=str(app): raise RuntimeError('writer root is not the reviewed APP checkout')
        absent=[os.environ.get(k,'') for k in ('ORCHESTRATOR_PATHS_LLAMA_CPP_BIN','ORCHESTRATOR_PATHS_LLAMA_MTMD','ORCHESTRATOR_PATHS_LLAMA_SERVER')]
        if not all(x.startswith('/absent/') and not Path(x).exists() for x in absent): raise RuntimeError('serving binaries are not absent')
        llm=Path(os.environ.get('ORCHESTRATOR_PATHS_LLM_ROOT',''))
        if not llm.is_dir() or any(llm.iterdir()): raise RuntimeError('isolated model directory is not empty')
        if os.environ.get('GITHUB_REF')!='refs/heads/codex/ni08-et13-w6-ci-20261007' or os.environ.get('GITHUB_EVENT_NAME')!='push': raise RuntimeError('unexpected workflow event/ref')
        if os.environ.get('ROOT_CARRIER_PIN')!=ROOT_PIN: raise RuntimeError('ROOT context environment pin mismatch')
        if git(app,'status','--porcelain','--untracked-files=all') or git(carrier,'status','--porcelain','--untracked-files=all'): raise RuntimeError('checkout is not clean, including untracked files')
        if git(carrier,'rev-parse','HEAD')!=ROOT_PIN: raise RuntimeError('native carrier pin mismatch')
        if git(app,'rev-parse','HEAD')!=os.environ.get('GITHUB_SHA'): raise RuntimeError('APP checkout differs from triggering source SHA')
        cases_p,_,_=tracked(app,'scripts/ci/ni08_et13_w6_native_cases.json'); cases=json.loads(cases_p.read_text())
        case_count=cases.get('case_count')
        if cases.get('base_commit')!=BASE_PIN or cases.get('w6_commit')!=W6_PIN or cases.get('source_commit')!=SOURCE_PIN or len(cases['cases'])!=case_count: raise RuntimeError('case/source manifest pin/count mismatch')
        if cases.get('root_carrier')!=ROOT_PIN or cases.get('root_context_files')!=ROOT_CONTEXT_FILES: raise RuntimeError('ROOT context identity mismatch')
        binding=cases.get('prospective_source_binding',{})
        if binding.get('vidya_task')!='VB-ET13-ROUTING-CONFORMANCE' or binding.get('parent_task')!='ET-13 honest routing buckets (E5 remaining LOWs sweep), handoffs/active/eval-tower-architecture-audit-2026-07-20.md': raise RuntimeError('ET13 owning task binding mismatch')
        if binding.get('current_table_row_status')!='actual ET13 task/source row is published and bound in the exact ROOT context; carrier registry remains the existing ci-fixture-conformance source': raise RuntimeError('ET13 prospective source-table approval gate missing')
        if git(app,'rev-parse',BASE_PIN+'^{tree}')!=cases.get('base_tree') or subprocess.run(['git','-C',str(app),'merge-base','--is-ancestor',BASE_PIN,'HEAD']).returncode!=0: raise RuntimeError('tested APP is not descended from exact reviewed W6 source')
        if git(app,'rev-parse',SOURCE_PIN+'^{tree}')!=cases.get('source_tree') or subprocess.run(['git','-C',str(app),'merge-base','--is-ancestor',SOURCE_PIN,'HEAD']).returncode!=0: raise RuntimeError('tested APP is not descended from exact reviewed ET13 source')
        inputs=[]
        declared=['.github/workflows/ni08-et13-w6-native.yml','scripts/ci/ni08_et13_w6_native_capture.py','scripts/ci/ni08_et13_w6_native_cases.json','scripts/ci/ni08_et13_w6_native_requirements.txt','pyproject.toml','uv.lock']
        for rel in declared:
            p,mode,oid=tracked(app,rel); inputs.append({'repo':'app','path':rel,'mode':mode,'blob':oid,'sha256':sha(p),'size_bytes':p.stat().st_size})
        root_inputs=[]
        expected_root={row['path']:row for row in cases.get('root_context',[])}
        if set(expected_root)!=set(ROOT_CONTEXT_FILES): raise RuntimeError('ROOT source closure is incomplete')
        for rel in ROOT_CONTEXT_FILES:
            p,mode,oid=tracked(carrier,rel); row={'repo':'root','path':rel,'mode':mode,'blob':oid,'sha256':sha(p),'size_bytes':p.stat().st_size}; pin=expected_root[rel]
            if mode!=pin['git_mode'] or oid!=pin['git_blob'] or row['sha256']!=pin['sha256'] or row['size_bytes']!=pin['size_bytes']: raise RuntimeError('ROOT bound source mismatch '+rel)
            root_inputs.append(row); inputs.append(row)
        for row in cases['bound_files']:
            p,mode,oid=tracked(app,row['path'])
            if mode!=row['git_mode'] or oid!=row['git_blob'] or sha(p)!=row['sha256']: raise RuntimeError('bound source/data mismatch '+row['path'])
            inputs.append({'repo':'app','path':row['path'],'mode':mode,'blob':oid,'sha256':row['sha256'],'size_bytes':row['size_bytes']})
        lock=tomllib.loads((app/'uv.lock').read_text()); pkgs,names=closure(lock,REQUIREMENT_SEEDS)
        req=(app/'scripts/ci/ni08_et13_w6_native_requirements.txt').read_text()
        got={}; current=None; requirement_count=0
        for line in req.splitlines():
            m=re.match(r'^([a-zA-Z0-9_.-]+)==([^ ]+) \\\\?$',line)
            if m:
                current=m.group(1).lower().replace('_','-'); requirement_count += 1
                if current in got: raise RuntimeError('duplicate locked package requirement '+current)
                got[current]={'version':m.group(2),'hashes':set()}; continue
            m=re.search(r'--hash=sha256:([0-9a-f]{64})',line)
            if m and current: got[current]['hashes'].add(m.group(1))
        if requirement_count!=len(names) or set(got)!=names: raise RuntimeError('requirement package set differs from exact uv.lock closure')
        for n in names:
            pkg=pkgs[n]; expected={h['hash'].split(':',1)[1] for h in pkg.get('wheels',[]) if h.get('hash','').startswith('sha256:')}
            if not expected or got[n]['version']!=pkg['version'] or got[n]['hashes']!=expected: raise RuntimeError('version or full wheel-hash set mismatch '+n)
            try: installed=importlib.metadata.version(pkg['name'])
            except importlib.metadata.PackageNotFoundError: raise RuntimeError('locked package missing '+n)
            if installed!=pkg['version']: raise RuntimeError('installed version mismatch '+n)
        manifest=result/'source-manifest.json'
        manifest.write_text(json.dumps({'schema':'epyc.ni08.et13_w6.source_manifest.v1','base_commit':BASE_PIN,'source_commit':SOURCE_PIN,'app_combined_sha':git(app,'rev-parse','HEAD'),'app_tree':git(app,'rev-parse','HEAD^{tree}'),'root_carrier':ROOT_PIN,'root_context_files':root_inputs,'uv_lock_blob':git(app,'ls-tree','HEAD','uv.lock').split()[2],'test_modules':cases['whole_test_modules'],'case_count':case_count,'bound_inputs':inputs,'runtime_data':cases['runtime_data'],'native_outcome_acceptance':None},sort_keys=True,indent=2)+'\n')
        pre_status=result/'pre-status.json'
        pre_status.write_text(json.dumps({'state':'setup_complete','fixture_execution_conformant':None,'promotion_or_release_acceptance':None},sort_keys=True)+'\n')
        environment=result/'environment.json'
        environment.write_text(json.dumps({'python_version':platform.python_version(),'python_executable':sys.executable,'venv_prefix':sys.prefix,'base_prefix':sys.base_prefix,'platform':platform.platform(),'arch':platform.machine(),'python_pin':PY_PIN,'workflow_ref':os.environ['GITHUB_REF'],'event':os.environ['GITHUB_EVENT_NAME'],'launch_mode':'env-i','path':os.environ.get('PATH'),'home':os.environ.get('HOME'),'tmpdir':os.environ.get('TMPDIR'),'inherited_pythonpath':False,'loader_environment_absent':True,'plugin_autoload':'disabled','environment':ENV,'decision_acceptance':None},sort_keys=True,indent=2)+'\n')
        context=[app/'scripts/ci/ni08_et13_w6_native_requirements.txt',app/'uv.lock',manifest,app/'scripts/ci/ni08_et13_w6_native_cases.json',app/'.github/workflows/ni08-et13-w6-native.yml',app/'scripts/ci/ni08_et13_w6_native_capture.py',pre_status,environment,result/'pip-install.log',result/'pip-freeze.txt']
        for row in cases['bound_files']: context.append(app/row['path'])
        for rel in ROOT_CONTEXT_FILES: context.append(carrier/rel)
        for p in (result/'status.json',result/'pip-install.log',result/'pip-freeze.txt',manifest,pre_status,environment): sha(p)
        input_custody={'schema':'epyc.ni08_et13_w6.input_custody.v1','state':'pre_capture','app_commit':git(app,'rev-parse','HEAD'),'carrier_commit':git(carrier,'rev-parse','HEAD'),'source_manifest_sha256':sha(manifest),'declared_input_count':len(inputs),'declared_inputs':inputs,'result_files_before_capture':snapshot(result),'native_outcome_kind':'not_captured','fixture_execution_conformant':None,'shared_grade_state':'not_started','journal_record_grade':'not_in_scope','promotion_or_release_acceptance':None}
        custody_path=result/'input-custody.json'
        custody_path.write_text(json.dumps(input_custody,sort_keys=True,indent=2)+'\n')
        context.append(custody_path)
        before=snapshot(result)
        spec=importlib.util.spec_from_file_location('pinned_native_conformance',carrier/'scripts/ci/native_conformance.py'); api=importlib.util.module_from_spec(spec); sys.modules[spec.name]=api; spec.loader.exec_module(api)
        selections=[x['nodeid'] for x in cases['cases']]
        junit=result/'original-junit.xml'; temp=result/'pytest-tmp'; temp.mkdir()
        argv=[sys.executable,'-m','pytest','-c','/dev/null','--noconftest','--rootdir',str(app),'--import-mode=importlib','-p','no:cacheprovider','-o','addopts=','-q',*cases['whole_test_modules'],f'--basetemp={temp}',f'--junitxml={junit}']
        os.environ['PYTHONPATH']=str(app)
        stage='outer_native_capture'
        status.update(state='running',case_count=case_count,fixture_execution_conformant=None,native_outcome_kind='not_captured',shared_grade_state='not_started'); status_path.write_text(json.dumps(status,sort_keys=True)+'\n')
        receipt=api.capture_fixture_execution(argv=argv,cwd=app,junit=junit,output=result/'native',repositories={'app':app,'carrier':carrier},read_paths=context,selections=selections)
        if git(app,'status','--porcelain','--untracked-files=all') or git(carrier,'status','--porcelain','--untracked-files=all'): raise RuntimeError('APP or ROOT checkout changed during selected-module capture')
        after=snapshot(result)
        junit_bytes=(result/'native'/'original-junit.xml').read_bytes(); jr=ET.fromstring(junit_bytes)
        actual={(x.get('classname',''),x.get('name','')) for x in jr.iter('testcase')}
        expected={(x['classname'],x['name']) for x in cases['cases']}; counts=receipt.get('summary',{}).get('counts',{})
        exact_case_set=actual==expected and len(actual)==case_count and counts.get('collected')==case_count
        all_passed=exact_case_set and counts.get('executed')==case_count and counts.get('passed')==case_count and counts.get('skipped')==0 and counts.get('failure')==0 and counts.get('error')==0
        inventory={'schema':'epyc.ni08_et13_w6.run_root_inventory.v1','before_capture':before,'after_capture':after,'after_shared_grade':None,'shared_grade_applicable':True,'selected_case_count':case_count,'original_outcome':receipt.get('fixture_execution_conformant'),'exact_case_set':exact_case_set,'all_passed':all_passed,'promotion_or_release_acceptance':None}
        outcome=receipt.get('fixture_execution_conformant')
        if outcome is not None and type(outcome) is not bool: raise RuntimeError('outer conformance result is not TRUE/FALSE/NULL')
        outcome_kind='true' if outcome is True else 'false' if outcome is False else 'null'
        native_receipt=result/'native'/'receipt.json'
        grade_source_hashes={row['path']:row['sha256'] for row in root_inputs}
        hashes_before_grade={row['path']:sha(carrier/row['path']) for row in root_inputs}
        if hashes_before_grade!=grade_source_hashes: raise RuntimeError('ROOT grader/readset changed after capture and before shared grading')
        pregrade={'schema':'epyc.ni08_et13_w6.shared_grade_request.v1','outer_receipt_path':'native/receipt.json','outer_receipt_sha256':sha(native_receipt),'outer_outcome_kind':outcome_kind,'fixture_execution_conformant':outcome,'selected_case_count':case_count,'exact_case_set':exact_case_set,'all_passed':all_passed,'root_carrier':ROOT_PIN,'grader_source_hashes':grade_source_hashes,'verified_source_hashes_before_grade':hashes_before_grade,'result_files_before_grade':snapshot(result),'adapter_id':'vidya.adapters.ci_conformance/v1','registry_source':'ci-fixture-conformance','registry_task':'VB-CI-CONFORMANCE','shared_grade_state':'pending','journal_record_grade':'not_in_scope','promotion_or_release_acceptance':None}
        pregrade_path=result/'pregrade-custody.json'
        pregrade_path.write_text(json.dumps(pregrade,sort_keys=True,indent=2)+'\n')
        grade_result={'state':'declined_null','native_outcome_kind':outcome_kind,'fixture_execution_conformant':outcome,'reason':'existing ci_conformance.native_rows emits no tuple for a NULL proposition','journal_record_grade':'not_in_scope','promotion_or_release_acceptance':None}
        if outcome is not None and not exact_case_set:
            grade_result={'state':'not_graded_case_set_mismatch','native_outcome_kind':outcome_kind,'fixture_execution_conformant':outcome,'reason':'outer receipt did not contain the exact 140 selected case identities','journal_record_grade':'not_in_scope','promotion_or_release_acceptance':None}
        if outcome is not None and exact_case_set:
            stage='shared_ci_conformance_projection'
            sys.path.insert(0,str(carrier)); sys.path.insert(0,str(carrier/'scripts'/'vidya'))
            import scripts.vidya.ingest_sources as registry_module
            from claim_tuple import grade as shared_grade
            import claim_tuple as claim_tuple_module
            if Path(registry_module.__file__).resolve()!=(carrier/'scripts/vidya/ingest_sources.py').resolve(): raise RuntimeError('ROOT source registry import escaped pinned carrier')
            source=registry_module.SOURCES.get('ci-fixture-conformance')
            if source is None or source.task!='VB-CI-CONFORMANCE' or source.module!='ci_conformance' or source.natives!='native_rows' or source.project!='project_ci_conformance': raise RuntimeError('ROOT CI-conformance registry row differs from the bound VB source')
            if binding['current_carrier_row']!={'name':source.name,'module':source.module,'natives':source.natives,'project':source.project,'task':source.task}: raise RuntimeError('actual ROOT carrier row differs from prebound source-table context')
            adapter=source.load()
            native_reader=getattr(adapter,source.natives); projector=getattr(adapter,source.project)
            if Path(adapter.__file__).resolve()!=(carrier/'scripts/vidya/adapters/ci_conformance.py').resolve() or Path(claim_tuple_module.__file__).resolve()!=(carrier/'scripts/vidya/claim_tuple.py').resolve(): raise RuntimeError('shared grader import escaped pinned ROOT carrier')
            if shared_grade is not claim_tuple_module.grade: raise RuntimeError('grader function identity mismatch')
            rows=native_reader(native_receipt)
            if len(rows)!=1: raise RuntimeError('TRUE/FALSE outer receipt did not yield exactly one existing ci_conformance native row')
            tuple_value=projector(rows[0])
            stage='shared_claim_tuple_grade'
            quality,traceability,reasons=shared_grade(tuple_value)
            hashes_after={row['path']:sha(carrier/row['path']) for row in root_inputs}
            if hashes_after!=grade_source_hashes: raise RuntimeError('ROOT grader/readset changed during shared grading')
            if git(app,'status','--porcelain','--untracked-files=all') or git(carrier,'status','--porcelain','--untracked-files=all'): raise RuntimeError('APP or ROOT checkout changed during shared grading')
            grade_result={'state':'graded','native_outcome_kind':outcome_kind,'fixture_execution_conformant':outcome,'adapter_id':'vidya.adapters.ci_conformance/v1','registry_source':'ci-fixture-conformance','registry_task':'VB-CI-CONFORMANCE','measurement_id':tuple_value.measurement_id,'metric':tuple_value.metric,'value':tuple_value.value,'claim':tuple_value.claim,'quality':quality,'traceability':traceability,'reasons':reasons,'expected_ceiling_match':(quality,traceability)==('Judged','Located'),'shared_grade_source_hashes':hashes_after,'journal_record_grade':'not_in_scope','promotion_or_release_acceptance':None}
        (result/'shared-grade-custody.json').write_text(json.dumps(grade_result,sort_keys=True,indent=2)+'\n')
        inventory['after_shared_grade']=snapshot(result)
        (result/'run-root-inventory.json').write_text(json.dumps(inventory,sort_keys=True,indent=2)+'\n')
        grade_ok=grade_result['state']=='graded' and grade_result.get('expected_ceiling_match') is True
        status.update(state='complete' if exact_case_set and all_passed and outcome is True and grade_ok else 'native_outcome_retained',fixture_execution_conformant=outcome,native_outcome_kind=outcome_kind,shared_grade_state=grade_result['state'],junit_exact_case_set=exact_case_set,junit_all_passed=all_passed,native_receipt='native/receipt.json',shared_grade='shared-grade-custody.json',promotion_or_release_acceptance=None)
        status_path.write_text(json.dumps(status,sort_keys=True)+'\n')
        return 0 if exact_case_set and all_passed and outcome is True and grade_ok else 1
    except Exception as exc:
        outer_value=receipt.get('fixture_execution_conformant') if isinstance(receipt,dict) else None
        outer_kind=('true' if outer_value is True else 'false' if outer_value is False else 'null') if isinstance(receipt,dict) else 'exception'
        error_record={'schema':'epyc.ni08_et13_w6.shared_grade_error_custody.v1','failed_stage':stage,'outer_native_outcome_kind':outer_kind,'fixture_execution_conformant':outer_value,'processing_outcome_kind':'exception','exception_type':type(exc).__name__,'diagnostic':str(exc),'shared_grade_state':'exception' if isinstance(receipt,dict) else 'not_started','journal_record_grade':'not_in_scope','promotion_or_release_acceptance':None}
        (result/'shared-grade-error-custody.json').write_text(json.dumps(error_record,sort_keys=True,indent=2)+'\n')
        status.update(state='processing_error' if isinstance(receipt,dict) else 'setup_failure',diagnostic=f'{type(exc).__name__}: {exc}',fixture_execution_conformant=outer_value,native_outcome_kind=outer_kind,shared_grade_state=error_record['shared_grade_state'],promotion_or_release_acceptance=None)
        status_path.write_text(json.dumps(status,sort_keys=True)+'\n'); return 1
if __name__=='__main__': raise SystemExit(main())
