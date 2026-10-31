"""Offline table replay; Zenodo 20048594, one SGLang fidelity. No model calls."""
import argparse,csv,json,hashlib,math,time
import numpy as np
import optuna
from scipy.stats import qmc
from optuna._hypervolume import compute_hypervolume
from optuna.samplers import NSGAIISampler
from optuna.trial import TrialState
optuna.logging.set_verbosity(optuna.logging.WARNING)

p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['fail','raw','bounded','botorch']);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True); args=p.parse_args()
source='/mnt/raid0/llm/tmp/codex3-ap66-20261031/sglang.csv'
rows=list(csv.DictReader(open(source))); keys=[k for k in rows[0] if k.startswith('cfg-') and k!='cfg-config_id']
choices={k:sorted({r[k] for r in rows}) for k in keys}
# Normalize numeric distances by each column's observed range. Non-numeric values
# use categorical mismatch. Empty values are distinct categorical states.
def numeric(k):
 try: return {v:float(v) for v in choices[k]} if all(v!='' for v in choices[k]) else None
 except ValueError: return None
nums={k:numeric(k) for k in keys}
def distance(a,b):
 total=0.
 for k in keys:
  v=nums[k]
  if v is None: total+=float(a[k]!=b[k])
  else:
   span=max(v.values())-min(v.values()); total+=((v[a[k]]-v[b[k]])/span if span else 0.)**2
 return math.sqrt(total/len(keys))
def project(params):
 ds=[distance(params,r) for r in rows]; idx=min(range(len(rows)),key=lambda i: (ds[i],i)); return rows[idx],ds[idx]
def viol(t): return (t.user_attrs['violation'],)
if args.arm=='botorch':
 import torch
 from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
 from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
 from botorch.fit import fit_gpytorch_mll
 from botorch.models import SingleTaskGP
 from botorch.models.transforms.outcome import Standardize
 from botorch.optim import optimize_acqf
 from botorch.utils.transforms import normalize,unnormalize
 from gpytorch.mlls import ExactMarginalLogLikelihood
 from optuna_integration import BoTorchSampler
 torch.set_num_threads(1); torch.manual_seed(args.seed)
 def candidates(x,y,c,bounds,pending):
  n=y.shape[-1]; x=normalize(x,bounds)
  train=torch.cat((y,c),dim=-1)
  model=SingleTaskGP(x,train,outcome_transform=Standardize(m=train.shape[-1]))
  fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model),optimizer_kwargs={'options':{'maxiter':50}})
  ref=(y.min(dim=0).values-1e-8).tolist()
  acq=qLogNoisyExpectedHypervolumeImprovement(model=model,ref_point=ref,X_baseline=x,
   objective=IdentityMCMultiOutputObjective(outcomes=list(range(n))),
   constraints=[lambda z: z[...,n]],prune_baseline=True)
  standard=torch.stack((torch.zeros_like(bounds[0]),torch.ones_like(bounds[1])))
  cand,_=optimize_acqf(acq,standard,q=1,num_restarts=3,raw_samples=32,options={'maxiter':50})
  return unnormalize(cand.detach(),bounds)
 sampler=BoTorchSampler(candidates_func=candidates,constraints_func=viol,seed=args.seed,n_startup_trials=20)
else: sampler=NSGAIISampler(seed=args.seed,constraints_func=None if args.arm=='fail' else viol)
study=optuna.create_study(directions=['maximize','maximize'],sampler=sampler)
# Exactly the same 20 Sobol-selected table rows across arms for each seed.
u=qmc.Sobol(d=1,scramble=True,seed=args.seed).random_base2(5).flatten()[:20]
for x in u:
 r=rows[min(int(x*len(rows)),len(rows)-1)];study.enqueue_trial({k:r[k] for k in keys})
measured=[r for r in rows if r['obj-request_throughput+'] and r['obj-p99_e2e_latency_ms-']]
ref=(min(float(r['obj-request_throughput+']) for r in measured),min(-float(r['obj-p99_e2e_latency_ms-']) for r in measured))
pending=[]; observed=[]; snapshots=[]; records=[]; infeasible=0; snapped=0; distances=[]; start=time.monotonic()
for i in range(100):
 t=study.ask(); params={k:t.suggest_categorical(k,choices[k]) for k in keys}; row,dist=project(params); distances.append(dist); snapped+=dist>0
 bad=int(row['cfg-context_length'])>=9216
 if bad: infeasible+=1
 t.set_user_attr('violation', ((int(row['cfg-context_length'])-9216+1)/9216 if args.arm in ('bounded','botorch') else int(row['cfg-context_length'])-9216+1) if bad else 0.)
 t.set_user_attr('source_row',row['ID']); t.set_user_attr('snap_distance',dist)
 records.append({'trial':i,'requested':params,'projected_row_id':row['ID'],'snap_distance':dist,'context_rule_infeasible':bad,'violation':t.user_attrs['violation'],'native_request_throughput_per_s':float(row['obj-request_throughput+']) if row['obj-request_throughput+'] else None,'native_p99_e2e_ms':float(row['obj-p99_e2e_latency_ms-']) if row['obj-p99_e2e_latency_ms-'] else None})
 if bad and args.arm=='fail': study.tell(t,state=TrialState.FAIL)
 elif bad:
  if observed: study.tell(t,[min(x[j] for x in observed) for j in range(2)])
  else: pending.append(t)
 else:
  vals=[float(row['obj-request_throughput+']),-float(row['obj-p99_e2e_latency_ms-'])]; observed.append(vals);study.tell(t,vals)
  if pending:
   anchor=[min(x[j] for x in observed) for j in range(2)]
   for old in pending: study.tell(old,anchor)
   pending=[]
 if i+1 in (50,100):
  front=np.array(observed,dtype=float);hv=float(compute_hypervolume(-front,np.array([-ref[0],-ref[1]]))) if len(front) else 0.
  snapshots.append({'n':i+1,'context_rule_infeasible_rate':infeasible/(i+1),'hypervolume':hv,'snap_rate':snapped/(i+1),'mean_normalized_snap_distance':sum(distances)/(i+1),'max_normalized_snap_distance':max(distances)})
  open(args.output,'w').write(json.dumps({'arm':args.arm,'seed':args.seed,'source_sha256':hashlib.sha256(open(source,'rb').read()).hexdigest(),'source_url':'https://zenodo.org/records/20048594','fidelity':'15.0-1.0-16-16-1024','script_sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest(),'protocol':'Synthetic task-specified context_length >= 9216 infeasibility, not a claim of actual OOM; unavailable native objectives stay null. Sobol samples table row indices; later independent categorical proposals use nearest observed row projection, so no optimizer advantage or out-of-sample claim.','trials':records,'reference':ref,'objective_units':['requests/s','negative p99 end-to-end latency ms'],'snap_metric':'RMS of per-knob min-max normalized numeric distance and categorical mismatch','snapshots':snapshots,'elapsed_s':time.monotonic()-start},indent=2))
