"""Bounded 32-worker stability; policy inference only, no PPO learn/update."""
import os,sys,json,time,traceback,hashlib
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'training/py')]
from candidate_env import CandidateEnv,EXP,save
import train_ppo_beam_unilateral as beam
class StabilityEnv(CandidateEnv):
 def __init__(self,*a,**kw):
  self.missing_records=0;self.missing_fields=0;self.first_missing=None;self.contact_frames=0
  super().__init__(*a,**kw)
 def step(self,action):
  r=super().step(action)
  _,records=self.native_records()
  for x in records:
   n=x.get('unavailable_data_count',0)
   if n:
    self.missing_records+=1;self.missing_fields+=n
    if self.first_missing is None:
     self.first_missing={'context':self.context(),'phase':x['phase'],'stage':x['stage'],
       'mapped_null_fields':[{k:v for k,v in m.items() if v is None or k=='path'} for m in x['mapped'] if any(v is None for v in m.values())]}
   if x['phase']=='built' and any(x['contact_geometry']):self.contact_frames+=1
  return r
 def stability_stats(self):
  return dict(self.totals,missing_records=self.missing_records,missing_fields=self.missing_fields,
   first_missing=self.first_missing,contact_frames=self.contact_frames)
def main():
 from mcr_sim.distributed import DistributedPPO,resolve_device
 import torch,psutil
 t=time.monotonic();vec=None;result={'result':'FAIL','ppo_optimizer_updates':0,'vector_steps':0}
 try:
  device=resolve_device('npu',distributed=False,local_rank=0)
  assert str(device.resolved)=='npu:0' and str(device.accelerator)=='npu'
  cfg=json.loads((ROOT.parent/'training_runs/ppo_beam_unilateral_rhs_consistent_v1_20261010_170927/candidate_configuration.json').read_text())['args']
  args=SimpleNamespace(**cfg);args.local_n_envs=32;args.distributed_rank=0
  os.environ['MCR_CONSTRAINT_SOLVER']='generic';beam.BeamSafetyMCREnv=StabilityEnv
  vec=beam.build_env(args)
  model=DistributedPPO.load(os.environ['MCR_CANDIDATE_RESUME'],env=vec,device=device.resolved)
  model.set_random_seed(42);model.min_action_std=.2;model.max_action_std=.6;model.clamp_action_std()
  initial_updates=model._n_updates
  obs=vec.reset()
  for i in range(64):
   action,_=model.predict(obs,deterministic=False)
   obs,rewards,dones,infos=vec.step(action);result['vector_steps']=i+1
   if (i+1)%16==0:print('STABILITY_VECTOR_STEPS',i+1,flush=True)
  stats=vec.env_method('stability_stats');save(EXP/'worker_stats.json',stats)
  assert model._n_updates==initial_updates
  assert len(stats)==32 and all(s['physics_substeps']==128 for s in stats)
  result.update(result='PASS',workers=32,physics_substeps=sum(s['physics_substeps'] for s in stats),
    free_negative=sum(s['free_negative'] for s in stats),committed_negative=sum(s['committed_negative'] for s in stats),
    min_committed_mm=min(s['min_committed_mm'] for s in stats),missing_records=sum(s['missing_records'] for s in stats),
    missing_fields=sum(s['missing_fields'] for s in stats),contact_frames=sum(s['contact_frames'] for s in stats),
    first_missing=next((s['first_missing'] for s in stats if s['first_missing']),None),npu=str(device.resolved))
 except BaseException:
  result['error']=traceback.format_exc();print(result['error'],flush=True)
 finally:
  if result['result']=='PASS' and vec is not None:vec.close()
  else:
   children=psutil.Process().children(recursive=True)
   for c in children:c.terminate()
   _,alive=psutil.wait_procs(children,timeout=5)
   for c in alive:c.kill()
  result['wall_s']=time.monotonic()-t;save(EXP/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
