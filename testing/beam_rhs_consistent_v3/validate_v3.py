#!/usr/bin/env python3
import argparse,json,math,os,sys,time
from pathlib import Path
import numpy as np
P=Path("/data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python")
T=P/"training/beam_rhs_consistent"
PL=T/"native/_build_cpp_v3/lib/libMCRContactReferenceConsistency.so"
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--integration-steps",type=int,default=128)
 ap.add_argument("--ppo-steps",type=int,default=8)
 ap.add_argument("--seed",type=int,default=15204)
 ap.add_argument("--run-dir",default="")
 a=ap.parse_args()
 run=Path(a.run_dir) if a.run_dir else P/"testing/beam_rhs_consistent_v3/_runtime"/("run_"+time.strftime("%Y%m%d_%H%M%S"))
 wr=run/"worker_runs/candidate"; wr.mkdir(parents=True,exist_ok=True)
 run.mkdir(parents=True,exist_ok=True)
 os.environ.update(MCR_CANDIDATE_RUN_DIR=str(wr),MCR_RHS_PLUGIN=str(PL),MCR_CONSTRAINT_SOLVER="generic",MCR_SOFA_DT="0.005")
 sys.path[:0]=[str(T),str(P),str(P/"training/py")]
 import train_ppo_beam_unilateral as beam
 from fast_guard_native import NativeCoarseBeamGuard
 tg={"calls":0,"s":0.0}; og=NativeCoarseBeamGuard.__call__
 def g(self,*x,**k):
  t=time.perf_counter()
  try:return og(self,*x,**k)
  finally:tg["calls"]+=1;tg["s"]+=time.perf_counter()-t
 NativeCoarseBeamGuard.__call__=g
 from candidate_env import CandidateEnv
 oa=CandidateEnv._beam_traced_animate
 def anim(self,root,dt):
  out=oa(self,root,dt)
  ev=getattr(self,"_v3_events",None)
  if ev is None:ev=[];self._v3_events=ev
  ev.append({"solves":int(self.native.nativeSolveCalls.value),
   "stages":int(self.native.completedStages.value),
   "mapping_refreshes":int(self.native.mappingRefreshCount.value),
   "status":str(self.native.relinearizationStatus.value),
   "solve_ms":float(self.native.nativeSolveTotalWallMs.value),
   "mapping_ms":float(self.native.mappingTotalWallMs.value),
   "rows":int(self.native.currentNumConstraints.value),
   "beam_rows":len(self.rows["freeViolations"]) if self.rows is not None else 0,
   "coarse_safe":bool(self.frame.get("coarse_safe",False)) if self.frame else False,
   "audit":bool(str(self.native.auditPath.value))})
  return out
 CandidateEnv._beam_traced_animate=anim;beam.BeamSafetyMCREnv=CandidateEnv
 saved=sys.argv[:]
 sys.argv=["validate_v3","--device","npu","--n-envs","1","--epochs","1","--episodes-per-epoch","1",
  "--n-steps",str(a.ppo_steps),"--batch-size",str(a.ppo_steps),"--n-epochs","1","--max-episode-steps","2048",
  "--physics-substeps","2","--frame-skip","1","--time-step","0.005","--training-curriculum",
  "--beam-safety-validation","audit","--beam-safety-audit-interval-substeps","16",
  "--beam-safety-audit-near-clearance-mm","0.300","--beam-safety-plugin",
  str(P/"mcr_sim/native/beam_unilateral/_build/lib/libMCRBeamLinearizedUnilateral.so"),
  "--seed",str(a.seed),"--exp-name","v3_isolated_validation"]
 try:args=beam.parse_args()
 finally:sys.argv=saved
 args.local_n_envs=1;args.distributed_rank=0;args.seed=a.seed
 from mcr_sim.distributed import initialize_distributed,configure_npu_execution,DistributedPPO
 ctx=initialize_distributed(False,"npu",1,0,"hccl")
 if ctx.device.resolved!="npu:0":raise RuntimeError("NPU_PREFLIGHT_FAILED "+repr(ctx.device))
 npu_exec=configure_npu_execution("npu",enabled=False)
 env=None
 try:
  env=beam.build_env(args);obs=env.reset()
  if not np.isfinite(np.asarray(obs)).all():raise RuntimeError("NONFINITE_RESET")
  env.action_space.seed(a.seed);unit=env.envs[0].unwrapped
  base_sub=int(unit.totals["physics_substeps"]);base_calls=int(unit._native_coarse_guard.calls)
  t=time.perf_counter();n=0;early=None
  for _ in range(a.integration_steps):
   act=np.asarray([env.action_space.sample()],dtype=np.float32)
   obs,r,d,info=env.step(act);n+=1
   if not np.isfinite(np.asarray(obs)).all() or not np.isfinite(np.asarray(r)).all():raise RuntimeError("NONFINITE_REAL_STEP")
   if d[0] and info[0].get("beam_surface_failure"):early=info[0].get("beam_surface_failure_reason");break
  wall=time.perf_counter()-t;subs=int(unit.totals["physics_substeps"])-base_sub
  calls=int(unit._native_coarse_guard.calls)-base_calls
  ev=list(getattr(unit,"_v3_events",[]))
  integ={"completed_rl_steps":n,"physics_substeps":subs,"expected_substeps":2*n,
   "integration_wall_s":wall,"aggregate_fps":n/max(wall,1e-9),"finite":True,
   "early_episode_failure":early,"native_coarse_calls":calls,
   "python_parity_checks":int(unit._native_coarse_guard._compared),
   "coarse_guard_total_ms":tg["s"]*1000,"coarse_guard_mean_ms":tg["s"]*1000/max(tg["calls"],1),
   "totals":dict(unit.totals),"substep_events":ev,"solver":unit.native.getClassName(),
   "coarseSafe_field":hasattr(unit.native,"coarseSafe"),
   "response":str(unit._sofa_root_node.getObject("Response").response.value),
   "beam_constraint":unit._beam_constraint.getClassName(),
   "root_objects":[o.getClassName() for o in unit._sofa_root_node.objects]}
  if subs!=2*n or calls!=subs:raise RuntimeError("PHYSICS_OR_NATIVE_GUARD_COUNT_MISMATCH")
  if int(unit._native_coarse_guard._compared)<3:raise RuntimeError("REAL_STATE_PARITY_NOT_THREE")
  if integ["solver"]!="TestOnlyRelinearizingGenericConstraintSolver":raise RuntimeError("V3_NATIVE_SOLVER_INACTIVE")
  if "ReferenceFrictionContactConstraint" not in integ["response"]:raise RuntimeError("CONTACT_RESPONSE_INACTIVE")
  if "Beam" not in integ["beam_constraint"]:raise RuntimeError("BEAM_UNILATERAL_INACTIVE")
  env.reset()
  from stable_baselines3.common.callbacks import BaseCallback
  class Finite(BaseCallback):
   def __init__(self):super().__init__(verbose=0);self.steps=0;self.ok=True
   def _on_step(self):
    self.steps+=1
    for k in ("new_obs","actions","rewards"):
     v=self.locals.get(k)
     if v is not None:
      try:self.ok=self.ok and bool(np.isfinite(np.asarray(v)).all())
      except Exception:pass
    return True
  cb=Finite()
  import torch as th
  kw={"log_std_init":math.log(float(args.initial_action_std)),
   "net_arch":dict(pi=[512,512,256],vf=[512,512,256]),"activation_fn":th.nn.SiLU}
  model=DistributedPPO("MlpPolicy",env,learning_rate=float(args.learning_rate),
   n_steps=a.ppo_steps,batch_size=a.ppo_steps,n_epochs=1,gamma=float(args.gamma),
   gae_lambda=float(args.gae_lambda),clip_range=float(args.clip_range),ent_coef=float(args.ent_coef),
   vf_coef=float(args.vf_coef),max_grad_norm=float(args.max_grad_norm),policy_kwargs=kw,
   tensorboard_log=str(run/"tb"),seed=a.seed,device=ctx.device.resolved,verbose=0,
   distributed_context=ctx,min_action_std=float(args.min_action_std),max_action_std=float(args.max_action_std))
  before=[x.detach().clone() for x in model.policy.parameters()]
  t=time.perf_counter();model.learn(total_timesteps=a.ppo_steps,callback=cb);learn_s=time.perf_counter()-t
  after=list(model.policy.parameters());delta=max(float((x.detach()-y).abs().max().cpu()) for x,y in zip(after,before))
  metrics={}
  for k,v in model.logger.name_to_value.items():
   try:
    z=float(v)
    if math.isfinite(z):metrics[str(k)]=z
   except Exception:pass
  if model._n_updates<1 or cb.steps<a.ppo_steps or not cb.ok or delta<=0:raise RuntimeError("PPO_UPDATE_OR_FINITE_CHECK_FAILED")
  if not any("approx_kl" in k for k in metrics) or not any("value_loss" in k for k in metrics):raise RuntimeError("PPO_METRICS_MISSING")
  (run/"models").mkdir(exist_ok=True);ck=run/"models/v3_ppo_smoke.zip";model.save(str(ck))
  ppo={"timesteps":int(model.num_timesteps),"updates":int(model._n_updates),"callback_steps":cb.steps,
   "finite":cb.ok and all(bool(th.isfinite(x).all().item()) for x in after),"parameter_max_delta":delta,
   "learn_wall_s":learn_s,"env_steps_per_s_including_update":a.ppo_steps/max(learn_s,1e-9),
   "metrics":metrics,"checkpoint":str(ck)}
 finally:
  if env is not None:env.close()
 statuses=[json.loads(p.read_text()) for p in wr.glob("beam_records/worker_*/status.json")]
 res={"result":"PASS","run_dir":str(run),"npu":{"device":ctx.device.resolved,"execution":str(npu_exec)},
  "physics":{"RL_hz":100,"physics_hz":200,"substeps":2,"dt":0.005},
  "integration":integ,"ppo":ppo,"worker_statuses":statuses}
 (run/"result.json").write_text(json.dumps(res,indent=2,allow_nan=False))
 print("V3_REAL_SOFA_PPO_VALIDATION_PASS");print(json.dumps(res,indent=2,allow_nan=False))
if __name__=="__main__":main()
