"""Isolated from-scratch PPO training using RHS-consistent native candidate."""
from pathlib import Path
import os,sys,json,time,hashlib,shutil,traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'training/py')]
import train_ppo_beam_unilateral as beam
from candidate_env import CandidateEnv,EXP,save
from mcr_sim.distributed import DistributedPPO
ORIGINAL_PARSE=beam.parse_args
CONFIG=HERE/'config.json'

class FailFastPPO(DistributedPPO):
 def learn(self,*a,**kw):
  states=self.policy.optimizer.state
  initial={'optimizer_parameter_states':len(states),'n_updates':int(self._n_updates),
   'num_timesteps':int(self.num_timesteps),'mode':'FROM_SCRATCH; no checkpoint loaded'}
  if states or self._n_updates!=0 or self.num_timesteps!=0:raise RuntimeError('FRESH_TRAINING_GATE_FAILED')
  save(EXP/'initial_state.json',initial);print('CANDIDATE_FRESH_TRAINING',json.dumps(initial),flush=True)
  try:
   result=super().learn(*a,**kw)
   save(EXP/'finished.json',{'optimizer_updates':int(self._n_updates),'num_timesteps':int(self.num_timesteps),'finite_parameters':all(bool(p.isfinite().all()) for p in self.policy.parameters())})
   return result
  except BaseException as exc:
   path=EXP/'models/failure_latest_model'
   try:self.save(str(path));print('FAIL_FAST_CHECKPOINT_SAVED',str(path)+'.zip',flush=True)
   except Exception as save_exc:print('FAIL_FAST_CHECKPOINT_SAVE_ERROR',repr(save_exc),flush=True)
   save(EXP/'master_failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'latest_model':str(path)+'.zip'})
   try:
    import psutil
    children=psutil.Process().children(recursive=True)
    for child in children:child.terminate()
    _,alive=psutil.wait_procs(children,timeout=5)
    for child in alive:child.kill()
   except Exception:traceback.print_exc()
   raise

class ProgressCallback(beam.BeamSafetyRolloutMetricsCallback):
 def _on_rollout_start(self):
  self.rollout_start=time.monotonic()
  return super()._on_rollout_start()
 def _on_step(self):
  keep=super()._on_step()
  self.penetration_stats=[i.get('beam_penetration',{}) for i in self.locals.get('infos',[])]
  if self.n_calls==1 or self.n_calls%32==0:
   save(EXP/'training_progress.json',{'num_timesteps':int(self.model.num_timesteps),'optimizer_updates':int(self.model._n_updates),
     'epoch':int(getattr(self.model,'epoch',0)),'global_completed_episodes':int(getattr(self.model,'global_completed_episodes',0)),
     'rollout_vector_calls':int(self.n_calls)})
   if self.n_calls==1:print('CANDIDATE_REAL_ROLLOUT_STARTED',int(self.model.num_timesteps),flush=True)
  return keep
 def _on_rollout_end(self):
  for key in ['committed_negative','single_solves','double_solves','audited_substeps','skipped_substeps']:
   self.logger.record('beam_adaptive/'+key,sum(i.get(key,0) for i in getattr(self,'penetration_stats',[])))
  wall=time.monotonic()-self.rollout_start
  save(EXP/'rollout_summary.json',{'wall_seconds':wall,'env_steps':self.model.n_steps*self.model.n_envs,'fps':self.model.n_steps*self.model.n_envs/wall,'workers':self.penetration_stats})
  super()._on_rollout_end();print('CANDIDATE_ROLLOUT_COMPLETED',int(self.model.num_timesteps),flush=True)

def parse_args():
 args=ORIGINAL_PARSE();assert not args.resume_from and EXP.is_absolute() and EXP.is_dir();old=json.loads(CONFIG.read_text())
 preserve=['seed','learning_rate','gamma','gae_lambda','clip_range','ent_coef','vf_coef','max_grad_norm',
 'initial_action_std','min_action_std','max_action_std','npu_fused_adam','max_episode_steps','settle_steps',
 'radius_observation_scale','randomize_start_target','start_window_mm','target_window_mm','randomize_initial_orientation',
 'initial_orientation_max_angle_deg','entry_tangent_points','soft_randomize_single_vessel','vessel_scale_min','vessel_scale_max',
 'valid_min_train_success_rate']
 for key in preserve:
  if key in old and hasattr(args,key):setattr(args,key,old[key])
 args.beam_safety_validation='audit';args.variant='beam_unilateral_rhs_consistent_adaptive_v1'
 assert args.n_envs in (32,64) and args.physics_substeps==2 and args.time_step==0.005
 assert args.epochs==100 and args.episodes_per_epoch==100 and args.n_steps==256 and args.batch_size==1024 and args.n_epochs==5
 assert args.training_curriculum and args.device=='npu' and not args.reset_num_timesteps
 save(EXP/'candidate_configuration.json',{'args':vars(args),'preserved_fields':preserve,
 'rotation_jacobian':'world','consistent_contact_reference':True,'native_solves_per_physics_substep':'adaptive 1 or 2',
 'penetration_policy':'episode_terminate_gt_0p100mm_or_invalid_sdf','penetration_report_threshold_mm':0.100,'all_checked_negative_clearances_recorded':True,'candidate_target_margin_mm':0.100,'dense_independent_check_every_physics_substep':False,'second_solve_threshold_mm':0.150,
 'extra_physics_substeps':0,'production_modified':False})
 return args

def main():
 os.environ['MCR_CONSTRAINT_SOLVER']='generic'
 beam.BeamSafetyMCREnv=CandidateEnv;beam.parse_args=parse_args
 beam.BeamSafetyRolloutMetricsCallback=ProgressCallback;beam.baseline.DistributedPPO=FailFastPPO
 protected=['training/py/train_ppo.py','training/py/train_ppo_beam_unilateral.py','mcr_sim/mcr_rl_env.py','mcr_sim/beam_safety_env.py','mcr_sim/beam_sdf_unilateral_fast.py']
 hashes={str(ROOT/p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected}
 try:beam.main()
 finally:
  save(EXP/'baseline_unchanged.json',{'unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())})
if __name__=='__main__':main()
