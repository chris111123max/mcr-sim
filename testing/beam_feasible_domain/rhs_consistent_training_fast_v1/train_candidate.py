"""Isolated real PPO continuation using RHS-consistent native candidate."""
from pathlib import Path
import os,sys,json,time,hashlib,shutil,traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'training/py')]
import train_ppo_beam_unilateral as beam
from candidate_env import CandidateEnv,EXP,save
from mcr_sim.distributed import DistributedPPO
ORIGINAL_PARSE=beam.parse_args
OLD_RUN=ROOT.parent/'training_runs/ppo_beam_unilateral_audit_formal_100ep_20260928_224143'

class FailFastPPO(DistributedPPO):
 def learn(self,*a,**kw):
  states=self.policy.optimizer.state
  resume={'restored_optimizer_parameter_states':len(states),'restored_n_updates':int(self._n_updates),
   'restored_num_timesteps':int(self.num_timesteps),'restored_epoch':int(getattr(self,'epoch',0)),
   'restored_episodes':int(getattr(self,'global_completed_episodes',0)),
   'curriculum_stage':int(getattr(self,'curriculum_stage',-1)),
   'mode':'MODEL_PLUS_OPTIMIZER_AND_PROGRESS; environment/RNG/partial rollout not restored'}
  if not states or self._n_updates!=635 or getattr(self,'epoch',0)!=5:raise RuntimeError('CHECKPOINT_RESUME_GATE_FAILED')
  save(EXP/'resume_actual.json',resume);print('CANDIDATE_RESUME_ACTUAL',json.dumps(resume),flush=True)
  try:return super().learn(*a,**kw)
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
 def _on_step(self):
  keep=super()._on_step()
  if self.n_calls==1 or self.n_calls%32==0:
   save(EXP/'training_progress.json',{'num_timesteps':int(self.model.num_timesteps),'optimizer_updates':int(self.model._n_updates),
     'epoch':int(getattr(self.model,'epoch',5)),'global_completed_episodes':int(getattr(self.model,'global_completed_episodes',500)),
     'rollout_vector_calls':int(self.n_calls)})
   if self.n_calls==1:print('CANDIDATE_REAL_ROLLOUT_STARTED',int(self.model.num_timesteps),flush=True)
  return keep
 def _on_rollout_end(self):
  super()._on_rollout_end();print('CANDIDATE_ROLLOUT_COMPLETED',int(self.model.num_timesteps),flush=True)

def parse_args():
 args=ORIGINAL_PARSE();assert args.resume_from and EXP.is_absolute() and EXP.is_dir();old=json.loads((OLD_RUN/'logs/run_config.json').read_text())
 preserve=['seed','learning_rate','gamma','gae_lambda','clip_range','ent_coef','vf_coef','max_grad_norm',
 'initial_action_std','min_action_std','max_action_std','npu_fused_adam','max_episode_steps','settle_steps',
 'radius_observation_scale','randomize_start_target','start_window_mm','target_window_mm','randomize_initial_orientation',
 'initial_orientation_max_angle_deg','entry_tangent_points','soft_randomize_single_vessel','vessel_scale_min','vessel_scale_max',
 'valid_min_train_success_rate']
 for key in preserve:
  if key in old and hasattr(args,key):setattr(args,key,old[key])
 args.beam_safety_validation='debug';args.variant='beam_unilateral_rhs_consistent_fast_v1'
 assert args.n_envs==32 and args.physics_substeps==2 and args.time_step==0.005
 assert args.epochs==100 and args.episodes_per_epoch==100 and args.n_steps==256 and args.batch_size==1024 and args.n_epochs==5
 assert args.training_curriculum and args.device=='npu' and not args.reset_num_timesteps
 save(EXP/'candidate_configuration.json',{'args':vars(args),'preserved_fields':preserve,
 'rotation_jacobian':'world','consistent_contact_reference':True,'native_solves_per_physics_substep':2,
 'penetration_policy':'count_only','penetration_report_threshold_mm':0.100,'all_negative_clearances_recorded':True,'candidate_target_margin_mm':0.100,'dense_independent_check_every_physics_substep':True,
 'extra_physics_substeps':0,'production_modified':False})
 return args

def main():
 os.environ['MCR_CONSTRAINT_SOLVER']='generic'
 beam.BeamSafetyMCREnv=CandidateEnv;beam.parse_args=parse_args
 beam.BeamSafetyRolloutMetricsCallback=ProgressCallback;beam.baseline.DistributedPPO=FailFastPPO
 try:beam.main()
 finally:
  hashes=json.loads((HERE/'runtime/baseline_hashes.json').read_text())
  save(EXP/'baseline_unchanged.json',{'unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())})
if __name__=='__main__':main()
