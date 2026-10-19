"""Test-only bounded Epoch5 replay; read-only capture of real Beam/rows."""
import os, sys, json, hashlib, time, copy, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'training/py')]
os.environ['MCR_CONSTRAINT_SOLVER']='generic'
from mcr_sim.beam_safety_env import BeamSafetyMCREnv
from mcr_sim import beam_sdf_unilateral_fast as fast
from mcr_sim.distributed import DistributedPPO
from mcr_sim.distributed.device import resolve_device
import train_ppo_beam_unilateral as factory
OUT=Path(__file__).resolve().parent
RUN=ROOT.parent/'training_runs/ppo_beam_unilateral_audit_formal_100ep_20260928_224143'
CKPT=RUN/'models/ppo_beam_unilateral_audit_epoch_005_episodes_00500.zip'

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return str(x)
    return x

def savejson(p,x): p.write_text(json.dumps(clean(x),indent=2))

class CaptureEnv(BeamSafetyMCREnv):
    case=None
    def __init__(self,*a,**kw):
        self.capture=None; self.actions=[]; self.danger=float('inf'); self.saved=False
        self.capture_dir=getattr(self,'capture_dir',OUT); self.capture_dir.mkdir(exist_ok=True,parents=True)
        super().__init__(*a,**kw)
    def _attach_beam_safety(self):
        super()._attach_beam_safety()
        planner=self._beam_planner; original=planner._collision_begin
        def captured(source):
            qprev=fast._as_array(self._beam_dofs.position).copy()
            qfree=fast._as_array(self._beam_dofs.free_position).copy()
            geo=copy.deepcopy(self._beam_adapter.solver_geometry())
            original(source)
            rec=planner._record
            if rec and rec.get('rows_required') and rec.get('planning_status')!='FAIL':
                points,clearance=fast._profile(self._beam_adapter,qfree)
                c=self._beam_constraint
                self.capture={'q_prev':qprev,'q_free':qfree,'free_points':points,'free_clearance':clearance,
                    'geometry':geo,'record':copy.deepcopy(rec)}
                for name in ('rowOffsets','dofIndices','linearJacobian','angularJacobian','freeViolations','sourceClearances'):
                    self.capture[name]=np.array(getattr(c,name).value,copy=True)
                self.capture['selected_indices']=fast._select_rows(clearance)
        planner._collision_begin=captured
    def step(self,action):
        self.actions.append(np.array(action,dtype=np.float32,copy=True))
        return super().step(action)
    def _audit_committed(self,reason):
        try:
            result=super()._audit_committed(reason)
        except RuntimeError as exc:
            if 'COMMITTED_PENETRATION' in str(exc) and self.capture:
                self.snapshot('failure',reason,str(exc)); self.saved=True
            raise
        if reason=='active_rows' and self.capture and result<self.danger:
            self.danger=result
            self.snapshot('most_dangerous',reason,None)
        return result
    def snapshot(self,name,reason,error):
        s=dict(self.capture)
        q=fast._as_array(self._beam_dofs.position).copy()
        points,clearance=fast._profile(self._beam_adapter,q)
        s.update(q_committed=q,committed_points=points,committed_clearance=clearance,
            raw_actions=np.array(self.actions,dtype=np.float32),smoothed_action=np.array(getattr(self,'_last_smoothed_action',[])))
        qprev_clear=fast._profile(self._beam_adapter,s['q_prev'])[1]
        solver=next(o for o in self._sofa_root_node.objects if o.getClassName()=='GenericConstraintSolver')
        solver_fields={}
        for key in ('currentIterations','currentError','constraintForces','graphErrors','maxIterations','tolerance','computeConstraintForces'):
            try: solver_fields[key]=clean(getattr(solver,key).value)
            except Exception: solver_fields[key]='CURRENT_INTERFACE_UNOBSERVABLE'
        metadata={'case':self.case,'exact_original_failure':False,'error':error,'reason':reason,
            'rl_step':self._beam_rl_step,'substep':self._beam_substep,'episode':1,
            'vessel':getattr(self,'current_sampling_model',None),'target':getattr(self,'current_target_route_id',None),
            'seed':self.case['seed'],'worker_env':'single diagnostic env; NOT original worker',
            'q_prev_clearance_mm':float(np.min(qprev_clear)*1000),
            'q_free_clearance_mm':float(np.min(s['free_clearance'])*1000),
            'q_committed_clearance_mm':float(np.min(clearance)*1000),
            'activeCount':int(self._beam_constraint.activeCount.value),'row_count':len(s['freeViolations']),
            'action_sha256':hashlib.sha256(np.array(self.actions,dtype=np.float32).tobytes()).hexdigest(),
            'geometry_free':s.pop('geometry'),'geometry_committed':self._beam_adapter.solver_geometry(),
            'record':s.pop('record'),'solver_fields':solver_fields,'curriculum_stage':self.episode_curriculum_stage,
            'curriculum_dr_fraction':self.episode_curriculum_dr_fraction,'checkpoint':str(CKPT)}
        np.savez_compressed(self.capture_dir/(name+'_snapshot.npz'),**s)
        metadata.update(asset_T_env_sim=clean(self.asset_T_env_sim),asset_offset_sim=clean(self.asset_offset_sim),asset_source_to_sim_scale=self.asset_source_to_sim_scale,catheter_radius=self.catheter_radius,sdf_path=self.sdf_grid.source_path)
        savejson(self.capture_dir/(name+'_metadata.json'),metadata)

def main():
    device=resolve_device('npu',distributed=False,local_rank=0)
    print('DEVICE',device,flush=True)
    cfg=json.loads((RUN/'logs/run_config.json').read_text())
    args=SimpleNamespace(**cfg)
    args.local_n_envs=1; args.n_envs=1; args.distributed_rank=0; args.render='headless'; args.force_model=''
    args.beam_safety_plugin=str(ROOT/'mcr_sim/native/beam_unilateral/_build/lib/libMCRBeamLinearizedUnilateral.so')
    factory.BeamSafetyMCREnv=CaptureEnv
    model=DistributedPPO.load(str(CKPT),device='npu:0')
    plan=[{'vessel':'B02','target':4,'seed':66},{'vessel':'B02','target':4,'seed':42},
          {'vessel':'B01','target':1,'seed':42},{'vessel':'B02','target':1,'seed':15204}]
    results=[]; started=time.monotonic()
    for case in plan:
        CaptureEnv.case=case
        vec=factory.build_env(args); env=vec.envs[0].unwrapped
        env.set_training_model_sampling_weights({case['vessel']:1.0})
        env.set_training_route_sampling_weights({case['vessel']:{case['target']:1.0}})
        vec.seed(case['seed'])
        try:
            obs=vec.reset()
            print('CASE',case,'stage',env.episode_curriculum_stage,'DR',env.episode_curriculum_dr_fraction,flush=True)
            for i in range(2048):
                action,_=model.predict(obs,deterministic=True)
                obs,reward,done,info=vec.step(action)
                if (i+1)%100==0: print('PROGRESS',case,i+1,'worst_audited',env._beam_worst_audited_clearance_m*1000,flush=True)
                if done[0]: break
            results.append({'case':case,'status':'COMPLETED_NO_CAPTURED_FAILURE','steps':i+1})
        except Exception as exc:
            results.append({'case':case,'status':'CAPTURED_SAME_CLASS_FAILURE' if env.saved else 'ERROR',
                'steps':env._beam_rl_step,'error':str(exc)})
            print('STOP',results[-1],flush=True)
            traceback.print_exc()
            if env.saved: savejson(OUT/'reproduction_result.json',{'results':results,'runtime_s':time.monotonic()-started}); return
            raise
        finally: vec.close()
    savejson(OUT/'reproduction_result.json',{'results':results,'runtime_s':time.monotonic()-started})
if __name__=='__main__': main()
