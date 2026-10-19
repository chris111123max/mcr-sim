"""Same CollisionBegin fork, using the previously validated native solver."""
from __future__ import annotations
import argparse, ctypes, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'testing/beam_feasible_domain/nonlinear_relinearization_test/run_native_relinearization.py'
spec=importlib.util.spec_from_file_location('validated_native_relinearization',OLD)
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
fast, Sofa, factory=base.fast,base.Sofa,base.factory
CASES=json.loads((HERE/'cases.json').read_text())
CASE=None; META=None; SNAP=None; TARGET=None; FIRST=None; OUT=None
ORIGINAL_MEASUREMENT=base._stage_measurement

def measurement(q0,q,profile,rows):
    if len(rows['free_violations']): return ORIGINAL_MEASUREMENT(q0,q,profile,rows)
    return {'committed_clearance_mm':float(np.min(profile)*1000),'dense_worst_index':int(np.argmin(profile)),
            'selected_count':0,'row_count':0,'min_linear_row_gap_mm':None,
            'min_linear_row_gap_world_angular_mm':None,'max_abs_nonlinear_sdf_error_mm':None,
            'max_abs_nonlinear_sdf_error_world_angular_mm':None,
            'min_selected_true_sdf_gap_mm':None,'worst_dense_is_selected':False}
base._stage_measurement=measurement

def save_json(path,value):
    path.write_text(json.dumps(base._clean(value),indent=2,allow_nan=False)+'\n')

def compare_state(actual,expected):
    if actual.shape!=expected.shape or not np.isfinite(actual).all():
        return {'pass':False,'reason':'shape_or_nonfinite'}
    trans=float(np.max(np.abs(actual[:,:3]-expected[:,:3])))
    quat=float(np.max(np.abs(actual[:,3:7]-expected[:,3:7])))
    angle=float(np.max((Rotation.from_quat(actual[:,3:7])*Rotation.from_quat(expected[:,3:7]).inv()).magnitude()))
    return {'translation_max_abs_m':trans,'quaternion_max_abs':quat,'quaternion_angle_max_rad':angle,
            'pass':bool(trans<=1e-10 and quat<=1e-10 and angle<=1e-9)}

class StopTarget(BaseException): pass

class TimedController(base.RelinearizationController):
    def arm(self,*args,**kwargs):
        super().arm(*args,**kwargs)
        if not bool(self.constraint.enabled.value):
            self.rows={'row_offsets':np.array([0],dtype=np.int64),'dof_indices':np.array([],dtype=np.int64),
                       'linear_jacobian':np.empty((0,3)),'angular_jacobian':np.empty((0,3)),
                       'free_violations':np.array([]),'source_clearances':np.array([]),
                       'selected_dense_indices':np.array([],dtype=np.int64)}
    def onScriptEvent(self,event):
        super().onScriptEvent(event)
        if not self.active or not isinstance(event,dict): return
        name=str(event.get('event_name',''))
        if not name.startswith('MCR_NONLINEAR_RELINEARIZATION_STAGE_'): return
        stage=int(name.rsplit('_',1)[1])
        if stage in self.records:
            record=self.records[stage]
            record['solver_scale_tolerance']=bool(self.solver.scaleTolerance.value)
            record['solver_all_verified']=bool(self.solver.allVerified.value)
            if stage<int(self.solver.requestedPasses.value):
                planner_record=self.env._beam_planner._record
                planner_record['rows_required']=bool(planner_record.get('rows_armed',False))
            for key,data in [('native_solver_wall_ms','lastNativeSolveWallMs'),('mapping_refresh_wall_ms','lastMappingRefreshWallMs')]:
                record[key]=float(getattr(self.solver,data).value)
base.RelinearizationController=TimedController

class MultiEnv(base.BeamSafetyMCREnv):
    def __init__(self,*a,**kw):
        self.branch=None; self.passes=None; self.child=False; self.forked=False
        self.state_match=None; self.substeps=[]; self.target_animation_calls=0
        self.current=None; self.continuity=None
        super().__init__(*a,**kw)
    def _validate_solver(self):
        names=[o.getClassName() for o in self._sofa_root_node.objects if o.getClassName() in
               ('GenericConstraintSolver','LCPConstraintSolver','TestOnlyRelinearizingGenericConstraintSolver')]
        if names!=['TestOnlyRelinearizingGenericConstraintSolver']: raise RuntimeError('NATIVE_SOLVER_CLASS_MISMATCH:'+str(names))
    def _attach_beam_safety(self):
        if self._beam_attached: return
        super()._attach_beam_safety()
        registry=base.REGISTRY[id(self._sofa_root_node)]
        self.native=registry['solver']; self.ctrl=registry['controller']; self.ctrl.bind(self,self.native)
        original=self._beam_planner._collision_begin
        def hook(source):
            original(source)
            if self._beam_rl_step!=TARGET or self._beam_substep<FIRST: return
            qprev=fast._as_array(self._beam_dofs.position).copy()
            qfree=fast._as_array(self._beam_dofs.free_position).copy()
            if not self.forked:
                self.state_match={'q_prev':compare_state(qprev,SNAP['q_prev']),
                                  'q_free':compare_state(qfree,SNAP['q_free'])}
                self.state_match['geometry_equal']=self._beam_adapter.solver_geometry()==META['geometry_free']
                self.state_match['asset_transform_max_abs']=float(np.max(np.abs(np.asarray(self.asset_T_env_sim)-np.asarray(META['asset_T_env_sim']))))
                self.state_match['asset_offset_max_abs']=float(np.max(np.abs(np.asarray(self.asset_offset_sim)-np.asarray(META['asset_offset_sim']))))
                self.state_match['radius_abs_diff_m']=abs(float(self.catheter_radius)-float(META['catheter_radius']))
                self.state_match['sdf_path_equal']=str(self.sdf_grid.source_path)==str(META['sdf_path'])
                self.state_match['pass']=bool(self.state_match['q_prev']['pass'] and self.state_match['q_free']['pass']
                    and self.state_match['geometry_equal'] and self.state_match['asset_transform_max_abs']<=1e-12
                    and self.state_match['asset_offset_max_abs']<=1e-12 and self.state_match['radius_abs_diff_m']<=1e-12
                    and self.state_match['sdf_path_equal'])
                self.forked=True
                if not self.state_match['pass']: return
                common_pre_fork_ms=(time.perf_counter()-self.target_started)*1000
                for branch,passes in [('A',1),('B',2)]:
                    pid=os.fork()
                    if pid==0:
                        self.child=True; self.branch=branch; self.passes=passes
                        break
                    _,status=os.waitpid(pid,0)
                    if not os.WIFEXITED(status) or os.WEXITSTATUS(status)!=0:
                        save_json(OUT/f'{branch}_process_error.json',{'wait_status':status})
                if self.branch is None: self.branch='C'; self.passes=3
                self.target_started=time.perf_counter()-common_pre_fork_ms/1000
            if not self.state_match['pass']: return
            points,profile=fast._profile(self._beam_adapter,qfree)
            if self.substeps:
                self.continuity=compare_state(qprev,self.last_committed)
                if not self.continuity['pass']: raise RuntimeError('SUBSTEP2_NOT_FROM_ACTUAL_SUBSTEP1_COMMIT')
            self.current={'substep':int(self._beam_substep),'q_prev':qprev,'q_free':qfree,'free_points':points,
                          'free_profile':profile,'geometry':self._beam_adapter.solver_geometry(),
                          'free_clearance_mm':float(np.min(profile)*1000)}
            self.native.requestedPasses.value=self.passes
            self.native.relinearizationEnabled.value=True
            self.native.tolerance.value=1e-9
            required=bool(self._beam_planner._record.get('rows_required'))
            self.ctrl.arm(qprev,qfree,profile,self.branch,
                          initial_row_build_wall_ms=base.LAST_ROW_BUILD_WALL_MS if required else None)
        self._beam_planner._collision_begin=hook

    def capture_substep(self,wall_ms,error=None):
        q=fast._as_array(self._beam_dofs.position).copy()
        points,profile=fast._profile(self._beam_adapter,q)
        independent=self._beam_adapter.measure(q,spacing_m=1e-5)
        if self.current is None: raise RuntimeError('MISSING_TARGET_FRAME_CAPTURE')
        sub=self.current['substep']; native=self.native
        summary={'substep':sub,'free_clearance_mm':self.current['free_clearance_mm'],
                 'committed_clearance_mm':float(np.min(profile)*1000),
                 'independent_committed_clearance_mm':float(independent['min_clearance_m']*1000),
                 'physics_substep_wall_ms':wall_ms,'native_status':str(native.relinearizationStatus.value),
                 'native_solve_calls':int(native.nativeSolveCalls.value),'completed_stages':int(native.completedStages.value),
                 'mapping_refresh_count':int(native.mappingRefreshCount.value),
                 'native_solver_total_wall_ms':float(native.nativeSolveTotalWallMs.value),
                 'mapping_total_wall_ms':float(native.mappingTotalWallMs.value),
                 'geometry':self.current['geometry'],'continuity_from_previous':self.continuity,
                 'error':error,'stages':[],'nan_inf':bool(not np.isfinite(q).all() or not np.isfinite(profile).all())}
        previous=self.current['q_free']
        for stage,record in sorted(self.ctrl.records.items()):
            if stage<1: summary['error']=record.get('error'); continue
            rows=record['rows']; qstage=record['q_stage_event']; stageprof=record['dense_clearance']
            linmatch=compare_state(record['q_linearization'],previous)
            if not linmatch['pass']: raise RuntimeError('LINEARIZATION_PARENT_STATE_MISMATCH')
            worst=int(np.argmin(stageprof))
            densepoints=self._beam_adapter.sample(qstage,spacing_m=1e-5)
            location,_=fast._selected_dense_records(adapter=self._beam_adapter,q_free=qstage,dense_points=densepoints,selected_indices=[worst])
            filename=f'{self.branch}_sub{sub}_stage{stage}.npz'
            np.savez_compressed(OUT/filename,q_prev=self.current['q_prev'],q_free=self.current['q_free'],
                q_linearization_state=record['q_linearization'],q_committed=qstage,
                dense_clearance=stageprof,**rows)
            metrics=record['measurement']
            entry={k:record[k] for k in ('solver_iterations','solver_error','solver_tolerance','row_build_wall_ms',
                 'active_count','solver_constraint_count','solver_scale_tolerance','solver_all_verified','native_solve_calls_cumulative','completed_stages','mapping_refresh_count','event_stage',
                 'native_solver_wall_ms','mapping_refresh_wall_ms')}
            entry.update(stage=stage,npz=filename,**metrics,
                         worst_element=location[0],linearization_parent_match=linmatch,
                         solver_error_below_tolerance=bool(record['solver_error']<record['solver_tolerance']),
                         effective_solver_tolerance=float(record['solver_tolerance']*(record['solver_constraint_count'] if record['solver_scale_tolerance'] and not record['solver_all_verified'] else 1)))
            summary['stages'].append(entry)
            previous=qstage
        summary['final_stage_to_committed']=compare_state(previous,q)
        np.savez_compressed(OUT/f'{self.branch}_sub{sub}_final.npz',q_prev=self.current['q_prev'],
            q_free=self.current['q_free'],q_committed=q,dense_clearance=profile,
            free_dense_clearance=self.current['free_profile'])
        summary['nonpenetration_pass']=bool(summary['committed_clearance_mm']>=-0.001 and not summary['nan_inf'])
        summary['margin_pass']=bool(summary['committed_clearance_mm']>=0.100 and not summary['nan_inf'])
        self.substeps.append(summary); self.last_committed=q.copy()
        save_json(OUT/f'branch_{self.branch}.json',self.branch_result())
        return summary
    def branch_result(self):
        return {'branch':self.branch,'passes':self.passes,'solver_tolerance':1e-9,
                'case':CASE,'state_match':self.state_match,'substeps':self.substeps,
                'target_animate_calls':self.target_animation_calls,'expected_animate_calls':3-FIRST,
                'target_substep2_executed':any(s['substep']==2 for s in self.substeps),
                'extra_animate_calls':max(0,self.target_animation_calls-(3-FIRST)),
                'production_modified':False,'ppo_started':False,'direct_position_write':False,
                'rollback':False,'projection':False,'action_shielding':False}

    def _beam_traced_animate(self,root,dt):
        upcoming=self._beam_substep+1
        is_target=self._beam_rl_step==TARGET and upcoming>=FIRST
        start=time.perf_counter()
        if is_target:
            if abs(float(dt)-0.005)>1e-12: raise RuntimeError('PHYSICS_DT_MISMATCH')
            self.target_animation_calls+=1
            self.target_started=start
        try:
            result=super()._beam_traced_animate(root,dt)
        except Exception as exc:
            if is_target and self.state_match and self.state_match.get('pass'):
                self.capture_substep((time.perf_counter()-self.target_started)*1000,f'{type(exc).__name__}:{exc}')
                raise StopTarget()
            raise
        wall=(time.perf_counter()-(self.target_started if is_target else start))*1000
        if is_target:
            if not self.state_match or not self.state_match['pass']:
                save_json(OUT/'state_mismatch.json',{'case':CASE,'state_match':self.state_match,'result':'INCONCLUSIVE'})
                raise StopTarget()
            summary=self.capture_substep(wall)
            if not summary['nonpenetration_pass'] or summary['error'] or upcoming==2: raise StopTarget()
        return result

def main():
    global CASE,META,SNAP,TARGET,FIRST,OUT
    parser=argparse.ArgumentParser(); parser.add_argument('--case-id',required=True)
    cli=parser.parse_args(); CASE=next(c for c in CASES if c['id']==cli.case_id)
    OUT=HERE/'results'/CASE['id']; OUT.mkdir(parents=True,exist_ok=False)
    META=json.loads(Path(CASE['metadata']).read_text())
    with np.load(CASE['snapshot'],allow_pickle=False) as z: SNAP={k:np.array(z[k]) for k in z.files}
    TARGET=int(CASE['step']); FIRST=int(CASE['substep'])
    actions=np.asarray(SNAP['raw_actions'],dtype=np.float32)
    sha=hashlib.sha256(actions.tobytes()).hexdigest()
    if sha!=CASE['action_sha256'] or len(actions)!=TARGET: raise RuntimeError('ACTION_SNAPSHOT_GATE_FAILED')
    if hashlib.sha256(Path(CASE['snapshot']).read_bytes()).hexdigest()!=CASE['snapshot_sha256']: raise RuntimeError('SNAPSHOT_CHANGED')
    plugin=HERE/'results/native_build/lib/libMCRMultistateNativeSafety.so'
    ctypes.CDLL(str(plugin),mode=getattr(ctypes,'RTLD_GLOBAL',0))
    base.ORIGINAL_ROW_BUILDER=fast.build_rows_fast_local_fd
    fast.build_rows_fast_local_fd=base._timed_row_builder
    Sofa.Simulation.init=base._patched_sim_init
    factory.BeamSafetyMCREnv=MultiEnv
    cfg=json.loads((base.RUN/'logs/run_config.json').read_text()); args=SimpleNamespace(**cfg)
    args.local_n_envs=1; args.n_envs=1; args.distributed_rank=CASE['worker']; args.render='headless'; args.force_model=''
    args.beam_safety_plugin=str(base.PROD_PLUGIN)
    vec=None; env=None; started=time.perf_counter(); outcome={}
    try:
        vec=factory.build_env(args); env=vec.envs[0].unwrapped
        vessel=CASE['case']['vessel']; route=int(CASE['case']['target'].rsplit('_',1)[1])
        env.set_training_model_sampling_weights({vessel:1.0}); env.set_training_route_sampling_weights({vessel:{route:1.0}})
        vec.seed(int(CASE['case']['seed'])); vec.reset()
        try:
            for index,action in enumerate(actions):
                _,_,done,_=vec.step(action[None,:])
                if done[0]: raise RuntimeError(f'PREFIX_TERMINATED_AT_{index+1}')
            raise RuntimeError('TARGET_NOT_STOPPED')
        except StopTarget: pass
        outcome={'case':CASE,'action_sha256':sha,'state_match':env.state_match,'branch':env.branch,
                 'status':'NATIVE_MATRIX_COMPLETE' if env.state_match and env.state_match['pass'] else 'INCONCLUSIVE_STATE_MISMATCH'}
    except Exception as exc:
        outcome={'case':CASE,'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()}
        traceback.print_exc()
    finally:
        if vec: vec.close()
        Sofa.Simulation.init=base.ORIGINAL_SIM_INIT
        fast.build_rows_fast_local_fd=base.ORIGINAL_ROW_BUILDER
    if env is not None and env.child:
        os._exit(0 if outcome.get('status')=='NATIVE_MATRIX_COMPLETE' else 1)
    outcome['prefix_and_matrix_wall_s']=time.perf_counter()-started
    save_json(OUT/'case_result.json',outcome)
    print(json.dumps(base._clean(outcome),allow_nan=False),flush=True)
    return 0 if outcome['status']=='NATIVE_MATRIX_COMPLETE' else 3
if __name__=='__main__': raise SystemExit(main())
