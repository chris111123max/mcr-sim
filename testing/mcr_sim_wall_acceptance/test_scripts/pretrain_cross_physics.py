"""Two production RL steps per accepted vessel; no learner, no physics changes."""
import sys,json,time,traceback,resource
from pathlib import Path
from dataclasses import replace
import numpy as np
HERE=Path(__file__).resolve().parent;TEST=HERE.parent;PYTHON=TEST.parents[1]
sys.path[:0]=[str(PYTHON),str(TEST)]
from mcr_sim.config import DEFAULT_CONFIG
from mcr_sim.vessel_bundle import load_vessel_bundle
from mcr_sim.rl_env import MagneticNavigationEnv
from mcr_sim.sofa_simulation import SofaBackend
from run_wall_challenge import geometry,object_data
from run_acceptance import jsonable,_log_messages
class StopPhysicalTest(BaseException):pass

def main(vessel,seed,stage_a,out):
    vessel=Path(vessel).resolve();seed=int(seed);out=Path(out);out.mkdir(exist_ok=False);backend=None;stream=None
    report={'seed':seed,'status':'FAIL','actions':[],'physics_substeps':0,'production_substeps_per_action':22,'errors':[],'limits':{'rl_actions':2,'physics_substeps':44,'tip_jump_per_substep_m':.005}}
    started=time.monotonic();cpu0=time.process_time();context={}
    try:
        if json.loads(Path(stage_a).read_text())['status']!='PASS':raise RuntimeError('Stage A dependency did not pass')
        cfg=replace(DEFAULT_CONFIG,vessel_root=vessel.parent);backend=SofaBackend(cfg);env=MagneticNavigationEnv(backend,config=cfg,bundles=[load_vessel_bundle(vessel)])
        obs,info=env.reset(seed=seed,options={'vessel_seed':seed,'target_branch':'lcca'})
        if obs.shape!=(41,) or not np.all(np.isfinite(obs)):raise ValueError('Reset Observation invalid')
        s=backend.state();s.validate();report['reset']={'observation':obs.tolist(),'info':info,'stages':backend.stages,'irc':object_data(backend._handles.ir_controller),'tip_m':s.tip_m.tolist(),'inserted_length_m':s.inserted_length_m,'dt_s':float(backend._root.dt.value)}
        if report['reset']['dt_s']!=.006:raise ValueError('Actual dt mismatch')
        original=backend.physics_substep;stream=(out/'physical_substeps.jsonl').open('w')
        def observe(field,delta):
            before=backend.state();t0=time.monotonic();diagnostic=original(field,delta)
            try:
                s=backend.state();s.validate();g=geometry(backend);logs=_log_messages(backend)
                jump=float(np.linalg.norm(s.tip_m-before.tip_m));row={**context,'physics_substep':backend.physics_substeps_executed,'integration_seconds':time.monotonic()-t0,'tip_m':s.tip_m.tolist(),'tip_displacement_m':(s.tip_m-before.tip_m).tolist(),'inserted_length_m':s.inserted_length_m,'magnetic_field_t':s.magnetic_field_world_t.tolist(),'geometry':g,'simulation_error':diagnostic.simulation_error,'logs':logs}
                stream.write(json.dumps(jsonable(row),allow_nan=False)+'\n');stream.flush()
                if diagnostic.simulation_error or logs['errors'] or jump>.005:raise StopPhysicalTest('Numerical/SOFA error or >5mm tip jump')
                if not np.all(np.isfinite(np.asarray(backend._handles.dofs.velocity.value))):raise StopPhysicalTest('Nonfinite DOF velocity')
                return diagnostic
            except StopPhysicalTest:raise
            except BaseException:raise StopPhysicalTest(traceback.format_exc())
        backend.physics_substep=observe
        for i,a in enumerate(([0.,0.,0.],[.1,0.,.125]),1):
            context.update({'rl_action':i,'action':a});t0=time.monotonic();count0=backend.physics_substeps_executed
            obs,reward,terminated,truncated,info=env.step(np.array(a,np.float32));s=backend.state();s.validate()
            n=backend.physics_substeps_executed-count0;finite=obs.shape==(41,) and bool(np.all(np.isfinite(obs)))
            report['actions'].append({'rl_action':i,'action':a,'substeps':n,'observation':obs.tolist(),'finite_41':finite,'reward':reward,'info':info,'terminated':terminated,'truncated':truncated,'tip_m':s.tip_m.tolist(),'insertion_m':s.inserted_length_m,'seconds':time.monotonic()-t0})
            print('REAL_ACTION',seed,i,'substeps',n,'41finite',finite,flush=True)
            if n!=22 or not finite or terminated or truncated or not info.get('transition_valid'):raise StopPhysicalTest('Invalid or ended production transition')
        report['status']='PASS'
    except BaseException:report['errors'].append(traceback.format_exc())
    finally:
        if backend is not None:
            report['physics_substeps']=backend.physics_substeps_executed;report['final_logs']=_log_messages(backend);backend.close()
        if stream:stream.close()
        report['seconds']=time.monotonic()-started;report['cpu_seconds']=time.process_time()-cpu0;report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (out/'report.json').write_text(json.dumps(jsonable(report),indent=2,allow_nan=False)+'\n');print('CROSS_PHYSICS',seed,report['status'],report['physics_substeps'],flush=True)
    return 0 if report['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main(*sys.argv[1:]))
