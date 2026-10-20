"""Production scenes, exact full-step replay, native timing, test-only observation."""
import sys,json,time,traceback,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
HERE=Path(__file__).resolve().parent;TEST=HERE.parent;PYTHON=TEST.parents[1]
sys.path[:0]=[str(TEST),str(PYTHON)]
from mcr_sim.config import DEFAULT_CONFIG
from mcr_sim.sofa_simulation import SofaBackend
from mcr_sim.vessel_bundle import load_vessel_bundle
from mcr_sim.rl_env import MagneticNavigationEnv
from run_wall_challenge import geometry,solver_metrics,object_data
from run_adaptive_contact import contact_objects
from run_acceptance import jsonable,_log_messages
import Sofa
from SofaRuntime import Timer
class PhysicalStop(BaseException):pass

def save(p,r):p.write_text(json.dumps(jsonable(r),indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(root,kind):
    if kind not in ('fine','candidate'): raise ValueError('kind must be fine or candidate')
    root=Path(root).resolve();gate=json.loads((root/'wrapper_retry/geometry_report.json').read_text())
    assert gate['stage_a_status']=='PASS' and gate['formal_validation_passed']
    assert gate['preservation']['status']=='PASS'
    plan=json.loads((root/'replay_plan.json').read_text());vessel=Path(gate['source_directory'] if kind=='fine' else gate['candidate_directory']);seed=int(gate.get('seed',101))
    assert int(json.loads((vessel/'metadata.json').read_text())['seed'])==seed
    assert len(plan['full_rl_actions'])==15 and all(len(item['action'])==3 for item in plan['full_rl_actions'])
    if kind=='fine':assert sha(vessel/'collision_inner.stl')==plan['source_stl_sha256']
    else:assert sha(vessel/'collision_inner.stl')==gate['candidate_stl_sha256']
    out=root/('sofa_'+kind);out.mkdir(exist_ok=False);started=time.monotonic()
    report={'status':'FAIL','seed':seed,'kind':kind,'vessel':str(vessel),'stl_sha256':sha(vessel/'collision_inner.stl'),'timing':[],'contact_actions':[],'native_timer_records':[],'errors':[],'candidate_searches':0,'contact_status':'NOT_REACHED','limits':{'contact_full_RL_actions':15,'contact_physics_substeps':330,'wall_seconds':600,'tip_jump_stop_m':.005},'physics_parameters_changed':False}
    protected={str(p):sha(p) for p in (PYTHON/'mcr_sim').rglob('*.py')}
    cfg=replace(DEFAULT_CONFIG,vessel_root=vessel.parent,vessel_seeds=(seed,));backend=SofaBackend(cfg);env=MagneticNavigationEnv(backend,config=cfg,bundles=[load_vessel_bundle(vessel)])
    animate=Sofa.Simulation.animate;phase='timing';stream=None
    def timed_animate(*a,**kw):
        Timer.begin('MeshPerf');t=time.perf_counter()
        try:return animate(*a,**kw)
        finally:
            duration=time.perf_counter()-t
            try:report['native_timer_records'].append({'animate_seconds':duration,'records':Timer.getRecords('MeshPerf')})
            finally:Timer.end('MeshPerf')
    try:
        o,info=env.reset(seed=seed,options={'vessel_seed':seed,'target_branch':'lcca'})
        Timer.setEnabled('MeshPerf',True);Timer.setInterval('MeshPerf',1000000);Timer.setOutputType('MeshPerf','stdout');Sofa.Simulation.animate=timed_animate
        for j in range(2):
            t=time.monotonic();before=backend.physics_substeps_executed
            a=np.array([0.,0.,.125 if j==0 else 0.],np.float32);o,reward,term,trunc,info=env.step(a)
            assert o.shape==(41,) and np.isfinite(o).all() and info['transition_valid'] and not term and not trunc and backend.physics_substeps_executed-before==22
            report['timing'].append({'action':a.tolist(),'seconds':time.monotonic()-t,'actual_substeps':22});save(out/'report.json',report);print('TIMING',kind,j,report['timing'][-1]['seconds'],flush=True)
        report['timing_physics_substeps']=backend.physics_substeps_executed;Sofa.Simulation.animate=animate
        # Reset production scene, not positions. Replay starts with original seed/target.
        o,info=env.reset(seed=seed,options={'vessel_seed':seed,'target_branch':'lcca'});phase='contact'
        report['reset_observation']=o.tolist();report['reset_info']=info
        original=backend.physics_substep;context={};stream=(out/'physical_substeps.jsonl').open('w')
        def observed(field,delta):
            if backend.physics_substeps_executed>=330 or time.monotonic()-started>600:raise PhysicalStop('Declared compute budget')
            before=backend.state();t=time.perf_counter();d=original(field,delta);integration=time.perf_counter()-t
            state=backend.state();state.validate();raw=geometry(backend);logs=_log_messages(backend);objects=contact_objects(backend._root)
            points=np.asarray(raw['mapped_collision_positions_m']);edges=np.asarray(backend._handles.instrument.CollisionModel.getObject('collisEdgeSet').edges.value,int).reshape(-1,2)
            lengths=np.linalg.norm(points[edges[:,1]]-points[edges[:,0]],axis=1)
            row={**context,'physical_substep':backend.physics_substeps_executed,'substep_within_action':(backend.physics_substeps_executed-1)%22+1,'integration_seconds':integration,'geometry':raw,'tip_m':state.tip_m.tolist(),'inserted_length_m':state.inserted_length_m,'field_t':state.magnetic_field_world_t.tolist(),'commanded_field_t':np.asarray(field).tolist(),'insertion_delta_m':float(delta),'contact_objects':objects,'solver':solver_metrics(backend),'collision_topology_edges':len(edges),'collision_points':len(points),'geometrically_nonzero_edges_gt_1e_12_m':int(np.count_nonzero(lengths>1e-12)),'sofa_logs':logs,'simulation_error':d.simulation_error}
            stream.write(json.dumps(jsonable(row),allow_nan=False)+'\n');stream.flush()
            if objects and report['contact_status']=='NOT_REACHED':report['contact_status']='CONSTRAINT_OBJECT_OBSERVED';report['first_constraint_object_substep']=backend.physics_substeps_executed
            if d.simulation_error or logs['errors'] or not np.isfinite(np.asarray(backend._handles.dofs.velocity.value)).all() or np.linalg.norm(state.tip_m-before.tip_m)>.005:raise PhysicalStop('SOFA numeric error or >5 mm substep jump')
            return d
        backend.physics_substep=observed
        for item in plan['full_rl_actions']:
            context.update({'rl_action':item['index'],'action':item['action']});before=backend.physics_substeps_executed;t=time.monotonic()
            o,reward,term,trunc,info=env.step(np.asarray(item['action'],np.float32))
            assert backend.physics_substeps_executed-before==22 and o.shape==(41,) and np.isfinite(o).all() and info['transition_valid'] and not term and not trunc
            report['contact_actions'].append({**context,'actual_substeps':22,'seconds':time.monotonic()-t,'observation':o.tolist(),'info':info});save(out/'report.json',report)
            print('CONTACT_REPLAY',kind,item['index'],'substeps',backend.physics_substeps_executed,'constraint',report['contact_status'],flush=True)
        assert backend.physics_substeps_executed==330;report['status']='PASS_CAPTURE_COMPLETE';report['penetration_status']='NOT_VERIFIED_PENDING_OFFLINE_GEOMETRY'
    except BaseException:report['errors'].append(traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        Sofa.Simulation.animate=animate
        report['last_phase']=phase;report['last_phase_physics_substeps']=backend.physics_substeps_executed;report['sofa_logs']=_log_messages(backend)
        if stream:stream.close()
        env.close();report['production_sources_unchanged']=all(sha(p)==h for p,h in protected.items());report['seconds']=time.monotonic()-started
        save(out/'report.json',report);print('SOFA_FINAL',kind,report['status'],flush=True)
    return 0 if report['status']=='PASS_CAPTURE_COMPLETE' else 2
if __name__=='__main__':raise SystemExit(main(sys.argv[1],sys.argv[2]))
