"""Read-only timings around the unchanged production SOFA scene and actions."""
import sys,json,time,traceback,hashlib
from pathlib import Path
from dataclasses import replace
from collections import defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent;TEST=HERE.parent;PYTHON=TEST.parents[1]
sys.path[:0]=[str(PYTHON),str(TEST)]
from mcr_sim.config import DEFAULT_CONFIG
from mcr_sim.rl_env import MagneticNavigationEnv
from mcr_sim.sofa_simulation import SofaBackend
from run_acceptance import jsonable,_log_messages
import Sofa
from SofaRuntime import Timer
from mcr_sim.mcr_mag_controller import MagController

def main(root):
    out=Path(root)/'single_profile_v2';out.mkdir(exist_ok=False)
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PYTHON/'mcr_sim').rglob('*.py')}
    cfg=replace(DEFAULT_CONFIG,vessel_root=Path(root)/'parallel22/test_catalog',vessel_seeds=(101,))
    report={'status':'FAIL','phases':[],'timer_records':[],'errors':[],'timing_units':'seconds; internal SOFA timer total_time is milliseconds','no_physics_change':True}
    phase='reset';totals=defaultdict(lambda:defaultdict(float));counts=defaultdict(lambda:defaultdict(int));raw=[]
    def wrap(cls,name,label):
        original=getattr(cls,name)
        def measured(*a,**kw):
            t=time.perf_counter()
            try:return original(*a,**kw)
            finally:totals[phase][label]+=time.perf_counter()-t;counts[phase][label]+=1
        setattr(cls,name,measured)
        return original
    wrap(SofaBackend,'state','state_read');physics=wrap(SofaBackend,'physics_substep','production_physics_substep')
    wrap(MagController,'onAnimateBeginEvent','magnetic_python_callback')
    wrap(MagneticNavigationEnv,'_observation','observation_build')
    animate=Sofa.Simulation.animate
    def measured_animate(*a,**kw):
        if phase=='timer':Timer.begin('PerfSubstep')
        t=time.perf_counter();c=time.process_time()
        try:return animate(*a,**kw)
        finally:
            dt=time.perf_counter()-t;cpu=time.process_time()-c
            totals[phase]['animate']+=dt;totals[phase]['animate_cpu']+=cpu;counts[phase]['animate']+=1
            if phase=='timer':
                t0=time.perf_counter()
                try:report['timer_records'].append(Timer.getRecords('PerfSubstep'))
                finally:Timer.end('PerfSubstep')
                totals[phase]['timer_extract_overhead']+=time.perf_counter()-t0
            raw.append({'phase':phase,'animate_seconds':dt,'animate_cpu_seconds':cpu})
    Sofa.Simulation.animate=measured_animate
    backend=SofaBackend(cfg);env=MagneticNavigationEnv(backend,config=cfg)
    try:
        t=time.perf_counter();o,info=env.reset(seed=101,options={'vessel_seed':101,'target_branch':'lcca'})
        report['reset_seconds']=time.perf_counter()-t
        # Preserve normal message checking, only measure its Python overhead.
        message_check=backend._messages.require_no_errors
        def measured_messages(*a,**kw):
            t=time.perf_counter()
            try:return message_check(*a,**kw)
            finally:totals[phase]['production_message_check']+=time.perf_counter()-t;counts[phase]['production_message_check']+=1
        backend._messages.require_no_errors=measured_messages
        models=[];topologies=[]
        def walk(node):
            for obj in node.objects:
                cls=obj.getClassName();name=obj.getName();path=obj.getPathName()
                if 'CollisionModel' in cls:
                    item={'class':cls,'name':name,'path':path,'data':{}}
                    for key in ['active','moving','simulated','proximity','bothSide','group']:
                        try:item['data'][key]=jsonable(obj.findData(key).value)
                        except Exception:pass
                    try:item['size']=int(obj.getSize())
                    except Exception:item['size']='NOT_OBSERVED'
                    models.append(item)
                if 'Topology' in cls or 'MeshSTLLoader' in cls:
                    item={'class':cls,'name':name,'path':path}
                    for key in ['triangles','edges','position']:
                        try:item[key+'_count']=len(obj.findData(key).value)
                        except Exception:pass
                    topologies.append(item)
            for child in node.children:walk(child)
        walk(backend._root);report['runtime_collision_models']=models;report['runtime_topologies']=topologies
        Timer.setEnabled('PerfSubstep',True);Timer.setInterval('PerfSubstep',1000000);Timer.setOutputType('PerfSubstep','stdout')
        for current in ['plain','timer','diagnostic']:
            phase=current
            if current=='diagnostic':
                def extra_diagnostics(self,field,delta):
                    t=time.perf_counter();self.state();dt=time.perf_counter()-t
                    result=physics(self,field,delta)
                    t=time.perf_counter();s=self.state();s.validate();np.all(np.isfinite(np.asarray(self._handles.dofs.velocity.value)));_log_messages(self)
                    totals[phase]['extra_diagnostics']+=dt+time.perf_counter()-t
                    return result
                SofaBackend.physics_substep=extra_diagnostics
            for j in range(2):
                t=time.perf_counter();n=backend.physics_substeps_executed
                action=np.array([0.,0.,.125 if current=='plain' and j==0 else 0.],dtype=np.float32)
                o,reward,terminated,truncated,info=env.step(action)
                assert o.shape==(41,) and np.isfinite(o).all() and not terminated and not truncated and info['transition_valid']
                assert backend.physics_substeps_executed-n==22
                assert not _log_messages(backend)['errors']
                report['phases'].append({'phase':phase,'rl_index':j,'wall_seconds':time.perf_counter()-t,'substeps':22,'observation':o.tolist(),'info':jsonable(info)})
                (out/'report.json').write_text(json.dumps(jsonable(report),indent=2)+'\n')
                print('PROFILE_ACTION_PASS',phase,j,'seconds',time.perf_counter()-t,flush=True)
        report['status']='PASS';report['actual_substeps']=backend.physics_substeps_executed
    except BaseException:report['errors'].append(traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        report['python_timings']=dict(totals);report['python_call_counts']=dict(counts);report['raw_animate_samples']=raw
        report['sofa_logs']=_log_messages(backend);env.close()
        report['production_sources_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
        (out/'report.json').write_text(json.dumps(jsonable(report),indent=2)+'\n')
    return 0 if report['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
