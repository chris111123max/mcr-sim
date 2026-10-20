"""Ten real production samplers after ten-asset geometry gate; no learner or optimizer."""
import sys,os,json,time,threading,traceback,re
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
HERE=Path(__file__).resolve().parent;TEST=HERE.parent;PYTHON=TEST.parents[1]
sys.path[:0]=[str(PYTHON),str(TEST)]
from mcr_sim.config import DEFAULT_CONFIG
from mcr_sim import sampling
from mcr_sim.recurrent_sac import EpisodeSequenceReplay
from run_acceptance import jsonable,_log_messages
PRODUCTION_WORKER=sampling._worker
SEEDS=(101,202,303,404,505,606,707,808,909,1010)
class PhysicalStop(BaseException):pass

def observed_production_worker(connection,config,mode):
    # Test-local post-call observation only: original worker, backend, integration and actions unchanged.
    from mcr_sim.sofa_simulation import SofaBackend
    instances=[];frames=[];init=SofaBackend.__init__;substep=SofaBackend.physics_substep
    def remember(self,*a,**kw):init(self,*a,**kw);instances.append(self)
    def observe(self,field,delta):
        before=self.state();count=self.physics_substeps_executed;t0=time.monotonic();d=substep(self,field,delta)
        s=self.state();s.validate();jump=float(np.linalg.norm(s.tip_m-before.tip_m))
        frames.append({'counter':self.physics_substeps_executed,'tip_m':s.tip_m.tolist(),'insertion_m':s.inserted_length_m,'field_t':s.magnetic_field_world_t.tolist(),'requested_insertion_delta_m':float(delta),'seconds':time.monotonic()-t0,'simulation_error':d.simulation_error,'dt_s':float(self._root.dt.value)})
        if self.physics_substeps_executed!=count+1 or d.simulation_error or jump>.005 or not np.all(np.isfinite(np.asarray(self._handles.dofs.velocity.value))):raise PhysicalStop('Invalid real physical substep or >5mm jump')
        logs=_log_messages(self)
        if logs['errors'] or any(re.search('no converg|not converg|singular|NaN|diverg',str(w),re.I) for w in logs['warnings']):raise PhysicalStop('SOFA solver error or serious warning')
        return d
    SofaBackend.__init__=remember;SofaBackend.physics_substep=observe
    class ReadOnlyConnection:
        def recv(self):return connection.recv()
        def close(self):return connection.close()
        def send(self,value):
            success,result=value
            if success and isinstance(result,tuple) and len(result) in (2,5) and isinstance(result[-1],dict):
                b=instances[-1];s=b.state();info=dict(result[-1]);info['acceptance_observed']={'pid':os.getpid(),'actual_counter':b.physics_substeps_executed,'tip_m':s.tip_m.tolist(),'inserted_length_m':s.inserted_length_m,'substeps':list(frames),'sofa_logs':_log_messages(b)};result=tuple(list(result[:-1])+[info]);frames.clear()
            elif not success:result=str(result)+'\nACTUAL_TEST_CONTEXT='+json.dumps({'pid':os.getpid(),'substeps':frames})
            connection.send((success,result))
    PRODUCTION_WORKER(ReadOnlyConnection(),config,mode)

def stat(pid):
    v=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'cpu_s':(int(v[11])+int(v[12]))/os.sysconf('SC_CLK_TCK'),'rss_mib':int(v[21])*os.sysconf('SC_PAGE_SIZE')/1048576}

def main(root):
    root=Path(root).resolve();out=root/'phase3_ten_parallel';out.mkdir(exist_ok=False)
    report={'status':'FAIL','rounds':[],'errors':[],'limits':{'workers':10,'rl_steps_per_worker':9,'substeps_per_full_step':22,'max_tip_jump_per_substep_m':.005},'training_started':False,'SAC_networks_initialized':False,'optimizer_calls':0,'observer_scope':'Test-only wrappers around original production worker and post-integration reads. No physics or control modification.'}
    envs=[];started=time.monotonic();sampling._worker=observed_production_worker
    try:
        manifest=json.loads((root/'ten_cohort_generation.json').read_text())
        if manifest['status']!='ALL_TEN_STAGE_A_PASS':raise RuntimeError('All ten assets must finish generation and Stage A BEFORE any SOFA environment')
        catalog=out/'test_catalog';catalog.mkdir()
        for seed in SEEDS:(catalog/('arch_type_I_seed_%d'%seed)).symlink_to(Path(manifest['assets'][str(seed)]['directory']),target_is_directory=True)
        cfg=replace(DEFAULT_CONFIG,vessel_root=catalog,vessel_seeds=SEEDS)
        mem_limit=int(Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text());mem_used=int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text())
        report['initial_resource_budget']={'memory_limit_bytes':mem_limit,'memory_used_bytes':mem_used,'cpu_affinity_count':len(os.sched_getaffinity(0))}
        if mem_limit-mem_used<20*1024**3:raise RuntimeError('Insufficient memory headroom for ten-scene test; no silent reduction of worker count')
        for _ in SEEDS:envs.append(sampling.ProcessEnvironment(cfg))
        pids=[e.process.pid for e in envs];assert len(set(pids))==10
        report['worker_pids']=pids
        def reset_one(i):return envs[i].reset(seed=SEEDS[i],options={'vessel_seed':SEEDS[i],'target_branch':'lcca'})
        t0=time.monotonic()
        with ThreadPoolExecutor(max_workers=10) as pool:initial=list(pool.map(reset_one,range(10)))
        report['parallel_reset_seconds']=time.monotonic()-t0;observations=[]
        for i,(o,info) in enumerate(initial):
            assert o.shape==(41,) and np.all(np.isfinite(o)) and info['acceptance_observed']['pid']==pids[i] and info['acceptance_observed']['actual_counter']==0
            observations.append(o)
        report['initial_resets']=[{'seed':SEEDS[i],'info':info,'observation':o.tolist()} for i,(o,info) in enumerate(initial)]
        replay=EpisodeSequenceReplay(cfg.sac.replay_buffer_size,cfg.sac.batch_size,sequence_length=8,observation_dim=41);reference_histories=[[] for _ in SEEDS];traces=[];dispatches=[];collections=[]
        for i,e in enumerate(envs):
            original_async=e.step_async;original_wait=e.step_wait
            def send(a,i=i,fn=original_async):dispatches.append({'env_id':i,'time':time.monotonic()});return fn(a)
            def receive(i=i,fn=original_wait):r=fn();collections.append({'env_id':i,'time':time.monotonic()});return r
            e.step_async=send;e.step_wait=receive
        for rnd in range(1,10):
            actions=[np.array([0.,0.,.125 if rnd==1 else 0.],np.float32) for _ in envs]
            samples=[];done=threading.Event();before=[stat(p) for p in pids]
            def monitor():
                while not done.is_set():samples.append({'time':time.monotonic(),'workers':[stat(p) for p in pids]});done.wait(.1)
            thread=threading.Thread(target=monitor,daemon=True);thread.start();t0=time.monotonic()
            try:results=sampling.step_environments(envs,actions)
            finally:done.set();thread.join(timeout=2)
            elapsed=time.monotonic()-t0;after=[stat(p) for p in pids]
            assert max(x['time'] for x in dispatches[-10:])<min(x['time'] for x in collections[-10:])
            concurrency=[sum(b['workers'][j]['cpu_s']>a['workers'][j]['cpu_s'] for j in range(10)) for a,b in zip(samples,samples[1:])]
            if max(concurrency,default=0)!=10:raise RuntimeError('No resource sample corroborates all10 workers CPU-active; do not claim ten-way execution')
            rr={'round':rnd,'wall_seconds':elapsed,'rl_steps_per_second':10/elapsed,'dispatch_to_collection_seconds':[next(x['time'] for x in collections[-10:] if x['env_id']==i)-next(x['time'] for x in dispatches[-10:] if x['env_id']==i) for i in range(10)],'cpu_percent_one_core_100':[100*(b['cpu_s']-a['cpu_s'])/elapsed for a,b in zip(before,after)],'peak_rss_mib':[max(x['workers'][i]['rss_mib'] for x in samples) for i in range(10)],'intervals_all_ten_cpu_active':concurrency.count(10),'results':[]}
            for i,(o,reward,term,trunc,info) in enumerate(results):
                observed=info['acceptance_observed'];frames=observed['substeps']
                assert info['vessel_seed']==SEEDS[i] and observed['pid']==pids[i] and len(frames)==22 and all(s['dt_s']==.006 and not s['simulation_error'] for s in frames)
                assert observed['actual_counter']==22*(rnd-1 if i==0 and rnd>1 else rnd)
                assert info['transition_valid'] and info['observation_valid'] and info['physics_substeps_attempted']==22 and o.shape==(41,) and np.all(np.isfinite(o)) and not term and not trunc and not observed['sofa_logs']['errors']
                assert replay.add(observations[i],info['executed_action'],reward,o,term,trunc,env_id=i,valid=True)
                reference_histories[i]=(reference_histories[i]+[observations[i].copy()])[-8:]
                assert np.array_equal(replay.data[-1].observation_history,np.stack(reference_histories[i]))
                assert np.array_equal(replay.data[-1].next_observation_history,np.concatenate((np.stack(reference_histories[i]),o[None]),axis=0)[-8:])
                if rnd>=2 and i>0:assert abs(float(o[29]-observations[i][29]))<1e-8 and o[28]<observations[i][28]
                observations[i]=o;rr['results'].append({'env_id':i,'seed':SEEDS[i],'observation':o.tolist(),'reward':reward,'info':info})
            report['rounds'].append(rr);traces.extend(samples);print('TEN_PARALLEL_ROUND',rnd,'seconds',elapsed,'RL_steps_per_s',10/elapsed,flush=True)
            if rnd==1:
                peers={i:replay._histories[i][0].copy() for i in range(1,10)};cpu0=[stat(p) for p in pids[1:]];replay.end_episode(0);reference_histories[0]=[]
                o,info=reset_one(0);observations[0]=o
                assert info['acceptance_observed']['actual_counter']==0 and not info['acceptance_observed']['substeps']
                assert all(np.array_equal(peers[i],replay._histories[i][0]) for i in peers)
                report['peer_cpu_during_env0_reset_s']=[stat(p)['cpu_s']-s['cpu_s'] for p,s in zip(pids[1:],cpu0)]
        records=list(replay.data);assert len(records)==90
        for i,r in enumerate(records):
            replay._sequence(i,records);assert r.env_id==i%10 and r.episode==(1 if i>=10 and i%10==0 else 0) and len(r.observation_history)==min(8,i//10 if i>=10 and i%10==0 else i//10+1)
        batch=replay.sample('cpu');assert tuple(batch['observation'].shape)==(32,8,41)
        report['replay_records']=[{'env_id':r.env_id,'episode':r.episode,'history_frames':len(r.observation_history),'next_frames':len(r.next_observation_history)} for r in records]
        report['replay_sample_shape']=[32,8,41];report['history_reset_and_continuity']='PASS';report['actual_observed_physics_substeps']=sum(len(r['info']['acceptance_observed']['substeps']) for q in report['rounds'] for r in q['results']);assert report['actual_observed_physics_substeps']==1980
        report['aggregate_rl_steps_per_second']=90/sum(r['wall_seconds'] for r in report['rounds']);report['status']='PASS';(out/'resource_samples.json').write_text(json.dumps(traces,indent=2)+'\n')
    except BaseException:report['errors'].append(traceback.format_exc())
    finally:
        for e in envs:e.close()
        report['worker_exit_codes']=[e.process.exitcode for e in envs];report['seconds_including_resets']=time.monotonic()-started
        (out/'report.json').write_text(json.dumps(jsonable(report),indent=2,allow_nan=False)+'\n');print('TEN_PARALLEL_FINAL',report['status'],flush=True)
    return 0 if report['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
