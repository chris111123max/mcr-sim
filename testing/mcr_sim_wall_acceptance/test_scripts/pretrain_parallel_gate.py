"""Two real production samplers, four real transitions, no SAC networks/optimizer."""
import sys,os,json,time,threading,traceback
from pathlib import Path
from dataclasses import replace
import numpy as np
HERE=Path(__file__).resolve().parent;TEST=HERE.parent;PYTHON=TEST.parents[1]
sys.path[:0]=[str(PYTHON),str(TEST)]
from mcr_sim.config import DEFAULT_CONFIG
from mcr_sim.sampling import ProcessEnvironment,step_environments
from mcr_sim.recurrent_sac import EpisodeSequenceReplay
from run_acceptance import jsonable

def stat(pid):
    v=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'cpu_s':(int(v[11])+int(v[12]))/os.sysconf('SC_CLK_TCK'),'rss_mib':int(v[21])*os.sysconf('SC_PAGE_SIZE')/1048576}

def main(report_root):
    root=Path(report_root).resolve();out=root/'phase3_parallel';out.mkdir(exist_ok=False)
    report={'status':'FAIL','rounds':[],'errors':[],'limits':{'workers':2,'rl_actions_per_worker':2,'reported_physics_calls_per_action':22},'training_started':False,'SAC_networks_initialized':False,'optimizer_calls':0}
    envs=[];started=time.monotonic()
    try:
        for seed in (202,303):
            if json.loads((root/('seed%d_stage_a.json'%seed)).read_text())['status']!='PASS':raise RuntimeError('Cross-vessel Stage A prerequisite failed')
            if json.loads((root/('seed%d_physics'%seed)/'report.json').read_text())['status']!='PASS':raise RuntimeError('Cross-vessel SOFA prerequisite failed')
        catalog=out/'test_catalog';catalog.mkdir()
        src101=PYTHON/'stEVE_anatomical_vessels/tmp_seed101_geometry_r3TeZG/refined_03/arch_type_I_seed_101'
        src202=PYTHON/'stEVE_anatomical_vessels/pretrain_20261020_d085_vox03/arch_type_I_seed_202'
        for s,p in ((101,src101),(202,src202)):(catalog/('arch_type_I_seed_%d'%s)).symlink_to(p,target_is_directory=True)
        cfg=replace(DEFAULT_CONFIG,vessel_root=catalog,vessel_seeds=(101,202))
        for _ in range(2):envs.append(ProcessEnvironment(cfg))
        pids=[e.process.pid for e in envs];assert len(set(pids))==2
        report['worker_pids']=pids;report['workers_spawn_isolated']=True
        observations=[]
        for e,seed in zip(envs,(101,202)):
            o,info=e.reset(seed=seed,options={'vessel_seed':seed,'target_branch':'lcca'});assert o.shape==(41,) and np.all(np.isfinite(o));observations.append(o)
        replay=EpisodeSequenceReplay(16,4,sequence_length=8,observation_dim=41)
        traces=[];dispatches=[];collections=[]
        for i,e in enumerate(envs):
            original_async=e.step_async;original_wait=e.step_wait
            def send(a,i=i,fn=original_async):
                dispatches.append({'env_id':i,'time':time.monotonic()});return fn(a)
            def receive(i=i,fn=original_wait):
                r=fn();collections.append({'env_id':i,'time':time.monotonic()});return r
            e.step_async=send;e.step_wait=receive
        for rnd in (1,2):
            actions=[np.array([0.,0.,.125 if rnd==1 else 0.],np.float32) for _ in envs]
            samples=[];done=threading.Event();before=[stat(p) for p in pids]
            def monitor():
                while not done.is_set():
                    samples.append({'time':time.monotonic(),'workers':[stat(p) for p in pids]});done.wait(.1)
            monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start();t0=time.monotonic()
            try:results=step_environments(envs,actions)
            finally:done.set();monitor_thread.join(timeout=2)
            elapsed=time.monotonic()-t0;after=[stat(p) for p in pids]
            assert max(x['time'] for x in dispatches[-2:])<min(x['time'] for x in collections[-2:])
            overlap=sum(all(b['workers'][j]['cpu_s']>a['workers'][j]['cpu_s'] for j in range(2)) for a,b in zip(samples,samples[1:]))
            assert overlap>0,'No sampled concurrent CPU activity; do not claim parallel execution'
            round_record={'round':rnd,'seconds_for_two_steps':elapsed,'aggregate_rl_steps_per_second':2/elapsed,'per_environment_dispatch_to_collection_s':[next(x['time'] for x in collections[-2:] if x['env_id']==i)-next(x['time'] for x in dispatches[-2:] if x['env_id']==i) for i in range(2)],'cpu_seconds':[b['cpu_s']-a['cpu_s'] for a,b in zip(before,after)],'cpu_percent_one_core_100':[100*(b['cpu_s']-a['cpu_s'])/elapsed for a,b in zip(before,after)],'peak_rss_mib_per_worker':[max(x['workers'][i]['rss_mib'] for x in samples) for i in range(2)],'shared_sampling_intervals_with_both_cpu_active':overlap,'results':[]}
            for i,(o,reward,term,trunc,info) in enumerate(results):
                assert info['vessel_seed']==(101,202)[i] and info['transition_valid'] and info['observation_valid'] and info['physics_substeps_attempted']==22 and o.shape==(41,) and np.all(np.isfinite(o)) and not term and not trunc
                assert replay.add(observations[i],info['executed_action'],reward,o,term,trunc,env_id=i,valid=info['transition_valid'])
                if rnd==2 and i==1:
                    assert abs(float(o[29]-observations[i][29]))<1e-8,'Other worker insertion changed on reset'
                    assert o[28]<observations[i][28],'Other worker episode time reset'
                observations[i]=o;round_record['results'].append({'env_id':i,'observation':o.tolist(),'reward':reward,'info':info})
            report['rounds'].append(round_record);traces.extend(samples)
            print('PARALLEL_ROUND',rnd,'seconds',elapsed,'RL_steps_per_s',2/elapsed,'overlap_intervals',overlap,flush=True)
            if rnd==1:
                history_b=replay._histories[1][0].copy();cpu_b0=stat(pids[1]);replay.end_episode(0)
                o,info=envs[0].reset(seed=101,options={'vessel_seed':101,'target_branch':'lcca'});cpu_b1=stat(pids[1]);observations[0]=o
                assert np.array_equal(history_b,replay._histories[1][0]);report['worker_B_cpu_during_A_reset_s']=cpu_b1['cpu_s']-cpu_b0['cpu_s']
        records=list(replay.data);assert [(r.env_id,r.episode,len(r.observation_history)) for r in records]==[(0,0,1),(1,0,1),(0,1,1),(1,0,2)]
        for i,r in enumerate(records):replay._sequence(i,records)
        batch=replay.sample('cpu');assert tuple(batch['observation'].shape)==(4,8,41)
        report['replay_records']=[{'env_id':r.env_id,'episode':r.episode,'history_frames':len(r.observation_history),'next_history_frames':len(r.next_observation_history)} for r in records]
        report['replay_sample_shape']=[4,8,41];report['history_reset_and_continuity']='PASS'
        total=sum(r['seconds_for_two_steps'] for r in report['rounds']);report['aggregate_rl_steps_per_second']=4/total
        report['total_reported_physics_substeps']=88;report['independent_cross_process_animate_counter']='NOT_OBSERVED; unmodified production worker reports physics_substeps_attempted'
        report['status']='PASS';(out/'resource_samples.json').write_text(json.dumps(traces,indent=2)+'\n')
    except BaseException:report['errors'].append(traceback.format_exc())
    finally:
        for e in envs:e.close()
        report['worker_exit_codes']=[e.process.exitcode for e in envs];report['seconds_including_resets']=time.monotonic()-started
        (out/'report.json').write_text(json.dumps(jsonable(report),indent=2,allow_nan=False)+'\n');print('PARALLEL_FINAL',report['status'],flush=True)
    return 0 if report['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
