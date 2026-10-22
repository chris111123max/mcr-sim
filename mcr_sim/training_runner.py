"""Valid-transition scheduling and bounded, per-worker solver failure isolation."""
import json,random,time,traceback,os
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .serialization import write_json,json_default
from .faults import SamplingStop,valid_transition,result_info,isolated_lcp_failure,diagnostic_value
from .sampling import collect_step_outcomes,reset_environments
from .memory_guard import record as memory_record,safe as memory_safe


def train(train_env,eval_env,config,output_dir,total_steps=None,device='cpu',sequence_length=8,
          seed=0,resume=None,profile=None,log_interval_updates=20,strict_failures=False):
    from .recurrent_sac import EpisodeSequenceReplay,RecurrentSAC,evaluate_policy
    from .resume_state import restore,finite_tree
    total_steps=int(total_steps or config.training_steps_script_default)
    if total_steps<1 or log_interval_updates<1:raise ValueError('Invalid transition budget')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'config.json').exists() or (output/'metrics.jsonl').exists():
        raise FileExistsError('Training requires a new output directory, including on resume')
    envs=list(train_env) if isinstance(train_env,(tuple,list)) else [train_env]
    if not envs:raise ValueError('At least one training environment is required')
    write_json(output/'config.json',dict(sac=asdict(config),seed=seed,num_envs=len(envs),sequence_length=sequence_length))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    agent=RecurrentSAC(config,device=device,sequence_length=sequence_length)
    if agent.device.type=='npu':torch.npu.manual_seed(seed)
    replay=EpisodeSequenceReplay(config.replay_buffer_size,config.batch_size,sequence_length)
    agent.profile=replay.profile=profile
    step=0;updates=0
    if resume:step,updates=restore(resume,agent,replay,output)
    if not 0<=step<total_steps:raise ValueError('Resume step must be below target')
    start=step;new_updates=0
    state=getattr(agent,'restored_training_state',{})
    attempted_base=state.get('attempted_transitions',0 if not resume else None)
    invalid_base=state.get('invalid_transitions',0 if not resume else None)
    attempted_run=invalid_run=0
    fault_total=int(state.get('tracked_fault_total',0))
    consecutive=[int(state.get('consecutive_lcp_failures',{}).get(str(i),0)) for i in range(len(envs))]
    next_save=(step//50000+1)*50000
    next_eval=int(state.get('next_evaluation_step',(step//250000+1)*250000))
    if next_eval<step:raise ValueError('Checkpoint evaluation schedule is inconsistent')
    consistent=True;last_saved=str(Path(resume).resolve()) if resume else None
    owned_saves=deque();snapshot_serial=0;last_signature=None
    log_path=output/'metrics.jsonl';pending=[];losses={};recent=deque(maxlen=1000)
    reset_count=0;round_cursor=0;started=time.monotonic();evaluation_seconds=0.
    histories=[None]*len(envs);observations=[None]*len(envs)
    episode_returns=np.zeros(len(envs));episode_steps=np.zeros(len(envs),dtype=int)
    reset_info=[{} for _ in envs]
    def region(name):return profile.region(name) if profile else nullcontext()
    def log(item):
        with log_path.open('a') as stream:
            stream.write(json.dumps(diagnostic_value(item),default=json_default,allow_nan=False)+'\n')
    def counters():
        return dict(valid_transitions=step,global_step=step,
                    attempted_transitions=None if attempted_base is None else attempted_base+attempted_run,
                    invalid_transitions=None if invalid_base is None else invalid_base+invalid_run,
                    attempted_transitions_this_run=attempted_run,invalid_transitions_this_run=invalid_run,
                    new_transitions=step-start,updates=updates,new_updates=new_updates,
                    tracked_fault_total=fault_total,
                    fault_rate_this_run=invalid_run/max(attempted_run,1))
    def flush_metrics():
        for at,metric in pending:
            vals=torch.stack(tuple(metric.values())).cpu().tolist()
            if not np.isfinite(vals).all():raise FloatingPointError('Nonfinite SAC metric')
            losses.update(dict(zip(metric,map(float,vals))))
            log(dict(step=at,update=dict(losses)))
        pending.clear()
    def save(kind='periodic'):
        nonlocal snapshot_serial,last_saved,last_signature
        if not consistent:raise RuntimeError('Refuse snapshot during incomplete optimizer update')
        signature=(step,updates,new_updates,attempted_run,invalid_run,fault_total,next_save,next_eval,tuple(consecutive))
        if signature==last_signature:return Path(last_saved)
        flush_metrics()
        # Check learner/optimizer finite state only at snapshot boundaries.
        cp=agent.checkpoint();finite_tree(cp)
        training_state=dict(**counters(),next_checkpoint_step=next_save,next_evaluation_step=next_eval,
                            consecutive_lcp_failures={str(i):v for i,v in enumerate(consecutive)},
                            fault_counter_scope='tracked since engineering resume; older faults may be unknown')
        cp.update(step=step,updates=updates,replay_format_version=replay.FORMAT_VERSION,
                  replay_capacity=replay.data.maxlen,replay_records=len(replay),replay=list(replay.data),
                  replay_episodes=dict(replay.episodes),training_state=training_state,
                  resume_source=getattr(agent,'resume_metadata',None),physical_episodes_resumed=False,
                  checkpoint_kind=kind,complete_training_state=True,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state())
        if agent.device.type=='npu':cp['npu_rng']=torch.npu.get_rng_state(agent.device)
        folder=output/'checkpoints';folder.mkdir(exist_ok=True)
        snapshot_serial+=1
        suffix='' if kind=='periodic' else '_%s_%04d'%(kind,snapshot_serial)
        destination=folder/('lstm_sac_%d_steps%s.pt'%(step,suffix))
        if destination.exists():raise FileExistsError(destination)
        temporary=folder/(destination.name+'.pending')
        memory_record(output/'memory.jsonl','checkpoint_before',step,envs,replay_records=len(replay),kind=kind)
        try:
            with temporary.open('xb') as stream:
                torch.save(cp,stream);stream.flush();os.fsync(stream.fileno())
            temporary.replace(destination)
        except BaseException:
            save_error=traceback.format_exc()
            try:write_json(output/('checkpoint_save_failure_%04d.json'%snapshot_serial),dict(step=step,kind=kind,traceback=save_error,previous_complete=last_saved))
            except Exception as log_error:print('Checkpoint failure log unavailable: '+repr(log_error),flush=True)
            raise
        finally:del cp
        last_saved=str(destination);last_signature=signature;owned_saves.append(destination)
        memory_record(output/'memory.jsonl','checkpoint_after',step,envs,checkpoint_bytes=destination.stat().st_size)
        # Only files successfully written by this invocation are eligible.
        while len(owned_saves)>3:
            old=owned_saves.popleft()
            try:old.unlink()
            except OSError as exc:log(dict(event='checkpoint_retention_error',path=str(old),error=repr(exc)))
        log(dict(event='checkpoint_saved',step=step,path=str(destination),kind=kind))
        return destination
    def accept_reset(i,result):
        nonlocal reset_count
        obs,info=result
        if np.asarray(obs).shape!=(41,) or not np.isfinite(obs).all():raise ValueError('Invalid reset observation')
        if strict_failures and info.get('physics_runtime_audit',{}).get('status')!='PASS':
            raise ValueError('Production physics runtime audit failed')
        observations[i]=obs;reset_info[i]=info;histories[i]=None
        episode_steps[i]=0;episode_returns[i]=0.;reset_count+=1
    def scheduled_evaluation():
        nonlocal next_eval,evaluation_seconds
        if eval_env is None:raise ValueError('Scheduled evaluation requires an evaluator')
        log(dict(event='evaluation_start',step=step,episodes=10,workers=2))
        t=time.monotonic()
        try:summary=evaluate_policy(agent,eval_env,sequence_length=sequence_length,training_step=step)
        except Exception as exc:raise SamplingStop('Evaluation failed; see fixed-plan failure report') from exc
        evaluation_seconds+=time.monotonic()-t
        next_eval+=250000
        log(dict(step=step,evaluation=summary))
    write_json(output/'restore_complete.json',dict(step=step,updates=updates,replay_records=len(replay),replay_capacity=replay.data.maxlen,num_envs=len(envs),target=total_steps))
    print('TRAIN_RESTORE',json.dumps(dict(step=step,updates=updates,replay_records=len(replay),replay_capacity=replay.data.maxlen,target=total_steps)),flush=True)
    try:
        for i,e in enumerate(envs):e.action_space.seed(seed+i)
        initial=reset_environments(envs,[dict(seed=seed+i+step) for i in range(len(envs))])
        for i,result in enumerate(initial):accept_reset(i,result)
        write_json(output/'runtime_physics.json',dict(workers=[x[1].get('physics_runtime_audit') for x in initial],count=len(envs)))
        if profile:profile.training_start()
        last_memory=last_progress=0.;last_progress_step=step
        while step<total_steps:
            if next_eval==step:scheduled_evaluation()
            now=time.monotonic()
            if now-last_memory>=30 or (output/'STOP').exists():
                mem=memory_record(output/'memory.jsonl','training',step,envs,reset_count=reset_count,
                                  worker_reset_counts=[e.reset_count for e in envs],worker_recycles=[e.recycles for e in envs])
                write_json(output/'process.json',dict(pid=os.getpid(),status='RUNNING',step=step,worker_pids=[e.process.pid for e in envs]))
                last_memory=now
                if not memory_safe(mem) or (output/'STOP').exists():
                    save('safe_stop');write_json(output/'safe_stop.json',dict(step=step,memory=mem,reason='STOP or reserve below 24 GiB'))
                    return agent
            # No in-flight future sample can cross a valid update/save/eval boundary.
            budget=min(len(envs),total_steps-step,20-step%20,next_save-step,next_eval-step)
            indices=[(round_cursor+j)%len(envs) for j in range(budget)]
            round_cursor=(round_cursor+budget)%len(envs)
            with region('actor_batch'):
                policy,following_history=agent.act_batch(np.stack([observations[i] for i in indices]),[histories[i] for i in indices])
            if policy.shape!=(len(indices),3) or not np.isfinite(policy).all():
                raise FloatingPointError('Nonfinite/invalid actor batch')
            actions=[envs[i].action_space.sample() if step+j+1<=config.heatup_steps else policy[j] for j,i in enumerate(indices)]
            outcomes=collect_step_outcomes([envs[i] for i in indices],actions,indices)
            reset_ids=[];failed_ids=[];fatal=[]
            for j,outcome in enumerate(outcomes):
                i=outcome.env_id;attempted_run+=1
                valid=outcome.error is None and valid_transition(outcome.result)
                if not valid:
                    invalid_run+=1;fault_total+=1
                    info=result_info(outcome.result)
                    log(dict(event='simulation_failure',step=step,env_id=i,worker_pid=outcome.worker_pid,
                             worker_exitcode=outcome.exitcode,worker_log=outcome.worker_log,
                             vessel_seed=info.get('vessel_seed',reset_info[i].get('vessel_seed')),
                             target_branch=info.get('target_branch',reset_info[i].get('target_branch')),
                             command_action=actions[j],failure_info=diagnostic_value(info),transport_error=repr(outcome.error),
                             validation_failure='invalid transition' if outcome.error is None else 'collector failure',
                             valid_episode_prefix_steps=int(episode_steps[i]),valid_episode_prefix_reward=float(episode_returns[i]),
                             counts=counters()))
                    replay.end_episode(i);histories[i]=None;observations[i]=None
                    episode_returns[i]=0.;episode_steps[i]=0
                    if outcome.error is None and isolated_lcp_failure(info):
                        consecutive[i]+=1;failed_ids.append(i)
                        if consecutive[i]>=2:fatal.append('env %d: consecutive numerical-failure episodes'%i)
                    else:fatal.append('env %d: unclassified numerical/transport/source fault'%i)
                    continue
                obs,reward,terminated,truncated,info=outcome.result
                replay.add(observations[i],actions[j],reward,obs,terminated,truncated,i)
                step+=1;observations[i]=obs;histories[i]=following_history[j]
                episode_returns[i]+=reward;episode_steps[i]+=1
                if terminated or truncated:
                    consecutive[i]=0
                    log(dict(event='navigation_episode',step=step,env_id=i,success=bool(info['success']),
                             reward=float(episode_returns[i]),steps=int(episode_steps[i]),termination_reason=info['termination_reason'],
                             vessel_seed=info['vessel_seed'],target_branch=info['target_branch'],episode=info.get('episode_metrics')))
                    recent.append((bool(info['success']),float(episode_returns[i]),int(episode_steps[i])))
                    reset_ids.append(i);histories[i]=None
                if step>config.heatup_steps and step%20==0 and len(replay)>=config.batch_size:
                    consistent=False
                    metrics=agent.update(replay.sample(agent.device),return_tensors=True)
                    if agent.device.type=='npu':torch.npu.synchronize()
                    consistent=True;new_updates+=1
                    if updates is not None:updates+=1
                    pending.append((step,metrics))
                    if len(pending)>=log_interval_updates:flush_metrics()
            # Every peer has been received and all valid results committed first.
            periodic_due=step==next_save
            if periodic_due:next_save+=50000
            if failed_ids or fatal:save('fault')
            if fatal:raise SamplingStop('; '.join(fatal))
            if periodic_due and not failed_ids:save('periodic')
            if step==next_eval:scheduled_evaluation()
            for i in failed_ids:
                try:
                    old_pid=envs[i].process.pid;cleanup=envs[i].rebuild()
                    accept_reset(i,envs[i].reset())
                    log(dict(event='worker_rebuilt',step=step,env_id=i,old_pid=old_pid,new_pid=envs[i].process.pid,
                             cleanup=cleanup,consecutive_fault_episodes=consecutive[i],
                             new_vessel_seed=reset_info[i].get('vessel_seed'),
                             new_target_branch=reset_info[i].get('target_branch'),
                             failed_episode_replayed=False,counts=counters()))
                except Exception as exc:raise SamplingStop('Single worker rebuild/reset failed; no further retry') from exc
            if reset_ids:
                try:
                    reset_results=reset_environments([envs[i] for i in reset_ids])
                    for i,result in zip(reset_ids,reset_results):accept_reset(i,result)
                except Exception as exc:raise SamplingStop('Normal episode reset failed') from exc
            if step-last_progress_step>=1024 or step==start+20 or step==total_steps:
                flush_metrics();now=time.monotonic()
                progress=dict(event='progress',step=step,**counters(),heatup=step<=config.heatup_steps,
                              num_envs=len(envs),replay_records=len(replay),replay_capacity=replay.data.maxlen,
                              elapsed_s=now-started,overall_fps=(step-start)/max(now-started,1e-9),
                              training_fps_excluding_evaluation=(step-start)/max(now-started-evaluation_seconds,1e-9),
                              recent_completed_episodes=len(recent),success_rate=float(np.mean([x[0] for x in recent])) if recent else None,
                              mean_episode_reward=float(np.mean([x[1] for x in recent])) if recent else None,losses=losses)
                log(progress);write_json(output/'status.json',progress)
                print('TRAIN_PROGRESS',json.dumps(progress,default=json_default),flush=True)
                last_progress_step=step
            if profile:profile.round_finished(step,new_updates,agent,replay)
        flush_metrics()
        if last_saved is None or not last_saved.endswith('lstm_sac_%d_steps.pt'%step):save('final')
        if profile:profile.training_end(step,new_updates,agent,replay)
        return agent
    except BaseException as exc:
        original=traceback.format_exc()
        if isinstance(exc,SamplingStop) and consistent:
            try:save('safe_stop_fault')
            except BaseException:
                save_error=traceback.format_exc()
                try:write_json(output/'fault_snapshot_failure.json',dict(original_exception=original,save_exception=save_error,last_complete=last_saved))
                except Exception as log_error:print('Fault snapshot log unavailable: '+repr(log_error),flush=True)
        # Unknown code/NPU/partial-update failures never claim a resumable new snapshot.
        try:write_json(output/'runner_failure.json',dict(step=step,counts=counters(),exception=original,
                       consistent_optimizer_boundary=consistent,last_complete_checkpoint=last_saved))
        except Exception as log_error:print('Runner failure log unavailable: '+repr(log_error),flush=True)
        raise
