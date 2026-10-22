"""Two isolated workers, each running five fixed held-out episodes sequentially."""
from collections import Counter
from pathlib import Path
import time,traceback
import numpy as np
from .sampling import ProcessEnvironment,reset_environments,collect_step_outcomes,close_environments
from .faults import valid_transition,result_info,diagnostic_value
from .serialization import write_json
from .memory_guard import record,safe

class ParallelEvaluation:
    def __init__(self,config,output_dir):
        if tuple(config.eval_vessel_seeds)!=(1111,1212):raise ValueError('Expected held-out 1111/1212')
        if config.task.max_episode_steps_eval!=150:raise ValueError('Evaluation requires 150 steps')
        self.config=config;self.output=Path(output_dir)
        self.plan=[dict(worker_id=w,env_id=w*16+i,vessel_seed=s,episode_seed=26010000+w*16+i) for w,s in enumerate((1111,1212)) for i in range(5)]
        write_json(self.output/'fixed_episode_plan.json',self.plan)

    def evaluate(self,agent,step,sequence_length=8):
        if sequence_length!=agent.sequence_length:raise ValueError('Mismatched history')
        start=time.monotonic();envs=[];results=[];was_training=agent.networks.training
        folder=self.output/('step_%09d'%step);folder.mkdir(parents=True,exist_ok=False)
        mempath=folder/'memory.jsonl';before=record(mempath,'evaluation_before',step)
        summary=None;simulation_failures=[]
        try:
            for _ in range(2):
                if not safe(record(mempath,'before_worker_creation',step,envs)):
                    raise MemoryError('Insufficient memory reserve for evaluation worker')
                envs.append(ProcessEnvironment(self.config,mode='eval',log_dir=folder/'workers'))
            assert len(set(e.process.pid for e in envs))==2
            agent.networks.eval()
            for episode in range(5):
                plans=[self.plan[w*5+episode] for w in range(2)]
                initial=reset_environments(envs,[dict(seed=p['episode_seed'],options={'vessel_seed':p['vessel_seed']}) for p in plans])
                observations=[x[0] for x in initial];histories=[None,None]
                traces=[[],[]];active=[0,1]
                for i,(_,info) in enumerate(initial):
                    if info['vessel_seed']!=plans[i]['vessel_seed'] or info['physics_runtime_audit']['status']!='PASS':
                        raise RuntimeError('Invalid evaluation reset')
                while active:
                    if max(map(len,traces))%20==0 and not safe(record(mempath,'evaluation_step',step,envs,episode=episode,rl_step=max(map(len,traces)))):
                        raise MemoryError('Evaluation memory reserve below 24 GiB')
                    actions,hs=agent.act_batch(np.stack([observations[i] for i in active]),[histories[i] for i in active],deterministic=True)
                    if actions.shape!=(len(active),3) or not np.isfinite(actions).all():raise ValueError('Invalid action')
                    batch=collect_step_outcomes([envs[i] for i in active],actions,active);following=[]
                    batch_failures=[]
                    for j,outcome in enumerate(batch):
                        i=outcome.env_id
                        if outcome.error is not None or not valid_transition(outcome.result):
                            failure=dict(**plans[i],status='SIMULATION_FAILURE',worker_pid=outcome.worker_pid,
                                         worker_exitcode=outcome.exitcode,worker_log=outcome.worker_log,
                                         target_branch=initial[i][1]['target_branch'],command_action=actions[j],
                                         valid_steps=len(traces[i]),transport_error=repr(outcome.error),
                                         info=diagnostic_value(result_info(outcome.result)))
                            simulation_failures.append(failure);batch_failures.append(failure)
                            write_json(folder/('failed_episode_%02d.json'%plans[i]['env_id']),dict(failure=failure,valid_prefix=traces[i]))
                            continue
                        obs,reward,term,trunc,info=outcome.result
                        traces[i].append(dict(reward=reward,**info));observations[i]=obs;histories[i]=hs[j]
                        if term or trunc:
                            t=traces[i];p0=initial[i][1]['path_length_mm']
                            ins=np.array([x['executed_insertion_m'] for x in t])
                            signs=np.where(ins>1e-5,1,np.where(ins< -1e-5,-1,0));nz=signs[signs!=0]
                            distances=[initial[i][1]['distance_to_target_m']]+[x['distance_to_target_m'] for x in t]
                            row=dict(**plans[i],worker_pid=envs[i].process.pid,steps=len(t),success=bool(info['success']),reward=float(sum(x['reward'] for x in t)),termination_reason=info['termination_reason'],target_branch=initial[i][1]['target_branch'],target_m=initial[i][1]['target_m'],initial_path_length_mm=p0,final_path_length_mm=info['path_length_mm'],final_bfs_completion_pct=100*(1-info['path_length_mm']/p0),min_distance_mm=1000*min(distances),advance_steps=int(sum(signs==1)),retract_steps=int(sum(signs==-1)),axial_stationary_steps=int(sum(signs==0)),direction_switches=int(sum(nz[1:]!=nz[:-1])))
                            results.append(row);write_json(folder/('episode_%02d.json'%plans[i]['env_id']),dict(result=row,trace=t))
                        else:
                            if len(traces[i])>=150:raise RuntimeError('Missing episode truncation')
                            following.append(i)
                    if batch_failures:raise RuntimeError("Evaluation task failed; no replacement tasks scheduled")
                    active=following
                record(mempath,'episode_pair_complete',step,envs,completed=len(results))
            if len(results)!=10:raise RuntimeError('Incomplete evaluation')
            def aggregate(rows):
                return dict(episodes=len(rows),successes=sum(x['success'] for x in rows),success_rate=float(np.mean([x['success'] for x in rows])),mean_reward=float(np.mean([x['reward'] for x in rows])),mean_final_bfs_completion_pct=float(np.mean([x['final_bfs_completion_pct'] for x in rows])),mean_min_distance_mm=float(np.mean([x['min_distance_mm'] for x in rows])),best_min_distance_mm=min(x['min_distance_mm'] for x in rows),failure_reasons=dict(Counter(x['termination_reason'] for x in rows if not x['success'])))
            summary=dict(training_step=step,workers=2,**aggregate(results),per_vessel={str(s):aggregate([r for r in results if r['vessel_seed']==s]) for s in (1111,1212)},results=sorted(results,key=lambda r:r['env_id']),deterministic=True,replay_writes=0,optimizer_updates=0)
        except BaseException:
            known={r['env_id']:dict(r,status='COMPLETE') for r in results}
            known.update({r['env_id']:r for r in simulation_failures})
            planned_results=[known.get(p['env_id'],dict(p,status='NOT_COMPLETED')) for p in self.plan]
            write_json(folder/'failure.json',dict(status='FAILED',training_step=step,
                       traceback=traceback.format_exc(),planned_episodes=10,completed_episodes=results,
                       simulation_failures=simulation_failures,planned_results=planned_results,
                       success_rate_planned=sum(r['success'] for r in results)/10,
                       incomplete=True,memory=record(mempath,'evaluation_failure',step,envs)))
            raise
        finally:
            agent.networks.train(was_training)
            cleanup_results=close_environments(envs)
            after=record(mempath,'evaluation_after',step)
            cleanup=dict(worker_cleanup=cleanup_results,all_workers_stopped=all(r.get('stopped',False) for r in cleanup_results),pids=[e.process.pid for e in envs],memory_before=before,memory_after=after,memory_delta_bytes=after['current']-before['current'])
            write_json(folder/'process_cleanup.json',cleanup)
            if summary is not None:
                summary.update(seconds=time.monotonic()-start,cleanup=cleanup)
                write_json(folder/'results.json',summary)
            if not cleanup['all_workers_stopped']:raise RuntimeError('Evaluation worker cleanup failed')
        return summary
