"""Lazy 32-process deterministic evaluation of the existing SOFA environment."""
from collections import Counter
from pathlib import Path
import time,traceback
import numpy as np
from .sampling import ProcessEnvironment,reset_environments,step_environments
from .serialization import write_json

class ParallelEvaluation:
    def __init__(self,config,output_dir):
        if tuple(config.eval_vessel_seeds)!=(1111,1212):raise ValueError("Expected held-out 1111/1212")
        self.config=config;self.output=Path(output_dir);self.plan=[{"env_id":i,"vessel_seed":1111 if i<16 else 1212,"episode_seed":26010000+i} for i in range(32)]
        write_json(self.output/"fixed_episode_plan.json",self.plan)

    def evaluate(self,agent,step,sequence_length=8):
        if sequence_length!=agent.sequence_length:raise ValueError("Mismatched evaluation history")
        start=time.monotonic();envs=[];results=[];model_training=agent.networks.training
        folder=self.output/("step_%09d"%step);folder.mkdir(parents=True,exist_ok=False)
        try:
            for _ in self.plan:envs.append(ProcessEnvironment(self.config,mode="eval",log_dir=folder/"workers"))
            if len(envs)!=32 or len({e.process.pid for e in envs})!=32:raise RuntimeError("Evaluation requires exactly 32 independent workers")
            initial=reset_environments(envs,[{"seed":p["episode_seed"],"options":{"vessel_seed":p["vessel_seed"]}} for p in self.plan])
            observations=[x[0] for x in initial];histories=[None]*32;active=list(range(32))
            rewards=np.zeros(32);steps=np.zeros(32,dtype=int)
            for i,(_,info) in enumerate(initial):
                if info["vessel_seed"]!=self.plan[i]["vessel_seed"]:raise RuntimeError("Evaluation vessel contamination")
            agent.networks.eval()
            while active:
                actions,new_histories=agent.act_batch(np.stack([observations[i] for i in active]),[histories[i] for i in active],deterministic=True)
                if actions.shape!=(len(active),3) or not np.all(np.isfinite(actions)):raise FloatingPointError("Invalid evaluation action")
                batch=step_environments([envs[i] for i in active],actions)
                following_active=[]
                for j,(i,result) in enumerate(zip(active,batch)):
                    obs,reward,terminated,truncated,info=result
                    if not (np.asarray(obs).shape==(41,) and np.all(np.isfinite(obs)) and np.isfinite(reward) and info.get("transition_valid",False) and not info.get("simulation_error",False) and info.get("physics_substeps_attempted")==22 and info["vessel_seed"]==self.plan[i]["vessel_seed"]):raise RuntimeError("Invalid evaluation state: "+repr(info))
                    observations[i]=obs;histories[i]=new_histories[j];rewards[i]+=reward;steps[i]+=1
                    if terminated or truncated:
                        histories[i]=None
                        results.append({**self.plan[i],"worker_pid":envs[i].process.pid,"success":bool(info["success"]),"reward":float(rewards[i]),"steps":int(steps[i]),"termination_reason":info["termination_reason"],"target_branch":initial[i][1]["target_branch"],"target_m":initial[i][1]["target_m"]})
                    else:
                        if steps[i]>=self.config.task.max_episode_steps_eval:raise RuntimeError("Evaluation episode did not terminate")
                        following_active.append(i)
                active=following_active
            if len(results)!=32:raise RuntimeError("Incomplete evaluation")
            summary={"training_step":step,"episodes":32,"successes":sum(x["success"] for x in results),"per_vessel":{},"results":sorted(results,key=lambda x:x["env_id"]),"seconds":time.monotonic()-start,"deterministic":True,"replay_writes":0,"optimizer_updates":0}
            summary["success_rate"]=summary["successes"]/32
            for seed in self.config.eval_vessel_seeds:
                rows=[x for x in results if x["vessel_seed"]==seed]
                if len(rows)!=16:raise RuntimeError("Expected 16 episodes per vessel")
                count=sum(x["success"] for x in rows)
                summary["per_vessel"][str(seed)]={"episodes":16,"successes":count,"success_rate":count/16,"mean_reward":float(np.mean([x["reward"] for x in rows])),"mean_steps":float(np.mean([x["steps"] for x in rows])),"failure_reasons":dict(Counter(x["termination_reason"] for x in rows if not x["success"]))}
            write_json(folder/"results.json",summary)
            return summary
        except BaseException:
            write_json(folder/"failure.json",{"step":step,"traceback":traceback.format_exc(),"completed_episodes":results})
            raise
        finally:
            agent.networks.train(model_training)
            for env in envs:env.close()
            write_json(folder/"process_cleanup.json",{"all_workers_stopped":all(not env.process.is_alive() for env in envs),"pids":[env.process.pid for env in envs]})
