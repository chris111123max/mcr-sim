"""Formal existing LSTM-SAC entry; execution requires --execute."""
import argparse,json,traceback,os
from dataclasses import asdict
from datetime import datetime
from .config import DEFAULT_CONFIG
from .serialization import json_default,write_json

def preflight():
    from .sofa_simulation import component_parameters
    from .vessel_bundle import discover_vessel_bundles,coordinate_consistency
    cfg=DEFAULT_CONFIG
    assets=[]
    for split,seeds in (("train",cfg.vessel_seeds),("eval",cfg.eval_vessel_seeds)):
        bundles=discover_vessel_bundles(cfg.vessel_root,split=split)
        if {b.seed for b in bundles}!=set(seeds):raise ValueError("Missing/extra configured "+split+" vessels")
        for b in bundles:
            check=coordinate_consistency(b)
            if not all(check[k] for k in ("mm_to_m_exact","all_branches_connected","centerlines_inside_stl_bbox")):raise ValueError("Invalid coordinate/topology record")
            assets.append({"split":split,"seed":b.seed,"directory":str(b.directory),"faces":b.metadata["validation"]["mesh_quality"]["triangles"],"original_validation_passed":b.metadata["validation"]["passed"],"diameter_exception_used":not b.metadata["validation"]["passed"],"coordinate_checks":check})
    return {"algorithm":"LSTM-SAC","observation_dim":cfg.observation.dimension,"action_dim":3,"sac":asdict(cfg.sac),"sofa":component_parameters(),"train_seeds":cfg.vessel_seeds,"eval_seeds":cfg.eval_vessel_seeds,"vessel_root":str(cfg.vessel_root),"assets":assets,"eval_episodes":10,"eval_episodes_per_vessel":5,"eval_workers":2,"formal_training_started":False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--total-steps",type=int)
    parser.add_argument("--num-envs",type=int,default=24)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--device",default="npu:0")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    args=parser.parse_args()
    report=preflight();print(json.dumps(report,default=json_default,indent=2),flush=True)
    if args.preflight or not args.execute:return
    if args.num_envs<1:parser.error("num-envs must be positive")
    from .native_threads import limit_worker_environment
    limit_worker_environment()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from .distributed.device import resolve_device
    from .recurrent_sac import run_training
    from .sampling import ProcessEnvironment,close_environments
    from .evaluation import ParallelEvaluation
    from pathlib import Path
    cfg=DEFAULT_CONFIG
    device=resolve_device(args.device,distributed=False)
    if not device.resolved.startswith("npu:"):raise RuntimeError("Formal run requires explicit Ascend NPU; no CPU fallback")
    output=Path(args.output or cfg.output_root/("lstm_sac_"+datetime.now().strftime("%Y%m%d_%H%M%S")))
    output.mkdir(parents=True,exist_ok=True)
    allowed_launch_files={'train.log','launch.pid'}
    existing={p.name for p in output.iterdir()}
    if existing-allowed_launch_files:
        raise FileExistsError('Output contains prior artifacts; resume also requires a new directory')
    import fcntl,signal
    lock=(cfg.output_root/'.formal_training.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another formal learner holds the project lock')
    envs=[];status='STARTING';claimed=False
    try:
        # Exclusive marker prevents two launches from claiming even the same new directory.
        with (output/'run_owner.json').open('x') as stream:
            json.dump({'pid':os.getpid()},stream)
        claimed=True
        write_json(output/'preflight.json',report)
        write_json(output/'experiment.json',dict(config=asdict(cfg),device=asdict(device),
                   num_envs=args.num_envs,total_transitions=args.total_steps or cfg.sac.training_steps_script_default,
                   seed=args.seed,history_frames=8,eval_plan='2 workers, 5 fixed episodes each, 150 steps',
                   checkpoint_interval_valid_transitions=50000,evaluation_interval_valid_transitions=250000,
                   training_episodes_preserved_during_evaluation=True))
        write_json(output/'process.json',dict(pid=os.getpid(),status=status))
        signal.signal(signal.SIGTERM,lambda *_:(output/'STOP').touch())
        signal.signal(signal.SIGINT,lambda *_:(output/'STOP').touch())
        evaluation=ParallelEvaluation(cfg,output/'evaluations')
        from .memory_guard import record,safe
        for _ in range(args.num_envs):
            if not safe(record(output/'memory.jsonl','before_training_worker',0,envs)):
                raise MemoryError('Insufficient memory reserve for another training worker')
            envs.append(ProcessEnvironment(cfg,log_dir=output/'workers'))
        status='RUNNING'
        write_json(output/'process.json',dict(pid=os.getpid(),status=status,worker_pids=[e.process.pid for e in envs]))
        run_training(envs,evaluation,cfg.sac,output,args.total_steps,device.resolved,
                     sequence_length=8,seed=args.seed,resume=args.resume,strict_failures=True)
        if (output/'safe_stop.json').exists():status='SAFE_STOPPED'
        else:
            status='COMPLETE'
            write_json(output/'completion.json',dict(status=status,steps=args.total_steps or cfg.sac.training_steps_script_default))
    except BaseException:
        status='FAILED';original=traceback.format_exc()
        try:
            if claimed:write_json(output/'failure.json',dict(status=status,traceback=original))
        except Exception as log_error:print('Failure log write error: '+repr(log_error),flush=True)
        print(original,flush=True)
        raise
    finally:
        cleanup=close_environments(envs)
        try:
            stopped=all(r.get('stopped',False) for r in cleanup)
            if claimed:
                write_json(output/'shutdown.json',dict(all_training_workers_stopped=stopped,workers=cleanup))
                write_json(output/'process.json',dict(pid=os.getpid(),status=status if stopped else 'CLEANUP_FAILED',worker_pids=[e.process.pid for e in envs]))
        except Exception as log_error:print('Shutdown log write error: '+repr(log_error),flush=True)
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
