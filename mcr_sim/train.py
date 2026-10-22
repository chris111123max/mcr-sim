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
    return {"algorithm":"LSTM-SAC","observation_dim":cfg.observation.dimension,"action_dim":3,"sac":asdict(cfg.sac),"sofa":component_parameters(),"train_seeds":cfg.vessel_seeds,"eval_seeds":cfg.eval_vessel_seeds,"vessel_root":str(cfg.vessel_root),"assets":assets,"eval_episodes":32,"eval_episodes_per_vessel":16,"formal_training_started":False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--total-steps",type=int)
    parser.add_argument("--num-envs",type=int,default=32)
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
    from .sampling import ProcessEnvironment
    from .evaluation import ParallelEvaluation
    from pathlib import Path
    cfg=DEFAULT_CONFIG
    device=resolve_device(args.device,distributed=False)
    if not device.resolved.startswith("npu:"):raise RuntimeError("Formal run requires explicit Ascend NPU; no CPU fallback")
    output=Path(args.output or cfg.output_root/("lstm_sac_"+datetime.now().strftime("%Y%m%d_%H%M%S")))
    output.mkdir(parents=True,exist_ok=True)
    if (output/"metrics.jsonl").exists() and not args.resume:raise FileExistsError("Never overwrite previous formal training")
    write_json(output/"preflight.json",report)
    write_json(output/"experiment.json",{"config":asdict(cfg),"device":asdict(device),"num_envs":args.num_envs,"total_transitions":args.total_steps or cfg.sac.training_steps_script_default,"seed":args.seed,"history_frames":8,"eval_plan":"32 independent episodes: 16 per held-out vessel","training_episodes_preserved_during_evaluation":True})
    write_json(output/"process.json",{"pid":os.getpid(),"status":"STARTING"})
    envs=[];evaluation=ParallelEvaluation(cfg,output/"evaluations")
    try:
        for _ in range(args.num_envs):envs.append(ProcessEnvironment(cfg,log_dir=output/"workers"))
        write_json(output/"process.json",{"pid":os.getpid(),"status":"RUNNING","worker_pids":[e.process.pid for e in envs]})
        run_training(envs,evaluation,cfg.sac,output,args.total_steps,device.resolved,sequence_length=8,seed=args.seed,resume=args.resume,strict_failures=True)
        write_json(output/"completion.json",{"status":"COMPLETE","steps":args.total_steps or cfg.sac.training_steps_script_default})
    except BaseException:
        write_json(output/"failure.json",{"status":"FAILED","traceback":traceback.format_exc()})
        print(traceback.format_exc(),flush=True);raise
    finally:
        for env in envs:env.close()
        write_json(output/"shutdown.json",{"all_training_workers_stopped":all(not e.process.is_alive() for e in envs),"worker_pids":[e.process.pid for e in envs]})

if __name__=="__main__":main()
