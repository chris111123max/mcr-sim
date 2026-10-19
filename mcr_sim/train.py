"""The only formal training entry: recurrent SAC. Execution requires --execute."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from .config import DEFAULT_CONFIG


def preflight():
    from .sofa_simulation import component_parameters
    return {"algorithm": "LSTM-SAC", "observation_dim": DEFAULT_CONFIG.observation.dimension, "action_dim": 3,
            "config": asdict(DEFAULT_CONFIG.sac), "sofa": component_parameters(),
            "train_seeds": DEFAULT_CONFIG.vessel_seeds, "eval_seeds": DEFAULT_CONFIG.eval_vessel_seeds,
            "vessel_root": str(DEFAULT_CONFIG.vessel_root), "formal_training_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(json.dumps(preflight(), indent=2))
    if args.preflight or not args.execute:
        return
    if args.num_envs < 1:
        parser.error("num-envs must be positive")
    from .distributed.device import resolve_device
    from .rl_env import MagneticNavigationEnv
    from .sofa_simulation import SofaBackend
    from .recurrent_sac import run_training
    from .sampling import ProcessEnvironment
    from .vessel_bundle import discover_vessel_bundles
    cfg = DEFAULT_CONFIG
    available = {b.seed for b in discover_vessel_bundles(cfg.vessel_root)}
    missing = set(cfg.vessel_seeds + cfg.eval_vessel_seeds) - available
    if missing:
        raise FileNotFoundError("Generate configured training/eval caches explicitly first; missing seeds: " + str(sorted(missing)))
    device = resolve_device(args.device, distributed=False).resolved
    output = args.output or str(cfg.output_root / ("lstm_sac_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    envs, eval_env = [], None
    try:
        for _ in range(args.num_envs):
            envs.append(ProcessEnvironment(cfg) if args.num_envs > 1 else MagneticNavigationEnv(SofaBackend(cfg), config=cfg))
        eval_env = MagneticNavigationEnv(SofaBackend(cfg), config=cfg, mode="eval")
        run_training(envs, eval_env, cfg.sac, output, args.total_steps, device,
                     seed=args.seed, resume=args.resume)
    finally:
        for env in envs:
            env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
