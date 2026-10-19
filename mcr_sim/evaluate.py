"""Evaluate a trusted LSTM-SAC checkpoint on the disjoint evaluation vessels."""
import argparse
import json
from .config import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import torch
    from .distributed.device import resolve_device
    from .recurrent_sac import RecurrentSAC, evaluate_policy
    from .rl_env import MagneticNavigationEnv
    from .sofa_simulation import SofaBackend
    device = resolve_device(args.device, distributed=False).resolved
    agent = RecurrentSAC(DEFAULT_CONFIG.sac, device=device)
    agent.restore(torch.load(args.checkpoint, map_location=device))
    env = MagneticNavigationEnv(SofaBackend(), mode="eval")
    try:
        print(json.dumps(evaluate_policy(agent, env), indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
