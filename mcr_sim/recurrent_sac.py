"""Small episode-aware recurrent SAC runner used by ``train_sac``.

Nothing in this module starts training on import.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch import nn

from .networks import SACNetworks
from .config import ObservationConfig


@dataclass
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    episode: int
    env_id: int


class EpisodeSequenceReplay:
    def __init__(self, capacity, batch_size, sequence_length=8, observation_dim=ObservationConfig().dimension):
        self.data = deque(maxlen=int(capacity))
        self.batch_size = int(batch_size)
        self.sequence_length = int(sequence_length)
        self.episodes = {}
        self.observation_dim = int(observation_dim)

    def add(self, observation, action, reward, next_observation, terminated, truncated=False, env_id=0):
        self._validate_observation(observation)
        self._validate_observation(next_observation)
        episode = self.episodes.setdefault(int(env_id), 0)
        self.data.append(Transition(np.asarray(observation, dtype=np.float32).copy(),
                                    np.asarray(action, dtype=np.float32).copy(), float(reward),
                                    np.asarray(next_observation, dtype=np.float32).copy(),
                                    bool(terminated), bool(truncated), episode, int(env_id)))
        if terminated or truncated:
            self.episodes[int(env_id)] += 1

    def _validate_observation(self, observation):
        value = np.asarray(observation)
        if value.shape != (self.observation_dim,) or not np.all(np.isfinite(value)):
            raise ValueError("Replay requires finite %dD observations; legacy states cannot be mixed" % self.observation_dim)

    def __len__(self):
        return len(self.data)

    def _sequence(self, index, records=None):
        records = list(self.data) if records is None else records
        current = records[index]
        self._validate_observation(current.next_observation)
        selected = []
        for i in range(index, -1, -1):
            record = records[i]
            if record.env_id != current.env_id:
                continue
            if record.episode != current.episode:
                break
            self._validate_observation(record.observation)
            selected.append(record.observation)
            if len(selected) == self.sequence_length:
                break
        selected.reverse()
        while len(selected) < self.sequence_length:
            selected.insert(0, selected[0])
        next_selected = selected[1:] + [current.next_observation]
        return np.stack(selected), np.stack(next_selected)

    def sample(self, device):
        if len(self.data) < self.batch_size:
            raise RuntimeError("Replay buffer has fewer items than batch size")
        records = list(self.data)
        indices = random.sample(range(len(records)), self.batch_size)
        sequences, next_sequences = zip(*(self._sequence(i, records) for i in indices))
        tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
        return {"observation": tensor(sequences), "next_observation": tensor(next_sequences),
                "action": tensor([records[i].action for i in indices]),
                "reward": tensor([[records[i].reward] for i in indices]),
                # Time limits/VesselEnd/SimError are truncations, not MDP terminal states.
                "done": tensor([[records[i].terminated] for i in indices])}


class RecurrentSAC:
    def __init__(self, config, device="cpu", tau=0.005):
        self.config = config
        self.device = torch.device(device)
        self.tau = float(tau)
        self.networks = SACNetworks(
            observation_dim=ObservationConfig().dimension,
            action_dim=3,
            lstm_nodes=config.lstm_nodes,
            lstm_layers=config.lstm_layers,
            hidden=config.actor_hidden,
        ).to(self.device)
        self.target = deepcopy(self.networks).to(self.device).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.critic_optimizer = torch.optim.Adam(
            list(self.networks.encoder.parameters()) + list(self.networks.critics.parameters()),
            lr=config.learning_rate,
        )
        self.actor_optimizer = torch.optim.Adam(self.networks.actor.parameters(), lr=config.learning_rate)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
        self.target_entropy = -3.0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def action(self, observation_sequence, deterministic=False):
        sequence = torch.as_tensor(np.asarray(observation_sequence), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            latent = self.networks.encoder(sequence)
            if deterministic:
                mean, _ = self.networks.actor(latent)
                result = torch.tanh(mean)
            else:
                result, _ = self.networks.actor.sample(latent)
        return result[0].cpu().numpy().astype(np.float32)

    def action_with_state(self, observation, hidden=None, deterministic=False):
        value = torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            latent, hidden = self.networks.encoder.step(value, hidden)
            if deterministic:
                action = torch.tanh(self.networks.actor(latent)[0])
            else:
                action, _ = self.networks.actor.sample(latent)
        return action[0].cpu().numpy().astype(np.float32), hidden

    def checkpoint(self):
        return {"networks": self.networks.state_dict(), "target": self.target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict()}

    def restore(self, checkpoint):
        self.networks.load_state_dict(checkpoint["networks"])
        self.target.load_state_dict(checkpoint["target"])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"].to(self.device))
        for name in ("critic_optimizer", "actor_optimizer", "alpha_optimizer"):
            if name in checkpoint:
                getattr(self, name).load_state_dict(checkpoint[name])

    def update(self, batch):
        with torch.no_grad():
            next_latent = self.target.encoder(batch["next_observation"])
            next_action, next_log_probability = self.networks.actor.sample(next_latent)
            target_q1, target_q2 = self.target.critics(next_latent, next_action)
            target_q = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_log_probability
            expected = self.config.reward_scaling * batch["reward"] + self.config.gamma * (1.0 - batch["done"]) * target_q

        latent = self.networks.encoder(batch["observation"])
        q1, q2 = self.networks.critics(latent, batch["action"])
        critic_loss = nn.functional.mse_loss(q1, expected) + nn.functional.mse_loss(q2, expected)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        for parameter in self.networks.critics.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            actor_latent = self.networks.encoder(batch["observation"])
        policy_action, log_probability = self.networks.actor.sample(actor_latent)
        policy_q1, policy_q2 = self.networks.critics(actor_latent, policy_action)
        actor_loss = (self.alpha.detach() * log_probability - torch.minimum(policy_q1, policy_q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for parameter in self.networks.critics.parameters():
            parameter.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (log_probability.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(self.target.parameters(), self.networks.parameters()):
                target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
        }


def evaluate_policy(agent, env, vessel_seeds=None, sequence_length=8):
    results = []
    seeds = vessel_seeds if vessel_seeds is not None else env.config.eval_vessel_seeds
    for seed in seeds:
        observation, _ = env.reset(seed=seed, options={"vessel_seed": seed})
        hidden, done, total_reward = None, False, 0.0
        while not done:
            action, hidden = agent.action_with_state(observation, hidden, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        results.append({"vessel_seed": seed, "success": bool(info["success"]),
                        "reward": float(total_reward), "path_length_mm": float(info["path_length_mm"]),
                        "termination_reason": info["termination_reason"]})
    return results


def run_training(train_env, eval_env, config, output_dir, total_steps=None, device="cpu", sequence_length=8, seed=0, resume=None):
    """Explicit-only runner. Each env has independent LSTM state and replay episodes.

    Online rollouts keep full hidden state; updates use the morning runner's
    episode-safe 8-step truncated BPTT. No PPO/HER/curriculum branches exist.
    """
    from dataclasses import asdict
    import time
    total_steps = int(total_steps or config.training_steps_script_default)
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    envs = list(train_env) if isinstance(train_env, (tuple, list)) else [train_env]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({"sac": asdict(config), "seed": seed, "num_envs": len(envs), "sequence_length": sequence_length}, indent=2))
    replay = EpisodeSequenceReplay(config.replay_buffer_size, config.batch_size, sequence_length)
    agent = RecurrentSAC(config, device=device)
    start = 0
    if resume:
        checkpoint = torch.load(resume, map_location=agent.device)
        agent.restore(checkpoint)
        start = int(checkpoint["step"])
        replay.data.extend(checkpoint.get("replay", []))
        # Resuming starts fresh physical episodes; never append across the reset.
        replay.episodes = {int(k): int(v)+1 for k, v in checkpoint.get("replay_episodes", {}).items()}
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
    observations, hidden = [], [None] * len(envs)
    for i, env in enumerate(envs):
        env.action_space.seed(seed+i)
        observation, _ = env.reset(seed=seed+i+start)
        observations.append(observation)
    update_interval = int(round(1.0 / config.update_per_explore_step))
    log_path = output / "metrics.jsonl"
    def log(record):
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
    def save(step):
        checkpoint = {**agent.checkpoint(), "step": step, "replay": list(replay.data),
                      "replay_episodes": replay.episodes, "python_rng": random.getstate(),
                      "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}
        temporary = output / "checkpoint_pending.pt"
        torch.save(checkpoint, temporary)
        temporary.replace(output / ("lstm_sac_%d_steps.pt" % step))
    for step in range(start+1, total_steps+1):
        index = (step-1) % len(envs)
        env, observation = envs[index], observations[index]
        # Maintain hidden state even during random-action heatup.
        policy_action, hidden[index] = agent.action_with_state(observation, hidden[index])
        action = env.action_space.sample() if step <= config.heatup_steps else policy_action
        next_observation, reward, terminated, truncated, info = env.step(action)
        replay.add(observation, action, reward, next_observation, terminated, truncated, index)
        observations[index] = next_observation
        if terminated or truncated:
            log({"step": step, "env_id": index, "episode": info.get("episode_metrics"),
                 "termination_reason": info["termination_reason"]})
            observations[index], _ = env.reset()
            hidden[index] = None
        if step > config.heatup_steps and step % update_interval == 0 and len(replay) >= config.batch_size:
            update = agent.update(replay.sample(agent.device))
            log({"step": step, "update": update})
        if step % int(config.explore_steps_between_eval) == 0:
            log({"step": step, "evaluation": evaluate_policy(agent, eval_env)})
            save(step)
    save(total_steps)
    return agent
