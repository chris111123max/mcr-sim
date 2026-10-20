"""Small episode-aware recurrent SAC runner used by ``train_sac``.

Nothing in this module starts training on import.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import random
from contextlib import nullcontext
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
    observation_history: np.ndarray
    next_observation_history: np.ndarray


class TransitionRing:
    """O(1) logical indexing while retaining chronological checkpoint order."""
    def __init__(self, capacity):
        self.maxlen = int(capacity)
        if self.maxlen < 1:
            raise ValueError("Replay capacity must be positive")
        self.clear()

    def clear(self):
        self._slots = [None] * self.maxlen
        self._start = self._size = 0

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError(index)
        return self._slots[(self._start + index) % self.maxlen]

    def __iter__(self):
        return (self[index] for index in range(self._size))

    def append(self, value):
        if self._size < self.maxlen:
            self._slots[(self._start + self._size) % self.maxlen] = value
            self._size += 1
        else:
            self._slots[self._start] = value
            self._start = (self._start + 1) % self.maxlen

    def extend(self, values):
        for value in values:
            self.append(value)


class EpisodeSequenceReplay:
    FORMAT_VERSION = 2

    def __init__(self, capacity, batch_size, sequence_length=8, observation_dim=ObservationConfig().dimension):
        self.data = TransitionRing(capacity)
        self.batch_size = int(batch_size)
        self.sequence_length = int(sequence_length)
        self.observation_dim = int(observation_dim)
        if self.sequence_length < 1:
            raise ValueError("History window must be positive")
        self.episodes = {}
        self._histories = {}
        self._expected_observations = {}

    def end_episode(self, env_id):
        env_id = int(env_id)
        self.episodes[env_id] = self.episodes.get(env_id, 0) + 1
        self._histories.pop(env_id, None)
        self._expected_observations.pop(env_id, None)

    def add(self, observation, action, reward, next_observation, terminated, truncated=False, env_id=0, *, valid=True):
        env_id = int(env_id)
        if not valid:
            # No failure transition, stale fallback observation or unsafe target
            # is admitted. Earlier trusted transitions remain usable.
            self.end_episode(env_id)
            return False
        observation = self._validate_observation(observation)
        next_observation = self._validate_observation(next_observation)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)) or not np.isfinite(reward):
            raise ValueError("Replay action/reward must be finite")
        expected = self._expected_observations.get(env_id)
        if expected is not None and not np.array_equal(expected, observation):
            raise ValueError("Environment history discontinuity; end_episode() is required at reset")
        history = self._histories.setdefault(env_id, deque(maxlen=self.sequence_length))
        history.append(observation.copy())
        current = np.stack(history)
        following = np.concatenate((current, next_observation[None]), axis=0)[-self.sequence_length:]
        episode = self.episodes.setdefault(env_id, 0)
        self.data.append(Transition(observation.copy(), action.copy(), float(reward), next_observation.copy(),
                                    bool(terminated), bool(truncated), episode, env_id,
                                    current.copy(), following.copy()))
        self._expected_observations[env_id] = next_observation.copy()
        if terminated or truncated:
            self.end_episode(env_id)
        return True

    def _validate_observation(self, observation):
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (self.observation_dim,) or not np.all(np.isfinite(value)):
            raise ValueError("Replay requires finite %dD observations" % self.observation_dim)
        return value

    def __len__(self):
        return len(self.data)

    def _sequence(self, index, records=None):
        record = (self.data if records is None else records)[index]
        current, following = record.observation_history, record.next_observation_history
        for sequence in (current, following):
            if sequence.ndim != 2 or sequence.shape[1] != self.observation_dim or not 1 <= len(sequence) <= self.sequence_length or not np.all(np.isfinite(sequence)):
                raise ValueError("Invalid or legacy replay history")
        if not np.array_equal(current[-1], record.observation) or not np.array_equal(following[-1], record.next_observation):
            raise ValueError("Replay history endpoint mismatch")
        return current, following

    def add_batch(self, transitions):
        """Publish a chronological prefix only; callers retain update boundaries."""
        return [self.add(*args, **kwargs) for args, kwargs in transitions]

    def sample(self, device):
        if len(self.data) < self.batch_size:
            raise RuntimeError("Replay buffer has fewer items than batch size")
        region = getattr(self, "profile", None)
        with region.region("replay_pack") if region is not None else nullcontext():
            indices = random.sample(range(len(self.data)), self.batch_size)
            history = np.zeros((2, self.batch_size, self.sequence_length, self.observation_dim), dtype=np.float32)
            lengths = np.empty((2, self.batch_size), dtype=np.int64)
            values = np.empty((self.batch_size, 5), dtype=np.float32)
            for row, index in enumerate(indices):
                current, following = self._sequence(index)
                history[0, row, :len(current)] = current
                history[1, row, :len(following)] = following
                lengths[:, row] = (len(current), len(following))
                record = self.data[index]
                values[row, :3] = record.action
                values[row, 3:] = record.reward, record.terminated
        with region.region("replay_transfer") if region is not None else nullcontext():
            history = torch.from_numpy(history).to(device, non_blocking=True)
            values = torch.from_numpy(values).to(device, non_blocking=True)
        # Host lengths avoid any accelerator -> host synchronization in the LSTM.
        return {"observation": history[0], "next_observation": history[1],
                "observation_lengths": torch.from_numpy(lengths[0]),
                "next_observation_lengths": torch.from_numpy(lengths[1]),
                "action": values[:, :3], "reward": values[:, 3:4], "done": values[:, 4:5]}


class RecurrentSAC:
    def __init__(self, config, device="cpu", tau=0.005, sequence_length=8):
        self.config = config
        self.device = torch.device(device)
        self.tau = float(tau)
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 1:
            raise ValueError("History window must be positive")
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
        window = np.asarray(observation_sequence, dtype=np.float32)
        if window.ndim == 1:
            window = window[None]
        if window.ndim != 2 or len(window) < 1 or not np.all(np.isfinite(window)):
            raise ValueError("Action inference requires a finite real-frame window")
        sequence = torch.as_tensor(window[-self.sequence_length:], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            latent = self.networks.encoder(sequence)
            if deterministic:
                mean, _ = self.networks.actor(latent)
                result = torch.tanh(mean)
            else:
                result, _ = self.networks.actor.sample(latent)
        return result[0].cpu().numpy().astype(np.float32)

    def action_with_state(self, observation, history=None, deterministic=False):
        value = torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            latent, history = self.networks.encoder.step(value, history, self.sequence_length)
            if deterministic:
                action = torch.tanh(self.networks.actor(latent)[0])
            else:
                action, _ = self.networks.actor.sample(latent)
        return action[0].cpu().numpy().astype(np.float32), history

    @torch.inference_mode()
    def act_batch(self, observations, histories=None, deterministic=False, return_parameters=False):
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != ObservationConfig().dimension or not np.all(np.isfinite(values)):
            raise ValueError("Batch inference requires finite (N, 41) observations")
        if histories is None:
            histories = [None] * len(values)
        if len(histories) != len(values):
            raise ValueError("One independent history is required per environment")
        windows, lengths = [], []
        storage = np.zeros((len(values), self.sequence_length, values.shape[1]), dtype=np.float32)
        for row, (observation, history) in enumerate(zip(values, histories)):
            if history is None:
                window = observation[None].copy()
            else:
                previous = np.asarray(history, dtype=np.float32)
                if previous.ndim != 2 or previous.shape[1] != values.shape[1] or not np.all(np.isfinite(previous)):
                    raise ValueError("Invalid environment history")
                window = np.concatenate((previous, observation[None]), axis=0)[-self.sequence_length:]
            windows.append(window)
            lengths.append(len(window))
            storage[row, :len(window)] = window
        sequence = torch.from_numpy(storage).to(self.device, non_blocking=True)
        latent = self.networks.encoder(sequence, lengths)
        mean, log_std = self.networks.actor(latent)
        if deterministic:
            actions = torch.tanh(mean)
        else:
            normal = torch.distributions.Normal(mean, log_std.exp(), validate_args=False)
            actions = torch.tanh(normal.rsample())
        # One mandatory device->host transfer for the whole SOFA action batch.
        packed = torch.cat((actions, mean, log_std), dim=-1) if return_parameters else actions
        packed = packed.cpu().numpy()
        if return_parameters:
            return packed[:, :3], windows, {"mean": packed[:, 3:6], "log_std": packed[:, 6:9]}
        return packed, windows

    def checkpoint(self):
        return {"history_semantics": "fixed_window_zero_init_v1", "sequence_length": self.sequence_length,
                "networks": self.networks.state_dict(), "target": self.target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict()}

    def restore(self, checkpoint):
        if checkpoint.get("history_semantics") != "fixed_window_zero_init_v1" or checkpoint.get("sequence_length") != self.sequence_length:
            raise ValueError("Legacy recurrent checkpoint cannot be resumed with fixed-window history semantics")
        self.networks.load_state_dict(checkpoint["networks"])
        self.target.load_state_dict(checkpoint["target"])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"].to(self.device))
        for name in ("critic_optimizer", "actor_optimizer", "alpha_optimizer"):
            if name in checkpoint:
                getattr(self, name).load_state_dict(checkpoint[name])

    def update(self, batch, return_tensors=False):
        profiler = getattr(self, "profile", None)
        def region(name):
            return profiler.region(name) if profiler is not None else nullcontext()
        with region("critic"):
            with torch.no_grad():
                next_online_latent = self.networks.encoder(batch["next_observation"], batch["next_observation_lengths"])
                next_action, next_log_probability = self.networks.actor.sample(next_online_latent)
                next_target_latent = self.target.encoder(batch["next_observation"], batch["next_observation_lengths"])
                target_q1, target_q2 = self.target.critics(next_target_latent, next_action)
                target_q = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_log_probability
                expected = self.config.reward_scaling * batch["reward"] + self.config.gamma * (1.0 - batch["done"]) * target_q

            latent = self.networks.encoder(batch["observation"], batch["observation_lengths"])
            q1, q2 = self.networks.critics(latent, batch["action"])
            critic_loss = nn.functional.mse_loss(q1, expected) + nn.functional.mse_loss(q2, expected)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizer.step()

        with region("actor"):
            for parameter in self.networks.critics.parameters():
                parameter.requires_grad_(False)
            with torch.no_grad():
                actor_latent = self.networks.encoder(batch["observation"], batch["observation_lengths"])
            policy_action, log_probability = self.networks.actor.sample(actor_latent)
            policy_q1, policy_q2 = self.networks.critics(actor_latent, policy_action)
            actor_loss = (self.alpha.detach() * log_probability - torch.minimum(policy_q1, policy_q2)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            for parameter in self.networks.critics.parameters():
                parameter.requires_grad_(True)

        with region("alpha"):
            alpha_loss = -(self.log_alpha * (log_probability.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        with region("target"):
            with torch.no_grad():
                for target_parameter, parameter in zip(self.target.parameters(), self.networks.parameters()):
                    target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)
        metrics = {"critic_loss": critic_loss.detach(), "actor_loss": actor_loss.detach(), "alpha": self.alpha.detach().squeeze()}
        if return_tensors:
            return metrics
        # Legacy callers may explicitly request immediately materialized metrics.
        packed = torch.stack(tuple(metrics.values())).cpu().tolist()
        return dict(zip(metrics, packed))


def evaluate_policy(agent, env, vessel_seeds=None, sequence_length=8):
    if sequence_length != agent.sequence_length:
        raise ValueError("Evaluation and training history windows must agree")
    results = []
    seeds = vessel_seeds if vessel_seeds is not None else env.config.eval_vessel_seeds
    for seed in seeds:
        observation, _ = env.reset(seed=seed, options={"vessel_seed": seed})
        history, done, total_reward = None, False, 0.0
        while not done:
            action, history = agent.action_with_state(observation, history, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        results.append({"vessel_seed": seed, "success": bool(info["success"]),
                        "reward": float(total_reward), "path_length_mm": float(info["path_length_mm"]),
                        "termination_reason": info["termination_reason"]})
    return results


def run_training(train_env, eval_env, config, output_dir, total_steps=None, device="cpu", sequence_length=8,
                 seed=0, resume=None, profile=None, log_interval_updates=20, strict_failures=False):
    """Batched causal inference and replay, preserving each transition update boundary."""
    from dataclasses import asdict
    from .sampling import step_environments, reset_environments
    total_steps = int(total_steps or config.training_steps_script_default)
    if total_steps < 1 or log_interval_updates < 1:
        raise ValueError("Positive transition budget and logging interval required")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    envs = list(train_env) if isinstance(train_env, (tuple, list)) else [train_env]
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({"sac": asdict(config), "seed": seed, "num_envs": len(envs), "sequence_length": sequence_length}, indent=2))
    replay = EpisodeSequenceReplay(config.replay_buffer_size, config.batch_size, sequence_length)
    agent = RecurrentSAC(config, device=device, sequence_length=sequence_length)
    agent.profile = replay.profile = profile
    def region(name):
        return profile.region(name) if profile is not None else nullcontext()
    if agent.device.type == "npu":
        torch.npu.manual_seed(seed)
    start = 0
    if resume:
        checkpoint = torch.load(resume, map_location=agent.device)
        if checkpoint.get("replay_format_version") != EpisodeSequenceReplay.FORMAT_VERSION:
            raise ValueError("Legacy replay cannot restore real-frame histories")
        agent.restore(checkpoint); start = int(checkpoint["step"])
        replay.data.extend(checkpoint.get("replay", []))
        replay.episodes = {int(k): int(v)+1 for k,v in checkpoint.get("replay_episodes",{}).items()}
        random.setstate(checkpoint["python_rng"]); np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if agent.device.type == "npu" and "npu_rng" in checkpoint:
            torch.npu.set_rng_state(checkpoint["npu_rng"].cpu(), agent.device)
    for i,env in enumerate(envs):
        env.action_space.seed(seed+i)
    with region("initial_resets"):
        initial = reset_environments(envs, [{"seed": seed+i+start} for i in range(len(envs))])
    observations = [result[0] for result in initial]
    histories = [None] * len(envs)
    if profile is not None:
        profile.training_start()
    update_interval = int(round(1.0 / config.update_per_explore_step))
    log_path = output / "metrics.jsonl"
    def log(record):
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record,default=asdict)+"\n")
    pending_metrics = []
    def flush_metrics():
        if not pending_metrics:
            return
        values = torch.stack([torch.stack(tuple(metric.values())) for _,metric in pending_metrics]).cpu().numpy()
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Nonfinite SAC metric")
        for (record_step,metric),row in zip(pending_metrics,values):
            log({"step":record_step,"update":dict(zip(metric,map(float,row)))})
        pending_metrics.clear()
    def save(step):
        checkpoint = {**agent.checkpoint(), "replay_format_version": EpisodeSequenceReplay.FORMAT_VERSION,
                      "step":step,"replay":list(replay.data),"replay_episodes":replay.episodes,
                      "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state()}
        if agent.device.type == "npu":
            checkpoint["npu_rng"] = torch.npu.get_rng_state(agent.device)
        temporary = output / "checkpoint_pending.pt"
        torch.save(checkpoint,temporary);temporary.replace(output/("lstm_sac_%d_steps.pt"%step))
    step = start; updates = 0
    while step < total_steps:
        indices = [(step+offset)%len(envs) for offset in range(min(len(envs),total_steps-step))]
        with region("actor_batch"):
            policy_actions,new_histories = agent.act_batch(np.stack([observations[i] for i in indices]), [histories[i] for i in indices])
        actions = []
        for offset,index in enumerate(indices):
            histories[index] = new_histories[offset]
            actions.append(envs[index].action_space.sample() if step+offset+1<=config.heatup_steps else policy_actions[offset])
        with region("sampling"):
            results = step_environments([envs[i] for i in indices],actions)
        reset_indices = []; cursor = 0
        while cursor < len(indices):
            # Do not publish future transitions before a scheduled update/evaluation.
            count = min(len(indices)-cursor,update_interval-step%update_interval,
                        int(config.explore_steps_between_eval)-step%int(config.explore_steps_between_eval))
            records = []; prepared = []
            for offset in range(cursor,cursor+count):
                index = indices[offset]; result = results[offset]
                following,reward,terminated,truncated,info = result
                candidate = np.asarray(following)
                valid = (bool(info.get("transition_valid",True)) and bool(info.get("observation_valid",True)) and
                         not info.get("simulation_error",False) and info.get("state_error") is None and
                         info.get("termination_reason")!="simulation_error" and candidate.shape==observations[index].shape and
                         np.all(np.isfinite(candidate)) and np.isfinite(reward))
                if strict_failures and not valid:
                    raise RuntimeError("Invalid SOFA transition: "+repr(info))
                records.append(((observations[index],actions[offset],reward,following,terminated,truncated,index),{"valid":valid}))
                prepared.append((index,result,valid))
            with region("replay_add"):
                replay.add_batch(records)
            for index,(following,reward,terminated,truncated,info),valid in prepared:
                step += 1
                if valid:
                    observations[index] = following
                if terminated or truncated or not valid:
                    log({"step":step,"env_id":index,"episode":info.get("episode_metrics"),
                         "termination_reason":info.get("termination_reason") if valid else "simulation_error",
                         "transition_stored":valid,"state_error":info.get("state_error")})
                    histories[index] = None; reset_indices.append(index)
            cursor += count
            if step>config.heatup_steps and step%update_interval==0 and len(replay)>=config.batch_size:
                batch = replay.sample(agent.device)
                with region("update_total"):
                    metrics = agent.update(batch,return_tensors=True)
                updates += 1;pending_metrics.append((step,metrics))
                if len(pending_metrics)>=log_interval_updates:
                    flush_metrics()
            if step%int(config.explore_steps_between_eval)==0:
                flush_metrics();log({"step":step,"evaluation":evaluate_policy(agent,eval_env,sequence_length=sequence_length)});save(step)
        if reset_indices:
            with region("episode_resets"):
                reset_results = reset_environments([envs[i] for i in reset_indices])
            for index,result in zip(reset_indices,reset_results):
                observations[index] = result[0]
        if profile is not None:
            profile.round_finished(step,updates,agent,replay)
    flush_metrics();save(total_steps)
    if profile is not None:
        profile.training_end(step,updates,agent,replay)
    return agent
