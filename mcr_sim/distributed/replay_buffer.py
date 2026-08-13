"""Accelerator-resident SB3 replay buffer for small vector observations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


def _torch_dtype(numpy_dtype) -> th.dtype:
    return th.from_numpy(np.empty((), dtype=numpy_dtype)).dtype


class AcceleratorReplayBuffer(ReplayBuffer):
    """Keep replay tensors on the learner device and gather batches in place.

    Environment outputs are copied once when a vector step is inserted.  SAC
    minibatches are then indexed directly on the NPU, avoiding repeated copies
    of hundreds or thousands of observations from NumPy for every update.
    NumPy still generates sample indices so the established replay RNG stream
    and uniform sampling semantics are retained.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        if optimize_memory_usage:
            raise ValueError(
                "AcceleratorReplayBuffer requires optimize_memory_usage=False."
            )
        if isinstance(observation_space, spaces.Dict):
            raise TypeError("Dictionary observations require a separate replay buffer.")

        # Bypass ReplayBuffer.__init__, which would allocate a complete NumPy
        # replay before replacing it with accelerator tensors.
        BaseBuffer.__init__(
            self,
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
        )
        self.buffer_size = max(int(buffer_size) // int(n_envs), 1)
        self.optimize_memory_usage = False
        self.handle_timeout_termination = bool(handle_timeout_termination)

        observation_dtype = _torch_dtype(observation_space.dtype)
        action_dtype = _torch_dtype(self._maybe_cast_dtype(action_space.dtype))
        observation_shape = (self.buffer_size, self.n_envs, *self.obs_shape)
        self.observations = th.empty(
            observation_shape, dtype=observation_dtype, device=self.device
        )
        self.next_observations = th.empty_like(self.observations)
        self.actions = th.empty(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=action_dtype,
            device=self.device,
        )
        self.rewards = th.empty(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.dones = th.empty_like(self.rewards)
        self.timeouts = th.empty_like(self.rewards)
        self._probe_device_indexing()

    def _probe_device_indexing(self) -> None:
        """Fail during startup, rather than after the full replay warm-up."""

        probe = th.tensor(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
            dtype=th.float32,
            device=self.device,
        )
        rows = th.tensor([0, 1], dtype=th.long, device=self.device)
        columns = th.tensor([1, 2], dtype=th.long, device=self.device)
        selected = probe[rows, columns]
        selected_values = selected.detach().cpu().tolist()
        if selected_values != [1.0, 6.0]:
            raise RuntimeError("Accelerator replay advanced-index probe failed.")

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.observations,
            self.next_observations,
            self.actions,
            self.rewards,
            self.dones,
            self.timeouts,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _copy_array(self, destination: th.Tensor, value) -> None:
        source = th.as_tensor(np.asarray(value), dtype=destination.dtype, device=self.device)
        destination.copy_(source)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
        action = action.reshape((self.n_envs, self.action_dim))

        self._copy_array(self.observations[self.pos], obs)
        self._copy_array(self.next_observations[self.pos], next_obs)
        self._copy_array(self.actions[self.pos], action)
        self._copy_array(self.rewards[self.pos], reward)
        self._copy_array(self.dones[self.pos], done)
        if self.handle_timeout_termination:
            timeouts = [info.get("TimeLimit.truncated", False) for info in infos]
            self._copy_array(self.timeouts[self.pos], timeouts)
        else:
            self.timeouts[self.pos].zero_()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> ReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=int(batch_size))
        return self._get_samples(batch_inds, env=env)

    def _normalize_tensor(self, tensor: th.Tensor, env, observation: bool) -> th.Tensor:
        if env is None:
            return tensor
        host_value = tensor.detach().cpu().numpy()
        normalized = (
            env.normalize_obs(host_value)
            if observation
            else env.normalize_reward(host_value).astype(np.float32)
        )
        return th.as_tensor(normalized, device=self.device)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=len(batch_inds))
        batch_tensor = th.as_tensor(batch_inds, dtype=th.long, device=self.device)
        env_tensor = th.as_tensor(env_indices, dtype=th.long, device=self.device)

        observations = self.observations[batch_tensor, env_tensor]
        next_observations = self.next_observations[batch_tensor, env_tensor]
        actions = self.actions[batch_tensor, env_tensor]
        dones = (
            self.dones[batch_tensor, env_tensor]
            * (1.0 - self.timeouts[batch_tensor, env_tensor])
        ).reshape(-1, 1)
        rewards = self.rewards[batch_tensor, env_tensor].reshape(-1, 1)
        return ReplayBufferSamples(
            observations=self._normalize_tensor(observations, env, observation=True),
            actions=actions,
            next_observations=self._normalize_tensor(
                next_observations, env, observation=True
            ),
            dones=dones,
            rewards=self._normalize_tensor(rewards, env, observation=False),
        )
