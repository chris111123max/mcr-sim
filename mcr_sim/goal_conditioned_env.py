"""Goal-conditioned wrappers and replay storage for the isolated Goal-SAC route.

The baseline :class:`MCREnv` is intentionally not modified.  This module adds
the desired-goal scalar (selected-route completion in [0, 1]) at the VecEnv
boundary and stores enough episode metadata to perform safe future HER.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnvWrapper


class GoalConditionedVecEnv(VecEnvWrapper):
    """Append a scalar desired route-completion goal to the baseline state.

    The real task always uses desired_goal=1.0.  HER changes the final scalar
    only inside replay samples; the SOFA environment and its V11 semantics are
    never changed.
    """

    def __init__(self, venv, final_goal: float = 1.0):
        super().__init__(venv)
        if getattr(venv.observation_space, "shape", None) is None:
            raise ValueError("GoalConditionedVecEnv requires a flat state observation.")
        self.base_observation_dim = int(venv.observation_space.shape[0])
        self.final_goal = float(np.clip(final_goal, 0.0, 1.0))
        low = np.concatenate(
            [np.asarray(venv.observation_space.low, dtype=np.float32), np.array([0.0], dtype=np.float32)]
        )
        high = np.concatenate(
            [np.asarray(venv.observation_space.high, dtype=np.float32), np.array([1.0], dtype=np.float32)]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.desired_goals = np.full((self.num_envs, 1), self.final_goal, dtype=np.float32)
        self._goal_episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._goal_episode_lengths = np.zeros(self.num_envs, dtype=np.int64)

    def _augment(self, observations):
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        return np.concatenate([obs, self.desired_goals], axis=1).astype(np.float32, copy=False)

    @staticmethod
    def _finite_risk(info, key):
        try:
            value = float(info.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0
        return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0

    def reset(self):
        self.desired_goals.fill(self.final_goal)
        self._goal_episode_returns.fill(0.0)
        self._goal_episode_lengths.fill(0)
        return self._augment(self.venv.reset())

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        out_infos = []
        sparse_rewards = np.empty(self.num_envs, dtype=np.float32)
        for index, info in enumerate(infos):
            info = dict(info or {})
            if "terminal_observation" in info and info["terminal_observation"] is not None:
                terminal = np.asarray(info["terminal_observation"], dtype=np.float32).reshape(-1)
                if terminal.shape[0] == self.base_observation_dim:
                    info["terminal_observation"] = np.concatenate(
                        [terminal, np.array([self.desired_goals[index, 0]], dtype=np.float32)]
                    )
            # These fields are consumed by SafeHerReplayBuffer and are also
            # useful in the standalone CSV/log diagnostics.
            info["original_desired_goal"] = float(self.desired_goals[index, 0])
            info["achieved_goal"] = float(np.clip(info.get("route_progress_ratio", 0.0), 0.0, 1.0))
            info["her_safe"] = bool(
                not info.get("out_of_vessel", False)
                and not info.get("wrong_branch", False)
                and not info.get("done_by_non_finite", False)
                and not info.get("route_projection_jump_rejected", False)
            )
            wall_risk = self._finite_risk(info, "sdf_body_warning_feature")
            branch_risk = self._finite_risk(info, "off_target_branch_feature")
            safety_penalty = -0.05 * wall_risk - 0.05 * branch_risk
            sparse_rewards[index] = np.float32(
                (0.0 if info.get("done_by_target", False) else -1.0) + safety_penalty
            )
            self._goal_episode_returns[index] += float(sparse_rewards[index])
            self._goal_episode_lengths[index] += 1
            info["goal_sparse_reward"] = float(sparse_rewards[index])
            info["goal_safety_penalty"] = float(safety_penalty)
            if dones[index]:
                # The next reset happens inside VecEnv.step_wait.  Keep the
                # original final goal in this transition's info.
                self.desired_goals[index, 0] = self.final_goal
                episode = dict(info.get("episode", {}))
                episode["r"] = float(self._goal_episode_returns[index])
                episode["l"] = int(self._goal_episode_lengths[index])
                info["episode"] = episode
                self._goal_episode_returns[index] = 0.0
                self._goal_episode_lengths[index] = 0
            out_infos.append(info)
        return self._augment(observations), sparse_rewards, dones, out_infos


class SafeHerReplayBuffer(ReplayBuffer):
    """Replay buffer with future HER filtered by route and safety validity.

    Storage remains a regular SB3 replay buffer so standard SAC serialization
    and sampling continue to work.  HER goals are route-completion scalars;
    candidate states must be on the correct route, finite, inside the vessel,
    and above ``her_safe_margin_m`` of catheter-surface clearance.
    """

    def __init__(
        self,
        *args,
        her_ratio: float = 0.5,
        her_safe_margin_m: float = 0.0005,
        goal_tolerance: float = 0.01,
        future_short_fraction: float = 0.25,
        future_medium_fraction: float = 0.35,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.her_ratio = float(np.clip(her_ratio, 0.0, 1.0))
        self.original_ratio = 1.0 - self.her_ratio
        self.her_safe_margin_m = float(max(0.0, her_safe_margin_m))
        self.goal_tolerance = float(max(1e-6, goal_tolerance))
        self.future_short_fraction = float(np.clip(future_short_fraction, 0.0, 1.0))
        self.future_medium_fraction = float(np.clip(future_medium_fraction, 0.0, 1.0))
        self.achieved_goals = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.safe_flags = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.episode_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.episode_steps = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self._episode_id = np.zeros(self.n_envs, dtype=np.int64)
        self._episode_step = np.zeros(self.n_envs, dtype=np.int64)
        self._episode_safe = np.ones(self.n_envs, dtype=np.bool_)
        self.safety_penalties = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.success_flags = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.her_stats = {
            "samples": 0,
            "her_samples": 0,
            "safe_candidates": 0,
            "rejected_candidates": 0,
            "future_short": 0,
            "future_medium": 0,
            "future_far": 0,
        }

    @staticmethod
    def _info_float(info: Dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            value = float(info.get(key, default))
            return value if np.isfinite(value) else float(default)
        except (TypeError, ValueError):
            return float(default)

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        slot = int(self.pos)
        infos = list(infos or [{} for _ in range(self.n_envs)])
        for env_index in range(self.n_envs):
            info = dict(infos[env_index] or {})
            self.achieved_goals[slot, env_index, 0] = np.clip(
                self._info_float(info, "achieved_goal", self._info_float(info, "route_progress_ratio")), 0.0, 1.0
            )
            clearance = self._info_float(info, "sdf_body_min_surface_clearance", np.inf)
            valid_clearance = np.isinf(clearance) or clearance >= self.her_safe_margin_m
            frame_safe = bool(info.get("her_safe", False)) and valid_clearance
            # Safe HER uses a prefix, not an isolated apparently-valid frame:
            # once an episode enters a wrong branch, invalid projection, or
            # unsafe clearance region, all later states in that episode are
            # excluded even if the geometry subsequently recovers.
            self._episode_safe[env_index] = bool(self._episode_safe[env_index] and frame_safe)
            self.safe_flags[slot, env_index] = self._episode_safe[env_index]
            self.safety_penalties[slot, env_index] = np.float32(
                -0.05 * np.clip(self._info_float(info, "sdf_body_warning_feature"), 0.0, 1.0)
                -0.05 * np.clip(self._info_float(info, "off_target_branch_feature"), 0.0, 1.0)
            )
            self.success_flags[slot, env_index] = bool(info.get("done_by_target", False))
            self.episode_ids[slot, env_index] = self._episode_id[env_index]
            self.episode_steps[slot, env_index] = self._episode_step[env_index]
            self._episode_step[env_index] += 1
            if bool(np.asarray(done).reshape(-1)[env_index]):
                self._episode_id[env_index] += 1
                self._episode_step[env_index] = 0
                self._episode_safe[env_index] = True
        super().add(obs, next_obs, action, reward, done, infos)

    def _valid_indices(self, env_index: int, episode_id: int, after_step: int):
        limit = self.buffer_size if self.full else self.pos
        if limit <= 0:
            return np.empty(0, dtype=np.int64)
        mask = (
            (self.episode_ids[:limit, env_index] == episode_id)
            & (self.episode_steps[:limit, env_index] > after_step)
            & self.safe_flags[:limit, env_index]
        )
        return np.flatnonzero(mask)

    def _sample_future_goal(self, env_index: int, episode_id: int, current_step: int):
        candidates = self._valid_indices(env_index, episode_id, current_step)
        if candidates.size == 0:
            self.her_stats["rejected_candidates"] += 1
            return None
        self.her_stats["safe_candidates"] += int(candidates.size)
        # Short/medium/far future mixture; far candidates are biased toward the
        # maximum safe progress of this episode.
        draw = np.random.random()
        if draw < self.future_short_fraction:
            subset = candidates[: max(1, int(np.ceil(candidates.size * 0.33)))]
            self.her_stats["future_short"] += 1
        elif draw < self.future_short_fraction + self.future_medium_fraction:
            lo = max(0, int(candidates.size * 0.25))
            hi = max(lo + 1, int(np.ceil(candidates.size * 0.75)))
            subset = candidates[lo:hi]
            self.her_stats["future_medium"] += 1
        else:
            subset = candidates[-max(1, int(np.ceil(candidates.size * 0.50))) :]
            self.her_stats["future_far"] += 1
        index = int(np.random.choice(subset))
        return float(self.achieved_goals[index, env_index, 0])

    def _sparse_reward(self, next_achieved, goal, safety_penalty):
        reached = bool(float(next_achieved) + self.goal_tolerance >= float(goal))
        return np.float32((0.0 if reached else -1.0) + float(safety_penalty))

    def sample(self, batch_size: int, env: Optional[Any] = None) -> ReplayBufferSamples:
        if self.n_envs <= 0:
            raise ValueError("SafeHerReplayBuffer requires at least one environment.")
        upper = self.buffer_size if self.full else self.pos
        if upper <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        batch_inds = np.random.randint(0, upper, size=batch_size)
        env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        observations = self.observations[batch_inds, env_indices].copy()
        next_observations = self.next_observations[batch_inds, env_indices].copy()
        actions = self.actions[batch_inds, env_indices].copy()
        rewards = self.rewards[batch_inds, env_indices].reshape(-1).copy()
        dones = self.dones[batch_inds, env_indices].reshape(-1).copy()
        relabel = np.random.random(batch_size) < self.her_ratio
        desired_goals = observations[:, -1].copy()
        for row in np.flatnonzero(relabel):
            current = int(batch_inds[row])
            env_index = int(env_indices[row])
            goal = self._sample_future_goal(
                env_index,
                int(self.episode_ids[current, env_index]),
                int(self.episode_steps[current, env_index]),
            )
            if goal is None:
                relabel[row] = False
                continue
            desired_goals[row] = np.clip(goal, 0.0, 1.0)
            next_achieved = float(self.achieved_goals[current, env_index, 0])
            rewards[row] = self._sparse_reward(
                next_achieved, desired_goals[row], self.safety_penalties[current, env_index]
            )
            dones[row] = float(next_achieved + self.goal_tolerance >= desired_goals[row])
        observations[:, -1] = desired_goals
        next_observations[:, -1] = desired_goals
        # Original-goal samples use the independent sparse reward too; the
        # stored V11 reward is deliberately never reused by this branch.
        original = ~relabel
        if np.any(original):
            original_success = self.success_flags[batch_inds[original], env_indices[original]]
            original_penalty = self.safety_penalties[batch_inds[original], env_indices[original]]
            rewards[original] = np.where(original_success, 0.0, -1.0).astype(np.float32)
            rewards[original] += original_penalty
        self.her_stats["samples"] += int(batch_size)
        self.her_stats["her_samples"] += int(np.count_nonzero(relabel))
        return ReplayBufferSamples(
            observations=self.to_torch(observations),
            actions=self.to_torch(actions),
            next_observations=self.to_torch(next_observations),
            dones=self.to_torch(dones.reshape(-1, 1)),
            rewards=self.to_torch(rewards.reshape(-1, 1)),
        )

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Gym goal API compatible sparse reward helper for diagnostics."""
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired = np.asarray(desired_goal, dtype=np.float32)
        return np.where(achieved[..., 0] + self.goal_tolerance >= desired[..., 0], 0.0, -1.0)
