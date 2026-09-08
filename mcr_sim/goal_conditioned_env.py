"""Goal-conditioned wrappers and replay storage for the isolated Goal-SAC route.

The baseline :class:`MCREnv` is intentionally not modified. This module adds
achieved/desired route fractions and reconditions the remaining distance.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnvWrapper
from .goal_contract import GoalReward, condition_observation, relabel_observation


class GoalConditionedVecEnv(VecEnvWrapper):
    """Real goal=1 retains physical success; virtual interior goals live in replay."""

    def __init__(
        self,
        venv,
        final_goal: float = 1.0,
        step_cost: float = 0.005,
        safety_weight: float = 0.002,
        action_smoothness_weight: float = 0.0001,
        failure_terminal_penalty: float = 120.0,
        timeout_penalty: float = 120.0,
        out_of_vessel_penalty: float = 150.0,
        non_finite_penalty: float = 200.0,
        max_episode_steps: int = 2048,
        gamma: float = 0.999,
        potential_scale: float = 10.0,
        success_bonus: float = 100.0,
    ):
        super().__init__(venv)
        if getattr(venv.observation_space, "shape", None) is None:
            raise ValueError("GoalConditionedVecEnv requires a flat state observation.")
        self.base_observation_dim = int(venv.observation_space.shape[0])
        self.final_goal = float(np.clip(final_goal, 0.0, 1.0))
        if self.final_goal != 1.0:
            raise ValueError("Rollout must use the real endpoint (final_goal=1)")
        self.step_cost = float(max(0.0, step_cost))
        self.safety_weight = float(max(0.0, safety_weight))
        self.action_smoothness_weight = float(max(0.0, action_smoothness_weight))
        self.failure_terminal_penalty = float(max(0.0, failure_terminal_penalty))
        self.timeout_penalty = float(max(0.0, timeout_penalty))
        self.out_of_vessel_penalty = float(max(0.0, out_of_vessel_penalty))
        self.non_finite_penalty = float(max(0.0, non_finite_penalty))
        self.max_episode_steps = max(1, int(max_episode_steps))
        self.reward_contract = GoalReward(gamma, step_cost, potential_scale, success_bonus,
                                         failure_terminal_penalty)
        low = np.concatenate(
            [np.asarray(venv.observation_space.low, dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)]
        )
        high = np.concatenate(
            [np.asarray(venv.observation_space.high, dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)]
        )
        low[9], high[9] = -1.0, 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._achieved = np.zeros(self.num_envs, dtype=np.float32)
        self.desired_goals = np.full((self.num_envs, 1), self.final_goal, dtype=np.float32)
        self._goal_episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._goal_episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._goal_progress_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._goal_terminal_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._goal_safety_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._goal_step_returns = np.zeros(self.num_envs, dtype=np.float64)
        action_shape = tuple(getattr(self.action_space, "shape", ()) or ())
        if not action_shape:
            raise ValueError("GoalConditionedVecEnv requires a continuous action space")
        self._previous_actions = np.zeros((self.num_envs,) + action_shape, dtype=np.float32)
        self._pending_actions = self._previous_actions.copy()

    def _augment(self, observations):
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        return condition_observation(obs, self._achieved, self.desired_goals[:, 0])

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
        self._goal_progress_returns.fill(0.0)
        self._goal_terminal_returns.fill(0.0)
        self._goal_safety_returns.fill(0.0)
        self._goal_step_returns.fill(0.0)
        self._previous_actions.fill(0.0)
        self._pending_actions.fill(0.0)
        observations = self.venv.reset()
        self._achieved[:] = self.venv.get_attr("current_route_progress_ratio")
        return self._augment(observations)

    def step_async(self, actions):
        pending = np.asarray(actions, dtype=np.float32)
        if pending.shape != self._previous_actions.shape:
            raise ValueError(
                f"Expected actions {self._previous_actions.shape}, got {pending.shape}"
            )
        self._pending_actions = pending.copy()
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        out_infos = []
        sparse_rewards = np.empty(self.num_envs, dtype=np.float32)
        for index, info in enumerate(infos):
            info = dict(info or {})
            raw_achieved = float(info.get("route_progress_ratio", 0.0))
            achieved = float(np.clip(raw_achieved, 0.0, 1.0)) if np.isfinite(raw_achieved) else float(self._achieved[index])
            previous = float(self._achieved[index])
            if "terminal_observation" in info and info["terminal_observation"] is not None:
                terminal = np.asarray(info["terminal_observation"], dtype=np.float32).reshape(-1)
                if terminal.shape[0] == self.base_observation_dim:
                    info["terminal_observation"] = condition_observation(
                        terminal, achieved, self.desired_goals[index, 0])
            # These fields are consumed by SafeHerReplayBuffer and are also
            # useful in the standalone CSV/log diagnostics.
            info["original_desired_goal"] = float(self.desired_goals[index, 0])
            info["achieved_goal"] = achieved
            info["her_safe"] = bool(
                not info.get("out_of_vessel", False)
                and not info.get("wrong_branch", False)
                and not info.get("done_by_non_finite", False)
                and not info.get("route_projection_jump_rejected", False)
                and np.isfinite(raw_achieved)
            )
            wall_risk = self._finite_risk(info, "sdf_body_warning_feature")
            branch_risk = self._finite_risk(info, "off_target_branch_feature")
            safety_penalty = -self.safety_weight * (wall_risk + branch_risk)
            action_delta = self._pending_actions[index] - self._previous_actions[index]
            smoothness_penalty = -self.action_smoothness_weight * float(
                np.mean(np.square(action_delta, dtype=np.float32))
            )
            auxiliary_penalty = safety_penalty + smoothness_penalty
            success = bool(info.get("done_by_target", False))
            terminal = bool(dones[index])
            if success:
                terminal_reward = self.reward_contract.success_bonus
            elif bool(info.get("done_by_non_finite", False)):
                terminal_reward = -self.non_finite_penalty
            elif bool(info.get("done_by_out_of_vessel", False) or info.get("out_of_vessel", False)):
                terminal_reward = -self.out_of_vessel_penalty
            else:
                terminal_reward = -self.timeout_penalty
            base_reward, progress_reward, auxiliary_reward = self.reward_contract.terms(
                previous, achieved, self.final_goal, success, terminal,
                auxiliary_penalty, terminal_reward,
            )
            sparse_rewards[index] = np.float32(
                base_reward + progress_reward + auxiliary_reward
            )
            self._achieved[index] = achieved
            self._previous_actions[index] = self._pending_actions[index]
            # This task has an observed finite horizon (state[10]). A timeout
            # is a task failure, not an artificial training rollout truncation.
            info["TimeLimit.truncated"] = False
            self._goal_episode_returns[index] += float(sparse_rewards[index])
            self._goal_episode_lengths[index] += 1
            self._goal_progress_returns[index] += float(progress_reward)
            self._goal_safety_returns[index] += float(auxiliary_reward)
            if terminal:
                self._goal_terminal_returns[index] += float(base_reward)
            else:
                self._goal_step_returns[index] += float(base_reward)
            info["goal_sparse_reward"] = float(sparse_rewards[index])
            info["goal_reward"] = float(sparse_rewards[index])
            info["goal_safety_penalty"] = float(safety_penalty)
            info["goal_action_smoothness_penalty"] = float(smoothness_penalty)
            info["goal_auxiliary_penalty"] = float(auxiliary_penalty)
            info["goal_reward_progress"] = float(progress_reward)
            info["goal_reward_terminal"] = float(base_reward) if terminal else 0.0
            info["goal_reward_step"] = 0.0 if terminal else float(base_reward)
            info["goal_terminal_base_reward"] = float(terminal_reward) if terminal else 0.0
            if dones[index]:
                # The next reset happens inside VecEnv.step_wait.  Keep the
                # original final goal in this transition's info.
                self.desired_goals[index, 0] = self.final_goal
                episode = dict(info.get("episode", {}))
                episode["r"] = float(self._goal_episode_returns[index])
                episode["l"] = int(self._goal_episode_lengths[index])
                info["episode"] = episode
                info["episode_goal_reward_progress"] = float(self._goal_progress_returns[index])
                info["episode_goal_reward_terminal"] = float(self._goal_terminal_returns[index])
                info["episode_goal_reward_safety"] = float(self._goal_safety_returns[index])
                info["episode_goal_reward_step"] = float(self._goal_step_returns[index])
                info["episode_goal_reward_total_components"] = float(
                    self._goal_progress_returns[index]
                    + self._goal_terminal_returns[index]
                    + self._goal_safety_returns[index]
                    + self._goal_step_returns[index]
                )
                self._goal_episode_returns[index] = 0.0
                self._goal_episode_lengths[index] = 0
                self._goal_progress_returns[index] = 0.0
                self._goal_terminal_returns[index] = 0.0
                self._goal_safety_returns[index] = 0.0
                self._goal_step_returns[index] = 0.0
                self._achieved[index] = self.venv.get_attr(
                    "current_route_progress_ratio", indices=index)[0]
                self._previous_actions[index].fill(0.0)
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
        her_ratio: float = 0.0,
        her_safe_margin_m: float = 0.0005,
        goal_tolerance: float = 0.01,
        min_goal_advance: float = 0.0,
        future_short_fraction: float = 0.25,
        future_medium_fraction: float = 0.35,
        step_cost: float = 0.005,
        gamma: float = 0.999,
        potential_scale: float = 10.0,
        success_bonus: float = 100.0,
        failure_terminal_penalty: float = 120.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.optimize_memory_usage:
            raise ValueError("Safe HER requires separate next-observation storage")
        self.reward_contract = GoalReward(gamma, step_cost, potential_scale, success_bonus,
                                         failure_terminal_penalty)
        self.her_ratio = float(np.clip(her_ratio, 0.0, 1.0))
        self.original_ratio = 1.0 - self.her_ratio
        self.her_safe_margin_m = float(max(0.0, her_safe_margin_m))
        self.goal_tolerance = float(max(1e-6, goal_tolerance))
        self.min_goal_advance = float(max(0.0, min_goal_advance))
        self.future_short_fraction = float(np.clip(future_short_fraction, 0.0, 1.0))
        self.future_medium_fraction = float(np.clip(future_medium_fraction, 0.0, 1.0))
        self.step_cost = float(max(0.0, step_cost))
        self.achieved_goals = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.safe_flags = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.episode_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.episode_steps = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self._episode_id = np.zeros(self.n_envs, dtype=np.int64)
        self._episode_step = np.zeros(self.n_envs, dtype=np.int64)
        self._episode_safe = np.ones(self.n_envs, dtype=np.bool_)
        # (env, episode) -> [steps, replay slots, monotonically increasing
        # achieved progresses].  Only genuine progress milestones are indexed:
        # a later timestamp at the same position must never become a HER goal.
        self._safe_episode_milestones = {}
        self._milestone_flags = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.auxiliary_penalties = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.success_flags = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.terminal_base_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.her_stats = {
            "samples": 0,
            "her_samples": 0,
            "her_successes": 0,
            "safe_candidates": 0,
            "rejected_candidates": 0,
            "future_short": 0,
            "future_medium": 0,
            "future_far": 0,
            "insufficient_progress_candidates": 0,
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
            if self.full and self._milestone_flags[slot, env_index]:
                old_key = (env_index, int(self.episode_ids[slot, env_index]))
                old_entry = self._safe_episode_milestones.get(old_key)
                if old_entry is not None:
                    steps, slots, progresses = old_entry
                    try:
                        old_position = slots.index(slot)
                    except ValueError:
                        old_position = -1
                    if old_position >= 0:
                        steps.pop(old_position)
                        slots.pop(old_position)
                        progresses.pop(old_position)
                    if not slots:
                        self._safe_episode_milestones.pop(old_key, None)
            self._milestone_flags[slot, env_index] = False
            info = dict(infos[env_index] or {})
            self.achieved_goals[slot, env_index, 0] = np.clip(
                self._info_float(info, "achieved_goal", self._info_float(info, "route_progress_ratio")), 0.0, 1.0
            )
            clearance = self._info_float(info, "sdf_body_min_surface_clearance", -np.inf)
            valid_clearance = np.isfinite(clearance) and clearance >= self.her_safe_margin_m
            frame_safe = bool(info.get("her_safe", False)) and valid_clearance
            # Safe HER uses a prefix, not an isolated apparently-valid frame:
            # once an episode enters a wrong branch, invalid projection, or
            # unsafe clearance region, all later states in that episode are
            # excluded even if the geometry subsequently recovers.
            self._episode_safe[env_index] = bool(
                self.her_ratio > 0.0 and self._episode_safe[env_index] and frame_safe
            )
            self.safe_flags[slot, env_index] = self._episode_safe[env_index]
            self.auxiliary_penalties[slot, env_index] = np.float32(
                self._info_float(info, "goal_auxiliary_penalty", 0.0)
            )
            self.success_flags[slot, env_index] = bool(info.get("done_by_target", False))
            self.terminal_base_rewards[slot, env_index] = np.float32(
                self._info_float(info, "goal_terminal_base_reward", 0.0)
            )
            self.episode_ids[slot, env_index] = self._episode_id[env_index]
            self.episode_steps[slot, env_index] = self._episode_step[env_index]
            if self.safe_flags[slot, env_index]:
                key = (env_index, int(self._episode_id[env_index]))
                entry = self._safe_episode_milestones.get(key)
                if entry is None:
                    entry = [[], [], []]
                    self._safe_episode_milestones[key] = entry
                progress = float(self.achieved_goals[slot, env_index, 0])
                # A tiny epsilon removes identical/stalled frames while the
                # stronger min_goal_advance filter is applied at sampling.
                if not entry[2] or progress > entry[2][-1] + 1e-6:
                    entry[0].append(int(self._episode_step[env_index]))
                    entry[1].append(slot)
                    entry[2].append(progress)
                    self._milestone_flags[slot, env_index] = True
            self._episode_step[env_index] += 1
            if bool(np.asarray(done).reshape(-1)[env_index]):
                self._episode_id[env_index] += 1
                self._episode_step[env_index] = 0
                self._episode_safe[env_index] = True
        super().add(obs, next_obs, action, reward, done, infos)

    def _sample_future_goal(
        self,
        env_index: int,
        episode_id: int,
        current_step: int,
        previous_achieved: float,
    ):
        entry = self._safe_episode_milestones.get((env_index, episode_id))
        if entry is None:
            self.her_stats["rejected_candidates"] += 1
            return None
        steps, slots, progresses = entry
        # Future strategy is inclusive of the sampled transition.  This lets
        # the transition that genuinely reaches a milestone provide HER's
        # sparse success sample.  Later stalled frames are absent from the
        # milestone index and therefore cannot repeat that success.
        time_start = bisect_left(steps, int(current_step))
        progress_start = bisect_left(
            progresses,
            previous_achieved + max(self.goal_tolerance, self.min_goal_advance) + 1e-7,
        )
        start = max(time_start, progress_start)
        # Goal=1 retains the physical 3 mm success definition. HER creates
        # interior route goals only, never a conflicting virtual endpoint.
        end = bisect_left(progresses, 1.0 - self.goal_tolerance)
        candidate_count = end - start
        if candidate_count <= 0:
            self.her_stats["rejected_candidates"] += 1
            self.her_stats["insufficient_progress_candidates"] += 1
            return None
        self.her_stats["safe_candidates"] += int(candidate_count)
        # Short/medium/far future mixture; far candidates are biased toward the
        # maximum safe progress of this episode.
        draw = np.random.random()
        if draw < self.future_short_fraction:
            low = start
            high = start + max(1, int(np.ceil(candidate_count * 0.33)))
            self.her_stats["future_short"] += 1
        elif draw < self.future_short_fraction + self.future_medium_fraction:
            low = start + int(candidate_count * 0.25)
            high = start + max(int(candidate_count * 0.25) + 1, int(np.ceil(candidate_count * 0.75)))
            self.her_stats["future_medium"] += 1
        else:
            low = start + candidate_count - max(1, int(np.ceil(candidate_count * 0.50)))
            high = start + candidate_count
            self.her_stats["future_far"] += 1
        position = int(np.random.randint(low, high))
        return float(progresses[position])

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
                float(observations[row, -2]),
            )
            if goal is None:
                relabel[row] = False
                continue
            desired_goals[row] = np.clip(goal, 0.0, 1.0)
            next_achieved = float(self.achieved_goals[current, env_index, 0])
            her_success = bool(
                next_achieved + self.goal_tolerance >= desired_goals[row]
            ) and bool(self.safe_flags[current, env_index])
            dones[row] = float(bool(dones[row]) or her_success)
            terminal_reward = (
                self.reward_contract.success_bonus if her_success
                else self.terminal_base_rewards[current, env_index]
            )
            rewards[row] = self.reward_contract(
                observations[row, -2], next_achieved, desired_goals[row],
                her_success, bool(dones[row]), self.auxiliary_penalties[current, env_index],
                terminal_reward,
            )
            self.her_stats["her_successes"] += int(her_success)
        observations = relabel_observation(observations, desired_goals)
        next_observations = relabel_observation(next_observations, desired_goals)
        # Original rows keep the wrapper reward; HER uses the same contract.
        self.her_stats["samples"] += int(batch_size)
        self.her_stats["her_samples"] += int(np.count_nonzero(relabel))
        return ReplayBufferSamples(
            observations=self.to_torch(observations),
            actions=self.to_torch(actions),
            next_observations=self.to_torch(next_observations),
            dones=self.to_torch(dones.reshape(-1, 1)),
            rewards=self.to_torch(rewards.reshape(-1, 1)),
        )

    def pop_window_stats(self):
        """Diagnostics cover the last log window, not the entire run."""
        result = self.her_stats.copy()
        for key in self.her_stats:
            self.her_stats[key] = 0
        return result
