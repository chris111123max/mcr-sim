"""Complete-episode replay with topology-safe future-goal relabeling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from mcr_sim.goal_contract import relabel_observation


@dataclass
class Batch:
    obs: np.ndarray
    next_obs: np.ndarray
    previous_actions: np.ndarray
    actions: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    risk_targets: np.ndarray
    her_mask: np.ndarray


def goal_reward(before, after, goal, success, failure, *, progress_scale=20.0,
                success_bonus=20.0, failure_penalty=25.0, step_cost=0.002,
                unsafe=False):
    """Bounded route-potential increment; no credit on an unsafe transition."""
    progress = 0.0 if failure or unsafe else progress_scale * (
        min(float(after), float(goal)) - min(float(before), float(goal))
    )
    return float(progress + success_bonus * bool(success)
                 - failure_penalty * bool(failure) - step_cost)


class TopologyHerReplay:
    def __init__(self, capacity: int, num_envs: int):
        self.capacity = int(capacity)
        self.pending = [[] for _ in range(num_envs)]
        self.episodes = deque()
        self.size = 0
        self.rng = np.random.default_rng()

    def add(self, index: int, obs, action, next_obs, info, done: bool):
        episode = self.pending[index]
        route = (str(info.get("chosen_model", "unknown")),
                 str(info.get("target_route_id", "default")))
        if episode and episode[-1]["route"] != route:
            raise RuntimeError("Route changed inside an episode; refusing to splice HER data")
        before = float(np.clip(obs[-2], 0.0, 1.0))
        after = float(np.clip(info.get("route_progress_ratio", before), 0.0, 1.0))
        unsafe = bool(info.get("done_by_out_of_vessel", False)
                      or info.get("done_by_non_finite", False)
                      or info.get("route_projection_jump_rejected", False)
                      or float(info.get("off_target_branch_feature", 0.0)) > 0.5)
        clearance = float(info.get("sdf_body_min_surface_clearance", np.nan))
        safe = not unsafe and np.isfinite(clearance) and clearance >= 0.0
        episode.append({
            "obs": np.asarray(obs, dtype=np.float32).copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "next_obs": np.asarray(next_obs, dtype=np.float32).copy(),
            "before": before, "after": after, "safe": safe,
            "unsafe": unsafe,
            "failure": bool(info.get("done_by_out_of_vessel", False)
                            or info.get("done_by_non_finite", False)),
            "success": bool(info.get("done_by_target", False)),
            "done": bool(done), "route": route,
            "risk": float(np.clip(max(
                float(info.get("sdf_body_warning_feature", 0.0)),
                float(info.get("off_target_branch_feature", 0.0)),
                float(unsafe)), 0.0, 1.0)),
        })
        if done:
            self.episodes.append(episode)
            self.size += len(episode)
            self.pending[index] = []
            while len(self.episodes) > 1 and self.size > self.capacity:
                self.size -= len(self.episodes.popleft())

    def can_sample(self, sequence_length: int):
        return any(len(ep) >= sequence_length for ep in self.episodes)

    def sample(self, batch_size: int, sequence_length: int, her_ratio: float,
               future_horizon: int, risk_horizon: int, goal_tolerance: float = 0.005):
        eligible = [ep for ep in self.episodes if len(ep) >= sequence_length]
        if not eligible:
            raise RuntimeError("No complete episode long enough for recurrent replay")
        result = {k: [] for k in Batch.__dataclass_fields__}
        for _ in range(batch_size):
            ep = eligible[int(self.rng.integers(len(eligible)))]
            end = int(self.rng.integers(sequence_length - 1, len(ep)))
            start = end - sequence_length + 1
            current = ep[end]
            goal = 1.0
            her = False
            if self.rng.random() < her_ratio and current["safe"]:
                # The entire segment from the sampled action to the future
                # goal must remain safe and on the same selected route.
                futures = []
                high = min(len(ep), end + max(1, future_horizon))
                for j in range(end, high):
                    if not ep[j]["safe"] or ep[j]["route"] != current["route"]:
                        break
                    value = ep[j]["after"]
                    if current["before"] + goal_tolerance < value < 1.0 - goal_tolerance:
                        futures.append(value)
                if futures:
                    goal = float(futures[int(self.rng.integers(len(futures)))])
                    her = True
            sequence = ep[start:end + 1]
            obs = np.stack([relabel_observation(item["obs"], goal) for item in sequence])
            next_obs = np.stack([relabel_observation(item["next_obs"], goal) for item in sequence])
            actions = np.stack([item["action"] for item in sequence])
            previous = np.stack([
                np.zeros_like(actions[0]) if i == 0 else ep[i - 1]["action"]
                for i in range(start, end + 1)
            ])
            virtual_success = bool(her and current["safe"]
                                   and current["after"] >= goal - goal_tolerance)
            success = current["success"] if not her else virtual_success
            failure = current["failure"]
            done = bool(current["done"] or virtual_success)
            reward = goal_reward(current["before"], current["after"], goal,
                                 success, failure, unsafe=current["unsafe"])
            risk_end = min(len(ep), end + max(1, risk_horizon))
            risk = max(item["risk"] for item in ep[end:risk_end])
            for key, value in (
                ("obs", obs), ("next_obs", next_obs),
                ("previous_actions", previous), ("actions", actions),
                ("dones", [float(done)]), ("rewards", [reward]),
                ("risk_targets", [risk]), ("her_mask", [float(her)]),
            ):
                result[key].append(value)
        return Batch(**{key: np.asarray(value, dtype=np.float32)
                        for key, value in result.items()})
