"""Algorithm-independent validation on unseen artificial vessel assets."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import random
import torch as th


STL_CANDIDATES = ("Segmentation.stl",)
CENTERLINE_CANDIDATES = ("Centerline model.vtk", "centerline.vtk")


@contextmanager
def _preserve_rng_state():
    """Keep validation seeds from perturbing subsequent training rollouts."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = th.random.get_rng_state()
    cuda_state = th.cuda.get_rng_state_all() if th.cuda.is_available() else None
    npu_api = getattr(th, "npu", None)
    npu_state = None
    if npu_api is not None:
        try:
            if npu_api.is_available() and hasattr(npu_api, "get_rng_state_all"):
                npu_state = npu_api.get_rng_state_all()
        except Exception:
            npu_state = None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        th.random.set_rng_state(torch_state)
        if cuda_state is not None:
            th.cuda.set_rng_state_all(cuda_state)
        if npu_state is not None and hasattr(npu_api, "set_rng_state_all"):
            npu_api.set_rng_state_all(npu_state)


@dataclass(frozen=True)
class ValidationEpisodeResult:
    vessel_id: str
    episode_index: int
    seed: int
    success: bool
    terminal_reason: str
    steps: int
    reward: float


@dataclass(frozen=True)
class ValidationResult:
    valid_vessels: int
    valid_episodes: int
    valid_success_count: int
    valid_success_rate: float
    episodes: Sequence[ValidationEpisodeResult]


def discover_validation_vessels(
    valid_dir: Path,
    expected_vessels: int = 5,
) -> List[str]:
    """Return valid vessel directory names or fail without changing protocol."""

    valid_dir = Path(valid_dir).expanduser().resolve()
    discovered = []
    if valid_dir.is_dir():
        for vessel_dir in sorted(path for path in valid_dir.iterdir() if path.is_dir()):
            stl_ok = any((vessel_dir / name).is_file() for name in STL_CANDIDATES)
            stl_ok = stl_ok or (vessel_dir / f"{vessel_dir.name}.stl").is_file()
            centerline_ok = any(
                (vessel_dir / name).is_file() for name in CENTERLINE_CANDIDATES
            )
            centerline_ok = centerline_ok or (
                vessel_dir / f"{vessel_dir.name}_centerline.vtk"
            ).is_file()
            if stl_ok and centerline_ok:
                discovered.append(vessel_dir.name)

    if len(discovered) != int(expected_vessels):
        raise RuntimeError(
            "Validation vessel protocol is not ready: "
            f"found {len(discovered)} validation vessels, expected {expected_vessels}. "
            f"Place each vessel's STL and centerline VTK under {valid_dir}. "
            "Use --skip-validation only for an explicit code smoke test."
        )
    return discovered


def evaluate_policy(
    vessel_ids: Sequence[str],
    env_factory: Callable[[str], object],
    deterministic_action: Callable[[np.ndarray], np.ndarray],
    episodes_per_vessel: int = 2,
    max_episode_steps: int = 4096,
    base_seed: int = 100_000,
    task_rank: int = 0,
    task_world_size: int = 1,
) -> ValidationResult:
    """Evaluate every vessel with fixed seeds and no training side effects.

    ``env_factory`` must return a separate single-environment Gym or SB3 VecEnv.
    ``deterministic_action`` is the only policy-specific adapter.
    """

    results: List[ValidationEpisodeResult] = []
    with _preserve_rng_state(), th.no_grad():
        task_rank = int(task_rank)
        task_world_size = max(1, int(task_world_size))
        for vessel_index, vessel_id in enumerate(vessel_ids):
            local_episode_indices = [
                episode_index
                for episode_index in range(int(episodes_per_vessel))
                if (
                    vessel_index * int(episodes_per_vessel) + episode_index
                )
                % task_world_size
                == task_rank
            ]
            if not local_episode_indices:
                continue
            env = env_factory(str(vessel_id))
            try:
                for episode_index in local_episode_indices:
                    seed = int(base_seed) + vessel_index * int(episodes_per_vessel) + episode_index
                    try:
                        reset_result = env.reset(seed=seed)
                    except TypeError:
                        if hasattr(env, "seed"):
                            env.seed(seed)
                        reset_result = env.reset()
                    observation = (
                        reset_result[0]
                        if isinstance(reset_result, tuple) and len(reset_result) == 2
                        else reset_result
                    )
                    total_reward = 0.0
                    final_info = {}
                    step_count = 0
                    done = False
                    for step_count in range(1, int(max_episode_steps) + 1):
                        action = deterministic_action(observation)
                        step_result = env.step(action)
                        if len(step_result) == 5:
                            observation, reward, terminated, truncated, info = step_result
                            done = bool(terminated or truncated)
                            final_info = info
                        else:
                            observation, reward, dones, infos = step_result
                            done = bool(np.asarray(dones).reshape(-1)[0])
                            final_info = infos[0] if isinstance(infos, (list, tuple)) else infos
                        total_reward += float(np.asarray(reward).reshape(-1)[0])
                        if done:
                            break
                    success = bool(done and final_info.get("done_by_target", False))
                    terminal_reason = (
                        str(final_info.get("terminal_reason", "other"))
                        if done
                        else "timeout"
                    )
                    results.append(
                        ValidationEpisodeResult(
                            vessel_id=str(vessel_id),
                            episode_index=episode_index,
                            seed=seed,
                            success=success,
                            terminal_reason=terminal_reason,
                            steps=step_count,
                            reward=total_reward,
                        )
                    )
            finally:
                env.close()

    success_count = sum(int(item.success) for item in results)
    episode_count = len(results)
    return ValidationResult(
        valid_vessels=len({item.vessel_id for item in results}),
        valid_episodes=episode_count,
        valid_success_count=success_count,
        valid_success_rate=float(success_count / episode_count) if episode_count else 0.0,
        episodes=tuple(results),
    )


def validation_result_to_json(result: ValidationResult) -> str:
    return json.dumps(
        [
            {
                "vessel_id": item.vessel_id,
                "episode_index": item.episode_index,
                "seed": item.seed,
                "success": item.success,
                "terminal_reason": item.terminal_reason,
                "steps": item.steps,
                "reward": item.reward,
            }
            for item in result.episodes
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def merge_validation_json(payloads: Sequence[str]) -> ValidationResult:
    """Merge rank-local episode payloads into the original global result."""

    episodes = []
    for payload in payloads:
        for item in json.loads(payload or "[]"):
            episodes.append(ValidationEpisodeResult(**item))
    episodes.sort(key=lambda item: (item.vessel_id, item.episode_index))
    success_count = sum(int(item.success) for item in episodes)
    episode_count = len(episodes)
    return ValidationResult(
        valid_vessels=len({item.vessel_id for item in episodes}),
        valid_episodes=episode_count,
        valid_success_count=success_count,
        valid_success_rate=float(success_count / episode_count) if episode_count else 0.0,
        episodes=tuple(episodes),
    )
