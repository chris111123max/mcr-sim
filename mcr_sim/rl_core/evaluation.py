"""Algorithm-independent validation on unseen artificial vessel assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Sequence

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
    route_completion: float = 0.0
    route_potential: float = 0.0
    final_distance_mm: float = math.inf
    min_distance_mm: float = math.inf
    error: str = ""
    diagnostics: Dict[str, object] = field(default_factory=dict)
    target_route_id: str = "default"


@dataclass(frozen=True)
class ValidationResult:
    valid_vessels: int
    valid_episodes: int
    valid_success_count: int
    valid_success_rate: float
    valid_route_completion_mean: float
    valid_route_potential_mean: float
    valid_final_distance_mm_mean: float
    valid_min_distance_mm_mean: float
    episodes: Sequence[ValidationEpisodeResult]


def _finite_mean(values, default: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float(default)


def _diagnostic_float(info, key: str, scale: float = 1.0) -> float:
    """Read an optional terminal diagnostic without changing evaluation."""

    try:
        value = float(info.get(key, math.nan)) * float(scale)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _terminal_diagnostics(info) -> Dict[str, object]:
    """Extract safety/control evidence from an environment's terminal info."""

    trace = info.get("terminal_diagnostic_trace", ())
    if not isinstance(trace, (list, tuple)):
        trace = ()
    return {
        "route_progress_m": _diagnostic_float(info, "route_progress"),
        "route_projection_segment": int(info.get("route_projection_segment", -1)),
        "route_projection_distance_mm": _diagnostic_float(
            info, "route_projection_distance", 1000.0
        ),
        "route_projection_jump_rejections": int(
            info.get("route_projection_jump_rejections_episode", 0)
        ),
        "centerline_local_radius_mm": _diagnostic_float(
            info, "centerline_local_radius", 1000.0
        ),
        "centerline_safety_ratio": _diagnostic_float(
            info, "centerline_safety_ratio"
        ),
        "centerline_safety_ratio_max_episode": _diagnostic_float(
            info, "centerline_safety_ratio_max_episode"
        ),
        "centerline_safety_margin": _diagnostic_float(
            info, "centerline_safety_margin"
        ),
        "centerline_safety_margin_min_episode": _diagnostic_float(
            info, "centerline_safety_margin_min_episode"
        ),
        "curve_bend_5mm": _diagnostic_float(info, "curve_bend_5mm"),
        "curve_bend_10mm": _diagnostic_float(info, "curve_bend_10mm"),
        "curve_bend_20mm": _diagnostic_float(info, "curve_bend_20mm"),
        "curve_alignment_error_20mm": _diagnostic_float(
            info, "curve_alignment_error_20mm"
        ),
        "sdf_tip_clearance_min_episode_mm": _diagnostic_float(
            info, "sdf_surface_clearance_min_episode", 1000.0
        ),
        "sdf_body_clearance_min_episode_mm": _diagnostic_float(
            info, "sdf_body_surface_clearance_min_episode", 1000.0
        ),
        "sdf_penetration_depth_max_episode_mm": _diagnostic_float(
            info, "sdf_penetration_depth_max_episode", 1000.0
        ),
        "sdf_near_wall_steps_episode": int(
            info.get("sdf_tip_near_wall_steps_episode", 0)
        ),
        "sdf_wall_contact_steps_episode": int(
            info.get("sdf_wall_contact_steps_episode", 0)
        ),
        "insert_action_mean_episode": _diagnostic_float(
            info, "insert_action_mean_episode"
        ),
        "insert_positive_fraction_episode": _diagnostic_float(
            info, "insert_positive_fraction_episode"
        ),
        "insert_negative_fraction_episode": _diagnostic_float(
            info, "insert_negative_fraction_episode"
        ),
        "insert_near_zero_fraction_episode": _diagnostic_float(
            info, "insert_near_zero_fraction_episode"
        ),
        "inserted_length_final_mm": _diagnostic_float(
            info, "inserted_length_final", 1000.0
        ),
        "inserted_length_max_episode_mm": _diagnostic_float(
            info, "inserted_length_max_episode", 1000.0
        ),
        "terminal_trace": tuple(dict(item) for item in trace if isinstance(item, dict)),
    }


def summarize_validation(
    episodes: Sequence[ValidationEpisodeResult],
) -> ValidationResult:
    """Build aggregate metrics shared by local and distributed validation."""

    episodes = tuple(episodes)
    success_count = sum(int(item.success) for item in episodes)
    episode_count = len(episodes)
    return ValidationResult(
        valid_vessels=len({item.vessel_id for item in episodes}),
        valid_episodes=episode_count,
        valid_success_count=success_count,
        valid_success_rate=(
            float(success_count / episode_count) if episode_count else 0.0
        ),
        valid_route_completion_mean=_finite_mean(
            (item.route_completion for item in episodes),
            0.0,
        ),
        valid_route_potential_mean=_finite_mean(
            (item.route_potential for item in episodes),
            0.0,
        ),
        valid_final_distance_mm_mean=_finite_mean(
            (item.final_distance_mm for item in episodes),
            math.inf,
        ),
        valid_min_distance_mm_mean=_finite_mean(
            (item.min_distance_mm for item in episodes),
            math.inf,
        ),
        episodes=episodes,
    )


def validation_selection_key(result: ValidationResult):
    """Return the deterministic lexicographic best-checkpoint score.

    Target success remains primary. Fixed-seed route completion, route
    potential, and final distance only break ties, so a merely closer failure
    can never replace a checkpoint with a higher target success rate.
    """

    final_distance = float(result.valid_final_distance_mm_mean)
    return (
        float(result.valid_success_rate),
        float(result.valid_route_completion_mean),
        float(result.valid_route_potential_mean),
        -final_distance if math.isfinite(final_distance) else -math.inf,
    )


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

    def record_failure(vessel_id, episode_index, seed, step_count, reward, exc):
        error_text = f"{type(exc).__name__}: {exc}"
        print(
            f"[VALID][Rank {task_rank}][Vessel {vessel_id}] "
            f"episode={episode_index} status=failed error={error_text}",
            flush=True,
        )
        results.append(
            ValidationEpisodeResult(
                vessel_id=str(vessel_id),
                episode_index=int(episode_index),
                seed=int(seed),
                success=False,
                terminal_reason="validation_error",
                steps=int(step_count),
                reward=float(reward),
                error=error_text,
            )
        )

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
            print(
                f"[VALID][Rank {task_rank}][Vessel {vessel_id}] "
                f"status=start episodes={local_episode_indices}",
                flush=True,
            )
            env = None
            try:
                env = env_factory(str(vessel_id))
            except Exception as exc:
                for episode_index in local_episode_indices:
                    seed = int(base_seed) + vessel_index * int(episodes_per_vessel) + episode_index
                    record_failure(vessel_id, episode_index, seed, 0, 0.0, exc)
                print(
                    f"[VALID][Rank {task_rank}][Vessel {vessel_id}] "
                    "status=finished_with_errors",
                    flush=True,
                )
                continue

            vessel_result_start = len(results)
            try:
                for episode_index in local_episode_indices:
                    seed = int(base_seed) + vessel_index * int(episodes_per_vessel) + episode_index
                    total_reward = 0.0
                    step_count = 0
                    print(
                        f"[VALID][Rank {task_rank}][Vessel {vessel_id}] "
                        f"episode={episode_index} seed={seed} status=start",
                        flush=True,
                    )
                    try:
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
                        # Recurrent predictors expose an episode-local reset
                        # hook. Plain MLP callables do not, so their validation
                        # behavior is unchanged.
                        reset_action_state = getattr(
                            deterministic_action, "reset", None
                        )
                        if callable(reset_action_state):
                            reset_action_state()
                        final_info = {}
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
                        route_ratio = float(
                            final_info.get("route_progress_ratio", math.nan)
                        )
                        if success:
                            route_completion = 1.0
                        elif math.isfinite(route_ratio):
                            route_completion = float(
                                np.clip(route_ratio, 0.0, 1.0)
                            )
                        else:
                            route_completion = 0.0
                        route_potential = float(
                            np.clip(
                                final_info.get("route_potential", 0.0),
                                0.0,
                                1.0,
                            )
                        )
                        final_distance = float(
                            final_info.get(
                                "final_dist_to_goal",
                                final_info.get("current_dist_to_goal", math.inf),
                            )
                        )
                        if not math.isfinite(final_distance):
                            final_distance = float(
                                final_info.get("current_dist_to_goal", math.inf)
                            )
                        min_distance = float(
                            final_info.get("min_dist_to_goal", math.inf)
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
                                route_completion=route_completion,
                                route_potential=route_potential,
                                final_distance_mm=(
                                    final_distance * 1000.0
                                    if math.isfinite(final_distance)
                                    else math.inf
                                ),
                                min_distance_mm=(
                                    min_distance * 1000.0
                                    if math.isfinite(min_distance)
                                    else math.inf
                                ),
                                diagnostics=_terminal_diagnostics(final_info),
                                target_route_id=str(
                                    final_info.get("target_route_id", "default")
                                ),
                            )
                        )
                        print(
                            f"[VALID][Rank {task_rank}][Vessel {vessel_id}] "
                            f"episode={episode_index} status=finished "
                            f"success={success} terminal_reason={terminal_reason} "
                            f"steps={step_count}",
                            flush=True,
                        )
                    except Exception as exc:
                        record_failure(
                            vessel_id,
                            episode_index,
                            seed,
                            step_count,
                            total_reward,
                            exc,
                        )
            finally:
                try:
                    env.close()
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[VALID][Rank {task_rank}][Vessel {vessel_id}] "
                        f"status=close_failed error={error_text}",
                        flush=True,
                    )
                    # A failed SOFA teardown can poison the next validation
                    # scene. Treat this vessel's local episodes as failed but
                    # keep all ranks alive and able to reach the collectives.
                    for result_index in range(vessel_result_start, len(results)):
                        item = results[result_index]
                        if item.vessel_id == str(vessel_id):
                            results[result_index] = ValidationEpisodeResult(
                                vessel_id=item.vessel_id,
                                episode_index=item.episode_index,
                                seed=item.seed,
                                success=False,
                                terminal_reason="validation_error",
                                steps=item.steps,
                                reward=item.reward,
                                route_completion=item.route_completion,
                                route_potential=item.route_potential,
                                final_distance_mm=item.final_distance_mm,
                                min_distance_mm=item.min_distance_mm,
                                error=error_text,
                                diagnostics=item.diagnostics,
                                target_route_id=item.target_route_id,
                            )
            vessel_errors = any(
                item.error
                for item in results[vessel_result_start:]
                if item.vessel_id == str(vessel_id)
            )
            print(
                f"[VALID][Rank {task_rank}][Vessel {vessel_id}] "
                f"status={'finished_with_errors' if vessel_errors else 'finished'}",
                flush=True,
            )

    return summarize_validation(results)


def evaluate_vector_policy(
    vessel_ids: Sequence[str],
    env_factory: Callable[[str, int], object],
    policy_action: Callable[[np.ndarray], np.ndarray],
    episodes_per_vessel: int = 32,
    max_parallel_envs: int = 32,
    max_episode_steps: int = 4096,
    base_seed: int = 100_000,
) -> ValidationResult:
    """Evaluate one fixed-seed episode per vector slot in parallel.

    Each vessel is evaluated separately so its statistics remain exact.  Up to
    ``max_parallel_envs`` slots run concurrently; larger requested sample counts
    are processed in independent batches.  The function is diagnostic-only and
    never mutates policy parameters.
    """

    requested = int(episodes_per_vessel)
    parallel = max(1, int(max_parallel_envs))
    if requested <= 0:
        raise ValueError("episodes_per_vessel must be positive")
    results = []
    with _preserve_rng_state():
        for vessel_offset, vessel_id in enumerate(vessel_ids):
            completed_for_vessel = 0
            while completed_for_vessel < requested:
                batch_size = min(parallel, requested - completed_for_vessel)
                print(
                    f"[VECTOR EVAL][Vessel {vessel_id}] "
                    f"episodes={completed_for_vessel + 1}-"
                    f"{completed_for_vessel + batch_size}/{requested} "
                    f"parallel_envs={batch_size} status=start",
                    flush=True,
                )
                env = env_factory(str(vessel_id), batch_size)
                batch_seed = (
                    int(base_seed)
                    + vessel_offset * requested
                    + completed_for_vessel
                )
                try:
                    seed_method = getattr(env, "seed", None)
                    if callable(seed_method):
                        seed_method(batch_seed)
                    observation = env.reset()
                    if isinstance(observation, tuple):
                        observation = observation[0]
                    rewards = np.zeros(batch_size, dtype=np.float64)
                    steps = np.zeros(batch_size, dtype=np.int64)
                    active = np.ones(batch_size, dtype=np.bool_)
                    action_sums = None
                    action_abs_sums = None
                    action_abs_max = None
                    hard_limit = max(1, int(max_episode_steps)) + 1
                    for _ in range(hard_limit):
                        action = policy_action(observation)
                        action_values = np.asarray(
                            action, dtype=np.float64
                        ).reshape(batch_size, -1)
                        if action_sums is None:
                            action_sums = np.zeros_like(action_values)
                            action_abs_sums = np.zeros_like(action_values)
                            action_abs_max = np.zeros_like(action_values)
                        action_sums[active] += action_values[active]
                        action_abs_sums[active] += np.abs(action_values[active])
                        action_abs_max[active] = np.maximum(
                            action_abs_max[active], np.abs(action_values[active])
                        )
                        step_result = env.step(action)
                        if len(step_result) == 5:
                            observation, reward, terminated, truncated, infos = step_result
                            dones = np.logical_or(terminated, truncated)
                        else:
                            observation, reward, dones, infos = step_result
                        reward = np.asarray(reward, dtype=np.float64).reshape(-1)
                        dones = np.asarray(dones, dtype=np.bool_).reshape(-1)
                        rewards[active] += reward[active]
                        steps[active] += 1
                        infos = list(infos)
                        for slot in np.flatnonzero(active & dones):
                            info = infos[int(slot)]
                            success = bool(info.get("done_by_target", False))
                            route_ratio = float(info.get("route_progress_ratio", math.nan))
                            route_completion = (
                                1.0
                                if success
                                else float(np.clip(route_ratio, 0.0, 1.0))
                                if math.isfinite(route_ratio)
                                else 0.0
                            )
                            route_potential = float(
                                np.clip(info.get("route_potential", 0.0), 0.0, 1.0)
                            )
                            final_distance = float(
                                info.get(
                                    "final_dist_to_goal",
                                    info.get("current_dist_to_goal", math.inf),
                                )
                            )
                            min_distance = float(info.get("min_dist_to_goal", math.inf))
                            episode_index = completed_for_vessel + int(slot)
                            diagnostics = _terminal_diagnostics(info)
                            action_names = ("rot_n", "rot_b", "insert")
                            action_count = max(1, int(steps[slot]))
                            for action_index, action_name in enumerate(action_names):
                                if action_sums is not None and action_index < action_sums.shape[1]:
                                    diagnostics[f"model_action_{action_name}_mean"] = float(
                                        action_sums[slot, action_index] / action_count
                                    )
                                    diagnostics[f"model_action_{action_name}_abs_mean"] = float(
                                        action_abs_sums[slot, action_index] / action_count
                                    )
                                    diagnostics[f"model_action_{action_name}_abs_max"] = float(
                                        action_abs_max[slot, action_index]
                                    )
                            results.append(
                                ValidationEpisodeResult(
                                    vessel_id=str(vessel_id),
                                    episode_index=episode_index,
                                    seed=batch_seed + int(slot),
                                    success=success,
                                    terminal_reason=str(
                                        info.get("terminal_reason", "other")
                                    ),
                                    steps=int(steps[slot]),
                                    reward=float(rewards[slot]),
                                    route_completion=route_completion,
                                    route_potential=route_potential,
                                    final_distance_mm=(
                                        final_distance * 1000.0
                                        if math.isfinite(final_distance)
                                        else math.inf
                                    ),
                                    min_distance_mm=(
                                        min_distance * 1000.0
                                        if math.isfinite(min_distance)
                                        else math.inf
                                    ),
                                    diagnostics=diagnostics,
                                    target_route_id=str(
                                        info.get("target_route_id", "default")
                                    ),
                                )
                            )
                            active[slot] = False
                        if not np.any(active):
                            break
                    for slot in np.flatnonzero(active):
                        results.append(
                            ValidationEpisodeResult(
                                vessel_id=str(vessel_id),
                                episode_index=completed_for_vessel + int(slot),
                                seed=batch_seed + int(slot),
                                success=False,
                                terminal_reason="evaluation_limit",
                                steps=int(steps[slot]),
                                reward=float(rewards[slot]),
                                error="environment did not terminate within evaluation limit",
                            )
                        )
                finally:
                    env.close()
                completed_for_vessel += batch_size
                print(
                    f"[VECTOR EVAL][Vessel {vessel_id}] "
                    f"completed={completed_for_vessel}/{requested} status=finished",
                    flush=True,
                )
    results.sort(key=lambda item: (item.vessel_id, item.episode_index))
    return summarize_validation(results)


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
                "route_completion": item.route_completion,
                "route_potential": item.route_potential,
                "final_distance_mm": item.final_distance_mm,
                "min_distance_mm": item.min_distance_mm,
                "error": item.error,
                "target_route_id": item.target_route_id,
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
            if "route_completion" not in item:
                item["route_completion"] = item.pop(
                    "waypoint_reached_ratio",
                    0.0,
                )
            episodes.append(ValidationEpisodeResult(**item))
    episodes.sort(key=lambda item: (item.vessel_id, item.episode_index))
    return summarize_validation(episodes)
