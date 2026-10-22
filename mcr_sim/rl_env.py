"""Gymnasium environment joining stEVE task logic to a magnetic backend."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Sequence, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .magnetic_controller import MagneticActionController
from .pathfinder import CenterlineRandomAdapter, StEVEBruteForceBFSAdapter
from .config import DEFAULT_CONFIG, ExperimentConfig
from .diagnostics import DiagnosticAccumulator, StepDiagnostics
from .metrics import EpisodeMetricTracker
from .observation import MagneticObservationBuilder
from .reward import StEVEReward
from .task import NavigationTask
from .vessel_bundle import VesselBundle, discover_vessel_bundles


@dataclass
class BackendState:
    """SI state; tracking contains only inserted catheter nodes, tip first."""
    catheter_tracking_m: np.ndarray
    tip_frame_world: np.ndarray
    magnetic_field_world_t: np.ndarray
    inserted_length_m: float
    simulation_error: bool = False
    vessel_end: bool = False
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def tip_m(self) -> np.ndarray:
        return np.asarray(self.catheter_tracking_m, dtype=float).reshape(-1, 3)[0]

    def validate(self) -> None:
        tracking = np.asarray(self.catheter_tracking_m, dtype=float)
        frame = np.asarray(self.tip_frame_world, dtype=float)
        field = np.asarray(self.magnetic_field_world_t, dtype=float)
        if tracking.ndim != 2 or tracking.shape[1] != 3 or len(tracking) == 0:
            raise ValueError("backend tracking must have shape (N, 3), tip first")
        if frame.shape != (3, 3) or not np.all(np.isfinite(frame)) or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(frame), 1.0, atol=1e-5):
            raise ValueError("backend tip frame must be an orthonormal 3x3 matrix")
        if field.shape != (3,) or not np.all(np.isfinite(field)):
            raise ValueError("backend magnetic field must be a finite 3-vector")
        if not np.all(np.isfinite(tracking)):
            raise ValueError("backend tracking contains non-finite coordinates")
        if not np.isfinite(self.inserted_length_m) or self.inserted_length_m < 0:
            raise ValueError("backend insertion length is invalid")
        if self.simulation_error:
            raise ValueError("backend reports a SOFA simulation error")


class MagneticBackend(Protocol):
    def reset(self, bundle: VesselBundle) -> BackendState: ...
    def state(self) -> BackendState: ...
    def physics_substep(self, magnetic_field_world_t: np.ndarray, insertion_delta_m: float) -> StepDiagnostics: ...
    def close(self) -> None: ...


class MagneticNavigationEnv(gym.Env):
    """No-curriculum stEVE Bench-style navigation environment."""
    metadata = {"render_modes": []}

    def __init__(self, backend: MagneticBackend, *, config: ExperimentConfig = DEFAULT_CONFIG, mode: str = "train", bundles: Optional[Sequence[VesselBundle]] = None):
        super().__init__()
        if backend is None:
            raise ValueError("A concrete magnetic physics backend is required")
        config.validate()
        self.backend, self.config, self.mode = backend, config, mode
        all_bundles = tuple(bundles or discover_vessel_bundles(config.vessel_root, split=mode))
        allowed_seeds = set(config.vessel_seeds if mode == "train" else config.eval_vessel_seeds)
        self.bundles: Tuple[VesselBundle, ...] = tuple(b for b in all_bundles if b.seed in allowed_seeds)
        if not self.bundles:
            raise ValueError("No configured vessel seeds are available")
        if set(config.vessel_seeds) & set(config.eval_vessel_seeds):
            raise ValueError("Training and evaluation vessel seeds must be disjoint")
        for bundle in self.bundles:
            if bundle.metadata.get("scale_diameter_array") != [0.85]:
                raise ValueError("Expected ArchVariety 0.85 cache, not adult 1.0 assets: " + str(bundle.directory))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(config.observation.dimension,), dtype=np.float32)
        self.action_controller = MagneticActionController(config.action, config.sofa, config.instrument)
        self.task = NavigationTask(mode, config.task)
        self.reward_model = StEVEReward(config.reward)
        self.metric_tracker = EpisodeMetricTracker(config.sofa.rl_dt_s)
        self.bundle = self.graph = self.target = self.observation_builder = None
        self._last_action = np.zeros(3, dtype=np.float32)
        self._episode_done = True

    def _select_bundle(self, options: dict) -> VesselBundle:
        requested = options.get("vessel_seed")
        if requested is not None:
            matches = [b for b in self.bundles if b.seed == int(requested)]
            if not matches:
                raise ValueError(f"Configured vessel seed is unavailable: {requested}")
            return matches[0]
        return self.bundles[int(self.np_random.integers(0, len(self.bundles)))]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        self.bundle = self._select_bundle(options)
        self.graph = StEVEBruteForceBFSAdapter(self.bundle)
        requested = self.bundle.available_target_branches(self.config.task.requested_target_branches)
        branch = options.get("target_branch")
        if branch is not None:
            if branch not in requested:
                raise ValueError(f"Requested target branch is unavailable: {branch}")
            requested = (branch,)
        sampler = CenterlineRandomAdapter(self.bundle, requested)
        target_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.target = sampler.reset(target_seed)
        state = self.backend.reset(self.bundle)
        state.validate()
        self._last_valid_state = state
        self._last_action = np.zeros(3, dtype=np.float32)
        self.observation_builder = MagneticObservationBuilder(
            self.bundle, self.config.observation,
            insertion_length_scale_m=self.config.instrument.total_length_m,
            insertion_step_scale_m=self.config.action.max_insertion_per_rl_step_m,
        )
        path_mm = self.graph.path_length_mm(state.tip_m * 1000.0, self.target.coordinates_mm)
        self.reward_model.reset(path_mm)
        self.task.reset()
        parts = self._observation(state)
        self._last_observation = parts.flatten().copy()
        self.metric_tracker.reset(state.tip_m, path_mm)
        self._episode_done = False
        info = {
            "vessel_seed": self.bundle.seed,
            "target_branch": self.target.branch,
            "target_m": self.target.coordinates_m.copy(),
            "path_length_mm": path_mm,
            "available_target_branches": requested,
            "rl_dt_s": self.config.sofa.rl_dt_s,
            "insertion_point_m": self.bundle.insertion_point_m.copy(),
            "insertion_direction": self.bundle.insertion_direction.copy(),
        }
        return parts.flatten(), info

    def step(self, action):
        if self._episode_done:
            raise RuntimeError("reset() is required before step()")
        try:
            before = self.backend.state()
            before.validate()
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            return self._failed_transition(repr(exc), 0, StepDiagnostics(simulation_error=True))
        plan = self.action_controller.plan(action, before.magnetic_field_world_t, before.tip_frame_world)
        diagnostics = DiagnosticAccumulator()
        substeps = 0
        for field_t, insertion_m in zip(plan.field_substeps_world_t, plan.insertion_substeps_m):
            substeps += 1
            try:
                sample = self.backend.physics_substep(field_t, insertion_m)
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                diag = diagnostics.finish()
                return self._failed_transition(repr(exc), substeps, diag)
            diagnostics.add(sample)
            if sample.simulation_error:
                break
        diag = diagnostics.finish()
        if diag.simulation_error:
            return self._failed_transition("SOFA substep reported numerical failure", substeps, diag)
        try:
            state = self.backend.state()
            state.validate()
            executed_insertion = state.inserted_length_m - before.inserted_length_m
            self._last_action = plan.executed_action.astype(np.float32).copy()
            self._last_action[:2] *= substeps / self.config.sofa.physics_substeps
            self._last_action[2] = np.clip(
                executed_insertion / self.config.action.max_insertion_per_rl_step_m, -1.0, 1.0)
            # Build with the time of the completed action before committing task
            # or reward state, so invalid geometry cannot advance either one.
            parts = self._observation(
                state, tip_displacement_world_m=state.tip_m - before.tip_m,
                last_insertion_m=executed_insertion, completed_steps=self.task.steps + 1,
            )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            return self._failed_transition(repr(exc), substeps, diag)
        path_mm = self.graph.path_length_mm(state.tip_m * 1000.0, self.target.coordinates_mm)
        outcome = self.task.step(
            state.tip_m, self.target.coordinates_m,
            vessel_end=bool(state.vessel_end or self.graph.is_at_vessel_end(state.tip_m * 1000.0)),
            simulation_error=False,
        )
        reward = self.reward_model.step(path_mm, outcome.success)
        observation = parts.flatten()
        self._last_valid_state = state
        self._last_observation = observation.copy()
        self.metric_tracker.step(state.tip_m, executed_insertion)
        self._episode_done = outcome.terminated or outcome.truncated
        info = {
            "vessel_seed": self.bundle.seed,
            "target_branch": self.target.branch,
            "path_length_mm": path_mm,
            "distance_to_target_m": outcome.distance_to_target_m,
            "termination_reason": outcome.reason,
            "success": outcome.success,
            "reward_terms": {"target": reward.target, "path_delta": reward.path_delta, "step": reward.step},
            "physics_substeps_attempted": substeps,
            "state_error": None,
            "observation_valid": True,
            "transition_valid": True,
            "bootstrap_allowed": not outcome.terminated,
            "raw_action": plan.raw_action.copy(),
            "quantized_action": plan.quantized_action.copy(),
            "executed_action": plan.executed_action.copy(),
            "requested_insertion_m": plan.requested_insertion_m,
            "executed_insertion_m": executed_insertion,
            "field_rotation_command_deg": plan.magnetic_rotation_command_deg,
            "field_rotation_executed_deg": plan.magnetic_rotation_executed_deg,
            **diag.as_info(),
        }
        if self._episode_done:
            info["episode_metrics"] = self.metric_tracker.finish(
                success=outcome.success, final_path_length_mm=path_mm, reason=outcome.reason)
        return observation, reward.total, outcome.terminated, outcome.truncated, info

    def _failed_transition(self, error, substeps, diagnostics):
        # Gym still needs a finite return value. This is explicitly a stale
        # placeholder, never a trustworthy next state or a replay transition.
        state = self._last_valid_state
        path_mm = self.graph.path_length_mm(state.tip_m * 1000.0, self.target.coordinates_mm)
        outcome = self.task.step(state.tip_m, self.target.coordinates_m, simulation_error=True)
        reward = self.reward_model.step(path_mm, False)
        diagnostics.simulation_error = True
        self._episode_done = True
        info = {
            "vessel_seed": self.bundle.seed, "target_branch": self.target.branch,
            "path_length_mm": path_mm, "distance_to_target_m": outcome.distance_to_target_m,
            "termination_reason": outcome.reason, "success": False,
            "reward_terms": {"target": reward.target, "path_delta": reward.path_delta, "step": reward.step},
            "physics_substeps_attempted": substeps, "state_error": error,
            "observation_valid": False, "transition_valid": False, "bootstrap_allowed": False,
            "executed_insertion_m": None,
            "episode_metrics": self.metric_tracker.finish(success=False, final_path_length_mm=path_mm, reason=outcome.reason),
            **diagnostics.as_info(),
        }
        return self._last_observation.copy(), reward.total, outcome.terminated, outcome.truncated, info

    def _observation(self, state, *, tip_displacement_world_m=None, last_insertion_m=0.0, completed_steps=None):
        # Read the same BFS path without changing its progress/reward semantics.
        path = self.graph.shortest_path_mm(state.tip_m * 1000.0, self.target.coordinates_mm)
        return self.observation_builder.build(
            state.catheter_tracking_m, self.target.coordinates_m, self._last_action,
            state.magnetic_field_world_t, state.tip_frame_world,
            path=path,
            remaining_time_ratio=max(0.0, 1.0 - (self.task.steps if completed_steps is None else completed_steps) / self.task.max_steps),
            inserted_length_m=state.inserted_length_m,
            tip_displacement_world_m=(np.zeros(3) if tip_displacement_world_m is None else tip_displacement_world_m),
            last_insertion_m=last_insertion_m,
        )

    def close(self):
        self.backend.close()
