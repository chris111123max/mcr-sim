"""SOFA scene and backend for the stEVE x Nelson environment.

The module is importable without SOFA. Runtime imports are intentionally delayed
so pure geometry, reward and controller tests can run in ``mcr_sofa``.
"""
from dataclasses import dataclass
import contextlib
import importlib
import io
from typing import Dict, Optional
import numpy as np

from .config import DEFAULT_CONFIG, ExperimentConfig
from .diagnostics import StepDiagnostics
from .rl_env import BackendState
from .instrument_layout import NelsonInstrumentLayout
from .magnetics import distributed_magnetic_torques, quaternion_xyzw_to_matrix
from .vessel_bundle import VesselBundle


SOFA_COMPONENT_MANIFEST = {
    "animation_loop": "FreeMotionAnimationLoop",
    "pipeline": "CollisionPipeline",
    "broad_phase": "BruteForceBroadPhase",
    "narrow_phase": "BVHNarrowPhase",
    "intersection": "LocalMinDistance",
    "contact_manager": "CollisionResponse",
    "constraint_solver": "LCPConstraintSolver",
    "ode_solver": "EulerImplicitSolver",
    "linear_solver": "BTDLinearSolver",
    "controller": "InterventionalRadiologyController",
    "beam_interpolation": "WireBeamInterpolation",
    "beam_force": "AdaptiveBeamForceFieldAndMass",
}


def component_parameters(config: ExperimentConfig = DEFAULT_CONFIG) -> Dict[str, object]:
    d = config.instrument.discretization
    return {
        "dt_s": config.sofa.dt_s,
        "physics_substeps": config.sofa.physics_substeps,
        "contact_distance_m": config.sofa.contact_distance_m,
        "alarm_distance_m": config.sofa.alarm_distance_m,
        "angle_cone": config.sofa.angle_cone,
        "friction": config.sofa.friction,
        "lcp_tolerance": config.sofa.lcp_tolerance,
        "lcp_max_iterations": config.sofa.lcp_max_iterations,
        "lcp_build": config.sofa.lcp_build,
        "physical_beams": d.total_beams,
        "mechanical_dofs": d.mechanical_dofs,
        "collision_edges": (d.collision_edges_body, d.collision_edges_tip),
        "visual_edges": d.visual_edges,
        "catheter_proximity_m": config.sofa.catheter_proximity_m,
        "vessel_proximity_m": config.sofa.vessel_proximity_m,
    }


def insertion_pose(point_m, direction) -> list:
    direction = np.asarray(direction, dtype=float).reshape(3)
    direction /= np.linalg.norm(direction)
    original = np.array([1.0, 0.0, 0.0])
    if np.allclose(direction, original):
        xyz, w = np.zeros(3), 1.0
    elif np.allclose(np.cross(direction, original), 0.0):
        xyz, w = np.array([0.0, 1.0, 0.0]), 0.0
    else:
        half = original + direction
        half /= np.linalg.norm(half)
        xyz, w = np.cross(original, half), float(np.dot(original, half))
    return list(np.asarray(point_m, dtype=float)) + list(xyz) + [w]


@dataclass
class SceneHandles:
    root: object
    dofs: object
    ir_controller: object
    magnetic_force: object
    torque_controller: object
    layout: NelsonInstrumentLayout
    simulator: object
    instrument: object
    vessel: object
    stages: dict
    compatibility: dict


def create_sofa_scene(root, bundle: VesselBundle, config: ExperimentConfig = DEFAULT_CONFIG, stages=None, messages=None):
    """The only production scene: original Nelson modules + SOFA21 components."""
    from .sofa_compat import load_plugins
    from .mcr_simulator import Simulator
    from .mcr_environment import Environment
    from .mcr_instrument import Instrument
    from .mcr_mag_controller import MagController
    config.validate()
    stages = stages if stages is not None else {}
    for key in "ABCDEFGH":
        stages.setdefault(key, {"status": "NOT_RUN"})
    def stage(key, operation):
        try:
            value = operation()
            if messages is not None:
                messages.require_no_errors(key)
            stages[key] = {"status": "PASS", "scope": "component construction only"}
            return value
        except Exception as exc:
            stages[key] = {"status": "FAIL", "error": repr(exc)}
            for later in "ABCDEFGH"["ABCDEFGH".index(key)+1:]:
                stages[later] = {"status": "BLOCKED", "reason": "earlier stage " + key + " failed"}
            raise
    compatibility = stage("A", lambda: load_plugins(root))
    # Simulator owns the animation loop and the whole contact pipeline. Keep B/C
    # independently reported in the backend's observed SOFA log checks.
    root.dt = config.sofa.dt_s
    root.gravity = list(config.sofa.gravity_m_s2)
    animation_loop = stage("B", lambda: root.addObject("FreeMotionAnimationLoop"))
    simulator = stage("C", lambda: Simulator(root, config, animation_loop))
    vessel = stage("D", lambda: Environment(
        root_node=root, environment_stl=str(bundle.collision_stl),
        T_env_sim=[0, 0, 0, 0, 0, 0, 1], scale="0.001", visual=False,
        use_line_collision=True, use_point_collision=False,
        collision_proximity=config.sofa.vessel_proximity_m,
        triangle_collision_proximity=config.sofa.vessel_proximity_m,
        line_point_collision_proximity=config.sofa.vessel_proximity_m, verbose=False))
    instrument = stage("E", lambda: Instrument(
        root, insertion_pose(bundle.insertion_point_m, bundle.insertion_direction), config))
    def add_magnet():
        controller = MagController(instrument, config.instrument.initial_field_t)
        instrument.InstrumentCombined.addObject(controller)
        return controller
    controller = stage("F", add_magnet)
    return SceneHandles(root, instrument.MO, instrument.IRC, instrument.CFF,
                        controller, instrument.layout, simulator, instrument, vessel, stages, compatibility)


class SofaBackend:
    """Lazy SOFA backend. Use under the project's SOFA Python launcher."""

    def __init__(self, config: ExperimentConfig = DEFAULT_CONFIG):
        self.config = config
        self._Sofa = None
        self._root = None
        self._handles: Optional[SceneHandles] = None
        self._simulation_error = False
        self.stages = {key: {"status": "NOT_RUN"} for key in "ABCDEFGH"}
        self.physics_substeps_executed = 0
        self._messages = None
        self.last_exception = None
        self.last_messages = []

    @staticmethod
    def availability() -> Dict[str, object]:
        # Some SOFA builds print environment checks during import. Keep callers'
        # JSON/log streams clean while retaining the real exception if unavailable.
        sink = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                importlib.import_module("Sofa")
                importlib.import_module("SofaRuntime")
        except Exception as exc:
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        return {"available": True, "reason": None}

    def reset(self, bundle: VesselBundle) -> BackendState:
        status = self.availability()
        if not status["available"]:
            raise RuntimeError("SOFA Python runtime unavailable: " + str(status["reason"]))
        self.close()
        self._Sofa = importlib.import_module("Sofa")
        self._root = self._Sofa.Core.Node("root")
        from .sofa_compat import message_collector
        self._messages = message_collector()
        self._messages.__enter__()
        self.stages = {key: {"status": "NOT_RUN"} for key in "ABCDEFGH"}
        self.physics_substeps_executed = 0
        try:
            self._handles = create_sofa_scene(self._root, bundle, self.config, self.stages, self._messages)
            self._messages.require_no_errors("A-F component construction")
            self._Sofa.Simulation.init(self._root)
            self._messages.require_no_errors("G Simulation.init")
            self.stages["G"] = {"status": "PASS", "scope": "Simulation.init; no integration"}
            self._simulation_error = False
        except Exception as exc:
            import traceback
            self.last_exception = traceback.format_exc()
            if self.stages["F"]["status"] == "PASS":
                self.stages["G"] = {"status": "FAIL", "error": str(exc)}
                self.stages["H"] = {"status": "BLOCKED"}
            self.close()
            raise
        return self.state()

    def state(self) -> BackendState:
        if self._handles is None:
            raise RuntimeError("SOFA backend has not been reset")
        positions = np.asarray(self._handles.dofs.position.value, dtype=float)
        if positions.ndim != 2 or positions.shape[1] < 7:
            raise RuntimeError(f"Unexpected SOFA Rigid3d positions: {positions.shape}")
        controller = self._handles.ir_controller
        try:
            inserted_values = np.asarray(controller.xtip.value, dtype=float).reshape(-1)
            first_values = np.asarray(controller.indexFirstNode.value).reshape(-1)
            curv_abs = np.asarray(controller.CurvAbs.value, dtype=float).reshape(-1)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("Inserted catheter geometry NOT_OBSERVED: IRC xtip/indexFirstNode/CurvAbs unavailable") from exc
        if len(inserted_values) != 1 or len(first_values) != 1 or not np.all(np.isfinite(first_values)):
            raise RuntimeError("Inserted geometry requires the existing single-instrument IRC layout")
        inserted = float(inserted_values[0])
        first = int(first_values[0])
        # SOFA21 BeamAdapter controller source: idP = numControlledNodes-nnode+p;
        # fixFirstNodesWithUntil() exposes that base node as indexFirstNode.
        # CurvAbs lists exactly those simulated DOFs, in entrance-to-tip order.
        # This scene has one catheter and no dropped instruments. Never infer
        # active nodes from a uniform rest grid or retain its parked prefix.
        if (first != first_values[0] or first < 0 or len(curv_abs) == 0 or
                first + len(curv_abs) != len(positions)):
            raise RuntimeError("IRC active-node range does not match mechanical DOFs")
        if (not np.isfinite(inserted) or not 0.0 <= inserted <= self.config.instrument.total_length_m or
                not np.all(np.isfinite(curv_abs)) or np.any(np.diff(curv_abs) <= 0)):
            raise RuntimeError("Invalid IRC insertion/curvilinear coordinates")
        # Only a consistency check in SI metres; no controller/physics tuning.
        if (not np.isclose(curv_abs[0], 0.0, atol=1e-8, rtol=0.0) or
                not np.isclose(curv_abs[-1], inserted, atol=1e-8, rtol=1e-6)):
            raise RuntimeError("IRC geometry does not cover entrance-to-xtip in metres; inserted shape NOT_OBSERVED")
        active = positions[first:first + len(curv_abs)]
        if not np.all(np.isfinite(active)):
            raise RuntimeError("Inserted SOFA catheter geometry contains NaN or Inf")
        tracking = active[::-1, :3].copy()
        tip_frame = quaternion_xyzw_to_matrix(active[-1, 3:7])
        field = np.asarray(self._handles.torque_controller.field_world_t, dtype=float).copy()
        error = self._simulation_error or not np.all(np.isfinite(positions))
        return BackendState(
            catheter_tracking_m=tracking,
            tip_frame_world=tip_frame,
            magnetic_field_world_t=field,
            inserted_length_m=inserted,
            simulation_error=bool(error),
            extra={"geometry_source": "SOFA21 IRC indexFirstNode + CurvAbs",
                   "inserted_dof_indices_tip_first": np.arange(first, first+len(curv_abs))[::-1].copy(),
                   "inserted_curv_abs_m_tip_first": curv_abs[::-1].copy()},
        )

    def physics_substep(self, magnetic_field_world_t, insertion_delta_m) -> StepDiagnostics:
        if self._handles is None:
            raise RuntimeError("SOFA backend has not been reset")
        self._handles.torque_controller.field_world_t = np.asarray(magnetic_field_world_t, dtype=float).reshape(3)
        xtip_data = self._handles.ir_controller.xtip
        current = float(np.asarray(xtip_data.value).reshape(-1)[0])
        updated = float(np.clip(current + float(insertion_delta_m), 0.0, self.config.instrument.total_length_m))
        xtip_data.value = [updated]
        reasons = {
            "contact_count": "SOFA binding exposes no stable public contact-count field in this runtime",
            "max_penetration_m": "no vessel SDF was generated; collision response does not expose penetration depth",
            "lcp_iterations": "LCP iteration Data field must be verified in the dedicated SOFA runtime",
            "minimum_clearance_m": "no vessel SDF was generated",
        }
        try:
            self._Sofa.Simulation.animate(self._root, self.config.sofa.dt_s)
            self.physics_substeps_executed += 1
            self._messages.require_no_errors("H physics substep")
            self.stages["H"] = {"status": "PASS" if self.physics_substeps_executed % self.config.sofa.physics_substeps == 0 else "NOT_RUN", "executed_substeps": self.physics_substeps_executed}
        except Exception:
            import traceback
            self.last_exception = traceback.format_exc()
            self.stages["H"] = {"status": "FAIL", "traceback": self.last_exception}
            self._simulation_error = True
            reasons["simulation_error"] = self.last_exception
            return StepDiagnostics(simulation_error=True, unavailable_reasons=reasons)
        if not np.all(np.isfinite(np.asarray(self._handles.dofs.position.value, dtype=float))):
            self._simulation_error = True
        return StepDiagnostics(simulation_error=self._simulation_error, unavailable_reasons=reasons)

    def close(self) -> None:
        if self._root is not None and self._Sofa is not None:
            try:
                self._Sofa.Simulation.unload(self._root)
            except Exception:
                pass
        self._root = None
        self._handles = None
        if self._messages is not None:
            self.last_messages = list(self._messages.messages)
            self._messages.__exit__(None, None, None)
            self._messages = None
