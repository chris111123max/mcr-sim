"""Immutable configuration for the unified Nelson / SOFA21 magnetic navigation system."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple
import math

MM_TO_M = 0.001
PACKAGE_DIR = Path(__file__).resolve().parent
PYTHON_DIR = PACKAGE_DIR.parent
DEFAULT_VESSEL_ROOT = PYTHON_DIR / "vessel_cache" / "archvariety_type_I_d085"
DEFAULT_OUTPUT_ROOT = PYTHON_DIR.parent / "training_runs" / "mcr_sim"


@dataclass(frozen=True)
class BeamDiscretization:
    body_beams: int
    tip_beams: int
    collision_edges_body: int
    collision_edges_tip: int
    visual_edges: int = 600

    @property
    def total_beams(self) -> int:
        return self.body_beams + self.tip_beams

    @property
    def mechanical_dofs(self) -> int:
        return self.total_beams + 1


BEAM_PRESETS: Dict[str, BeamDiscretization] = {
    "nelson_reference": BeamDiscretization(30, 3, 30, 6, 600),
    "default_100_34": BeamDiscretization(100, 34, 100, 68, 600),
    "refined_150_51": BeamDiscretization(150, 51, 150, 102, 600),
}


@dataclass(frozen=True)
class InstrumentConfig:
    body_length_m: float = 0.500
    tip_length_m: float = 0.034
    outer_diameter_m: float = 0.00133
    inner_diameter_m: float = 0.0008
    young_modulus_body_pa: float = 100e6
    young_modulus_tip_pa: float = 21e6
    mass_density_kg_m3: float = 155.0
    magnet_length_m: float = 0.004
    magnet_outer_diameter_m: float = 0.00133
    magnet_inner_diameter_m: float = 0.00086
    magnet_remanence_t: float = 1.45
    initial_field_t: Tuple[float, float, float] = (0.01, 0.01, 0.0)
    discretization: BeamDiscretization = field(
        default_factory=lambda: BEAM_PRESETS["default_100_34"]
    )

    @property
    def total_length_m(self) -> float:
        return self.body_length_m + self.tip_length_m

    @property
    def radius_m(self) -> float:
        return self.outer_diameter_m / 2.0

    @property
    def initial_field_magnitude_t(self) -> float:
        return math.sqrt(sum(value * value for value in self.initial_field_t))


@dataclass(frozen=True)
class SofaConfig:
    dt_s: float = 0.006
    physics_substeps: int = 22
    gravity_m_s2: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    contact_distance_m: float = 0.0003
    alarm_distance_m: float = 0.0005
    angle_cone: float = 0.02
    friction: float = 0.1
    lcp_tolerance: float = 1e-4
    lcp_max_iterations: int = 2000
    lcp_build: bool = False
    rayleigh_stiffness: float = 0.2
    rayleigh_mass: float = 0.1
    catheter_proximity_m: float = 0.000665
    vessel_proximity_m: float = 0.0

    @property
    def rl_dt_s(self) -> float:
        return self.dt_s * self.physics_substeps


@dataclass(frozen=True)
class ActionConfig:
    magnetic_resolution: float = 0.001
    max_field_rotation_deg: float = 24.0
    max_insertion_per_rl_step_m: float = 0.004


@dataclass(frozen=True)
class ObservationConfig:
    # Fixed SI scales shared by every vessel/episode; no vessel-bounds scaling.
    lookahead_distances_m: Tuple[float, ...] = (0.0, 0.008, 0.020, 0.036, 0.056)
    shape_offsets_m: Tuple[float, ...] = (0.010, 0.030, 0.060)
    spatial_scale_m: float = 0.100
    remaining_path_scale_m: float = 0.500
    target_scale_m: float = 0.500
    radius_scale_m: float = 0.020
    tip_displacement_scale_m: float = 0.010

    @property
    def dimension(self) -> int:
        return 41


@dataclass(frozen=True)
class RewardConfig:
    target: float = 1.0
    path_delta_per_mm: float = 0.001
    step: float = -0.005


@dataclass(frozen=True)
class TaskConfig:
    requested_target_branches: Tuple[str, ...] = (
        "lcca", "rcca", "lsa", "rsa", "bct", "co"
    )
    target_threshold_m: float = 0.003
    max_episode_steps_train: int = 150
    max_episode_steps_eval: int = 450


@dataclass(frozen=True)
class SACConfig:
    gamma: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 32
    replay_buffer_size: int = 10_000
    heatup_steps: int = 500_000
    update_per_explore_step: float = 1.0 / 20.0
    explore_steps_between_eval: int = 250_000
    learning_rate: float = 0.0003218
    lstm_layers: int = 1
    lstm_nodes: int = 900
    actor_hidden: Tuple[int, ...] = (400, 400, 400)
    critic_hidden: Tuple[int, ...] = (400, 400, 400)
    training_steps_script_default: int = 20_000_000


@dataclass(frozen=True)
class ExperimentConfig:
    vessel_root: Path = DEFAULT_VESSEL_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    vessel_seeds: Tuple[int, ...] = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
    eval_vessel_seeds: Tuple[int, ...] = (1101, 1202)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    sofa: SofaConfig = field(default_factory=SofaConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    sac: SACConfig = field(default_factory=SACConfig)

    def validate(self) -> None:
        assert self.instrument.total_length_m == 0.534
        assert self.instrument.discretization.total_beams == 134
        assert self.instrument.discretization.mechanical_dofs == 135
        assert math.isclose(self.sofa.rl_dt_s, 0.132, abs_tol=1e-12)
        assert self.observation.dimension == 41
        assert self.task.target_threshold_m == 0.003
        assert self.action.max_insertion_per_rl_step_m == 0.004
        assert self.sofa.catheter_proximity_m == self.instrument.radius_m


DEFAULT_CONFIG = ExperimentConfig()
DEFAULT_CONFIG.validate()
