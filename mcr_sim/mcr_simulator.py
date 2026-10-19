"""Nelson SOFA21 root solver/collision pipeline; no RL or environment overrides.

Uses the legacy component spellings of the original local Simulator.
"""
from .config import DEFAULT_CONFIG


class Simulator:
    def __init__(self, root_node, config=DEFAULT_CONFIG, animation_loop=None):
        self.root_node = root_node
        self.config = config
        s = config.sofa
        root_node.dt = s.dt_s
        root_node.gravity = list(s.gravity_m_s2)
        self.animation_loop = animation_loop if animation_loop is not None else root_node.addObject("FreeMotionAnimationLoop")
        self.lcp_solver = root_node.addObject(
            "LCPConstraintSolver", name="LCP", mu=s.friction,
            tolerance=s.lcp_tolerance, maxIt=s.lcp_max_iterations, build_lcp=s.lcp_build)
        self.pipeline = root_node.addObject("CollisionPipeline", draw=False, depth=6, verbose=False)
        self.broad_phase = root_node.addObject("BruteForceBroadPhase")
        self.narrow_phase = root_node.addObject("BVHNarrowPhase")
        self.intersection = root_node.addObject(
            "LocalMinDistance", name="localmindistance", contactDistance=s.contact_distance_m,
            alarmDistance=s.alarm_distance_m, angleCone=s.angle_cone)
        self.response = root_node.addObject("CollisionResponse", response="FrictionContactConstraint")
        self.group_manager = root_node.addObject("DefaultCollisionGroupManager")
        self.visual_loop = root_node.addObject("DefaultVisualManagerLoop")
