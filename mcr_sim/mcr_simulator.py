import Sofa
import os

from .training_config import (
    CONSTRAINT_MAX_ITERATIONS,
    CONSTRAINT_TOLERANCE,
    FRICTION_COEFFICIENT,
    LMD_ALARM_DISTANCE_M,
    LMD_ANGLE_CONE,
    LMD_CONTACT_DISTANCE_M,
    SOFA_TIME_STEP_S,
)


class Simulator(Sofa.Core.Controller):
    """
    A class used to define the physics and the solver of the SOFA simulation.

    :param root_node: The sofa root node
    :type root_node:
    :param dt: The time step (s)
    :type dt: float
    :param gravity: The gravity verctor (m/s^2)
    :type gravity: float
    :param friction_coef: The coeficient of friction
    :type friction_coef: float
    """

    def __init__(
            self,
            root_node,
            dt=SOFA_TIME_STEP_S,
            gravity=[0, 0, 0],
            friction_coef=FRICTION_COEFFICIENT,
            *args, **kwargs):

        # These are needed (and the normal way to override from a python class)
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root_node = root_node
        self.dt = dt
        self.gravity = gravity
        self.friction_coef = friction_coef

        self.root_node.addObject(
            'RequiredPlugin',
            name='ImportSoftRob',
            pluginName='SoftRobots')
        self.root_node.addObject(
            'RequiredPlugin',
            name='ImportBeamAdapt',
            pluginName='BeamAdapter')
        self.root_node.addObject(
            'RequiredPlugin',
            name='ImportSofaPython3',
            pluginName='SofaPython3')
        self.root_node.addObject(
            'RequiredPlugin',
            name='ImportSofaConstraint',
            pluginName='SofaConstraint')
        self.root_node.addObject(
            'RequiredPlugin',
            name='ImportSofaMiscCollision',
            pluginName='SofaMiscCollision')
        self.root_node.addObject(
            'RequiredPlugin',
            name='ImportSofaGeneralLoader',
            pluginName='SofaGeneralLoader')
        self.root_node.addObject(
            'RequiredPlugin',
            name='ImportSofaOpenglVisual',
            pluginName='SofaOpenglVisual')
        self.root_node.addObject(
            'RequiredPlugin',
            name='ImportSofaGeneralLinearSolver',
            pluginName='SofaGeneralLinearSolver')

        self.root_node.dt = self.dt
        self.root_node.animate = True
        self.root_node.gravity = self.gravity

        self.root_node.addObject(
            'VisualStyle',
            displayFlags='showVisualModels hideBehaviorModels \
                hideCollisionModels hideMappings hideForceFields \
                    hideInteractionForceFields')
        self.root_node.addObject('FreeMotionAnimationLoop')

        # Lightweight contact solving for RL training:
        #   - use a relaxed tolerance for speed;
        #   - keep enough iterations for stable triangle-vessel contact;
        #   - stronger anti-shortcut behavior is handled by reward/out-of-vessel penalty
        #     and by smaller SOFA substeps, not by heavy collision geometry.
        constraint_solver_type = os.environ.get(
            "MCR_CONSTRAINT_SOLVER", "lcp"
        ).strip().lower()
        constraint_tolerance = os.environ.get(
            "MCR_CONSTRAINT_TOLERANCE", str(CONSTRAINT_TOLERANCE)
        )
        constraint_max_it = os.environ.get(
            "MCR_CONSTRAINT_MAX_IT", str(CONSTRAINT_MAX_ITERATIONS)
        )

        if constraint_solver_type in ("generic", "genericconstraintsolver"):
            self.lcp_solver = self.root_node.addObject(
                'GenericConstraintSolver',
                tolerance=constraint_tolerance,
                maxIterations=constraint_max_it,
                printLog='false')
            print(
                "[CONSTRAINT_SOLVER]",
                "type=GenericConstraintSolver",
                "tolerance=", constraint_tolerance,
                "maxIterations=", constraint_max_it,
            )
        else:
            self.lcp_solver = self.root_node.addObject(
                'LCPConstraintSolver',
                mu=str(friction_coef),
                tolerance=constraint_tolerance,
                maxIt=constraint_max_it,
                build_lcp='false')
            print(
                "[CONSTRAINT_SOLVER]",
                "type=LCPConstraintSolver",
                "mu=", friction_coef,
                "tolerance=", constraint_tolerance,
                "maxIt=", constraint_max_it,
            )

        self.root_node.addObject(
            'CollisionPipeline',
            draw='0',
            depth='6',
            verbose='0')

        # NOTE: BruteForceDetection is deprecated in this pipeline.
        # Keep sofa_env baseline broad/narrow phase setup.
        self.root_node.addObject(
            'BruteForceBroadPhase',
            name='N2_1')
        self.root_node.addObject(
            'BVHNarrowPhase',
            name='N2_2')

        # Lightweight LocalMinDistance.  The static vessel is triangle-only;
        # catheter Line/Point primitives carry the catheter-radius proximity.
        # Vessel Line/Point models stay disabled because they multiply contacts
        # and previously destabilized the LCP solve.
        lmd_contact_distance = os.environ.get(
            "MCR_LMD_CONTACT_DISTANCE", str(LMD_CONTACT_DISTANCE_M)
        )
        lmd_alarm_distance = os.environ.get(
            "MCR_LMD_ALARM_DISTANCE", str(LMD_ALARM_DISTANCE_M)
        )
        lmd_angle_cone = os.environ.get(
            "MCR_LMD_ANGLE_CONE", str(LMD_ANGLE_CONE)
        )
        print(
            "[LOCAL_MIN_DISTANCE]",
            "contactDistance=", lmd_contact_distance,
            "alarmDistance=", lmd_alarm_distance,
            "angleCone=", lmd_angle_cone,
        )
        self.root_node.addObject(
            'LocalMinDistance',
            contactDistance=lmd_contact_distance,
            alarmDistance=lmd_alarm_distance,
            name='localmindistance',
            angleCone=lmd_angle_cone)

        self.root_node.addObject(
            'CollisionResponse',
            name='Response',
            response='FrictionContactConstraint')
        self.root_node.addObject(
            'DefaultCollisionGroupManager',
            name='Group')
        self.root_node.addObject(
            'DefaultVisualManagerLoop',
            name='VisualLoop')

        # set backbround color
        self.root_node.addObject('BackgroundSetting', color='1 1 1')
