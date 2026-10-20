"""Nelson BeamAdapter beam/force/collision assembly with the new task parameters.

Derived from the server's original Instrument: WireRestShape, wire interpolation,
AdaptiveBeamForceFieldAndMass, IRC insertion, ConstantForceField and beam mapping.
No SDF wall force, position projection, or legacy RL control is attached.
"""
import numpy as np
from .config import DEFAULT_CONFIG
from .instrument_layout import NelsonInstrumentLayout


class Instrument:
    def __init__(self, root_node, T_start_sim, config=DEFAULT_CONFIG, name="mag_instrument"):
        cfg, s = config.instrument, config.sofa
        d = cfg.discretization
        self.layout = NelsonInstrumentLayout(cfg)
        self.config = config
        # The old code computed an equivalent radius then divided it by two.
        # Use the morning configuration's actual annular second moment of area.
        self.equivalent_radius_m = ((cfg.outer_diameter_m / 2) ** 4 - (cfg.inner_diameter_m / 2) ** 4) ** 0.25
        topo = root_node.addChild(name + "_topo_lines")
        self.rest_shape = topo.addObject(
            "WireRestShape", name="InstrRestShape", template="Rigid3d",
            straightLength=cfg.body_length_m, length=cfg.total_length_m,
            numEdges=d.visual_edges, densityOfBeams=[d.body_beams, d.tip_beams],
            numEdgesCollis=[d.collision_edges_body, d.collision_edges_tip],
            youngModulus=cfg.young_modulus_body_pa, youngModulusExtremity=cfg.young_modulus_tip_pa,
            radius=self.equivalent_radius_m, radiusExtremity=self.equivalent_radius_m,
            spireDiameter=250.0, spireHeight=0.0, printLog=False)
        topo.addObject("EdgeSetTopologyContainer", name="meshLinesGuide")
        topo.addObject("EdgeSetTopologyModifier", name="Modifier")
        topo.addObject("EdgeSetGeometryAlgorithms", name="GeomAlgo", template="Rigid3d")
        topo.addObject("MechanicalObject", name="dofTopo2", template="Rigid3d")
        device = self.InstrumentCombined = root_node.addChild(name)
        self.ode = device.addObject("EulerImplicitSolver", rayleighStiffness=s.rayleigh_stiffness, rayleighMass=s.rayleigh_mass)
        self.linear = device.addObject("BTDLinearSolver", verification=False, subpartSolve=False, verbose=False)
        self.RG = device.addObject(
            "RegularGrid", name="meshLinesCombined", nx=d.mechanical_dofs, ny=1, nz=1,
            xmin=0.0, xmax=cfg.total_length_m, ymin=0.0, ymax=0.0, zmin=0.0, zmax=0.0)
        # IRC parks nodes at the insertion pose in bwdInit. Its moving entrance
        # spring must use that same reference pose, not RegularGrid world-X positions.
        self.MO = device.addObject(
            "MechanicalObject", name="DOFs", template="Rigid3d",
            rest_position=[list(T_start_sim) for _ in range(d.mechanical_dofs)])
        self.IC = device.addObject(
            "WireBeamInterpolation", name="InterpolGuide", WireRestShape="@../" + name + "_topo_lines/InstrRestShape",
            radius=self.equivalent_radius_m, printLog=False)
        self.force_field = device.addObject("AdaptiveBeamForceFieldAndMass", massDensity=cfg.mass_density_kg_m3,
                                            name="GuideForceField", interpolation="@InterpolGuide")
        self.CFF = device.addObject(
            "ConstantForceField", name="DistributedMagneticTorque", indices=list(self.layout.magnetic_node_indices),
            forces=[[0.0] * 6 for _ in self.layout.magnet_overlaps], indexFromEnd=False)
        self.IRC = device.addObject(
            "InterventionalRadiologyController", name="m_ircontroller", template="Rigid3d",
            instruments="InterpolGuide", startingPos=list(T_start_sim), xtip=[0.001],
            # BeamAdapter length precision: 0.01 mm expressed in this SI-metre scene.
            threshold=1e-5,
            rotationInstrument=[0.0], speed=0.0, listening=True, controlledInstrument=0, printLog=False)
        self.correction = device.addObject("LinearSolverConstraintCorrection", wire_optimization=True, printLog=False)
        device.addObject("FixedConstraint", indices=0, name="FixedConstraint")
        device.addObject("RestShapeSpringsForceField", points="@m_ircontroller.indexFirstNode", angularStiffness=1e8, stiffness=1e8)
        self.CollisionModel = collision = device.addChild("CollisionModel")
        collision.addObject("EdgeSetTopologyContainer", name="collisEdgeSet")
        collision.addObject("EdgeSetTopologyModifier", name="colliseEdgeModifier")
        self.collision_dofs = collision.addObject("MechanicalObject", name="CollisionDOFs")
        collision.addObject("MultiAdaptiveBeamMapping", controller="../m_ircontroller", useCurvAbs=True, printLog=False, name="collisMap")
        self.line_model = collision.addObject("LineCollisionModel", proximity=s.catheter_proximity_m, group=1)
        self.point_model = collision.addObject("PointCollisionModel", proximity=s.catheter_proximity_m, group=1)
