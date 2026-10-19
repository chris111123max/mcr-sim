"""SOFA21 plugin boundary, verified against this server's ObjectFactory.

Aliases are registered by SOFA21 itself, not guessed replacements:
CollisionPipeline -> DefaultPipeline, CollisionResponse -> DefaultContactManager,
RegularGrid -> RegularGridTopology, Mesh -> MeshTopology.
"""
import importlib

REQUIRED_PLUGINS = (
    "SofaPython3", "BeamAdapter", "SofaConstraint", "SofaGeneralAnimationLoop",
    "SofaMiscCollision", "SofaGeneralLoader", "SofaGeneralLinearSolver",
    "SofaImplicitOdeSolver", "SofaGeneralTopology", "SofaTopologyMapping",
    "SofaBoundaryCondition", "SofaRigid", "SofaSimpleFem",
    "SofaBaseCollision", "SofaMeshCollision", "SofaBaseTopology",
)
COMPONENTS = (
    "FreeMotionAnimationLoop", "CollisionPipeline", "BruteForceBroadPhase",
    "BVHNarrowPhase", "LocalMinDistance", "CollisionResponse",
    "DefaultCollisionGroupManager", "LCPConstraintSolver", "EulerImplicitSolver",
    "BTDLinearSolver", "LinearSolverConstraintCorrection", "WireRestShape",
    "WireBeamInterpolation", "AdaptiveBeamForceFieldAndMass",
    "InterventionalRadiologyController", "MeshSTLLoader", "Mesh", "RegularGrid",
    "TriangleCollisionModel", "LineCollisionModel", "PointCollisionModel",
    "MultiAdaptiveBeamMapping", "ConstantForceField", "FixedConstraint",
    "RestShapeSpringsForceField", "DefaultVisualManagerLoop", "EdgeSetTopologyContainer",
    "EdgeSetTopologyModifier", "EdgeSetGeometryAlgorithms", "MechanicalObject",
)


def load_plugins(root=None):
    runtime = importlib.import_module("SofaRuntime")
    sofa = importlib.import_module("Sofa")
    loaded = {}
    for name in REQUIRED_PLUGINS:
        result = runtime.importPlugin(name)
        if result is not True:
            raise RuntimeError("SOFA plugin load not confirmed: " + name)
        if root is not None:
            root.addObject("RequiredPlugin", name="Import" + name, pluginName=name)
        loaded[name] = True
    sources = {}
    for name in COMPONENTS:
        entry = sofa.Core.ObjectFactory.getComponent(name)
        if entry is None:
            raise RuntimeError("Required SOFA21 component unavailable: " + name)
        sources[name] = str(entry)
    return {"plugins": loaded, "components": sources}


def message_collector():
    sofa = importlib.import_module("Sofa")

    class Collector(sofa.Helper.MessageHandler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def process(self, message):
            if str(message["type"]) in ("Warning", "Error", "Fatal", "Deprecated"):
                self.messages.append({key: str(message[key]) for key in ("type", "sender", "message")})

        @property
        def errors(self):
            return [m for m in self.messages if m["type"] in ("Error", "Fatal")]

        def require_no_errors(self, stage):
            if self.errors:
                raise RuntimeError("SOFA errors at " + stage + ": " + repr(self.errors))

    return Collector()
