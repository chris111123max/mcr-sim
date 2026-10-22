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
    # Fail closed: only the audited SOFA21 IRC bootstrap library is permitted.
    from pathlib import Path
    import hashlib, json
    native = Path(__file__).resolve().parent / "native/beamadapter21_irc_bootstrap"
    library = native / "libBeamAdapter.so.21.12"
    expected = json.loads((native / "manifest.json").read_text())["library_sha256"]
    if hashlib.sha256(library.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Production BeamAdapter library hash mismatch")
    runtime.PluginRepository.addFirstPath(str(native))
    if runtime.importPlugin(str(library)) is not True:
        raise RuntimeError("Patched production BeamAdapter failed to load")
    mapped = {str(Path(line.split()[-1]).resolve())
              for line in Path("/proc/self/maps").read_text().splitlines()
              if "libBeamAdapter.so" in line}
    if mapped != {str(library.resolve())}:
        raise RuntimeError("Unexpected BeamAdapter mapping: " + repr(mapped))
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
    return {"plugins": loaded, "components": sources, "beamadapter_library": str(library), "beamadapter_sha256": expected}


def message_collector():
    sofa = importlib.import_module("Sofa")

    class Collector(sofa.Helper.MessageHandler):
        def __init__(self):
            super().__init__()
            from collections import deque,Counter
            self.messages=deque(maxlen=256)
            self._errors=deque(maxlen=32)
            self.counts=Counter();self._lcp_error=False;self._other_error=False

        def process(self,message):
            import json,sys
            row={key:str(message[key]) for key in ('type','sender','message')}
            if row['type'] not in ('Warning','Error','Fatal','Deprecated'):return
            key=(row['type'],row['sender'])
            if key not in self.counts and len(self.counts)>=64:key=('other','other')
            self.counts[key]+=1
            numerical=(row['sender']=='LCPConstraintSolver' and
                       'No convergence' in row['message'])
            fatal=row['type'] in ('Error','Fatal') or 'No convergence' in row['message']
            if numerical:self._lcp_error=True
            if fatal and not numerical:self._other_error=True
            if fatal:
                self._errors.append(row)
                # Detailed errors survive bounded in-memory diagnostics.
                print('SOFA_ERROR '+json.dumps(row),file=sys.stderr,flush=True)
            if fatal or self.counts[key]<=3:self.messages.append(row)

        @property
        def failure_kind(self):
            return 'lcp_nonconvergence' if self._lcp_error and not self._other_error else None

        @property
        def errors(self):return list(self._errors)

        def require_no_errors(self,stage):
            # Errors remain latched until this scene/handler is discarded.
            if self._errors:
                raise RuntimeError('SOFA errors at '+stage+': '+repr(list(self._errors)))

    return Collector()
