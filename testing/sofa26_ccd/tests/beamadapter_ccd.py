#!/usr/bin/env python3
"""Small BeamAdapter / MultiAdaptiveBeamMapping collision experiment."""
import json
import time
from pathlib import Path

import numpy as np
import Sofa
import SofaRuntime


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_runtime/results/beamadapter_ccd_test.json"
DT = 0.005
PLUGIN_NAMES = [
    "BeamAdapter",
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Collision.Detection.Algorithm",
    "Sofa.Component.Collision.Detection.Intersection",
    "Sofa.Component.Collision.Geometry",
    "Sofa.Component.Collision.Response.Contact",
    "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.Constraint.Lagrangian.Solver",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Constant",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Topology.Container.Grid",
]


def arr(data):
    try:
        return np.asarray(data.array(), dtype=float).copy()
    except Exception:
        return np.asarray(data.value, dtype=float).copy()


def get_value(obj, name):
    try:
        return obj.findData(name).value
    except Exception:
        return None


def contacts(node):
    return sum(o.getClassName() == "FrictionContact" for o in node.objects) + sum(
        contacts(child) for child in node.children
    )


def create_scene(method="ccd"):
    root = Sofa.Core.Node("root")
    root.dt.value = DT
    root.gravity.value = [0, 0, 0]
    root.addObject("FreeMotionAnimationLoop")
    solver = root.addObject("BlockGaussSeidelConstraintSolver", name="constraints",
                            maxIterations=1000, tolerance=1e-6)
    root.addObject("CollisionPipeline")
    root.addObject("BruteForceBroadPhase")
    root.addObject("BVHNarrowPhase")
    if method == "ccd":
        root.addObject("CCDTightInclusionIntersection", continuousCollisionType="FreeMotion",
                       alarmDistance=1e-4, contactDistance=1e-5, maxIterations=1000)
    else:
        root.addObject("NewProximityIntersection", alarmDistance=1e-4,
                       contactDistance=1e-5)
    root.addObject("CollisionResponse", response="FrictionContactConstraint",
                   responseParams="mu=0.0")

    topology = root.addChild("edge_topology")
    topology.addObject("RodStraightSection", name="section", length=0.02,
                       radius=0.000665, nbBeams=2, nbEdgesCollis=8,
                       nbEdgesVisu=8, youngModulus=1e6, massDensity=1000,
                       poissonRatio=0.3)
    topology.addObject("WireRestShape", name="rest_shape", template="Rigid3d",
                       wireMaterials="@section")
    topology.addObject("EdgeSetTopologyContainer", name="edges")
    topology.addObject("EdgeSetTopologyModifier")
    topology.addObject("EdgeSetGeometryAlgorithms", template="Rigid3d")
    topology.addObject("MechanicalObject", name="topology_dofs", template="Rigid3d")

    beam = root.addChild("beam")
    beam.addObject("EulerImplicitSolver", rayleighStiffness=0, rayleighMass=0)
    beam.addObject("BTDLinearSolver", name="linear")
    beam.addObject("RegularGridTopology", name="beam_grid", n=[3, 1, 1],
                   min=[0, 0, 0], max=[0, 0, 0])
    beam_dofs = beam.addObject("MechanicalObject", name="beam_dofs", template="Rigid3d")
    beam.addObject("WireBeamInterpolation", name="interpolation",
                   WireRestShape="@../edge_topology/rest_shape")
    beam.addObject("AdaptiveBeamForceFieldAndMass", name="beam_force",
                   interpolation="@interpolation", massDensity=1000)
    beam.addObject("InterventionalRadiologyController", name="controller",
                   template="Rigid3d", instruments="interpolation",
                   topology="@beam_grid", startingPos=[0, 0, 0.001, 0, 0, 0, 1],
                   xtip=[0.02], rotationInstrument=[0], step=0, speed=0,
                   listening=False, controlledInstrument=0)
    beam.addObject("LinearSolverConstraintCorrection")

    beam.addObject("FixedProjectiveConstraint", name="base_fix", indices="0")
    collision = beam.addChild("collision")
    collision.addObject("EdgeSetTopologyContainer", name="collision_edges")
    collision.addObject("EdgeSetTopologyModifier")
    collision_dofs = collision.addObject("MechanicalObject", name="collision_dofs",
                                         template="Vec3d")
    collision.addObject("MultiAdaptiveBeamMapping", name="mapping",
                        controller="@../controller", useCurvAbs=True)
    collision.addObject("PointCollisionModel", contactDistance=0.0)
    collision.addObject("LineCollisionModel", contactDistance=0.0)

    wall = root.addChild("wall")
    vertices = "-0.05 -0.05 0 0.05 -0.05 0 0 0.05 0"
    wall.addObject("MeshTopology", position=vertices, triangles="0 1 2")
    wall.addObject("MechanicalObject", template="Vec3d", position=vertices)
    wall.addObject("TriangleCollisionModel", moving=False, simulated=False,
                   bothSide=True, contactDistance=0.0)
    return root, beam_dofs, collision_dofs, solver


def run(method):
    root, beam, collision, solver = create_scene(method)
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, DT)  # Establish the deployed beam geometry before the crossing.
    before = {
        "beam_position": arr(beam.position).tolist(),
        "collision_position": arr(collision.position).tolist(),
    }
    # Distal-node velocity is applied once after deployment; no state reset after animate.
    velocity = arr(beam.velocity)
    velocity[:, 2] = 0.0
    velocity[1:, 2] = -1.0
    beam.velocity.value = velocity.tolist()
    records = []
    before["input_velocity"] = velocity.tolist()
    try:
        for i in range(20):
            t0 = time.perf_counter()
            Sofa.Simulation.animate(root, DT)
            beam_pos, coll_pos = arr(beam.position), arr(collision.position)
            beam_free, coll_free = arr(beam.free_position), arr(collision.free_position)
            rows = get_value(solver, "currentNumConstraints")
            records.append({
                "step": i + 1,
                "beam_position": beam_pos.tolist(),
                "collision_position": coll_pos.tolist(),
                "beam_min_z": float(np.min(beam_pos[:, 2])) if beam_pos.size else None,
                "collision_min_z": float(np.min(coll_pos[:, 2])) if coll_pos.size else None,
                "beam_correction_m": float(np.max(np.linalg.norm(
                    beam_pos[:, :3] - beam_free[:, :3], axis=1))) if beam_pos.size else None,
                "collision_correction_m": float(np.max(np.linalg.norm(
                    coll_pos - coll_free, axis=1))) if coll_pos.size else None,
                "contact_components": contacts(root),
                "constraint_rows": int(rows) if rows is not None else None,
                "solver_iterations": get_value(solver, "currentIterations"),
                "solver_error": get_value(solver, "currentError"),
                "finite": bool(np.isfinite(beam_pos).all() and np.isfinite(coll_pos).all()),
                "runtime_s": time.perf_counter() - t0,
            })
            if not records[-1]["finite"]:
                break
    finally:
        Sofa.Simulation.unload(root)
    return {"method": method, "initial": before, "records": records}


def main():
    for name in PLUGIN_NAMES:
        if not SofaRuntime.importPlugin(name):
            raise RuntimeError(f"Plugin unavailable: {name}")
    result = {"discrete": run("discrete"), "ccd": run("ccd")}
    a, b = result["discrete"], result["ccd"]
    same_initial = (
        np.array_equal(np.asarray(a["initial"]["beam_position"]),
                       np.asarray(b["initial"]["beam_position"]))
        and np.array_equal(np.asarray(a["initial"]["collision_position"]),
                           np.asarray(b["initial"]["collision_position"]))
        and a["initial"]["input_velocity"] == b["initial"]["input_velocity"]
    )
    result["summary"] = {
        "same_initial_state_and_velocity": bool(same_initial),
        "discrete_first_penetration_m": max(0.0, -a["records"][0]["collision_min_z"]),
        "ccd_first_penetration_m": max(0.0, -b["records"][0]["collision_min_z"]),
        "discrete_max_20_step_penetration_m": max(
            max(0.0, -r["collision_min_z"]) for r in a["records"]),
        "ccd_max_20_step_penetration_m": max(
            max(0.0, -r["collision_min_z"]) for r in b["records"]),
        "ccd_max_solver_iterations": max(r["solver_iterations"] for r in b["records"]),
        "ccd_max_solver_error": max(r["solver_error"] for r in b["records"]),
        "ccd_runtime_s": sum(r["runtime_s"] for r in b["records"]),
    }
    s = result["summary"]
    result["pass"] = bool(same_initial and s["discrete_first_penetration_m"] > 0.001
                          and s["ccd_max_20_step_penetration_m"] < 1e-5
                          and b["records"][0]["contact_components"] > 0
                          and b["records"][0]["constraint_rows"] > 0
                          and b["records"][0]["beam_correction_m"] > 0
                          and b["records"][0]["collision_correction_m"] > 0
                          and all(r["finite"] for r in b["records"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"pass": result["pass"], **s}))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
