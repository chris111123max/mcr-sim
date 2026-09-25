#!/usr/bin/env python3
"""Run identical point/triangle crossings with discrete and continuous detection."""
import json
import math
import time
from pathlib import Path

import numpy as np
import Sofa
import SofaRuntime


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "_runtime/results"
DT = 0.005
INITIAL_Z = 0.001
INITIAL_VZ = -1.0
WALL_Z = 0.0
CONTACT_DISTANCE = 1e-5
ALARM_DISTANCE = 1e-4
PLUGINS = [
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Collision.Detection.Algorithm",
    "Sofa.Component.Collision.Detection.Intersection",
    "Sofa.Component.Collision.Geometry",
    "Sofa.Component.Collision.Response.Contact",
    "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.Constraint.Lagrangian.Solver",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.Mass",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Constant",
    "Sofa.Component.MechanicalLoad",
]


def array(data):
    try:
        return np.asarray(data.array(), dtype=float).copy()
    except Exception:
        return np.asarray(data.value, dtype=float).copy()


def field(obj, name):
    try:
        return obj.findData(name).value
    except Exception:
        return None


def contact_count(node):
    count = sum(obj.getClassName() == "FrictionContact" for obj in node.objects)
    for child in node.children:
        count += contact_count(child)
    return int(count)


def make_scene(method):
    root = Sofa.Core.Node("root")
    root.dt.value = DT
    root.gravity.value = [0.0, 0.0, 0.0]
    root.addObject("FreeMotionAnimationLoop")
    solver = root.addObject(
        "BlockGaussSeidelConstraintSolver", name="constraints",
        maxIterations=1000, tolerance=1e-9,
    )
    root.addObject("CollisionPipeline")
    root.addObject("BruteForceBroadPhase")
    root.addObject("BVHNarrowPhase")
    if method == "discrete":
        root.addObject("NewProximityIntersection", alarmDistance=ALARM_DISTANCE,
                       contactDistance=CONTACT_DISTANCE)
    else:
        root.addObject("CCDTightInclusionIntersection",
                       continuousCollisionType="FreeMotion",
                       alarmDistance=ALARM_DISTANCE,
                       contactDistance=CONTACT_DISTANCE,
                       maxIterations=1000)
    root.addObject("CollisionResponse", response="FrictionContactConstraint",
                   responseParams="mu=0.0")

    moving = root.addChild("moving_point")
    moving.addObject("EulerImplicitSolver")
    moving.addObject("SparseLDLSolver", name="linear", template="CompressedRowSparseMatrixd")
    dofs = moving.addObject(
        "MechanicalObject", name="dofs", template="Vec3d",
        position=f"0 0 {INITIAL_Z}", velocity=f"0 0 {INITIAL_VZ}",
    )
    moving.addObject("UniformMass", totalMass=1.0)
    moving.addObject("ConstantForceField", forces="0 0 -1")
    moving.addObject("PointCollisionModel", contactDistance=0.0)
    moving.addObject("LinearSolverConstraintCorrection")

    wall = root.addChild("static_triangle")
    vertices = "-0.01 -0.01 0 0.01 -0.01 0 0 0.01 0"
    wall.addObject("MeshTopology", position=vertices, triangles="0 1 2")
    wall.addObject("MechanicalObject", template="Vec3d", position=vertices)
    wall.addObject("TriangleCollisionModel", moving=False, simulated=False,
                   bothSide=True, contactDistance=0.0)
    return root, dofs, solver


def step_record(root, dofs, solver, index):
    start = time.perf_counter()
    Sofa.Simulation.animate(root, DT)
    elapsed = time.perf_counter() - start
    position = array(dofs.position)
    velocity = array(dofs.velocity)
    free = array(dofs.free_position)
    z = float(position[0, 2])
    correction = float(np.max(np.linalg.norm(position - free, axis=1)))
    rows = field(solver, "currentNumConstraints")
    iterations = field(solver, "currentIterations")
    error = field(solver, "currentError")
    data = {
        "step": index,
        "position": position[0].tolist(),
        "velocity": velocity[0].tolist(),
        "free_position": free[0].tolist(),
        "clearance_m": z - WALL_Z,
        "penetration_m": max(0.0, WALL_Z - z),
        "correction_m": correction,
        "contact_components": contact_count(root),
        "constraint_rows": int(rows) if rows is not None else None,
        "solver_iterations": int(iterations) if iterations is not None else None,
        "solver_error": float(error) if error is not None else None,
        "finite": bool(np.isfinite(position).all() and np.isfinite(velocity).all()),
        "runtime_s": elapsed,
    }
    return data


def run(method, steps):
    root, dofs, solver = make_scene(method)
    Sofa.Simulation.init(root)
    initial = array(dofs.position)[0].tolist()
    initial_velocity = array(dofs.velocity)[0].tolist()
    records = []
    try:
        for index in range(1, steps + 1):
            records.append(step_record(root, dofs, solver, index))
            if not records[-1]["finite"]:
                break
    finally:
        Sofa.Simulation.unload(root)
    return {
        "method": method, "dt_s": DT, "initial_position": initial,
        "initial_velocity": initial_velocity,
        "predicted_free_position_without_force": [0.0, 0.0, INITIAL_Z + INITIAL_VZ * DT],
        "alarm_distance_m": ALARM_DISTANCE,
        "contact_distance_m": CONTACT_DISTANCE,
        "records": records,
    }


def summarize(records):
    pen = np.array([r["penetration_m"] for r in records])
    return {
        "steps": len(records),
        "max_penetration_m": float(np.max(pen)),
        "p95_penetration_m": float(np.percentile(pen, 95)),
        "p99_penetration_m": float(np.percentile(pen, 99)),
        "contact_active_steps": sum(r["contact_components"] > 0 for r in records),
        "constraint_active_steps": sum((r["constraint_rows"] or 0) > 0 for r in records),
        "max_constraint_rows": max((r["constraint_rows"] or 0) for r in records),
        "max_solver_iterations": max((r["solver_iterations"] or 0) for r in records),
        "max_solver_error": max((r["solver_error"] or 0.0) for r in records),
        "all_finite": all(r["finite"] for r in records),
        "runtime_s": sum(r["runtime_s"] for r in records),
    }


def main():
    for name in PLUGINS:
        if not SofaRuntime.importPlugin(name):
            raise RuntimeError(f"Could not load {name}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    discrete = run("discrete", 1)
    ccd = run("ccd", 1)
    result = {"discrete": discrete, "ccd": ccd}
    result["pass"] = (
        discrete["records"][0]["penetration_m"] > 0.001
        and ccd["records"][0]["penetration_m"] < 1e-5
        and (ccd["records"][0]["constraint_rows"] or 0) > 0
    )
    (RESULTS / "ccd_single_step_ab.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if result["pass"]:
        long_run = run("ccd", 1000)
        snapshots = {str(n): summarize(long_run["records"][:n])
                     for n in (20, 100, 1000)}
        long_result = {"method": "ccd", "snapshots": snapshots,
                       "records": long_run["records"],
                       "pass": snapshots["1000"]["max_penetration_m"] < 1e-5
                       and snapshots["1000"]["all_finite"]}
    else:
        long_result = {"skipped": "single_step_ccd_not_passed"}
    (RESULTS / "ccd_multistep.json").write_text(
        json.dumps(long_result, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "single_step_pass": result["pass"],
        "discrete_final": discrete["records"][0],
        "ccd_final": ccd["records"][0],
        "multistep": long_result.get("snapshots", long_result),
    }))


if __name__ == "__main__":
    main()
