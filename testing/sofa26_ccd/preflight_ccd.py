"""SOFA v26.06 Tight-Inclusion CCD feasibility test.

This is intentionally isolated from the MCR environment and training code.

The test performs two one-step experiments with the same fast moving point:
1. ordinary discrete proximity intersection;
2. CCDTightInclusionIntersection using FreeMotion.

The point starts above a static triangle wall and its free motion crosses the
wall within one 5 ms step.  The CCD case is expected to retain the point on the
original side of the wall while the discrete case is expected to tunnel.

It also verifies that the v26.06 BeamAdapter plugin can be loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime


DT = 0.005
INITIAL_Z = 0.004
INITIAL_VZ = -4.0
CONTACT_DISTANCE = 1.0e-5


CORE_PLUGINS = [
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Collision.Detection.Algorithm",
    "Sofa.Component.Collision.Detection.Intersection",
    "Sofa.Component.Collision.Geometry",
    "Sofa.Component.Collision.Response.Contact",
    "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.Constraint.Lagrangian.Solver",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.Mass",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Dynamic",
]


def _plugin_paths() -> list[str]:
    runtime = Path(os.environ["SOFA26_RUNTIME"]).resolve()
    build = runtime / "build"
    return [
        str(build),
        str(build / "external_directories" / "SofaPython3"),
        str(build / "external_directories" / "BeamAdapter"),
        str(runtime / "install" / "plugins"),
    ]


def _load_plugins() -> dict:
    for path in _plugin_paths():
        if Path(path).exists():
            SofaRuntime.PluginRepository.addFirstPath(path)

    loaded = {}
    for plugin in CORE_PLUGINS:
        loaded[plugin] = bool(SofaRuntime.importPlugin(plugin))

    # BeamAdapter is not needed for the particle wall-crossing itself. Loading
    # it here verifies the exact plugin needed by the catheter migration.
    loaded["BeamAdapter"] = bool(SofaRuntime.importPlugin("BeamAdapter"))
    return loaded


def _rows(dofs) -> int:
    try:
        raw = str(dofs.constraint.value)
    except Exception:
        return -1
    return len([line for line in raw.splitlines() if line.strip()])


def _build_root(use_ccd: bool):
    root = Sofa.Core.Node("root")
    root.dt.value = DT
    root.gravity.value = [0.0, 0.0, 0.0]

    root.addObject("FreeMotionAnimationLoop")
    root.addObject(
        "BlockGaussSeidelConstraintSolver",
        name="constraintSolver",
        maxIterations=100,
        tolerance=1.0e-9,
    )
    root.addObject("CollisionPipeline", name="pipeline")
    root.addObject("BruteForceBroadPhase", name="broadPhase")
    root.addObject("BVHNarrowPhase", name="narrowPhase")
    root.addObject(
        "CollisionResponse",
        name="contactManager",
        response="FrictionContactConstraint",
        responseParams="mu=0.0",
    )

    if use_ccd:
        root.addObject(
            "CCDTightInclusionIntersection",
            name="intersection",
            continuousCollisionType="FreeMotion",
            maxIterations=100,
            alarmDistance=0.0,
            contactDistance=CONTACT_DISTANCE,
        )
    else:
        root.addObject(
            "NewProximityIntersection",
            name="intersection",
            alarmDistance=0.0,
            contactDistance=CONTACT_DISTANCE,
        )

    moving = root.addChild("movingPoint")
    moving.addObject("EulerImplicitSolver", rayleighStiffness=0.0, rayleighMass=0.0)
    moving.addObject(
        "CGLinearSolver",
        name="linearSolver",
        iterations=100,
        tolerance=1.0e-12,
        threshold=1.0e-12,
    )
    dofs = moving.addObject(
        "MechanicalObject",
        name="dofs",
        template="Vec3d",
        position=[[0.0, 0.0, INITIAL_Z]],
        velocity=[[0.0, 0.0, INITIAL_VZ]],
    )
    moving.addObject("UniformMass", totalMass=0.001)
    moving.addObject("PointCollisionModel", contactDistance=0.0)
    moving.addObject("LinearSolverConstraintCorrection", linearSolver="@linearSolver")

    wall = root.addChild("wall")
    wall.addObject(
        "TriangleSetTopologyContainer",
        name="topology",
        position=[
            [-0.05, -0.05, 0.0],
            [0.05, -0.05, 0.0],
            [0.05, 0.05, 0.0],
            [-0.05, 0.05, 0.0],
        ],
        triangles=[[0, 1, 2], [0, 2, 3]],
    )
    wall.addObject(
        "MechanicalObject",
        name="dofs",
        template="Vec3d",
        position="@topology.position",
    )
    wall.addObject(
        "TriangleCollisionModel",
        name="collision",
        contactDistance=CONTACT_DISTANCE,
        moving=False,
        simulated=False,
    )

    return root, dofs


def _run_case(use_ccd: bool) -> dict:
    root, dofs = _build_root(use_ccd)
    Sofa.Simulation.init(root)

    before = np.asarray(dofs.position.array(), dtype=np.float64)[0, :3].copy()
    predicted_no_contact_z = float(INITIAL_Z + INITIAL_VZ * DT)

    Sofa.Simulation.animate(root, DT)

    after = np.asarray(dofs.position.array(), dtype=np.float64)[0, :3].copy()
    velocity_after = np.asarray(dofs.velocity.array(), dtype=np.float64)[0, :3].copy()

    result = {
        "mode": "ccd" if use_ccd else "discrete",
        "initial_position_m": before.tolist(),
        "initial_velocity_m_per_s": [0.0, 0.0, INITIAL_VZ],
        "predicted_no_contact_z_m": predicted_no_contact_z,
        "final_position_m": after.tolist(),
        "final_velocity_m_per_s": velocity_after.tolist(),
        "constraint_rows": _rows(dofs),
        "finite": bool(np.all(np.isfinite(after)) and np.all(np.isfinite(velocity_after))),
        "crossed_wall": bool(after[2] < -1.0e-6),
    }

    try:
        Sofa.Simulation.unload(root)
    except Exception:
        pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    loaded = _load_plugins()

    missing = [name for name, ok in loaded.items() if not ok]
    if missing:
        raise RuntimeError(f"Required plugins failed to load: {missing}")

    discrete = _run_case(use_ccd=False)
    ccd = _run_case(use_ccd=True)

    ccd_pass = (
        ccd["finite"]
        and not ccd["crossed_wall"]
        and ccd["final_position_m"][2] >= -1.0e-6
    )
    demonstration_pass = (
        discrete["finite"]
        and discrete["crossed_wall"]
        and ccd_pass
    )

    result = {
        "test": "SOFA v26.06 Tight-Inclusion CCD one-step tunneling preflight",
        "dt_s": DT,
        "initial_z_m": INITIAL_Z,
        "initial_vz_m_per_s": INITIAL_VZ,
        "contact_distance_m": CONTACT_DISTANCE,
        "plugins": loaded,
        "discrete": discrete,
        "ccd": ccd,
        "ccd_nonpenetration_pass": ccd_pass,
        "discrete_vs_ccd_demonstration_pass": demonstration_pass,
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))

    if not ccd_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
