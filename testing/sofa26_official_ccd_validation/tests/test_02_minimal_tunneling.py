#!/usr/bin/env python3
"""Minimal tunneling validation for official SOFA 26.06 TightInclusion CCD.

Four one-step cases are run with dt=5 ms:
  A. Point through a static triangle, discrete collision.
  B. Point through a static triangle, TightInclusion FreeMotion CCD.
  C. Long line segment through a static triangle while BOTH endpoints cross
     the plane outside the triangle footprint, discrete collision.
  D. Same line-interior case with TightInclusion FreeMotion CCD.

Case D is the critical catheter-body test.  The official v26.06 source natively
supports Triangle/Point but not Triangle/Line.  The geometry is designed so
PointCollisionModel endpoints cannot save the line interior.  If D tunnels,
official CCD as-is cannot provide a body nonpenetration guarantee for the
current mCR Triangle-vs-Line/Point representation.

No mCR production files are imported or modified.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS = ROOT_DIR / "_runtime" / "results"
OUT = RESULTS / "test_02_minimal_tunneling.json"

DT = 0.005
START_Z = 0.002
VELOCITY_Z = -0.8  # 4 mm travel in one 5 ms step: +2 mm -> -2 mm.
TOL_M = 1e-6
CONTACT_DISTANCE = 1e-5

TRIANGLE = np.array(
    [
        [-0.005, -0.006, 0.0],
        [ 0.005, -0.006, 0.0],
        [ 0.000,  0.006, 0.0],
    ],
    dtype=np.float64,
)

POINT_START = np.array([[0.0, 0.0, START_Z]], dtype=np.float64)
POINT_VELOCITY = np.array([[0.0, 0.0, VELOCITY_Z]], dtype=np.float64)

LINE_START = np.array(
    [
        [-0.020, 0.0, START_Z],
        [ 0.020, 0.0, START_Z],
    ],
    dtype=np.float64,
)
LINE_VELOCITY = np.array(
    [
        [0.0, 0.0, VELOCITY_Z],
        [0.0, 0.0, VELOCITY_Z],
    ],
    dtype=np.float64,
)

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
]


def _arr(data: Any) -> np.ndarray:
    if hasattr(data, "array"):
        try:
            return np.asarray(data.array(), dtype=np.float64).copy()
        except Exception:
            pass
    try:
        return np.asarray(data.value, dtype=np.float64).copy()
    except Exception:
        return np.asarray(data, dtype=np.float64).copy()


def _load_plugins(SofaRuntime: Any) -> None:
    failed = []
    for name in PLUGINS:
        try:
            result = SofaRuntime.importPlugin(name)
            if result is False:
                failed.append(name)
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    if failed:
        raise RuntimeError("plugin load failure: " + " | ".join(failed))


def _add_root_pipeline(root: Any, continuous: bool) -> None:
    root.dt.value = DT
    root.gravity.value = [0.0, 0.0, 0.0]

    root.addObject("FreeMotionAnimationLoop")
    root.addObject(
        "BlockGaussSeidelConstraintSolver",
        name="ConstraintSolver",
        maxIterations=100,
        tolerance=1e-10,
    )
    root.addObject("CollisionPipeline", name="Pipeline", depth=6)
    root.addObject("BruteForceBroadPhase", name="BroadPhase")
    root.addObject("BVHNarrowPhase", name="NarrowPhase")
    root.addObject(
        "CollisionResponse",
        name="ContactManager",
        response="FrictionContactConstraint",
        responseParams="mu=0.0",
    )

    if continuous:
        root.addObject(
            "CCDTightInclusionIntersection",
            name="Intersection",
            continuousCollisionType="FreeMotion",
            tolerance=1e-10,
            maxIterations=1000,
            alarmDistance=0.0,
            contactDistance=CONTACT_DISTANCE,
        )
    else:
        # Discrete control case.  LocalMinDistance is intentionally not used
        # because it introduces alarm-distance semantics; NewProximity is the
        # closest ordinary discrete proximity control in the same pipeline.
        root.addObject(
            "NewProximityIntersection",
            name="Intersection",
            alarmDistance=0.0,
            contactDistance=CONTACT_DISTANCE,
        )


def _add_static_triangle(root: Any) -> Any:
    node = root.addChild("StaticTriangle")
    node.addObject(
        "MeshTopology",
        name="topology",
        position=TRIANGLE.tolist(),
        triangles=[[0, 1, 2]],
    )
    node.addObject(
        "MechanicalObject",
        name="dofs",
        template="Vec3d",
        position=TRIANGLE.tolist(),
    )
    node.addObject(
        "TriangleCollisionModel",
        name="TriangleCM",
        moving=False,
        simulated=False,
        contactDistance=CONTACT_DISTANCE,
    )
    return node


def _add_dynamic_point(root: Any) -> Any:
    node = root.addChild("MovingPoint")
    node.addObject("EulerImplicitSolver")
    node.addObject(
        "SparseLDLSolver",
        name="ldl",
        template="CompressedRowSparseMatrixMat3x3",
    )
    node.addObject(
        "MeshTopology",
        name="topology",
        position=POINT_START.tolist(),
    )
    mo = node.addObject(
        "MechanicalObject",
        name="dofs",
        template="Vec3d",
        position=POINT_START.tolist(),
        velocity=POINT_VELOCITY.tolist(),
    )
    node.addObject("UniformMass", totalMass=1.0)
    node.addObject(
        "PointCollisionModel",
        name="PointCM",
        contactDistance=CONTACT_DISTANCE,
    )
    node.addObject(
        "LinearSolverConstraintCorrection",
        linearSolver="@ldl",
    )
    return mo


def _add_dynamic_line(root: Any) -> Any:
    node = root.addChild("MovingLine")
    node.addObject("EulerImplicitSolver")
    node.addObject(
        "SparseLDLSolver",
        name="ldl",
        template="CompressedRowSparseMatrixMat3x3",
    )
    node.addObject(
        "MeshTopology",
        name="topology",
        position=LINE_START.tolist(),
        edges=[[0, 1]],
    )
    mo = node.addObject(
        "MechanicalObject",
        name="dofs",
        template="Vec3d",
        position=LINE_START.tolist(),
        velocity=LINE_VELOCITY.tolist(),
    )
    node.addObject("UniformMass", totalMass=1.0)
    node.addObject(
        "LineCollisionModel",
        name="LineCM",
        contactDistance=CONTACT_DISTANCE,
    )
    # Keep the same representation style as mCR: line body plus sampled points.
    node.addObject(
        "PointCollisionModel",
        name="PointCM",
        contactDistance=CONTACT_DISTANCE,
    )
    node.addObject(
        "LinearSolverConstraintCorrection",
        linearSolver="@ldl",
    )
    return mo


def _point_in_triangle_xy(p: np.ndarray, tri: np.ndarray) -> bool:
    p = np.asarray(p[:2], dtype=np.float64)
    a, b, c = tri[:, :2]

    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = float(v0 @ v0)
    dot01 = float(v0 @ v1)
    dot02 = float(v0 @ v2)
    dot11 = float(v1 @ v1)
    dot12 = float(v1 @ v2)
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-20:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -1e-12 and v >= -1e-12 and (u + v) <= 1.0 + 1e-12


def _independent_point_sweep_expected() -> bool:
    z1 = START_Z + VELOCITY_Z * DT
    crosses = START_Z > 0.0 and z1 < 0.0
    return bool(crosses and _point_in_triangle_xy(np.array([0.0, 0.0, 0.0]), TRIANGLE))


def _independent_line_interior_sweep_expected() -> bool:
    z1 = START_Z + VELOCITY_Z * DT
    if not (START_Z > 0.0 and z1 < 0.0):
        return False

    # At the time of plane crossing the line is x in [-20,+20] mm, y=0, z=0.
    # The origin lies on its interior and is strictly inside the triangle.
    origin_inside_triangle = _point_in_triangle_xy(
        np.array([0.0, 0.0, 0.0]), TRIANGLE
    )
    endpoints_outside_triangle = (
        not _point_in_triangle_xy(np.array([-0.020, 0.0, 0.0]), TRIANGLE)
        and not _point_in_triangle_xy(np.array([0.020, 0.0, 0.0]), TRIANGLE)
    )
    return bool(origin_inside_triangle and endpoints_outside_triangle)


def _run_case(kind: str, continuous: bool) -> dict[str, Any]:
    import Sofa
    import SofaRuntime

    _load_plugins(SofaRuntime)

    root = Sofa.Core.Node(
        f"{'ccd' if continuous else 'discrete'}_{kind}_tunneling"
    )
    _add_root_pipeline(root, continuous=continuous)
    _add_static_triangle(root)

    if kind == "point":
        mo = _add_dynamic_point(root)
        start = POINT_START
        sweep_expected = _independent_point_sweep_expected()
    elif kind == "line":
        mo = _add_dynamic_line(root)
        start = LINE_START
        sweep_expected = _independent_line_interior_sweep_expected()
    else:
        raise ValueError(kind)

    result: dict[str, Any] = {
        "kind": kind,
        "continuous": continuous,
        "continuousCollisionType": "FreeMotion" if continuous else "None",
        "dt_s": DT,
        "start_position": start.tolist(),
        "start_min_z_m": float(np.min(start[:, 2])),
        "unconstrained_end_z_m": float(START_Z + VELOCITY_Z * DT),
        "independent_swept_intersection_expected": bool(sweep_expected),
    }

    try:
        Sofa.Simulation.init(root)
        result["after_init_position"] = _arr(mo.position).tolist()
        result["after_init_velocity"] = _arr(mo.velocity).tolist()

        Sofa.Simulation.animate(root, DT)

        final_pos = _arr(mo.position)
        final_vel = _arr(mo.velocity)
        result["final_position"] = final_pos.tolist()
        result["final_velocity"] = final_vel.tolist()
        result["final_min_z_m"] = float(np.min(final_pos[:, 2]))
        result["final_max_z_m"] = float(np.max(final_pos[:, 2]))
        result["finite"] = bool(
            np.all(np.isfinite(final_pos)) and np.all(np.isfinite(final_vel))
        )

        # For these deliberately one-sided sweeps, ending fully below the
        # zero-thickness triangle after an independently known swept crossing
        # means temporal tunneling occurred.
        fully_other_side = bool(np.max(final_pos[:, 2]) < -TOL_M)
        result["fully_on_other_side"] = fully_other_side
        result["tunneled"] = bool(sweep_expected and fully_other_side)

        if not result["finite"]:
            result["case_decision"] = "FAIL"
            result["case_reason"] = "NON_FINITE_STATE"
        elif result["tunneled"]:
            result["case_decision"] = "FAIL"
            result["case_reason"] = "SWEPT_INTERSECTION_BUT_FINAL_STATE_TUNNELED"
        else:
            result["case_decision"] = "PASS"
            result["case_reason"] = "NO_COMPLETE_TUNNEL_OBSERVED"
    except Exception as exc:
        result["case_decision"] = "INCONCLUSIVE"
        result["case_reason"] = f"RUNTIME_ERROR: {type(exc).__name__}: {exc}"
    finally:
        try:
            Sofa.Simulation.unload(root)
        except Exception:
            pass

    return result


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "test": "SOFA26 official TightInclusion minimal tunneling",
        "official_reference_tag": "sofa-framework/sofa v26.06.00",
        "dt_s": DT,
        "triangle_line_native_support_v26_06": False,
        "triangle_point_native_support_v26_06": True,
        "cases": [],
    }

    try:
        import Sofa  # noqa: F401
        import SofaRuntime  # noqa: F401
    except Exception as exc:
        payload["decision"] = "INCONCLUSIVE"
        payload["reason"] = f"SOFA_IMPORT_FAILED: {type(exc).__name__}: {exc}"
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return

    cases = [
        _run_case("point", continuous=False),
        _run_case("point", continuous=True),
        _run_case("line", continuous=False),
        _run_case("line", continuous=True),
    ]
    payload["cases"] = cases

    lookup = {(c["kind"], c["continuous"]): c for c in cases}
    point_ccd = lookup[("point", True)]
    line_ccd = lookup[("line", True)]

    if any(c["case_decision"] == "INCONCLUSIVE" for c in cases):
        decision = "INCONCLUSIVE"
        reason = "AT_LEAST_ONE_MINIMAL_SCENE_DID_NOT_EXECUTE"
    elif point_ccd["case_decision"] != "PASS":
        decision = "FAIL"
        reason = "OFFICIAL_TRIANGLE_POINT_CCD_DID_NOT_PREVENT_POINT_TUNNEL"
    elif line_ccd["case_decision"] != "PASS":
        decision = "FAIL_BODY_GUARANTEE"
        reason = "LINE_INTERIOR_TUNNELED_TRIANGLE_POINT_CCD_CANNOT_COVER_BODY_INTERIOR"
    else:
        decision = "PASS"
        reason = "POINT_AND_LINE_INTERIOR_MINIMAL_TUNNELING_CASES_PREVENTED"

    payload["decision"] = decision
    payload["reason"] = reason
    payload["migration_implication"] = (
        "OFFICIAL_CCD_AS_IS_NOT_SUFFICIENT_FOR_MCR_BODY_NONPENETRATION"
        if decision == "FAIL_BODY_GUARANTEE"
        else (
            "PROCEED_TO_BEAMADAPTER_MAPPING_AUDIT"
            if decision == "PASS"
            else "DO_NOT_MIGRATE_YET"
        )
    )

    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
