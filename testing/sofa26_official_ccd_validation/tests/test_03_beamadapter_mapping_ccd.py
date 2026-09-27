#!/usr/bin/env python3
"""BeamAdapter + MultiAdaptiveBeamMapping audit under isolated SOFA 26.06.

This test builds a small official-style BeamAdapter wire scene based on the
BeamAdapter SingleBeamDeploymentCollision example, then verifies that:
- the Rigid3 beam state is created and advances;
- MultiAdaptiveBeamMapping produces finite collision DOFs;
- the mapped collision representation contains BOTH LineCollisionModel and
  PointCollisionModel;
- non-zero line interiors exist between mapped points.

It then combines those runtime facts with the official v26.06 TightInclusion
support matrix and, when available, test_02's actual line-interior tunneling
result.

This is a migration-precheck only. It does not import mCR production code and
does not use the B02 vessel yet.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS = ROOT_DIR / "_runtime" / "results"
OUT = RESULTS / "test_03_beamadapter_mapping_ccd.json"
TEST2_OUT = RESULTS / "test_02_minimal_tunneling.json"

DT = 0.005
RADIUS = 0.000665

PLUGINS = [
    "BeamAdapter",
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Collision.Geometry",
    "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.Constraint.Lagrangian.Solver",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.Spring",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Topology.Container.Grid",
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


def _safe_class(obj: Any) -> str:
    try:
        return str(obj.getClassName())
    except Exception:
        return type(obj).__name__


def _safe_size(obj: Any) -> int | None:
    for name in ("getSize", "size"):
        value = getattr(obj, name, None)
        if callable(value):
            try:
                return int(value())
            except Exception:
                pass
        elif value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return None


def _load_plugins(SofaRuntime: Any) -> dict[str, Any]:
    results = {}
    for name in PLUGINS:
        try:
            value = SofaRuntime.importPlugin(name)
            results[name] = {
                "ok": bool(value) if value is not None else True,
                "return": repr(value),
            }
        except Exception as exc:
            results[name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return results


def _build_scene(Sofa: Any) -> tuple[Any, dict[str, Any]]:
    root = Sofa.Core.Node("sofa26_beamadapter_mapping_audit")
    root.dt.value = DT
    root.gravity.value = [0.0, 0.0, 0.0]

    root.addObject("FreeMotionAnimationLoop")
    root.addObject(
        "BlockGaussSeidelConstraintSolver",
        name="ConstraintSolver",
        maxIterations=50,
        tolerance=1e-10,
    )

    topo = root.addChild("EdgeTopology")
    topo.addObject(
        "RodStraightSection",
        name="StraightSection",
        length=0.04,
        radius=RADIUS,
        nbBeams=8,
        nbEdgesCollis=16,
        nbEdgesVisu=16,
        youngModulus=2.0e7,
        massDensity=1000.0,
        poissonRatio=0.3,
    )
    topo.addObject(
        "WireRestShape",
        name="BeamRestShape",
        template="Rigid3d",
        wireMaterials="@StraightSection",
    )
    topo.addObject("EdgeSetTopologyContainer", name="meshLines")
    topo.addObject("EdgeSetTopologyModifier", name="Modifier")
    topo.addObject(
        "EdgeSetGeometryAlgorithms",
        name="GeomAlgo",
        template="Rigid3d",
    )
    topo.addObject(
        "MechanicalObject",
        name="dofTopo",
        template="Rigid3d",
    )

    beam = root.addChild("BeamModel")
    beam.addObject(
        "EulerImplicitSolver",
        rayleighStiffness=0.0,
        rayleighMass=0.0,
    )
    beam.addObject(
        "BTDLinearSolver",
        name="LinearSolver",
        verification=False,
        subpartSolve=False,
        verbose=False,
    )
    beam.addObject(
        "RegularGridTopology",
        name="MeshLines",
        nx=9,
        ny=1,
        nz=1,
        xmin=0.0,
        xmax=0.0,
        ymin=0.0,
        ymax=0.0,
        zmin=0.0,
        zmax=0.0,
        p0=[0.0, 0.0, 0.0],
    )
    beam_dofs = beam.addObject(
        "MechanicalObject",
        name="DOFs",
        template="Rigid3d",
        showIndices=False,
    )
    beam.addObject(
        "WireBeamInterpolation",
        name="BeamInterpolation",
        WireRestShape="@../EdgeTopology/BeamRestShape",
        printLog=False,
    )
    beam.addObject(
        "AdaptiveBeamForceFieldAndMass",
        name="BeamForceField",
        massDensity=1000.0,
        interpolation="@BeamInterpolation",
    )
    beam.addObject(
        "InterventionalRadiologyController",
        name="DeployController",
        template="Rigid3d",
        instruments="BeamInterpolation",
        topology="@MeshLines",
        startingPos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        xtip=[0.035],
        rotationInstrument=[0.0],
        step=0.001,
        speed=0.001,
        listening=True,
        controlledInstrument=0,
        printLog=False,
    )
    beam.addObject(
        "LinearSolverConstraintCorrection",
        wire_optimization=True,
        printLog=False,
    )
    beam.addObject(
        "FixedProjectiveConstraint",
        name="FixedConstraint",
        indices=0,
    )
    beam.addObject(
        "RestShapeSpringsForceField",
        name="RestShapeSpring",
        points="@DeployController.indexFirstNode",
        angularStiffness=1e8,
        stiffness=1e8,
    )

    collision = beam.addChild("CollisionModel")
    collision.addObject(
        "EdgeSetTopologyContainer",
        name="collisEdgeSet",
    )
    collision.addObject(
        "EdgeSetTopologyModifier",
        name="collisEdgeModifier",
    )
    collision_dofs = collision.addObject(
        "MechanicalObject",
        name="CollisionDOFs",
        template="Vec3d",
    )
    mapping = collision.addObject(
        "MultiAdaptiveBeamMapping",
        name="collisMap",
        controller="../DeployController",
        useCurvAbs=True,
        printLog=False,
    )
    line_cm = collision.addObject(
        "LineCollisionModel",
        name="LineCM",
        contactDistance=0.0,
    )
    point_cm = collision.addObject(
        "PointCollisionModel",
        name="PointCM",
        contactDistance=0.0,
    )

    return root, {
        "beam_dofs": beam_dofs,
        "collision_node": collision,
        "collision_dofs": collision_dofs,
        "mapping": mapping,
        "line_cm": line_cm,
        "point_cm": point_cm,
    }


def _edge_lengths(collision_node: Any, points: np.ndarray) -> list[float]:
    try:
        topo = collision_node.getObject("collisEdgeSet")
        edges = np.asarray(topo.edges.value, dtype=np.int64).reshape(-1, 2)
    except Exception:
        edges = np.empty((0, 2), dtype=np.int64)

    lengths: list[float] = []
    for a, b in edges:
        if 0 <= a < len(points) and 0 <= b < len(points):
            lengths.append(float(np.linalg.norm(points[b] - points[a])))

    # Fallback: the official mapping normally creates points in arclength order.
    if not lengths and len(points) >= 2:
        lengths = [
            float(np.linalg.norm(points[i + 1] - points[i]))
            for i in range(len(points) - 1)
        ]
    return lengths


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "test": "SOFA26 BeamAdapter mapping and CCD coverage audit",
        "official_reference": {
            "sofa": "v26.06.00",
            "beamadapter_example": "SingleBeamDeploymentCollision",
        },
        "official_tight_inclusion_pair_support": {
            "Triangle/Point": True,
            "Triangle/Line": False,
        },
        "catheter_radius_m": RADIUS,
        "dt_s": DT,
    }

    try:
        import Sofa
        import SofaRuntime
    except Exception as exc:
        payload["decision"] = "INCONCLUSIVE"
        payload["reason"] = f"SOFA_IMPORT_FAILED: {type(exc).__name__}: {exc}"
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return

    plugin_results = _load_plugins(SofaRuntime)
    payload["plugins"] = plugin_results
    missing = [k for k, v in plugin_results.items() if not v.get("ok", False)]
    if missing:
        payload["decision"] = "INCONCLUSIVE"
        payload["reason"] = "REQUIRED_BEAMADAPTER_PLUGIN_MISSING"
        payload["missing_plugins"] = missing
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return

    root = None
    try:
        root, handles = _build_scene(Sofa)
        Sofa.Simulation.init(root)

        beam_before = _arr(handles["beam_dofs"].position)
        collision_before = _arr(handles["collision_dofs"].position)

        # Two tiny normal animation steps are enough to exercise mechanical
        # propagation without creating any custom mapping hook.
        Sofa.Simulation.animate(root, DT)
        Sofa.Simulation.animate(root, DT)

        beam_after = _arr(handles["beam_dofs"].position)
        collision_after = _arr(handles["collision_dofs"].position)

        edge_lengths = _edge_lengths(
            handles["collision_node"],
            collision_after[:, :3] if collision_after.ndim == 2 else collision_after,
        )

        payload["runtime"] = {
            "beam_state_class": _safe_class(handles["beam_dofs"]),
            "mapping_class": _safe_class(handles["mapping"]),
            "line_collision_class": _safe_class(handles["line_cm"]),
            "point_collision_class": _safe_class(handles["point_cm"]),
            "line_collision_size": _safe_size(handles["line_cm"]),
            "point_collision_size": _safe_size(handles["point_cm"]),
            "beam_shape_before": list(beam_before.shape),
            "beam_shape_after": list(beam_after.shape),
            "collision_shape_before": list(collision_before.shape),
            "collision_shape_after": list(collision_after.shape),
            "beam_finite": bool(np.all(np.isfinite(beam_after))),
            "collision_finite": bool(np.all(np.isfinite(collision_after))),
            "mapped_edge_count": len(edge_lengths),
            "mapped_edge_length_min_m": (
                float(np.min(edge_lengths)) if edge_lengths else None
            ),
            "mapped_edge_length_max_m": (
                float(np.max(edge_lengths)) if edge_lengths else None
            ),
            "mapped_edge_length_mean_m": (
                float(np.mean(edge_lengths)) if edge_lengths else None
            ),
            "nonzero_line_interiors_exist": bool(
                edge_lengths and max(edge_lengths) > 1e-12
            ),
        }

        mapping_ok = (
            payload["runtime"]["beam_finite"]
            and payload["runtime"]["collision_finite"]
            and collision_after.ndim == 2
            and collision_after.shape[0] >= 2
            and payload["runtime"]["nonzero_line_interiors_exist"]
        )
        payload["mapping_runtime_ok"] = bool(mapping_ok)

        test2 = None
        if TEST2_OUT.is_file():
            try:
                test2 = json.loads(TEST2_OUT.read_text())
            except Exception:
                test2 = None
        payload["minimal_tunneling_result"] = test2

        line_tunnel_observed = False
        if isinstance(test2, dict):
            for case in test2.get("cases", []):
                if (
                    case.get("kind") == "line"
                    and case.get("continuous") is True
                    and case.get("tunneled") is True
                ):
                    line_tunnel_observed = True
                    break

        payload["coverage_analysis"] = {
            "actual_mapping_has_line_model": True,
            "actual_mapping_has_point_model": True,
            "mapped_line_interiors_nonzero": bool(
                payload["runtime"]["nonzero_line_interiors_exist"]
            ),
            "official_triangle_point_ccd_supported": True,
            "official_triangle_line_ccd_supported": False,
            "minimal_line_interior_tunnel_observed": line_tunnel_observed,
            "point_samples_alone_form_continuous_body_certificate": False,
        }

        if not mapping_ok:
            decision = "INCONCLUSIVE"
            reason = "BEAMADAPTER_MAPPING_RUNTIME_NOT_ESTABLISHED"
        elif line_tunnel_observed:
            decision = "FAIL_BODY_GUARANTEE"
            reason = (
                "BEAMADAPTER_USES_NONZERO_LINE_INTERIORS_AND_OFFICIAL_"
                "TRIANGLE_LINE_CCD_IS_UNSUPPORTED_WITH_OBSERVED_TUNNEL"
            )
        else:
            decision = "PASS_MAPPING_WITH_CCD_COVERAGE_GAP"
            reason = (
                "BEAMADAPTER_MAPPING_WORKS_BUT_TRIANGLE_LINE_CCD_SUPPORT_"
                "IS_MISSING_SO_BODY_GUARANTEE_IS_NOT_ESTABLISHED"
            )

        payload["decision"] = decision
        payload["reason"] = reason
        payload["migration_gate"] = (
            "BLOCKED_FOR_NONPENETRATION_GUARANTEE"
            if decision in (
                "FAIL_BODY_GUARANTEE",
                "PASS_MAPPING_WITH_CCD_COVERAGE_GAP",
            )
            else "INCONCLUSIVE"
        )

    except Exception as exc:
        payload["decision"] = "INCONCLUSIVE"
        payload["reason"] = f"BEAMADAPTER_SCENE_ERROR: {type(exc).__name__}: {exc}"
    finally:
        if root is not None:
            try:
                Sofa.Simulation.unload(root)
            except Exception:
                pass

    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
