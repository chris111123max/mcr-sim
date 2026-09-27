#!/usr/bin/env python3
"""SOFA 26.06 official collision/BeamAdapter component audit.

Run only inside an isolated SOFA 26.06 Python environment.  This test does not
modify the mCR production environment.

It checks:
- actual runtime version;
- importability of the official plugins needed by the migration candidate;
- creation of the key official components;
- the official v26.06 TightInclusion primitive-pair support matrix relevant to
  the catheter/vessel representation.

The support matrix below is copied from the official v26.06.00
CCDTightInclusionIntersection source:
  supported: Cube/Cube, Line/Line, Triangle/Point
  ignored:   Triangle/Line and several other pairs.

This test intentionally records Triangle/Line as NOT SUPPORTED unless the
runtime demonstrates an externally registered extension in a later test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "_runtime" / "results"
OUT = RESULTS / "test_01_component_audit.json"

EXPECTED_SOFA_PREFIX = "26.06"

REQUIRED_PLUGINS = [
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
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Topology.Container.Grid",
    "BeamAdapter",
]

OFFICIAL_V26_06_TIGHT_INCLUSION_SUPPORT = {
    "Cube/Cube": True,
    "Line/Line": True,
    "Triangle/Point": True,
    "Triangle/Line": False,
    "Line/Point": False,
    "Triangle/Triangle": False,
    "Triangle/Sphere": False,
}


def _runtime_version(Sofa: Any) -> str:
    candidates = [
        getattr(Sofa, "__version__", None),
        getattr(Sofa, "version", None),
    ]
    for value in candidates:
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        text = str(value)
        if text and text.lower() != "none":
            return text

    # Some SofaPython3 builds expose version helpers below Sofa.Core.
    try:
        core = Sofa.Core
        for name in ("getSofaVersion", "getVersion", "version"):
            fn = getattr(core, name, None)
            if callable(fn):
                try:
                    value = fn()
                except Exception:
                    continue
                if value:
                    return str(value)
    except Exception:
        pass

    return "UNKNOWN"


def _create_probe(root: Any, component: str, **kwargs: Any) -> dict[str, Any]:
    try:
        obj = root.addObject(component, name=f"probe_{component}", **kwargs)
        return {
            "created": True,
            "class_name": str(obj.getClassName()) if hasattr(obj, "getClassName") else type(obj).__name__,
        }
    except Exception as exc:
        return {
            "created": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "test": "SOFA26 official component audit",
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "official_reference_tag": "sofa-framework/sofa v26.06.00",
        "official_tight_inclusion_support": OFFICIAL_V26_06_TIGHT_INCLUSION_SUPPORT,
    }

    try:
        import Sofa
        import SofaRuntime
    except Exception as exc:
        payload.update(
            decision="INCONCLUSIVE",
            reason=f"SOFA_IMPORT_FAILED: {type(exc).__name__}: {exc}",
        )
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return

    version = _runtime_version(Sofa)
    payload["sofa_runtime_version"] = version

    plugin_results: dict[str, Any] = {}
    for plugin in REQUIRED_PLUGINS:
        try:
            result = SofaRuntime.importPlugin(plugin)
            plugin_results[plugin] = {
                "ok": bool(result) if result is not None else True,
                "return": repr(result),
            }
        except Exception as exc:
            plugin_results[plugin] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    payload["plugins"] = plugin_results

    root = Sofa.Core.Node("sofa26_component_audit")
    probe = root.addChild("probes")

    component_results: dict[str, Any] = {}
    component_results["FreeMotionAnimationLoop"] = _create_probe(
        probe, "FreeMotionAnimationLoop"
    )
    component_results["BlockGaussSeidelConstraintSolver"] = _create_probe(
        probe, "BlockGaussSeidelConstraintSolver"
    )
    component_results["CollisionPipeline"] = _create_probe(
        probe, "CollisionPipeline"
    )
    component_results["CompositeCollisionPipeline"] = _create_probe(
        probe, "CompositeCollisionPipeline"
    )
    component_results["SubCollisionPipeline"] = _create_probe(
        probe, "SubCollisionPipeline"
    )
    component_results["CCDTightInclusionIntersection"] = _create_probe(
        probe,
        "CCDTightInclusionIntersection",
        continuousCollisionType="FreeMotion",
        tolerance=1e-10,
        maxIterations=1000,
        alarmDistance=0.0,
        contactDistance=1e-5,
    )
    component_results["BruteForceBroadPhase"] = _create_probe(
        probe, "BruteForceBroadPhase"
    )
    component_results["BVHNarrowPhase"] = _create_probe(
        probe, "BVHNarrowPhase"
    )
    component_results["CollisionResponse"] = _create_probe(
        probe,
        "CollisionResponse",
        response="FrictionContactConstraint",
        responseParams="mu=0.0",
    )

    # BeamAdapter creation probes. These are only factory/API checks; the full
    # mapping is built in test_03.
    beam_parent = root.addChild("beam_probe")
    component_results["BeamAdapter.MechanicalObjectRigid3d"] = _create_probe(
        beam_parent,
        "MechanicalObject",
        template="Rigid3d",
        position=[[0, 0, 0, 0, 0, 0, 1]],
    )
    beam_child = beam_parent.addChild("mapped_probe")
    component_results["BeamAdapter.CollisionDOFs"] = _create_probe(
        beam_child,
        "MechanicalObject",
        template="Vec3d",
        position=[[0, 0, 0]],
    )
    component_results["MultiAdaptiveBeamMapping"] = _create_probe(
        beam_child,
        "MultiAdaptiveBeamMapping",
    )

    payload["components"] = component_results

    missing_plugins = [
        name for name, data in plugin_results.items() if not data.get("ok", False)
    ]
    missing_components = [
        name for name, data in component_results.items() if not data.get("created", False)
    ]

    version_ok = EXPECTED_SOFA_PREFIX in version
    payload["version_ok"] = version_ok
    payload["missing_plugins"] = missing_plugins
    payload["missing_components"] = missing_components

    # Important migration fact: current catheter body uses Triangle vessel
    # against mapped Line + Point primitives. TightInclusion v26.06 natively
    # covers Triangle/Point but not Triangle/Line.
    payload["mcr_relevant_pair_audit"] = {
        "vessel": "TriangleCollisionModel",
        "catheter_body": ["LineCollisionModel", "PointCollisionModel"],
        "triangle_point_ccd": True,
        "triangle_line_ccd": False,
        "body_interior_guarantee_from_official_matrix": False,
    }

    if not version_ok:
        decision = "INCONCLUSIVE"
        reason = "RUNTIME_IS_NOT_CONFIRMED_SOFA_26_06"
    elif missing_plugins:
        decision = "INCONCLUSIVE"
        reason = "REQUIRED_OFFICIAL_PLUGIN_MISSING"
    elif not component_results["CCDTightInclusionIntersection"]["created"]:
        decision = "INCONCLUSIVE"
        reason = "TIGHT_INCLUSION_COMPONENT_UNAVAILABLE"
    elif not component_results["CompositeCollisionPipeline"]["created"]:
        decision = "INCONCLUSIVE"
        reason = "COMPOSITE_COLLISION_PIPELINE_UNAVAILABLE"
    elif not plugin_results.get("BeamAdapter", {}).get("ok", False):
        decision = "INCONCLUSIVE"
        reason = "BEAMADAPTER_PLUGIN_UNAVAILABLE"
    else:
        decision = "PASS_WITH_PRIMITIVE_SUPPORT_WARNING"
        reason = "OFFICIAL_COMPONENTS_AVAILABLE_BUT_TRIANGLE_LINE_CCD_NOT_NATIVELY_SUPPORTED"

    payload["decision"] = decision
    payload["reason"] = reason
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
