"""Preflight for the external SDFUnilateralConstraint SOFA plugin.

This does not run an MCR episode. It verifies that the compiled plugin can be
loaded, its component can be created on a Vec3 MechanicalObject, and its Data
fields are visible through SofaPython3.
"""

from pathlib import Path
import sys

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import Sofa
import Sofa.Simulation

from mcr_sim.sdf_unilateral_constraint import load_sdf_unilateral_plugin


def main():
    plugin_path = load_sdf_unilateral_plugin()

    root = Sofa.Core.Node("root")
    node = root.addChild("points")
    node.addObject(
        "MechanicalObject",
        name="dofs",
        template="Vec3d",
        position=[[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]],
    )
    constraint = node.addObject(
        "SDFUnilateralConstraint",
        name="constraint",
        enabled=True,
        indices0=[0],
        indices1=[1],
        weights0=[0.5],
        weights1=[0.5],
        normals=[[1.0, 0.0, 0.0]],
        anchors=[[0.0004, 0.0, 0.0]],
        sourceClearances=[0.0001],
    )

    Sofa.Simulation.init(root)

    required = (
        "enabled",
        "indices0",
        "indices1",
        "weights0",
        "weights1",
        "normals",
        "anchors",
        "sourceClearances",
        "activeCount",
    )
    missing = [name for name in required if not hasattr(constraint, name)]
    if missing:
        raise RuntimeError(f"Plugin loaded but Data fields are missing: {missing}")

    print(
        "[SDF_UNILATERAL_PREFLIGHT] PASS",
        "plugin_path=", plugin_path,
        "class=", constraint.getClassName(),
        "enabled=", bool(constraint.enabled.value),
        flush=True,
    )


if __name__ == "__main__":
    main()
