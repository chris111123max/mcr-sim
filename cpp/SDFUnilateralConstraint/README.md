# SDFUnilateralConstraint (SOFA 21.12 prototype)

This plugin adds one component, `SDFUnilateralConstraint`, to a Vec3 mapped
collision MechanicalObject.  Each active row enforces

```
g(x) = dot(sample(x) - anchor, inward_normal) >= 0
lambda >= 0
lambda * g(x) = 0
```

The component does not project positions and does not apply a penalty force.
It contributes rows to the existing SOFA Lagrange constraint solve.

## Build

Activate the same environment used to run MCR/SOFA, then point CMake at the
SOFA installation:

```bash
cd cpp/SDFUnilateralConstraint

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${SOFA_ROOT:-$CONDA_PREFIX}"

cmake --build build -j2
```

The Python bridge searches these locations automatically:

- `cpp/SDFUnilateralConstraint/build`
- `cpp/SDFUnilateralConstraint/build/lib`
- `cpp/SDFUnilateralConstraint/build/bin`

You can override the search path with:

```bash
export MCR_SDF_UNILATERAL_PLUGIN_DIR=/absolute/path/to/plugin/directory
```

Run the plugin preflight before the trajectory diagnostic:

```bash
python testing/py/preflight_sdf_unilateral_plugin.py
```
