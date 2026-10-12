"""ctypes bridge for candidate-only native Beam/SDF coarse gate.

Every buffer passed to C++ is a live, contiguous NumPy allocation; native
never retains pointers.  The immutable SDF array is cached by grid identity.
The first few actual SOFA states are checked against the V2 Python guard.
"""
import ctypes
import numpy as np
from fast_guard import coarse_beam_clearance_guard, SDF_SLOPE_ALLOWANCE, EXTRA_ERROR_CUSHION_M
from mcr_sim import beam_sdf_unilateral_fast as fast

_DOUBLE = ctypes.POINTER(ctypes.c_double)
_FLOAT = ctypes.POINTER(ctypes.c_float)
_INT = ctypes.POINTER(ctypes.c_int32)


class NativeCoarseBeamGuard:
    def __init__(self, library, spacing_m=1.0e-4, switch_margin_m=2.5e-4):
        self.spacing_m = float(spacing_m)
        self.switch_margin_m = float(switch_margin_m)
        function = library.mcr_fast_coarse_sdf_v1
        function.argtypes = [
            _DOUBLE, ctypes.c_int32, _INT, _DOUBLE, ctypes.c_int32,
            _FLOAT, _INT, _DOUBLE, _DOUBLE, _DOUBLE, _DOUBLE,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, _DOUBLE
        ]
        function.restype = ctypes.c_int
        self.function = function
        self._grid = None
        self._values = None
        self._dimensions = None
        self._origin = None
        self._spacing = None
        self._compared = 0
        self.calls = 0

    def _set_grid(self, grid):
        if grid is self._grid:
            return
        values = np.asarray(grid.values)
        if values.ndim != 3 or min(values.shape) < 2:
            raise RuntimeError("NATIVE_GUARD_INVALID_SDF_GRID_SHAPE")
        self._values = np.ascontiguousarray(values, dtype=np.float32)
        nz, ny, nx = values.shape
        self._dimensions = np.asarray([nx, ny, nz], dtype=np.int32)
        self._origin = np.ascontiguousarray(grid.origin, dtype=np.float64)
        self._spacing = np.ascontiguousarray(grid.spacing, dtype=np.float64)
        if self._origin.shape != (3,) or self._spacing.shape != (3,):
            raise RuntimeError("NATIVE_GUARD_INVALID_SDF_GRID_METADATA")
        self._grid = grid

    def __call__(self, adapter, q_free):
        env = adapter.env
        self._set_grid(env.sdf_grid)
        q = np.ascontiguousarray(q_free, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
            raise RuntimeError("NATIVE_GUARD_INVALID_Q_FREE")
        # Use direct NumPy arrays rather than creating a Python dictionary for
        # every deployed element in LiveBeamSDFAdapter._geometry().
        edge_list = fast._data_array(adapter.interpolation, "edgeList", dtype=np.int64).reshape(-1)
        lengths = fast._data_array(adapter.interpolation, "lengthList", dtype=np.float64).reshape(-1)
        edges = fast._data_array(adapter.topology, "edges", dtype=np.int64).reshape((-1, 2))
        if len(edge_list) == 0 or len(edge_list) != len(lengths):
            raise RuntimeError("NATIVE_GUARD_INVALID_ACTIVE_ELEMENTS")
        if np.any(edge_list < 0) or np.any(edge_list >= len(edges)):
            raise RuntimeError("NATIVE_GUARD_INVALID_EDGE_IDS")
        if np.any(lengths <= 0) or not np.isfinite(lengths).all():
            raise RuntimeError("NATIVE_GUARD_INVALID_ACTIVE_LENGTHS")
        inserted = adapter._inserted_length()
        if abs(float(np.sum(lengths)) - inserted) > 1e-6:
            raise RuntimeError("NATIVE_GUARD_INSERTED_LENGTH_MISMATCH")
        active_nodes = np.ascontiguousarray(edges[edge_list], dtype=np.int32)
        if np.any(active_nodes < 0) or np.any(active_nodes >= len(q)):
            raise RuntimeError("NATIVE_GUARD_INVALID_NODE_IDS")
        lengths = np.ascontiguousarray(lengths, dtype=np.float64)
        pose = np.ascontiguousarray(env.asset_T_env_sim, dtype=np.float64).reshape(7)
        offset = np.ascontiguousarray(env.asset_offset_sim, dtype=np.float64).reshape(3)
        result = np.empty(7, dtype=np.float64)
        ptr = lambda arr, ctype: arr.ctypes.data_as(ctypes.POINTER(ctype))
        rc = self.function(
            ptr(q, ctypes.c_double), int(len(q)),
            ptr(active_nodes, ctypes.c_int32), ptr(lengths, ctypes.c_double), int(len(lengths)),
            ptr(self._values, ctypes.c_float), ptr(self._dimensions, ctypes.c_int32),
            ptr(self._origin, ctypes.c_double), ptr(self._spacing, ctypes.c_double),
            ptr(pose, ctypes.c_double), ptr(offset, ctypes.c_double),
            float(env.asset_source_to_sim_scale), float(env.catheter_radius),
            self.spacing_m, self.switch_margin_m,
            float(SDF_SLOPE_ALLOWANCE), float(EXTRA_ERROR_CUSHION_M),
            ptr(result, ctypes.c_double),
        )
        if rc != 0:
            raise RuntimeError("NATIVE_GUARD_ERROR_CODE_{}".format(rc))
        invalid = bool(result[4])
        safe = bool(result[6])
        # An early out can return a partial (but never falsely safe) minimum.
        minimum = float(result[0]) if np.isfinite(result[0]) else None
        lower = float(result[1]) if not invalid and np.isfinite(result[1]) else None
        probe = {
            "safe": safe, "invalid_sdf": invalid,
            "minimum_clearance_m": minimum, "lower_bound_m": lower,
            "worst_index": int(result[2]),
            "sample_count": int(result[3]),
            "max_curve_gap_bound_m": float(result[5]),
            "interpolation_allowance_m": (
                float(SDF_SLOPE_ALLOWANCE)*float(result[5])/2.0+
                float(EXTRA_ERROR_CUSHION_M)
            ),
            "coarse_spacing_m": self.spacing_m,
            "switch_margin_m": self.switch_margin_m,
            "backend": "native_cpp_v1",
        }
        self.calls += 1
        # Exercise real deployed geometry + transformed vessel SDF at startup.
        # Never silently accept a native SAFE verdict that Python disproves.
        if self._compared < 3:
            reference = coarse_beam_clearance_guard(
                adapter, q, spacing_m=self.spacing_m,
                switch_margin_m=self.switch_margin_m,
            )
            if safe != bool(reference["safe"]):
                raise RuntimeError(
                    "NATIVE_GUARD_PYTHON_SAFE_MISMATCH native={} python={}".format(
                        safe, reference["safe"]
                    )
                )
            if safe:
                if (
                    abs(float(result[0])-float(reference["minimum_clearance_m"])) > 2e-7
                    or abs(float(result[1])-float(reference["lower_bound_m"])) > 2e-7
                ):
                    raise RuntimeError("NATIVE_GUARD_PYTHON_CLEARANCE_MISMATCH")
            self._compared += 1
        return probe
