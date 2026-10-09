#!/usr/bin/env python3
"""TEST ONLY: real Beam/SDF directional relinearization on immutable snapshots.

This is a read-only geometric counterfactual, NOT a SOFA constraint re-solve.
An offline PASS cannot establish that a native same-substep iteration is possible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

# python/testing/... => the repository's python source directory
PYTHON_ROOT = Path(__file__).resolve().parents[3]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from mcr_sim.vessel_assets import (  # noqa: E402
    load_signed_distance_grid,
    sim_points_to_asset_source,
)

DENSE_SPACING_M = 1e-5
MARGIN_M = 1e-4


def _validate_q(q, name):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 7 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError(f"{name}: expected finite (N,7) Rigid3")
    lengths = np.linalg.norm(q[:, 3:7], axis=1)
    if np.max(np.abs(lengths - 1.0)) > 1e-4:
        raise ValueError(f"{name}: quaternions are not unit length")
    return q


def _sample_element(q, spec, spacing=DENSE_SPACING_M):
    pair = spec["nodes"]
    n0, n1 = int(pair[0]), int(pair[1])
    if n0 == n1 or min(n0, n1) < 0 or max(n0, n1) >= len(q):
        raise ValueError("Invalid active element Beam node indices")
    length = float(spec["rest_length_m"])
    if length <= 0 or not np.isfinite(length):
        raise ValueError("Invalid active Beam length")
    p0, p3 = q[n0, :3], q[n1, :3]
    r0, r1 = Rotation.from_quat(q[n0, 3:7]), Rotation.from_quat(q[n1, 3:7])
    p1 = p0 + r0.apply([length / 3.0, 0.0, 0.0])
    p2 = p3 + r1.apply([-length / 3.0, 0.0, 0.0])
    linear = (
        np.dot(p1 - p0, p2 - p1) < 0.0
        and np.linalg.norm(p3 - p0) < 0.4 * length
    )
    t = np.linspace(0.0, 1.0, max(2, int(np.ceil(length / spacing)) + 1))
    u = 1.0 - t
    if linear:
        return u[:, None] * p0 + t[:, None] * p3
    return (
        u[:, None] ** 3 * p0
        + (3.0 * u**2 * t)[:, None] * p1
        + (3.0 * u * t**2)[:, None] * p2
        + t[:, None] ** 3 * p3
    )


def sample_beam(q, active_elements):
    """Replicates production BeamAdapter sample count/order, without SOFA."""
    chunks = []
    if not isinstance(active_elements, list) or not active_elements:
        raise ValueError("Need recorded active_elements; never infer topology")
    for i, spec in enumerate(active_elements):
        points = _sample_element(q, spec)
        chunks.append(points if i == 0 else points[1:])
    return np.concatenate(chunks, axis=0)


class GeometryOracle:
    def __init__(self, bundle):
        required = (
            "active_elements",
            "sdf_vti",
            "asset_T_env_sim",
            "asset_offset_sim",
            "asset_source_to_sim_scale",
            "catheter_radius_m",
        )
        missing = [k for k in required if k not in bundle]
        if missing:
            raise ValueError("MISSING_CAPTURED_GEOMETRY: " + ", ".join(missing))
        self.specs = bundle["active_elements"]
        self.source = Path(bundle["sdf_vti"]).expanduser().resolve()
        if not self.source.is_file():
            raise FileNotFoundError(f"Recorded SDF is unavailable: {self.source}")
        self.grid = load_signed_distance_grid(self.source)
        self.transform = np.asarray(bundle["asset_T_env_sim"], dtype=float).reshape(7)
        self.offset = np.asarray(bundle["asset_offset_sim"], dtype=float).reshape(3)
        self.scale = float(bundle["asset_source_to_sim_scale"])
        self.radius = float(bundle["catheter_radius_m"])
        if self.scale <= 0 or self.radius <= 0:
            raise ValueError("Non-positive SDF scale or catheter radius")
        self.count = 0

    def profile(self, q):
        points = sample_beam(q, self.specs)
        source = sim_points_to_asset_source(points, self.transform, self.offset, self.scale)
        clearance = -np.asarray(self.grid.sample(source), dtype=float) * self.scale - self.radius
        if clearance.shape != (len(points),) or not np.isfinite(clearance).all():
            raise ValueError("SDF_PROFILE_INVALID_OR_OUTSIDE_GRID")
        self.count += 1
        return clearance


def interpolate_correction(q_free, q_committed, alpha):
    """World-frame geodesic displacement along the actual recorded native correction."""
    alpha = float(alpha)
    if not np.isfinite(alpha):
        raise ValueError("Nonfinite correction scale")
    q = q_free.copy()
    q[:, :3] = q_free[:, :3] + alpha * (q_committed[:, :3] - q_free[:, :3])
    r0 = Rotation.from_quat(q_free[:, 3:7])
    r1 = Rotation.from_quat(q_committed[:, 3:7])
    world_delta = (r1 * r0.inv()).as_rotvec()
    q[:, 3:7] = (Rotation.from_rotvec(alpha * world_delta) * r0).as_quat()
    return q


def _read_capture(path):
    with np.load(path, allow_pickle=False) as source:
        arrays = {k: np.asarray(source[k]).copy() for k in source.files}
    if "q_free" not in arrays or "q_committed" not in arrays:
        raise ValueError("Snapshot must have canonical q_free and q_committed")
    qf = _validate_q(arrays["q_free"], "q_free")
    qc = _validate_q(arrays["q_committed"], "q_committed")
    if qf.shape != qc.shape:
        raise ValueError("q_free / q_committed shape mismatch")
    for key in ("q_free_dense_clearance", "q_committed_dense_clearance"):
        if key not in arrays:
            raise ValueError("Missing independent 10um capture: " + key)
    return arrays, qf, qc


def _max_difference(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1), np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Capture/recomputed dense sample counts differ")
    return float(np.max(np.abs(a - b)))


def _sha256_array(q):
    return hashlib.sha256(np.asarray(q, dtype="<f8").tobytes()).hexdigest()


def probe(arrays, qf, qc, oracle, *,
          margin_m=MARGIN_M, match_tolerance_m=1e-8,
          fd_alpha=0.0025, max_newton_rounds=3,
          max_alpha=1.5, max_alpha_increment=0.2):
    """1D directional Newton diagnostic; NEVER a native physics solve."""
    if not (0 < fd_alpha < 0.1 and 1 < max_alpha <= 2
            and max_newton_rounds in (1, 2, 3)
            and 0 < max_alpha_increment <= 0.5):
        raise ValueError("Invalid bounded diagnostic parameters")

    q_at_one = interpolate_correction(qf, qc, 1.0)
    state_position_error = float(np.max(np.abs(q_at_one[:, :3] - qc[:, :3])))
    quaternion_mismatch_rad = float(np.max((
        Rotation.from_quat(q_at_one[:, 3:7]) *
        Rotation.from_quat(qc[:, 3:7]).inv()
    ).magnitude()))
    if state_position_error > 1e-10 or quaternion_mismatch_rad > 1e-9:
        raise ValueError("CORRECTION_INTERPOLATION_BASELINE_MISMATCH")

    original_free = oracle.profile(qf)
    original_committed = oracle.profile(qc)
    free_difference = _max_difference(original_free, arrays["q_free_dense_clearance"])
    committed_difference = _max_difference(
        original_committed, arrays["q_committed_dense_clearance"]
    )
    if max(free_difference, committed_difference) > match_tolerance_m:
        raise ValueError(
            f"CAPTURE_GEOMETRY_MISMATCH:free_m={free_difference:.12g}"
            f",committed_m={committed_difference:.12g},"
            f"allowed_m={match_tolerance_m:.12g}"
        )

    cache = {0.0: original_free, 1.0: original_committed}

    def measure(alpha):
        alpha = float(alpha)
        if alpha not in cache:
            cache[alpha] = oracle.profile(interpolate_correction(qf, qc, alpha))
        x = cache[alpha]
        return float(np.min(x)), int(np.argmin(x))

    baseline, baseline_worst_index = measure(1.0)
    alpha = 1.0
    rows = [{
        "iteration": 0,
        "alpha": 1.0,
        "dense_min_clearance_mm": baseline * 1000,
        "dense_worst_index": baseline_worst_index,
        "remaining_margin_mm": (margin_m - baseline) * 1000,
    }]
    state = "MARGIN_MET_AT_BASELINE" if baseline >= margin_m else "MAX_NEWTON_ROUNDS"
    for iteration in range(1, max_newton_rounds + 1):
        if measure(alpha)[0] >= margin_m:
            state = "DIRECTIONAL_MARGIN_MET"
            break
        # True nonlinear SDF directional derivative; no native Jacobian or solve.
        left, _ = measure(alpha - fd_alpha)
        right, _ = measure(alpha + fd_alpha)
        derivative = (right - left) / (2.0 * fd_alpha)
        if not np.isfinite(derivative) or derivative <= 1e-10:
            state = "NO_POSITIVE_CLEARANCE_SLOPE_ALONG_NATIVE_CORRECTION"
            rows.append({"iteration": iteration, "alpha": alpha,
                         "derivative_m_per_alpha": derivative, "status": state})
            break
        clearance, _ = measure(alpha)
        increment = float(np.clip((margin_m - clearance) / derivative,
                                  0.0, max_alpha_increment))
        next_alpha = min(max_alpha, alpha + increment)
        if next_alpha - alpha < 1e-8:
            state = "BOUNDED_DIRECTIONAL_STAGNATION"
            break
        alpha = next_alpha
        value, worst_index = measure(alpha)
        rows.append({
            "iteration": iteration,
            "alpha": alpha,
            "derivative_m_per_alpha": derivative,
            "dense_min_clearance_mm": value * 1000,
            "dense_worst_index": worst_index,
            "remaining_margin_mm": (margin_m - value) * 1000,
            "note": "Hypothetical numpy state ONLY; not SOFA committed state",
        })
        if value >= margin_m:
            state = "DIRECTIONAL_MARGIN_MET"
            break
        if alpha >= max_alpha - 1e-10:
            state = "BOUNDED_DIRECTIONAL_ALPHA_LIMIT"
            break

    return {
        "status": "MEASURED_READ_ONLY_GEOMETRY",
        "scope": "one-dimensional correction-direction Newton; NOT native relinearization",
        "frame_q_free_sha256": _sha256_array(qf),
        "frame_q_committed_sha256": _sha256_array(qc),
        "baseline_capture_gate": "PASS",
        "max_free_profile_difference_m": free_difference,
        "max_committed_profile_difference_m": committed_difference,
        "interpolation_max_position_difference_m": state_position_error,
        "interpolation_max_rotation_difference_rad": quaternion_mismatch_rad,
        "baseline_clearance_mm": baseline * 1000,
        "baseline_worst_index": baseline_worst_index,
        "requested_margin_mm": margin_m * 1000,
        "directional_outcome": state,
        "iterations": rows,
        "geometry_sdf_queries": oracle.count,
        "native_physics_solve_count": 0,
        "native_relinearization_verified": False,
        "historical_deep_penetration_cause_identified": False,
    }


def _selftest():
    qf = np.array([
        [0.0, 0.0, 0.0, 0, 0, 0, 1],
        [0.01, 0.0, 0.0, 0, 0, 0, 1],
    ], dtype=float)
    qc = qf.copy()
    qc[0, 2] = 0.002
    qc[0, 3:7] = Rotation.from_euler("z", 30, degrees=True).as_quat()
    for alpha in (0, 0.25, 0.5, 0.75, 1):
        q = interpolate_correction(qf, qc, alpha)
        assert np.max(np.abs(q[:, :3] - (qf[:, :3] + alpha * (qc[:, :3] - qf[:, :3])))) < 1e-14
        assert np.isfinite(q).all()
    assert np.allclose(interpolate_correction(qf, qc, 0), qf, atol=1e-13)
    assert np.allclose(interpolate_correction(qf, qc, 1), qc, atol=1e-13)
    spec = [{"nodes": [0, 1], "rest_length_m": 0.01}]
    p = sample_beam(qf, spec)
    assert len(p) == 1001, len(p)  # 10 mm / 10 um + 1
    assert np.allclose(p[0], qf[0, :3])
    assert np.allclose(p[-1], qf[1, :3])
    return {"selftest": "PASS", "synthetic_only": True, "sample_count": len(p)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, help="Normalized saved same-frame NPZ")
    ap.add_argument("--geometry", type=Path, help="Recorded JSON: exact active elements + SDF transform/path")
    ap.add_argument("--output", type=Path, help="New report JSON (never overwrites snapshot)")
    ap.add_argument("--max-rounds", type=int, default=3, choices=(1, 2, 3))
    ap.add_argument("--margin-mm", type=float, default=0.100)
    ap.add_argument("--capture-tolerance-m", type=float, default=1e-8)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        result = _selftest()
    else:
        if not args.snapshot or not args.geometry:
            ap.error("Both --snapshot and --geometry are required")
        if args.margin_mm <= 0 or not np.isfinite(args.margin_mm):
            ap.error("--margin-mm must be positive and finite")
        bundle = json.loads(args.geometry.read_text(encoding="utf-8"))
        if not isinstance(bundle, dict):
            raise ValueError("Geometry JSON must contain a dict")
        arrays, qf, qc = _read_capture(args.snapshot)
        oracle = GeometryOracle(bundle)
        result = probe(
            arrays, qf, qc, oracle,
            margin_m=args.margin_mm * 1e-3,
            match_tolerance_m=args.capture_tolerance_m,
            max_newton_rounds=args.max_rounds,
        )
        result["snapshot"] = str(args.snapshot.resolve())
        result["geometry"] = str(args.geometry.resolve())
        result["vessel"] = bundle.get("vessel", "UNKNOWN")
        result["target"] = bundle.get("target", "UNKNOWN")
        result["original_failure_frame_reproduced"] = False
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    print(text)
    if args.output:
        dest = args.output.expanduser().resolve()
        if args.snapshot and dest == args.snapshot.resolve():
            raise ValueError("Refuse to overwrite source snapshot")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"OFFLINE_PROBE_INCONCLUSIVE:{type(error).__name__}:{error}", file=sys.stderr)
        sys.exit(2)
