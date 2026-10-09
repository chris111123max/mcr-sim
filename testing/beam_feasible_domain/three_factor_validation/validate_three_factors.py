#!/usr/bin/env python3
"""Read-only Beam unilateral three-factor diagnostic on saved SAME-FRAME snapshots.

No SOFA animation, policy inference, training, state writes, or production edits.
Input numbers are metres; outputs report millimetres. The frame may be SAFE.
A PASS here never implies the historical -0.766410143 mm crash was reproduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

MARGIN_M = 1.0e-4
ROW_TOL_M = 1.0e-6
NONLINEAR_TOL_M = 1.0e-6

ALIASES = {
    "q_prev": ("q_prev", "beam_q_prev", "prev_beam_state"),
    "q_free": ("q_free", "beam_q_free", "free_position"),
    "q_committed": ("q_committed", "beam_q_committed", "committed_position"),
    "row_offsets": ("row_offsets", "rowOffsets"),
    "dof_indices": ("dof_indices", "dofIndices"),
    "linear_jacobian": ("linear_jacobian", "linearJacobian"),
    "angular_jacobian": ("angular_jacobian", "angularJacobian"),
    "free_violations": ("free_violations", "freeViolations"),
    "selected_dense_indices": ("selected_dense_indices", "selected_indices"),
    "q_free_dense_clearance": (
        "q_free_dense_clearance", "free_dense_clearance",
        "q_free_dense_profile", "free_clearance_profile",
    ),
    "q_committed_dense_clearance": (
        "q_committed_dense_clearance", "committed_dense_clearance",
        "q_committed_dense_profile", "committed_clearance_profile",
    ),
    "angular_world_oracle": (
        "angular_world_oracle", "independent_angular_fd_world",
        "oracle_angular_jacobian_world",
    ),
}
REQUIRED = (
    "q_free", "q_committed", "row_offsets", "dof_indices",
    "linear_jacobian", "angular_jacobian", "free_violations",
    "selected_dense_indices", "q_free_dense_clearance",
    "q_committed_dense_clearance",
)


def _arr(data, canonical, *, optional=False):
    for key in ALIASES[canonical]:
        if key in data:
            return np.asarray(data[key])
    if optional:
        return None
    raise ValueError(
        "MISSING_SNAPSHOT_FIELD:" + canonical + " aliases="
        + ",".join(ALIASES[canonical]) + "; available=" + ",".join(sorted(data))
    )


def _shape_numeric(a, name):
    if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise ValueError("NONFINITE_OR_NONNUMERIC:" + name)


def load_snapshot(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as src:
        data = {key: np.array(src[key]) for key in src.files}
    out = {k: _arr(data, k) for k in REQUIRED}
    out["q_prev"] = _arr(data, "q_prev", optional=True)
    out["angular_world_oracle"] = _arr(data, "angular_world_oracle", optional=True)
    out["path"] = str(path)
    out["metadata"] = {}
    for sibling in (path.with_suffix(".json"), path.parent / "failure_metadata.json"):
        if sibling.is_file():
            try:
                obj = json.loads(sibling.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    out["metadata"] = obj
                    break
            except (OSError, ValueError):
                pass
    q0 = np.asarray(out["q_free"], dtype=np.float64)
    q1 = np.asarray(out["q_committed"], dtype=np.float64)
    if q0.ndim != 2 or q0.shape[1] != 7 or q1.shape != q0:
        raise ValueError("INVALID_RIGID3_BEAM_SHAPE: expected matching (N,7)")
    _shape_numeric(q0, "q_free")
    _shape_numeric(q1, "q_committed")
    for name in ("q_prev",):
        if out[name] is not None:
            out[name] = np.asarray(out[name], dtype=np.float64)
            if out[name].shape != q0.shape:
                raise ValueError("INVALID_SHAPE:" + name)
            _shape_numeric(out[name], name)
    # xyzw is scipy/SOFA quaternion convention; reject invalid rotations.
    for key, q in (("q_free", q0), ("q_committed", q1)):
        if np.any(np.linalg.norm(q[:, 3:7], axis=1) < 1e-9):
            raise ValueError("ZERO_QUATERNION:" + key)

    offset = np.asarray(out["row_offsets"], dtype=np.int64).reshape(-1)
    ids = np.asarray(out["dof_indices"], dtype=np.int64).reshape(-1)
    jl = np.asarray(out["linear_jacobian"], dtype=np.float64).reshape(-1, 3)
    ja = np.asarray(out["angular_jacobian"], dtype=np.float64).reshape(-1, 3)
    g = np.asarray(out["free_violations"], dtype=np.float64).reshape(-1)
    selected = np.asarray(out["selected_dense_indices"], dtype=np.int64).reshape(-1)
    fc = np.asarray(out["q_free_dense_clearance"], dtype=np.float64).reshape(-1)
    cc = np.asarray(out["q_committed_dense_clearance"], dtype=np.float64).reshape(-1)
    for name, array in (
        ("linear_jacobian", jl), ("angular_jacobian", ja),
        ("free_violations", g), ("free_dense_clearance", fc),
        ("committed_dense_clearance", cc),
    ):
        _shape_numeric(array, name)
    if len(g) == 0 or len(selected) != len(g) or len(offset) != len(g) + 1:
        raise ValueError("ROW_COUNT_OR_SELECTED_COUNT_MISMATCH")
    if offset[0] != 0 or np.any(np.diff(offset) <= 0) or offset[-1] != len(ids):
        raise ValueError("INVALID_CSR_OFFSETS")
    if len(ids) != len(jl) or len(ids) != len(ja):
        raise ValueError("INVALID_CSR_JACOBIAN_LENGTH")
    if np.any(ids < 0) or np.any(ids >= len(q0)):
        raise ValueError("INVALID_CSR_BEAM_DOF")
    if len(fc) != len(cc):
        raise ValueError("DENSE_SAMPLE_COUNT_MISMATCH")
    if np.any(selected < 0) or np.any(selected >= len(fc)):
        raise ValueError("INVALID_SELECTED_DENSE_INDEX")
    if len(set(selected.tolist())) != len(selected):
        raise ValueError("DUPLICATE_SELECTED_DENSE_INDEX")
    out.update(
        q_free=q0, q_committed=q1, row_offsets=offset, dof_indices=ids,
        linear_jacobian=jl, angular_jacobian=ja, free_violations=g,
        selected_dense_indices=selected, q_free_dense_clearance=fc,
        q_committed_dense_clearance=cc,
    )
    oracle = out["angular_world_oracle"]
    if oracle is not None:
        oracle = np.asarray(oracle, dtype=np.float64).reshape(-1, 3)
        if oracle.shape != ja.shape:
            raise ValueError("ANGULAR_ORACLE_JACOBIAN_SHAPE_MISMATCH")
        _shape_numeric(oracle, "angular_world_oracle")
        out["angular_world_oracle"] = oracle
    return out


def _displacements(snapshot):
    q0, q1 = snapshot["q_free"], snapshot["q_committed"]
    r0 = Rotation.from_quat(q0[:, 3:7])
    r1 = Rotation.from_quat(q1[:, 3:7])
    delta_linear = q1[:, :3] - q0[:, :3]
    # R_new = dR_world * R_free, so dR_world = R_new * inv(R_free).
    delta_world = (r1 * r0.inv()).as_rotvec()
    return delta_linear, delta_world, r0


def linearized_g(snapshot, *, convert_angular):
    dt, dr_world, r0 = _displacements(snapshot)
    ids = snapshot["dof_indices"]
    jl = snapshot["linear_jacobian"]
    ja = snapshot["angular_jacobian"]
    # Production FD uses R_free * Exp(delta_local); the native SOFA angular
    # increments are expressed in world coordinates in this hypothesis.
    jw = r0[ids].apply(ja) if convert_angular else ja
    blocks = np.einsum("ij,ij->i", jl, dt[ids])
    blocks += np.einsum("ij,ij->i", jw, dr_world[ids])
    out = snapshot["free_violations"].copy()
    offsets = snapshot["row_offsets"]
    for row in range(len(out)):
        out[row] += float(np.sum(blocks[offsets[row]:offsets[row + 1]]))
    return out, jw


def jacobian_case(snapshot):
    _, corrected = linearized_g(snapshot, convert_angular=True)
    raw = snapshot["angular_jacobian"]
    difference = np.linalg.norm(corrected - raw, axis=1)
    result = {
        "case": "rotational_jacobian_frame",
        "raw_vs_converted_max_m_per_rad": float(np.max(difference)),
        "raw_vs_converted_mean_m_per_rad": float(np.mean(difference)),
        "requires_independent_oracle": True,
    }
    oracle = snapshot["angular_world_oracle"]
    if oracle is None:
        result.update(
            status="INCONCLUSIVE",
            reason="Independent world-frame FD oracle missing from this snapshot; "
                   "a frame conversion difference alone does not prove which "
                   "coordinate convention native SOFA expects.",
        )
    else:
        raw_error = np.linalg.norm(raw - oracle, axis=1)
        fixed_error = np.linalg.norm(corrected - oracle, axis=1)
        result.update(
            raw_max_error_m_per_rad=float(np.max(raw_error)),
            converted_max_error_m_per_rad=float(np.max(fixed_error)),
            status="FRAME_MISMATCH_CONFIRMED"
            if np.max(fixed_error) < np.max(raw_error) / 100.0
            and np.max(fixed_error) < 1e-7
            else "FRAME_MISMATCH_NOT_CONFIRMED",
        )
    return result


def solver_case(snapshot, tol_m):
    original, _ = linearized_g(snapshot, convert_angular=False)
    corrected, _ = linearized_g(snapshot, convert_angular=True)
    return {
        "case": "linear_row_residual",
        "status": "LINEAR_RESIDUAL_EXCEEDS_TOLERANCE"
        if float(np.min(original)) < -tol_m else "LINEAR_ROWS_WITHIN_TOLERANCE",
        "row_count": int(len(original)),
        "tolerance_mm": float(tol_m * 1000),
        "min_original_jacobian_gap_mm": float(np.min(original) * 1000),
        "violated_original_rows": int(np.count_nonzero(original < -tol_m)),
        "min_converted_jacobian_gap_mm": float(np.min(corrected) * 1000),
        "violated_converted_rows": int(np.count_nonzero(corrected < -tol_m)),
        "note": "Reconstructed first-order row gaps, not native solver "
                "lambda/residual data; quaternion increments use finite rotation log.",
    }


def nonlinear_case(snapshot, tol_m):
    selected = snapshot["selected_dense_indices"]
    actual = snapshot["q_committed_dense_clearance"][selected] - MARGIN_M
    pred_original, _ = linearized_g(snapshot, convert_angular=False)
    pred_converted, _ = linearized_g(snapshot, convert_angular=True)
    mismatch_raw = actual - pred_original
    mismatch_corrected = actual - pred_converted
    cc = snapshot["q_committed_dense_clearance"]
    fc = snapshot["q_free_dense_clearance"]
    worst_final = int(np.argmin(cc))
    selected_set = set(selected.tolist())
    status = (
        "REAL_SDF_MARGIN_MISSED"
        if float(np.min(cc)) < MARGIN_M - tol_m
        else "REAL_SDF_MARGIN_MET"
    )
    return {
        "case": "nonlinear_sdf_vs_linear_prediction",
        "status": status,
        "requested_margin_mm": MARGIN_M * 1000,
        "tolerance_mm": tol_m * 1000,
        "min_free_clearance_mm": float(np.min(fc) * 1000),
        "min_committed_clearance_mm": float(np.min(cc) * 1000),
        "min_selected_actual_gap_mm": float(np.min(actual) * 1000),
        "max_abs_nonlinear_mismatch_original_mm":
            float(np.max(np.abs(mismatch_raw)) * 1000),
        "max_abs_nonlinear_mismatch_converted_mm":
            float(np.max(np.abs(mismatch_corrected)) * 1000),
        "free_worst_index": int(np.argmin(fc)),
        "committed_worst_index": worst_final,
        "committed_worst_selected": worst_final in selected_set,
        "min_clearance_minus_requested_margin_mm":
            float((np.min(cc) - MARGIN_M) * 1000),
        "note": "Nonlinear geometry is evaluated from recorded independent dense "
                "SDF, not a new SOFA integration. Same-index correspondence and "
                "recorded interpolation topology must be checked by capture code.",
    }


def _hash_state(snapshot):
    h = hashlib.sha256()
    for key in ("q_prev", "q_free"):
        if snapshot[key] is None:
            raise ValueError("NATIVE_AB_REQUIRES:" + key)
        h.update(np.asarray(snapshot[key], dtype="<f8").tobytes())
    return h.hexdigest()


def compare_native(base, variant, kind, *, state_tolerance_m=1e-10):
    if base["q_prev"] is None or variant["q_prev"] is None:
        raise ValueError("NATIVE_AB_REQUIRES:q_prev")
    if base["q_prev"].shape != variant["q_prev"].shape:
        raise ValueError("AB_PREV_STATE_SHAPE_MISMATCH")
    if base["q_free"].shape != variant["q_free"].shape:
        raise ValueError("AB_FREE_STATE_SHAPE_MISMATCH")
    # translations in metres; quaternions dimensionless, checked separately.
    worst_pos = max(
        float(np.max(np.abs(base[k][:, :3] - variant[k][:, :3])))
        for k in ("q_prev", "q_free")
    )
    worst_quat = max(
        float(np.max(np.abs(base[k][:, 3:] - variant[k][:, 3:])))
        for k in ("q_prev", "q_free")
    )
    if worst_pos > state_tolerance_m or worst_quat > 1e-10:
        raise ValueError(
            "AB_FREE_OR_PREV_STATE_MISMATCH: translation_m="
            + repr(worst_pos) + " quaternion=" + repr(worst_quat)
        )
    for key in ("row_offsets", "dof_indices", "selected_dense_indices",
                "linear_jacobian", "free_violations",
                "q_free_dense_clearance"):
        if not np.array_equal(base[key], variant[key]):
            raise ValueError("AB_CONFOUNDED_CHANGED:" + key)
    for meta_key in (
        "vessel", "target", "seed", "episode", "rl_step", "substep",
        "geometry_sha256", "sdf_sha256", "action_prefix_sha256",
    ):
        base_meta = base["metadata"]
        variant_meta = variant["metadata"]
        if meta_key in base_meta and meta_key in variant_meta:
            if base_meta[meta_key] != variant_meta[meta_key]:
                raise ValueError("AB_METADATA_MISMATCH:" + meta_key)
    if kind == "solver_tight" and not np.array_equal(
        base["angular_jacobian"], variant["angular_jacobian"]
    ):
        raise ValueError("AB_SOLVER_TIGHT_CHANGED_ANGULAR_JACOBIAN")
    if kind == "jacobian_fixed":
        expected = linearized_g(base, convert_angular=True)[1]
        if not np.allclose(variant["angular_jacobian"], expected,
                           atol=1e-9, rtol=1e-9):
            raise ValueError("AB_JACOBIAN_NOT_ONLY_EXPECTED_FRAME_ROTATION")
    return {
        "variant": kind,
        "same_q_prev_and_q_free": True,
        "max_translation_delta_m": worst_pos,
        "max_quaternion_component_delta": worst_quat,
        "base_state_sha256": _hash_state(base),
        "variant_state_sha256": _hash_state(variant),
        "baseline_committed_clearance_mm":
            float(np.min(base["q_committed_dense_clearance"]) * 1000),
        "variant_committed_clearance_mm":
            float(np.min(variant["q_committed_dense_clearance"]) * 1000),
        "delta_committed_clearance_mm": float(
            (np.min(variant["q_committed_dense_clearance"])
             - np.min(base["q_committed_dense_clearance"])) * 1000
        ),
        "variant_reconstructed_row_residual":
            solver_case(variant, ROW_TOL_M),
    }


def _round_trip_self_test():
    # Numeric fixtures prove the implementation runs, not physics validity.
    q0 = np.array([
        [0, 0, 0, 0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
        [1, 0, 0, 0, 0, 0, 1],
    ], dtype=float)
    q1 = q0.copy()
    q1[0, 0] += 2.0e-5
    fixture = {
        "q_free": q0, "q_committed": q1, "q_prev": q0.copy(),
        "row_offsets": np.array([0, 1]),
        "dof_indices": np.array([0]),
        "linear_jacobian": np.array([[1.0, 0.0, 0.0]]),
        "angular_jacobian": np.array([[1.0, 0.0, 0.0]]),
        "free_violations": np.array([-1e-5]),
        "selected_dense_indices": np.array([0]),
        "q_free_dense_clearance": np.array([9e-5, 2e-4]),
        "q_committed_dense_clearance": np.array([1.1e-4, 2e-4]),
        "angular_world_oracle": np.array([[0.0, 1.0, 0.0]]),
        "metadata": {}, "path": "synthetic",
    }
    j = jacobian_case(fixture)
    s = solver_case(fixture, ROW_TOL_M)
    n = nonlinear_case(fixture, NONLINEAR_TOL_M)
    assert j["status"] == "FRAME_MISMATCH_CONFIRMED", j
    assert abs(s["min_original_jacobian_gap_mm"] - 0.01) < 1e-9, s
    assert n["status"] == "REAL_SDF_MARGIN_MET", n
    assert compare_native(fixture, fixture, "solver_tight")["same_q_prev_and_q_free"]
    return {"selftest": "PASS", "synthetic_only": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, help="Baseline normalized .npz")
    ap.add_argument("--jacobian-fixed", type=Path,
                    help="Same-frame native result after ONLY rotation frame fix")
    ap.add_argument("--solver-tight", type=Path,
                    help="Same-frame native result after ONLY tighter solver")
    ap.add_argument("--case", choices=("jacobian", "solver", "nonlinear", "all"),
                    default="all")
    ap.add_argument("--row-tolerance-mm", type=float, default=0.001)
    ap.add_argument("--nonlinear-tolerance-mm", type=float, default=0.001)
    ap.add_argument("--output", type=Path, help="Write report JSON; opt-in only")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        result = _round_trip_self_test()
    else:
        if args.snapshot is None:
            ap.error("--snapshot is required unless --self-test is used")
        if args.row_tolerance_mm < 0 or args.nonlinear_tolerance_mm < 0:
            ap.error("tolerances must be non-negative")
        snap = load_snapshot(args.snapshot)
        result = {
            "snapshot": str(args.snapshot.resolve()),
            "historical_failure_reproduced": False,
            "capture_metadata": snap["metadata"],
            "cases": {},
            "native_ab": {},
            "interpretation_limit": "This checks the provided captured frame ONLY. "
                                    "It does not identify the original training crash.",
        }
        if args.case in ("jacobian", "all"):
            result["cases"]["jacobian"] = jacobian_case(snap)
        if args.case in ("solver", "all"):
            result["cases"]["solver"] = solver_case(
                snap, args.row_tolerance_mm * 1e-3
            )
        if args.case in ("nonlinear", "all"):
            result["cases"]["nonlinear"] = nonlinear_case(
                snap, args.nonlinear_tolerance_mm * 1e-3
            )
        if args.jacobian_fixed:
            fixed = load_snapshot(args.jacobian_fixed)
            result["native_ab"]["jacobian_fixed"] = compare_native(
                snap, fixed, "jacobian_fixed"
            )
        if args.solver_tight:
            tight = load_snapshot(args.solver_tight)
            result["native_ab"]["solver_tight"] = compare_native(
                snap, tight, "solver_tight"
            )
    txt = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(txt)
    if args.output:
        path = args.output.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, AssertionError) as exc:
        print("DIAGNOSTIC_INVALID_OR_INCONCLUSIVE: " + str(exc), file=sys.stderr)
        sys.exit(2)
