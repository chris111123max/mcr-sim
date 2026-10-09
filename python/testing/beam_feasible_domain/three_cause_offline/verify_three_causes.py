#!/usr/bin/env python3
"""Read-only three-way Beam unilateral numerical diagnostics.

Never advances SOFA, alters simulation state, or changes production modules.
Run on a snapshot exported by testing/formal_training_failure capture harness.

Three independent investigations:
 (1) rotational FD Jacobian frame convention;
 (2) linear unilateral row satisfaction;
 (3) nonlinear clearance gap versus the linear prediction.

Unknown/absent snapshot schemas are INCONCLUSIVE, not a fabricated PASS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

EPS_M = 1e-12
MARGIN_M = 1e-4
ALIASES = {
    "q_free": ("q_free", "free_position", "beam_q_free"),
    "q_committed": ("q_committed", "committed_position", "beam_q_committed"),
    "q_prev": ("q_prev", "previous_position", "beam_q_prev"),
    "free_profile": ("q_free_dense_clearance", "q_free_dense_profile", "free_dense_clearance", "free_clearance_profile", "free_dense_profile"),
    "committed_profile": ("q_committed_dense_clearance", "q_committed_dense_profile", "committed_dense_clearance", "committed_clearance_profile", "committed_dense_profile"),
    "indices": ("selected_dense_indices", "row_dense_indices"),
    "row_offsets": ("row_offsets", "rowOffsets"),
    "dofs": ("dof_indices", "dofIndices"),
    "jl": ("linear_jacobian", "linearJacobian", "linear_blocks"),
    "ja": ("angular_jacobian", "angularJacobian", "angular_blocks"),
    "g_free": ("free_violations", "freeViolations", "row_free_violations"),
    "source": ("source_clearances", "sourceClearances"),
    "margin": ("requested_margin_m", "margin_m"),
    "rotation_expected_world": ("rotation_jacobian_world_reference", "rotation_jacobian_world_fd", "angular_jacobian_world_reference"),
    "rotation_local": ("rotation_jacobian_local", "angular_jacobian_local"),
    "rotation_nodes": ("rotation_jacobian_nodes", "angular_jacobian_nodes"),
}

def locate(data, key):
    for k in ALIASES[key]:
        if k in data:
            return np.asarray(data[k])
    return None

def require(data, *keys):
    values = [locate(data, k) for k in keys]
    missing = [key for key, value in zip(keys, values) if value is None]
    if missing:
        raise ValueError("Missing snapshot fields: " + ", ".join(missing))
    return values

def quat_world_increment(q_from, q_to):
    """SOFA-world-axis rotation-vector of q_to * inverse(q_from).

    Used only for numeric comparison. Verify the SOFA convention separately;
    this method never assumes the angular Jacobian provided is world-frame.
    """
    a = Rotation.from_quat(q_from[:, 3:7])
    b = Rotation.from_quat(q_to[:, 3:7])
    return (b * a.inv()).as_rotvec()

def row_prediction(data):
    qf, qc, offsets, dofs, jl, ja, gf = require(
        data, "q_free", "q_committed", "row_offsets", "dofs", "jl", "ja", "g_free"
    )
    qf = np.asarray(qf, dtype=float)
    qc = np.asarray(qc, dtype=float)
    offsets = np.asarray(offsets, dtype=int).reshape(-1)
    dofs = np.asarray(dofs, dtype=int).reshape(-1)
    jl = np.asarray(jl, dtype=float).reshape(-1, 3)
    ja = np.asarray(ja, dtype=float).reshape(-1, 3)
    gf = np.asarray(gf, dtype=float).reshape(-1)
    if qf.shape != qc.shape or qf.ndim != 2 or qf.shape[1] != 7:
        raise ValueError("Rigid3 q_free / q_committed shapes must match (N, 7)")
    if len(offsets) != len(gf) + 1 or offsets[0] != 0:
        raise ValueError("CSR row_offsets mismatch")
    if offsets[-1] != len(dofs) or len(dofs) != len(jl) or len(dofs) != len(ja):
        raise ValueError("CSR nnz mismatch")
    if np.any(np.diff(offsets) < 0) or np.any(dofs < 0) or np.any(dofs >= len(qf)):
        raise ValueError("Invalid CSR row or DOF index")
    if not all(np.isfinite(x).all() for x in (qf, qc, jl, ja, gf)):
        raise ValueError("Nonfinite input")
    dq = qc[:, :3] - qf[:, :3]
    drot_world = quat_world_increment(qf, qc)
    # SOFA world convention: left quaternion increment. A native SOFA A/B
    # remains the authoritative physical solver test.
    predicted = gf.astype(float).copy()
    contributions = []
    for r in range(len(gf)):
        subtotal = 0.0
        for j in range(offsets[r], offsets[r + 1]):
            n = dofs[j]
            subtotal += float(jl[j] @ dq[n] + ja[j] @ drot_world[n])
        predicted[r] += subtotal
        contributions.append(subtotal)
    return predicted, gf, contributions, qf, qc, ja, dofs

def check_rotation(data):
    """Test J_local * R.T against *independent* stored world-FD reference.

    Crucially, a transformed Jacobian alone cannot demonstrate correctness:
    this check requires independently captured world-axis FD derivatives.
    """
    try:
        (reference, local, nodes, qf) = require(
            data, "rotation_expected_world", "rotation_local", "rotation_nodes", "q_free"
        )
        reference = np.asarray(reference, dtype=float).reshape(-1, 3)
        local = np.asarray(local, dtype=float).reshape(-1, 3)
        nodes = np.asarray(nodes, dtype=int).reshape(-1)
        qf = np.asarray(qf, dtype=float)
        if not (len(reference) == len(local) == len(nodes)):
            raise ValueError("Rotation reference/Jacobian/node counts differ")
        if len(reference) == 0:
            raise ValueError("No rotation derivatives")
        if np.any(nodes < 0) or np.any(nodes >= len(qf)):
            raise ValueError("Rotation DOF indices invalid")
        if not all(np.isfinite(x).all() for x in (reference, local, qf)):
            raise ValueError("Nonfinite values")
        matrices = Rotation.from_quat(qf[nodes, 3:7]).as_matrix()
        transformed = np.einsum("ni,nji->nj", local, matrices)
        # local-row @ R.T; einsum uses matrices[j,i].
        original_err = np.linalg.norm(local - reference, axis=1)
        corrected_err = np.linalg.norm(transformed - reference, axis=1)
        return {"status":"MEASURED", "sample_count":len(nodes),
                "raw_max_error_m_per_rad":float(original_err.max()),
                "converted_max_error_m_per_rad":float(corrected_err.max()),
                "conversion_reduces_max_error":bool(corrected_err.max() < original_err.max()),
                "note":"Reference must be independently computed using SOFA world-axis perturbations; inspect provenance."}
    except (ValueError, TypeError) as exc:
        return {"status":"INCONCLUSIVE", "reason":str(exc)}

def check_rows(data):
    try:
        pred, gf, _, _, _, _, _ = row_prediction(data)
        return {"status":"MEASURED", "row_count":len(pred),
                "min_free_linear_gap_mm":float(np.min(gf)*1000),
                "min_committed_linear_gap_mm":float(np.min(pred)*1000),
                "violated_row_indices":np.flatnonzero(pred < -EPS_M).tolist(),
                "all_rows_satisfied_within_1e-12_m":bool(np.all(pred >= -EPS_M)),
                "note":"First-order prediction in world rotation increments; not a native lambda/residual readout."}
    except (ValueError, TypeError) as exc:
        return {"status":"INCONCLUSIVE", "reason":str(exc)}

def check_nonlinearity(data):
    try:
        pred, gf, _, _, _, _, _ = row_prediction(data)
        free, committed, indices = require(data, "free_profile", "committed_profile", "indices")
        free = np.asarray(free, dtype=float).reshape(-1)
        committed = np.asarray(committed, dtype=float).reshape(-1)
        indices = np.asarray(indices, dtype=int).reshape(-1)
        if len(free) != len(committed) or len(free) == 0:
            raise ValueError("Dense profiles need identical nonzero sample count")
        if len(indices) != len(pred) or np.any(indices < 0) or np.any(indices >= len(free)):
            raise ValueError("Row indices are not aligned with dense profile")
        if not np.isfinite(free).all() or not np.isfinite(committed).all():
            raise ValueError("Nonfinite dense profiles")
        margin_arr = locate(data, "margin")
        margin = float(margin_arr.reshape(-1)[0]) if margin_arr is not None else MARGIN_M
        actual_g = committed[indices] - margin
        errors = actual_g - pred
        free_worst = int(np.argmin(free))
        committed_worst = int(np.argmin(committed))
        return {"status":"MEASURED", "sample_count":len(free),
                "row_count":len(indices), "margin_mm":margin*1000,
                "max_abs_selected_nonlinear_prediction_error_mm":float(np.max(np.abs(errors))*1000),
                "selected_actual_min_gap_mm":float(np.min(actual_g)*1000),
                "global_committed_min_clearance_mm":float(np.min(committed)*1000),
                "free_worst_index":free_worst, "committed_worst_index":committed_worst,
                "committed_worst_selected":bool(committed_worst in set(indices.tolist())),
                "minimum_migrated":bool(free_worst != committed_worst),
                "note":"Requires saved independent full-beam committed profile. A nonzero error alone does not establish original deep-penetration cause."}
    except (ValueError, TypeError) as exc:
        return {"status":"INCONCLUSIVE", "reason":str(exc)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Path to .npz snapshot")
    parser.add_argument("--metadata", type=Path, help="Optional capture .json metadata")
    parser.add_argument("--output", type=Path, help="Report .json (defaults next to snapshot)")
    args = parser.parse_args()
    if not args.snapshot.is_file():
        parser.error(f"Snapshot not found: {args.snapshot}")
    with np.load(args.snapshot, allow_pickle=False) as archive:
        data = {k:archive[k] for k in archive.files}
    metadata = {}
    if args.metadata:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        # Numeric data may be split between capture npz and metadata JSON.
        for k,v in metadata.items():
            if k not in data and isinstance(v,(int,float,list)):
                try:
                    data[k]=np.asarray(v)
                except Exception:
                    pass
    report = {
        "snapshot":str(args.snapshot.resolve()), "metadata":metadata,
        "available_npz_fields":sorted(data),
        "rotation_jacobian_frame":check_rotation(data),
        "linear_row_satisfaction":check_rows(data),
        "nonlinear_sdf_gap":check_nonlinearity(data),
        "original_deep_penetration_reproduced":False,
        "production_modified":False,
        "remarks":"This is read-only offline arithmetic; SOFA native same-frame A/B is required to establish physical causal effects."
    }
    output = args.output or args.snapshot.with_name(args.snapshot.stem+"_three_causes_report.json")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print(f"REPORT={output}")
    return 2 if all(report[k]["status"]=="INCONCLUSIVE" for k in (
        "rotation_jacobian_frame","linear_row_satisfaction","nonlinear_sdf_gap"
    )) else 0

if __name__ == "__main__":
    sys.exit(main())
