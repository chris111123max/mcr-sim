from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

import diagnose_step665_fast_local_fd_v2 as harness
import mcr_sim.beam_sdf_unilateral_fast as production
from beam_unilateral_fast_fd_v2 import (
    build_rows_fast_local_fd_v2,
)
from beam_unilateral_fast_fd import compare_row_builds, row_sha256


BASE_ADAPTER_COMPAT = harness.AdapterCompat


class ProductionAdapterCompat(BASE_ADAPTER_COMPAT):
    def __init__(self, env, instrument):
        super().__init__(env, instrument)
        self.production_adapter = production.LiveBeamSDFAdapter(env, instrument)
        self.production_measure_enabled = False
        self.production_measure_sink = None

    def measure(self, state, spacing_m=None):
        measured = super().measure(state, spacing_m=spacing_m)
        if self.production_measure_enabled:
            production_measured = self.production_adapter.measure(
                np.asarray(state, dtype=np.float64),
                spacing_m=(
                    production.DENSE_SPACING_M
                    if spacing_m is None
                    else float(spacing_m)
                ),
            )
            sink = self.production_measure_sink
            if sink is not None:
                sink.update({
                    "sample_count": int(production_measured["sample_count"]),
                    "min_clearance_m": float(
                        production_measured["min_clearance_m"]
                    ),
                    "min_clearance_mm": float(
                        production_measured["min_clearance_mm"]
                    ),
                    "worst_index": int(production_measured["worst_index"]),
                    "matches_test_adapter_exact": bool(
                        int(production_measured["sample_count"])
                        == int(measured["sample_count"])
                        and float(production_measured["min_clearance_m"])
                        == float(measured["min_clearance_m"])
                        and np.array_equal(
                            np.asarray(
                                production_measured["worst_point_m"],
                                dtype=np.float64,
                            ),
                            np.asarray(
                                measured["worst_point_m"],
                                dtype=np.float64,
                            ),
                        )
                    ),
                })
        return measured


class ProductionVsV2Controller(harness.Step665FastFDV2Controller):
    def __init__(self, *, beam_dofs, adapter, constraint, **kwargs):
        super().__init__(
            beam_dofs=beam_dofs,
            adapter=adapter,
            constraint=constraint,
            **kwargs,
        )
        self.production_adapter = adapter.production_adapter

    def _collision_begin(self, source):
        if (
            not self.enabled
            or self.processed
            or self.current_rl_step != harness.TARGET_STEP
            or self.current_substep != harness.TARGET_SUBSTEP
        ):
            return
        self.processed = True

        q_prev = harness._as_array(self.beam_dofs.position)
        q_free = harness._as_array(self.beam_dofs.free_position)
        prev_mm = self.previous_committed_clearance_mm

        profile_start = time.perf_counter()
        dense_points, dense_clearance = production._profile(
            self.production_adapter, q_free
        )
        profile_runtime_s = time.perf_counter() - profile_start
        free_mm = float(np.min(dense_clearance)) * 1000.0

        self.record = {
            "event_source": source,
            "q_prev_clearance_mm": prev_mm,
            "q_free_clearance_mm": free_mm,
            "state_matched": bool(
                prev_mm is not None
                and abs(float(prev_mm) - harness.EXPECTED_QPREV_MM)
                <= harness.STATE_TOL_MM
                and abs(free_mm - harness.EXPECTED_QFREE_MM)
                <= harness.STATE_TOL_MM
            ),
            "dense_sample_count": int(len(dense_clearance)),
            "q_prev_full_dense_profile_count": 0,
            "q_free_full_dense_profile_count": 1,
            "q_free_external_dense_profile_count": 1,
            "v2_builder_internal_q_free_dense_count": 0,
            "production_builder_internal_q_free_dense_count": 0,
            "q_free_profile_runtime_s": float(profile_runtime_s),
            "q_prev_state_source": (
                "previous_substep_independent_committed_dense_validation"
            ),
            "target_q_candidate_solver_used": False,
            "target_native_candidate_injection_used": False,
            "uses_collision_dofs_as_constraint_source": False,
            "writes_free_position": False,
            "writes_committed_position": False,
            "production_plugin_path": str(production.resolve_plugin_path()),
        }

        if not self.record["state_matched"]:
            self.record["status"] = "STEP665_PROTECTED_STATE_MISMATCH"
            self.record["jacobian_equivalent"] = False
            self.constraint.enabled.value = False
            return

        # Refresh the archived V2 adapter's live topology metadata without
        # rebuilding a second dense q_free profile.
        self.adapter.adapter._refresh_live_interpolation(
            self.adapter._inserted_length()
        )
        v2_start = time.perf_counter()
        v2_rows = build_rows_fast_local_fd_v2(
            adapter=self.adapter,
            q_prev=q_prev,
            q_free=q_free,
            dense_points=dense_points,
            dense_clearance=dense_clearance,
            requested_margin_m=production.MARGIN_M,
        )
        v2_runtime_s = time.perf_counter() - v2_start

        production_start = time.perf_counter()
        production_rows = production.build_rows_fast_local_fd(
            adapter=self.production_adapter,
            q_prev=q_prev,
            q_free=q_free,
            dense_points=dense_points,
            dense_clearance=dense_clearance,
            requested_margin_m=production.MARGIN_M,
        )
        production_runtime_s = time.perf_counter() - production_start
        production_records, _production_support = (
            production._selected_dense_records(
                adapter=self.production_adapter,
                q_free=q_free,
                dense_points=dense_points,
                selected_indices=production_rows["selected_dense_indices"],
            )
        )
        production_rows["selected_t"] = [
            float(item["t"]) for item in production_records
        ]

        comp = compare_row_builds(v2_rows, production_rows)
        comp["selected_elements_equal"] = (
            v2_rows.get("selected_element_indices")
            == production_rows.get("selected_element_indices")
        )
        comp["support_nodes_equal"] = (
            v2_rows.get("row_support_nodes")
            == production_rows.get("row_support_nodes")
        )
        comp["selected_t_equal"] = np.array_equal(
            np.asarray(v2_rows.get("selected_t", []), dtype=np.float64),
            np.asarray(production_rows.get("selected_t", []), dtype=np.float64),
        )
        comp["source_clearances_exact_equal"] = np.array_equal(
            np.asarray(v2_rows["source_clearances"], dtype=np.float64),
            np.asarray(production_rows["source_clearances"], dtype=np.float64),
        )
        comp["free_violations_exact_equal"] = np.array_equal(
            np.asarray(v2_rows["free_violations"], dtype=np.float64),
            np.asarray(production_rows["free_violations"], dtype=np.float64),
        )
        comp["dense_sample_count_equal"] = (
            int(v2_rows["dense_sample_count"])
            == int(production_rows["dense_sample_count"])
        )
        comp["q_free_clearance_equal"] = (
            float(v2_rows["free_dense_min_clearance_m"])
            == float(production_rows["free_dense_min_clearance_m"])
        )
        comp["q_free_worst_index_equal"] = (
            int(v2_rows["free_dense_worst_index"])
            == int(production_rows["free_dense_worst_index"])
        )
        comp["row_count_equal"] = (
            len(v2_rows["free_violations"])
            == len(production_rows["free_violations"])
        )
        comp["selected_t_values_v2"] = v2_rows.get("selected_t", [])
        comp["selected_t_values_production"] = production_rows.get(
            "selected_t", []
        )
        comp["v2_row_sha256"] = row_sha256(v2_rows)
        comp["production_row_sha256"] = row_sha256(production_rows)
        comp["row_sha_equal"] = (
            comp["v2_row_sha256"] == comp["production_row_sha256"]
        )
        comp["support_mode_v2"] = v2_rows["support_metadata"]["support_mode"]
        comp["support_mode_production"] = production_rows[
            "support_metadata"
        ]["support_mode"]
        comp["support_mode_equal"] = (
            comp["support_mode_v2"] == comp["support_mode_production"]
        )

        structural_exact = all(
            bool(comp.get(key))
            for key in (
                "all_structure_equal",
                "selected_elements_equal",
                "support_nodes_equal",
                "selected_t_equal",
                "source_clearances_exact_equal",
                "free_violations_exact_equal",
                "dense_sample_count_equal",
                "q_free_clearance_equal",
                "q_free_worst_index_equal",
                "row_count_equal",
                "row_sha_equal",
                "support_mode_equal",
            )
        )
        jacobian_exact = bool(
            comp["linear_max_abs_error"] == 0.0
            and comp["angular_max_abs_error"] == 0.0
        )
        exact = bool(structural_exact and jacobian_exact)

        self.record.update({
            "status": "FAST_ROWS_ARMED" if exact else "PRODUCTION_V2_ROW_MISMATCH",
            "jacobian_equivalent": exact,
            "comparison": comp,
            "production_v2_comparison": comp,
            "v1_runtime_s": float(production_runtime_s),
            "v2_runtime_s": float(v2_runtime_s),
            "production_builder_runtime_s": float(production_runtime_s),
            "validated_v2_builder_runtime_s": float(v2_runtime_s),
            "row_count": len(production_rows["free_violations"]),
            "selected_dense_indices": production_rows[
                "selected_dense_indices"
            ],
            "selected_element_indices": production_rows[
                "selected_element_indices"
            ],
            "selected_t": production_rows.get("selected_t", []),
            "row_support_nodes": production_rows["row_support_nodes"],
            "row_offsets": production_rows["row_offsets"],
            "dof_indices": production_rows["dof_indices"],
            "row_block_counts": production_rows["row_block_counts"],
            "source_clearances": production_rows["source_clearances"],
            "free_violations": production_rows["free_violations"],
            "linear_jacobian": production_rows["linear_jacobian"],
            "angular_jacobian": production_rows["angular_jacobian"],
            "v1_row_sha256": comp["production_row_sha256"],
            "v2_row_sha256": comp["v2_row_sha256"],
            "support_mode": comp["support_mode_production"],
            "selected_point_reconstruction_max_error_m": (
                production_rows["support_metadata"].get(
                    "selected_point_reconstruction_max_error_m"
                )
            ),
            "batched_fd_point_count": int(
                production_rows["batched_fd_point_count"]
            ),
            "batched_fd_sdf_query_count": int(
                production_rows["batched_fd_sdf_query_count"]
            ),
            "full_beam_profile_evaluations_after_selection": int(
                production_rows[
                    "full_beam_profile_evaluations_after_selection"
                ]
            ),
            "production_rows_installed": bool(exact),
            "target_constraint_math_source": "production",
        })

        if exact:
            production.write_rows(self.constraint, production_rows)
            self.adapter.production_measure_sink = self.record.setdefault(
                "production_committed_measure", {}
            )
            self.adapter.production_measure_enabled = True
        else:
            self.constraint.enabled.value = False


def _production_plugin():
    return production.resolve_plugin_path()


def main():
    harness.RUNTIME = (
        harness.BEAM_ROOT
        / "_runtime"
        / "production_beam_audit_acceptance"
        / "step665"
    )
    harness.AdapterCompat = ProductionAdapterCompat
    harness.Step665FastFDV2Controller = ProductionVsV2Controller
    harness._find_beam_plugin = _production_plugin
    harness.main()

    output_path = None
    if "--output" in sys.argv:
        output_path = Path(sys.argv[sys.argv.index("--output") + 1]).expanduser()
    else:
        output_path = (
            harness.RUNTIME
            / "results"
            / "step665_fast_fd_v2_result.json"
        )
    payload = json.loads(output_path.read_text())
    payload["acceptance_stage"] = "STEP665_PRODUCTION_VS_VALIDATED_V2"
    payload["production_plugin_path"] = str(production.resolve_plugin_path())
    payload["validated_v2_builder"] = (
        "testing/beam_feasible_domain/beam_unilateral_lcp/"
        "beam_unilateral_fast_fd_v2.py"
    )
    payload["production_builder"] = "mcr_sim/beam_sdf_unilateral_fast.py"
    v2_reference_path = (
        harness.BEAM_ROOT
        / "_runtime"
        / "beam_unilateral_fast_fd_v2"
        / "step665"
        / "results"
        / "step665_fast_fd_v2_result.json"
    )
    v2_reference = json.loads(v2_reference_path.read_text())
    payload["step665_reference_action_sha256"] = (
        v2_reference["action_prefix_sha256"]
    )
    payload["action_sha_matches_step665_v2_reference"] = (
        payload.get("action_prefix_sha256")
        == payload["step665_reference_action_sha256"]
    )
    production_measure = (
        payload.get("target", {}).get("production_committed_measure", {})
    )
    production_clearance_m = production_measure.get("min_clearance_m")
    production_commit_safe = bool(
        production_clearance_m is not None
        and np.isfinite(float(production_clearance_m))
        and float(production_clearance_m)
        >= -harness.COMMITTED_PEN_LIMIT_M
    )
    payload["production_committed_safe"] = production_commit_safe
    target = payload.get("target", {})
    comparison = target.get("comparison", {})
    target_equivalent = bool(
        target.get("state_matched")
        and target.get("status") == "FAST_ROWS_ARMED"
        and target.get("production_rows_installed")
        and comparison.get("row_sha_equal")
        and comparison.get("linear_max_abs_error") == 0.0
        and comparison.get("angular_max_abs_error") == 0.0
        and target.get("constraint_active_count") == target.get("row_count")
        and target.get("committed_finite")
        and float(target.get("committed_clearance_m", float("-inf")))
        >= -harness.COMMITTED_PEN_LIMIT_M
    )
    if not payload["action_sha_matches_step665_v2_reference"]:
        payload["decision"] = "FAIL"
        payload["reason"] = "STEP665_ACTION_PREFIX_DIFFERS_FROM_V2_REFERENCE"
    elif not production_commit_safe:
        payload["decision"] = "FAIL"
        payload["reason"] = "PRODUCTION_COMMITTED_CLEARANCE_FAILED"
    elif target_equivalent:
        payload["decision"] = "PASS"
        payload["reason"] = "PRODUCTION_ROWS_IDENTICAL_TO_V2_AND_COMMITTED_SAFE"
    else:
        payload["decision"] = "FAIL"
        payload["reason"] = "PRODUCTION_V2_EQUIVALENCE_OR_SAFETY_GATE_FAILED"
    output_path.write_text(
        json.dumps(harness._json_safe(payload), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
