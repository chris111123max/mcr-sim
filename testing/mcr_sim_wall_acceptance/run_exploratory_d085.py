"""Exploratory SOFA run for the known Stage-A-failing seed-101 d085 vessel.

This test-only entry point does not modify production code or vessel assets. It
records the failing asset gate and then runs the production SOFA backend for one
zero action followed, only if healthy, by up to five low-amplitude RL actions.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import sys
import os
import re

import numpy as np

HERE = Path(__file__).resolve().parent
PYTHON_ROOT = HERE.parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from run_acceptance import (  # noqa: E402
    _log_messages, _capture_one_step, inspect_asset, jsonable, sha256,
    sofa_snapshot, source_version,
)

LOW_AMPLITUDE_ACTIONS = (
    (0.10, 0.00, 0.10),
    (-0.10, 0.00, 0.10),
    (0.00, 0.10, 0.10),
    (0.00, -0.10, 0.10),
    (0.00, 0.00, -0.10),
)


def _step_is_numerically_healthy(step):
    info = step.get("info", {})
    finite = True
    for item in step.get("substep_records", []):
        if item.get("tip_position_m") is None:
            continue
        try:
            finite &= bool(np.all(np.isfinite(np.asarray(item["tip_position_m"], dtype=float))))
            finite &= bool(np.all(np.isfinite(np.asarray(item["magnetic_field_world_t"], dtype=float))))
            finite &= bool(np.isfinite(float(item["inserted_length_m"])))
        except (KeyError, TypeError, ValueError):
            finite = False
    return bool(
        step.get("substeps_executed") == 22
        and step.get("observation_shape") == [41]
        and step.get("observation_finite")
        and step.get("transition_valid")
        and not info.get("simulation_error", False)
        and finite
        and step.get("tip_jump_check", {}).get("status") == "PASS"
        and step.get("insertion_delta_check", {}).get("status") == "PASS"
        and step.get("sofa_logs", {}).get("error_status") == "PASS"
    )



def _raw_scene_snapshot(backend):
    result = {"stage_status": getattr(backend, "stages", None)}
    handles = getattr(backend, "_handles", None)
    if handles is None:
        result["handles_available"] = False
        return result
    result["handles_available"] = True

    def value(obj, name):
        try:
            array = np.asarray(getattr(obj, name).value)
            return {"shape": list(array.shape), "values": array.tolist()}
        except Exception as exc:
            return {"read_error": "%s: %s" % (type(exc).__name__, exc)}

    controller = getattr(handles, "ir_controller", None)
    dofs = getattr(handles, "dofs", None)
    if controller is not None:
        result["irc_xtip"] = value(controller, "xtip")
        result["irc_indexFirstNode"] = value(controller, "indexFirstNode")
        result["irc_CurvAbs"] = value(controller, "CurvAbs")
    if dofs is not None:
        positions = value(dofs, "position")
        result["sofa_dofs_position"] = positions
        try:
            raw = np.asarray(dofs.position.value, dtype=float)
            first = int(np.asarray(controller.indexFirstNode.value).reshape(-1)[0])
            curv = np.asarray(controller.CurvAbs.value, dtype=float).reshape(-1)
            xtip = np.asarray(controller.xtip.value, dtype=float).reshape(-1)
            result["irc_consistency"] = {
                "positions_count": int(len(raw)),
                "first_node": first,
                "curv_abs_count": int(len(curv)),
                "curv_abs_first_m": float(curv[0]) if len(curv) else None,
                "curv_abs_last_m": float(curv[-1]) if len(curv) else None,
                "xtip_m": float(xtip[0]) if len(xtip) else None,
                "curv_abs_strictly_increasing": bool(len(curv) > 0 and np.all(np.diff(curv) > 0)),
                "active_range_end": first + len(curv),
                "active_range_matches_dofs": bool(first >= 0 and first + len(curv) == len(raw)),
                "xtip_minus_curv_abs_last_m": float(xtip[0] - curv[-1]) if len(xtip) and len(curv) else None,
            }
        except Exception as exc:
            result["irc_consistency_error"] = "%s: %s" % (type(exc).__name__, exc)
    result["last_exception"] = getattr(backend, "last_exception", None)
    try:
        result["messages"] = list(backend.last_messages)
    except Exception:
        result["messages"] = None
    return result


def _review_tip_bounds(step, mesh_info):
    # Leaving the entire STL bounding box proves exterior location even when
    # signed wall distances are unavailable. Remaining inside does not prove
    # containment and is never used to declare wall clearance.
    vertices = np.asarray(mesh_info[0].vertices_m, dtype=float)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    tol = float(np.finfo(np.float32).eps * max(float(np.max(upper-lower)), 1e-9) * 8)
    failures = []
    for record in step.get("substep_records", []):
        if record.get("tip_position_m") is None:
            continue
        point = np.asarray(record["tip_position_m"], dtype=float)
        if np.any(point < lower-tol) or np.any(point > upper+tol):
            failures.append({"physical_substep": record.get("physical_substep_index"),
                             "tip_position_m": point.tolist()})
    return {"status": "FAIL" if failures else "PASS", "bbox_min_m": lower.tolist(),
            "bbox_max_m": upper.tolist(), "numeric_roundoff_margin_m": tol,
            "outside_tip_records": failures,
            "scope": "Gross position sanity only; not proof of lumen containment or wall clearance."}

def _write_report(report):
    report_dir = HERE / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = report_dir / ("exploratory_seed101_d085_%s.json" % stamp)
    path.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "exploratory_status": report.get("exploratory_status"),
        "report": str(path),
        "stage_a_status": report.get("stage_a_context", {}).get("status"),
        "sofa_physics_status": report.get("stage_b", {}).get("status"),
    }, indent=2, ensure_ascii=False))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vessel",
        type=Path,
        default=PYTHON_ROOT / "stEVE_anatomical_vessels/generated/arch_type_I_seed_101_d085",
    )
    parser.add_argument("--extra-steps", type=int, default=5, choices=range(0, 6))
    args = parser.parse_args()
    vessel = args.vessel.expanduser().resolve()
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Exploratory physical run only; not vessel geometry acceptance.",
        "source_version": source_version(),
        "instrument_source_sha256": sha256(PYTHON_ROOT / "mcr_sim/mcr_instrument.py"),
        "vessel_directory": str(vessel),
        "requested_seed": 101,
        "expected_scale_diameter_array": [0.85],
        "stage_a_context": {"status": "NOT_RUN", "reason": None},
        "stage_b": {"status": "NOT_RUN", "reset": {}, "steps": [], "errors": []},
        "wall_diagnostics": {
            "status": "NOT_VERIFIED",
            "max_penetration_depth_mm": None,
            "reason": "Mesh self-intersection is not independently verified; wall results are diagnostic only.",
        },
        "contact_response": {
            "status": "NOT_OBSERVED",
            "reason": "Production SOFA binding exposes no stable public contact count or penetration-depth field.",
        },
        "timing_seconds": {},
        "numerical_anomalies": [],
    }
    backend = None
    env = None
    try:
        from mcr_sim.config import DEFAULT_CONFIG
        from mcr_sim.rl_env import MagneticNavigationEnv
        from mcr_sim.sofa_simulation import SofaBackend

        if not vessel.is_dir() or "seed_101_d085" not in vessel.name:
            raise ValueError("Expected the explicit seed-101 d085 directory, got %s" % vessel)

        stage_a_started = time.perf_counter()
        asset_result, bundle, mesh_info = inspect_asset(vessel, 101)
        report["stage_a_context"] = {
            "status": asset_result.get("status", "FAIL"),
            "reason": "Stage A remains failed and is recorded as-is; only its workflow gate is bypassed for this exploratory run.",
            "elapsed_seconds": time.perf_counter() - stage_a_started,
            "asset_record": asset_result,
            "generator_validation_passed": bool(bundle.metadata.get("validation", {}).get("passed", False)) if bundle else False,
        }
        if bundle is None or mesh_info is None:
            raise RuntimeError("Test-only parser could not construct the recorded d085 bundle.")
        if bundle.seed != 101 or bundle.metadata.get("scale_diameter_array") != [0.85]:
            raise ValueError("Asset seed/scale mismatch; refusing to run SOFA with the wrong model.")
        declared_hashes = bundle.metadata.get("asset_sha256", {})
        for name in ("collision_inner.stl", "visual_wall.stl", "centerline_graph.vtk"):
            path = vessel / name
            if not path.is_file() or declared_hashes.get(name) != sha256(path):
                raise ValueError("Missing or mismatched d085 asset hash: %s" % path)

        config = replace(DEFAULT_CONFIG, vessel_root=vessel.parent)
        backend = SofaBackend(config)
        # The explicit bundle came from the read-only acceptance parser. This avoids
        # changing the failed validation metadata or weakening the production loader.
        env = MagneticNavigationEnv(backend, config=config, mode="train", bundles=[bundle])

        reset_started = time.perf_counter()
        observation, reset_info = env.reset(seed=101, options={"vessel_seed": 101})
        state = backend.state()
        snapshot = sofa_snapshot(env, state, observation)
        snapshot["irc_threshold_m"] = float(backend._handles.ir_controller.threshold.value)
        snapshot["sofa_dofs_rest_position"] = np.asarray(backend._handles.dofs.rest_position.value).tolist()
        # This SofaPython3 binding has no __version__; use the version manifest
        # belonging to the actual loaded runtime rather than a configured guess.
        if not snapshot.get("sofa_python_version"):
            runtime_root = Path(os.environ["SOFA_ROOT"]).resolve()
            loaded_module = Path(backend._Sofa.__file__).resolve()
            manifest = runtime_root / "lib/cmake/SofaFrameworkConfigVersion.cmake"
            if runtime_root in loaded_module.parents and manifest.is_file():
                found = re.search(r'set\(PACKAGE_VERSION "([^"\n]+)"\)', manifest.read_text())
                if found:
                    snapshot["sofa_python_version"] = found.group(1)
                    snapshot["sofa_version_source"] = str(manifest)
        logs_after_reset = _log_messages(backend)
        statuses = {
            "SCENE_INIT": "PASS" if snapshot["sofa_dofs_position_finite"] and snapshot["sofa_dt_s"] is not None and np.isclose(snapshot["sofa_dt_s"], config.sofa.dt_s) else "FAIL",
            "SOFA_VERSION": ("NOT_VERIFIED" if not snapshot.get("sofa_python_version") else "PASS" if "21.12" in str(snapshot["sofa_python_version"]) else "FAIL"),
            "IRC_NODE_MAPPING": "PASS" if snapshot["irc_active_range_valid"] and snapshot["irc_xtip_matches_backend_insertion"] and snapshot["irc_curv_abs_matches_entrance_to_xtip"] and snapshot["active_geometry_matches_sofa_dof_slice"] else "FAIL",
            "BEAM_MODEL": "PASS" if snapshot["sofa_dofs_position_shape"] == [135, 7] and snapshot["physical_beam_count"] == 134 and snapshot["configured_body_beams"] == 100 and snapshot["configured_tip_beams"] == 34 else "FAIL",
            "OBSERVATION_41D": "PASS" if snapshot["observation_shape"] == [41] and snapshot["observation_finite"] else "FAIL",
            "SOFA_LOGS_AFTER_RESET": logs_after_reset["error_status"],
        }
        report["timing_seconds"]["reset_and_scene_init"] = time.perf_counter() - reset_started
        report["stage_b"]["reset"] = {
            "statuses": statuses,
            "reset_info": jsonable(reset_info),
            "snapshot": snapshot,
            "sofa_logs": logs_after_reset,
            "seed": bundle.seed,
            "scale_diameter_array": bundle.metadata.get("scale_diameter_array"),
        }
        report["stage_b"]["expected_substeps_per_rl_action"] = config.sofa.physics_substeps
        report["stage_b"]["extra_action_limit"] = int(args.extra_steps)
        report["stage_b"]["status"] = "FAIL" if "FAIL" in statuses.values() else "NOT_VERIFIED" if "NOT_VERIFIED" in statuses.values() else "READY"
        report["trajectory"] = [{
            "phase": "after_reset", "tip_position_m": state.tip_m.tolist(),
            "inserted_length_m": float(state.inserted_length_m),
            "magnetic_field_world_t": state.magnetic_field_world_t.tolist(),
        }]

        if report["stage_b"]["status"] == "READY":
            actions = [(0.0, 0.0, 0.0)] + list(LOW_AMPLITUDE_ACTIONS[:args.extra_steps])
            for index, action in enumerate(actions):
                label = "zero_action" if index == 0 else "low_amplitude_%02d" % index
                step_started = time.perf_counter()
                step = _capture_one_step(env, backend, action, mesh_info=mesh_info)
                step["label"] = label
                step["index"] = index
                step["numerically_healthy"] = _step_is_numerically_healthy(step)
                step["gross_position_check"] = _review_tip_bounds(step, mesh_info)
                step["wall_status_counts"] = {}
                for record in step.get("substep_records", []):
                    wall = record.get("wall", {})
                    key = wall.get("status", "NOT_VERIFIED")
                    step["wall_status_counts"][key] = step["wall_status_counts"].get(key, 0) + 1
                step["elapsed_seconds"] = time.perf_counter() - step_started
                report["timing_seconds"][label] = step["elapsed_seconds"]
                report["stage_b"]["steps"].append(step)
                for sub in step.get("substep_records", []):
                    if sub.get("tip_position_m") is not None:
                        report["trajectory"].append({
                            "phase": label, "physical_substep": sub.get("physical_substep_index"),
                            "tip_position_m": sub.get("tip_position_m"),
                            "inserted_length_m": sub.get("inserted_length_m"),
                            "magnetic_field_world_t": sub.get("magnetic_field_world_t"),
                        })
                if not step["numerically_healthy"] or step["gross_position_check"]["status"] == "FAIL":
                    report["numerical_anomalies"].append({
                        "step": label,
                        "gross_position_check": step["gross_position_check"],
                        "simulation_error": step.get("info", {}).get("simulation_error"),
                        "state_error": step.get("info", {}).get("state_error"),
                        "sofa_log_errors": step.get("sofa_logs", {}).get("errors"),
                    })
                    break
                report["trajectory"].append({
                    "phase": label, "tip_position_m": step.get("next_tip_m"),
                    "inserted_length_m": step.get("next_inserted_length_m"),
                    "magnetic_field_world_t": step.get("next_magnetic_field_world_t"),
                })
                if step.get("terminated") or step.get("truncated"):
                    report["stage_b"]["stop_reason"] = "episode ended after %s" % label
                    break

            steps = report["stage_b"]["steps"]
            total_substeps = sum(int(step.get("substeps_executed", 0)) for step in steps)
            report["stage_b"]["actual_rl_actions"] = len(steps)
            report["stage_b"]["actual_physics_substeps"] = total_substeps
            report["stage_b"]["status"] = "PASS" if steps and all(step["numerically_healthy"] and step["gross_position_check"]["status"] == "PASS" for step in steps) else "FAIL"
            report["stage_b"]["physics_substeps_status"] = "PASS" if all(step.get("substeps_executed") == 22 for step in steps) else "FAIL"
            wall_statuses = [rec.get("wall", {}).get("status", "NOT_VERIFIED") for step in steps for rec in step.get("substep_records", [])]
            report["wall_diagnostics"]["status"] = "PASS" if wall_statuses and all(x == "PASS" for x in wall_statuses) else "FAIL" if any(x == "FAIL" for x in wall_statuses) else "NOT_VERIFIED"
            report["wall_diagnostics"]["max_penetration_depth_mm"] = None
            report["wall_diagnostics"]["observed_status_counts"] = {x: wall_statuses.count(x) for x in sorted(set(wall_statuses))}
            reasons = {
                str(reason)
                for step in steps
                for reason in (step.get("info", {}).get("diagnostic_unavailable_reasons") or {})
            }
            report["stage_b"]["contact_diagnostics_unavailable_reasons"] = sorted(reasons)
            report["contact_response"]["status"] = "NOT_OBSERVED"
        elif report["stage_b"]["status"] == "NOT_VERIFIED":
            report["stage_b"]["status"] = "NOT_RUN"
            report["stage_b"]["reason"] = "SOFA version could not be confirmed; no physical action was started."

        if report["stage_b"]["status"] == "FAIL" and not report["stage_b"]["steps"]:
            report["stage_b"]["actual_rl_actions"] = 0
            report["stage_b"]["actual_physics_substeps"] = 0
    except Exception as exc:
        report["stage_b"]["status"] = "FAIL"
        report["stage_b"]["errors"].append("%s: %s" % (type(exc).__name__, exc))
        report["stage_b"]["traceback"] = traceback.format_exc()
        report["stage_b"].setdefault("actual_rl_actions", 0)
        report["stage_b"].setdefault("actual_physics_substeps", 0)
        if backend is not None:
            report["stage_b"]["sofa_logs_at_failure"] = _log_messages(backend)
            report["stage_b"]["raw_sofa_state_at_failure"] = _raw_scene_snapshot(backend)
    finally:
        if env is not None:
            env.close()
        elif backend is not None:
            backend.close()
        report["timing_seconds"]["total"] = time.perf_counter() - started

    if report["stage_a_context"]["status"] == "NOT_RUN":
        report["exploratory_status"] = "FAIL_BEFORE_STAGE_A_CONTEXT"
    elif report["stage_b"]["status"] == "PASS":
        report["exploratory_status"] = "SOFA_PHYSICS_PASS_GEOMETRY_STAGE_A_FAIL"
    else:
        report["exploratory_status"] = "SOFA_PHYSICS_%s_GEOMETRY_STAGE_A_%s" % (
            report["stage_b"]["status"], report["stage_a_context"]["status"])
    _write_report(report)
    return 0 if report["stage_b"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
