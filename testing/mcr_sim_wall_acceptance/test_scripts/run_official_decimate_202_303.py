"""Run the original stEVE decimate(0.99) acceptance recipe for seeds 202/303.

Only orchestrates existing, previously verified test scripts. Does not alter the
generator, production assets, SOFA parameters, rewards, or RL training.
The seed-101 action sequence is reused as a fixed comparison probe, NOT a
guarantee that a different vessel will reach contact.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
PYTHON_ROOT = TEST.parents[1]
REPORTS = TEST / "reports"
DEFAULT_OUTPUT = REPORTS / "official_decimate99_seed202_303"
COHORT = REPORTS / "pretrain_final_20261020" / "ten_cohort_generation.json"
REFERENCE_PLAN = REPORTS / "official_decimate99_20261020_111627" / "replay_plan.json"
GEOMETRY_PYTHON = PYTHON_ROOT / "stEVE_anatomical_vessels" / ".venv" / "bin" / "python"
SEEDS = (202, 303)
TOTAL_ACTIONS = 15
SUBSTEPS_PER_ACTION = 22


def digest(path: Path) -> str:
    hash_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_value.update(chunk)
    return hash_value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def source_inventory():
    # Exclude unrelated reports, generated data, and Git state.
    return {str(p): digest(p) for p in sorted((PYTHON_ROOT / "mcr_sim").rglob("*.py"))}


def run_process(label, cmd, log_dir: Path, timeout_seconds: int):
    logfile = log_dir / (label + ".log")
    print("RUN", label, " ".join(map(str, cmd)), flush=True)
    start = time.monotonic()
    with logfile.open("w", encoding="utf-8") as output:
        try:
            result = subprocess.run(
                [str(v) for v in cmd],
                cwd=str(PYTHON_ROOT),
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                env=os.environ.copy(),
            )
            code = result.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            code = 124
            timed_out = True
            output.write("\nTIMEOUT after %s seconds\n" % timeout_seconds)
    info = {
        "label": label, "argv": list(map(str, cmd)), "exit_code": code,
        "timed_out": timed_out, "seconds": time.monotonic() - start,
        "log": str(logfile),
    }
    print("FINISH", label, "exit", code, "seconds", round(info["seconds"], 2), flush=True)
    if code != 0:
        print("DETAILS", logfile, flush=True)
    return info


def inspect_cohort():
    cohort = read_json(COHORT)
    if cohort.get("status") != "ALL_TEN_STAGE_A_PASS":
        raise ValueError("Frozen ten-vessel cohort was not accepted")
    sources = {}
    for seed in SEEDS:
        item = cohort["assets"][str(seed)]
        if item.get("independent_stage_a") != "PASS":
            raise ValueError("Independent original Stage A not PASS for seed %d" % seed)
        directory = Path(item["directory"]).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        metadata = read_json(directory / "metadata.json")
        parameters = read_json(directory / "parameters.json")
        if (
            metadata.get("seed") != seed
            or metadata.get("arch_type") != "I"
            or metadata.get("scale_diameter_array") != [0.85]
            or metadata.get("meshing", {}).get("voxel_spacing_mm") != [0.3] * 3
            or not metadata.get("validation", {}).get("passed")
            or parameters.get("effective_profile", {}).get("meshing", {}).get("voxel_spacing_mm") != [0.3] * 3
        ):
            raise ValueError("Incorrect frozen 0.85/0.3mm input for seed %d" % seed)
        stl = directory / "collision_inner.stl"
        if item.get("stl_sha256") != digest(stl):
            raise ValueError("Input STL hash differs from accepted manifest for seed %d" % seed)
        sources[seed] = {"directory": str(directory), "stl_sha256": digest(stl)}
    return sources


def prepare_replay_plan(seed: int, source_stl_sha256: str, folder: Path):
    template = read_json(REFERENCE_PLAN)
    actions = template.get("full_rl_actions")
    if not isinstance(actions, list) or len(actions) != TOTAL_ACTIONS:
        raise ValueError("Seed-101 fixed replay template must have 15 actions")
    for i, item in enumerate(actions):
        if not isinstance(item, dict) or "index" not in item or "action" not in item:
            raise ValueError("Malformed replay action %d" % i)
        action = np.asarray(item["action"], dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.0):
            raise ValueError("Replay action outside production action space at %d" % i)
    plan = copy.deepcopy(template)
    plan["source_stl_sha256"] = source_stl_sha256
    plan["cross_seed_provenance"] = {
        "target_vessel_seed": seed,
        "template": str(REFERENCE_PLAN),
        "template_sha256": digest(REFERENCE_PLAN),
        "reuses_101_actions_without_search": True,
        "not_a_guaranteed_wall_contact_challenge": True,
        "target_branch": "lcca",
        "training_started": False,
    }
    save_json(folder / "replay_plan.json", plan)


def geometry_phase(seed: int, source: dict, output: Path, status: dict):
    wrapper = output / "wrapper_retry"
    wrapper.mkdir(parents=True, exist_ok=False)
    result = run_process(
        "geometry",
        [GEOMETRY_PYTHON, "-B", "-u", HERE / "official_decimate_geometry.py", wrapper, seed],
        output, 2400,
    )
    status["geometry_run"] = result
    report_file = wrapper / "geometry_report.json"
    if report_file.is_file():
        gate = read_json(report_file)
        status["geometry"] = {
            "stage_a": gate.get("stage_a_status"),
            "formal_passed": gate.get("formal_validation_passed"),
            "independent_stage_a": gate.get("independent_stage_a_status"),
            "input_triangles": gate.get("input_triangles"),
            "output_triangles": gate.get("output_triangles_before_serialization"),
            "surface_deviation": gate.get("surface_deviation"),
            "source_directory": gate.get("source_directory"),
            "candidate_directory": gate.get("candidate_directory"),
            "candidate_stl_sha256": gate.get("candidate_stl_sha256"),
            "strict_hausdorff_upper": "NOT_VERIFIED",
            "strict_self_intersection": "NOT_VERIFIED",
            "errors": gate.get("errors", []),
        }
        good = (
            result["exit_code"] == 0
            and gate.get("seed") == seed
            and gate.get("source_stl_sha256") == source["stl_sha256"]
            and gate.get("stage_a_status") == "PASS"
            and gate.get("formal_validation_passed") is True
            and gate.get("independent_stage_a_status") == "PASS"
            and gate.get("preservation", {}).get("status") == "PASS"
        )
    else:
        status["geometry"] = {"stage_a": "FAIL", "error": "geometry_report.json missing"}
        good = False
    status["geometry_status"] = "PASS" if good else "FAIL"
    return good


def physics_phase(seed: int, source: dict, output: Path, status: dict):
    wrapper = output / "wrapper_retry"
    gate = read_json(wrapper / "geometry_report.json")
    if gate.get("seed") != seed or gate.get("stage_a_status") != "PASS" or gate.get("preservation", {}).get("status") != "PASS":
        raise ValueError("SOFA forbidden before matching candidate Stage A PASS")
    if digest(Path(gate["source_directory"]) / "collision_inner.stl") != source["stl_sha256"]:
        raise ValueError("Frozen fine STL changed after geometry phase")
    if digest(Path(gate["candidate_directory"]) / "collision_inner.stl") != gate["candidate_stl_sha256"]:
        raise ValueError("Candidate STL changed after geometry phase")
    prepare_replay_plan(seed, source["stl_sha256"], output)
    status["physics"] = {}
    for kind in ("fine", "candidate"):
        result = run_process(
            "sofa_" + kind,
            [sys.executable, "-B", "-u", HERE / "compare_decimate_sofa.py", output, kind],
            output, 1500,
        )
        status["physics"][kind] = result
        report_file = output / ("sofa_" + kind) / "report.json"
        if report_file.is_file():
            r = read_json(report_file)
            status["physics"][kind]["sofa_status"] = r.get("status")
            status["physics"][kind]["contact_constraint_observation"] = r.get("contact_status", "NOT_OBSERVED")
            status["physics"][kind]["first_constraint_object_substep"] = r.get("first_constraint_object_substep")
            status["physics"][kind]["observed_contact_actions"] = len(r.get("contact_actions", []))
            status["physics"][kind]["mean_rl_step_seconds"] = (
                sum(row["seconds"] for row in r.get("timing", [])) / len(r["timing"])
                if r.get("timing") else None
            )
        if result["exit_code"] != 0 or status["physics"][kind].get("sofa_status") != "PASS_CAPTURE_COMPLETE":
            status["physics_status"] = "FAIL"
            return False
    status["physics_status"] = "PASS_CAPTURE_COMPLETE"
    return True


def offline_phase(output: Path, status: dict):
    result = run_process(
        "offline_geometry",
        [GEOMETRY_PYTHON, "-B", "-u", HERE / "check_decimate_contacts.py", output],
        output, 1500,
    )
    status["offline_run"] = result
    path = output / "contact_geometry_report.json"
    if path.is_file():
        report = read_json(path)
        status["offline_geometry"] = {"status": report.get("status"), "errors": report.get("errors", [])}
        for trajectory, row in report.get("measurements", {}).items():
            status["offline_geometry"][trajectory] = {
                "by_wall": {
                    wall: {
                        "status": metrics.get("status"),
                        "max_sampled_penetration_mm": metrics.get("max_sampled_entity_penetration_mm"),
                        "continuous_penetration_upper_mm": metrics.get("maximum_continuous_entity_depth_upper_mm"),
                        "uncertain_substeps": metrics.get("uncertain_frames"),
                        "total_substeps": metrics.get("frames"),
                    }
                    for wall, metrics in row.get("by_wall", {}).items()
                }
            }
    else:
        status["offline_geometry"] = {"status": "FAIL", "error": "Report missing"}
    offline_report_status = status["offline_geometry"].get("status")
    if offline_report_status == "COMPLETED_WITH_PENETRATION":
        # A positive sampled entity breach is not just an inconclusive comparison.
        status["offline_status"] = "PENETRATION_DETECTED"
    elif result["exit_code"] == 0 and offline_report_status == "COMPLETED_ENDPOINT_GEOMETRY_CHECK":
        status["offline_status"] = "COMPLETED_ENDPOINT_ONLY"
    else:
        status["offline_status"] = "FAIL"
    # This run never upgrades finite-trajectory evidence to global equivalence.
    status["collision_equivalence"] = "NOT_VERIFIED"
    status["strict_anti_penetration"] = "NOT_VERIFIED"
    return status["offline_status"] == "COMPLETED_ENDPOINT_ONLY"


def trajectory_comparison(output: Path, status: dict):
    traces = {}
    for kind in ("fine", "candidate"):
        path = output / ("sofa_" + kind) / "physical_substeps.jsonl"
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        if len(rows) != TOTAL_ACTIONS * SUBSTEPS_PER_ACTION:
            raise ValueError("Unexpected incomplete SOFA replay for " + kind)
        traces[kind] = rows
    step_errors = []
    for a, b in zip(traces["fine"], traces["candidate"]):
        if a["physical_substep"] != b["physical_substep"]:
            raise ValueError("Physical substep indices differ")
        step_errors.append(float(np.linalg.norm(np.asarray(a["tip_m"]) - np.asarray(b["tip_m"]))))
    status["trajectory"] = {
        "fine_vs_candidate_max_tip_delta_mm": max(step_errors) * 1000,
        "fine_vs_candidate_mean_tip_delta_mm": float(np.mean(step_errors)) * 1000,
        "matched_physics_substeps": len(step_errors),
    }


def write_summary(folder: Path, record: dict):
    save_json(folder / "CROSS_SEED_SUMMARY.json", record)
    lines = [
        "# Official stEVE 99% decimation, seeds 202 and 303", "",
        "Reference: seed 101 official-decimate experiment; same mesh.decimate(0.99) method, no new geometry tolerance.",
        "The original fine vessels and production SOFA code have not been modified.", "",
        "| Seed | Fine triangles | Candidate triangles | Stage A | SOFA | Contact proxy evidence | Penetration geometry |",
        "|---:|---:|---:|---|---|---|---|",
    ]
    for seed in SEEDS:
        item = record["seeds"].get(str(seed), {})
        geom = item.get("geometry", {})
        p = item.get("physics", {})
        contact = "/".join(p.get(k, {}).get("contact_constraint_observation", "NOT_RUN") for k in ("fine", "candidate"))
        lines.append("| %d | %s | %s | %s | %s | %s | %s |" % (
            seed, geom.get("input_triangles", "-"), geom.get("output_triangles", "-"),
            item.get("geometry_status", "NOT_RUN"), item.get("physics_status", "NOT_RUN"),
            contact, item.get("offline_status", "NOT_RUN")))
    lines.extend([
        "", "## Limitations", "",
        "- Replayed actions originate from seed 101. No candidate action search, no guarantee of contact in seed 202/303.",
        "- Constraint object observation is not a direct contact-force/count measurement.",
        "- Sampled/endpoint geometry is not a continuous-time, whole-surface anti-penetration certification.",
        "- Strict self-intersection and full collision equivalence remain NOT_VERIFIED.",
        "- No training, no output published in production asset directory.", "",
        "Overall execution: " + record["status"], "",
    ])
    (folder / "CROSS_SEED_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("geometry", "physics", "all"), default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPORTS.resolve()):
        parser.error("Output must remain inside the dedicated test reports directory")
    output.mkdir(parents=True, exist_ok=True)
    summary_file = output / "CROSS_SEED_SUMMARY.json"
    if args.stage in ("geometry", "all") and summary_file.exists():
        parser.error("Existing run detected; use --stage physics after geometry or choose a new --output")
    if args.stage in ("geometry", "all") and not GEOMETRY_PYTHON.is_file():
        parser.error("Approved local geometry virtualenv was not found")
    before = source_inventory()
    started = time.monotonic()
    record = (
        {"status": "IN_PROGRESS", "stage_requested": args.stage, "seeds": {}, "production_modified": None}
        if not summary_file.exists() else read_json(summary_file)
    )
    record["stage_requested"] = args.stage
    try:
        sources = inspect_cohort()
        for seed in SEEDS:
            record["seeds"].setdefault(str(seed), {"seed": seed, "source": sources[seed]})
        if args.stage in ("geometry", "all"):
            for seed in SEEDS:
                folder = output / ("seed_%d" % seed)
                folder.mkdir(parents=True, exist_ok=False)
                geometry_phase(seed, sources[seed], folder, record["seeds"][str(seed)])
                write_summary(output, record)
            if any(record["seeds"][str(seed)]["geometry_status"] != "PASS" for seed in SEEDS):
                raise RuntimeError("At least one candidate failed Stage A. SOFA testing blocked.")
        if args.stage in ("physics", "all"):
            for seed in SEEDS:
                folder = output / ("seed_%d" % seed)
                status = record["seeds"][str(seed)]
                if status.get("geometry_status") != "PASS":
                    raise RuntimeError("Missing geometry PASS for seed %d" % seed)
                if not physics_phase(seed, sources[seed], folder, status):
                    raise RuntimeError("SOFA capture failed for seed %d" % seed)
                trajectory_comparison(folder, status)
                if not offline_phase(folder, status):
                    raise RuntimeError("Offline geometric examination failed for seed %d" % seed)
                write_summary(output, record)
        record["status"] = (
            "GEOMETRY_PASS_PHYSICS_NOT_RUN" if args.stage == "geometry"
            else "COMPLETED_NOT_VERIFIED_FOR_COLLISION_EQUIVALENCE"
        )
    except BaseException:
        record["status"] = "STOPPED_WITH_FAILURE"
        record["error"] = traceback.format_exc()
        print(record["error"], flush=True)
    finally:
        after = source_inventory()
        record["production_modified"] = before != after
        record["production_source_unchanged"] = before == after
        record["total_seconds_this_invocation"] = time.monotonic() - started
        if before != after:
            record["status"] = "FAIL_PRODUCTION_CHANGED"
        write_summary(output, record)
    print("CROSS_SEED_FINAL", record["status"], "report", summary_file, flush=True)
    return 0 if record["status"] in ("GEOMETRY_PASS_PHYSICS_NOT_RUN", "COMPLETED_NOT_VERIFIED_FOR_COLLISION_EQUIVALENCE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
