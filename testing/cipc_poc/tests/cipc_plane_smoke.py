#!/usr/bin/env python3
"""Isolated C-IPC rod/triangle finite-thickness gravity crossing smoke test."""
import argparse
import json
import math
import time
from pathlib import Path
from JGSL import Vector3d
import Drivers

ROOT = Path(__file__).resolve().parents[1]
RADIUS = 0.000665
DT = 0.005


def read_vertices(path):
    return [[float(x) for x in line.split()[1:4]] for line in path.read_text().splitlines() if line.startswith("v ")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1)
    args = ap.parse_args()
    work = ROOT / "_runtime" / "sandbox"
    work.mkdir(parents=True, exist_ok=True)
    plane = work / "plane.obj"
    plane.write_text("v -0.02 0 -0.02\nv 0.02 0 -0.02\nv 0.02 0 0.02\nv -0.02 0 0.02\nf 1 2 3\nf 1 3 4\n")
    sim = Drivers.FEMDiscreteShellBase("double", 3)
    zero = Vector3d(0, 0, 0)
    sim.add_shell_3D(str(plane), zero, zero, Vector3d(0, 1, 0), 0)
    sim.set_DBC(Vector3d(-0.01, -0.01, -0.01), Vector3d(1.01, 1.01, 1.01), zero, zero, Vector3d(0, 1, 0), 0)
    sim.make_and_add_rod_3D(0.008, 8, Vector3d(0, 0.003, 0), zero, Vector3d(0, 1, 0), 0, Vector3d(1, 1, 1))
    sim.dt = DT
    sim.withCollision = True
    sim.epsv2 = 1e-10
    sim.PNTol = 1e-5
    sim.initialize(500, 1e5, 0.3, 0.0001, 0)
    sim.initialize_rod(1000, 1e3, 1, RADIUS)
    sim.initialize_OIPC(RADIUS, RADIUS)
    sim.write(0)
    initial = read_vertices(Path(sim.output_folder) / "rod0.obj")
    records = []
    for k in range(args.steps):
        tic = time.perf_counter()
        iterations_before = sim.PNIterCount
        sim.advance_one_time_step(DT)
        sim.write(k+1)
        elapsed = time.perf_counter() - tic
        vertices = read_vertices(Path(sim.output_folder) / f"rod{k+1}.obj")
        shell = read_vertices(Path(sim.output_folder) / f"shell{k+1}.obj")[:4]
        shell_initial = read_vertices(Path(sim.output_folder) / "shell0.obj")[:4]
        min_clearance = min(v[1] for v in vertices) - RADIUS
        max_shell_motion = max(math.dist(a,b) for a,b in zip(shell,shell_initial))
        records.append(dict(step=k+1, min_clearance_m=min_clearance, static_shell_max_motion_m=max_shell_motion, solver_iterations=sim.PNIterCount-iterations_before, runtime_s=elapsed, finite=all(math.isfinite(x) for v in vertices for x in v)))
        if not records[-1]["finite"]:
            raise RuntimeError(f"nonfinite position at {k+1}")
    result = dict(model="C-IPC rod vs static two-triangle plane", radius_m=RADIUS, dt_s=DT, steps=len(records), initial_clearance_m=min(v[1] for v in initial)-RADIUS, min_clearance_m=min(r["min_clearance_m"] for r in records), max_penetration_m=max(0,-min(r["min_clearance_m"] for r in records)), total_runtime_s=sum(r["runtime_s"] for r in records), max_solver_iterations=max(r["solver_iterations"] for r in records), records=records)
    out=ROOT / "_runtime" / "results" / f"cipc_plane_{args.steps}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
