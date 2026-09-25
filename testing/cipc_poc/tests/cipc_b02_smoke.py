#!/usr/bin/env python3
"""Isolated C-IPC discrete rod against the actual B02 inner vessel mesh."""
import argparse
import json
import math
import struct
import time
from pathlib import Path
import numpy as np
from JGSL import Vector3d, MeshIO
import Drivers
from mcr_sim.vessel_assets import load_signed_distance_grid

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
ASSET = PROJECT / "mesh/train/B02"
RADIUS = 0.000665
DT = 0.005


def convert_stl_to_obj(stl, obj):
    data = stl.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84+50*count:
        raise ValueError("expected binary STL")
    records=np.frombuffer(data, dtype=np.dtype([("normal","<f4",3),("vertices","<f4",(3,3)),("attribute","<u2")]), offset=84, count=count)
    vertices, inverse=np.unique(records["vertices"].reshape((-1,3)), axis=0, return_inverse=True)
    faces=inverse.reshape((-1,3))
    with obj.open("w") as f:
        for xyz in vertices*0.001:
            f.write("v %.9g %.9g %.9g\n" % tuple(xyz))
        for abc in faces+1:
            f.write("f %d %d %d\n" % tuple(abc))
    return len(vertices),len(faces)


def read_vertices(path):
    return np.array([[float(x) for x in line.split()[1:4]] for line in path.read_text().splitlines() if line.startswith("v ")],float)


def dense_rod_clearance_m(vertices, sdf):
    points=np.vstack([a[None,:]*(1-t)[:,None]+b[None,:]*t[:,None] for a,b in zip(vertices[:-1],vertices[1:]) for t in [np.linspace(0,1,9,endpoint=False)]])
    signed_mm=np.asarray(sdf.sample(points*1000),float)
    if not np.all(np.isfinite(signed_mm)):
        return float("-inf"),False
    return float(np.min(-signed_mm*0.001-RADIUS)),True


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--steps",type=int,default=1);args=ap.parse_args()
    work=ROOT/"_runtime/sandbox";work.mkdir(parents=True,exist_ok=True)
    obj=work/"B02_collision_inner_m.obj"
    if not obj.exists():
        vertex_count,face_count=convert_stl_to_obj(ASSET/"collision_inner.stl",obj)
    else:
        vertex_count=sum(line.startswith("v ") for line in obj.open())
        face_count=sum(line.startswith("f ") for line in obj.open())
    sdf=load_signed_distance_grid(ASSET/"vessel_sdf.vti")
    tic_init=time.perf_counter()
    sim=Drivers.FEMDiscreteShellBase("double",3)
    zero=Vector3d(0,0,0)
    sim.add_shell_3D(str(obj),zero,zero,Vector3d(0,1,0),0)
    sim.set_DBC(Vector3d(-0.01,-0.01,-0.01),Vector3d(1.01,1.01,1.01),zero,zero,Vector3d(0,1,0),0)
    sim.make_and_add_rod_3D(0.008,8,Vector3d(0.0013,0.008,-0.0004),zero,Vector3d(0,0,1),90,Vector3d(1,1,1))
    sim.gravity=Vector3d(9.81,0,0)
    sim.dt=DT;sim.withCollision=True;sim.epsv2=1e-10;sim.PNTol=1e-5
    sim.initialize(500,1e5,0.3,0.0001,0)
    sim.initialize_rod(1000,1e3,1,RADIUS)
    sim.initialize_OIPC(RADIUS,RADIUS)
    init_s=time.perf_counter()-tic_init
    rod_file=work/"b02_rod_latest.obj"
    MeshIO.Write_SegMesh_Obj(sim.X,sim.rod,str(rod_file))
    initial,valid=dense_rod_clearance_m(read_vertices(rod_file),sdf)
    if not valid or initial<0: raise RuntimeError("initial rod not SDF-feasible")
    records=[]
    for k in range(args.steps):
        before=sim.PNIterCount;tic=time.perf_counter()
        sim.advance_one_time_step(DT)
        MeshIO.Write_SegMesh_Obj(sim.X,sim.rod,str(rod_file))
        elapsed=time.perf_counter()-tic
        xyz=read_vertices(rod_file)
        clearance,valid=dense_rod_clearance_m(xyz,sdf)
        record=dict(step=k+1,clearance_m=clearance if valid else None,valid=valid,solver_iterations=sim.PNIterCount-before,runtime_s=elapsed,finite=bool(np.isfinite(xyz).all()))
        records.append(record)
        if not valid or not record["finite"] or clearance < -0.0001: break
    good=[r["clearance_m"] for r in records if r["clearance_m"] is not None]
    result=dict(model="C-IPC rod vs full B02 vessel",mesh_vertices=vertex_count,mesh_triangles=face_count,radius_m=RADIUS,dt_s=DT,requested_steps=args.steps,completed_steps=len(records),initial_clearance_m=initial,min_clearance_m=min(good) if good else None,max_penetration_m=max(0,-min(good)) if good else None,init_runtime_s=init_s,total_runtime_s=sum(r["runtime_s"] for r in records),max_solver_iterations=max(r["solver_iterations"] for r in records),records=records)
    out=ROOT/"_runtime/results"/f"cipc_b02_{args.steps}.json";out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!="records"}))

if __name__=="__main__": main()
