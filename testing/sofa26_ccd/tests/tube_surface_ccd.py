#!/usr/bin/env python3
"""Isolated BeamAdapter real-radius tube versus plane wall CCD test."""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import Sofa, SofaRuntime
from beamadapter_ccd import PLUGIN_NAMES, arr, get_value

ROOT = Path(__file__).resolve().parents[1]
DT, R, L, NAX, NC = .005, .000665, .020, 40, 12
PAIR_MODE="all"
def serial(rows):
    return " ".join(" ".join(map(str, row)) for row in rows)
def mesh():
    pts=[]; edges=set(); tris=[]
    for i in range(NAX+1):
        for j in range(NC):
            a=2*math.pi*j/NC
            pts.append((L*i/NAX,R*math.cos(a),R*math.sin(a)))
            u=i*NC+j; v=i*NC+(j+1)%NC
            edges.add(tuple(sorted((u,v))))
            if i:
                p=(i-1)*NC+j; q=(i-1)*NC+(j+1)%NC
                edges.update((tuple(sorted((p,u))),tuple(sorted((p,q)))))
                tris.extend(((p,u,q),(q,u,v)))
    for i in (0,NAX):
        c=len(pts); pts.append((L*i/NAX,0.,0.))
        for j in range(NC):
            u=i*NC+j; v=i*NC+(j+1)%NC
            edges.add(tuple(sorted((c,u))))
            tris.append((c,v,u) if i==0 else (c,u,v))
    return pts,sorted(edges),tris
def contacts(node):
    out=[]
    def walk(n):
        for o in n.objects:
            if o.getClassName()=="FrictionContact": out.append(o.getName())
        for c in n.children: walk(c)
    walk(node); return out
def scene(method,pts,edges,tris):
    root=Sofa.Core.Node("root"); root.dt.value=DT; root.gravity.value=[0,0,0]
    root.addObject("FreeMotionAnimationLoop")
    solver=root.addObject("BlockGaussSeidelConstraintSolver",maxIterations=1000,tolerance=1e-6)
    root.addObject("CollisionPipeline"); root.addObject("BruteForceBroadPhase")
    root.addObject("BVHNarrowPhase")
    kw=dict(alarmDistance=1e-4,contactDistance=1e-5)
    if method=="ccd": kw.update(continuousCollisionType="FreeMotion",maxIterations=11000)
    root.addObject("CCDTightInclusionIntersection" if method=="ccd" else "NewProximityIntersection",**kw)
    root.addObject("CollisionResponse",response="FrictionContactConstraint",responseParams="mu=0.0")
    topo=root.addChild("edge_topology")
    topo.addObject("RodStraightSection",name="section",length=L,radius=R,nbBeams=2,nbEdgesCollis=8,nbEdgesVisu=8,youngModulus=1e6,massDensity=1000,poissonRatio=.3)
    topo.addObject("WireRestShape",name="rest_shape",template="Rigid3d",wireMaterials="@section")
    topo.addObject("EdgeSetTopologyContainer",name="edges")
    topo.addObject("EdgeSetTopologyModifier")
    topo.addObject("EdgeSetGeometryAlgorithms",template="Rigid3d")
    topo.addObject("MechanicalObject",name="topology_dofs",template="Rigid3d")
    beam=root.addChild("beam")
    beam.addObject("EulerImplicitSolver",rayleighStiffness=0,rayleighMass=0)
    beam.addObject("BTDLinearSolver",name="linear")
    beam.addObject("RegularGridTopology",name="beam_grid",n=[3,1,1],min=[0,0,0],max=[0,0,0])
    b=beam.addObject("MechanicalObject",name="beam_dofs",template="Rigid3d")
    beam.addObject("WireBeamInterpolation",name="interpolation",WireRestShape="@../edge_topology/rest_shape")
    beam.addObject("AdaptiveBeamForceFieldAndMass",name="beam_force",interpolation="@interpolation",massDensity=1000)
    beam.addObject("InterventionalRadiologyController",name="controller",template="Rigid3d",instruments="interpolation",topology="@beam_grid",startingPos=[0,0,R+.001,0,0,0,1],xtip=[L],rotationInstrument=[0],step=0,speed=0,listening=False,controlledInstrument=0)
    beam.addObject("LinearSolverConstraintCorrection")
    beam.addObject("FixedProjectiveConstraint",indices="0")
    tube=beam.addChild("tube")
    tube.addObject("MeshTopology",name="tube_topology",position=serial(pts),edges=serial(edges),triangles=serial(tris))
    t=tube.addObject("MechanicalObject",name="tube_dofs",template="Vec3d",position=serial(pts))
    mapping=tube.addObject("AdaptiveBeamMapping",name="tube_mapping",interpolation="@../interpolation",input="@../beam_dofs",output="@tube_dofs",points=serial(pts),useCurvAbs=True,isMechanical=True,mapForces=True,mapMasses=False)
    for kind in (("Point","Line","Triangle") if PAIR_MODE=="all" else (("Point",) if PAIR_MODE=="point_triangle" else ("Line",))):
        tube.addObject(kind+"CollisionModel",name="tube_"+kind,selfCollision=False)
    wall=root.addChild("wall")
    wp=[(-.01,-.02,0),(.03,-.02,0),(.03,.02,0),(-.01,.02,0)]
    wall.addObject("MeshTopology",position=serial(wp),edges="0 1 1 2 2 3 0 3 0 2",triangles="0 1 2 0 2 3")
    wall.addObject("MechanicalObject",template="Vec3d",position=serial(wp))
    for kind in (("Point","Line","Triangle") if PAIR_MODE=="all" else (("Triangle",) if PAIR_MODE=="point_triangle" else ("Line",))):
        wall.addObject(kind+"CollisionModel",name="wall_"+kind,moving=False,simulated=False,bothSide=True,selfCollision=False)
    return root,b,t,mapping,solver
def run(method,steps,pts,edges,tris,sustained=False):
    root,b,t,mapping,solver=scene(method,pts,edges,tris)
    Sofa.Simulation.init(root); Sofa.Simulation.animate(root,DT)
    initial_tube=arr(t.position)
    rings=initial_tube[:(NAX+1)*NC].reshape(NAX+1,NC,3)
    radial=np.linalg.norm(rings-rings.mean(axis=1,keepdims=True),axis=2)
    initial=dict(beam=arr(b.position).tolist(),tube_min_z_m=float(np.min(initial_tube[:,2])),
                 ring_radius_min_m=float(radial.min()),ring_radius_max_m=float(radial.max()),
                 ring_radius_max_error_m=float(np.max(np.abs(radial-R))))
    velocity=arr(b.velocity); velocity[:,2]=0; velocity[2:,2]=-1.
    b.velocity.value=velocity.tolist(); initial["velocity"]=velocity.tolist()
    records=[]
    try:
        for i in range(steps):
            if sustained and i:
                v=arr(b.velocity); v[2,2]=-1.; b.velocity.value=v.tolist()
            start=time.perf_counter(); Sofa.Simulation.animate(root,DT)
            bp,bf,tp,tf=arr(b.position),arr(b.free_position),arr(t.position),arr(t.free_position)
            names=contacts(root); rows=get_value(solver,"currentNumConstraints")
            solver_error=get_value(solver,"currentError")
            error_finite=solver_error is not None and bool(np.isfinite(solver_error))
            finite=bool(np.isfinite(bp).all() and np.isfinite(tp).all() and error_finite)
            min_z=float(np.min(tp[:,2])) if finite else None
            rec=dict(step=i+1,tube_min_z_m=min_z,penetration_m=float(max(0,-min_z)) if finite else None,contacts=len(names),contact_names=names,constraint_rows=int(rows) if rows is not None else None,solver_iterations=get_value(solver,"currentIterations"),solver_error=float(solver_error) if error_finite else None,solver_error_nonfinite=not error_finite,beam_correction_m=float(np.max(np.linalg.norm(bp[:,:3]-bf[:,:3],axis=1))) if finite else None,tube_correction_m=float(np.max(np.linalg.norm(tp-tf,axis=1))) if finite else None,beam_positions=bp.tolist() if finite else None,tube_positions=tp.tolist() if finite else None,finite=finite,runtime_s=time.perf_counter()-start)
            records.append(rec)
            if not rec["finite"]: break
    finally: Sofa.Simulation.unload(root)
    return dict(method=method,initial=initial,mapping_state=str(get_value(mapping,"componentState")),records=records)
def main():
    global PAIR_MODE
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=1)
    ap.add_argument("--pairs",choices=("all","point_triangle","line_line"),default="all")
    ap.add_argument("--sustained",action="store_true"); args=ap.parse_args(); PAIR_MODE=args.pairs
    for name in PLUGIN_NAMES:
        if not SofaRuntime.importPlugin(name): raise RuntimeError(name)
    pts,edges,tris=mesh(); manifest=dict(radius_m=R,length_m=L,axial_spacing_m=L/NAX,circumferential_samples=NC,vertices=len(pts),edges=len(edges),triangles=len(tris),geometry_source="AdaptiveBeamMapping local radial offsets from WireBeamInterpolation")
    out=ROOT/"_runtime/results"; out.mkdir(parents=True,exist_ok=True)
    (out/"tube_geometry_manifest.json").write_text(json.dumps(manifest,indent=2))
    if args.steps==1:
        a=run("discrete",1,pts,edges,tris); b=run("ccd",1,pts,edges,tris)
        result=dict(manifest=manifest,discrete=a,ccd=b,pass_test=bool(a["records"][0]["finite"] and b["records"][0]["finite"] and a["records"][0]["penetration_m"]>1e-3 and b["records"][0]["penetration_m"]<=1e-5 and b["records"][0]["constraint_rows"]>0 and b["records"][0]["beam_correction_m"]>0))
        path=out/("tube_single_step_ab.json" if PAIR_MODE=="all" else "tube_single_step_ab_"+PAIR_MODE+".json")
        print(json.dumps(dict(discrete=a["records"][0]["penetration_m"],ccd=b["records"][0]["penetration_m"],ccd_rows=b["records"][0]["constraint_rows"],ccd_contacts=b["records"][0]["contacts"])))
    else:
        b=run("ccd",args.steps,pts,edges,tris,sustained=args.sustained); p=np.array([r["penetration_m"] for r in b["records"] if r["finite"]])
        summary=dict(steps=len(b["records"]),sustained_drive=args.sustained,max_penetration_m=float(np.max(p)) if len(p) else None,p95_penetration_m=float(np.percentile(p,95)) if len(p) else None,p99_penetration_m=float(np.percentile(p,99)) if len(p) else None,penetrating_steps=int(np.sum(p>1e-5)),contact_loss_while_penetrating=sum(r["contacts"]==0 and r["penetration_m"] is not None and r["penetration_m"]>1e-5 for r in b["records"]),max_solver_iterations=max(r["solver_iterations"] for r in b["records"]),max_solver_error=max(r["solver_error"] for r in b["records"] if np.isfinite(r["solver_error"])) if any(np.isfinite(r["solver_error"]) for r in b["records"]) else None,nonfinite_steps=sum(not r["finite"] for r in b["records"]),runtime_s=sum(r["runtime_s"] for r in b["records"]))
        result=dict(manifest=manifest,ccd=b,summary=summary); path=out/"tube_multistep_ccd.json"; print(json.dumps(summary))
    manifest["observed_initial_ring_radius_max_error_m"]=b["initial"]["ring_radius_max_error_m"]
    (out/"tube_geometry_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
    if args.steps==1 and not result["pass_test"]:
        raise SystemExit(1)
if __name__=="__main__": main()
