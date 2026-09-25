#!/usr/bin/env python3
"""Test-only pre-commit constrained beam update; not a SOFA time integrator."""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.spatial.transform import Rotation
import Sofa, SofaRuntime

ROOT=Path(__file__).resolve().parents[1]
R=.000665
DT=.005
L=.01
TOL=1e-8
DRIVE=-.70
PLUGIN_NAMES=("BeamAdapter","Sofa.Component.AnimationLoop","Sofa.Component.LinearSolver.Direct",
              "Sofa.Component.ODESolver.Backward","Sofa.Component.StateContainer",
              "Sofa.Component.Topology.Container.Constant","Sofa.Component.Constraint.Projective")
def make_scene():
    for n in PLUGIN_NAMES:
        if not SofaRuntime.importPlugin(n): raise RuntimeError("missing plugin "+n)
    root=Sofa.Core.Node("root"); root.dt.value=DT; root.gravity.value=[0,0,0]
    root.addObject("DefaultAnimationLoop")
    beam=root.addChild("beam")
    beam.addObject("EulerImplicitSolver")
    beam.addObject("BTDLinearSolver")
    init=np.array([[0,0,R+.001,0,0,0,1],
                   [L,0,R+.001,0,0,0,1],
                   [2*L,0,R+.001,0,0,0,1]],float)
    mo=beam.addObject("MechanicalObject",name="beam_dofs",template="Rigid3d",
                      position=" ".join(map(str,init.ravel())))
    beam.addObject("MeshTopology",name="beam_edges",edges="0 1 1 2")
    interp=beam.addObject("BeamInterpolation",name="interpolation",radius=R)
    beam.addObject("AdaptiveBeamForceFieldAndMass",name="beam_force",interpolation="@interpolation",
                   computeMass=True,massDensity=1000)
    beam.addObject("FixedProjectiveConstraint",indices="0")
    Sofa.Simulation.init(root)
    if interp.getClassName()!="BeamInterpolation" or mo.getClassName()!="MechanicalObject":
        raise RuntimeError("BeamAdapter model failed")
    return root,mo,np.asarray(mo.position.array(),float).copy()
def poses(base,u):
    q=base.copy(); d=u.reshape(2,6)
    q[1:,:3]+=d[:,:3]
    for i in (1,2):
        q[i,3:7]=Rotation.from_rotvec(d[i-1,3:6]).as_quat()
    return q
def controls(q,k):
    a,b=q[k],q[k+1]
    ta=Rotation.from_quat(a[3:7]).apply([1,0,0])
    tb=Rotation.from_quat(b[3:7]).apply([1,0,0])
    p0=a[:3]; p3=b[:3]
    return np.array([p0,p0+L*ta/3,p3-L*tb/3,p3])
def bezier(c,t):
    v=1-t
    return v**3*c[0]+3*v*v*t*c[1]+3*v*t*t*c[2]+t**3*c[3]
def split(c):
    a=(c[0]+c[1])/2; b=(c[1]+c[2])/2; d=(c[2]+c[3])/2
    e=(a+b)/2; f=(b+d)/2; m=(e+f)/2
    return np.array([c[0],a,e,m]),np.array([m,f,d,c[3]])
def cert(q):
    # For a plane, g(z)=z-R. Bernstein convex hull gives a rigorous lower bound.
    subdivisions=0; unresolved=0; min_bound=np.inf; min_sample=np.inf; worst_unresolved_length=0.
    all_samples=[]
    def rec(c,depth):
        nonlocal subdivisions,unresolved,min_bound,min_sample,worst_unresolved_length
        g=c[:,2]-R
        lo=float(np.min(g)); hi=float(np.max(g))
        min_sample=min(min_sample,float(min(g[0],g[-1],bezier(c,.5)[2]-R)))
        if lo>=-TOL:
            min_bound=min(min_bound,lo)
            return
        if hi < -TOL:
            worst_unresolved_length=max(worst_unresolved_length,L/(2**depth))
            unresolved+=1; min_bound=min(min_bound,lo)
            return
        if depth>=11 or np.linalg.norm(c[-1]-c[0])<1e-5:
            worst_unresolved_length=max(worst_unresolved_length,L/(2**depth))
            unresolved+=1; min_bound=min(min_bound,lo)
            return
        l,r=split(c); subdivisions+=1; rec(l,depth+1); rec(r,depth+1)
    for k in range(2):
        c=controls(q,k); rec(c,0)
        for t in np.linspace(0,1,17):
            all_samples.append((k,float(t),float(bezier(c,t)[2]-R)))
    return dict(min_certified_clearance_m=float(min_bound),
                min_sampled_clearance_m=float(min_sample),
                subdivisions=subdivisions,unresolved_segments=unresolved,
                max_unresolved_segment_m=worst_unresolved_length,
                samples=all_samples,feasible=bool(unresolved==0 and min_bound>=-TOL))
def sample_g(base,u,knots):
    q=poses(base,u)
    return np.array([bezier(controls(q,k),t)[2]-R for k,t in knots])
def solve_step(base,prev):
    # Incremental constrained update. q_free is an uncommitted inertial prediction.
    target=prev.copy(); target[8]+=DRIVE*DT
    free_q=poses(base,target); free_cert=cert(free_q)
    x=prev.copy(); trace=[]; reductions=0
    weights=np.array([1,1,1,1e-6,1e-6,1e-6]*2,float)
    for it in range(12):
        q=poses(base,x); state=cert(q)
        knots=[(k,float(t)) for k in range(2) for t in np.linspace(0,1,17)]
        g=sample_g(base,x,knots)
        active=np.where(g<.0005)[0]
        if len(active)==0: active=np.array([int(np.argmin(g))])
        use=[knots[i] for i in active]
        ga=g[active]
        J=np.zeros((len(use),len(x)))
        for j in range(len(x)):
            h=1e-7 if j%6<3 else 1e-5
            xp=x.copy(); xm=x.copy(); xp[j]+=h; xm[j]-=h
            J[:,j]=(sample_g(base,xp,use)-sample_g(base,xm,use))/(2*h)
        def objective(d):
            v=x+d-target
            return .5*float(np.dot(weights*v,v))
        def jac(d): return weights*(x+d-target)
        opt=minimize(objective,np.zeros_like(x),jac=jac,method="SLSQP",
                     constraints={"type":"ineq","fun":lambda d:ga+J@d-1e-7,
                                  "jac":lambda d:J},
                     options={"ftol":1e-14,"maxiter":100})
        if not opt.success: raise RuntimeError("linearized QP failed: "+str(opt.message))
        direction=opt.x
        alpha=1.
        for _ in range(30):
            trial=cert(poses(base,x+alpha*direction))
            if trial["feasible"]: break
            alpha*=.5; reductions+=1
        else:
            if np.linalg.norm(direction)>1e-10:
                raise RuntimeError("feasible line search failed")
            alpha=0.
        x=x+alpha*direction
        done=cert(poses(base,x))
        trace.append(dict(iteration=it+1,min_clearance_m=done["min_sampled_clearance_m"],
                          min_certified_clearance_m=done["min_certified_clearance_m"],
                          active_constraints=len(active),max_violation_m=max(0,-done["min_sampled_clearance_m"]),
                          line_search_alpha=alpha,qp_iterations=int(opt.nit),
                          qp_success=bool(opt.success),
                          solver_residual=float(nnls(J.T,jac(direction))[1]),
                          max_linearized_violation_m=float(max(0,np.max(1e-7-ga-J@direction))),
                          subdivisions=done["subdivisions"],unresolved_segments=done["unresolved_segments"]))
        if np.linalg.norm(alpha*direction)<1e-8 and done["feasible"]: break
    accepted=cert(poses(base,x))
    return x,dict(free_proposal=free_cert,accepted=accepted,nonlinear_iterations=len(trace),
                  line_search_alpha_min=min(t["line_search_alpha"] for t in trace),
                  reductions=reductions,active_constraints_max=max(t["active_constraints"] for t in trace),
                  trace=trace,correction_m=float(np.linalg.norm((x-target).reshape(2,6)[:,:3],axis=1).max()),
                  finite=bool(np.isfinite(x).all()),runtime_s=None)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=1); a=ap.parse_args()
    root,mo,base=make_scene()
    u=np.zeros(12); records=[]
    try:
        for i in range(a.steps):
            start=time.perf_counter(); u,out=solve_step(base,u)
            q=poses(base,u)
            if not out["accepted"]["feasible"]: raise RuntimeError("accepted infeasible state")
            mo.position.value=q.tolist()
            out["step"]=i+1; out["runtime_s"]=time.perf_counter()-start
            out["beam_dofs"]=q.tolist() if i==0 or i==a.steps-1 else None
            records.append(out)
    finally: Sofa.Simulation.unload(root)
    p=np.array([max(0,-r["accepted"]["min_sampled_clearance_m"]) for r in records])
    summary=dict(steps=len(records),max_violation_m=float(p.max()),p95_violation_m=float(np.percentile(p,95)),
                 p99_violation_m=float(np.percentile(p,99)),nonlinear_failure_count=0,
                 line_search_reduction_count=sum(r["reductions"] for r in records),
                 max_nonlinear_iterations=max(r["nonlinear_iterations"] for r in records),
                 max_active_constraints=max(r["active_constraints_max"] for r in records),
                 max_subdivisions=max(r["accepted"]["subdivisions"] for r in records),
                 unresolved_steps=sum(r["accepted"]["unresolved_segments"]>0 for r in records),
                 nan_count=sum(not r["finite"] for r in records),
                 runtime_s=sum(r["runtime_s"] for r in records))
    out=ROOT/"_runtime/results"; out.mkdir(parents=True,exist_ok=True)
    path=out/("beam_single_step.json" if a.steps==1 else "beam_multistep.json")
    path.write_text(json.dumps(dict(model="BeamAdapter Rigid3d/BeamInterpolation; test-only pre-commit SQP, not native SOFA integrator",
                                    radius_m=R,dt_s=DT,summary=summary,records=records),indent=2,allow_nan=False))
    print(json.dumps(summary))
if __name__=="__main__": main()
