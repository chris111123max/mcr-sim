#!/usr/bin/env python3
"""Test-only audit of captured step-776 real BeamAdapter geometry and production SDF."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from mcr_sim.vessel_assets import load_signed_distance_grid, sim_points_to_asset_source
ROOT=Path(__file__).resolve().parents[3]
RESULTS=ROOT/'testing/beam_feasible_domain/_runtime/results'
CAP=RESULTS/'b02_step776_capture.json'; NPZ=RESULTS/'b02_step776_states.npz'; OUT=RESULTS/'b02_step776_adapter_audit.json'
def bezier(p0,p1,p2,p3,t):
 t=np.asarray(t,dtype=np.float64).reshape(-1,1); u=1-t
 return u**3*p0+3*u*u*t*p1+3*u*t*t*p2+t**3*p3
def beam_curve(q,c,max_step=.00025):
 ip=c['interpolation']; edges=np.asarray(c['topology']['edges'],int); xyz=[]; meta=[]
 for i,(eid,L) in enumerate(zip(ip['edgeList'],ip['lengthList'])):
  n0,n1=edges[int(eid)]; a,b=q[n0],q[n1]
  # Scene mcr_instrument.py sets rotationInstrument=[0.], so addBeam stores identity offsets.
  r0=R.from_quat(a[3:7]); r1=R.from_quat(b[3:7]); p0=a[:3]; p3=b[:3]
  p1=p0+r0.apply([1.,0,0])*(L/3); p2=p3+r1.apply([-1.,0,0])*(L/3)
  linear=np.dot(p1-p0,p2-p1)<0 and np.linalg.norm(p3-p0)<.4*L
  t=np.linspace(0,1,max(2,int(np.ceil(L/max_step))+1))
  pts=p0[None,:]*(1-t[:,None])+p3[None,:]*t[:,None] if linear else bezier(p0,p1,p2,p3,t)
  if xyz and np.linalg.norm(xyz[-1][-1]-pts[0])<1e-10: pts=pts[1:]
  xyz.append(pts); meta.append({'edge_list_index':i,'topology_edge_id':int(eid),'nodes':[int(n0),int(n1)],'rest_length_m':float(L),'samples':len(pts),'compressed_linear_fallback':bool(linear)})
 return np.concatenate(xyz),meta
def src_to_sim(p,tr,off,scale):
 tr=np.asarray(tr); return R.from_quat(tr[3:7]).apply(np.asarray(p)*scale)+tr[:3]+off
def clearance(points,sdf,tr,off,scale,radius):
 src=sim_points_to_asset_source(points,tr,off,scale); sd=np.asarray(sdf.sample(src)); g=-sd*scale-radius; finite=np.isfinite(g); j=int(np.argmin(g[finite])) if np.any(finite) else None
 vals=g[finite]
 return {'valid_count':int(finite.sum()),'sample_count':len(g),'min_clearance_m':float(np.min(vals)) if len(vals) else None,'max_penetration_m':float(max(0,-np.min(vals))) if len(vals) else None,'clearance_p01_m':float(np.quantile(vals,.01)) if len(vals) else None,'clearance_p05_m':float(np.quantile(vals,.05)) if len(vals) else None,'min_sample_index':int(np.flatnonzero(finite)[j]) if j is not None else None}
def clean(x):
 if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [clean(v) for v in x]
 if isinstance(x,(float,np.floating)) and not np.isfinite(x): return str(x)
 if isinstance(x,np.generic): return x.item()
 return x
def main():
 c=json.loads(CAP.read_text()); s=np.load(NPZ)
 if not c.get('action_prefix_matches') or c['rl_step']!=776 or c['physics_substep']!=1: raise RuntimeError('capture target/action mismatch')
 qp=np.asarray(s['q_prev'],float); qf=np.asarray(s['q_free'],float); qn=np.asarray(s['q_native'],float)
 if not np.allclose(np.linalg.norm(qf[:,3:7],axis=1),1,atol=1e-4): raise RuntimeError('bad quaternion')
 scale=float(c['sdf_metadata']['asset_source_to_sim_scale']); tr=np.asarray(c['sdf_metadata']['asset_T_env_sim']); off=np.asarray(c['sdf_metadata']['asset_offset_sim']); rad=float(c['sdf_metadata']['catheter_radius_m']); sdf=load_signed_distance_grid(c['sdf_metadata']['path'])
 bp,_=beam_curve(qp,c); bf,meta=beam_curve(qf,c); bn,_=beam_curve(qn,c); cf=np.asarray(s['collision_free_diagnostic_only'],float)[:,:3]; cn=np.asarray(s['collision_native_diagnostic_only'],float)[:,:3]
 df=cKDTree(bf).query(cf)[0]; dn=cKDTree(bn).query(cn)[0]
 def mmstats(x): return {'mean':float(np.mean(x)*1000),'p95':float(np.quantile(x,.95)*1000),'max':float(np.max(x)*1000)}
 consistency={'free_collision_to_beam_mm':mmstats(df),'native_collision_to_beam_mm':mmstats(dn)}
 previous=clearance(bp,sdf,tr,off,scale,rad); free=clearance(bf,sdf,tr,off,scale,rad); native=clearance(bn,sdf,tr,off,scale,rad); cfclear=clearance(cf,sdf,tr,off,scale,rad); cnclear=clearance(cn,sdf,tr,off,scale,rad)
 c['real_beam_clearances']={'previous':previous,'free':free,'native_contact_baseline_only':native,'geometry':'BeamAdapter Rigid3d + active edge/length + cubic spline; 0.25 mm samples'}
 c['collision_dof_clearances_diagnostic_only']={'free':cfclear,'native':cnclear,'constraint_source':False}
 CAP.write_text(json.dumps(clean(c),indent=2,allow_nan=False)+'\n')
 vals=sdf.values; picks=[]
 inner=vals[1:-1,1:-1,1:-1]; inner_flat=int(np.argmax(inner)); iz,iy,ix=np.unravel_index(inner_flat,inner.shape); outer_flat=np.ravel_multi_index((iz+1,iy+1,ix+1),vals.shape)
 for label,flat in [('inside_min',int(np.argmin(vals))),('outside_max',int(outer_flat))]:
  z,y,x=np.unravel_index(flat,vals.shape); src=sdf.origin+sdf.spacing*np.array([x,y,z]); sim=src_to_sim(src,tr,off,scale); back=sim_points_to_asset_source(sim,tr,off,scale)[0]; sample=float(sdf.sample(back)); exact=float(vals.reshape(-1)[flat])
  picks.append({'label':label,'signed_distance_mm':exact,'roundtrip_sample_mm':sample,'roundtrip_error_mm':abs(sample-exact),'source_mm':src.tolist(),'sim_m':sim.tolist()})
 signok=picks[0]['signed_distance_mm']<0<picks[1]['signed_distance_mm'] and max(x['roundtrip_error_mm'] for x in picks)<1e-4
 geomok=max(consistency['free_collision_to_beam_mm']['p95'],consistency['native_collision_to_beam_mm']['p95'])<=.20 and max(consistency['free_collision_to_beam_mm']['max'],consistency['native_collision_to_beam_mm']['max'])<=.75
 out={'test':'B02 step776 BeamAdapter reconstruction + independent SDF audit','frame':'RL step 776 / physics substep 1 free Beam state','action_sha256':c['action_sha256'],'capture_sha256':hashlib.sha256(CAP.read_bytes()).hexdigest(),'state_npz_sha256':hashlib.sha256(NPZ.read_bytes()).hexdigest(),'interpolation_source':'BeamInterpolation.inl getControlPointsFromFrame/InterpolateTransformUsingSpline; controller mcr_instrument.py rotationInstrument=[0.]; exact compressed fallback included','active_elements':meta,'max_sample_step_mm':.25,'dense_beam_sample_count':len(bf),'collision_dofs_are_diagnostic_only':True,'sdf_sign_unit_check':{'negative_inside_positive_outside':bool(signok),'min_mm':float(vals.min()),'max_mm':float(vals.max()),'grid_roundtrip_examples':picks,'production_clearance_formula':'-signed_distance_mm*asset_source_to_sim_scale-catheter_radius_m'},'geometry_check':consistency,'geometry_check_passed':bool(geomok),'free_state_clearance':free,'previous_state_clearance':previous,'native_state_clearance':native,'collision_dof_clearance_diagnostic_only':{'free':cfclear,'native':cnclear},'safe_to_run_solver':bool(signok and geomok and free['valid_count']==free['sample_count'] and free['min_clearance_m'] is not None and free['min_clearance_m']<0),'solver_run':False,'production_files_modified':False}
 out.update({'beam_state_source':'captured beam DOFs free_position (q_free), arbitrary copied Rigid3d array; q_native used only for diagnostic baseline','beam_geometry_source':'production BeamAdapter active edgeList/lengthList/curvAbsList and BeamInterpolation Bezier/fallback formula over physically inserted elements','uses_collision_dofs_as_constraint_source':False,'production_sdf_usage':{'used':True,'path':c['sdf_metadata']['path'],'source_scalar':sdf.scalar_name,'asset_source_to_sim_scale':scale,'asset_T_env_sim':tr.tolist(),'asset_offset_sim':off.tolist()},'native_contact_input_contamination':False,'used_native_post_contact_state_as_solver_input':False,'catheter_radius_m':rad,'rotation_parameterization':'Rigid3d xyzw quaternion read through scipy Rotation for geometry; no quaternion subtraction; feasible solve uses SO(3) tangent-composition (recorded in solve JSON)','production_files_modified':False})
 OUT.write_text(json.dumps(clean(out),indent=2,allow_nan=False)+'\n'); print(json.dumps(clean({'audit':'PASS' if out['safe_to_run_solver'] else 'STOP','geometry_passed':geomok,'sdf_sign_passed':signok,'free_clearance_mm':None if free['min_clearance_m'] is None else free['min_clearance_m']*1000,'free_state':free,'native_state':native,'geometry':consistency,'output':str(OUT)}),allow_nan=False))
if __name__=='__main__': main()
