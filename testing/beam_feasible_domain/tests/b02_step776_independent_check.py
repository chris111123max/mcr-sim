#!/usr/bin/env python3
"""Independent dense re-evaluation of the accepted real-beam state; no SOFA scene writeback."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from mcr_sim.vessel_assets import load_signed_distance_grid,sim_points_to_asset_source
ROOT=Path(__file__).resolve().parents[3]; RES=ROOT/'testing/beam_feasible_domain/_runtime/results'
CAP=RES/'b02_step776_capture.json'; SOL=RES/'b02_step776_feasible_solve.json'; AUD=RES/'b02_step776_adapter_audit.json'; OUT=RES/'b02_step776_independent_check.json'
def curve_independent(q,specs):
 allpts=[]; meta=[]
 for ei,e in enumerate(specs):
  i,j=e['nodes']; L=e['rest_length_m']; a=q[i]; b=q[j]
  # Rebuild the controller's local frame offsets independently (rotationInstrument is zero in the captured MCR scene).
  qa=R.from_quat(a[3:7]); qb=R.from_quat(b[3:7]); P0=a[:3]; P3=b[:3]
  C1=P0+(qa.apply(np.array([1.,0.,0.])))*(L/3.)
  C2=P3+(qb.apply(np.array([-1.,0.,0.])))*(L/3.)
  compressed=(float(np.dot(C1-P0,C2-C1))<0. and float(np.linalg.norm(P3-P0))<.4*L)
  # Keep a fixed 0.01 mm parametric/arclength grid, including both endpoints.
  n=max(2,int(np.ceil(L/1e-5))+1); u=np.linspace(0.,1.,n)
  if compressed: P=P0[None,:]*(1-u[:,None])+P3[None,:]*u[:,None]
  else:
   v=1-u[:,None]; P=v**3*P0+3*v*v*u[:,None]*C1+3*v*u[:,None]**2*C2+u[:,None]**3*P3
  start,end=e['curv_abs_m']; arc=start+u*(end-start)
  if allpts and np.linalg.norm(allpts[-1][-1]-P[0])<1e-10: P=P[1:]; arc=arc[1:]; u=u[1:]
  allpts.append(P); meta.extend({'element_index':ei,'edge_id':e['topology_edge_id'],'t':float(x),'curv_abs_m':float(s),'point_sim_m':p.tolist()} for x,s,p in zip(u,arc,P))
 return np.concatenate(allpts),meta

def main():
 t0=time.perf_counter(); c=json.loads(CAP.read_text()); audit=json.loads(AUD.read_text()); s=json.loads(SOL.read_text()); q=np.asarray(s['accepted_q_rigid3d'],float); specs=[]
 for i,e in enumerate(audit['active_elements']):
  e=dict(e); e['curv_abs_m']=c['interpolation']['curvAbsList'][i]; specs.append(e)
 points,meta=curve_independent(q,specs)
 sdf=load_signed_distance_grid(c['sdf_metadata']['path']); tr=np.asarray(c['sdf_metadata']['asset_T_env_sim'],float); off=np.asarray(c['sdf_metadata']['asset_offset_sim'],float); scale=float(c['sdf_metadata']['asset_source_to_sim_scale']); radius=float(c['sdf_metadata']['catheter_radius_m'])
 source=sim_points_to_asset_source(points,tr,off,scale); signed=np.asarray(sdf.sample(source),float); clearance=-signed*scale-radius
 finite=np.isfinite(clearance); valid=np.flatnonzero(finite)
 if len(valid)==0: raise RuntimeError('independent SDF check returned no valid samples')
 idx=int(valid[np.argmin(clearance[finite])]); worst=meta[idx]; minc=float(clearance[idx]);
 depth_counts={f'deeper_than_{mm:.2f}_mm':int(np.count_nonzero(finite & (clearance < -mm/1000.))) for mm in (.01,.05,.1)}
 elapsed=(time.perf_counter()-t0)*1000
 out={'test':'B02 step776 accepted real-beam state independent dense safety check','frame':s['frame'],'action_prefix_sha256':s['action_prefix_sha256'],'source_paths':{'capture':str(CAP),'feasible_solve':str(SOL),'production_sdf':c['sdf_metadata']['path']},'state_source':'accepted_q_rigid3d from offline feasible solve; not native contact state','geometry_path':'Separate 0.01 mm dense BeamAdapter cubic-Bezier reconstruction over all active element interiors, independently implemented from captured rigid nodes/topology/lengths.','sdf_query':'Production B02 vessel_sdf.vti and production source-to-sim transform; clearance=-signed_distance_m-catheter_radius','sample_count':len(clearance),'valid_sample_count':int(finite.sum()),'sample_spacing_mm':.01,'minimum_clearance_m':minc,'minimum_clearance_mm':minc*1000.,'max_penetration_m':max(0.,-minc),'worst_point':{'element_index':worst['element_index'],'edge_id':worst['edge_id'],'t':worst['t'],'curv_abs_m':worst['curv_abs_m'],'point_sim_m':worst['point_sim_m'],'signed_distance_source_mm':float(signed[idx]),'clearance_mm':minc*1000.},'penetration_counts':depth_counts,'finite':bool(finite.all() and np.isfinite(points).all()),'classification_tolerance_m':1e-6,'passed':bool(finite.all() and minc>=-1e-6),'runtime_ms':elapsed,'uses_collision_dofs_as_constraint_source':False,'used_native_post_contact_state_as_solver_input':False,'production_files_modified':False}
 OUT.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 # Complete the solver runtime record with this separate verification phase.
 solver=json.loads(SOL.read_text()); rb=solver['runtime_breakdown_ms']; rb['independent_safety_check_ms']=elapsed; rb['total_with_independent_check']=float(rb['total'])+elapsed; solver['independent_check_path']=str(OUT); solver['independent_check_passed']=out['passed']; SOL.write_text(json.dumps(solver,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'independent_check':'PASS' if out['passed'] else 'FAIL','sample_count':len(clearance),'min_clearance_mm':minc*1000,'penetration_counts':depth_counts,'runtime_ms':elapsed,'output':str(OUT)},allow_nan=False))
if __name__=='__main__': main()
