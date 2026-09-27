#!/usr/bin/env python3
"""One-frame offline feasible-domain solve from captured B02 step776 free Beam state."""
from __future__ import annotations
import hashlib,json,time,tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from mcr_sim.vessel_assets import load_signed_distance_grid,sim_points_to_asset_source
ROOT=Path(__file__).resolve().parents[3]; RES=ROOT/'testing/beam_feasible_domain/_runtime/results'
CAP=RES/'b02_step776_capture.json'; NPZ=RES/'b02_step776_states.npz'; AUD=RES/'b02_step776_adapter_audit.json'; OUT=RES/'b02_step776_feasible_solve.json'
TOL_M=1e-6; MARGIN_M=1e-4; ACTIVE_MM=.60; SOLVE_SAMPLE_M=.0001; CERT_LIP=2.0; MAX_NONLINEAR=12
STRICT_DENSE_CHECKER=None  # installed only by the explicit current-frame online wrapper
STRICT_TRACE_PATH=None  # optional target-only diagnostic JSONL
STRICT_FLOAT_TOL_M=1e-12  # 0.000000001 mm, only roundoff allowance
TIMERS={k:0.0 for k in ['beam_geometry_evaluation_ms','production_sdf_queries_ms','adaptive_certification_ms','jacobian_finite_differences_ms','optimization_solver_ms','relinearization_ms','line_search_ms']}
def bez(c,t):
 t=np.asarray(t,dtype=float).reshape(-1,1); u=1-t
 return u**3*c[0]+3*u*u*t*c[1]+3*u*t*t*c[2]+t**3*c[3]
def split(c):
 a=(c[0]+c[1])*.5; b=(c[1]+c[2])*.5; d=(c[2]+c[3])*.5; e=(a+b)*.5; f=(b+d)*.5; m=(e+f)*.5
 return np.array([c[0],a,e,m]),np.array([m,f,d,c[3]])
def controls(q,spec):
 n0,n1=spec['nodes']; L=spec['rest_length_m']; a=q[n0]; b=q[n1]
 r0=R.from_quat(a[3:7]); r1=R.from_quat(b[3:7]); p0=a[:3]; p3=b[:3]
 p1=p0+r0.apply([1.,0.,0.])*(L/3); p2=p3+r1.apply([-1.,0.,0.])*(L/3)
 if np.dot(p1-p0,p2-p1)<0 and np.linalg.norm(p3-p0)<.4*L:
  return np.array([p0,p0+(p3-p0)/3,p0+2*(p3-p0)/3,p3]),True
 return np.array([p0,p1,p2,p3]),False
def rot_delta(a,b): return (R.from_quat(a[3:7]).inv()*R.from_quat(b[3:7])).as_rotvec()
def clean(x):
 if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [clean(v) for v in x]
 if isinstance(x,(float,np.floating)) and not np.isfinite(x): return str(x)
 if isinstance(x,np.generic): return x.item()
 return x
def emit_trace(row):
 if STRICT_TRACE_PATH is not None:
  with Path(STRICT_TRACE_PATH).open('a') as f: f.write(json.dumps(clean(row),allow_nan=False)+'\n')
def main():
 if not callable(STRICT_DENSE_CHECKER): raise RuntimeError('strict independent dense checker unavailable')
 audit=json.loads(AUD.read_text()); c=json.loads(CAP.read_text())
 if not audit.get('safe_to_run_solver'): raise RuntimeError('adapter/SDF/free-hazard gate failed; solver not run')
 state=np.load(NPZ); qprev=np.asarray(state['q_prev'],float); qfree=np.asarray(state['q_free'],float)
 specs=audit['active_elements']; nodes=sorted({n for sp in specs for n in sp['nodes']})
 # Nodes 0..27 are fixed by the active controller. Only free-moving nodes in the captured active region are solver variables.
 moved=[]
 for n in nodes:
  td=np.linalg.norm(qfree[n,:3]-qprev[n,:3]); rd=np.linalg.norm(rot_delta(qprev[n],qfree[n]))
  if td>1e-11 or rd>1e-10: moved.append(n)
 if not moved: raise RuntimeError('free state has no moving Beam DOFs')
 char_mm={n:float(np.mean([sp['rest_length_m']*1000 for sp in specs if n in sp['nodes']])) for n in moved}
 # Decision vector is a right-composed SO(3) tangent increment, scaled to equivalent local beam length in mm.
 def encode(q):
  y=[]
  for n in moved:
   y.extend(((q[n,:3]-qprev[n,:3])*1000).tolist())
   y.extend((rot_delta(qprev[n],q[n])*char_mm[n]).tolist())
  return np.asarray(y,float)
 yfree=encode(qfree); dim=len(yfree)
 def state_from(y):
  q=qprev.copy(); y=np.asarray(y).reshape(len(moved),6)
  for i,n in enumerate(moved):
   q[n,:3]=qprev[n,:3]+y[i,:3]/1000.
   q[n,3:7]=(R.from_quat(qprev[n,3:7])*R.from_rotvec(y[i,3:6]/char_mm[n])).as_quat()
  return q
 transform=np.asarray(c['sdf_metadata']['asset_T_env_sim'],float); offset=np.asarray(c['sdf_metadata']['asset_offset_sim'],float); scale=float(c['sdf_metadata']['asset_source_to_sim_scale']); radius=float(c['sdf_metadata']['catheter_radius_m']); sdf=load_signed_distance_grid(c['sdf_metadata']['path'])
 def sdf_gap(points):
  t0=time.perf_counter(); src=sim_points_to_asset_source(points,transform,offset,scale); signed=np.asarray(sdf.sample(src),float); gaps=-signed*scale-radius; TIMERS['production_sdf_queries_ms']+=(time.perf_counter()-t0)*1000
  return gaps
 def labels_for(step):
  labels=[]
  for ei,sp in enumerate(specs):
   n=max(2,int(np.ceil(sp['rest_length_m']/step))+1)
   for t in np.linspace(0,1,n): labels.append((ei,float(t)))
  return labels
 solve_labels=labels_for(SOLVE_SAMPLE_M)
 def eval_labels(y,labels):
  t0=time.perf_counter(); q=state_from(y); byedge={i:controls(q,sp)[0] for i,sp in enumerate(specs)}
  pts=np.asarray([bez(byedge[i],[t])[0] for i,t in labels],float); TIMERS['beam_geometry_evaluation_ms']+=(time.perf_counter()-t0)*1000
  return sdf_gap(pts),pts
 def adaptive_cert(y):
  tm=time.perf_counter(); q=state_from(y); rootc={i:controls(q,sp)[0] for i,sp in enumerate(specs)}; TIMERS['beam_geometry_evaluation_ms']+=(time.perf_counter()-tm)*1000
  # Broad sampled pass supplies the independent worst point for the candidate; certification recursively tightens only non-proven intervals.
  gaps,pts=eval_labels(y,labels_for(.0001)); min_sample=float(np.min(gaps)); worst=int(np.argmin(gaps)); worst_label=labels_for(.0001)[worst]
  cert={'subdivisions':0,'certified_segments':0,'unresolved_safe_segments':0,'unresolved_unsafe_segments':0,'max_depth':0,'min_lower_bound_m':float('inf'),'sample_count':len(gaps),'max_base_sample_step_mm':.1}
  refine=set()
  def recurse(ei,ctrl,sp,t0,t1,depth):
   cert['max_depth']=max(cert['max_depth'],depth)
   u=np.array([0.,.5,1.]); pp=bez(ctrl,u); gg=sdf_gap(pp)
   for tt,gv in zip((t0,(t0+t1)*.5,t1),gg):
    if gv<ACTIVE_MM/1000.: refine.add((ei,round(float(tt),12)))
   arc_upper=float(np.sum(np.linalg.norm(np.diff(ctrl,axis=0),axis=1)))
   lower=float(np.min(gg)-CERT_LIP*arc_upper/4)
   if lower>=-TOL_M:
    cert['min_lower_bound_m']=min(cert['min_lower_bound_m'],lower); cert['certified_segments']+=1; return
   if np.max(gg)<-TOL_M:
    cert['min_lower_bound_m']=min(cert['min_lower_bound_m'],lower); cert['unresolved_unsafe_segments']+=1; return
   if depth>=18 or arc_upper<=2e-6:
    cert['min_lower_bound_m']=min(cert['min_lower_bound_m'],lower)
    if np.min(gg)<-TOL_M: cert['unresolved_unsafe_segments']+=1
    else: cert['unresolved_safe_segments']+=1
    return
   left,right=split(ctrl); tm=(t0+t1)*.5; cert['subdivisions']+=1
   recurse(ei,left,sp,t0,tm,depth+1); recurse(ei,right,sp,tm,t1,depth+1)
  tcert=time.perf_counter()
  for i,sp in enumerate(specs): recurse(i,rootc[i],sp,0.,1.,0)
  TIMERS['adaptive_certification_ms']+=(time.perf_counter()-tcert)*1000
  cert['active_refinement_count']=len(refine)
  dense_min=float(STRICT_DENSE_CHECKER(q))
  return {'dense_min_clearance_m':dense_min,'min_sampled_clearance_m':min_sample,'worst_sample_index':worst,'worst_element_index':int(worst_label[0]),'worst_t':float(worst_label[1]),'sample_count':len(gaps),'finite':bool(np.isfinite(gaps).all()),'certification':cert,'active_refinement_labels':sorted(refine),'feasible':bool(np.isfinite(gaps).all() and np.isfinite(dense_min) and min_sample+STRICT_FLOAT_TOL_M>=MARGIN_M and dense_min+STRICT_FLOAT_TOL_M>=MARGIN_M and cert['unresolved_unsafe_segments']==0)}
 # Current/free clearance and active set; starts exactly at q_free (not q_native).
 free_cert=adaptive_cert(yfree); z=yfree.copy(); trace=[]; line_trace=[]; alpha_min=1.0; status='FAIL'; solver_errors=[]; best_dense_safe=None
 emit_trace({'event':'start','requested_margin_m':MARGIN_M,'q_prev_sha256':c['state_sha256']['q_prev'],'q_free_sha256':c['state_sha256']['q_free'],'free_internal_clearance_m':free_cert['min_sampled_clearance_m'],'free_dense_clearance_m':free_cert['dense_min_clearance_m']})
 start=time.perf_counter(); total_phase_start=start
 for it in range(MAX_NONLINEAR):
  tr0=time.perf_counter(); g,_=eval_labels(z,solve_labels); current_min=float(np.min(g)); worst=int(np.argmin(g)); active=np.flatnonzero(g<ACTIVE_MM/1000.)
  current_dense=adaptive_cert(z)
  label_set={tuple(x) for x in [solve_labels[int(i)] for i in active]}
  label_set.update(tuple(x) for x in current_dense['active_refinement_labels'])
  if not label_set: label_set.add(tuple(solve_labels[worst]))
  active_labels=sorted(label_set); ga,_=eval_labels(z,active_labels)
  jt=time.perf_counter(); J=np.zeros((len(active_labels),dim)); h=.001
  for j in range(dim):
   yp=z.copy(); ym=z.copy(); yp[j]+=h; ym[j]-=h
   gp,_=eval_labels(yp,active_labels); gm,_=eval_labels(ym,active_labels); J[:,j]=(gp-gm)/(2*h)
  TIMERS['jacobian_finite_differences_ms']+=(time.perf_counter()-jt)*1000
  # Work in millimetres; clearance constraint gets only a 0.002 mm numerical interior margin (2e-6 m).
  margin_m=MARGIN_M # gap values use metres; decision variables use millimetres
  affine=ga-J@z
  def obj(v): d=v-yfree; return .5*float(d@d)
  def objjac(v): return v-yfree
  def con(v): return affine+J@v-margin_m
  def conjac(v): return J
  opt0=time.perf_counter(); opt=minimize(obj,z,jac=objjac,method='SLSQP',constraints=[{'type':'ineq','fun':con,'jac':conjac}],options={'ftol':1e-12,'maxiter':250,'disp':False}); TIMERS['optimization_solver_ms']+=(time.perf_counter()-opt0)*1000
  if not opt.success: solver_errors.append(str(opt.message))
  emit_trace({'event':'outer_iteration','iteration':it+1,'requested_margin_m':MARGIN_M,'current_internal_clearance_m':current_dense['min_sampled_clearance_m'],'current_dense_clearance_m':current_dense['dense_min_clearance_m'],'current_objective':obj(z),'current_gradient_norm':float(np.linalg.norm(objjac(z))),'direction_norm':float(np.linalg.norm(opt.x-z)),'opt_success':bool(opt.success),'opt_message':str(opt.message),'opt_iterations':int(getattr(opt,'nit',0)),'active_constraints':len(active_labels),'opt_constraint_violation':float(max(0.,-np.min(con(opt.x)))),'best_dense_safe':best_dense_safe})
  lin_res=float(max(0.,-float(np.min(con(opt.x))))) if len(active) else 0.
  ls0=time.perf_counter(); candidates=[]; chosen=None
  alpha=1.0
  for li in range(18):
   cand=z+alpha*(opt.x-z); cert=adaptive_cert(cand)
   entry={'iteration':it+1,'candidate_index':li+1,'alpha':float(alpha),'candidate_sampled_clearance_m':cert['min_sampled_clearance_m'],'candidate_lower_bound_m':cert['certification']['min_lower_bound_m'],'candidate_feasible':cert['feasible'],'accepted':False,'unresolved_unsafe_segments':cert['certification']['unresolved_unsafe_segments']}
   candidates.append(entry)
   dense_safe=bool(np.isfinite(cert['dense_min_clearance_m']) and cert['dense_min_clearance_m']+STRICT_FLOAT_TOL_M>=MARGIN_M)
   correction=np.asarray(cand-yfree).reshape(len(moved),6)
   trans=np.linalg.norm(correction[:,:3],axis=1)
   rotations=np.asarray([np.linalg.norm(correction[i,3:6])/char_mm[n]*180/np.pi for i,n in enumerate(moved)])
   trial={'event':'line_search_trial','iteration':it+1,'trial':li+1,'alpha':float(alpha),'requested_margin_m':MARGIN_M,'objective':obj(cand),'merit':None,'constraint_violation':float(max(0.,-np.min(con(cand)))),'internal_clearance_m':cert['min_sampled_clearance_m'],'dense_clearance_m':cert['dense_min_clearance_m'],'dense_margin_deficit_m':float(MARGIN_M-cert['dense_min_clearance_m']),'max_translation_correction_mm':float(np.max(trans)),'rms_translation_correction_mm':float(np.sqrt(np.mean(trans**2))),'max_rotation_correction_deg':float(np.max(rotations)),'rms_rotation_correction_deg':float(np.sqrt(np.mean(rotations**2))),'gradient_norm':float(np.linalg.norm(objjac(cand))),'direction_norm':float(np.linalg.norm(opt.x-z)),'step_norm':float(np.linalg.norm(cand-z)),'dense_safe':dense_safe,'full_cert_feasible':bool(cert['feasible']),'unresolved_unsafe_segments':cert['certification']['unresolved_unsafe_segments']}
   if dense_safe and (best_dense_safe is None or cert['dense_min_clearance_m']>best_dense_safe['dense_clearance_m']):
    best_dense_safe={'iteration':it+1,'trial':li+1,'dense_clearance_m':cert['dense_min_clearance_m'],'internal_clearance_m':cert['min_sampled_clearance_m'],'full_cert_feasible':bool(cert['feasible'])}
   if cert['feasible']: trial['decision']='accept_strict_feasible'
   elif cert['dense_min_clearance_m']>current_dense['dense_min_clearance_m']+1e-12: trial['decision']='progress_or_superseded'
   else: trial['decision']='reject_no_dense_progress'
   trial['best_dense_safe']=best_dense_safe
   emit_trace(trial)
   if cert['feasible']:
    chosen=(cand,cert,alpha,'feasible'); entry['accepted']=True; break
   if cert['dense_min_clearance_m']>current_dense['dense_min_clearance_m']+1e-12:
    if chosen is None or cert['dense_min_clearance_m']>chosen[1]['dense_min_clearance_m']:
     chosen=(cand,cert,alpha,'progress')
   alpha*=.5
  if chosen is None:
   # If the QP did not find a feasible/improving direction, stop as a genuine stagnation, not a rollback.
   line_trace.extend(candidates); TIMERS['line_search_ms']+=(time.perf_counter()-ls0)*1000
   trace.append({'iteration':it+1,'active_constraints':int(len(active_labels)),'minimum_clearance_before_m':current_min,'worst_sample_index':worst,'worst_element_index':int(solve_labels[worst][0]),'worst_t':float(solve_labels[worst][1]),'jacobian_rows':int(J.shape[0]),'adaptive_refinement_constraints':int(current_dense['certification']['active_refinement_count']),'solver_success':bool(opt.success),'solver_message':str(opt.message),'solver_iterations':int(getattr(opt,'nit',0)),'linearized_residual_m':lin_res,'line_search_alpha':0.,'candidate_minimum_clearance_m':None,'accepted_minimum_clearance_m':current_min,'line_search_status':'stagnation'})
   status='FAIL: FEASIBLE LINE SEARCH STAGNATION'; emit_trace({'event':'termination','reason':status,'iteration':it+1,'current_internal_clearance_m':current_dense['min_sampled_clearance_m'],'current_dense_clearance_m':current_dense['dense_min_clearance_m'],'best_dense_safe':best_dense_safe}); break
  z_new,accepted_cert,alpha,kind=chosen; line_trace.extend(candidates); TIMERS['line_search_ms']+=(time.perf_counter()-ls0)*1000
  alpha_min=min(alpha_min,float(alpha)); z=z_new
  trace.append({'iteration':it+1,'active_constraints':int(len(active_labels)),'minimum_clearance_before_m':current_min,'worst_sample_index':worst,'worst_element_index':int(solve_labels[worst][0]),'worst_t':float(solve_labels[worst][1]),'jacobian_rows':int(J.shape[0]),'adaptive_refinement_constraints':int(current_dense['certification']['active_refinement_count']),'solver_success':bool(opt.success),'solver_message':str(opt.message),'solver_iterations':int(getattr(opt,'nit',0)),'linearized_residual_m':lin_res,'line_search_alpha':float(alpha),'candidate_minimum_clearance_m':accepted_cert['min_sampled_clearance_m'],'accepted_minimum_clearance_m':accepted_cert['min_sampled_clearance_m'],'candidate_feasible':accepted_cert['feasible'],'line_search_status':kind,'adaptive_certification':accepted_cert['certification']})
  TIMERS['relinearization_ms']+=(time.perf_counter()-tr0)*1000
  if accepted_cert['feasible']:
   status='PASS'; break
  if it==MAX_NONLINEAR-1: status='FAIL: nonlinear iteration limit'
 if status=='FAIL' and not trace: status='FAIL: no nonlinear iteration'
 qacc=state_from(z); acc_cert=adaptive_cert(z)
 if status=='PASS' and not acc_cert['feasible']: status='FAIL: strict requested margin not certified'
 emit_trace({'event':'final','status':status,'iterations':len(trace),'requested_margin_m':MARGIN_M,'accepted_internal_clearance_m':acc_cert['min_sampled_clearance_m'],'accepted_dense_clearance_m':acc_cert['dense_min_clearance_m'],'best_dense_safe':best_dense_safe})
 # Independent-style dense constraint evaluation is performed again by a separate script.
 delta_nodes=[]; rot_corr=[]
 for n in moved:
  dt=np.linalg.norm(qacc[n,:3]-qfree[n,:3])*1000.; da=np.linalg.norm((R.from_quat(qfree[n,3:7]).inv()*R.from_quat(qacc[n,3:7])).as_rotvec())
  delta_nodes.append({'node':int(n),'translation_correction_mm':float(dt),'rotation_correction_deg':float(np.rad2deg(da))}); rot_corr.append(float(np.rad2deg(da)))
 trans=np.asarray([x['translation_correction_mm'] for x in delta_nodes]); elapsed=(time.perf_counter()-total_phase_start)*1000
 out={'test':'B02 step776 substep1 offline real-beam feasible-domain solve','frame':{'model':'B02','target':'target_04','seed':15204,'rl_step':776,'physics_substep':1,'dt_s':.005},'action_prefix_sha256':c['action_sha256'],'action_prefix_matches':c['action_prefix_matches'],'state_source':{'q_prev_sha256':c['state_sha256']['q_prev'],'q_free_sha256':c['state_sha256']['q_free'],'used_free_position':True,'used_native_post_contact_state_as_solver_input':False,'native_state_loaded_by_solver':False},'geometry_source':'Actual Rigid3d beam DOFs + captured BeamAdapter active edgeList/lengthList/curvAbsList + BeamInterpolation cubic Bezier; exact compressed linear fallback; identity local offsets from rotationInstrument=0.','constraint_source':'Production B02 vessel_sdf.vti queried directly at real BeamAdapter curve samples; CollisionDOFs are validation-only and not used by constraints.','rotation_parameterization':'SO(3) tangent increments: relative quaternion q_prev^-1*q_free -> rotvec; candidate R=R_prev*Exp(delta_rotvec); optimization rotation variables scaled by local mean active Beam rest length in mm. Quaternion coefficients are never subtracted.','catheter_radius_m':radius,'sdf_sign_units':'source signed_distance_mm converted by production source_to_sim scale; clearance=-signed_distance_m-radius','free_solver_min_clearance_m':free_cert['min_sampled_clearance_m'],'free_solver_min_clearance_mm':free_cert['min_sampled_clearance_m']*1000,'certification_kind':'adaptive sampled certification; subdivision Lipschitz factor is a heuristic, not a mathematical proof','free_adaptive_certification':free_cert['certification'],'requested_margin_m':MARGIN_M,'accepted_independent_dense_min_clearance_m':acc_cert['dense_min_clearance_m'],'accepted_solver_min_clearance_m':acc_cert['min_sampled_clearance_m'],'accepted_solver_min_clearance_mm':acc_cert['min_sampled_clearance_m']*1000,'accepted_worst_location':{'element_index':acc_cert['worst_element_index'],'t':acc_cert['worst_t'],'sample_index':acc_cert['worst_sample_index']},'accepted_adaptive_certification':acc_cert['certification'],'nonlinear_iterations':len(trace),'active_constraints_per_iteration':[x['active_constraints'] for x in trace],'jacobian_rows_per_iteration':[x['jacobian_rows'] for x in trace],'trace':trace,'line_search_trace':line_trace,'minimum_alpha':float(alpha_min),'solver_residual_max_m':max([x['linearized_residual_m'] for x in trace],default=0.),'solver_errors':solver_errors,'solver_status':status,'moved_beam_nodes':moved,'max_translation_correction_mm':float(np.max(trans)) if len(trans) else 0.,'rms_translation_correction_mm':float(np.sqrt(np.mean(trans**2))) if len(trans) else 0.,'tip_translation_correction_mm':float(delta_nodes[-1]['translation_correction_mm']) if delta_nodes else 0.,'max_rotation_correction_deg':float(max(rot_corr,default=0.)),'rms_rotation_correction_deg':float(np.sqrt(np.mean(np.asarray(rot_corr)**2))) if rot_corr else 0.,'tip_rotation_correction_deg':float(rot_corr[-1]) if rot_corr else 0.,'per_node_corrections':delta_nodes,'accepted_q_rigid3d':qacc.tolist(),'finite':bool(np.isfinite(qacc).all() and np.isfinite(z).all() and np.isfinite(acc_cert['min_sampled_clearance_m'])),'nan_count':int(np.isnan(qacc).sum()),'inf_count':int(np.isinf(qacc).sum()),'runtime_breakdown_ms':{**TIMERS,'total':elapsed,'independent_safety_check_ms':None,'timers_note':'geometry/SDF/adaptive phase timers are inclusive within Jacobian and line-search wall time; do not sum inclusive categories as disjoint'},'production_files_modified':False,'training_started':False,'multi_step_replay_started':False}
 OUT.write_text(json.dumps(clean(out),indent=2,allow_nan=False)+'\n'); print(json.dumps(clean({'solver_status':status,'free_min_clearance_mm':out['free_solver_min_clearance_mm'],'accepted_min_clearance_mm':out['accepted_solver_min_clearance_mm'],'iterations':len(trace),'active_counts':out['active_constraints_per_iteration'],'minimum_alpha':alpha_min,'max_translation_correction_mm':out['max_translation_correction_mm'],'max_rotation_correction_deg':out['max_rotation_correction_deg'],'runtime_ms':elapsed,'output':str(OUT)}),allow_nan=False))
def solve_feasible_state(q_prev,q_free,adapter,context,requested_margin_m=0.0001):
 # Thin online wrapper around the exact previously-PASSed main() solver.
 # It supplies this CollisionBeginEvent's Beam states and current live
 # BeamAdapter active interpolation without changing the optimization logic.
 requested_margin_m=float(requested_margin_m)
 if not np.isfinite(requested_margin_m) or requested_margin_m<=0: raise ValueError('invalid requested margin')
 if not callable(getattr(adapter,'measure',None)): raise RuntimeError('independent dense checker unavailable')
 qprev=np.asarray(q_prev,dtype=np.float64); qfree=np.asarray(q_free,dtype=np.float64)
 if qprev.shape!=qfree.shape or qprev.ndim!=2 or qprev.shape[1]!=7:
  raise ValueError(f'invalid current Beam state shapes: {qprev.shape} / {qfree.shape}')
 impl=getattr(adapter,'adapter',adapter)
 geometry_fn=getattr(impl,'solver_geometry',None)
 if not callable(geometry_fn):
  raise RuntimeError('live production BeamAdapter solver_geometry unavailable')
 geometry=geometry_fn()
 capture=json.loads(CAP.read_text()); audit=json.loads(AUD.read_text())
 if not audit.get('safe_to_run_solver'):
  raise RuntimeError('previous PASS audit did not authorize validated solver')
 capture['interpolation']=geometry['interpolation']
 capture['topology']=geometry['topology']
 capture['inserted_length_m']=float(geometry['inserted_length_m'])
 capture['physics_substep']=int(context['substep'])
 capture['state_sha256']={
  'q_prev':hashlib.sha256(np.ascontiguousarray(qprev).tobytes()).hexdigest(),
  'q_free':hashlib.sha256(np.ascontiguousarray(qfree).tobytes()).hexdigest()}
 audit['active_elements']=geometry['active_elements']
 with tempfile.TemporaryDirectory(prefix='b02_online_feasible_',dir=str(RES)) as tmp:
  temp=Path(tmp); cap_path=temp/'capture.json'; audit_path=temp/'audit.json'
  npz_path=temp/'states.npz'; out_path=temp/'solve.json'
  cap_path.write_text(json.dumps(capture,allow_nan=False))
  audit_path.write_text(json.dumps(audit,allow_nan=False))
  np.savez_compressed(npz_path,q_prev=qprev,q_free=qfree)
  original={name:globals()[name] for name in ('CAP','AUD','NPZ','OUT','TIMERS','MARGIN_M','STRICT_DENSE_CHECKER','STRICT_TRACE_PATH')}
  try:
   globals()['CAP']=cap_path; globals()['AUD']=audit_path
   globals()['NPZ']=npz_path; globals()['OUT']=out_path
   globals()['TIMERS']={key:0.0 for key in original['TIMERS']}
   globals()['MARGIN_M']=requested_margin_m
   globals()['STRICT_DENSE_CHECKER']=lambda q: float(adapter.measure(q,spacing_m=0.00001)['min_clearance_m'])
   globals()['STRICT_TRACE_PATH']=context.get('solver_trace_path')
   if globals()['STRICT_TRACE_PATH'] is not None and Path(globals()['STRICT_TRACE_PATH']).exists(): raise RuntimeError('refuse overwrite solver trace')
   main()
   result=json.loads(out_path.read_text())
  finally:
   for name,value in original.items(): globals()[name]=value
 if result.get('solver_status')!='PASS':
  raise RuntimeError(f"strict margin solver returned {result.get('solver_status')}")
 accepted=np.asarray(result.get('accepted_q_rigid3d'),dtype=np.float64)
 if accepted.shape!=qfree.shape or not np.isfinite(accepted).all():
  raise RuntimeError(f'strict solver returned invalid candidate {accepted.shape}')
 independent_dense=float(adapter.measure(accepted,spacing_m=0.00001)['min_clearance_m'])
 if not np.isfinite(independent_dense) or independent_dense+STRICT_FLOAT_TOL_M<requested_margin_m:
  raise RuntimeError(f'strict dense margin not achieved: {independent_dense:.12g} < {requested_margin_m:.12g} m')
 result['accepted_independent_dense_min_clearance_m']=independent_dense
 result['requested_margin_m']=requested_margin_m
 result['accepted']=accepted
 result['solver_source']='strict_b02_feasible_solve.main (validated B02 solver with strict dense margin termination)'
 result['solver_frame']={'step':int(context['step']),'substep':int(context['substep']),
                         'dt_s':float(context['dt'])}
 result['active_edge_ids']=list(geometry['interpolation']['edgeList'])
 result['live_inserted_length_m']=float(geometry['inserted_length_m'])
 result['uses_collision_dofs_as_constraint_source']=False
 result['used_native_post_contact_state_as_solver_input']=False
 result['rollback_used']=False
 result['projection_used']=False
 return result


if __name__=='__main__':main()
