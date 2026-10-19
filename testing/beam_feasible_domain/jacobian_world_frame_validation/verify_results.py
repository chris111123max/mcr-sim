"""Read-only independent artifact audit; no new SOFA animate calls."""
from pathlib import Path
import json,sys,hashlib,subprocess
import numpy as np
from scipy.spatial.transform import Rotation
from preflight import SavedAdapter,fast,verify_fd,convert_rows
from checks import audit_substep,linear_gaps
HERE=Path(__file__).resolve().parent;R=HERE/'results'
LABELS={'A1':1,'A2':2,'B1':1,'B2':2}
def read(p):return json.loads(p.read_text())
def unpack(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k]) for k in z.files}
def main():
 summary={'cases':{},'faults':[],'branches':{},'production_modified':False,'training_started':False}
 stats={b:[] for b in LABELS};fdall=[];stages=[]
 for case in read(HERE/'cases.json'):
  cid=case['id'];d=R/cid;meta=read(Path(case['metadata']));snap=unpack(Path(case['snapshot']))
  if not (d/'case_result.json').exists():
   summary['cases'][cid]={'status':'NOT_COMPLETED','branches':{}};continue
  outcome=read(d/'case_result.json');item={'status':outcome['status'],'branches':{}};summary['cases'][cid]=item
  if outcome['status']!='NATIVE_MATRIX_COMPLETE':continue
  adapter=SavedAdapter(meta);firsts={}
  for b,passes in LABELS.items():
   data=read(d/f'branch_{b}.json');item['branches'][b]=data
   assert data['state_match']['pass'] and data['extra_animate_calls']==0
   assert data['target_animate_calls']==3-case['substep']
   assert data['passes']==passes
   if b.startswith('A'):
    oldb='A' if passes==1 else 'B'
    old=read(HERE.parent/'multistate_native_safety/results'/cid/f'branch_{oldb}.json')
    assert len(old['substeps'])==len(data['substeps'])
    for oldsub,newsub in zip(old['substeps'],data['substeps']):
     assert abs(oldsub['committed_clearance_mm']-newsub['committed_clearance_mm'])<1e-10,'ORIGINAL_BASELINE_NOT_REPRODUCED'
   lastq=None
   for sub in data['substeps']:
    faults=audit_substep(sub,passes)
    for fault in faults:summary['faults'].append(f'{cid}/{b}/{sub["substep"]}:{fault}')
    if sub['substep']>case['substep']:assert sub['continuity_from_previous']['pass']
    adapter.meta=dict(meta,geometry_free=sub['geometry'])
    final=unpack(d/f'{b}_sub{sub["substep"]}_final.npz')
    if lastq is not None:assert np.array_equal(final['q_prev'],lastq)
    else:
     assert np.max(np.abs(final['q_prev']-snap['q_prev']))<=1e-10
     assert np.max(np.abs(final['q_free']-snap['q_free']))<=1e-10
    sub['free_to_committed_max_translation_mm']=float(np.max(np.linalg.norm(final['q_committed'][:,:3]-final['q_free'][:,:3],axis=1))*1000)
    sub['previous_to_committed_max_translation_mm']=float(np.max(np.linalg.norm(final['q_committed'][:,:3]-final['q_prev'][:,:3],axis=1))*1000)
    lastq=final['q_committed'];stats[b].append(sub)
    previous=final['q_free']
    for stage in sub['stages']:
     z=unpack(d/stage['npz']);q=z['q_linearization_state'];qnew=z['q_committed']
     assert np.array_equal(q,previous);previous=qnew
     points,profile=fast._profile(adapter,q)
     actual=fast._profile(adapter,qnew)[1]
     assert np.max(np.abs(actual-z['dense_clearance']))<1e-14
     for k,v in z.items():assert np.isfinite(v).all(),(cid,b,k)
     if stage['row_count']:
      raw=fast.build_rows_fast_local_fd(adapter=adapter,q_prev=q,q_free=q,dense_points=points,dense_clearance=profile)
      expected=convert_rows(raw,q) if b.startswith('B') else raw
      for key in ['row_offsets','dof_indices','selected_dense_indices','linear_jacobian','angular_jacobian','free_violations','source_clearances']:
       assert np.max(np.abs(np.asarray(expected[key])-z[key]))<1e-12,(cid,b,stage['stage'],key)
      fd=verify_fd(adapter,q,raw);assert fd['pass'];fdall.append(fd)
      gap=linear_gaps(q,qnew,z)
      actualgap=actual[z['selected_dense_indices']]-1e-4
      assert abs(float(gap.min()*1000)-stage['min_linear_row_gap_mm'])<1e-9
      assert abs(float(np.max(np.abs(actualgap-gap))*1000)-stage['max_abs_nonlinear_sdf_error_mm'])<1e-9
     if sub['substep']==case['substep'] and stage['stage']==1:firsts[b]=z
     stages.append(dict(case=cid,branch=b,substep=sub['substep'],**stage))
   item['branches'][b]['nonpenetration_state_pass']=all(s['nonpenetration_pass'] for s in data['substeps'])
   item['branches'][b]['margin_state_pass']=all(s['margin_pass'] for s in data['substeps'])
  for b,z in firsts.items():
   a=firsts['A1']
   for k in ['q_prev','q_free','row_offsets','dof_indices','selected_dense_indices','linear_jacobian','free_violations']:
    assert np.array_equal(a[k],z[k]),(cid,b,'initial condition',k)
 for b,subs in stats.items():
  ts=[t for t in stages if t['branch']==b]
  valid=[c['branches'][b] for c in summary['cases'].values() if b in c['branches']]
  def mean(key):
   vals=[t[key] for t in ts if t.get(key) is not None];return float(np.mean(vals)) if vals else None
  summary['branches'][b]=dict(states=len(valid),nonpenetration_states=sum(v['nonpenetration_state_pass'] for v in valid),margin_states=sum(v['margin_state_pass'] for v in valid),substeps=len(subs),nonpenetration_substeps=sum(s['nonpenetration_pass'] for s in subs),margin_substeps=sum(s['margin_pass'] for s in subs),
   worst_committed_mm=min([s['committed_clearance_mm'] for s in subs],default=None),
   native_solve_count=len(ts),mean_native_solve_ms=mean('native_solver_wall_ms'),mean_row_build_ms=mean('row_build_wall_ms'),mean_mapping_ms=mean('mapping_refresh_wall_ms'),
   mean_physics_substep_ms=float(np.mean([s['physics_substep_wall_ms'] for s in subs])) if subs else None,
   max_free_to_committed_translation_mm=max([t['free_to_committed_max_translation_mm'] for t in subs],default=None),
   max_translation_mm=max([t['max_beam_translation_mm'] for t in ts],default=None),max_rotation_deg=max([t['max_beam_rotation_deg'] for t in ts],default=None),
   raw_tolerance_passes=sum(t['solver_error']<1e-9 for t in ts),native_scaled_tolerance_passes=sum(t['solver_error']<t['effective_solver_tolerance'] for t in ts),
   max_nonlinear_prediction_error_mm=max([t['max_abs_nonlinear_sdf_error_mm'] for t in ts if t['max_abs_nonlinear_sdf_error_mm'] is not None],default=None))
 before=read(R/'production_before.json')
 after={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in before['source_sha256']}
 diff=subprocess.check_output(['git','diff','--','mcr_sim','training'],text=True)
 summary['production_modified']=bool(after!=before['source_sha256'] or diff!=before['production_diff'])
 summary['valid_states']=sum(c['status']=='NATIVE_MATRIX_COMPLETE' for c in summary['cases'].values())
 summary['native_stages']=stages
 initial=read(R/'independent_fd.json')
 summary['fd_original_max_error']=max(r['original_max_error'] for r in initial.values())
 summary['fd_corrected_max_error']=max(r['corrected_max_error'] for r in initial.values())
 summary['native_stage_fd_corrected_max_error']=max((r['corrected_max_error'] for r in fdall),default=None)
 summary['initial_fd_pass']=all(r['pass'] for r in initial.values())
 summary['translation_unchanged']=all(r['translation_unchanged'] for r in initial.values())
 if summary['valid_states']<6:summary['final']='INCONCLUSIVE'
 elif summary['faults'] or summary['production_modified'] or summary['branches']['B2']['margin_states']<6:summary['final']='FAIL'
 else:summary['final']='PASS'
 (R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','native_stages']},indent=2))
if __name__=='__main__':main()
