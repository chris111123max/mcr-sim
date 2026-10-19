from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
H=Path(__file__).resolve().parent;R=H/'results'
def peak(x):
 x=np.asarray(x);return float(np.max(np.abs(x))) if x.size else 0.
def mapped(x):return {m['path']:m for m in x['mapped']}
def audit_stage(d,b,sub,st,records):
 z=np.load(d/st['npz']);n=st['stage'];a={x['phase']:x for x in records if x['stage']==n}
 built=a['built'];pre=a['pre_apply'];post=a['post_apply'];sol=a['solved'];q=z['q_linearization_state'];qc=z['q_committed']
 J=np.zeros((built['dimension'],len(q),6))
 for item in built['jacobian']:J[int(item[0]),int(item[1])]+=np.asarray(item[2:])
 beamids=[];je=0.
 for i in range(len(z['free_violations'])):
  expected=np.zeros((len(q),6));lo,hi=z['row_offsets'][i:i+2]
  for j in range(lo,hi):expected[z['dof_indices'][j]]+=np.r_[z['linear_jacobian'][j],z['angular_jacobian'][j]]
  err=np.max(np.abs(J-expected),axis=(1,2));idx=int(np.argmin(err));je=max(je,float(err[idx]));beamids.append(idx)
 rhs=np.asarray(built['rhs']);lam=np.asarray(sol['lambda']);dx=np.asarray(post['applied_dx'])
 dq=np.zeros_like(dx);dq[:,:3]=qc[:,:3]-q[:,:3];dq[:,3:]=(Rotation.from_quat(qc[:,3:])*Rotation.from_quat(q[:,3:]).inv()).as_rotvec()
 groups=[c for g in built.get('contact_geometry',[]) for c in g];m=mapped(built);referr=[];rhserr=[];jerr=[]
 for c in groups:
  row=int(c['row']);rhserr.append(abs(rhs[row]-c['normal_gap']))
  for key,pt,ind,sgn in [('state1','Q','m1',-1),('state2','P','m2',1)]:
   referr.extend([peak(np.asarray(c[pt])-c['cparams_'+pt]),peak(np.asarray(m[c[key]]['reference_x'])[c[ind]]-c['cparams_'+pt])])
   jm=np.zeros(3)
   for jr in m[c[key]]['jacobian']:
    if int(jr[0])==row and int(jr[1])==c[ind]:jm+=np.asarray(jr[2:])
   jerr.append(peak(jm-sgn*np.asarray(c['normal'])))
 out={'native_beam_J_max_error':je,'contact_count':len(groups),'contact_rows':len(groups)*3,'beam_rows':st['row_count'],
  'active_count':st['active_count'],'global_rows':built['dimension'],'rhs':rhs.tolist(),'lambda':lam.tolist(),
  'contact_reference_error_mm':max(referr or [0])*1000,'contact_RHS_error_mm':max(rhserr or [0])*1000,'contact_normal_J_error':max(jerr or [0]),
  'reference_matches_linearization_max_abs':peak(np.asarray(pre['reference_x'])-q),
  'free_state_unchanged_max_abs':peak(np.asarray(pre['free_position'])-z['q_free']),
  'correction_matches_actual_translation_mm':peak(dx[:,:3]-dq[:,:3])*1000,
  'correction_matches_actual_rotation_rad':peak(dx[:,3:]-dq[:,3:]),
  'solver_iterations':st['solver_iterations'],'solver_error':st['solver_error'],'effective_tolerance':st['effective_solver_tolerance'],
  'solver_converged':st['solver_error']<st['effective_solver_tolerance'],'native_status':sub['native_status'],
  'native_solve_calls':sub['native_solve_calls'],'mapping_refresh_count':sub['mapping_refresh_count'],
  'min_linear_Beam_gap_mm':st['min_linear_row_gap_mm'],'nonlinear_SDF_error_mm':st['max_abs_nonlinear_sdf_error_mm'],
  'worst_dense_is_selected':st['worst_dense_is_selected'],'min_selected_true_gap_mm':st['min_selected_true_sdf_gap_mm'],
  'independent_entire_beam':st['entire_beam']}
 out['reference_and_execution_pass']=all([je<1e-9,max(referr or [0])<1e-12,max(rhserr or [0])<1e-12,max(jerr or [0])<1e-9,
  out['reference_matches_linearization_max_abs']<1e-12,out['free_state_unchanged_max_abs']<1e-12,
  out['correction_matches_actual_translation_mm']<1e-9,out['correction_matches_actual_rotation_rad']<1e-9,
  st['active_count']==st['row_count'],out['solver_converged']])
 if n>1:
  prev={x['phase']:x for x in records if x['stage']==n-1};mp=mapped(prev['post_map'])
  out['mapped_reference_previous_postmap_error_mm']=max([peak(np.asarray(x['reference_x'])-np.asarray(mp[x['path']]['position']))*1000 for x in pre['mapped']] or [0])
  out['reference_and_execution_pass'] &= out['mapped_reference_previous_postmap_error_mm']<1e-9
 return out

cases={};totals={b:{'prevented':0,'worst_committed_mm':None,'body_penetration_substeps':0,'valid_replays':0} for b in 'ABC'}
for d in sorted(R.glob('state*_step*_sub*')):
 meta=json.loads((d/'metadata.json').read_text());out={'metadata':meta,'branches':{}}
 for b in 'ABC':
  file=d/f'branch_{b}.json'
  if not file.exists():continue
  branch=json.loads(file.read_text());subs=[]
  for sub in branch['substeps']:
   records=[json.loads(x) for x in (d/f'{b}_sub{sub["substep"]}_native.jsonl').read_text().splitlines()]
   audits=[audit_stage(d,b,sub,st,records) for st in sub['stages']]
   subs.append({'physics_substep':sub['substep'],'free':sub['entire_beam_free'],'committed':sub['entire_beam_committed'],
    'stage_audits':audits,'continuity':sub['continuity_from_previous'],'error':sub['error'],
    'final_stage_matches_committed':sub['final_stage_to_committed']})
  valid=branch['state_match']['pass'] and len(subs)==branch['expected_animate_calls'] and branch['extra_animate_calls']==0
  valid &= all(s['stage_audits'] and not s['error'] and s['final_stage_matches_committed']['pass'] for s in subs)
  physical_pass=valid and all(s['committed']['strict_nonpenetration'] for s in subs)
  native_pass=valid and all(a['reference_and_execution_pass'] for s in subs for a in s['stage_audits'])
  native_pass &= all(len(s['stage_audits'])==branch['passes_per_physics_substep'] for s in subs)
  out['branches'][b]={'same_state_valid':valid,'physical_nonpenetration_pass':physical_pass,'native_execution_pass':native_pass,'substeps':subs}
  t=totals[b];t['valid_replays']+=int(valid);t['prevented']+=int(physical_pass)
  t['body_penetration_substeps']+=sum(s['committed']['body_negative_samples']>0 for s in subs)
  for s in subs:
   cl=s['committed']['clearance_mm'];t['worst_committed_mm']=cl if t['worst_committed_mm'] is None else min(t['worst_committed_mm'],cl)
 cases[d.name]=out
before=json.loads((R/'production_before.json').read_text());unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in before.items())
run=json.loads((R/'run_summary.json').read_text())
unsafe=sum(c['metadata']['free']['clearance_mm']<0 for c in cases.values())
valid=sum(sum(c['branches'].get(b,{}).get('same_state_valid',False) for b in 'ABC') for c in cases.values())
cgate=bool(cases) and all(c['branches'].get('C',{}).get('native_execution_pass',False) for c in cases.values())
physical_failure=any(not c['branches']['C']['physical_nonpenetration_pass'] for c in cases.values() if c['branches'].get('C',{}).get('same_state_valid'))
final='FAIL' if physical_failure else 'PASS' if unsafe>=2 and cgate and unchanged and run['status']=='BOUNDED_CAPTURE_FINISHED' else 'INCONCLUSIVE'
summary={'final':final,'production_unchanged':unchanged,'real_unsafe_states':unsafe,'valid_same_state_replays':valid,
 'tip_free_penetration_cases':sum(c['metadata']['free']['tip_negative_samples']>0 for c in cases.values()),
 'body_free_penetration_cases':sum(c['metadata']['free']['body_negative_samples']>0 for c in cases.values()),
 'branches':totals,'C_native_execution_gate':cgate,'cases':cases,'run':run,'historical_minus_0p766410_root_cause':'UNPROVEN'}
(R/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))
