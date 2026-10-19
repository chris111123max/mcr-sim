"""Audit real native captures; synthetic fixtures are never counted as cases."""
import json,hashlib,math
from pathlib import Path
import numpy as np
from checks import audit_substep,linear_gaps,classify
HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
CASES=json.loads((HERE/'cases.json').read_text())

def read(p):return json.loads(p.read_text())
def max_abs(a,b):return float(np.max(np.abs(a-b)))
def audit_branch(case,branch):
    folder=RESULTS/case['id']; p=folder/f'branch_{branch}.json'
    if not p.exists():return {'status':'INCONCLUSIVE','reason':'missing_branch'}
    r=read(p);faults=[];passes={'A':1,'B':2,'C':3}[branch]
    with np.load(case['snapshot'],allow_pickle=False) as z: source={k:np.array(z[k]) for k in ('q_prev','q_free')}
    if not r['state_match']['pass']:faults.append('state_gate')
    if r['passes']!=passes or r['solver_tolerance']!=1e-9:faults.append('branch_settings')
    previous=None
    for s in r['substeps']:
        faults+=audit_substep(s,passes)
        if s['continuity_from_previous'] is not None and not s['continuity_from_previous']['pass']:faults.append('substep_continuity')
        final=folder/f'{branch}_sub{s["substep"]}_final.npz'
        with np.load(final,allow_pickle=False) as z:
            qprev=np.array(z['q_prev']);qfree=np.array(z['q_free']);qfinal=np.array(z['q_committed']);dense=np.array(z['dense_clearance'])
        if not all(np.isfinite(x).all() for x in (qprev,qfree,qfinal,dense)):faults.append('nonfinite_capture')
        if previous is None:
            for name,q in [('q_prev',qprev),('q_free',qfree)]:
                if max_abs(q[:,:3],source[name][:,:3])>1e-10 or max_abs(q[:,3:7],source[name][:,3:7])>1e-10:faults.append('snapshot_numeric_gate')
        elif max_abs(qprev,previous)>1e-10:faults.append('substep2_not_actual_commit')
        if abs(float(np.min(dense)*1000)-s['committed_clearance_mm'])>1e-9:faults.append('final_clearance')
        parent=qfree
        for stage in s['stages']:
            with np.load(folder/stage['npz'],allow_pickle=False) as z: snap={k:np.array(z[k]) for k in z.files}
            if any(not np.isfinite(v).all() for v in snap.values() if np.issubdtype(v.dtype,np.number)):faults.append('nonfinite_stage_array')
            if max_abs(snap['q_linearization_state'],parent)>1e-10:faults.append('stage_parent')
            if len(snap['free_violations'])!=stage['row_count']:faults.append('stage_row_count')
            gaps=linear_gaps(snap['q_linearization_state'],snap['q_committed'],snap)
            if len(gaps):
                actual=snap['dense_clearance'][snap['selected_dense_indices']]-0.0001
                residual=float(np.min(gaps)*1000);nonlinear=float(np.max(np.abs(actual-gaps))*1000)
                if abs(residual-stage['min_linear_row_gap_mm'])>1e-9:faults.append('row_gap_arithmetic')
                if abs(nonlinear-stage['max_abs_nonlinear_sdf_error_mm'])>1e-9:faults.append('nonlinear_arithmetic')
            if stage['solver_error_below_tolerance']!=(stage['solver_error']<1e-9):faults.append('solver_convergence_claim')
            parent=snap['q_committed']
        if max_abs(parent,qfinal)>1e-10:faults.append('stage_vs_final')
        previous=qfinal
    if r['extra_animate_calls']!=0:faults.append('extra_animate')
    for flag in ('production_modified','ppo_started','direct_position_write','rollback','projection','action_shielding'):
        if r[flag]:faults.append(flag)
    complete=len(r['substeps'])==3-case['substep'] and r['target_animate_calls']==3-case['substep']
    nonpen=complete and all(s['nonpenetration_pass'] for s in r['substeps'])
    margin=complete and all(s['margin_pass'] for s in r['substeps'])
    return {'status':'AUDIT_PASS' if not faults else 'AUDIT_FAIL','faults':sorted(set(faults)),
            'complete':complete,'nonpenetration':nonpen,'margin':margin,'data':r}

def main():
    summary={'tested_states':len(CASES),'valid_same_state_replays':0,'cases':{},'branches':{},'native_audit_faults':[]}
    for case in CASES:
        p=RESULTS/case['id']/'case_result.json'
        if not p.exists():summary['cases'][case['id']]={'status':'NOT_RUN'};continue
        result=read(p);item={'result':result,'branches':{}}
        if result.get('status')=='NATIVE_MATRIX_COMPLETE':
            summary['valid_same_state_replays']+=1
            for branch in 'ABC':item['branches'][branch]=audit_branch(case,branch)
            initial=[]
            for branch in 'ABC':
                if 'data' not in item['branches'][branch]:continue
                first=item['branches'][branch]['data']['substeps'][0]['substep']
                with np.load(RESULTS/case['id']/f'{branch}_sub{first}_final.npz',allow_pickle=False) as z:
                    initial.append((np.array(z['q_prev']),np.array(z['q_free'])))
            if len(initial)!=3 or any(not np.array_equal(initial[0][j],value[j]) for value in initial[1:] for j in (0,1)):
                summary['native_audit_faults'].append(case['id']+':ABC_initial_state')
        summary['cases'][case['id']]=item
    valid=[c for c in CASES if summary['cases'].get(c['id'],{}).get('branches')]
    summary['vessels']=sorted({c['case']['vessel'] for c in valid});summary['targets']=sorted({c['case']['target'] for c in valid})
    for branch in 'ABC':
        audits=[summary['cases'][c['id']]['branches'][branch] for c in valid]
        data=[a['data'] for a in audits if 'data'in a]
        subs=[s for d in data for s in d['substeps']];stages=[t for s in subs for t in s['stages']]
        native=[t['native_solver_wall_ms'] for t in stages];rowtime=[t['row_build_wall_ms'] for t in stages if t['row_build_wall_ms'] is not None]
        r={'states_with_data':len(data),'nonpenetration_pass_count':sum(a.get('nonpenetration',False) for a in audits),
           'margin_pass_count':sum(a.get('margin',False) for a in audits),
           'substeps':len(subs),'nonpenetration_substeps':sum(s['nonpenetration_pass'] for s in subs),
           'margin_substeps':sum(s['margin_pass'] for s in subs),
           'worst_committed_clearance_mm':min((s['committed_clearance_mm'] for s in subs),default=None),
           'native_solve_count':len(stages),'mean_native_solver_ms':float(np.mean(native)) if native else None,
           'mean_native_solver_total_per_substep_ms':float(np.mean([s['native_solver_total_wall_ms'] for s in subs])) if subs else None,
           'mean_row_build_ms':float(np.mean(rowtime)) if rowtime else None,
           'mean_physics_substep_ms':float(np.mean([s['physics_substep_wall_ms'] for s in subs])) if subs else None,
           'mean_mapping_refresh_ms':float(np.mean([t['mapping_refresh_wall_ms'] for t in stages if t['mapping_refresh_wall_ms']>0])) if branch!='A' and stages else None,
           'row_count_range':[min((t['row_count'] for t in stages),default=None),max((t['row_count'] for t in stages),default=None)],
           'min_linear_row_gap_mm':min((t['min_linear_row_gap_mm'] for t in stages if t['min_linear_row_gap_mm'] is not None),default=None),
           'max_nonlinear_error_mm':max((t['max_abs_nonlinear_sdf_error_mm'] for t in stages if t['max_abs_nonlinear_sdf_error_mm'] is not None),default=None),
           'solver_raw_error_below_1e9_count':sum(t['solver_error_below_tolerance'] for t in stages),
           'row_active_mismatches':sum(t['row_count']!=t['active_count'] for t in stages)}
        for case,a in zip(valid,audits):
            if a['status']!='AUDIT_PASS':summary['native_audit_faults'].append(case['id']+':'+branch+':'+str(a.get('faults',a.get('reason'))))
        summary['branches'][branch]=r
    summary['historical_deep_failure_root_cause']='UNPROVEN'
    summary['production_modified']=False;summary['ppo_started']=False
    summary['final_result']='INCONCLUSIVE' if len(valid)<2 or summary['native_audit_faults'] else ('PASS' if all(summary['branches'][b]['margin_pass_count']==len(valid) and summary['branches'][b]['solver_raw_error_below_1e9_count']==summary['branches'][b]['native_solve_count'] for b in 'BC') else 'FAIL')
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))
    return summary
if __name__=='__main__': main()
