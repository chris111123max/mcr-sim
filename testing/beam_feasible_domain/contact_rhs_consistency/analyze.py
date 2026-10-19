from pathlib import Path
import json, hashlib
import numpy as np
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent
R=P/'results'
def maxtrans(a,b):return float(np.linalg.norm(np.array(a)[:,:3]-np.array(b)[:,:3],axis=1).max()*1000)
def peak(a):
    a=np.array(a);return float(np.max(np.abs(a))) if a.size else 0.
def mappings(a):return {x['path']:x for x in a['mapped']}
results={}
for c in json.loads((P/'cases.json').read_text()):
    d=R/c['id'];out={}
    for bf in sorted(d.glob('branch_*.json')):
        branch=json.loads(bf.read_text());b=branch['branch'];sub=c['substep']
        records=[json.loads(l) for l in (d/f'{b}_sub{sub}_native.jsonl').read_text().splitlines()]
        stages=[];initial=np.load(d/f'{b}_sub{sub}_stage1.npz')['q_free'];prev=initial.copy();prevnative=None
        for st in branch['substeps'][0]['stages']:
            n=st['stage'];z=np.load(d/st['npz']);rows={k:z[k] for k in z.files}
            audit={x['phase']:x for x in records if x['stage']==n}
            built=audit['built'];sol=audit['solved'];pre=audit['pre_apply'];post=audit['post_apply'];prep=audit['prepared']
            assert np.max(np.abs(np.array(pre['reference_x'])-rows['q_linearization_state']))<1e-12
            assert np.max(np.abs(np.array(post['position'])-rows['q_committed']))<1e-12
            assert np.max(np.abs(np.array(pre['free_position'])-initial))<1e-12
            dofs=len(initial);dim=built['dimension'];J=np.zeros((dim,dofs,6))
            for item in built['jacobian']:J[int(item[0]),int(item[1])]=item[2:]
            beamids=[];jacerr=0.
            for ri in range(len(rows['free_violations'])):
                expect=np.zeros((dofs,6));a,e=rows['row_offsets'][ri:ri+2]
                for k in range(a,e):expect[rows['dof_indices'][k]]+=np.r_[rows['linear_jacobian'][k],rows['angular_jacobian'][k]]
                errors=np.max(np.abs(J-expect),axis=(1,2));idx=int(np.argmin(errors));jacerr=max(jacerr,float(errors[idx]));beamids.append(idx)
            assert jacerr<1e-10,(c['id'],b,n,jacerr)
            rhs=np.array(built['rhs']);lam=np.array(sol['lambda']);dx=np.array(post['applied_dx']);native=np.array(post['native_dx'])
            remaining=rhs+np.einsum('ijk,jk->i',J,dx)
            system_remaining=rhs+np.array(sol['W'])@lam
            contactids=[i for i in range(dim) if i not in beamids]
            dq=np.zeros_like(dx);dq[:,:3]=np.array(post['position'])[:,:3]-prev[:,:3]
            dq[:,3:]=(Rotation.from_quat(np.array(post['position'])[:,3:7])*Rotation.from_quat(prev[:,3:7]).inv()).as_rotvec()
            info=dict(stage=n,clearance_mm=st['committed_clearance_mm'],max_displacement_mm=maxtrans(post['position'],prev),
                max_rotation_deg=float(np.rad2deg(np.linalg.norm(dq[:,3:],axis=1).max())),row_count=st['row_count'],active_count=st['active_count'],
                global_rows=dim,beam_global_row_ids=beamids,native_jacobian_max_abs_error=jacerr,
                rhs=rhs.tolist(),lambda_actual=lam.tolist(),native_contact_row_ids=contactids,
                rhs_min_mm=float(rhs.min()*1000),lambda_max_abs=peak(lam),lambda_before_solve_max_abs=peak(built['lambda']),
                cleared_native_dx_max_abs=peak(prep['native_dx']),applied_native_dx_max_translation_mm=float(np.linalg.norm(dx[:,:3],axis=1).max()*1000),
                correction_vs_actual_translation_max_abs_mm=peak(dx[:,:3]-dq[:,:3])*1000,
                correction_vs_actual_rotation_max_abs_rad=peak(dx[:,3:]-dq[:,3:]),
                native_raw_to_applied_ratio=float(np.linalg.norm(dx)/np.linalg.norm(native)) if np.linalg.norm(native)>0 else None,
                beam_min_linear_gap_mm=st['min_linear_row_gap_mm'],solver_error=st['solver_error'],solver_iterations=st['solver_iterations'],
                effective_tolerance=st['effective_solver_tolerance'],reference_aliases_position=pre['reference_aliases_position'],
                reference_aliases_free=pre['reference_aliases_free'],native_linear_residual=remaining.tolist(),native_system_residual=system_remaining.tolist(),
                min_native_contact_normal_residual_mm=float(remaining[contactids[::3]].min()*1000) if contactids else None,
                min_native_contact_normal_system_residual_mm=float(system_remaining[contactids[::3]].min()*1000) if contactids else None,
                beam_lambda_max_abs=peak(lam[beamids]),contact_lambda_max_abs=peak(lam[contactids]),
                native_correction_npz=str(d/st['npz']),audit_jsonl=str(d/f'{b}_sub{sub}_native.jsonl'))
            if prevnative is not None:
                oldcontact=prevnative['native_contact_row_ids'];oldrhs=np.array(prevnative['rhs'])[oldcontact];newrhs=rhs[contactids]
                info['contact_rhs_vs_previous_max_abs']=peak(newrhs-oldrhs) if newrhs.shape==oldrhs.shape else None
                oldlambda=np.array(prevnative['lambda_actual'])[oldcontact]
                info['contact_lambda_vs_previous_max_abs']=peak(lam[contactids]-oldlambda) if len(oldlambda)==len(contactids) else None
                oldremaining=np.array(prevnative['native_linear_residual'])[oldcontact]
                info['contact_rhs_vs_previous_satisfied_residual_max_abs_mm']=peak(newrhs-oldremaining)*1000 if oldremaining.shape==newrhs.shape else None
                info['native_contact_jacobian_vs_previous_max_abs']=peak(J[contactids]-Jprev[oldcontact]) if len(contactids)==len(oldcontact) else None
                info['increment_vs_previous_correction_max_abs_mm']=peak(dx[:,:3]-dxprev[:,:3])*1000
                oldm=mappings(postprev);newm=mappings(pre)
                info['mapped_free_unchanged_max_abs_mm']=max([peak(np.array(newm[k]['free_position'])-np.array(oldm[k]['free_position']))*1000 for k in oldm] or [0])
                if 'post_map' in prev_audit:
                    info['mapped_reference_matches_previous_postmap_max_abs_mm']=max([peak(np.array(newm[k]['reference_x'])-np.array(mappings(prev_audit['post_map'])[k]['position']))*1000 for k in newm] or [0])
            if 'pre_map' in audit:
                am=mappings(audit['pre_map']);bm=mappings(audit['post_map'])
                info['mapping_position_change_mm']=max([peak(np.array(bm[k]['position'])-np.array(am[k]['position']))*1000 for k in am] or [0])
                info['mapping_free_change_mm']=max([peak(np.array(bm[k]['free_position'])-np.array(am[k]['free_position']))*1000 for k in am] or [0])
            stages.append(info);prev=np.array(post['position']);prevnative=info;Jprev=J;dxprev=dx;postprev=post;prev_audit=audit
        out[b]=dict(state_match=branch['state_match'],extra_animate_calls=branch['extra_animate_calls'],stages=stages,
                    substeps=branch['substeps'],nan_inf=any(s['nan_inf'] for s in branch['substeps']))
    results[c['id']]=out
before=json.loads((R/'production_before.json').read_text());after={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in before}
assert before==after
summary=dict(cases=results,production_unchanged=before==after)
(R/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))

def flatten_geometry(a):return [c for group in a.get('contact_geometry',[]) for c in group]
for case,bs in results.items():
    d=R/case
    for b,v in bs.items():
        bj=json.loads((d/f'branch_{b}.json').read_text());sub=bj['case']['substep']
        rec=[json.loads(l) for l in (d/f'{b}_sub{sub}_native.jsonl').read_text().splitlines()]
        final={x['path']:x for x in bj['substeps'][0]['final_mapped']}
        for st in v['stages']:
            audit={x['phase']:x for x in rec if x['stage']==st['stage']}
            built=audit['built'];geo=flatten_geometry(built);m=mappings(built)
            rhs=np.array(built['rhs']);gap_errors=[];ref_errors=[];mapped_errors=[];jac_errors=[];final_gaps=[]
            for c in geo:
                ri=int(c['row']);normal=np.asarray(c['normal'])
                gap_errors.append(abs(rhs[ri]-float(c['normal_gap'])))
                ref_errors.extend([peak(np.array(c['P'])-c['cparams_P']),peak(np.array(c['Q'])-c['cparams_Q'])])
                for key,point,mi,sgn in [('state1','Q','m1',-1),('state2','P','m2',1)]:
                    path=c[key];idx=c[mi]
                    mapped_errors.append(peak(np.asarray(m[path]['reference_x'])[idx]-c['cparams_'+point]))
                    jm=np.zeros(3)
                    for jr in m[path]['jacobian']:
                        if int(jr[0])==ri and int(jr[1])==idx:jm+=np.asarray(jr[2:])
                    jac_errors.append(peak(jm-sgn*normal))
                fp=np.asarray(final[c['state2']]['position'])[c['m2']]
                fq=np.asarray(final[c['state1']]['position'])[c['m1']]
                final_gaps.append(float(np.dot(fp-fq,normal)-c['distance'])*1000)
            st.update(contact_count=len(geo),contact_geometry_rhs_error_mm=max(gap_errors or [0])*1000,
                contact_used_vs_cparams_reference_error_mm=max(ref_errors or [0])*1000,
                contact_cparams_vs_native_mapped_reference_error_mm=max(mapped_errors or [0])*1000,
                contact_local_normal_jacobian_error=max(jac_errors or [0]),
                geometric_contact_rhs_normal_min_mm=min([c['normal_gap']*1000 for c in geo] or [0]),
                final_mapped_normal_gap_min_mm=min(final_gaps or [0]),
                final_mapped_contact_normal_gaps_mm=final_gaps,
                native_converged=st['solver_error']<st['effective_tolerance'],
                contact_used_selected_reference=all(c['selected_reference'] for c in geo))
        checks=[v['state_match']['pass'],not v['nan_inf'],v['extra_animate_calls']==0]
        checks+= [s['contact_geometry_rhs_error_mm']<1e-9 and
            s['contact_used_vs_cparams_reference_error_mm']<1e-9 and
            s['contact_cparams_vs_native_mapped_reference_error_mm']<1e-9 and
            s['contact_local_normal_jacobian_error']<1e-9 and
            s['native_jacobian_max_abs_error']<1e-9 and
            s['correction_vs_actual_translation_max_abs_mm']<1e-9 and
            s['correction_vs_actual_rotation_max_abs_rad']<1e-9 and
            s['row_count']==s['active_count'] and s['native_converged'] for s in v['stages']]
        v['incremental_reference_consistency_pass']=all(checks)
        v['nonpenetration']=all(x['nonpenetration_pass'] for x in v['substeps'])
        v['strict_margin']=all(x['margin_pass'] for x in v['substeps'])
    if 'A' in bs and 'C' in bs:
        za=np.load(d/f'A_sub{bs["A"]["substeps"][0]["substep"]}_stage1.npz')
        zc=np.load(d/f'C_sub{bs["C"]["substeps"][0]["substep"]}_stage1.npz')
        err=float(np.max(np.abs(za['q_committed']-zc['q_committed'])))
        bs['C']['first_solve_vs_A_max_abs']=err
        bs['C']['first_solve_identical']=err<1e-12
summary['cases']=results
(R/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
for c,bs in results.items():
    for b,v in bs.items():
        print(c,b,'state_match',v['state_match']['pass'],'consistent',v['incremental_reference_consistency_pass'],
              'nonpenetration',v['nonpenetration'],'strict_margin',v['strict_margin'])
        for s in v['stages']:
            print({k:s.get(k) for k in ['stage','clearance_mm','max_displacement_mm','row_count','active_count','global_rows',
             'contact_count','geometric_contact_rhs_normal_min_mm','contact_used_vs_cparams_reference_error_mm',
             'contact_geometry_rhs_error_mm','contact_local_normal_jacobian_error','native_jacobian_max_abs_error',
             'contact_lambda_max_abs','min_native_contact_normal_residual_mm','native_converged',
             'contact_rhs_vs_previous_satisfied_residual_max_abs_mm','mapped_reference_matches_previous_postmap_max_abs_mm']})
main=results.get('worker9_step1736_sub2',{})
if main:
    gate=main.get('C',{}).get('incremental_reference_consistency_pass',False) and main.get('C',{}).get('first_solve_identical',False)
    gate=gate and main.get('C',{}).get('nonpenetration',False)
    if 'B' in main and 'C' in main:
        gate=gate and main['C']['stages'][1]['max_displacement_mm']<main['B']['stages'][1]['max_displacement_mm']*0.1
    else:gate=False
    (R/'main_gate.json').write_text(json.dumps({'pass':gate,'scientific_iterations':1},indent=2)+'\n')
    print('MAIN_CONSISTENCY_GATE',gate)
