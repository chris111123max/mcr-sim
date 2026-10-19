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
for c,bs in results.items():
    for b,v in bs.items():
        print(c,b,'match',v['state_match']['pass'])
        for s in v['stages']:
            print({k:s.get(k) for k in ['stage','clearance_mm','max_displacement_mm','row_count','global_rows','rhs_min_mm','lambda_max_abs','cleared_native_dx_max_abs','correction_vs_actual_translation_max_abs_mm','contact_rhs_vs_previous_max_abs','contact_lambda_vs_previous_max_abs','contact_rhs_vs_previous_satisfied_residual_max_abs_mm','native_contact_jacobian_vs_previous_max_abs','mapping_free_change_mm','mapped_reference_matches_previous_postmap_max_abs_mm']})
