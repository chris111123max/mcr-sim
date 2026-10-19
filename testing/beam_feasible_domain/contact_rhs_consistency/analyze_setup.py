from pathlib import Path
HERE=Path(__file__).resolve().parent
old=HERE.parent/'native_repeat_solve_consistency/analyze.py'
text=old.read_text()
text=text[:text.index('for c,bs in results.items():')]
text+='''
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
(R/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\\n')
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
    (R/'main_gate.json').write_text(json.dumps({'pass':gate,'scientific_iterations':1},indent=2)+'\\n')
    print('MAIN_CONSISTENCY_GATE',gate)
'''
(HERE/'analyze.py').write_text(text)
print('ANALYSIS_SCRIPT_WRITTEN')
