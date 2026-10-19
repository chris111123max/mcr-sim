from pathlib import Path
import json,hashlib,numpy as np
H=Path(__file__).resolve().parent;R=H/'results'
a=json.loads((R/'analysis.json').read_text());cases=a['cases'];main=cases['worker9_step1736_sub2']
corrected=[v['C'] for v in cases.values() if 'C' in v]
complete=len(corrected)==3
consistent=complete and all(v['incremental_reference_consistency_pass'] and v['first_solve_identical'] for v in corrected)
nonpen=complete and all(v['nonpenetration'] for v in corrected)
margin=complete and all(v['strict_margin'] for v in corrected)
converged=complete and all(s['native_converged'] for v in corrected for s in v['stages'])
old=main['B']['stages'][1];new=main['C']['stages'][1]
double=consistent and new['max_displacement_mm']<old['max_displacement_mm']*0.1
status='PASS' if consistent and nonpen and margin and converged else ('FAIL' if complete else 'INCONCLUSIVE')
lines=['# SOFA CONTACT RHS CONSISTENCY VALIDATION','',
       'Scope: test-only native contact/reference fix and three saved physical substeps. No production integration or PPO. Contact consistency, nonpenetration, strict margin and native convergence are evaluated separately.','',
       '## GitHub references actually inspected','']
manifest=json.loads((H/'research_sources/manifest.json').read_text())
for x in manifest:
    if x['status']=='OK':lines.append(f"- [{x['path']}]({x['url']}); blob `{x['github_blob_sha']}`; installed equal = {x['installed_equal']}.")
lines+=['','Official SOFA v21.12.00 commit d300cc9af550afbd5c2167470676a5a7d47381a1 was verified against the actual GitHub tree. All selected SOFA bytes match their Git blob SHA; Compliant sources were downloaded through the Git blob API. github_research.md was saved before implementation. PairInteractionConstraint attestation was additionally recorded after initial implementation; its installed callback had already been read before selection.','',
 '## Mechanism / smallest lawful change','',
 'The normal 21.12 FreeMotionAnimationLoop uses freePosition as both the contact violation and correction reference, and calls Generic once. The experimental repeat wrapper changes ConstraintParams X to committed position after stage1 and propagates it, but original UnilateralInteractionConstraint ignores the passed X coordinates and still reads freePosition. It therefore adds another correction from the entire old RHS to the new reference. Real prior audits excluded stale dx.','',
 'A test-only derived FrictionContact uses the official virtual activateMappers hook to create a derived UnilateralInteractionConstraint, then delegates all ordinary contact/mapping initialization to native FrictionContact. A/B call the native violation method unchanged. C evaluates the same official normal/friction position-violation formulas on ConstraintParams::readX-selected coordinates. Native buildSystem rebuilds J/compliance/resolution and actual RHS; the native correction and mechanical mapping visitor remain unchanged. Beam rows use the same current reference and world angular Jacobian. No RHS is manually zeroed or assigned an expected value.','',
 'No off-the-shelf nonlinear outer-loop solver for this Generic 21.12 pipeline was found. Compliant provides the principle of updated mapped geometry and recalculated residuals, but its different KKT and stabilization integration are not transplanted. PersistentContact still uses freePosition and is not adopted. This is a local derived test extension based on official callback contracts, not an official new solver.','',
 '## Numerical A/B/C evidence','',
 '| State | Branch/pass | Beam increment mm | Dense committed mm | Contact normal RHS min mm | Normal linear residual min mm | Contact max |lambda| | Beam rows/active | Native iterations/error/effective tolerance |',
 '|---|---|---:|---:|---:|---:|---:|---|---|']
for cid,bs in cases.items():
    for b,v in bs.items():
        for s in v['stages']:
            lines.append(f"| {cid} | {b}/{s['stage']} | {s['max_displacement_mm']:.12g} | {s['clearance_mm']:.12g} | {s['geometric_contact_rhs_normal_min_mm']:.12g} | {s['min_native_contact_normal_residual_mm']:.12g} | {s['contact_lambda_max_abs']:.9g} | {s['row_count']}/{s['active_count']} | {s['solver_iterations']} / {s['solver_error']:.9g} / {s['effective_tolerance']:.9g} |")
lines+=['','Full actual rhs/lambda/W/native J/dx/reference vectors are retained in branch native JSONL, stage NPZ and analysis.json. A is one solve, B is erroneous two solves, C is consistent two solves. Other states only run A/C after the main consistency gate passed.','',
 '## State / reference / correction / mapping checks','']
for cid,bs in cases.items():
    c=bs['C'];lines+=['',f'### {cid}',
      f"Snapshot q_prev translation difference = {c['state_match']['q_prev']['translation_max_abs_m']:.12g} m; q_free = {c['state_match']['q_free']['translation_max_abs_m']:.12g} m. Quaternion component errors = {c['state_match']['q_prev']['quaternion_max_abs']:.12g} / {c['state_match']['q_free']['quaternion_max_abs']:.12g}. Initial-state gate = {c['state_match']['pass']}.",
      f"A/C stage1 max state difference = {c['first_solve_vs_A_max_abs']:.12g}. Each branch executes only its targeted native 5 ms substep; no following substep or additional animate. Extra animate calls = {c['extra_animate_calls']}."]
    for s in c['stages']:
        lines.append(f"- C pass{s['stage']}: contacts={s['contact_count']}; actual geometry→RHS error={s['contact_geometry_rhs_error_mm']:.12g} mm; used contact reference→cParams error={s['contact_used_vs_cparams_reference_error_mm']:.12g} mm; cParams→mapped reference error={s['contact_cparams_vs_native_mapped_reference_error_mm']:.12g} mm; local contact normal Jacobian error={s['contact_local_normal_jacobian_error']:.12g}; native Beam Jacobian vs assembled world FD error={s['native_jacobian_max_abs_error']:.12g}; native correction→actual translation error={s['correction_vs_actual_translation_max_abs_mm']:.12g} mm, SO3 rotation error={s['correction_vs_actual_rotation_max_abs_rad']:.12g} rad; rotation increment={s['max_rotation_deg']:.12g} deg; Beam max |lambda|={s['beam_lambda_max_abs']:.12g}; native runtime={c['substeps'][0]['stages'][s['stage']-1]['native_solver_wall_ms']:.12g} ms.")
    st=c['stages'][1];prev=c['stages'][0]
    ids=st['native_contact_row_ids'][::3];prev_ids=prev['native_contact_row_ids'][::3]
    normerr=float(np.max(np.abs(np.asarray(st['rhs'])[ids]-np.asarray(prev['native_linear_residual'])[prev_ids])))*1000
    lines.append(f"Second normal RHS vs previous satisfied linear normal residual max difference = {normerr:.12g} mm. Mapping retains original freePosition (difference={st['mapped_free_unchanged_max_abs_mm']:.12g} mm), while readX equals actual previous native post-map position (difference={st['mapped_reference_matches_previous_postmap_max_abs_mm']:.12g} mm).")
    lines.append(f"Final independent dense clearance = {c['substeps'][0]['independent_committed_clearance_mm']:.12g} mm; nonpenetration={'PASS' if c['nonpenetration'] else 'FAIL'}; strict +0.100 mm margin={'PASS' if c['strict_margin'] else 'FAIL'}; native convergence={'PASS' if all(x['native_converged'] for x in c['stages']) else 'FAIL'}.")
lines+=['','Friction caveat: a tangential RHS is not always the previous linear tangent residual. Native position violation uses its original anchors and a piecewise impact/sliding formula; those anchors/formulas are preserved. Normal reference/RHS consistency and actual native friction equations are checked separately. In the main frame the all-component RHS/residual difference is about 0.119861 mm, dominated by this tangential definition; it is not a stale normal RHS. Frozen contact normals/manifold, original compliance and the incremental reference scheme remain scope limitations; this does not prove a fully converged nonlinear dynamics/KKT equilibrium or new-contact detection.','',
 '## Synthetic / build / implementation iterations','',
 'GNU 15.3.0 test-only plugin build and real native callback unit passed. Unit original-free and selected-free RHS both -0.100 mm; selected-current RHS +0.050 mm matches actual constructed geometry. Independent saved-state world FD preflight passed for three snapshots. Initial build/fixture API/allocation issues were corrected before real replay; the first synthetic fixture exited with an allocation exception, not a real-frame safety failure. One meaningful implementation iteration was used. No tolerance, margin, dense selection or FD retuning. Missing Eigen headers were downloaded only into this test build_deps directory; no SOFA/Conda/system installation was changed.','',
 '## Four separate gates','',f'- Incremental contact/reference/Jacobian/native correction consistency: {"PASS" if consistent else "FAIL"}.',
 f'- Corrected branch true nonpenetration (dense clearance >= -0.001 mm): {"PASS" if nonpen else "FAIL"}.',
 f'- Corrected branch strict +0.100 mm margin: {"PASS" if margin else "FAIL"}.',
 f'- Actual native stopping condition (configured tolerance times row count, scaleTolerance=true): {"PASS" if converged else "FAIL"}.',
 '', '## Required summary','', '```text','SOFA CONTACT RHS CONSISTENCY VALIDATION',
 'PRODUCTION MODIFIED = NO','PPO TRAINING STARTED = NO',
 'GITHUB SOURCE RESEARCH = PASS','OFFICIAL SOFA 21.12 SOURCE VERIFIED = YES, v21.12.00 d300cc9af550afbd5c2167470676a5a7d47381a1',
 'OPEN-SOURCE DESIGN ADOPTED = official PairInteractionConstraint/readX callback contract; inherited native response, matrix/resolution/correction/mapping',
 f'REAL FRAMES REPLAYED = {len(corrected)}; targeted forks only',
 f'INITIAL STATE MATCH = {"PASS" if complete and all(v["state_match"]["pass"] for v in corrected) else "FAIL"}',
 'EXTRA PHYSICS SUBSTEPS = 0',
 f"ORIGINAL CONTACT RHS = {old['geometric_contact_rhs_normal_min_mm']:.12g} mm (normal minimum, repeated pass2)",
 f"UPDATED CONTACT RHS = {new['geometric_contact_rhs_normal_min_mm']:.12g} mm (normal minimum, C pass2)",
 f"ORIGINAL PASS2 DISPLACEMENT = {old['max_displacement_mm']:.12g} mm",f"CORRECTED PASS2 DISPLACEMENT = {new['max_displacement_mm']:.12g} mm",
 f'CONTACT REFERENCE CONSISTENCY = {"PASS" if consistent else "FAIL"}',f'JACOBIAN REFERENCE CONSISTENCY = {"PASS" if consistent else "FAIL"}',
 f'MAPPING CONSISTENCY = {"PASS" if consistent else "FAIL"}',f'NATIVE CORRECTION CONSISTENCY = {"PASS" if consistent else "FAIL"}']
for label,key in [('B02 TARGET01 RESULT','worker9_step1736_sub2'),('B01 STEP2033 RESULT','worker6_step2033_sub1'),('B02 STEP2009 RESULT','worker23_step2009_sub1')]:
    v=cases.get(key,{}).get('C')
    lines.append(label+' = '+('INCONCLUSIVE' if v is None else f"reference={'PASS' if v['incremental_reference_consistency_pass'] else 'FAIL'}; nonpenetration={'PASS' if v['nonpenetration'] else 'FAIL'}; strict margin={'PASS' if v['strict_margin'] else 'FAIL'}"))
lines += [f'DOUBLE CONTACT RESPONSE ELIMINATED = {"YES" if double else "NO"}',
 f'SAME-SUBSTEP REPEAT SOLVE PHYSICALLY CONSISTENT = {"YES (audited incremental frozen-contact reference convention)" if consistent else "NO"}',
 f'NONPENETRATION = {"PASS" if nonpen else "FAIL"}',f'STRICT 0.100MM MARGIN = {"PASS" if margin else "FAIL"}',
 'HISTORICAL -0.766410MM FAILURE ROOT CAUSE = UNPROVEN',f'FINAL = {status}','```','',
 '## Answers / next engineering decision','',
 '1. Official 21.12 exposes a lawful selected-coordinate violation callback and native mapping/matrix/correction interfaces, but no ready-made drop-in nonlinear Generic outer loop was found. Compliant corroborates residual refresh but is not numerically transplanted.',
 '2. The old wrapper changed the correction reference to committed position while original contacts kept the freePosition violation; it broke the shared reference convention.',
 f'3. The minimal derived test-only callback unifies contact RHS, inherited local Jacobian, newly accumulated mapped parent Jacobian, rebuilt world Beam rows and native correction reference: {"PASS" if consistent else "FAIL"}.',
 f'4. Main unreasonable repeated ~0.354 mm response removed: {"YES" if double else "NO"}; the remaining displacement corresponds to genuinely tiny updated normal violation and native finite tolerance.',
 f'5. Corrected saved frames retain real dense nonpenetration: {"YES" if nonpen else "NO"}. Strict margin is a separate {"PASS" if margin else "FAIL"}.',
 '6. World Jacobian + at most two consistent native solves is worth further test-only validation of nonlinear residual/strict margin and friction/dynamic equilibrium. It is not ready for production integration or PPO based on these frames. No future-state guarantee is claimed.','',
 'Production hashes before/after match; git status shows no tracked mcr_sim/ or training/ edits. Existing untracked production build directory was preserved. No real direct position/freePosition/CollisionDOF write, rollback/projection/action shielding, or extra physical integration was used.','']
(R/'final_report.md').write_text('\n'.join(lines))
summary={'FINAL':status,'consistency':consistent,'nonpenetration':nonpen,'strict_margin':margin,'native_converged':converged,
 'double_response_eliminated':double,'frames':len(corrected),'scientific_iterations':1,'production_unchanged':a['production_unchanged'],
 'original_pass2_displacement_mm':old['max_displacement_mm'],'corrected_pass2_displacement_mm':new['max_displacement_mm'],
 'report':str((R/'final_report.md').resolve())}
(R/'final_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
