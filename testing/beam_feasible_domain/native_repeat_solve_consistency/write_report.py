from pathlib import Path
import json, hashlib
P=Path(__file__).resolve().parent;R=P/'results'
a=json.loads((R/'analysis.json').read_text());cases=a['cases']
main=cases['worker9_step1736_sub2']['P3']['stages']
assert len(main)==3
assert all(v['state_match']['pass'] and not v['nan_inf'] and v['extra_animate_calls']==0 for bs in cases.values() for v in bs.values())
assert {k:len(v) for k,v in cases.items()}=={'worker9_step1736_sub2':3,'worker6_step2033_sub1':2,'worker23_step2009_sub1':2}
for cid,bs in cases.items():
 for branch,v in bs.items():
  passes=int(branch[-1]);assert len(v['substeps'])==(1 if cid.endswith('sub2') else 2)
  for sub in v['substeps']:
   assert sub['error'] is None and sub['native_status']=='PASS_NATIVE_SAME_SUBSTEP_RELINEARIZATION'
   assert sub['native_solve_calls']==passes and len(sub['stages'])==passes
   assert all(s['active_count']==s['row_count'] for s in sub['stages'])
   if sub['continuity_from_previous'] is not None:assert sub['continuity_from_previous']['pass']
confirmed=all(s['contact_rhs_vs_previous_max_abs']<1e-14 and s['contact_rhs_vs_previous_satisfied_residual_max_abs_mm']>0.1 for s in main[1:])
assert all(s['cleared_native_dx_max_abs']<1e-15 and s['correction_vs_actual_translation_max_abs_mm']<1e-9 for s in main)
assert confirmed
lines=['# NATIVE REPEAT SOLVE CONSISTENCY','','PRODUCTION MODIFIED = NO','PPO TRAINING STARTED = NO',
 'REAL FRAMES REPLAYED = 3 saved frames, 7 same-state branches','INITIAL STATE MATCH = PASS (q_prev/q_free translation and quaternion)',
 'EXTRA PHYSICS SUBSTEPS = 0','','## Evidence by frame','','All targets use corrected world angular Jacobians; tolerance remains 1e-9; margin and row selection unchanged.',
 'Native instrumentation calls the original Generic prepare/build/solve/apply functions without changing their behavior.',
 'Full native RHS, lambda, assembled Rigid3 Jacobian, W, correction, q/v/reference vectors and mapped CollisionDOFs are in *_native.jsonl.',
 'Stage/final NPZ and branch JSON hold independent dense checks, linear row gaps, rotations, state match and continuity.','']
for cid,bs in cases.items():
 lines+=['### '+cid,'','| Branch/pass | Dense clearance mm | Increment mm | Beam rows/active | Global rows | RHS min mm | max abs lambda | solver iterations/error |',
 '|---|---:|---:|---:|---:|---:|---:|---|']
 for b,v in bs.items():
  for s in v['stages']:
   lines.append(f"| {b}/{s['stage']} | {s['clearance_mm']:.12f} | {s['max_displacement_mm']:.12f} | {s['row_count']}/{s['active_count']} | {s['global_rows']} | {s['rhs_min_mm']:.12f} | {s['lambda_max_abs']:.12g} | {s['solver_iterations']}/{s['solver_error']:.12g} |")
 lines+=['','State-match details: '+json.dumps(next(iter(bs.values()))['state_match']), '']
 for b,v in bs.items():
  for s in v['stages']:
   keys=['stage','beam_global_row_ids','native_contact_row_ids','cleared_native_dx_max_abs','lambda_before_solve_max_abs',
    'reference_aliases_position','reference_aliases_free','native_raw_to_applied_ratio','native_jacobian_max_abs_error',
    'correction_vs_actual_translation_max_abs_mm','correction_vs_actual_rotation_max_abs_rad','contact_rhs_vs_previous_max_abs',
    'contact_lambda_vs_previous_max_abs','contact_rhs_vs_previous_satisfied_residual_max_abs_mm',
    'native_contact_jacobian_vs_previous_max_abs','increment_vs_previous_correction_max_abs_mm',
    'mapping_position_change_mm','mapping_free_change_mm','mapped_reference_matches_previous_postmap_max_abs_mm',
    'beam_min_linear_gap_mm','effective_tolerance','beam_lambda_max_abs','contact_lambda_max_abs',
    'min_native_contact_normal_residual_mm','min_native_contact_normal_system_residual_mm']
   lines+=['Native details '+b+'/'+str(s['stage'])+':','```json',json.dumps({k:s.get(k) for k in keys},indent=2),'```','']
  for sub in v['substeps'][1:]:lines.append(f"Original remaining substep {b}/{sub['substep']}: clearance {sub['committed_clearance_mm']:.12f} mm; continuity {sub['continuity_from_previous']['pass']}.")
 lines.append('')
lines+=['## Mechanism / hypotheses','',
 'H1 OLD CORRECTION REAPPLIED = REFUTED as a stale dx buffer: prepareStates clears native dx before every solve. The same contact-driven correction is nevertheless recomputed and applied again.',
 'H2 RHS/REFERENCE MISMATCH = SUPPORTED: subsequent ConstraintParams X/V aliases current committed vectors, while SOFA 21.12 UnilateralInteractionConstraint::getPositionViolation explicitly reads contact MechanicalState freePosition. That vector is unchanged; contact RHS is therefore identical, not the previous corrected residual.',
 'H3 LAMBDA/WARMSTART PROBLEM = REFUTED for this contact path: actual built lambda is zero; UnilateralInteractionConstraint constructs its friction resolution with previous-force cache nullptr, and resolution init only loads old forces if that pointer is non-null. Instantaneous force values inside init are UNAVAILABLE, but the audited route has no such warm-start source. Fresh solved lambda is recorded directly.',
 'H4 MAPPING STATE MISMATCH = REFUTED as a failed propagation: native visitor changes mapped position, and next-pass ConstraintParams reference matches the preceding post-map position. It intentionally does not update mapped freePosition. This is the reference inconsistency in H2, not evidence of a broken Beam mapping.',
 'H5 VALID INCREMENTAL CORRECTION = REFUTED for the main frame: contact RHS remains the original violation despite the already corrected contact linear residual; Beam-specific rows are rebuilt around the current state. These two constraint families are solving different reference problems.',
 '', 'DOUBLE APPLICATION CONFIRMED = YES (effective repeated full contact correction, not reuse of an uncleared dx vector)',
 'NATIVE REPEAT SOLVE PHYSICALLY CONSISTENT = NO',
 'HISTORICAL -0.766410MM ROOT CAUSE = UNPROVEN',
 'FINAL = FAIL (physical consistency); native diagnostic execution = PASS','',
 '## Answers / next step','',
 '1. Second solve does rebuild Beam unilateral rows for their remaining SDF error, but the complete native system does not consistently represent the remaining error: native contact rows retain original free-state RHS.',
 '2. No old correction buffer survives prepareStates. Freshly recomputed full native contact correction is re-applied to the already corrected position/velocity.',
 '3. Actual Jacobian rows match intended world rows; RHS/lambda/state evidence is recorded. Contact reference and correction reference are inconsistent in repeats.',
 '4. The approximately 0.354 mm increments in B02 target01 originate from repeatedly solving the original contact RHS and applying that contact response on top of the prior committed state.',
 '5. One Generic solve follows SOFA 21.12 semantics. Calling the entire solver again with updated X/V alone does not convert all native contact constraints to an incremental problem.',
 '6. Do not continue production repeated solves. First implement a test-only genuinely incremental contact/Beam RHS and correction reference lifecycle, preserving current force/compliance/mapping semantics; validate all three snapshots before integration. Merely clearing dx/lambda again or copying positions is not a fix.',
 '7. Double application is established for this orchestration/frame, not a claim that unmodified SOFA single solves are wrong. Historical deep training failure remains unproven.',
 '', '## Source audit','',
 'SOFA source root: /data/home/3220251075/mcr_sim/mcr_env/sofa/src/sofa. All below were read-only.',
 'modules/SofaConstraint/src/SofaConstraint/GenericConstraintSolver.cpp: prepareStates clears m_lambdaId and m_dxId; buildSystem resets/builds/accumulates the Jacobian and reads constraint violation; applyCorrection computes motion from current_cp->f and applies the correction.',
 'modules/SofaConstraint/src/SofaConstraint/UnilateralInteractionConstraint.inl: getPositionViolation reads both contact MechanicalState freePosition directly; getConstraintViolation ignores incoming coordinate arguments for position order; getConstraintResolution constructs friction resolution with previous-force cache nullptr.',
 'modules/SofaConstraint/src/SofaConstraint/UnilateralInteractionConstraint.cpp: friction resolution init only restores previous forces when _prev is non-null.',
 'modules/SofaConstraint/src/SofaConstraint/LinearSolverConstraintCorrection.inl: applyMotionCorrection reads cParams X/V and assigns x=x_reference+dx, v=v_reference+dv.',
 'The test-only wrapper changes stageParams X/V to current position/velocity on passes 2/3 and propagates position/velocity only between calls; it does not rebuild collision detection/contact free-state RHS.',
 '', '## Isolation / audit','',
 'Production/training hashes before and after are identical (production_before.json / analysis.json). No training, q writes, CollisionDOF writes, projection, rollback or extra animate.',
 'An interrupted initial import produced no target physics samples. A probe-selection attempt was archived under instrumentation_wrong_state_archive; it selected topology rest DOFs and is excluded from conclusions. The test-only selector was corrected to the actual DOFs and the same fixed states revalidated.',
 'Internal warm-start initialization force values are UNAVAILABLE; no lambda is inferred from displacement. All lambda values in the tables are actual current_cp->f. The source audit confirms no previous-force cache is attached on this contact route.',
 'Artifacts: analysis.json plus one directory per saved frame. No production integration performed.']
(R/'final_report.md').write_text('\n'.join(lines)+'\n')
(R/'final_summary.json').write_text(json.dumps(dict(final='FAIL',diagnostic_execution='PASS',double_application=True,physically_consistent=False,production_modified=False,ppo_started=False,extra_physics_substeps=0,historical_root_cause='UNPROVEN'),indent=2))
print('REPORT',R/'final_report.md')
