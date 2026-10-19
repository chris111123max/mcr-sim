"""Read-only physical-result analysis; only writes reports in this directory."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np
HERE=Path(__file__).resolve().parent;R=HERE/'results'
s=json.loads((R/'summary.json').read_text());records=[];movements=[]
for cid,item in s['cases'].items():
 for b,a in item.get('branches',{}).items():
  for sub in a['data']['substeps']:
   with np.load(R/cid/f'{b}_sub{sub["substep"]}_final.npz',allow_pickle=False) as z:
    prev_index=int(np.argmin(z['free_dense_clearance']))
   for t in sub['stages']:
    with np.load(R/cid/t['npz'],allow_pickle=False) as z:
     movement=float(np.max(np.linalg.norm(z['q_committed'][:,:3]-z['q_linearization_state'][:,:3],axis=1))*1000)
    records.append({'state':cid,'branch':b,'substep':sub['substep'],'stage':t['stage'],
       'worst_index':t['dense_worst_index'],'worst_changed':prev_index!=t['dense_worst_index'],
       'worst_selected':t['worst_dense_is_selected'],'max_native_translation_mm':movement,
       'raw_error':t['solver_error'],'effective_tolerance':t['effective_solver_tolerance'],
       'raw_tolerance_pass':t['solver_error_below_tolerance']})
    prev_index=t['dense_worst_index']
    if cid=='worker9_step1736_sub2' and b=='C':movements.append(movement)
fail=s['cases']['worker6_step2033_sub1']['branches']['C']['data']['substeps'][0]
f=fail['stages'][-1]
result={'native_passes':len(records),'native_audit_faults':s['native_audit_faults'],
 'raw_error_below_configured_tolerance_count':sum(t['raw_tolerance_pass'] for t in records),
 'error_below_effective_native_tolerance_count':sum(t['raw_error']<t['effective_tolerance'] for t in records),
 'worst_point_migration_events':sum(t['worst_changed'] for t in records),
 'worst_point_not_selected_events':sum(not t['worst_selected'] for t in records),
 'penetrating_free_substeps':sum(sub['free_clearance_mm']<0 for item in s['cases'].values() for a in item.get('branches',{}).values() for sub in a['data']['substeps']),
 'failure_frame':{'state':'worker6_step2033_sub1','branch':'C','clearance_mm':fail['committed_clearance_mm'],
  'dense_index':f['dense_worst_index'],'beam_element':f['worst_element'],'rows':f['row_count'],'active':f['active_count'],
  'raw_linear_gap_mm':f['min_linear_row_gap_mm'],'world_converted_linear_gap_mm':f['min_linear_row_gap_world_angular_mm'],
  'true_sdf_gap_mm':f['min_selected_true_sdf_gap_mm'],'raw_prediction_error_mm':f['max_abs_nonlinear_sdf_error_mm'],
  'world_converted_prediction_error_mm':f['max_abs_nonlinear_sdf_error_world_angular_mm']},
 'target01_each_pass_max_translation_mm':movements,
 'preferred_future_candidate':'2 solves with dense verification and early stop; first resolve the angular Jacobian coordinate convention in a test-only A/B.',
 'production_ready':False,'historical_deep_failure_root_cause':'UNPROVEN','records':records}
(R/'failure_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
report=R/'final_report.md';text=report.read_text().split('\n## Failure analysis',1)[0]
text+='\n## Failure analysis\n\n'
text+=f"B01 / target_04 / seed48 / step2033 / substep1: A {s['cases']['worker6_step2033_sub1']['branches']['A']['data']['substeps'][0]['committed_clearance_mm']:.12f} mm; B {s['branches']['B']['worst_committed_clearance_mm']:.12f} mm; C {fail['committed_clearance_mm']:.12f} mm. The 0.001508 mm C margin shortfall is real, not floating-point rounding.\n\n"
text+=f"All three C passes keep worst index 4950, Beam element3 / nodes28,29, selected in all five active rows. The last raw linear gap is {f['min_linear_row_gap_mm']:.12g} mm, while actual SDF gap is {f['min_selected_true_sdf_gap_mm']:.12g} mm. With the stored local angular Jacobian rotated into world coordinates, the predicted gap is {f['min_linear_row_gap_world_angular_mm']:.12g} mm and maximum prediction error drops from {f['max_abs_nonlinear_sdf_error_mm']:.12g} to {f['max_abs_nonlinear_sdf_error_world_angular_mm']:.12g} mm. This supports an angular coordinate mismatch in THIS failed margin frame, rather than missing sampling, worst-point switching or appreciable geometric nonlinearity.\n\n"
text+='Read-only source evidence: mcr_sim/beam_sdf_unilateral_fast.py _perturbed_point uses rotation * exp(delta) (local/body perturbation); BeamLinearizedUnilateralConstraint.cpp copies the resulting angular Jacobian directly into Rigid3 Deriv; installed SOFA21.12 RigidTypes.h operator+=(Deriv) applies exp(omega) * orientation (world increment). No conversion is present. This was not repaired during this task.\n\n'
text+=f"Native solver scaled-threshold criterion passes {result['error_below_effective_native_tolerance_count']}/{len(records)}; unscaled currentError<1e-9 passes {result['raw_error_below_configured_tolerance_count']}/{len(records)}. No native convergence failure follows merely from currentError>1e-9 when scaleTolerance=True.\n\n"
text+='B01 step771 failures are floating point only: B reaches 0.09999999999965765 / 0.09999999999970417 mm; C reaches 0.09999999999999994 / 0.10000000000000482 mm. Strict comparisons are preserved, so this state is not counted as a full margin pass.\n\n'
text+=f"Worst point migration events: {result['worst_point_migration_events']}/{len(records)}; new worst point not in current rows: {result['worst_point_not_selected_events']}/{len(records)}. These migrated points remain nonpenetrating in this corpus. Initial and continued free states are all nonpenetrating: {result['penetrating_free_substeps']} penetrating free substeps. Deep free-state penetration is not covered.\n\n"
text+=f"B02 target_01 C passes each move the most displaced Beam translation by {movements} mm. Clearance rises 0.410877 -> 0.734158 -> 1.049684 mm; extra calls are not merely reducing a small residual and can repeatedly move the Beam. This does not establish faithful incremental response or prove the historical deep failure root cause.\n\n"
text+='IS 2-SOLVE STRATEGY SUFFICIENT? = For observed nonpenetration YES; for strict universal 0.100mm margin and unscaled residual NO.\n\nDOES 3RD SOLVE GIVE MEANINGFUL BENEFIT? = More clearance in several states, but no improvement in full-state margin pass count (4/6 for both); does not repair the substantive B01 failure.\n\n'
a=s['branches'];text+=f"C/B target-physics wall cost ratio = {a['C']['mean_physics_substep_ms']/a['B']['mean_physics_substep_ms']:.6f}x; additional mean cost = {a['C']['mean_physics_substep_ms']-a['B']['mean_physics_substep_ms']:.6f} ms/substep. These include diagnostics. Native-only total cost ratio = {a['C']['mean_native_solver_total_per_substep_ms']/a['B']['mean_native_solver_total_per_substep_ms']:.6f}x.\n\n"
text+='Recommendation: prioritize a maximum-two-solve candidate with dense verification/early stopping AFTER a test-only coordinate-consistent Jacobian validation. Do not integrate a fixed three-solve strategy solely on the present margin results. No production integration or training is authorized by these results.\n'
report.write_text(text)
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
