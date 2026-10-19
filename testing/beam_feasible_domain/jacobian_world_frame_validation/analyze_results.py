"""Finalize actual-result interpretation without additional physics execution."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent;R=HERE/'results'
s=json.loads((R/'summary.json').read_text());report=R/'final_report.md';text=report.read_text().split('\n## Focused states and actual failure reasons',1)[0]
analysis={'historical_deep_failure_root_cause':'UNPROVEN','boundary_cases':[],'substantive_margin_failures':[]}
for cid,c in s['cases'].items():
 for b,d in c['branches'].items():
  for sub in d['substeps']:
   clear=sub['committed_clearance_mm'];deficit=.1-clear
   if deficit>0:
    e=dict(state=cid,branch=b,substep=sub['substep'],clearance_mm=clear,deficit_mm=deficit)
    analysis['boundary_cases' if deficit<1e-10 else 'substantive_margin_failures'].append(e)
text+='\n## Focused states and actual failure reasons\n\n'
for cid in ['worker6_step2033_sub1','worker23_step2009_sub1']:
 c=s['cases'][cid]['branches'];a=c['A2']['substeps'][0];b=c['B2']['substeps'][0]
 t=b['stages'][-1]
 analysis[cid]=dict(original=a['committed_clearance_mm'],corrected=b['committed_clearance_mm'],margin_pass=b['margin_pass'],regression=b['committed_clearance_mm']<a['committed_clearance_mm'],final_stage=t)
 text+=f"{cid}: original A2 {a['committed_clearance_mm']:.17g} -> corrected B2 {b['committed_clearance_mm']:.17g} mm; strict margin pass={b['margin_pass']}. Final {t['row_count']}/{t['active_count']} rows/active; worst dense index {t['dense_worst_index']}, Beam element {t['worst_element']['element_index']}, selected={t['worst_dense_is_selected']}. Minimum linear gap {t['min_linear_row_gap_mm']:.17g} mm vs selected true SDF gap {t['min_selected_true_sdf_gap_mm']:.17g} mm; max prediction error {t['max_abs_nonlinear_sdf_error_mm']:.17g} mm. Native error {t['solver_error']:.17g}, effective tolerance {t['effective_solver_tolerance']:.17g}.\n\n"
text+='B01 step2033 coordinate mismatch is largely removed: the corrected final predicted gap -3.820465e-7 mm agrees with true SDF gap -4.062657e-7 mm (maximum prediction error 2.430709e-8 mm). The remaining strict-margin deficit is about 0.000000406 mm, mainly already present in the linear row residual; the native solver nevertheless met its unchanged scaled stopping threshold. This supports residual permitted by numerical stopping plus a smaller nonlinear term, not persistent frame mismatch, unselected worst point or native propagation failure. No tolerance fix was tested.\n\n'
text+='World angular rows are independently rebuilt and compared against captured native component Data on EVERY nonempty stage. This verifies the correction was actually supplied, not merely applied during report calculations. ActiveCount equals intended row count and native event/stage counters are audited.\n\n'
text+='B02 step2009 has a genuine margin regression, not rounding: the corrected 2nd-pass linear rows are satisfied, but the true selected SDF falls short by about 0.000029 mm. This is remaining finite-motion/nonlinear error, not a failed coordinate conversion or missed native solve. No tolerance, margin, FD step or selection was retuned.\n\n'
text+='Strict numerical boundary cases (exact deficit retained; no rounding to PASS):\n'
for e in analysis['boundary_cases']:text+=f"- {e['state']}/{e['branch']}/sub{e['substep']}: {e['clearance_mm']:.17g} mm, deficit {e['deficit_mm']:.17g} mm\n"
text+='\n## Repeated physical response\n\n'
c=s['cases']['worker9_step1736_sub2']['branches'];moves={}
for b,d in c.items():
 ts=d['substeps'][0]['stages'];moves[b]=[t['max_beam_translation_mm'] for t in ts]
 text+=f"B02 target_01 {b}: per-native-call max translation {moves[b]} mm; final clearance {d['substeps'][0]['committed_clearance_mm']:.17g} mm.\n\n"
analysis['target01_movements']=moves
analysis['target01_repeat_motion_persists']=moves['B2'][1]>0.9*moves['B2'][0]
text+=f"Corrected second/first displacement ratio = {moves['B2'][1]/moves['B2'][0]:.17g}; repeated approximately 0.354 mm motion persists={analysis['target01_repeat_motion_persists']}. It is unchanged by the coordinate fix in this state, not a newly introduced correction anomaly.\n\n"
text+='A larger clearance does not establish a faithful response: inspect whether the second native call repeats a comparable large motion. This test does not alter contacts, integration or the validated relinearization framework to suppress it.\n\n'
a=s['branches']['A2'];b=s['branches']['B2'];ratio=b['mean_physics_substep_ms']/a['mean_physics_substep_ms'];analysis['B2_to_A2_physics_cost_ratio']=ratio
text+=f"B2/A2 target physics wall ratio = {ratio:.9g}x. Maximum corrected B2 per-native-call translation = {b['max_translation_mm']:.17g} mm; free-to-final translation = {b['max_free_to_committed_translation_mm']:.17g} mm; rotation = {b['max_rotation_deg']:.17g} deg.\n\n"
text+='## Final decision and limits\n\n'
text+=f"WORLD-FRAME FIX NUMERICALLY VALID = YES\n\nWORLD-FRAME FIX IMPROVES PHYSICAL SAFETY = INCONCLUSIVE (both routes are nonpenetrating here; strict full-state margins A2 {a['margin_states']}/6 vs B2 {b['margin_states']}/6, including a real B02 regression).\n\nTWO-SOLVE STRATEGY ACCEPTABLE = NO for the required uniform +0.100 mm gate; not production-ready.\n\n"
text+='Prioritize coordinate-consistent world Jacobians as a test-only candidate because the independent derivative is correct. Do NOT accept fixed two-call relinearization as a validated production safety layer yet: remaining nonlinear margin error and repeated large native motion must be understood. Do not increase margin or run further physics in this task.\n\n'
text+='Historical -0.766410 mm root cause = UNPROVEN. The six initial saved free states have positive clearance and do not cover deep unsafe free states or long episodes. Production source hashes remain unchanged, no PPO was launched, and no position/freePosition/CollisionDOF override, projection, rollback, shielding or extra substep was used. Initial missing-plugin startup attempts executed zero action steps and are archived, not counted as physical evidence.\n'
text+=f"\nFINAL = {s['final']}\n"
text+='\n## Independent FD per-state max / mean error (m/rad)\n\n| state | original max | original mean | corrected max | corrected mean | translation error |\n|---|---:|---:|---:|---:|---:|\n'
for cid,r in json.loads((R/'independent_fd.json').read_text()).items():
 text+=f"| {cid} | {r['original_max_error']:.17g} | {r['original_mean_error']:.17g} | {r['corrected_max_error']:.17g} | {r['corrected_mean_error']:.17g} | {r['translation_max_error']:.17g} |\n"
report.write_text(text);(R/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
print(json.dumps({k:v for k,v in analysis.items() if k not in ['worker6_step2033_sub1','worker23_step2009_sub1','boundary_cases','substantive_margin_failures']},indent=2))
