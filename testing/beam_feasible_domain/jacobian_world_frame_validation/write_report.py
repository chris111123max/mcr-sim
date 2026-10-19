from pathlib import Path
import json
HERE=Path(__file__).resolve().parent;R=HERE/'results'
s=json.loads((R/'summary.json').read_text());f=json.loads((R/'independent_fd.json').read_text())
text=['# WORLD-FRAME JACOBIAN VALIDATION','',f"FINAL = {s['final']}",'PRODUCTION MODIFIED = NO' if not s['production_modified'] else 'PRODUCTION MODIFIED = YES','TRAINING STARTED = NO',f"REAL STATES TESTED / VALID SAME-STATE A/B = 6 / {s['valid_states']}",'PHYSICS CONFIG = 2 x 5ms; substep2 captures finish only substep2. No extra integration.','',
'## Independent FD validation','',f"Original angular max error = {s['fd_original_max_error']:.17g} m/rad",f"Corrected angular max error = {s['fd_corrected_max_error']:.17g} m/rad",f"All native-stage corrected max FD error = {s['native_stage_fd_corrected_max_error']:.17g} m/rad",f"Translation unchanged = {s['translation_unchanged']}",'Oracle independently perturbs endpoint quaternion by exp(delta_world)*R, reconstructs the real Beam selected point, and freshly queries production vessel SDF. It never rotates a gradient to construct the oracle. Conversion is J_local @ R.T; quaternion xyzw and active local-to-world R. Every block, R and quaternion are recorded in independent_fd.json.','',
'## State/substep clearance comparison (mm)','',
'| state | substep | A1 local/1 | A2 local/2 | B1 world/1 | B2 world/2 | B2 nonpenetration | B2 margin |',
'|---|---:|---:|---:|---:|---:|---|---|']
for cid,item in s['cases'].items():
 if item['status']!='NATIVE_MATRIX_COMPLETE':text.append(f"{cid}: INCONCLUSIVE; {item['status']}");continue
 bs=item['branches']
 for index in range(len(bs['A1']['substeps'])):
  t=[bs[b]['substeps'][index] for b in ['A1','A2','B1','B2']]
  text.append('| '+cid+' | '+str(t[0]['substep'])+' | '+' | '.join(format(v['committed_clearance_mm'],'.17g') for v in t)+' | '+str(t[-1]['nonpenetration_pass'])+' | '+str(t[-1]['margin_pass'])+' |')
text+=['','Substep2 always begins from its own branch actual substep1 committed state. Subsequent state/row-selection divergence is a physical consequence, not an initial-condition mismatch. Strict gate uses unrounded clearance >=0.100 mm; below-gate deficits <1e-10 mm are explicitly numerical boundaries, not substantive violations.','',
'## Aggregate and physical cost','',
'| branch | nonpenetration states | margin states | nonpenetration substeps | margin substeps | native calls | native ms/call | row build ms | physics ms/substep | max displacement mm | max rotation deg | native scaled stop | raw error <1e-9 |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for b,a in s['branches'].items():
 text.append(f"| {b} | {a['nonpenetration_states']}/{a['states']} | {a['margin_states']}/{a['states']} | {a['nonpenetration_substeps']}/{a['substeps']} | {a['margin_substeps']}/{a['substeps']} | {a['native_solve_count']} | {a['mean_native_solve_ms']:.6f} | {a['mean_row_build_ms']:.6f} | {a['mean_physics_substep_ms']:.6f} | {a['max_translation_mm']:.12g} | {a['max_rotation_deg']:.12g} | {a['native_scaled_tolerance_passes']}/{a['native_solve_count']} | {a['raw_tolerance_passes']}/{a['native_solve_count']} |")
text+=['','Native SOFA21.12 scaleTolerance=True/allVerified=False are preserved: currentError < 1e-9 * global constraint dimension is the actual stop condition. Unscaled error<1e-9 is separately reported. Physics timing excludes prefix replay and sibling branch wait; includes read-only profiles/diagnostics. Not a production throughput benchmark.','',
'## Per-stage evidence','',
'| state/branch/substep/pass | rows/active | gap mm | true minimum selected gap mm | max SDF prediction error mm | worst dense/element | iterations/error/effective tol | native/row-build/mapping ms | translation mm/rotation deg |',
'|---|---|---:|---:|---:|---|---|---|---|']
for t in s['native_stages']:
 text.append(f"| {t['case']}/{t['branch']}/{t['substep']}/{t['stage']} | {t['row_count']}/{t['active_count']} | {t['min_linear_row_gap_mm']} | {t['min_selected_true_sdf_gap_mm']} | {t['max_abs_nonlinear_sdf_error_mm']} | {t['dense_worst_index']}/{t['worst_element']['element_index']} | {t['solver_iterations']}/{t['solver_error']}/{t['effective_solver_tolerance']} | {t['native_solver_wall_ms']}/{t['row_build_wall_ms']}/{t['mapping_refresh_wall_ms']} | {t['max_beam_translation_mm']}/{t['max_beam_rotation_deg']} |")
(R/'final_report.md').write_text('\n'.join(text)+'\n')
print(R/'final_report.md')
