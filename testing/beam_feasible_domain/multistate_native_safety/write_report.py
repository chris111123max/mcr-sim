from pathlib import Path
import json,numpy as np
HERE=Path(__file__).resolve().parent
R=HERE/'results'
s=json.loads((R/'summary.json').read_text())
lines=['# MULTISTATE NATIVE BEAM SAFETY VALIDATION','',
 'PRODUCTION MODIFIED = NO','PPO TRAINING STARTED = NO',
 f'TESTED STATES = {s["tested_states"]}',f'VALID SAME-STATE REPLAYS = {s["valid_same_state_replays"]}',
 f'VESSELS COVERED = {s["vessels"]}',f'TARGETS COVERED = {s["targets"]}',
 'PHYSICS SUBSTEPS = 2 x 5 ms; a snapshot already at substep2 tests only that substep.',
 'EXTRA PHYSICS SUBSTEPS = 0','',
 'State pass counts require every remaining planned substep to pass. Substep counts are also reported.',
 '| branch | nonpenetration states | margin states | nonpenetration substeps | margin substeps | worst committed mm | mean native solve ms | native total per substep ms | row-build mean ms | physics substep mean ms |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for b in 'ABC':
 a=s['branches'][b]
 def val(k):return str(a[k]) if a[k] is None else f'{a[k]:.6f}'
 lines.append(f'| {b}: {"ABC".index(b)+1} solves | {a["nonpenetration_pass_count"]}/{a["states_with_data"]} | {a["margin_pass_count"]}/{a["states_with_data"]} | {a["nonpenetration_substeps"]}/{a["substeps"]} | {a["margin_substeps"]}/{a["substeps"]} | {val("worst_committed_clearance_mm")} | {val("mean_native_solver_ms")} | {val("mean_native_solver_total_per_substep_ms")} | {val("mean_row_build_ms")} | {val("mean_physics_substep_ms")} |')
lines+=['','## Every target substep','',
 '| state / vessel / target | branch/substep | free mm | committed mm | margin | native solve count | rows/active per pass | linear gap mm per pass | nonlinear error mm per pass | error / iterations per pass |',
 '|---|---|---:|---:|---|---:|---|---|---|---|']
stages=[];two=[]
for cid,item in s['cases'].items():
 if not item.get('branches'):
  lines.append(f'| {cid} | excluded: {item.get("result",item).get("status")} | | | | | | | | |');continue
 case=item['result']['case']
 for b,branch in item['branches'].items():
  if 'data' not in branch:lines.append(f'| {cid} | {b}: INCONCLUSIVE | | | | | | | | |');continue
  data=branch['data']
  if case['substep']==1:two.append(branch)
  for sub in data['substeps']:
   t=sub['stages'];stages+=t
   def series(key):return ', '.join('N/A' if x[key] is None else f'{x[key]:.8g}' for x in t)
   rows=', '.join(f'{x["row_count"]}/{x["active_count"]}' for x in t)
   errors=', '.join(f'{x["solver_error"]:.4g}/{x["solver_iterations"]}' for x in t)
   lines.append(f'| {cid} / {case["case"]["vessel"]} / {case["case"]["target"]} | {b}/{sub["substep"]} | {sub["free_clearance_mm"]:.6f} | {sub["committed_clearance_mm"]:.6f} | {sub["margin_pass"]} | {sub["native_solve_calls"]} | {rows} | {series("min_linear_row_gap_mm")} | {series("max_abs_nonlinear_sdf_error_mm")} | {errors} |')
lines+=['','## Audit and interpretation','',f'NATIVE EXECUTION AUDIT = {"PASS" if not s["native_audit_faults"] else s["native_audit_faults"]}',
 f'ROW COUNT / ACTIVE COUNT mismatches = {sum(a["row_active_mismatches"] for a in s["branches"].values())}',
 f'SOLVER ERROR < configured 1e-9 = {sum(x["solver_error_below_tolerance"] for x in stages)}/{len(stages)} native passes',
 'SOFA21.12 scaleTolerance=True, allVerified=False are preserved. Its native stopping threshold is tolerance * global constraint dimension; this is distinct from the raw error<1e-9 requirement.',
 f'Native scaled-threshold passes = {sum(x["solver_error"]<x["effective_solver_tolerance"] for x in stages)}/{len(stages)}',
 f'TWO-SUBSTEP SAFETY INVARIANCE = {sum(x["nonpenetration"] for x in two)}/{len(two)} branch runs; margin = {sum(x["margin"] for x in two)}/{len(two)}',
 'Timings use steady_clock inside the native solver/mapping and perf_counter around the target substep. Prefix replay and sibling fork waits are excluded. Physics wall includes diagnostic callbacks, so it is not a production training throughput benchmark.',
 'Historical -0.766410mm failure root cause = UNPROVEN',f'FINAL RESULT = {s["final_result"]}']
for b in 'BC':
 a=s['branches'][b];ref=s['branches']['A']
 if a['mean_physics_substep_ms'] and ref['mean_physics_substep_ms']:
  lines.append(f'NUMERICAL COST RATIO {b}/A: physics wall {a["mean_physics_substep_ms"]/ref["mean_physics_substep_ms"]:.3f}x; native total {a["mean_native_solver_total_per_substep_ms"]/ref["mean_native_solver_total_per_substep_ms"]:.3f}x')
(R/'final_report.md').write_text('\n'.join(lines)+'\n')
print(R/'final_report.md')
