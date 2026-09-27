#!/usr/bin/env python3
"""Detailed deterministic multistep audit for the isolated planar Beam PoC."""
import json, sys, time, traceback
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import beam_feasible_poc as poc
ROOT=HERE.parent
OUT=ROOT/'_runtime/results'; LOG=ROOT/'_runtime/logs'
OUT.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)

def pct(v,p): return float(np.percentile(np.asarray(v,dtype=float),p)) if v else 0.0

def run_length(n):
    root,mo,base=poc.make_scene(); u=np.zeros(12); rows=[]; error=None
    try:
        for step in range(1,n+1):
            tic=time.perf_counter_ns()
            try:
                u,out=poc.solve_step(base,u)
                q=poc.poses(base,u)
                if not out['accepted']['feasible']: raise RuntimeError('accepted state infeasible')
                mo.position.value=q.tolist()
                elapsed=time.perf_counter_ns()-tic
                free=float(out['free_proposal']['min_sampled_clearance_m'])
                accepted=float(out['accepted']['min_sampled_clearance_m'])
                rows.append({'run_steps':n,'step':step,
                    'free_proposal_min_clearance_m':free,'free_proposal_min_clearance_mm':free*1000,
                    'accepted_min_clearance_m':accepted,'accepted_min_clearance_mm':accepted*1000,
                    'active_constraints':out['active_constraints_max'],
                    'nonlinear_iterations':out['nonlinear_iterations'],
                    'relinearization_count':out['relinearization_count'],
                    'line_search_iterations':len(out['line_search_trace']),
                    'line_search_candidate_count':len(out['line_search_trace']),
                    'accepted_alpha':out['line_search_alpha_min'],
                    'segments_checked':out['accepted']['segments_checked'],
                    'adaptive_subdivision_count':out['accepted']['subdivisions'],
                    'max_subdivision_depth':out['accepted']['max_subdivision_depth'],
                    'max_constraint_violation_m':max(0.0,-accepted),
                    'max_constraint_violation_mm':max(0.0,-accepted*1000),
                    'finite':bool(np.isfinite(u).all() and np.isfinite(q).all()),
                    'nan_count':int(np.isnan(u).sum()+np.isnan(q).sum()),
                    'inf_count':int(np.isinf(u).sum()+np.isinf(q).sum()),
                    'step_runtime_ns':elapsed,'step_runtime_ms':elapsed/1e6})
                if not rows[-1]['finite']: raise FloatingPointError('non-finite state')
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                rows.append({'run_steps':n,'step':step,'failure':error,
                             'step_runtime_ns':time.perf_counter_ns()-tic})
                break
    finally:
        import Sofa
        Sofa.Simulation.unload(root)
    runtime=[r['step_runtime_ms'] for r in rows if 'step_runtime_ms' in r]
    acc=[r['max_constraint_violation_mm'] for r in rows if 'max_constraint_violation_mm' in r]
    free=[r['free_proposal_min_clearance_mm'] for r in rows if 'free_proposal_min_clearance_mm' in r]
    iters=[r['nonlinear_iterations'] for r in rows if 'nonlinear_iterations' in r]
    subdiv=[r['adaptive_subdivision_count'] for r in rows if 'adaptive_subdivision_count' in r]
    alpha=[r['accepted_alpha'] for r in rows if 'accepted_alpha' in r]
    summary={'requested_steps':n,'completed_steps':len(runtime),'status':'PASS' if len(runtime)==n and not error and max(acc or [0])<=0.001 and all(r['finite'] for r in rows if 'finite' in r) else 'FAIL',
      'free_proposal_penetrating_steps':sum(x<0 for x in free),
      'accepted_penetrating_steps':sum(x>0 for x in acc),
      'accepted_violation_max_mm':max(acc or [0]),'accepted_violation_p95_mm':pct(acc,95),'accepted_violation_p99_mm':pct(acc,99),
      'accepted_violation_threshold_counts':{f'>{t:g}_mm':sum(x>t for x in acc) for t in [.01,.05,.1,.2,.5]},
      'active_constraint_steps':sum(r.get('active_constraints',0)>0 for r in rows),
      'line_search_triggered_steps':sum(r.get('line_search_iterations',0)>1 for r in rows),
      'alpha_lt_0_5_count':sum(x<.5 for x in alpha),'alpha_lt_0_1_count':sum(x<.1 for x in alpha),
      'nonlinear_iterations_mean':float(np.mean(iters)) if iters else 0,'nonlinear_iterations_p95':pct(iters,95),'nonlinear_iterations_max':max(iters or [0]),
      'subdivisions_mean':float(np.mean(subdiv)) if subdiv else 0,'subdivisions_p95':pct(subdiv,95),'max_subdivision_depth':max((r.get('max_subdivision_depth',0) for r in rows),default=0),
      'nan_count':sum(r.get('nan_count',0) for r in rows),'inf_count':sum(r.get('inf_count',0) for r in rows),
      'failure_count':int(error is not None),'failure':error,
      'step_runtime_mean_ms':float(np.mean(runtime)) if runtime else 0,'step_runtime_median_ms':pct(runtime,50),'step_runtime_p95_ms':pct(runtime,95),'step_runtime_p99_ms':pct(runtime,99),'step_runtime_max_ms':max(runtime or [0]),
      'total_runtime_s':sum(runtime)/1000,'steps_per_second':(len(runtime)/(sum(runtime)/1000)) if sum(runtime)>0 else 0,
      'model_limitation':'solver/geometry pseudo-steps only; no SOFA Simulation.animate or SDF query'}
    return rows,summary

def main():
    all_rows=[]; summaries={}
    for n in (20,100,1000):
        rows,s=run_length(n); all_rows.extend(rows); summaries[str(n)]=s
        print(f'{n}: {s["status"]}; accepted max={s["accepted_violation_max_mm"]:.6g} mm; mean={s["step_runtime_mean_ms"]:.3f} ms')
    with (OUT/'beam_multistep_detailed.jsonl').open('w') as f:
        for row in all_rows: f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    (OUT/'beam_multistep_summary.json').write_text(json.dumps({'runs':summaries},indent=2,allow_nan=False)+'\n')
    print(f'WROTE {OUT}/beam_multistep_detailed.jsonl and beam_multistep_summary.json')
if __name__=='__main__': main()
