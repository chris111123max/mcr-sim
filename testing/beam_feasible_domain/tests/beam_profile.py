#!/usr/bin/env python3
"""10 warmup + 100 measured steps, minimal vs detailed profile; PoC only."""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import beam_feasible_poc as poc
ROOT=HERE.parent; OUT=ROOT/'_runtime/results'; LOG=ROOT/'_runtime/logs'
OUT.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)
CTX={'profile':None}
ORIG={n:getattr(poc,n) for n in ('controls','bezier','sample_g','cert')}
ORIG_ROTATION=poc.Rotation

def _phase_bucket(p,key):
    return p.setdefault(key,{})
def add_nested(name,elapsed,geo_delta=0):
    p=CTX['profile']
    if p is None: return
    phase=p.get('_active_phase','unassigned')
    bucket=p.setdefault('_nested_by_phase_ns',{})
    values=bucket.setdefault(name,{})
    values[phase]=values.get(phase,0)+max(0,elapsed-geo_delta)

def _geo_clock(p):
    return sum(p.get('_geometry_by_phase_ns',{}).values())+sum(p.get('_bezier_by_phase_ns',{}).values())+sum(p.get('_transform_by_phase_ns',{}).values())
def _note_transform(ns):
    p=CTX['profile']
    if p is None: return
    phase=p.get('_active_phase','unassigned'); b=_phase_bucket(p,'_transform_by_phase_ns')
    b[phase]=b.get(phase,0)+ns; p['coordinate_transform_calls']=p.get('coordinate_transform_calls',0)+1
class _RotationProxy:
    def __init__(self,obj): self._obj=obj
    def _call(self,name,*a,**k):
        tic=time.perf_counter_ns(); value=getattr(self._obj,name)(*a,**k); _note_transform(time.perf_counter_ns()-tic); return value
    def apply(self,*a,**k): return self._call('apply',*a,**k)
    def as_quat(self,*a,**k): return self._call('as_quat',*a,**k)
class _RotationFactory:
    @staticmethod
    def _new(name,*a,**k):
        tic=time.perf_counter_ns(); value=getattr(ORIG_ROTATION,name)(*a,**k); _note_transform(time.perf_counter_ns()-tic); return _RotationProxy(value)
    @staticmethod
    def from_quat(*a,**k): return _RotationFactory._new('from_quat',*a,**k)
    @staticmethod
    def from_rotvec(*a,**k): return _RotationFactory._new('from_rotvec',*a,**k)

def timed_controls(*a,**k):
    p=CTX['profile']; before=_geo_clock(p) if p is not None else 0; tic=time.perf_counter_ns()
    v=ORIG['controls'](*a,**k); ns=time.perf_counter_ns()-tic
    if p is not None:
        transform_delta=_geo_clock(p)-before
        phase=p.get('_active_phase','unassigned'); b=_phase_bucket(p,'_geometry_by_phase_ns')
        b[phase]=b.get(phase,0)+max(0,ns-transform_delta)
        p['geometry_interpolation_calls']=p.get('geometry_interpolation_calls',0)+1
    return v

def timed_bezier(*a,**k):
    tic=time.perf_counter_ns(); v=ORIG['bezier'](*a,**k); ns=time.perf_counter_ns()-tic
    p=CTX['profile']
    if p is not None:
        phase=p.get('_active_phase','unassigned'); b=_phase_bucket(p,'_bezier_by_phase_ns')
        b[phase]=b.get(phase,0)+ns; p['bezier_evaluations']=p.get('bezier_evaluations',0)+1
    return v

def timed_sample_g(*a,**k):
    p=CTX['profile']; tic=time.perf_counter_ns(); before=_geo_clock(p) if p is not None else 0
    v=ORIG['sample_g'](*a,**k); ns=time.perf_counter_ns()-tic
    if p is not None:
        after=_geo_clock(p); add_nested('sample_clearance_eval',ns,after-before)
        p['sample_g_calls']=p.get('sample_g_calls',0)+1; p['sample_points_evaluated']=p.get('sample_points_evaluated',0)+len(v)
    return v

def timed_cert(*a,**k):
    p=CTX['profile']; tic=time.perf_counter_ns(); before=_geo_clock(p) if p is not None else 0
    v=ORIG['cert'](*a,**k); ns=time.perf_counter_ns()-tic
    if p is not None:
        after=_geo_clock(p); add_nested('adaptive_certification',ns,after-before)
        p['certification_calls']=p.get('certification_calls',0)+1
        p['segments_checked']=p.get('segments_checked',0)+v['segments_checked']
        p['subdivision_nodes_created']=p.get('subdivision_nodes_created',0)+v['subdivisions']
        p['max_subdivision_depth']=max(p.get('max_subdivision_depth',0),v['max_subdivision_depth'])
    return v

# The small wrappers are only installed for PROFILE MODE; MINIMAL MODE has no per-call instrumentation.
def install_wrappers():
    poc.controls=timed_controls; poc.bezier=timed_bezier; poc.sample_g=timed_sample_g; poc.cert=timed_cert; poc.Rotation=_RotationFactory

def restore_wrappers():
    for n,v in ORIG.items(): setattr(poc,n,v)
    poc.Rotation=ORIG_ROTATION

def run_mode(mode):
    profile_mode=mode=='profile'
    if profile_mode: install_wrappers()
    root,mo,base=poc.make_scene(); u=np.zeros(12); measured=[]
    try:
        for i in range(110):
            pr={} if profile_mode else None
            CTX['profile']=pr
            tic=time.perf_counter_ns()
            u,out=poc.solve_step(base,u,profile=pr)
            q=poc.poses(base,u)
            mo.position.value=q.tolist()
            wall=time.perf_counter_ns()-tic
            if i>=10:
                row={'mode':mode,'measured_step':i-9,'runtime_ns':wall,'runtime_ms':wall/1e6,
                     'accepted_clearance_m':out['accepted']['min_sampled_clearance_m'],
                     'free_clearance_m':out['free_proposal']['min_sampled_clearance_m'],
                     'nonlinear_iterations':out['nonlinear_iterations'],
                     'relinearization_count':out['relinearization_count'],
                     'active_constraints':out['active_constraints_max'],
                     'jacobian_rows':sum(t['active_constraints'] for t in out['trace']),
                     'line_search_candidate_count':len(out['line_search_trace']),
                     'line_search_triggered':len(out['line_search_trace'])>out['nonlinear_iterations'],
                     'accepted_alpha':out['line_search_alpha_min'],
                     'finite':bool(np.isfinite(u).all() and np.isfinite(q).all())}
                if profile_mode:
                    raw=pr.get('_phase_raw_ns',{}); geo=pr.get('_geometry_by_phase_ns',{})
                    bez=pr.get('_bezier_by_phase_ns',{}); trans=pr.get('_transform_by_phase_ns',{})
                    nested=pr.get('_nested_by_phase_ns',{})
                    geo_by={k:int(v) for k,v in geo.items()}; bez_by={k:int(v) for k,v in bez.items()}; trans_by={k:int(v) for k,v in trans.items()}
                    sample_by=nested.get('sample_clearance_eval',{})
                    cert_by=nested.get('adaptive_certification',{})
                    categories={'geometry_interpolation_ns':sum(geo_by.values()),
                                'bezier_evaluation_ns':sum(bez_by.values()),
                                'coordinate_transform_ns':sum(trans_by.values()),
                                'sample_clearance_eval_ns':sum(sample_by.values()),
                                'adaptive_segment_certification_ns':sum(cert_by.values())}
                    names={'proposal':'proposal_residual_ns','active_set':'active_set_construction_ns',
                           'jacobian':'jacobian_assembly_ns','optimizer':'nonlinear_solve_ns',
                           'iteration_diagnostics':'iteration_diagnostics_ns','line_search':'line_search_bookkeeping_ns',
                           'commit_diagnostics':'state_commit_diagnostics_ns','finalize':'finalization_ns'}
                    for phase,outname in names.items():
                        residual=raw.get(phase,0)-geo_by.get(phase,0)-bez_by.get(phase,0)-trans_by.get(phase,0)-sample_by.get(phase,0)-cert_by.get(phase,0)
                        categories[outname]=max(0,int(residual))
                    classified=sum(categories.values()); phase_total=sum(raw.values())
                    categories['other_ns']=max(0,int(wall-phase_total))
                    row['exclusive_stage_ns']=categories
                    row['work_counts']={'sdf_queries':0,'gradient_queries':0,
                        'geometry_interpolations':pr.get('geometry_interpolation_calls',0),
                        'coordinate_transform_calls':pr.get('coordinate_transform_calls',0),
                        'bezier_evaluations':pr.get('bezier_evaluations',0),
                        'sample_g_calls':pr.get('sample_g_calls',0),
                        'sample_points_evaluated':pr.get('sample_points_evaluated',0),
                        'certification_calls':pr.get('certification_calls',0),
                        'segments_checked':pr.get('segments_checked',0),
                        'subdivision_nodes_created':pr.get('subdivision_nodes_created',0),
                        'max_subdivision_depth':pr.get('max_subdivision_depth',0),
                        'active_constraint_count':out['active_constraints_max'],
                        'jacobian_rows':row['jacobian_rows'],
                        'nonlinear_iterations':out['nonlinear_iterations'],
                        'relinearization_count':out['relinearization_count'],
                        'line_search_candidate_count':len(out['line_search_trace']),
                        'final_alpha':out['line_search_alpha_min']}
                measured.append(row)
    finally:
        CTX['profile']=None
        import Sofa
        Sofa.Simulation.unload(root)
        if profile_mode: restore_wrappers()
    return measured

def stats(values):
    x=np.asarray(values,dtype=float)
    return {'mean':float(np.mean(x)),'median':float(np.median(x)),'p95':float(np.percentile(x,95)),
            'p99':float(np.percentile(x,99)),'max':float(np.max(x))}

def main():
    minimal=run_mode('minimal'); profiled=run_mode('profile'); rows=minimal+profiled
    with (OUT/'beam_profile_per_step.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    m=stats([r['runtime_ms'] for r in minimal]); p=stats([r['runtime_ms'] for r in profiled])
    stage_ns={}
    for r in profiled:
        for k,v in r['exclusive_stage_ns'].items(): stage_ns.setdefault(k,[]).append(v)
    mean_total_ns=float(np.mean([r['runtime_ns'] for r in profiled]))
    stages={k:{'mean_ms':float(np.mean(v))/1e6,'median_ms':float(np.median(v))/1e6,
               'p95_ms':float(np.percentile(v,95))/1e6,'percentage_of_profiled_wall':100*float(np.mean(v))/mean_total_ns}
            for k,v in stage_ns.items()}
    top=sorted(stages.items(),key=lambda kv:kv[1]['mean_ms'],reverse=True)[:5]
    summary={'warmup_steps_excluded':10,'measured_steps':100,'model_limitation':'planar Bezier PoC; no production SDF, no SOFA animate/physics integration',
      'minimal_mode_step_ms':m,'profile_mode_step_ms':p,
      'profile_overhead_percent':100*(p['mean']-m['mean'])/m['mean'] if m['mean'] else 0,
      'stage_exclusive':stages,'top_5_bottlenecks':[{'name':k,**v} for k,v in top],
      'work_counts_mean':{k:float(np.mean([r['work_counts'][k] for r in profiled])) for k in profiled[0]['work_counts']}}
    (OUT/'beam_profile_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(f"minimal={m['mean']:.3f} ms; profile={p['mean']:.3f} ms; overhead={summary['profile_overhead_percent']:.1f}%")
    for x in summary['top_5_bottlenecks']: print(f"{x['name']}: {x['mean_ms']:.3f} ms ({x['percentage_of_profiled_wall']:.1f}%)")
    print(f'WROTE {OUT}/beam_profile_per_step.jsonl and beam_profile_summary.json')
if __name__=='__main__': main()
