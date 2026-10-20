"""Refine only previously unresolved seed-303 saved SOFA endpoint geometries.

Compares the true saved Beam Bezier catheter against both original and 99%
decimated vessel surfaces. Dense physical tube surface WITNESSES are measured;
a conservative swept-radius upper bound is retained from the original report.
This is not a continuous-time contact certificate and it starts NO SOFA.
"""
import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
TEST=HERE.parent
PYTHON=TEST.parents[1]
sys.path[:0]=[str(TEST),str(HERE),str(PYTHON)]
from penetration_geometry import Detector,curve_samples
from run_acceptance import jsonable

SPACING_M=0.00005
ANGLES=96
MAX_SAMPLES_PER_CASE=200000
MAX_CASES=12


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')


def unsigned_tube_witness(raw,detector,deadline):
    """Sample true swept circular rod cross-sections and planar endcaps.

    The sample set does NOT supply a certified angular/arc continuous bound.
    The previously computed conservative swept BALL upper bound remains valid
    and is reported separately. A positive surface witness is decisive.
    """
    c,t,gap=curve_samples(raw,spacing=SPACING_M)
    if time.monotonic()>deadline:raise TimeoutError('Geometry refinement deadline exceeded')
    signed,inside=detector.q.query(c)
    mismatch=((signed<0)!=inside)&(np.abs(signed)>detector.tol)
    if mismatch.any():raise ValueError('Centerline signed distance / containment mismatch')
    near=signed+detector.radius+gap>=-0.0001
    theta=np.arange(ANGLES)*2*np.pi/ANGLES
    cos=np.cos(theta);sin=np.sin(theta)
    indices=np.flatnonzero(near)
    if len(indices)*ANGLES+(2*7*ANGLES)>MAX_SAMPLES_PER_CASE:
        raise RuntimeError('Actual tube surface samples exceed declared cap; no false PASS')
    best_depth=-float('inf');best_point=None;best_contained=None
    sampled=0;outside_count=0
    def examine(points):
        nonlocal best_depth,best_point,best_contained,sampled,outside_count
        arr=np.asarray(points,float).reshape(-1,3)
        for start in range(0,len(arr),4096):
            segment=arr[start:start+4096]
            depths,enclosed=detector.q.query(segment)
            bad=((depths<0)!=enclosed)&(np.abs(depths)>detector.tol)
            if bad.any():raise ValueError('Tube signed distance / containment mismatch')
            j=int(np.argmax(depths))
            if depths[j]>best_depth:
                best_depth=float(depths[j]);best_point=segment[j].tolist()
                best_contained=bool(enclosed[j])
            outside_count+=int(np.count_nonzero(depths>detector.tol))
            sampled+=len(segment)
            if time.monotonic()>deadline:raise TimeoutError('Geometry refinement deadline exceeded')
    def frame(k):
        z=np.array([0.,0.,1.]) if abs(t[k,2])<.9 else np.array([0.,1.,0.])
        u=np.cross(t[k],z);u/=np.linalg.norm(u)
        v=np.cross(t[k],u)
        return u,v
    for k in indices:
        u,v=frame(k)
        examine(c[k]+detector.radius*(cos[:,None]*u+sin[:,None]*v))
    # Include the entire rod's open Bezier end disks; no artificial inlet exclusion.
    for k in (0,len(c)-1):
        u,v=frame(k)
        examine(c[k])
        for frac in np.linspace(1/6,1.,6):
            examine(c[k]+frac*detector.radius*(cos[:,None]*u+sin[:,None]*v))
    depth=max(0.,best_depth)
    return {
        'status':'PENETRATION_DETECTED' if depth>detector.tol else 'NO_SAMPLED_TUBE_SURFACE_OUTSIDE',
        'maximum_actual_sampled_tube_surface_depth_mm':depth*1000,
        'max_signed_witness_distance_mm':best_depth*1000,
        'surface_witness_m':best_point if depth else None,
        'surface_witness_inside_boolean':best_contained,
        'independent_triangle_distance_witness':detector.q.witness(best_point),
        'actual_sample_points':sampled,
        'samples_outside_numeric_tolerance':outside_count,
        'num_centerline_samples':len(c),
        'num_critical_cross_sections':len(indices),
        'cross_section_angles':ANGLES,
        'longitudinal_curve_spacing_request_mm':SPACING_M*1000,
        'longitudinal_cover_half_interval_upper_mm':gap*1000,
        'continuous_tube_surface_depth_upper_mm':'NOT_VERIFIED',
        'continuous_time_depth_upper_mm':'NOT_VERIFIED',
        'scope':'Actual sampled perpendicular circular rod body and both end disks at prior SOFA saved endpoints; no angular or time continuum proof',
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--comparison',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--budget-seconds',type=int,default=900)
    args=ap.parse_args()
    comparison=args.comparison.resolve()
    gate=json.loads((comparison/'wrapper_retry/geometry_report.json').read_text())
    if int(gate['seed'])!=303 or gate['stage_a_status']!='PASS':
        raise ValueError('Only already accepted seed-303 comparison is allowed')
    sources={'fine':Path(gate['source_directory']).resolve(),
             'candidate':Path(gate['candidate_directory']).resolve()}
    for kind,root in sources.items():
        known=gate['source_stl_sha256'] if kind=='fine' else gate['candidate_stl_sha256']
        if sha(root/'collision_inner.stl')!=known:raise ValueError('Input vessel STL changed: '+kind)
    output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    started=time.monotonic()
    report={'status':'FAIL','seed':303,'wall_stl_hashes':{
                kind:sha(root/'collision_inner.stl') for kind,root in sources.items()},
            'physics_substeps_executed':0,'new_actions':0,
            'max_cases':MAX_CASES,'spacing_mm':SPACING_M*1000,'angles':ANGLES,
            'cases':[],'uncertain_frames':[],'errors':[],
            'global_collision_equivalence':'NOT_VERIFIED',
            'strict_self_intersection':'NOT_VERIFIED',
            'continuous_time_penetration':'NOT_VERIFIED'}
    try:
        geometry_cases=[]
        for trajectory in ('fine','candidate'):
            file=comparison/(trajectory+'_geometry_substeps.jsonl')
            if not file.is_file():raise FileNotFoundError(file)
            with file.open() as f:
                for line in f:
                    item=json.loads(line)
                    uncertain={wall:details for wall,details in item['walls'].items()
                               if details['classification'] in ('NOT_VERIFIED','PENETRATION_DETECTED')}
                    for wall,prior in uncertain.items():
                        geometry_cases.append((trajectory,int(item['physical_substep']),wall,prior))
        if len(geometry_cases)>MAX_CASES:
            raise RuntimeError('Unexpected more than %d unresolved cases; no silent truncation'%MAX_CASES)
        wanted={}
        for trajectory,index,wall,prior in geometry_cases:
            wanted.setdefault(trajectory,set()).add(index)
        saved={}
        for trajectory,indices in wanted.items():
            src=comparison/('sofa_'+trajectory)/'physical_substeps.jsonl'
            records={}
            with src.open() as f:
                for line in f:
                    item=json.loads(line)
                    i=int(item['physical_substep'])
                    if i in indices:records[i]=item
            if set(records)!=indices:raise ValueError('Missing original saved physical frame: '+trajectory)
            saved[trajectory]=records
        report['uncertain_frames']=sorted(set(i for _,i,_,_ in geometry_cases))
        save(output/'critical_frames.json',{
            'seed':303,'physical_substeps':report['uncertain_frames'],
            'cases':[{'trajectory':t,'physical_substep':i,'wall':w}
                     for t,i,w,_ in geometry_cases]})
        detectors={kind:Detector(path) for kind,path in sources.items()}
        report['detectors']={k:v.report for k,v in detectors.items()}
        for trajectory,i,wall,prior in geometry_cases:
            deadline=time.monotonic()+args.budget_seconds
            measurement={'trajectory':trajectory,'physical_substep':i,'wall':wall,
                         'old_continuous_swept_ball_upper_mm':prior['whole_entity_depth_upper_mm'],
                         'old_status':prior['classification'],
                         'old_critical_curve_bound_mm':prior.get('max_critical_longitudinal_bound_mm'),
                         'old_interval_not_a_measured_breach':True}
            try:
                measured=unsigned_tube_witness(saved[trajectory][i]['geometry'],
                                               detectors[wall],deadline)
                measurement['refined_sampled_tube']=measured
                measurement['status']=measured['status']
            except Exception:
                measurement['status']='NOT_VERIFIED'
                measurement['error']=traceback.format_exc()
            report['cases'].append(measurement)
            print('CRITICAL_303',trajectory,i,wall,measurement['status'],flush=True)
            save(output/'critical303_report.json',report)
        positives=[x for x in report['cases'] if x['status']=='PENETRATION_DETECTED']
        failures=[x for x in report['cases'] if x['status']=='NOT_VERIFIED']
        report['status']='PENETRATION_DETECTED' if positives else (
            'COMPLETED_WITH_UNCERTAIN_CASES' if failures else 'DENSE_SAMPLES_COMPLETE_NOT_CONTINUOUSLY_CERTIFIED')
        report['sampled_positive_cases']=len(positives)
        report['unresolved_measurement_cases']=len(failures)
    except BaseException:
        report['errors'].append(traceback.format_exc())
        print(report['errors'][-1],flush=True)
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['input_stls_unchanged']=all(
            sha(root/'collision_inner.stl')==report['wall_stl_hashes'][kind]
            for kind,root in sources.items())
        if not report['input_stls_unchanged']:report['status']='FAIL_SOURCE_CHANGED'
        save(output/'critical303_report.json',report)
        print('CRITICAL_303_FINAL',report['status'],'cases',len(report['cases']),flush=True)
    return 0 if report['status']=='DENSE_SAMPLES_COMPLETE_NOT_CONTINUOUSLY_CERTIFIED' else 2


if __name__=='__main__':
    raise SystemExit(main())
