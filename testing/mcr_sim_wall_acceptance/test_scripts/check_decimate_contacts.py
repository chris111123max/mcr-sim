"""Offline whole inserted-Beam solid certification against BOTH unchanged STLs."""
import sys,json,time,traceback,hashlib
from pathlib import Path
import numpy as np
from penetration_geometry import Detector
from refine_saved_contact import certify,calibration
HERE=Path(__file__).resolve().parent;TEST=HERE.parent

def save(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
def main(root):
    root=Path(root).resolve();start=time.monotonic();gate=json.loads((root/'wrapper_retry/geometry_report.json').read_text())
    assert gate['stage_a_status']=='PASS'
    report={'status':'NOT_VERIFIED','measurements':{},'errors':[],'new_physics_substeps':0,'candidate_searches':0,'self_intersection':'NOT_VERIFIED','scope':'Saved substep endpoint geometries, exact Beam Bezier controls, swept 0.665 mm radius ball upper bound plus actual sampled tube witnesses when the upper bound is positive; no between-integrator-endpoint certificate'}
    deadline=time.monotonic()+600
    try:
        report['synthetic_calibration']=calibration();detectors={}
        for name,path in [('fine',gate['source_directory']),('candidate',gate['candidate_directory'])]:
            d=Detector(Path(path));detectors[name]=d
        report['detector_validation']={name:d.report for name,d in detectors.items()}
        for trajectory in ['fine','candidate']:
            capture=json.loads((root/('sofa_'+trajectory)/'report.json').read_text());assert capture['status']=='PASS_CAPTURE_COMPLETE'
            counts={};summaries={name:{'max_sampled_entity_penetration_mm':0.,'maximum_continuous_entity_depth_upper_mm':0.,'minimum_continuous_gap_lower_mm':None,'uncertain_frames':0,'sampled_penetration_frames':0,'max_critical_sampling_bound_mm':0.,'worst_frame':None,'frames':0} for name in detectors}
            budgets={name:{'queries':0,'deadline':deadline} for name in detectors}
            with (root/(trajectory+'_geometry_substeps.jsonl')).open('w') as stream:
                with (root/('sofa_'+trajectory)/'physical_substeps.jsonl').open() as f:
                    for line in f:
                        row=json.loads(line);i=row['physical_substep'];assert row['substep_within_action']==(i-1)%22+1
                        counts[row['rl_action']]=counts.get(row['rl_action'],0)+1;measure={'physical_substep':i,'rl_action':row['rl_action'],'walls':{}}
                        for wall,d in detectors.items():
                            res=certify(row['geometry'],d.q,d.tol,budgets[wall]);sampled=0.;witness=None
                            if res['whole_entity_depth_upper_mm']>0:
                                actual=d.measure(row['geometry']);sampled=actual['max_sampled_surface_penetration_mm'];witness=actual['max_penetration_position_m'];res['actual_surface_samples']=actual['surface_sample_count'];res['actual_sampled_surface_penetration_mm']=sampled
                            res['classification']='PENETRATION_DETECTED' if sampled>d.tol*1000 else ('NO_ENDPOINT_PENETRATION_CERTIFIED' if res['status']=='PASS' else 'NOT_VERIFIED')
                            measure['walls'][wall]=res;s=summaries[wall];s['frames']+=1
                            upper=res['whole_entity_depth_upper_mm'];s['maximum_continuous_entity_depth_upper_mm']=max(s['maximum_continuous_entity_depth_upper_mm'],upper)
                            gap=res['continuous_gap_lower_mm'];s['minimum_continuous_gap_lower_mm']=gap if s['minimum_continuous_gap_lower_mm'] is None else min(s['minimum_continuous_gap_lower_mm'],gap)
                            s['max_critical_sampling_bound_mm']=max(s['max_critical_sampling_bound_mm'],res['max_critical_longitudinal_bound_mm']);s['uncertain_frames']+=int(res['classification']=='NOT_VERIFIED');s['sampled_penetration_frames']+=int(res['classification']=='PENETRATION_DETECTED')
                            if sampled>s['max_sampled_entity_penetration_mm']:s['max_sampled_entity_penetration_mm']=sampled;s['worst_frame']={'physical_substep':i,'rl_action':row['rl_action'],'action':row['action'],'position_m':witness}
                        stream.write(json.dumps(measure,allow_nan=False)+'\n');stream.flush()
                        if i%22==0:print('GEOMETRY_REPLAY_CHECK',trajectory,i,{k:(v['max_sampled_entity_penetration_mm'],v['maximum_continuous_entity_depth_upper_mm']) for k,v in summaries.items()},flush=True)
            assert len(counts)==15 and all(v==22 for v in counts.values())
            for wall,s in summaries.items():
                s['status']='PENETRATION_DETECTED' if s['sampled_penetration_frames'] else ('PASS_ENDPOINT_GEOMETRY' if not s['uncertain_frames'] else 'NOT_VERIFIED');s['point_queries']=budgets[wall]['queries']
            report['measurements'][trajectory]={'by_wall':summaries,'RL_substep_counts':counts};save(root/'contact_geometry_report.json',report)
        report['status']='COMPLETED_WITH_PENETRATION' if any(s['sampled_penetration_frames'] for t in report['measurements'].values() for s in t['by_wall'].values()) else 'COMPLETED_ENDPOINT_GEOMETRY_CHECK'
    except BaseException:report['errors'].append(traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:report['seconds']=time.monotonic()-start;save(root/'contact_geometry_report.json',report);print('CONTACT_GEOMETRY_FINAL',report['status'],flush=True)
    return 0 if not report['errors'] else 2
if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
