"""Offline adaptive certification of every saved real Beam frame. No SOFA imports."""
import sys,json,time,math,hashlib,traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import vtk
from penetration_geometry import Detector,DistanceQuery

HERE=Path(__file__).resolve().parent; TEST=HERE.parent
RADIUS=.000665; MESH_GUARD=.000000020; LOCAL_BOUND=.000002; CRITICAL=.0001

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def certify(raw,q,tol,budget):
    controls=np.asarray(raw['bezier_control_points_m'],float)
    if controls.ndim!=3 or controls.shape[1:]!=(4,3) or len(controls)==0 or not np.all(np.isfinite(controls)):raise ValueError('Invalid saved true Beam controls')
    result={'whole_entity_depth_upper_mm':0.,'continuous_gap_lower_mm':float('inf'),'point_queries':0,'accepted_patches':0,'critical_patches':0,'max_critical_longitudinal_bound_mm':0.,'unresolved_patches':0}
    for beam,hull in enumerate(controls):
        speed=3*float(np.max(np.linalg.norm(np.diff(hull,axis=0),axis=1)))
        stack=[(0.,1.,0)]
        while stack:
            a,b,depth=stack.pop();t=(a+b)/2;half=(b-a)/2
            c=(1-t)**3*hull[0]+3*(1-t)**2*t*hull[1]+3*(1-t)*t*t*hull[2]+t**3*hull[3]
            signed,inside=q.query([c]);d=float(signed[0]);result['point_queries']+=1;budget['queries']+=1
            if (d<0)!=bool(inside[0]) and abs(d)>tol:raise ValueError('Sign/enclosure disagreement')
            if budget['queries']>500000 or time.monotonic()>budget['deadline']:raise RuntimeError('Offline refinement compute cap; no false PASS')
            cover=speed*half
            upper=d+RADIUS+cover+MESH_GUARD+tol
            critical=upper>=-CRITICAL
            if critical and cover>LOCAL_BOUND and depth<24:
                stack.extend([(a,t,depth+1),(t,b,depth+1)]);continue
            result['accepted_patches']+=1
            if critical:
                result['critical_patches']+=1
                result['max_critical_longitudinal_bound_mm']=max(result['max_critical_longitudinal_bound_mm'],cover*1000)
            if upper>=0:result['unresolved_patches']+=1
            result['continuous_gap_lower_mm']=min(result['continuous_gap_lower_mm'],-upper*1000)
            result['whole_entity_depth_upper_mm']=max(result['whole_entity_depth_upper_mm'],max(0.,upper)*1000)
    result['status']='PASS' if result['unresolved_patches']==0 else 'NOT_VERIFIED'
    return result

def calibration():
    src=vtk.vtkCubeSource();src.SetXLength(.010);src.SetYLength(.010);src.SetZLength(.010);src.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION);src.Update()
    tri=vtk.vtkTriangleFilter();tri.SetInputConnection(src.GetOutputPort());tri.Update()
    clean=vtk.vtkCleanPolyData();clean.SetInputConnection(tri.GetOutputPort());clean.Update();q=DistanceQuery(clean.GetOutput())
    cases=[]
    for name,y,clear in [('inside',.003,True),('tangent',.005-RADIUS,False),('known_0.2mm_breach',.005-RADIUS+.0002,False)]:
        h=np.array([[-.003,y,0],[-.001,y,0],[.001,y,0],[.003,y,0]])
        res=certify({'bezier_control_points_m':[h.tolist()]},q,1e-12,{'queries':0,'deadline':time.monotonic()+60})
        cases.append({'case':name,'result':res,'expected_clear':clear})
        if (res['status']=='PASS')!=clear:raise ValueError('Synthetic coverage calibration failed')
        if 'breach' in name and res['whole_entity_depth_upper_mm']<.2:raise ValueError('Known penetration upper bound missed')
    return {'status':'PASS','cases':cases,'scope':'Mathematical cover calibration only, not fake SOFA data'}

def main():
    source=Path(sys.argv[1]).resolve();old=json.loads((source/'report.json').read_text())
    out=TEST/'reports'/('contact_refinement_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'));out.mkdir()
    print('OUTPUT',out,flush=True);started=time.monotonic()
    saved=source/'physical_substeps.jsonl';protected=json.loads((source/'protected_sha256.json').read_text())
    report={'status':'NOT_VERIFIED','source_run':str(source),'source_trace_sha256':sha(saved),'new_sofa_physics_substeps':0,'candidate_action_searches':0,'rl_substeps_per_full_action':22,'saved_frames':0,'limits':{'point_queries':500000,'seconds_after_detector_init':600,'local_longitudinal_bound_mm':LOCAL_BOUND*1000},'radius_mm':RADIUS*1000,'mesh_guard_mm':MESH_GUARD*1000,'errors':[]}
    stream=None;frames=[]
    try:
        report['calibration']=calibration();print('COVER_CALIBRATION_PASS',flush=True)
        detector=Detector(Path(old['vessel']));report['inherited_detector']=detector.report
        budget={'queries':0,'deadline':time.monotonic()+600};stream=(out/'refined_substeps.jsonl').open('w')
        counts={};iteration_rows=[];pair_rows=[]
        with saved.open() as f:
            for text in f:
                row=json.loads(text);i=row['physical_substep'];rl=row['rl_action'];within=i-22*(rl-1)
                if not 1<=within<=22:raise ValueError('Saved frame violates 22-substep grouping')
                if i!=len(frames)+1:raise ValueError('Missing or reordered physical frame')
                res=certify(row['geometry'],detector.q,detector.tol,budget)
                res.update({'physical_substep':i,'rl_action':rl,'substep_within_action':within,'saved_sampled_entity_depth_mm':row['wall']['max_penetration_depth_mm']})
                frames.append(res);counts[rl]=counts.get(rl,0)+1;stream.write(json.dumps(res)+'\n');stream.flush()
                graph=str(row['solver']['lcp']['data']['graph']);data=row['solver']['lcp']['data']
                if not str(graph).startswith('Error '):raise ValueError('LCP graph layout differs; iterations NOT_OBSERVED')
                nums=graph[6:].split()
                try:vals=[float(x) for x in nums]
                except ValueError:raise ValueError('LCP graph not reliably parseable')
                if data['build_lcp']!=0 or data['mu']!=.1:raise ValueError('Different saved LCP branch')
                # Source-verified: exactly one residual push per unbuilt GS sweep.
                iteration_rows.append({'physical_substep':i,'sweeps_from_residual_vector':len(vals),'raw_graph':graph})
                if row['observed_contact_constraint_objects']:pair_rows.append({'physical_substep':i,'objects':row['observed_contact_constraint_objects']})
                if i%22==0:print('REFINED_RL',rl,'frames',i,'queries',budget['queries'],'upper_mm',max(x['whole_entity_depth_upper_mm'] for x in frames),flush=True)
        report.update({'saved_frames':len(frames),'per_rl_action_substep_counts':counts,'partial_final_action':{'rl_action':16,'substeps':counts.get(16),'reason':old['stop_reason']},'continuous_swept_outer_radius_depth_upper_mm':max(x['whole_entity_depth_upper_mm'] for x in frames),'minimum_continuous_gap_lower_mm':min(x['continuous_gap_lower_mm'] for x in frames),'max_sampled_entity_depth_mm':max(x['saved_sampled_entity_depth_mm'] for x in frames),'max_critical_local_cover_bound_mm':max(x['max_critical_longitudinal_bound_mm'] for x in frames),'point_queries':budget['queries'],'geometry_saved_frames_status':'PASS' if all(x['status']=='PASS' for x in frames) else 'NOT_VERIFIED','lcp_sweep_count':{'status':'SOURCE_DERIVED_OBSERVED','min':min(x['sweeps_from_residual_vector'] for x in iteration_rows),'max':max(x['sweeps_from_residual_vector'] for x in iteration_rows),'method':'Length of per-step Error residual vector, one append per unbuilt GS sweep; not a direct Python iteration getter.'},'contact_count':'NOT_OBSERVED','contact_force':'NOT_OBSERVED','contact_pair_object_frames':len(pair_rows),'strict_anti_penetration':'NOT_VERIFIED','limitations':['Offline processing covers saved substep endpoints, not motion inside an integrator substep.','Union of radius0.665mm balls over exact saved cubic beams bounds the whole catheter solid, including terminal caps.','STL global robust self-intersection certificate and physical contact-force observation remain missing.','Mesh-coordinate guard comes from earlier runtime/file comparison; strict identity remained NOT_VERIFIED.','Single vessel and bounded load cannot establish all-condition prevention.']})
        (out/'lcp_sweeps.json').write_text(json.dumps(iteration_rows,indent=2)+'\n');(out/'contact_pair_objects.json').write_text(json.dumps(pair_rows,indent=2)+'\n')
    except BaseException:report['errors'].append(traceback.format_exc())
    finally:
        if stream:stream.close()
        report['preservation']={'status':'PASS' if all(Path(p).is_file() and sha(p)==h for p,h in protected.items()) else 'FAIL','files':len(protected)}
        report['seconds']=time.monotonic()-started
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('FINAL',report.get('geometry_saved_frames_status'),'STRICT',report['status'],'frames',report['saved_frames'],'REPORT',out/'report.json',flush=True)
    return 0 if not report['errors'] and report.get('geometry_saved_frames_status')=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
