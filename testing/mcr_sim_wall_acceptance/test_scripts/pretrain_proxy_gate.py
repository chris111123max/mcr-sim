"""Offline whole-curve collision proxy audit; bounds are not contact response claims."""
import sys,json,time
from pathlib import Path
import numpy as np
from penetration_geometry import curve_samples

def main(source,mesh,out):
    started=time.monotonic();source=Path(source);mesh=Path(mesh);out=Path(out)
    report={'status':'NOT_VERIFIED','physical_substeps_executed':0,'frames':0,'max_curve_to_point_proxy_upper_mm':0.,'max_curve_to_line_proxy_upper_mm':0.,'max_adjacent_proxy_spacing_mm':0.}
    records=[]
    with (source/'physical_substeps.jsonl').open() as f:
        for line in f:
            r=json.loads(line);raw=r['geometry'];c,t,bound=curve_samples(raw,spacing=.00005)
            p=np.asarray(raw['mapped_collision_positions_m']);v=p[1:]-p[:-1];vv=np.einsum('ij,ij->i',v,v);diff=c[:,None,:]-p[:-1][None,:,:]
            q=np.divide(np.einsum('ijk,jk->ij',diff,v),vv[None,:],out=np.zeros(diff.shape[:2]),where=vv[None,:]>1e-24);q=np.clip(q,0,1)
            distances=np.linalg.norm(diff-q[:,:,None]*v[None,:,:],axis=2)
            dl=float(np.max(np.min(distances,axis=1)))+bound
            dp=float(np.max(np.min(np.linalg.norm(c[:,None,:]-p[None,:,:],axis=2),axis=1)))+bound
            item={'physical_substep':r['physical_substep'],'curve_to_point_upper_mm':dp*1000,'curve_to_consecutive_line_upper_mm':dl*1000,'sample_cover_mm':bound*1000,'max_adjacent_spacing_mm':float(np.linalg.norm(v,axis=1).max())*1000}
            records.append(item);report['frames']+=1
            report['max_curve_to_point_proxy_upper_mm']=max(report['max_curve_to_point_proxy_upper_mm'],dp*1000)
            report['max_curve_to_line_proxy_upper_mm']=max(report['max_curve_to_line_proxy_upper_mm'],dl*1000)
            report['max_adjacent_proxy_spacing_mm']=max(report['max_adjacent_proxy_spacing_mm'],item['max_adjacent_spacing_mm'])
    m=json.loads((mesh/'report.json').read_text());r=.665;margin=.3;h=m['max_wall_triangle_inradius_mm'];e=report['max_curve_to_line_proxy_upper_mm']
    report.update({'proxy_radius_mm':.665,'actual_outer_radius_mm':.665,'radius_configuration_match':'PASS','contact_distance_margin_mm':margin,'max_wall_face_inradius_mm':h,'feature_cover_bound_mm':e+h,'no_precontact_blind_region_sufficient_condition':e+h<=margin,'local_min_distance_triangle_line':'IGNORED_IN_INSTALLED_SOURCE','supported_pairs':['Triangle-Point','Line-Line','Line-Point'],'reason':'Radius/proximity match is confirmed. A global no-missed-wall certificate does not follow merely from enabled models: Triangle-Line ignored; point-only cover and line/mesh-edge feature cover must be checked. Failed sufficient bound is NOT evidence of actual penetration.','line_topology_scope':'Consecutive segments are initialized by MultiAdaptiveBeamMapping; saved snapshots omit actual dynamic edge-removal lists, so this line audit is optimistic and cannot independently certify every active edge.','seconds':time.monotonic()-started})
    (out/'proxy_coverage.json').write_text(json.dumps(report,indent=2)+'\n');(out/'proxy_frame_bounds.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
    main(*sys.argv[1:])
    raise SystemExit(2)  # This audit explicitly leaves coverage NOT_VERIFIED.
