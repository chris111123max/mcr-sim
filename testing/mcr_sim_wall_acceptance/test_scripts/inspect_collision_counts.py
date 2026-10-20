"""Initialize only the approved fine production scene; never integrate."""
import sys,json,time,math,traceback,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;TEST=HERE.parent;PYTHON=TEST.parents[1]
sys.path[:0]=[str(TEST),str(PYTHON)]
from mcr_sim.sofa_simulation import SofaBackend
from mcr_sim.vessel_bundle import load_vessel_bundle
from run_wall_challenge import object_data
from run_acceptance import jsonable,_log_messages

def main(root):
    root=Path(root);report={'status':'FAIL','physics_substeps':0,'candidate_scene_created':False};b=None
    manifest=json.loads((TEST/'reports/pretrain_final_20261020/ten_cohort_generation.json').read_text());fine=Path(manifest['assets']['101']['directory'])
    try:
        b=SofaBackend();b.reset(load_vessel_bundle(fine));h=b._handles;c=h.instrument.CollisionModel
        topology=c.getObject('collisEdgeSet');edges=np.asarray(topology.edges.value,dtype=int).reshape(-1,2)
        points=np.asarray(h.instrument.collision_dofs.position.value,dtype=float).reshape(-1,3)
        lengths=np.linalg.norm(points[edges[:,1]]-points[edges[:,0]],axis=1)
        models=[{'class':o.getClassName(),'path':o.getPathName(),'data':object_data(o),'exposed_methods':[name for name in ['getSize','isActive','getNumberOfElements','isElementEnabled'] if name in dir(o)]} for o in c.objects if o.getClassName() in ['LineCollisionModel','PointCollisionModel']]
        connected=bool(len(points)==len(edges)+1 and np.array_equal(edges,np.c_[np.arange(len(edges)),np.arange(1,len(edges)+1)]))
        report.update({'source_stl':str(fine/'collision_inner.stl'),'stl_sha256':hashlib.sha256((fine/'collision_inner.stl').read_bytes()).hexdigest(),'configured_segments':[b.config.instrument.discretization.collision_edges_body,b.config.instrument.discretization.collision_edges_tip],'runtime_topology_segments':len(edges),'runtime_point_DOFs':len(points),'runtime_referenced_point_ids':len(np.unique(edges)),'continuous_open_polyline':connected,'segments_geometrically_nonzero_gt_1e_12_m':int(np.count_nonzero(lengths>1e-12)),'min_segment_length_m':float(lengths.min()),'max_segment_length_m':float(lengths.max()),'models':models,'per_element_active_mask':'NOT_OBSERVED: binding does not expose a reliable per-element activation mask; topological count is not a contact count','IRC_xtip_m':np.asarray(h.ir_controller.xtip.value).tolist(),'IRC_indexFirstNode':np.asarray(h.ir_controller.indexFirstNode.value).tolist(),'actual_inserted_mechanical_DOFs':len(np.asarray(h.ir_controller.CurvAbs.value)),'dt_s':float(b._root.dt.value),'status':'PASS' if len(edges)==168 and len(points)==169 and connected else 'FAIL'})
        np.savez_compressed(root/'runtime_fine_collision_geometry.npz',points_m=points,edges=edges)
    except BaseException:report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
    finally:
        if b is not None:report['sofa_logs']=_log_messages(b);report['physics_substeps']=b.physics_substeps_executed;b.close()
        (root/'runtime_collision_counts.json').write_text(json.dumps(jsonable(report),indent=2)+'\n');print('RUNTIME_COUNTS',report['status'],report.get('runtime_topology_segments'),report.get('runtime_point_DOFs'),flush=True)
    return 0 if report['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
