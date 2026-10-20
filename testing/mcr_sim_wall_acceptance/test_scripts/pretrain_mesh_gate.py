"""Read actual production scene mesh and proxies; initialization only, no animate."""
import sys,json,struct,time,traceback,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;TEST=HERE.parent;PYTHON=TEST.parents[1]
sys.path[:0]=[str(PYTHON),str(TEST)]
from mcr_sim.sofa_simulation import SofaBackend
from mcr_sim.vessel_bundle import load_vessel_bundle
from run_wall_challenge import object_data,geometry

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(f):
    f=np.asarray(f,np.int64); k=np.argmin(f,axis=1);x=np.take_along_axis(f,(k[:,None]+np.arange(3))%3,axis=1)
    return x[np.lexsort((x[:,2],x[:,1],x[:,0]))]

def main(vessel,out):
    vessel=Path(vessel).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=False);backend=None;started=time.monotonic()
    report={'status':'NOT_VERIFIED','physics_substeps':0,'strict_self_intersection':'NOT_VERIFIED'}
    try:
        old=json.loads((vessel/'parameters.json').read_text());report['generator_parameters']=old['invocation']
        report['generator_0_3mm']=old['effective_profile']['meshing']['voxel_spacing_mm']==[.3]*3
        backend=SofaBackend();backend.reset(load_vessel_bundle(vessel));h=backend._handles
        blob=(vessel/'collision_inner.stl').read_bytes();n=struct.unpack('<I',blob[80:84])[0]
        assert len(blob)==84+50*n
        facets=np.frombuffer(blob,offset=84,dtype=[('normal','<f4',(3,)),('p','<f4',(3,3)),('attr','<u2')]);mm=facets['p'].reshape(-1,3).astype(float)
        # MeshLoader::parse uses getAttributeAsFloat, then casts float scale to SReal.
        effective_scale=float(np.float32(.001));expected=mm*effective_scale
        v,inv=np.unique(expected,axis=0,return_inverse=True)
        actual=np.asarray(h.vessel.MO.position.value,float);av,ai=np.unique(actual,axis=0,return_inverse=True)
        report['runtime_source_backed_scale']=effective_scale
        report['exact_vertex_match_under_source_arithmetic']=bool(np.array_equal(av,v))
        report['source_arithmetic_max_error_m']=float(np.max(np.abs(av-v))) if av.shape==v.shape else None
        report['max_deviation_from_exact_0_001_m']=float(np.max(np.abs(v-np.unique(mm,axis=0)*.001)))
        topo=[o for o in h.vessel.CollisionModel.objects if o.getClassName()=='MeshTopology']
        if len(topo)!=1:raise ValueError('Unique actual vessel MeshTopology unavailable: '+str([(o.getName(),o.getClassName()) for o in h.vessel.CollisionModel.objects]))
        tr=np.asarray(topo[0].triangles.value,np.int64);edges=np.asarray(topo[0].edges.value,np.int64)
        report['triangle_count']=len(tr);report['wall_edge_count']=len(edges)
        report['oriented_triangle_connectivity_match']=bool(np.array_equal(canonical(ai[tr]),canonical(inv.reshape(-1,3))))
        points=actual[tr];a=np.linalg.norm(points[:,1]-points[:,0],axis=1);b=np.linalg.norm(points[:,2]-points[:,1],axis=1);c=np.linalg.norm(points[:,0]-points[:,2],axis=1)
        rin=np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1)/(a+b+c)
        report['max_wall_triangle_inradius_mm']=float(rin.max()*1000)
        loader=h.vessel.CollisionModel.getObject('meshLoader');report['loader']=object_data(loader)
        report['wall_models']=[{'class':o.getClassName(),'path':o.getPathName(),'data':object_data(o)} for o in h.vessel.CollisionModel.objects if 'CollisionModel' in o.getClassName()]
        report['catheter_models']=[{'class':o.getClassName(),'path':o.getPathName(),'data':object_data(o)} for o in h.instrument.CollisionModel.objects if o.getClassName() in ('LineCollisionModel','PointCollisionModel')]
        report['intersection']=object_data(h.simulator.intersection)
        report['dt_s']=float(backend._root.dt.value);report['initial_geometry']=geometry(backend)
        np.savez_compressed(out/'actual_collision_mesh.npz',vertices=actual,triangles=tr,edges=edges)
        report['stl_sha256']=sha(vessel/'collision_inner.stl');report['declared_stl']=h.vessel.environment_stl
        report['mesh_status']='PASS_SOURCE_EXACT' if report['source_arithmetic_max_error_m'] is not None and report['source_arithmetic_max_error_m']<=1e-12 and report['oriented_triangle_connectivity_match'] and len(edges)>0 else 'FAIL'
        report['status']='PASS' if report['mesh_status']=='PASS_SOURCE_EXACT' and report['generator_0_3mm'] else 'FAIL'
    except BaseException:report['traceback']=traceback.format_exc();report['status']='FAIL'
    finally:
        if backend is not None:report['physics_substeps']=backend.physics_substeps_executed;backend.close()
        report['seconds']=time.monotonic()-started;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('MESH_GATE',report['status'],report.get('traceback',''),flush=True)
    return 0 if report['status']=='PASS' else 2
def reconcile_saved(directory,output):
    directory=Path(directory);r=json.loads((directory/'report.json').read_text());stl=Path(r['declared_stl'])
    if sha(stl)!=r['stl_sha256']:raise ValueError('Original STL has changed')
    blob=stl.read_bytes();facets=np.frombuffer(blob,offset=84,dtype=[('normal','<f4',(3,)),('p','<f4',(3,3)),('attr','<u2')])
    mm=np.unique(facets['p'].reshape(-1,3).astype(float),axis=0)
    av=np.unique(np.load(directory/'actual_collision_mesh.npz')['vertices'],axis=0)
    scale=np.array(r['loader']['scale3d']);err=float(np.max(np.abs(av-mm*scale)))
    result={'status':'PASS_SOURCE_PRECISION' if err<=1e-12 and r['oriented_triangle_connectivity_match'] else 'FAIL','reused_runtime_report':str(directory/'report.json'),'source_arithmetic_max_error_m':err,'unmodified_comparison_precision_m':1e-12,'actual_scale':scale.tolist(),'ideal_scale_max_error_m':float(np.max(np.abs(av-mm*.001))),'oriented_triangle_connectivity_match':r['oriented_triangle_connectivity_match'],'new_physics_substeps':0,'reason':'Installed MeshLoader.cpp parse uses getAttributeAsFloat before SReal conversion. Corrected expected arithmetic, not relaxed Stage A tolerances. Initial bitwise-equality failure remains preserved.'}
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print('SAVED_MESH_RECONCILIATION',result['status']);return 0 if result['status']=='PASS_SOURCE_PRECISION' else 2
if __name__=='__main__':
    raise SystemExit(reconcile_saved(sys.argv[2],sys.argv[3]) if sys.argv[1]=='--saved' else main(sys.argv[1],sys.argv[2]))
