"""Initialization-only actual SOFA mesh/units check; never calls animate."""
import sys,json,traceback,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; TEST=HERE.parent; PYTHON=TEST.parents[1]
sys.path.insert(0,str(PYTHON));sys.path.insert(0,str(TEST))
from wall_geometry import read_stl
from mcr_sim.vessel_bundle import load_vessel_bundle
from mcr_sim.sofa_simulation import SofaBackend

def main():
    vessel=Path(sys.argv[1]).resolve(); out=Path(sys.argv[2]); backend=SofaBackend(); report={'physics_substeps':0,'status':'FAIL'}
    try:
        bundle=load_vessel_bundle(vessel); backend.reset(bundle); h=backend._handles
        actual=np.unique(np.asarray(h.vessel.MO.position.value,float),axis=0)
        expected=np.unique(read_stl(vessel/'collision_inner.stl').triangles_m.reshape(-1,3),axis=0)
        error=None if actual.shape!=expected.shape else float(np.max(np.abs(actual-expected)))
        loader=h.vessel.CollisionModel.getObject('meshLoader'); fields={}
        for data in loader.getDataFields():
            if data.getName() in ('filename','scale','scale3d','translation','rotation'):fields[data.getName()]=data.getValueString()
        report.update({'status':'PASS' if error is not None and error<=1e-12 else 'NOT_VERIFIED','loaded_filename_and_transform_Data':fields,'declared_production_environment_stl':h.vessel.environment_stl,'expected_stl_sha256':hashlib.sha256((vessel/'collision_inner.stl').read_bytes()).hexdigest(),'actual_unique_vertex_count':len(actual),'expected_unique_vertex_count':len(expected),'actual_bbox_min_m':actual.min(0).tolist(),'actual_bbox_max_m':actual.max(0).tolist(),'max_sorted_vertex_coordinate_error_m':error,'observed_dt_s':float(backend._root.dt.value),'geometry_provenance':'Actual production vessel MechanicalObject positions vs binary final STL vertices converted mm -> m. No integration.'})
    except Exception:report['traceback']=traceback.format_exc()
    finally:
        report['physics_substeps']=backend.physics_substeps_executed; backend.close();out.write_text(json.dumps(report,indent=2)+'\n')
    print('LOADED_MESH_CHECK',report['status'],'substeps',report['physics_substeps']);return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
