"""Independent exact PyVista decimate(0.99), read-back acceptance, no SOFA."""
import sys,json,time,hashlib,shutil,traceback,inspect,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy
HERE=Path(__file__).resolve().parent;TEST=HERE.parent;PYTHON=TEST.parents[1]
sys.path[:0]=[str(TEST),str(PYTHON),str(PYTHON/'stEVE_anatomical_vessels/src')]
from export_assets import read_stl,mesh_quality,save_stl
from validate_geometry import validate
from run_acceptance import inspect_asset,jsonable
from mcr_sim.vessel_bundle import load_vessel_bundle

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):p.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')
def main(out, seed=101):
    seed=int(seed)
    if seed not in (101,202,303): raise ValueError('Only previously approved cohort seeds 101, 202, 303 are supported')
    out=Path(out).resolve();started=time.monotonic()
    manifest=json.loads((TEST/'reports/pretrain_final_20261020/ten_cohort_generation.json').read_text())
    source=Path(manifest['assets'][str(seed)]['directory']);meta=json.loads((source/'metadata.json').read_text());params=json.loads((source/'parameters.json').read_text())
    protected={str(p):sha(p) for p in (PYTHON/'mcr_sim').rglob('*.py')}
    protected.update({str(p):sha(p) for p in source.rglob('*') if p.is_file()})
    save(out/'protected_sha256.json',protected)
    result={'status':'FAIL','seed':seed,'source_directory':str(source),'source_stl_sha256':sha(source/'collision_inner.stl'),'method':'pv.read(fine_stl).decimate(0.99), no extra kwargs; empty-metadata output compatibility bridge; VTK STL serialization only','physics_substeps':0,'production_published':False,'errors':[],'stages':{'geometry':'NOT_RUN','performance':'NOT_RUN','contact_replay':'NOT_RUN'}}
    candidate_dir=out/'candidate'/('arch_type_I_seed_%d'%seed);candidate_dir.mkdir(parents=True,exist_ok=False)
    try:
        assert manifest['status']=='ALL_TEN_STAGE_A_PASS' and meta['validation']['passed'] and int(meta['seed'])==seed
        assert meta['meshing']['voxel_spacing_mm']==[.3]*3 and meta['scale_diameter_array']==[.85]
        for q in source.iterdir():
            if q.is_file() and q.name not in ['collision_inner.stl','metadata.json']:shutil.copy2(q,candidate_dir/q.name)
        mesh=pv.read(str(source/'collision_inner.stl'));assert mesh.is_all_triangles
        result['software']={'pyvista':pv.__version__,'vtk':vtk.vtkVersion.GetVTKVersion(),'decimate_signature':str(inspect.signature(pv.PolyDataFilters.decimate)),'underlying_filter':'vtkQuadricDecimation'}
        (out/'pyvista_decimate_implementation.py.txt').write_text(inspect.getsource(pv.PolyDataFilters.decimate))
        # VTK9.5 shadows old PyVista metadata properties. Input STL has no arrays;
        # bypass only empty Python metadata propagation, not the VTK filter.
        import pyvista.core.filters.poly_data as filters
        get_output=filters._get_output
        def geometry_only_output(alg,*args,**kwargs):
            incoming=alg.GetInputDataObject(0,0);outgoing=alg.GetOutputDataObject(0)
            assert all(x.GetNumberOfArrays()==0 for obj in [incoming,outgoing] for x in [obj.GetPointData(),obj.GetCellData(),obj.GetFieldData()])
            assert alg.GetClassName()=='vtkQuadricDecimation' and alg.GetTargetReduction()==.99
            result['actual_vtk_filter']={'class':alg.GetClassName(),'target_reduction':alg.GetTargetReduction(),'volume_preservation':alg.GetVolumePreservation(),'attribute_error_metric':alg.GetAttributeErrorMetric(),'scalars':alg.GetScalarsAttribute(),'vectors':alg.GetVectorsAttribute(),'normals':alg.GetNormalsAttribute(),'tcoords':alg.GetTCoordsAttribute(),'tensors':alg.GetTensorsAttribute()}
            result['pyvista_compatibility']={'scope':'Test-process empty metadata output only; exact original PyVista decimate function executes vtkQuadricDecimation; no filter parameter change','input_and_output_array_counts_all_zero':True,'source_function_sha256':hashlib.sha256(inspect.getsource(pv.PolyDataFilters.decimate).encode()).hexdigest()}
            return pv.wrap(outgoing)
        filters._get_output=geometry_only_output
        t=time.monotonic()
        try:candidate=mesh.decimate(0.99)
        finally:filters._get_output=get_output
        result['decimation_seconds']=time.monotonic()-t
        save_stl(candidate,candidate_dir/'collision_inner.stl')
        result.update({'candidate_directory':str(candidate_dir),'input_triangles':int(mesh.n_cells),'output_triangles_before_serialization':int(candidate.n_cells),'candidate_stl_sha256':sha(candidate_dir/'collision_inner.stl')})
        print('DECIMATE_COMPLETE',result['input_triangles'],result['output_triangles_before_serialization'],result['decimation_seconds'],flush=True)
        bundle=load_vessel_bundle(source)
        branches=[SimpleNamespace(name=b.name,coordinates=b.coordinates_mm,radii=b.radii_mm) for b in bundle.branches.values()]
        formal=validate(candidate_dir,branches,meta['anatomical_landmarks'],params['effective_profile'])
        save(out/'formal_geometry_validation.json',formal)
        meta['validation']=formal;meta['asset_sha256']['collision_inner.stl']=result['candidate_stl_sha256']
        meta['candidate_experiment']={'source':str(source),'source_stl_sha256':result['source_stl_sha256'],'method':result['method'],'not_production':True,'unchanged_centerlines':True}
        save(candidate_dir/'metadata.json',meta)
        independent,_,_=inspect_asset(candidate_dir,seed);save(out/'independent_stage_a.json',independent)
        result['formal_validation_passed']=formal['passed'];result['formal_failure_reasons']=formal['failure_reasons'];result['independent_stage_a_status']=independent['status'];result['mesh_quality']=formal['mesh_quality']
        sections=[]
        for key,items in formal['multi_section_diameters'].items():
            previous=bundle.metadata['validation']['multi_section_diameters'][key]
            for item,old in zip(items,previous):sections.append({'region':key,'branch':item['branch'],'index':item['centerline_index'],'nominal_mm':item['nominal_target_diameter_mm'],'randomized_design_mm':item['randomized_design_diameter_mm'],'fine_mm':old['diameter_median_mm'],'candidate_mm':item['diameter_median_mm'],'candidate_within_existing_tolerance':item['within_configured_tolerance'],'fine_to_candidate_delta_mm':None if item['diameter_median_mm'] is None else item['diameter_median_mm']-old['diameter_median_mm']})
        save(out/'diameter_comparison.json',sections)
        # Exact point-to-triangle distances at every vertex and triangle centroid.
        fine=read_stl(source/'collision_inner.stl');coarse=read_stl(candidate_dir/'collision_inner.stl')
        distance=vtk.vtkDistancePolyDataFilter();distance.SetInputData(0,fine);distance.SetInputData(1,coarse);distance.SignedDistanceOff();distance.ComputeSecondDistanceOn()
        if hasattr(distance,'ComputeCellCenterDistanceOn'):distance.ComputeCellCenterDistanceOn()
        distance.Update();surfaces=[];maximum=0.
        for name,poly in [('fine_to_candidate',distance.GetOutput()),('candidate_to_fine',distance.GetSecondDistanceOutput())]:
            d=np.abs(vtk_to_numpy(poly.GetPointData().GetArray('Distance')));k=int(np.argmax(d));row={'direction':name,'vertex_samples':len(d),'max_vertex_distance_mm':float(d[k]),'p95_vertex_distance_mm':float(np.percentile(d,95)),'p99_vertex_distance_mm':float(np.percentile(d,99)),'vertex_witness_mm':list(poly.GetPoint(k))};maximum=max(maximum,float(d.max()))
            cell=poly.GetCellData().GetArray('Distance')
            if cell is not None:
                c=np.abs(vtk_to_numpy(cell));row['centroid_samples']=len(c);row['max_centroid_distance_mm']=float(c.max());maximum=max(maximum,float(c.max()))
            surfaces.append(row)
        reference_budget=float(params['effective_profile']['meshing']['simplification_error_mm'])
        result['surface_deviation']={'method':'Bidirectional exact sampled-point to opposing triangle-surface distance; all vertices and available triangle centroids','measurements':surfaces,'sampled_hausdorff_lower_bound_mm':maximum,'strict_continuous_hausdorff_upper_mm':None,'strict_continuous_hausdorff_status':'NOT_VERIFIED','existing_simplification_parameter_mm':reference_budget,'parameter_is_not_preexisting_hausdorff_certificate':True,'exceeds_existing_parameter_reference':maximum>reference_budget}
        result['stage_a_status']='PASS' if formal['passed'] and independent['status']=='PASS' else 'FAIL'
        # A failed existing Stage A blocks physical testing unconditionally.
        result['stages']['geometry']=result['stage_a_status']
        result['status']=result['stage_a_status']
        if result['status']!='PASS':
            result['stages']['performance']='BLOCKED';result['stages']['contact_replay']='BLOCKED';result['stop_reason']='Candidate failed unchanged geometric acceptance. No candidate SOFA scene or integration permitted.'
        else:result['next_gate']='Review measured surface deviation and pre-existing geometry fidelity budget before any SOFA integration. No continuous surface equivalence claim.'
        tip_length=12.1*.4*math.pi;straight=450-tip_length
        result['official_default_jshaped']={'length_mm':450,'tip_length_mm':tip_length,'straight_length_mm':straight,'tip_collision_segments':math.ceil(2*tip_length),'straight_collision_segments':math.ceil(.1*straight),'total_collision_segments':math.ceil(2*tip_length)+math.ceil(.1*straight),'theoretical_connected_open_polyline_points':1+math.ceil(2*tip_length)+math.ceil(.1*straight),'runtime_official_count':'NOT_OBSERVED: official scene not instantiated; counts derived from official defaults'}
    except BaseException:result['errors'].append(traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        result['preservation']={'status':'PASS' if all(Path(p).is_file() and sha(p)==h for p,h in protected.items()) else 'FAIL','protected_files':len(protected)}
        result['seconds']=time.monotonic()-started;save(out/'geometry_report.json',result)
        print('GEOMETRY_FINAL',result['status'],'REASONS',result.get('formal_failure_reasons'),'REPORT',out/'geometry_report.json',flush=True)
    return 0 if result['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 101))
