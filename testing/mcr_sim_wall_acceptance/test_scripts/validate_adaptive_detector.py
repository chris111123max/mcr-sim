"""Analytic measurement calibration only; no fake SOFA acceptance data."""
import sys,json
from pathlib import Path
import numpy as np
import vtk
from adaptive_wall_geometry import AdaptiveDetector
from penetration_geometry import DistanceQuery,calibration

def main():
    src=vtk.vtkCubeSource();src.SetXLength(.01);src.SetYLength(.01);src.SetZLength(.01);src.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION);src.Update()
    tri=vtk.vtkTriangleFilter();tri.SetInputConnection(src.GetOutputPort());tri.Update();clean=vtk.vtkCleanPolyData();clean.SetInputConnection(tri.GetOutputPort());clean.Update()
    d=object.__new__(AdaptiveDetector);d.q=DistanceQuery(clean.GetOutput());d.radius=.000665;d.tol=3.2e-14
    cases=[]
    for name,y,expected in [('inside',.003,0.),('tangent',.005-.000665,0.),('side_penetration',.005-.000665+.0002,.2)]:
        controls=np.array([[-.003,y,0],[-.001,y,0],[.001,y,0],[.003,y,0]])
        raw={'dofs':[[-.003,y,0,0,0,0,1],[.003,y,0,0,0,0,1]],'edges':[[0,1]],'lengths':[.006],'first':0,'aligned':False,'bezier_control_points_m':[controls.tolist()],'mapped_collision_positions_m':[controls[0].tolist(),controls[-1].tolist()],'geometry_source':'SYNTHETIC_UNIT_ONLY_NOT_SOFA'}
        w=d.measure(raw);cases.append({'case':name,'expected_depth_mm':expected,'measured_depth_mm':w['max_penetration_depth_mm'],'gap_interval_mm':[w['minimum_entity_gap_lower_bound_mm'],w['minimum_entity_gap_sampled_upper_bound_mm']],'pass':abs(w['max_penetration_depth_mm']-expected)<1e-7})
    report={'status':'PASS' if all(x['pass'] for x in cases) else 'FAIL','cases':cases,'inherited_signed_distance_and_open_boundary_calibration':calibration(),'note':'Synthetic detector validation, not real physical test results.'}
    if report['inherited_signed_distance_and_open_boundary_calibration']['status']!='PASS':report['status']='FAIL'
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n');print(report['status']);return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
