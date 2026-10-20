"""Isolated read-only VTK worker; all units SI except explicitly suffixed mm."""
import sys,json,math,traceback
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_acceptance import inspect_asset,jsonable
from wall_geometry import _vtk_geometry

class DistanceQuery:
    def __init__(self,poly):
        self.poly=poly
        self.implicit=vtk.vtkImplicitPolyDataDistance(); self.implicit.SetInput(poly)
        self.enclosed=vtk.vtkSelectEnclosedPoints(); self.enclosed.SetSurfaceData(poly); self.enclosed.SetTolerance(1e-10); self.enclosed.Initialize(poly)
        self.locator=vtk.vtkStaticCellLocator(); self.locator.SetDataSet(poly); self.locator.BuildLocator()
    def query(self,p):
        p=np.asarray(p,dtype=np.float64).reshape(-1,3)
        inside=np.array([bool(self.enclosed.IsInsideSurface(float(x[0]),float(x[1]),float(x[2]))) for x in p])
        signed=np.array([self.implicit.EvaluateFunction(x) for x in p],dtype=float)
        return signed,inside
    def witness(self,p):
        closest=[0.,0.,0.]; cell=vtk.reference(0); sub=vtk.reference(0); d2=vtk.reference(0.)
        self.locator.FindClosestPoint(p,closest,cell,sub,d2)
        return {'closest_wall_m':closest,'cell_id':int(cell),'distance_mm':math.sqrt(float(d2))*1000}

def calibration():
    source=vtk.vtkCubeSource(); source.SetXLength(.010); source.SetYLength(.010); source.SetZLength(.010); source.SetCenter(0,0,0); source.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION); source.Update()
    tri=vtk.vtkTriangleFilter(); tri.SetInputConnection(source.GetOutputPort()); tri.Update()
    clean=vtk.vtkCleanPolyData(); clean.SetInputConnection(tri.GetOutputPort()); clean.Update()
    q=DistanceQuery(clean.GetOutput())
    points=np.array([[0,0,0],[.004,0,0],[.005,0,0],[.0052,0,0],[-.0052,0,0],[0,.0052,0],[0,0,.0052],[.006,.006,0]],float)
    expected=np.array([-.005,-.001,0,.0002,.0002,.0002,.0002,math.sqrt(2)*.001])
    values,inside=q.query(points); err=float(np.max(np.abs(values-expected)))
    witness_error=max(abs(q.witness(p)['distance_mm']/1000-abs(v)) for p,v in zip(points,values))
    r=.000665; theta=np.arange(64)*2*np.pi/64
    cases=[]
    for name,cy,expected_depth in [('contained',.003,0.),('tangent',.005-r,0.),('side_breach',.005-r+.0002,.0002)]:
        ring=np.c_[np.zeros(64),cy+r*np.cos(theta),r*np.sin(theta)]; d,inside_ring=q.query(ring)
        measured=max(0.,float(d.max())); cases.append({'name':name,'expected_depth_mm':expected_depth*1000,'measured_depth_mm':measured*1000,'error_mm':abs(measured-expected_depth)*1000})
    # A centerline inside the cube with a cap outside must not be missed.
    cap=np.array([[.0052,0,0],[.0052,r,0],[.0052,0,r]])
    cap_d,_=q.query(cap); cap_error=abs(float(cap_d.max())-.0002)
    tol=max(err,witness_error,cap_error,1e-15)*32
    passed=bool(err<1e-10 and witness_error<1e-10 and cap_error<1e-10 and all(c['error_mm']<1e-7 for c in cases) and np.all(inside[[0,1]]) and not np.any(inside[[3,4,5,6,7]]))
    # An open artificial surface is not assigned trusted signed distances.
    mesh=clean.GetOutput(); opened=vtk.vtkPolyData(); opened.DeepCopy(mesh); opened.BuildCells(); opened.DeleteCell(0); opened.RemoveDeletedCells()
    edges=vtk.vtkFeatureEdges(); edges.SetInputData(opened); edges.BoundaryEdgesOn(); edges.FeatureEdgesOff(); edges.NonManifoldEdgesOff(); edges.ManifoldEdgesOff(); edges.Update()
    open_rejected=edges.GetOutput().GetNumberOfCells()>0
    return {'status':'PASS' if passed and open_rejected else 'FAIL','analytic_cube_max_distance_error_mm':err*1000,'independent_locator_distance_error_mm':witness_error*1000,'point_queries_mm':(points*1000).tolist(),'expected_signed_distances_mm':(expected*1000).tolist(),'actual_signed_distances_mm':(values*1000).tolist(),'radius_cases':cases,'endcap_breach_error_mm':cap_error*1000,'open_mesh_rejected':open_rejected,'numeric_tolerance_m':tol,'scope':'Pointwise distance and containment calibration, not a geometry safety allowance.'}

def curve_samples(raw,spacing=.0001):
    dofs=np.asarray(raw['dofs'],float); edges=np.asarray(raw['edges'],int).reshape(-1,2); lengths=np.asarray(raw['lengths'],float).reshape(-1)
    if len(edges)!=len(lengths) or len(edges)==0 or (not raw['aligned'] and 'bezier_control_points_m' not in raw): raise ValueError('BeamAdapter edge/length/alignment data not reliable')
    def direction(q):
        q=np.asarray(q); q=q/np.linalg.norm(q); x,y,z,w=q
        return np.array([1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)])
    centers=[]; tangents=[]; bounds=[]
    for beam,((i,j),length) in enumerate(zip(edges,lengths)):
        if i<raw['first'] or j<raw['first']: raise ValueError('Interpolation contains a parked mechanical node')
        if 'bezier_control_points_m' in raw:
            p0,p1,p2,p3=np.asarray(raw['bezier_control_points_m'][beam],float)
        else:
            p0=dofs[i,:3]; p3=dofs[j,:3]; p1=p0+direction(dofs[i,3:7])*length/3; p2=p3-direction(dofs[j,3:7])*length/3
        hull=np.array([p0,p1,p2,p3]); control_length=float(np.linalg.norm(np.diff(hull,axis=0),axis=1).sum())
        # Derivative norm <= 3*max(control edge); gives a guaranteed parameter-step spatial bound.
        derivative_bound=3*float(np.max(np.linalg.norm(np.diff(hull,axis=0),axis=1)))
        count=max(2,int(math.ceil(derivative_bound/spacing))+1); t=np.linspace(0,1,count)
        p=(1-t[:,None])**3*p0+3*(1-t[:,None])**2*t[:,None]*p1+3*(1-t[:,None])*t[:,None]**2*p2+t[:,None]**3*p3
        dp=3*(1-t[:,None])**2*(p1-p0)+6*(1-t[:,None])*t[:,None]*(p2-p1)+3*t[:,None]**2*(p3-p2)
        n=np.linalg.norm(dp,axis=1)
        if np.any(n<1e-12): raise ValueError('Degenerate spline tangent')
        centers.extend(p); tangents.extend(dp/n[:,None]); bounds.append(derivative_bound/(count-1)/2)
    return np.asarray(centers),np.asarray(tangents),max(bounds)

class Detector:
    def __init__(self,vessel):
        self.cal=calibration()
        if self.cal['status']!='PASS': raise ValueError('Analytic detector calibration failed')
        asset,bundle,mesh_info=inspect_asset(vessel,101)
        if asset['status']!='PASS' or bundle is None: raise ValueError('Existing Stage A did not PASS: '+str(asset.get('reason')))
        if not bundle.metadata['validation']['passed']: raise ValueError('Generator validation not PASS')
        mesh,mesh_report=mesh_info; data,reason=_vtk_geometry(mesh)
        if data is None: raise ValueError(reason)
        self.q=DistanceQuery(data['surface']); self.radius=.000665
        self.tol=max(self.cal['numeric_tolerance_m'],np.finfo(float).eps*max(float(np.max(np.abs(mesh.vertices_m))),1.)*128)
        centers=np.vstack([b.coordinates_m for b in bundle.branches.values()]); low,high=mesh.vertices_m.min(0),mesh.vertices_m.max(0)
        exterior=high+(high-low); d,inside=self.q.query(np.vstack([centers,exterior]))
        sign_pass=bool(np.all(inside[:-1]) and np.all(d[:-1]<0) and not inside[-1] and d[-1]>0)
        if not sign_pass: raise ValueError('Actual vessel known interior/exterior sign validation failed')
        self.report={'status':'PASS','scope':'Calibrated pointwise signed STL distances and sampled physical tube; no full-mesh self-intersection certificate.','asset':asset,'calibration':self.cal,'actual_vessel_sign_check':'PASS','numeric_tolerance_mm':self.tol*1000,'radius_mm':.665,'spacing_upper_bound_mm':.1,'ring_samples':64,'inlet_policy':{'status':'PASS','exemption':'NONE','reason':'Actual STL is closed; insertion origin is inside. Only IRC active inserted beams are measured, parked/uninserted nodes excluded. Open artificial inlet fails topology gate; no arbitrary spatial exemptions.'},'self_intersection':mesh_report['self_intersection']}
    def measure(self,raw):
        c,t,gap=curve_samples(raw)
        signed,inside=self.q.query(c)
        mismatch=np.abs(signed)>self.tol
        if np.any(((signed<0)!=inside)&mismatch): raise ValueError('Actual centerline signed distance / containment disagreement')
        r=self.radius; near=signed+r+gap>=-self.tol
        theta=np.arange(64)*2*np.pi/64
        z=np.tile([0.,0.,1.],(len(c),1)); z[np.abs(t[:,2])>.9]=[0,1,0]
        u=np.cross(t,z); u/=np.linalg.norm(u,axis=1)[:,None]; v=np.cross(t,u)
        candidate=c[near,None,:]+r*(np.cos(theta)[None,:,None]*u[near,None,:]+np.sin(theta)[None,:,None]*v[near,None,:])
        # End disks include radial samples, not just the terminal center/rim.
        caps=[]
        for k in (0,-1):
            for frac in np.linspace(.00043/r,1,9): caps.extend(c[k]+frac*r*(np.cos(theta)[:,None]*u[k]+np.sin(theta)[:,None]*v[k]))
        points=np.vstack([candidate.reshape(-1,3),np.asarray(caps)])
        d,contained=self.q.query(points)
        if np.any(((d<0)!=contained)&(np.abs(d)>self.tol)): raise ValueError('Surface signed distance / containment disagreement')
        k=int(np.argmax(d)); maximum=max(0.,float(d[k])); witness=self.q.witness(points[k])
        if abs(witness['distance_mm']/1000-abs(d[k]))>self.tol*8: raise ValueError('Independent nearest-triangle distance disagrees')
        upper=max(0.,float(np.max(signed+r))+gap)
        classification='CONFIRMED_SAMPLED_SURFACE_OUTSIDE' if maximum>self.tol else ('CONTACT_WITHIN_NUMERIC_ERROR' if np.min(np.abs(d))<=self.tol else 'NO_SAMPLED_SURFACE_OUTSIDE')
        arc=np.linalg.norm(np.diff(c,axis=0),axis=1)
        mapped=np.asarray(raw.get('mapped_collision_positions_m',[]),float).reshape(-1,3)
        nearest_mapped_distance=float(np.max(np.min(np.linalg.norm(mapped[:,None,:]-c[None,:,:],axis=2),axis=1))) if len(mapped) else None
        return {'status':'PASS','classification':classification,'max_sampled_surface_penetration_mm':maximum*1000,'max_penetration_position_m':points[k].tolist() if maximum else None,'independent_witness':witness,'conservative_whole_tube_penetration_upper_bound_mm':upper*1000,'no_penetration_certificate_for_swept_radius':upper<=self.tol,'centerline_outside_count':int(np.count_nonzero(signed>self.tol)),'sampled_surface_outside_count':int(np.count_nonzero(d>self.tol)),'minimum_centerline_clearance_mm':-float(np.max(signed))*1000,'minimum_sampled_surface_distance_mm':float(np.min(np.abs(d)))*1000,'sampled_curve_m':c.tolist(),'sampled_curve_arc_length_mm':float(arc.sum()*1000),'sampling_spatial_bound_mm':gap*1000,'curvature_proxy':float(np.sum(np.linalg.norm(np.diff(t,axis=0),axis=1))),'mapped_collision_to_dense_curve_max_distance_mm':None if nearest_mapped_distance is None else nearest_mapped_distance*1000,'mapped_curve_consistency': 'PASS' if nearest_mapped_distance is not None and nearest_mapped_distance<=gap+self.tol else 'NOT_VERIFIED','numeric_tolerance_mm':self.tol*1000,'surface_sample_count':len(points),'tip_center_m':np.asarray(raw['dofs'])[-1,:3].tolist(),'scope':'Maximum sampled tube surface witness vs exact final STL, plus conservative swept-ball upper bound; not an absence-of-self-intersections certificate.'}

def main():
    detector=Detector(Path(sys.argv[1]))
    print(json.dumps(jsonable({'ready':detector.report})),flush=True)
    for line in sys.stdin:
        try:
            item=json.loads(line)
            if item.get('stop'): break
            response=detector.measure(item)
        except Exception:
            response={'status':'FAIL','traceback':traceback.format_exc()}
        print(json.dumps(jsonable(response),allow_nan=False),flush=True)
if __name__=='__main__':
    try: main()
    except Exception:
        print(json.dumps({'ready':{'status':'FAIL','traceback':traceback.format_exc()}}),flush=True)
        raise SystemExit(1)
