"""Adaptive challenge observer: reuse validated STL, sign and true Bezier pipeline."""
import sys,json,math,traceback
from pathlib import Path
import numpy as np
from penetration_geometry import Detector,curve_samples
from run_acceptance import jsonable

class AdaptiveDetector(Detector):
    def measure(self,raw):
        c,t,gap=curve_samples(raw); d,inside=self.q.query(c); r=self.radius
        if np.any(((d<0)!=inside)&(np.abs(d)>self.tol)):raise ValueError('Center sign/containment disagreement')
        k=int(np.argmax(d)); nearest=self.q.witness(c[k]); delta=np.asarray(nearest['closest_wall_m'])-c[k]
        # Exact physical radial witness toward the nearest wall, not a radius-subtracted point masquerading as material.
        radial=delta-np.dot(delta,t[k])*t[k]
        if np.linalg.norm(radial)<1e-12:radial=np.cross(t[k],[1,0,0] if abs(t[k,0])<.9 else [0,1,0])
        radial/=np.linalg.norm(radial)
        points=[c[k]+r*radial]
        # Far surfaces are certified by the signed-distance Lipschitz bound. Dense rings only near contact/possible breach.
        near=(-d-r-gap)<=.001; angles=np.arange(64)*2*np.pi/64
        for i in np.flatnonzero(near):
            ref=np.array([0.,0.,1.]) if abs(t[i,2])<.9 else np.array([0.,1.,0.]); u=np.cross(t[i],ref);u/=np.linalg.norm(u);v=np.cross(t[i],u)
            points.extend(c[i]+r*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v))
        for i in (0,len(c)-1):
            if near[i]:
                ref=np.array([0.,0.,1.]) if abs(t[i,2])<.9 else np.array([0.,1.,0.]);u=np.cross(t[i],ref);u/=np.linalg.norm(u);v=np.cross(t[i],u)
                for rr in np.linspace(.00043,r,9):points.extend(c[i]+rr*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v))
        points=np.asarray(points); sd,contained=self.q.query(points)
        if np.any(((sd<0)!=contained)&(np.abs(sd)>self.tol)):raise ValueError('Surface sign/containment disagreement')
        j=int(np.argmax(sd)); witness=self.q.witness(points[j]); error=abs(witness['distance_mm']*.001-abs(sd[j]))
        if error>self.tol*8:raise ValueError('Independent closest triangle disagreement')
        mapped=np.asarray(raw['mapped_collision_positions_m'],float).reshape(-1,3)
        mapped_error=float(np.max(np.min(np.linalg.norm(mapped[:,None,:]-c[None,:,:],axis=2),axis=1)))
        if mapped_error>gap+self.tol:raise ValueError('Mapped collision points do not match current curve within proven sample bound')
        tip=c[-1]; probe=tip+t[-1]*.012; probe_wall=self.q.witness(probe); tip_wall=self.q.witness(tip)
        return {'status':'PASS','geometry_source':raw['geometry_source'],'minimum_entity_gap_lower_bound_mm':float(-np.max(d)-r-gap)*1000,'minimum_entity_gap_sampled_upper_bound_mm':float(-np.max(sd))*1000,'max_penetration_depth_mm':max(0.,float(np.max(sd)))*1000,'max_penetration_upper_bound_mm':max(0.,float(np.max(d)+r+gap))*1000,'material_witness_m':points[j].tolist(),'closest_wall_witness':witness,'nearest_center_m':c[k].tolist(),'nearest_center_wall':nearest,'surface_outside_samples':int(np.count_nonzero(sd>self.tol)),'numeric_tolerance_mm':self.tol*1000,'sample_half_interval_bound_mm':gap*1000,'surface_samples':len(points),'sampled_curve_m':c.tolist(),'tip_tangent_world':t[-1].tolist(),'tip_wall':tip_wall,'forward_probe_wall':probe_wall,'forward_probe_m':probe.tolist(),'mapped_curve_max_error_mm':mapped_error*1000,'mapped_curve_consistency':'PASS','shape_turning_proxy':float(np.linalg.norm(np.diff(t,axis=0),axis=1).sum()),'limitations':['Whole-mesh robust self-intersection certificate remains NOT_VERIFIED.','Gap is an interval: lower bound via swept balls, sampled material witness gives upper bound.','Actual SOFA/file vertex scale precision differs by <=0.000020 mm conservative margin from previous initialization-only measurement.']}

def main():
    detector=AdaptiveDetector(Path(sys.argv[1]));print(json.dumps(jsonable({'ready':detector.report})),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        try:result=detector.measure(request)
        except Exception:result={'status':'FAIL','traceback':traceback.format_exc()}
        print(json.dumps(jsonable(result),allow_nan=False),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        print(json.dumps({'ready':{'status':'FAIL','traceback':traceback.format_exc()}}),flush=True);raise SystemExit(1)
