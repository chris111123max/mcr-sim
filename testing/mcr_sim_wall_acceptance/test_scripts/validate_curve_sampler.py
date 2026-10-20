"""Pure synthetic calibration of the test-only Bezier sampler, no SOFA."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from penetration_geometry import curve_samples,calibration

def main():
    data={'dofs':[[0,0,0,0,0,0,1],[.01,0,0,0,0,0,1]],'edges':[[0,1]],'lengths':[.01],'first':0,'aligned':False,'bezier_control_points_m':[[[0,0,0],[.01/3,0,0],[.02/3,0,0],[.01,0,0]]]}
    c,t,b=curve_samples(data); linear_error=float(np.max(np.abs(c[:,1:]))); endpoint_error=float(np.linalg.norm(c[-1]-[.01,0,0]))
    rotation=Rotation.from_euler('xyz',[31,-47,63],degrees=True).as_matrix(); shift=np.array([.023,-.017,.04])
    rotated=dict(data); rotated['bezier_control_points_m']=((np.asarray(data['bezier_control_points_m'])@rotation.T)+shift).tolist()
    rc,rt,rb=curve_samples(rotated); rigid_error=float(np.max(np.linalg.norm(rc-(np.linspace(0,.01,len(rc))[:,None]*rotation[:,0]+shift),axis=1)))
    curved=dict(data); curved['bezier_control_points_m']=[[[0,0,0],[0,.005,0],[.01,.005,0],[.01,0,0]]]
    cc,ct,cb=curve_samples(curved); curved_nonzero=bool(np.max(cc[:,1])>.0037)
    mesh=calibration(); passed=mesh['status']=='PASS' and linear_error<1e-14 and endpoint_error<1e-14 and rigid_error<1e-14 and curved_nonzero and max(b,rb,cb)<=.00005+1e-14
    report={'status':'PASS' if passed else 'FAIL','distance_calibration':mesh,'straight_curve_error_mm':linear_error*1000,'endpoint_error_mm':endpoint_error*1000,'rotation_translation_error_mm':rigid_error*1000,'curved_interior_sampled':curved_nonzero,'largest_half_interval_spatial_bound_mm':max(b,rb,cb)*1000,'scope':'Synthetic mathematical geometry only; not a SOFA physical test.'}
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n'); print(report['status']); return 0 if passed else 1
if __name__=='__main__': raise SystemExit(main())
