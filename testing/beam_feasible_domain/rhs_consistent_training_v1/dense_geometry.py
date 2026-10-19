import numpy as np
from scipy.spatial.transform import Rotation
def dense_check(env,q):
 adapter=env._beam_adapter;geometry=adapter._geometry();parts=[];tags=[];locations=[];bounds=[]
 elements=geometry['active_elements'];tip_first=max(0,len(elements)-int(getattr(env.mcr_controller_sofa.instrument,"num_elem_tip",3)))
 for k,e in enumerate(elements):
  a,b=q[np.asarray(e['nodes'],dtype=int)];L=e['rest_length_m'];p0=a[:3];p3=b[:3]
  p1=p0+Rotation.from_quat(a[3:]).apply([L/3,0,0]);p2=p3+Rotation.from_quat(b[3:]).apply([-L/3,0,0])
  linear=np.dot(p1-p0,p2-p1)<0 and np.linalg.norm(p3-p0)<0.4*L
  bound=np.linalg.norm(p3-p0) if linear else 3*max(np.linalg.norm(p1-p0),np.linalg.norm(p2-p1),np.linalg.norm(p3-p2))
  n=max(1,int(np.ceil(max(L,bound)/1e-5)));t=np.linspace(0,1,n+1);u=1-t
  pts=u[:,None]*p0+t[:,None]*p3 if linear else u[:,None]**3*p0+3*(u*u*t)[:,None]*p1+3*(u*t*t)[:,None]*p2+t[:,None]**3*p3
  parts.append(pts);tags.extend([k>=tip_first]*len(pts));locations.extend([(k,float(x)) for x in t]);bounds.append(bound/n)
 points=np.concatenate(parts);clearance=np.asarray(adapter.query(points),dtype=float)
 if not np.isfinite(clearance).all():raise RuntimeError('INDEPENDENT_SDF_NONFINITE_NO_IGNORING')
 from mcr_sim.sdf_hard_constraint import sample_sdf_clearance_and_outward
 _,_,valid=sample_sdf_clearance_and_outward(points,env.sdf_grid,env.asset_T_env_sim,env.asset_offset_sim,env.asset_source_to_sim_scale,env.catheter_radius)
 mask=np.asarray(tags,dtype=bool);worst=int(np.argmin(clearance));i,t=locations[worst]
 return {'clearance_mm':float(clearance.min()*1000),'body_clearance_mm':float(clearance[~mask].min()*1000) if (~mask).any() else None,
  'tip_clearance_mm':float(clearance[mask].min()*1000),'negative_samples':int((clearance<0).sum()),
  'body_negative_samples':int(((clearance<0)&~mask).sum()),'tip_negative_samples':int(((clearance<0)&mask).sum()),
  'sample_count':len(clearance),'invalid_normal_count':int((~np.asarray(valid,dtype=bool)).sum()),
  'negative_invalid_normal_count':int(((clearance<0)&~np.asarray(valid,dtype=bool)).sum()),
  'max_spacing_bound_mm':float(max(bounds)*1000),'covered_active_elements':len(elements),
  'coverage':'all physically deployed Beam elements; inactive tail outside inlet excluded',
  'radius_subtracted_m':float(env.catheter_radius),'worst_element':elements[i],'worst_t':t,'worst_point_m':points[worst].tolist(),
  'strict_nonpenetration':bool(clearance.min()>=0),'aux_margin_0p100':bool(clearance.min()>=1e-4)}
