"""Read-only runtime density, mapping and magnetic allocation checks."""
import numpy as np
from mcr_sim.magnetics import nelson_dipole_moment_a_m2

def audit(backend,initial=False):
 i=backend._handles.instrument
 layout=i.layout
 positions=np.asarray(i.MO.position.value,dtype=float)
 assert positions.shape==(299,7) and np.isfinite(positions).all()
 assert list(i.rest_shape.densityOfBeams.value)==[250,48]
 assert list(i.rest_shape.numEdgesCollis.value)==[50,68]
 edges=np.asarray(i.RG.edges.value,dtype=int).reshape(-1,2)
 assert edges.shape==(298,2) and edges.min()==0 and edges.max()==298
 indices=np.asarray(i.CFF.indices.value,dtype=int).reshape(-1)
 assert indices.tolist()==list(layout.magnetic_node_indices)==list(range(293,299))
 weights=np.asarray([x.weight for x in layout.magnet_overlaps])
 overlap=sum(x.overlap_m for x in layout.magnet_overlaps)
 assert abs(weights.sum()-1.)<1e-12 and abs(overlap-.004)<1e-12
 forces=np.asarray(i.CFF.forces.value,dtype=float).reshape(-1,6)
 assert forces.shape==(6,6) and np.isfinite(forces).all() and np.all(forces[:,:3]==0)
 if hasattr(i.CFF,'requested_forces'):assert np.allclose(forces,i.CFF.requested_forces,rtol=1e-10,atol=1e-15)
 controller=backend._handles.torque_controller
 total=controller.last_total_torque_n_m
 if total is not None:assert np.allclose(forces[:,3:].sum(axis=0),total,rtol=1e-10,atol=1e-15)
 moment=nelson_dipole_moment_a_m2(layout.config)
 bound=moment*np.linalg.norm(controller.field_world_t)*weights
 assert np.all(np.linalg.norm(forces[:,3:],axis=1)<=bound+1e-12), dict(forces=forces.tolist(),bound=bound.tolist(),field=np.asarray(controller.field_world_t).tolist(),moment=moment,weights=weights.tolist())
 collision=np.asarray(i.collision_dofs.position.value,dtype=float)
 collision_edges=np.asarray(i.CollisionModel.collisEdgeSet.edges.value,dtype=int).reshape(-1,2)
 assert np.isfinite(collision).all()
 if collision_edges.size:assert collision_edges.min()>=0 and collision_edges.max()<len(collision)
 result=dict(physical_dofs=len(positions),physical_topology_edges=len(edges),collision_points=len(collision),collision_topology_edges=len(collision_edges),torque_finite=True,torque_sum_consistent=total is not None,torque_norm_sum_n_m=float(np.linalg.norm(forces[:,3:],axis=1).sum()))
 if initial:result.update(density=[250,48],collision_density=[50,68],magnetic_node_indices=indices.tolist(),magnet_overlap_m=overlap,magnetic_weights=weights.tolist(),total_dipole_moment_a_m2=moment,scope='Weights sum to one; per-node magnitude bounded by weighted full moment times B. Callback quaternions are not inferred from post-integration poses.')
 h=backend._handles
 solver=h.simulator.lcp_solver
 def scalar(obj,name):
  value=obj.findData(name).value
  return np.asarray(value).reshape(-1)[0].item()
 actual={k:scalar(solver,k) for k in ('tolerance','maxIt','mu','build_lcp')}
 assert actual=={'tolerance':1e-5,'maxIt':2000,'mu':.1,'build_lcp':False},actual
 assert scalar(h.root,'dt')==.006
 vessel_lines=[o for o in h.vessel.CollisionModel.objects if o.getClassName()=='LineCollisionModel']
 assert len(vessel_lines)==1 and bool(scalar(vessel_lines[0],'bothSide'))
 assert h.instrument.line_model.getClassName()=='LineCollisionModel'
 assert h.instrument.point_model.getClassName()=='PointCollisionModel'
 assert scalar(h.instrument.line_model,'proximity')==.000665
 assert scalar(h.instrument.point_model,'proximity')==.000665
 assert len(collision)==119 and len(collision_edges)==118
 backend._messages.require_no_errors('runtime physics audit')
 result.update(status='PASS',lcp=actual,dt_s=scalar(h.root,'dt'),substeps_per_rl=backend.config.sofa.physics_substeps,
  vessel_line_both_side=bool(scalar(vessel_lines[0],'bothSide')),catheter_proximity_m=.000665,
  plugin=h.compatibility,physics_substeps_executed=backend.physics_substeps_executed)
 return result
