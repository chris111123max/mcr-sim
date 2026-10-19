"""Test-only one-time local-to-world row conversion and independent world-axis FD."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from mcr_sim import beam_sdf_unilateral_fast as fast

def convert_rows(rows,q):
    out=copy.deepcopy(rows)
    ids=np.asarray(rows['dof_indices'],dtype=int)
    local=np.asarray(rows['angular_jacobian'],dtype=float).reshape((-1,3))
    if rows.get('angular_frame')=='world': raise RuntimeError('DOUBLE_WORLD_CONVERSION')
    out['angular_jacobian']=Rotation.from_quat(q[ids,3:7]).apply(local).tolist() if len(ids) else []
    out['angular_frame']='world'
    return out

def world_fd_point(q,record,node,axis,amount,rotation):
    nodes=record['nodes']; pair=np.array(q[nodes],copy=True); k=nodes.index(int(node))
    if rotation:
        delta=np.zeros(3);delta[axis]=amount
        pair[k,3:7]=(Rotation.from_rotvec(delta)*Rotation.from_quat(pair[k,3:7])).as_quat()
    else: pair[k,axis]+=amount
    return fast._element_point(pair,node0=0,node1=1,length_m=record['rest_length_m'],t=record['t'])

def verify_fd(adapter,q,rows):
    points=adapter.sample(q,spacing_m=1e-5)
    selected=rows['selected_dense_indices']
    records,meta=fast._selected_dense_records(adapter=adapter,q_free=q,dense_points=points,selected_indices=selected)
    ids=np.asarray(rows['dof_indices']);off=rows['row_offsets'];query=[];slots=[]
    for row,rec in enumerate(records):
        for k in range(off[row],off[row+1]):
            for rot,h in [(False,fast.TRANSLATION_FD_M),(True,fast.ROTATION_FD_RAD)]:
                for axis in range(3):
                    slots.append((k,rot,axis,h))
                    for sign in [-1,1]:query.append(world_fd_point(q,rec,int(ids[k]),axis,sign*h,rot))
    values=adapter.query(np.asarray(query))
    ja=np.zeros((len(ids),3));jt=np.zeros_like(ja)
    for i,(k,rot,axis,h) in enumerate(slots):
        (ja if rot else jt)[k,axis]=(values[2*i+1]-values[2*i])/(2*h)
    corrected=convert_rows(rows,q);local=np.asarray(rows['angular_jacobian']);world=np.asarray(corrected['angular_jacobian'])
    original_error=np.abs(local-ja);corrected_error=np.abs(world-ja)
    trans_error=np.abs(np.asarray(rows['linear_jacobian'])-jt)
    blocks=[]
    for row in range(len(selected)):
        for k in range(off[row],off[row+1]):
            r=Rotation.from_quat(q[int(ids[k]),3:7]);mat=r.as_matrix()
            assert np.max(np.abs(world[k]-local[k]@mat.T))<1e-15
            blocks.append(dict(row=row,dense_index=int(selected[row]),dof=int(ids[k]),quaternion_xyzw=q[int(ids[k]),3:7].tolist(),R_local_to_world=mat.tolist(),local_j=local[k].tolist(),world_j=world[k].tolist(),independent_world_fd=ja[k].tolist(),translation_fd=jt[k].tolist()))
    return dict(**{'pass':bool(corrected_error.max()<1e-8 and trans_error.max()<1e-8)},
        original_max_error=float(original_error.max()),original_mean_error=float(original_error.mean()),
        corrected_max_error=float(corrected_error.max()),corrected_mean_error=float(corrected_error.mean()),
        translation_max_error=float(trans_error.max()),translation_unchanged=bool(np.array_equal(rows['linear_jacobian'],corrected['linear_jacobian'])),
        quaternion_convention='xyzw; active local-to-world rotation; SOFA increment exp(world)*R',
        fd_translation_m=fast.TRANSLATION_FD_M,fd_rotation_rad=fast.ROTATION_FD_RAD,blocks=blocks,
        angular_abs_gate=1e-8,oracle='Independent left-multiplied quaternion perturbations and fresh Beam/SDF queries; not rotated local gradients')
