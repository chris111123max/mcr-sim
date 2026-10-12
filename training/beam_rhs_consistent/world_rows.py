"""World-frame rotation Jacobian conversion used by standalone training."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation

def convert_rows(rows,q):
    out=copy.deepcopy(rows)
    ids=np.asarray(rows['dof_indices'],dtype=int)
    local=np.asarray(rows['angular_jacobian'],dtype=float).reshape((-1,3))
    if rows.get('angular_frame')=='world': raise RuntimeError('DOUBLE_WORLD_CONVERSION')
    out['angular_jacobian']=Rotation.from_quat(q[ids,3:7]).apply(local).tolist() if len(ids) else []
    out['angular_frame']='world'
    return out
