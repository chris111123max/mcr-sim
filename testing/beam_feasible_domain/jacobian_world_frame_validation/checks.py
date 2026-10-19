"""Independent arithmetic and acceptance gates, without SOFA execution."""
import numpy as np
from scipy.spatial.transform import Rotation

def classify(clearance_mm):
    if not np.isfinite(clearance_mm): return False,False
    return bool(clearance_mm>=-0.001),bool(clearance_mm>=0.100)

def linear_gaps(q0,q,rows):
    ids=rows['dof_indices']; offsets=rows['row_offsets']
    r0=Rotation.from_quat(q0[:,3:7]); r1=Rotation.from_quat(q[:,3:7])
    dr=(r1*r0.inv()).as_rotvec()
    blocks=np.einsum('ij,ij->i',rows['linear_jacobian'],(q[:,:3]-q0[:,:3])[ids])
    blocks+=np.einsum('ij,ij->i',rows['angular_jacobian'],dr[ids])
    gap=rows['free_violations'].copy()
    for row in range(len(gap)): gap[row]+=np.sum(blocks[offsets[row]:offsets[row+1]])
    return gap

def audit_substep(s,passes):
    faults=[]
    if s['native_status']!='PASS_NATIVE_SAME_SUBSTEP_RELINEARIZATION': faults.append('native_status')
    if s['native_solve_calls']!=passes or s['completed_stages']!=passes or len(s['stages'])!=passes: faults.append('solve_count')
    if s['mapping_refresh_count']!=passes-1: faults.append('mapping_count')
    if s['error'] or s['nan_inf']: faults.append('native_error_or_nonfinite')
    if not s['final_stage_to_committed']['pass']: faults.append('final_commit_changed')
    if abs(s['committed_clearance_mm']-s['independent_committed_clearance_mm'])>1e-9: faults.append('dense_disagreement')
    for i,stage in enumerate(s['stages'],1):
        if stage['stage']!=i or stage['native_solve_calls_cumulative']!=i: faults.append('event_sequence')
        if stage['row_count']!=stage['active_count']: faults.append('row_count')
        if not stage['linearization_parent_match']['pass']: faults.append('linearization_parent')
        if stage['solver_tolerance']!=1e-9: faults.append('tolerance_setting')
        if not np.isfinite(stage['solver_error']): faults.append('solver_nonfinite')
        if not np.isfinite(stage['native_solver_wall_ms']) or not np.isfinite(stage['mapping_refresh_wall_ms']):faults.append('timer_nonfinite')
        if stage['native_solver_wall_ms']<=0 or stage['mapping_refresh_wall_ms']<0: faults.append('timer_invalid')
    if abs(sum(t['native_solver_wall_ms'] for t in s['stages'])-s['native_solver_total_wall_ms'])>1e-6:faults.append('native_timer_sum')
    if abs(sum(t['mapping_refresh_wall_ms'] for t in s['stages'])-s['mapping_total_wall_ms'])>1e-6:faults.append('mapping_timer_sum')
    return sorted(set(faults))
