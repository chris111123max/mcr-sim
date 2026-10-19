"""Frozen-frame numerical attribution: no writes to SOFA states or parameters."""
import sys,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
from mcr_sim import beam_sdf_unilateral_fast as fast
from mcr_sim.vessel_assets import load_signed_distance_grid

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    return x

def savejson(p,x):p.write_text(json.dumps(clean(x),indent=2))
from scipy.spatial.transform import Rotation as R

def main():
    tag=Path(sys.argv[1]) if len(sys.argv)>1 else OUT/'failure'
    m=json.loads(Path(str(tag)+'_metadata.json').read_text())
    z=np.load(Path(str(tag)+'_snapshot.npz'))
    env=SimpleNamespace(sdf_grid=load_signed_distance_grid(m['sdf_path']),
        asset_T_env_sim=np.array(m['asset_T_env_sim']),asset_offset_sim=np.array(m['asset_offset_sim']),
        asset_source_to_sim_scale=m['asset_source_to_sim_scale'],catheter_radius=m['catheter_radius'])
    adapter=fast.LiveBeamSDFAdapter.__new__(fast.LiveBeamSDFAdapter)
    adapter.env=env;adapter._geometry=lambda:m['geometry_free']
    qf=z['q_free'];qc=z['q_committed']; qp=z['q_prev']
    pf,cf=fast._profile(adapter,qf);pc,cc=fast._profile(adapter,qc)
    check={'free_points_max_error_m':float(np.max(abs(pf-z['free_points']))),
           'free_clearance_max_error_m':float(np.max(abs(cf-z['free_clearance']))),
           'committed_points_max_error_m':float(np.max(abs(pc-z['committed_points']))),
           'committed_clearance_max_error_m':float(np.max(abs(cc-z['committed_clearance']))),
           'geometry_changed':m['geometry_free']!=m['geometry_committed']}
    assert max(check[k] for k in check if k.endswith('_m'))<1e-10,check
    sel=z['selected_indices'].astype(int).tolist()
    records,reconstruction=fast._selected_dense_records(adapter=adapter,q_free=qf,dense_points=pf,selected_indices=sel)
    rf=R.from_quat(qf[:,3:7]);rc=R.from_quat(qc[:,3:7])
    dw=(rc*rf.inv()).as_rotvec();dl=(rf.inv()*rc).as_rotvec();dt=qc[:,:3]-qf[:,:3]
    offsets=z['rowOffsets'].astype(int);dofs=z['dofIndices'].astype(int)
    jt=z['linearJacobian'];ja=z['angularJacobian'];gf=z['freeViolations']
    native=[];local=[];world_j_errors=[];converted_j_errors=[];fd_blocks=[]
    for row,rec in enumerate(records):
        nw=0.;nl=0.
        for block in range(offsets[row],offsets[row+1]):
            node=dofs[block]
            nw+=jt[block].dot(dt[node])+ja[block].dot(dw[node])
            nl+=jt[block].dot(dt[node])+ja[block].dot(dl[node])
            jw=[]
            for axis in range(3):
                axisvec=np.zeros(3);axisvec[axis]=fast.ROTATION_FD_RAD
                pairs=[]
                for sign in (1,-1):
                    qq=qf.copy();qq[node,3:7]=(R.from_rotvec(sign*axisvec)*rf[node]).as_quat()
                    pairs.append(fast._element_point(qq,node0=rec['nodes'][0],node1=rec['nodes'][1],length_m=rec['rest_length_m'],t=rec['t']))
                vals=adapter.query(np.array(pairs));jw.append((vals[0]-vals[1])/(2*fast.ROTATION_FD_RAD))
            jw=np.array(jw);converted=rf[node].apply(ja[block])
            world_j_errors.append(float(np.max(abs(jw-ja[block]))))
            converted_j_errors.append(float(np.max(abs(jw-converted))))
            fd_blocks.append({'row':row,'node':int(node),'local_j':ja[block].tolist(),'world_fd_j':jw.tolist(),'rotated_local_j':converted.tolist()})
        native.append(gf[row]+nw);local.append(gf[row]+nl)
    native=np.array(native);local=np.array(local)
    actual=cc[sel]-fast.MARGIN_M
    alphas=[]
    for alpha in (0.0001,0.001,.25,.5,.75,1.):
        qa=qf.copy();qa[:,:3]+=alpha*dt
        qa[:,3:7]=(R.from_rotvec(alpha*dw)*rf).as_quat()
        pa,ca=fast._profile(adapter,qa)
        pred=gf+alpha*(native-gf);predlocal=gf+alpha*(local-gf)
        alphas.append({'alpha':alpha,'normalized_native_error_mm_per_alpha':float(np.max(abs(pred-(ca[sel]-fast.MARGIN_M)))*1000/alpha),'normalized_world_consistent_error_mm_per_alpha':float(np.max(abs(predlocal-(ca[sel]-fast.MARGIN_M)))*1000/alpha),'dense_clearance_mm':float(ca.min()*1000),
            'native_pred_vs_actual_max_error_mm':float(np.max(abs(pred-(ca[sel]-fast.MARGIN_M)))*1000),
            'local_pred_vs_actual_max_error_mm':float(np.max(abs(predlocal-(ca[sel]-fast.MARGIN_M)))*1000)})
    fi=int(np.argmin(cf));ci=int(np.argmin(cc))
    minima=fast._local_minima(cf)
    omitted=[i for i in minima if cf[i]<=fast.MARGIN_M+fast.ACTIVE_BAND_M and i not in sel]
    result={'metadata':m,'profile_reproduction':check,'selected_indices':sel,'selected_records':records,
        'linear_predicted_native_gap_mm':(native*1000).tolist(),
        'linear_predicted_local_gap_mm':(local*1000).tolist(),
        'selected_true_clearance_mm':(cc[sel]*1000).tolist(),
        'native_pred_vs_actual_max_error_mm':float(np.max(abs(native-actual))*1000),
        'local_pred_vs_actual_max_error_mm':float(np.max(abs(local-actual))*1000),
        'all_linear_native_rows_satisfied':bool(np.min(native)>=-1e-6),
        'all_selected_nonlinear_nonpenetrating':bool(np.min(cc[sel])>=-1e-6),
        'all_selected_nonlinear_meet_margin':bool(np.min(cc[sel])>=fast.MARGIN_M-1e-6),
        'free_worst_index':fi,'committed_worst_index':ci,'committed_worst_selected':ci in sel,
        'min_migrated':fi!=ci,'free_worst_xyz':pf[fi].tolist(),'committed_worst_xyz':pc[ci].tolist(),
        'worst_point_nearest_selected_distance_mm':float(np.min(np.linalg.norm(pc[sel]-pc[ci],axis=1))*1000),
        'omitted_active_band_local_minima':omitted,
        'omitted_minima_free_clearance_mm':(cf[omitted]*1000).tolist(),
        'angular_fd_world_vs_supplied_max_abs_m_per_rad':max(world_j_errors),
        'angular_fd_world_vs_rotated_local_max_abs_m_per_rad':max(converted_j_errors),
        'angular_fd_blocks':fd_blocks,'alpha_scan':alphas,
        'max_correction_translation_mm':float(np.max(np.linalg.norm(dt,axis=1))*1000),
        'max_correction_rotation_deg':float(np.max(np.linalg.norm(dw,axis=1))*180/np.pi),
        'solver_observability':m['solver_fields'],
        'no_sofa_state_writes':True,'production_modified':False}
    savejson(Path(str(tag)+'_attribution.json'),result)
    print('RESULT',json.dumps(clean({k:v for k,v in result.items() if k not in ('metadata','angular_fd_blocks','selected_records')})),flush=True)

if __name__=='__main__':main()
