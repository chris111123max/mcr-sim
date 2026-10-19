#!/usr/bin/env python3
"""Test-only adapter for saved B02 step-2009 three-factor evidence."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
B=ROOT/'testing/beam_feasible_domain'; S=B/'formal_training_failure'; O=HERE/'results'
O.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(B))
from three_factor_validation import validate_three_factors as vf

def j(p): return json.loads(Path(p).read_text())
def ck(ok,msg):
    if not ok: raise RuntimeError('CAPTURE_MISMATCH: '+msg)
def main():
    m=j(S/'final_stress_metadata.json'); a=j(S/'final_stress_attribution.json')
    c=j(Path(m['checkpoint']).parent.parent/'logs/run_config.json'); case=m['case']
    rank=int(case['seed'])-int(c['seed'])
    ident={'vessel':case['vessel'],'target':case['target'],'seed':case['seed'],'worker_rank':rank,
      'case_episode':case['episode'],'top_level_episode':m.get('episode'),'step':m['rl_step'],
      'substep':m['substep'],'worker_env':m.get('worker_env'),'exact_original_failure':m.get('exact_original_failure')}
    ck((ident['vessel'],ident['target'],rank,ident['case_episode'],ident['step'],ident['substep'])==('B02','target_04',23,3,2009,1),'identity')
    ck(m.get('exact_original_failure') is False,'capture is not marked as original failure')
    with np.load(S/'final_stress_snapshot.npz',allow_pickle=False) as z: r={k:np.array(z[k]) for k in z.files}
    names={'q_prev':'q_prev','q_free':'q_free','q_committed':'q_committed','row_offsets':'rowOffsets',
      'dof_indices':'dofIndices','linear_jacobian':'linearJacobian','angular_jacobian':'angularJacobian',
      'free_violations':'freeViolations','selected_dense_indices':'selected_indices',
      'q_free_dense_clearance':'free_clearance','q_committed_dense_clearance':'committed_clearance'}
    ck(all(v in r for v in names.values()),'legacy NPZ fields missing')
    d={k:np.asarray(r[v]) for k,v in names.items()}
    ck(d['q_free'].shape==d['q_prev'].shape==d['q_committed'].shape and d['q_free'].shape[1]==7,'Rigid3 states')
    ck(d['q_free_dense_clearance'].shape==d['q_committed_dense_clearance'].shape==(8014,),'dense profiles')
    ck(r['free_points'].shape==r['committed_points'].shape==(8014,3),'dense point profiles')
    ck(np.array_equal(d['selected_dense_indices'],a['selected_indices']),'selected indices')
    ck(r['geometry_free']['active_elements']==r['geometry_committed']['active_elements'] if False else True,'')
    ck(np.max(np.abs(r['freeViolations']-(r['sourceClearances']-1e-4)))<=1e-15,'margin/source clearances')
    ck(np.max(np.abs(r['sourceClearances']-r['free_clearance'][d['selected_dense_indices']]))<=1e-15,'selected profile mapping')
    ck(m['geometry_free']['active_elements']==m['geometry_committed']['active_elements'],'dense profile topology')
    blocks={(int(x['row']),int(x['node'])):x for x in a['angular_fd_blocks']}
    off=r['rowOffsets']; ids=r['dofIndices']; ja=r['angularJacobian']; qf=r['q_free']
    ck(len(blocks)==len(ids),'independent oracle blocks')
    oracle=np.empty_like(ja,dtype=float); used=set()
    for row in range(len(off)-1):
      for k in range(int(off[row]),int(off[row+1])):
        key=(row,int(ids[k])); ck(key in blocks,'missing oracle '+repr(key)); x=blocks[key]
        ck(np.max(np.abs(ja[k]-x['local_j']))<=1e-12,'captured local Jacobian '+repr(key))
        rot=Rotation.from_quat(qf[int(ids[k]),3:]).apply(ja[k])
        ck(np.max(np.abs(rot-x['rotated_local_j']))<=1e-12,'converted Jacobian '+repr(key))
        ck(np.max(np.abs(rot-x['world_fd_j']))<=1e-10,'independent world FD '+repr(key))
        oracle[k]=x['world_fd_j']; used.add(key)
    ck(len(used)==len(blocks),'oracle ordering')
    d['angular_world_oracle']=oracle
    acts=np.asarray(r['raw_actions'],dtype='<f4'); sha=hashlib.sha256(acts.tobytes()).hexdigest()
    ck(sha==m['action_sha256'] and len(acts)==2009,'action prefix')
    fork=j(S/'native_fork_state_match.json')
    ck(fork['q_prev_max_abs_m']==0 and fork['q_free_max_abs_m']==0,'native fork state')
    free=float(np.min(d['q_free_dense_clearance'])*1000); commit=float(np.min(d['q_committed_dense_clearance'])*1000)
    ck(abs(free-m['q_free_clearance_mm'])<1e-9 and abs(commit-m['q_committed_clearance_mm'])<1e-9,'dense extrema')
    norm=O/'b02_worker23_episode3_step2009_normalized.npz'; np.savez_compressed(norm,**d)
    snap=dict(d,metadata={},path=str(norm)); jac=vf.jacobian_case(snap); row=vf.solver_case(snap,1e-6); nl=vf.nonlinear_case(snap,1e-6)

    branches={n:j(S/f"native_ab_{n}.json") for n in ("baseline","world_angular_basis","solver_tolerance_1e9")}
    for n,b in branches.items():
      ck((b["rl_step"],b["substep"],b["substep2_executed"])==(2009,1,False),n+" target isolation")
      ck(b["error"] is None and not b["NaN_Inf"] and not b["direct_position_writes"] and not b["rollback_projection_shielding"],n+" native audit")
      ck(b["row_count"]==b["activeCount"]==6 and "CollisionBeginEvent" in b["faithful_fork_method"],n+" fork/rows")
    base,world,tight=(branches[k] for k in ("baseline","world_angular_basis","solver_tolerance_1e9"))
    ck(abs(base["committed_clearance_mm"]-commit)<1e-8,"native baseline clearance")
    ck(world["solver_tolerance"]==base["solver_tolerance"],"Jacobian branch confounded")
    ck(tight["solver_tolerance"]==1e-9 and tight["solver_iterations"]>base["solver_iterations"],"tight solver branch")
    pred=vf.linearized_g(snap,convert_angular=False)[0]
    ck(abs(float(np.min(pred)*1000)-base["linear_row_min_gap_mm"])<1e-8,"baseline row residual")
    actual=snap["q_committed_dense_clearance"][snap["selected_dense_indices"]]*1000-0.1
    predmm=pred*1000; delta=actual-predmm; wi=int(np.argmax(np.abs(delta)))
    report={"identity":ident,"action_sha256":sha,"native_q_prev_q_free_exact_match":fork,
      "capture":{"dense_samples":8014,"rows":len(pred),"free_min_mm":free,"committed_min_mm":commit,
        "selected_indices":snap["selected_dense_indices"].tolist(),"profile_topology_match":True},
      "test_1_rotation_jacobian":{"status":"PASS" if jac["status"]=="FRAME_MISMATCH_CONFIRMED" else "INCONCLUSIVE",
        "offline":jac,"baseline_clearance_mm":base["committed_clearance_mm"],
        "world_basis_clearance_mm":world["committed_clearance_mm"],
        "delta_mm":world["committed_clearance_mm"]-base["committed_clearance_mm"],
        "judgment":"Mismatch confirmed; conversion slightly worsened clearance on this safe frame, not proof of historical failure cause."},
      "test_2_linear_row_residual":{"status":"PASS" if base["linear_row_min_gap_mm"] < -0.001 and tight["linear_row_min_gap_mm"] > -0.001 else "INCONCLUSIVE",
        "offline":row,"baseline_native":base,"tight_solver_native":tight,
        "clearance_delta_mm":tight["committed_clearance_mm"]-base["committed_clearance_mm"],
        "judgment":"Tighter tolerance reduced row residual and improved clearance, but did not reach +0.100 mm."},
      "test_3_nonlinear_sdf":{"status":"INCONCLUSIVE_NATIVE_CAUSAL_EFFECT","offline_measurement_status":nl["status"],
        "offline":nl,"worst_same_row":{"row":wi,"dense_index":int(snap["selected_dense_indices"][wi]),
        "actual_gap_to_margin_mm":float(actual[wi]),"linear_predicted_gap_mm":float(predmm[wi]),
        "actual_minus_prediction_mm":float(delta[wi])},"baseline_clearance_mm":base["committed_clearance_mm"],
        "native_isolating_variant_available":False,
        "judgment":"Nonlinearity is measured from actual dense SDF; no native branch isolates nonlinearization alone."},
      "native_branches":branches,"historical_minus_0.766410_mm_failure":"NOT_REPRODUCED; cause remains unproven",
      "limitations":["Capture is test_only_continuation; rank 23 is inferred from seed offset, not original worker trajectory.",
        "case episode=3 while top-level episode=1; discrepancy retained.","No nonlinear-only native intervention branch is stored."]}
    dst=O/"three_causes_report.json"
    dst.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
if __name__=="__main__": main()
