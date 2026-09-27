#!/usr/bin/env python3
"""Deterministic mechanism regressions for the isolated Beam feasible-domain PoC."""
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import beam_feasible_poc as poc

ROOT=HERE.parent
RESULTS=ROOT/"_runtime/results"
RESULTS.mkdir(parents=True,exist_ok=True)


def dump(name,data):
    path=RESULTS/name
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    print(f"{name}: {'PASS' if data.get('pass') else 'FAIL'} -> {path}")


def dense_min(q, count=2049):
    rows=[]
    for k in range(2):
        c=poc.controls(q,k)
        for t in np.linspace(0,1,count):
            x=poc.bezier(c,float(t))
            rows.append((float(x[2]-poc.R),k,float(t),x.tolist()))
    return min(rows,key=lambda x:x[0]),len(rows)


def main():
    sofa_root,mo,base=poc.make_scene()
    zero=np.zeros(12)
    try:
        # A: an explicit, deterministic free proposal through z=0.
        crossing=np.zeros(12);crossing[8]=poc.DRIVE*poc.DT
        free=poc.poses(base,crossing);free_cert=poc.cert(free)
        a_min,a_n=dense_min(free)
        a={'test':'A_free_crossing','geometry':'same two-element BeamAdapter cubic Bezier test rod; z=0 plane',
           'initial_dofs':base.tolist(),'initial_min_clearance_m':poc.cert(base)['min_sampled_clearance_m'],
           'proposed_dof_delta':crossing.tolist(),'free_proposal_min_clearance_m':a_min[0],
           'free_proposal_worst_element':a_min[1],'free_proposal_worst_t':a_min[2],
           'free_proposal_worst_point_m':a_min[3],'dense_samples':a_n,
           'adaptive_certificate_feasible':free_cert['feasible'],'pass':bool(a_min[0]<0)}
        dump('beam_test_A_free_crossing.json',a)

        # B and C share A's exact base state and proposal.
        tic=time.perf_counter_ns();accepted,out=poc.solve_step(base,zero,free_delta=crossing);elapsed=time.perf_counter_ns()-tic
        accepted_q=poc.poses(base,accepted);accepted_cert=poc.cert(accepted_q);b_min,b_n=dense_min(accepted_q)
        b={'test':'B_single_step_feasible','same_initial_dofs_as_A':bool(np.array_equal(base,np.asarray(a['initial_dofs']))),
           'same_proposed_dof_delta_as_A':bool(np.array_equal(crossing,np.asarray(a['proposed_dof_delta']))),
           'free_proposal_min_clearance_m':a_min[0],'accepted_min_clearance_m':b_min[0],
           'accepted_certified_clearance_lower_bound_m':accepted_cert['min_certified_clearance_m'],
           'active_constraint_rows_max':out['active_constraints_max'],'nonlinear_iterations':out['nonlinear_iterations'],
           'relinearization_count':out['relinearization_count'],'line_search_candidate_count':len(out['line_search_trace']),
           'accepted_alpha':out['line_search_alpha_min'],'adaptive_subdivisions':accepted_cert['subdivisions'],
           'max_subdivision_depth':accepted_cert['max_subdivision_depth'],'max_constraint_violation_m':max(0,-b_min[0]),
           'beam_dof_correction_m':out['correction_m'],'finite':out['finite'],'runtime_ns':elapsed,
           'dense_samples':b_n,'pass':bool(a_min[0]<0 and b_min[0]>=-1e-6 and accepted_cert['feasible'] and out['finite'])}
        dump('beam_test_B_single_step.json',b)

        # C: inspect every line-search candidate actually evaluated in B.
        ls=out['line_search_trace'];accepted_ls=[x for x in ls if x['accepted']]
        first_full=next((x for x in ls if x['iteration']==1 and x['alpha']==1.0),None)
        c={'test':'C_line_search','candidate_trace':ls,'first_full_step_candidate':first_full,
           'accepted_alpha_by_iteration':[x['alpha'] for x in accepted_ls],
           'pass':bool(first_full is not None and not first_full['accepted'] and accepted_ls and min(x['alpha'] for x in accepted_ls)<1.0 and accepted_cert['feasible'])}
        dump('beam_test_C_line_search.json',c)

        # D: relinearization is demonstrated only if active geometry/J changes across iterations.
        tr=out['trace'];set_changed=any(x['active_sample_indices']!=tr[0]['active_sample_indices'] for x in tr[1:])
        jac_changed=any(not np.isclose(x['jacobian_frobenius_norm'],tr[0]['jacobian_frobenius_norm'],rtol=1e-8,atol=1e-12) for x in tr[1:])
        loc_changed=any(not np.allclose(x['active_sample_positions_m'],tr[0]['active_sample_positions_m'],rtol=0,atol=1e-10) for x in tr[1:] if len(x['active_sample_positions_m'])==len(tr[0]['active_sample_positions_m']))
        d={'test':'D_nonlinear_relinearization','iterations':tr,'active_set_changed':set_changed,
           'constraint_location_changed':loc_changed,'jacobian_norm_changed':jac_changed,
           'normal_changed':False,'pass':bool(len(tr)>1 and (set_changed or jac_changed or loc_changed))}
        dump('beam_test_D_relinearization.json',d)

        # E: endpoints clear, interior violates; dense sampling independently identifies the trap.
        q=base.copy();q[0,3:7]=Rotation.from_euler('y',90,degrees=True).as_quat();q[1,3:7]=Rotation.from_euler('y',-90,degrees=True).as_quat()
        c0=poc.controls(q,0);end_a=float(c0[0,2]-poc.R);end_b=float(c0[-1,2]-poc.R);mid=float(poc.bezier(c0,.5)[2]-poc.R)
        ev,ec=dense_min(q);ecert=poc.cert(q)
        e={'test':'E_midpoint_trap','endpoint_a_clearance_m':end_a,'endpoint_b_clearance_m':end_b,
           'true_midpoint_clearance_m':mid,'dense_min_clearance_m':ev[0],'worst_element':ev[1],
           'worst_t':ev[2],'worst_point_m':ev[3],'dense_sample_count':ec,
           'adaptive_subdivision_count':ecert['subdivisions'],'unresolved_segments':ecert['unresolved_segments'],
           'max_subdivision_depth':ecert['max_subdivision_depth'],'max_unresolved_segment_m':ecert['max_unresolved_segment_m'],
           'certificate_rejected':not ecert['feasible'],
           'pass':bool(end_a>=0 and end_b>=0 and mid<0 and ev[0]<0 and not ecert['feasible'] and ecert['subdivisions']>0)}
        dump('beam_test_E_midpoint_trap.json',e)

        # F: safe far-field free proposal; expose both near-active samples and actual optimizer rows.
        far=base.copy();far[:,2]+=.010
        far_delta=np.zeros(12);far_delta[2]=-.002;far_delta[8]=-.002
        far_free=poc.poses(far,far_delta);far_free_c=poc.cert(far_free)
        far_knots=[(k,float(t)) for k in range(2) for t in np.linspace(0,1,17)]
        far_g=poc.sample_g(far,np.zeros(12),far_knots);near_active=int(np.sum(far_g<.0005))
        far_u,far_out=poc.solve_step(far,np.zeros(12),free_delta=far_delta);far_q=poc.poses(far,far_u)
        far_diff=float(np.linalg.norm(far_q-far_free));far_rows=far_out['active_constraints_max']
        f={'test':'F_no_false_constraint','free_proposal_min_clearance_m':far_free_c['min_sampled_clearance_m'],
           'near_active_sample_count':near_active,'optimizer_constraint_rows_max':far_rows,
           'accepted_alpha':far_out['line_search_alpha_min'],'accepted_free_state_dof_difference':far_diff,
           'accepted_min_clearance_m':poc.cert(far_q)['min_sampled_clearance_m'],
           'fallback_minimum_row_used':bool(near_active==0 and far_rows>0),
           'pass':bool(far_free_c['feasible'] and near_active==0 and far_out['line_search_alpha_min']==1.0 and far_diff<1e-8)}
        dump('beam_test_F_no_false_constraint.json',f)

        # G: near-wall tangential distal translation.
        tangent=base.copy();tangent[:,2]+=0.0001
        tangent_delta=np.zeros(12);tangent_delta[0]=0.001;tangent_delta[6]=0.001
        tq_free=poc.poses(tangent,tangent_delta);tc_free=poc.cert(tq_free)
        tu,tout=poc.solve_step(tangent,np.zeros(12),free_delta=tangent_delta);tq=poc.poses(tangent,tu)
        free_tan=float(tq_free[2,0]-tangent[2,0]);accepted_tan=float(tq[2,0]-tangent[2,0])
        free_norm=float(tq_free[2,2]-tangent[2,2]);accepted_norm=float(tq[2,2]-tangent[2,2])
        ratio=accepted_tan/free_tan if abs(free_tan)>1e-12 else 0.0
        g={'test':'G_near_boundary_tangential_motion','free_proposal_min_clearance_m':tc_free['min_sampled_clearance_m'],
           'free_displacement_m':[free_tan,free_norm],'accepted_displacement_m':[accepted_tan,accepted_norm],
           'tangential_retention_ratio':ratio,'active_constraint_rows_max':tout['active_constraints_max'],
           'accepted_alpha':tout['line_search_alpha_min'],'accepted_min_clearance_m':poc.cert(tq)['min_sampled_clearance_m'],
           'pass':bool(tc_free['feasible'] and poc.cert(tq)['min_sampled_clearance_m']>=-1e-6 and ratio>0.9)}
        dump('beam_test_G_tangential_motion.json',g)

        # H: 100 non-resetting contacts under the same persistent free-motion push.
        u=np.zeros(12);rows=[]
        for step in range(1,101):
            tic=time.perf_counter_ns();u,hout=poc.solve_step(base,u);ns=time.perf_counter_ns()-tic
            fq=poc.poses(base,u.copy());ac=poc.cert(fq)
            free_state=u.copy();free_state[8]+=poc.DRIVE*poc.DT;fc=poc.cert(poc.poses(base,free_state))
            rows.append({'step':step,'free_proposal_min_clearance_m':fc['min_sampled_clearance_m'],
                         'accepted_min_clearance_m':ac['min_sampled_clearance_m'],'active_constraints':hout['active_constraints_max'],
                         'nonlinear_iterations':hout['nonlinear_iterations'],'relinearization_count':hout['relinearization_count'],
                         'line_search_iterations':len(hout['line_search_trace']),'accepted_alpha':hout['line_search_alpha_min'],
                         'adaptive_subdivisions':ac['subdivisions'],'max_subdivision_depth':ac['max_subdivision_depth'],
                         'max_constraint_violation_m':max(0,-ac['min_sampled_clearance_m']),
                         'finite':bool(np.isfinite(u).all()),'runtime_ns':ns})
            if not rows[-1]['finite'] or not ac['feasible']:
                break
        max_pen=max(x['max_constraint_violation_m'] for x in rows)
        nonfinite_count=sum(not x['finite'] for x in rows)
        hs={'steps':len(rows),'max_accepted_penetration_m':max_pen,
            'alpha_zero_count':sum(x['accepted_alpha']<=1e-12 for x in rows),
            'alpha_lt_0_5_count':sum(x['accepted_alpha']<.5 for x in rows),
            'max_nonlinear_iterations':max(x['nonlinear_iterations'] for x in rows),
            'nonfinite_count':nonfinite_count,
            'max_active_constraints':max(x['active_constraints'] for x in rows),
            'pass':bool(len(rows)==100 and max_pen<=1e-6 and nonfinite_count==0)}
        h={'test':'H_repeated_contact','pass':hs['pass'],'summary':hs,'records':rows}
        dump('beam_test_H_repeated_contact.json',h)
    finally:
        import Sofa
        Sofa.Simulation.unload(sofa_root)

if __name__=='__main__':main()
