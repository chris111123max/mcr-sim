"""Read-only analysis of completed real adaptive contact measurements."""
import json,sys
from pathlib import Path
import numpy as np

def main(directory):
    out=Path(directory); report=json.loads((out/'report.json').read_text())
    rows=[json.loads(s) for s in (out/'physical_substeps.jsonl').read_text().splitlines()]
    valid=[r for r in rows if 'wall' in r]
    if not valid:raise ValueError('No valid geometric records; no zero-depth inference')
    contacts=[r for r in valid if r['constraint_contact_evidence']]
    event=next((e for e in report['events'] if e['type']=='CONTACT_REACHED'),None)
    loaded=[] if event is None else [r for r in valid if r['physical_substep']>event['physical_substep']]
    first=contacts[0] if contacts else None
    def phase(group):
        if not group:return None
        return {'substeps':len(group),'insertion_delta_mm':sum(r['actual_insertion_delta_m'] for r in group)*1000,'tip_path_length_mm':sum(np.linalg.norm(r['tip_displacement_m']) for r in group)*1000,'sum_tip_normal_motion_mm':sum(r['tip_normal_motion_mm'] for r in group),'sum_tip_tangential_motion_mm':sum(r['tip_tangential_motion_mm'] for r in group),'shape_turning_proxy_start':group[0]['wall']['shape_turning_proxy'],'shape_turning_proxy_end':group[-1]['wall']['shape_turning_proxy'],'gap_lower_min_mm':min(r['wall']['minimum_entity_gap_lower_bound_mm'] for r in group),'sampled_gap_upper_min_mm':min(r['wall']['minimum_entity_gap_sampled_upper_bound_mm'] for r in group),'max_sampled_entity_depth_mm':max(r['wall']['max_penetration_depth_mm'] for r in group),'max_continuous_entity_depth_upper_mm':max(r['wall']['max_penetration_upper_bound_mm'] for r in group)}
    positive=[r for r in valid if r['wall']['max_penetration_depth_mm']>r['wall']['numeric_tolerance_mm']]
    peak=max(valid,key=lambda r:r['wall']['max_penetration_depth_mm'])
    mu=sorted({float(o['data']['mu']) for r in contacts for o in r['observed_contact_constraint_objects'] if o['class']=='FrictionContact' and 'mu' in o['data']})
    result={'status':report['status'],'source_report':str((out/'report.json').resolve()),'contact_status':report['contact_status'],'contact_definition':'Actual SOFA wall/catheter constraint activation within configured proximity envelope; NOT a claim of zero material gap.','first_constraint_substep':None if first is None else first['physical_substep'],'contact_confirmation':event,'physics_substeps':report['physics_substeps'],'valid_geometry_substeps':len(valid),'constraint_active_substeps':len(contacts),'search_candidates':report['search_candidates'],'physical_candidate_rollouts':0,'wall_seconds':report['wall_seconds'],'stop_reason':report.get('stop_reason'),'overall':phase(valid),'controlled_contact_loading':phase(loaded),'penetration_detected':bool(positive),'penetration_event':None if not positive else {k:peak[k] for k in ('physical_substep','rl_action','action')},'constraint_classes':sorted({o['class'] for r in contacts for o in r['observed_contact_constraint_objects']}),'friction_contact_object_mu':mu,'lcp_solver_mu':report['final_solver']['lcp']['data']['mu'],'friction_note':'Distinct runtime Data: FrictionContact mu and LCP mu are reported separately, no effective friction equivalence assumed. No parameter modified.','contact_force_count_iterations':'NOT_OBSERVED','state_41_all_completed_actions_finite':all(s['observation_shape']==[41] and s['observation_finite'] for s in report['steps']),'numerical_errors':[r['physical_substep'] for r in rows if r.get('simulation_error') or r.get('measurement_or_state_error')],'sofa_logs':report.get('final_sofa_logs'),'preservation':report['preservation'],'mesh_self_intersection_certificate':'NOT_VERIFIED','limitations':['One vessel, one wall region and short controlled loading do not certify general anti-penetration capability.','No direct contact reaction force measurement or constraint-disabled control; changes in shape/motion are observed responses, not isolated causal estimates.','Geometry is an interval, actual/file mesh uncertainty remains approximately 0.000020 mm.']}
    worst_bound=max(valid,key=lambda r:r['wall']['max_penetration_upper_bound_mm'])
    result['unresolved_upper_bound_event']={k:worst_bound[k] for k in ('physical_substep','rl_action','action')}
    result['unresolved_upper_bound_event'].update({'depth_upper_mm':worst_bound['wall']['max_penetration_upper_bound_mm'],'sampled_depth_mm':worst_bound['wall']['max_penetration_depth_mm'],'material_witness_m':worst_bound['wall']['material_witness_m']})
    result['actions']=[{'rl_action':i,'action':g[-1]['action'],'actual_substeps':len(g),'insertion_end_mm':g[-1]['inserted_length_m']*1000,'gap_lower_min_mm':min(r['wall']['minimum_entity_gap_lower_bound_mm'] for r in g),'gap_sampled_upper_min_mm':min(r['wall']['minimum_entity_gap_sampled_upper_bound_mm'] for r in g),'constraint_substeps':sum(r['constraint_contact_evidence'] for r in g),'measured_depth_mm':max(r['wall']['max_penetration_depth_mm'] for r in g)} for i in sorted({r['rl_action'] for r in valid}) for g in [[r for r in valid if r['rl_action']==i]]]
    result['evidence_assessment']='FINITE_LOAD_EFFECTIVE_BEHAVIOR_OBSERVED_NOT_GENERAL_PASS' if contacts and loaded and not positive and result['overall']['gap_lower_min_mm']>0 and not result['numerical_errors'] else 'NOT_VERIFIED'
    (out/'interpretation.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Adaptive SOFA wall-contact challenge','','Status: '+result['status'],'','This is finite-load evidence, not a general anti-penetration PASS.','',json.dumps(result,indent=2),'','Full catheter geometry, tip trajectory, field, insertion, constraints and LCP graph per physical substep: physical_substeps.jsonl.','Candidate scores: search_*.json. Protected source/asset hashes: protected_sha256.json.']
    (out/'interpretation.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':main(sys.argv[1])
