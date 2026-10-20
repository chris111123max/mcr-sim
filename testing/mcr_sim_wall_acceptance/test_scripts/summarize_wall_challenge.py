"""Summarize persisted real measurements, never rerun or alter SOFA."""
import json,sys,re,hashlib
from pathlib import Path
import numpy as np

def main(directory):
    directory=Path(directory); rows=[]
    for text in (directory/'physical_substeps.jsonl').read_text().splitlines():
        try: rows.append(json.loads(text))
        except ValueError: pass
    raw_path=directory/'report.json'; raw=json.loads(raw_path.read_text()) if raw_path.exists() else {}
    valid=[r for r in rows if 'wall' in r]
    if not valid: raise ValueError('No valid geometry records; do not invent zero depth')
    def depth(r):return r['wall']['max_sampled_surface_penetration_mm']
    outside=[r for r in valid if depth(r)>r['wall']['numeric_tolerance_mm']]
    worst=max(valid,key=depth); upper=max(r['wall']['conservative_whole_tube_penetration_upper_bound_mm'] for r in valid)
    errors=[r for r in rows if r.get('simulation_error') or r.get('capture_or_numeric_error')]
    def event(r):
        return {**{k:r.get(k) for k in ('physical_substep','rl_step','pattern','action','tip_m','inserted_length_m','consecutive_outside_substeps')},'rl_substep':r['physical_substep']-22*(r['rl_step']-1),'depth_mm':depth(r),'position_mm':None if r['wall']['max_penetration_position_m'] is None else (np.array(r['wall']['max_penetration_position_m'])*1000).tolist(),'nearest_wall':r['wall']['independent_witness']}
    # Python 3.8 uses dict unpacking rather than union below.
    actions=[]
    for index in sorted({r['rl_step'] for r in rows}):
        group=[r for r in valid if r['rl_step']==index]
        if not group: continue
        last=group[-1]; graph=[r.get('solver',{}).get('lcp',{}).get('data',{}).get('graph') for r in group]
        actions.append({'rl_step':index,'pattern':last['pattern'],'action':last['action'],'substeps_recorded':len(group),'inserted_length_end_mm':last['inserted_length_m']*1000,'tip_end_mm':(np.asarray(last['tip_m'])*1000).tolist(),'insertion_executed_mm':sum(r['actual_insertion_delta_m'] for r in group)*1000,'tip_travel_mm':sum(np.linalg.norm(r['tip_displacement_m']) for r in group)*1000,'max_sampled_surface_depth_mm':max(depth(r) for r in group),'min_center_clearance_minus_radius_mm':min(r['wall']['minimum_centerline_clearance_mm']-.665 for r in group),'curvature_proxy_start':group[0]['wall']['curvature_proxy'],'curvature_proxy_end':last['wall']['curvature_proxy'],'lcp_graph_distinct':list(dict.fromkeys(str(g) for g in graph)),'integration_wall_seconds':sum(r['integration_seconds'] for r in group),'measurement_plus_integration_seconds':sum(r['total_record_seconds'] for r in group)})
    near=[r for r in valid if r['wall']['minimum_centerline_clearance_mm']-.665-r['wall']['sampling_spatial_bound_mm']<=.3]
    result={'status':'RUNNING' if not raw_path.exists() else ('GEOMETRIC_EXCEEDANCE_OBSERVED' if outside else 'NOT_VERIFIED'),'finished_report_present':raw_path.exists(),'source_report':str(raw_path.resolve()),'physical_substeps_recorded':len(rows),'valid_geometry_substeps':len(valid),'outside_substeps':len(outside),'max_consecutive_outside_substeps':max(r.get('consecutive_outside_substeps',0) for r in valid),'max_sampled_entity_penetration_mm':max(depth(r) for r in valid),'max_continuous_swept_radius_upper_bound_mm':upper,'first_outside':event(outside[0]) if outside else None,'maximum_event':event(worst) if outside else None,'possible_configured_contact_envelope_first_step':near[0]['physical_substep'] if near else None,'envelope_definition':'center clearance minus 0.665 mm radius minus spatial sample bound <= actual configured 0.3 mm contactDistance; potential contact only, not a measured contact force.','contact_response':'NOT_VERIFIED','lcp_iterations_and_contact_force':'NOT_VERIFIED; see actual raw Data/getter availability, no force/count invented','physical_solver_error_budget':'NOT_VERIFIED; geometry roundoff calibration is not a physical LCP convergence study','mesh_self_intersection_certificate':'NOT_VERIFIED','state_or_measurement_errors':[{'substep':r['physical_substep'],'error':r.get('capture_or_numeric_error'),'simulation_error':r.get('simulation_error')} for r in errors],'sofa_error_messages':raw.get('final_sofa_logs',{}).get('errors'),'all_mapped_collision_curve_checks_pass':all(r['wall'].get('mapped_curve_consistency')=='PASS' for r in valid),'actions':actions,'production_and_assets_preserved':raw.get('preservation')}
    (directory/'interpretation.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# SOFA wall challenge measurements','', 'Status: '+result['status'], '', 'This is geometric STL-relative evidence, not a solver-tolerance waiver or whole-mesh self-intersection certificate.','', '| RL step | Action | Actual substeps | Inserted end mm | Max sampled entity outside mm | Min center clearance - radius mm |','|---:|---|---:|---:|---:|---:|']
    for a in actions:lines.append('| %d | %s | %d | %.6f | %.6f | %.6f |'%(a['rl_step'],a['action'],a['substeps_recorded'],a['inserted_length_end_mm'],a['max_sampled_surface_depth_mm'],a['min_center_clearance_minus_radius_mm']))
    lines+=['', 'First outside: '+str(result['first_outside']),'', 'Maximum outside: '+str(result['maximum_event']),'', 'Continuous swept-radius penetration upper bound across all recorded states (mm): '+str(upper),'', 'Direct contact force/count: NOT_VERIFIED. Actual LCP graphs/parameters are retained in JSONL. Near-wall geometry alone does not prove a reaction force.','', 'Physical solver numerical-error budget and robust full-mesh self-intersection certificate remain NOT_VERIFIED. A small geometric exceedance is not automatically classified as severe physical penetration.','', 'Full raw per-substep shapes, fields, insertion and solver Data: physical_substeps.jsonl. No position correction or projection was used.']
    (directory/'interpretation.md').write_text('\n'.join(lines)+'\n'); print('SUMMARY',result['status'],'substeps',len(rows),'max_sampled_depth_mm',result['max_sampled_entity_penetration_mm']); return result
if __name__=='__main__': main(sys.argv[1])
