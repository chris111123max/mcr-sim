#!/usr/bin/env python3
"""Measure production SDF clearance on CollisionDOFs captured from live B02 replay."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from mcr_sim.mcr_rl_env import EnvType, MCREnv
from mcr_sim.rl_core.base import RenderMode
ROOT=Path(__file__).resolve().parents[3]
RESULTS=ROOT/'testing/beam_feasible_domain/_runtime/results'
def densify(poly,step):
    out=[]
    for a,b in zip(poly[:-1],poly[1:]):
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/step)))
        out.extend(a+(b-a)*(j/n) for j in range(n))
    if len(poly):out.append(poly[-1])
    return np.asarray(out,dtype=np.float64).reshape((-1,3))
def clearance(points,env,step):
    dense=densify(np.asarray(points,dtype=np.float64)[:,:3],step)
    src=env._sim_points_to_asset_source(dense)
    signed=np.asarray(env.sdf_grid.sample(src),dtype=np.float64)*float(env.asset_source_to_sim_scale)
    sentinel=max(0.05,2.0*float(env.sdf_outside_center_tolerance))
    signed=np.where(np.isfinite(signed),signed,sentinel)
    gap=-signed-float(env.catheter_radius)
    i=int(np.argmin(gap))
    return {'sample_count':int(len(dense)),'min_surface_clearance_mm':float(gap[i]*1000),
            'max_penetration_mm':float(max(0.0,-gap[i]*1000)),
            'worst_sample_sim_m':dense[i].tolist(),'outside_grid_samples':int(np.sum(~np.isfinite(np.asarray(env.sdf_grid.sample(env._sim_points_to_asset_source(dense))))))}
def main():
    actions=json.loads((RESULTS/'beam_b02_action_manifest.json').read_text())
    state=json.loads((RESULTS/'beam_b02_dangerous_state.json').read_text())
    if actions['sha256']!=state['action_sha256'] or len(actions['action_sequence'])!=776:
        raise RuntimeError('action/state manifest mismatch')
    os_env={'force_model':'B02','centerline_file':'target_04_centerline.vtk','verbose_scene':False,
       'training_curriculum_enabled':False,'vessel_scale_min':1.0,'vessel_scale_max':1.0,
       'start_window_distance_m':0.0,'target_window_distance_m':0.0,'initial_orientation_max_angle_deg':0.0,
       'sdf_physics_wall_enabled':True,'sdf_wall_stiffness_n_per_m':10.0,
       'diagnostic_intersection_method':'local_min_distance','use_vessel_line_point_collision':False,
       'use_vessel_point_collision':False,'use_vessel_line_collision':False,
       'sdf_hard_constraint_construct':False,'sdf_hard_constraint_enabled':False,
       'sdf_unilateral_constraint_construct':True,'sdf_unilateral_constraint_enabled':False}
    env=MCREnv(create_scene_kwargs=os_env,env_type=EnvType.AORTIC,render_mode=RenderMode.NONE,
               max_episode_steps=2048,time_step=0.005,frame_skip=1,physics_substeps=2)
    try:
        env.reset(seed=15204)
        spacing=float(np.min(env.sdf_grid.spacing)*env.asset_source_to_sim_scale)
        step=max(1e-6,spacing*0.25)
        rows=[]
        for r in state['captures']:
            free=np.asarray(r['collision_free_after'],dtype=np.float64)
            accepted=np.asarray(r['collision_position_after'],dtype=np.float64)
            bfree=np.asarray(r['beam_free_after'],dtype=np.float64)
            bpos=np.asarray(r['beam_position_after'],dtype=np.float64)
            rows.append({'rl_step':r['rl_step'],'substep':r['substep'],'dt_s':r['dt_s'],
              'raw_action':r['raw_action'],'inserted_length_m':r['inserted_length_m'],
              'collision_dof_count':int(len(free)),
              'mapping_provenance':'CollisionDOFs.free_position/position captured immediately after live Sofa.Simulation.animate in exact PPO replay; no reinjection or updateVisual used',
              'free_proposal':clearance(free,env,step),'post_solver_accepted':clearance(accepted,env,step),
              'max_collision_dof_correction_mm':float(np.max(np.linalg.norm((accepted[:,:3]-free[:,:3])*1000,axis=1))),
              'max_beam_translation_correction_mm':float(np.max(np.linalg.norm((bpos[:,:3]-bfree[:,:3])*1000,axis=1)))})
        out={'test':'B02_target04_step775_776_live_capture_sdf_clearance','diagnostic_only':True,
             'checkpoint':state['checkpoint'],'seed':state['seed'],'action_sha256':actions['sha256'],
             'action_count':len(actions['action_sequence']),'action_hash_matches_capture':actions['sha256']==state['action_sha256'],
             'dt_s':state['physics_dt_s'],'physics_substeps':state['physics_substeps'],
             'sdf_grid':str(env.sdf_grid.source_path),'sdf_spacing_source_mm':env.sdf_grid.spacing.tolist(),
             'sdf_sample_step_mm':step*1000,'clearance_sign':'production convention: -signed_sim - catheter_radius; negative is outside/penetration',
             'rows':rows,'solver_status':'NOT_RUN','reason':'uses production replay-captured CollisionDOFs only; feasibility optimization not attached to SOFA pre-solve path'}
        (RESULTS/'beam_b02_single_danger_step.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'action_hash_matches':out['action_hash_matches_capture'],'rows':[
          {'step':x['rl_step'],'substep':x['substep'],'free_clearance_mm':x['free_proposal']['min_surface_clearance_mm'],
           'accepted_clearance_mm':x['post_solver_accepted']['min_surface_clearance_mm'],
           'collision_correction_mm':x['max_collision_dof_correction_mm'],'beam_correction_mm':x['max_beam_translation_correction_mm']} for x in rows]}))
    finally: env.close()
if __name__=='__main__':main()
