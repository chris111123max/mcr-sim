#!/usr/bin/env python3
"""Capture true B02 Beam/CollisionDOFs and policy actions at the known hazard window."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
from mcr_sim.distributed import DistributedPPO
from mcr_sim.mcr_rl_env import EnvType, MCREnv
from mcr_sim.rl_core.base import RenderMode
ROOT=Path(__file__).resolve().parents[3]
RESULTS=ROOT/'testing/beam_feasible_domain/_runtime/results'

def arr(data):
    try: return np.asarray(data.array(),dtype=np.float64).copy()
    except Exception: return np.asarray(data.value,dtype=np.float64).copy()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--seed',type=int,default=15204); ap.add_argument('--end-step',type=int,default=776); ap.add_argument('--wall-stiffness',type=float,default=10.0); args=ap.parse_args()
    checkpoint=Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    os.environ['MCR_CONSTRAINT_SOLVER']='generic'; os.environ['MCR_SOFA_DT']='0.005'
    model=DistributedPPO.load(str(checkpoint),device='cpu'); model.policy.set_training_mode(False)
    env=MCREnv(create_scene_kwargs={
        'force_model':'B02','centerline_file':'target_04_centerline.vtk','verbose_scene':False,
        'training_curriculum_enabled':False,'vessel_scale_min':1.0,'vessel_scale_max':1.0,
        'start_window_distance_m':0.0,'target_window_distance_m':0.0,'initial_orientation_max_angle_deg':0.0,
        'sdf_physics_wall_enabled':True,'sdf_wall_stiffness_n_per_m':float(args.wall_stiffness),
        'diagnostic_intersection_method':'local_min_distance','use_vessel_line_point_collision':False,
        'use_vessel_point_collision':False,'use_vessel_line_collision':False,
        'sdf_hard_constraint_construct':False,'sdf_hard_constraint_enabled':False,
        'sdf_unilateral_constraint_construct':True,'sdf_unilateral_constraint_enabled':False},
        env_type=EnvType.AORTIC,render_mode=RenderMode.NONE,max_episode_steps=2048,
        time_step=0.005,frame_skip=1,physics_substeps=2)
    observation,_=env.reset(seed=int(args.seed))
    instrument=env.mcr_controller_sofa.instrument.InstrumentCombined
    beam=instrument.getObject('DOFs'); collision=instrument.getChild('mcr_collis').getObject('CollisionDOFs')
    original=env.sofa_simulation.animate; step=0; substeps={}; action=np.zeros(3,dtype=np.float32); actions=[]; captures=[]
    def traced_animate(root,dt):
        before=arr(beam.position); before_free=arr(beam.free_position)
        result=original(root,dt)
        sub=int(substeps.get(step,0)+1); substeps[step]=sub
        if step in (args.end_step-1,args.end_step):
            captures.append({'rl_step':step,'substep':sub,'dt_s':float(dt),
                'raw_action':action.astype(float).tolist(),
                'smoothed_action':np.asarray(env._last_smoothed_action,dtype=np.float64).reshape(-1).tolist(),
                'requested_insert_mm':float(getattr(env,'current_raw_insert',0.0)*1000.0),
                'effective_insert_mm':float(getattr(env,'current_effective_insert',0.0)*1000.0),
                'inserted_length_m':float(env.mcr_controller_sofa._getXTipValue()),
                'inserted_length_m':float(env.mcr_controller_sofa._getXTipValue()),
                'beam_position_before':before.tolist(),'beam_free_before':before_free.tolist(),
                'beam_position_after':arr(beam.position).tolist(),'beam_free_after':arr(beam.free_position).tolist(),
                'collision_position_after':arr(collision.position).tolist(),
                'collision_free_after':arr(collision.free_position).tolist(),
                'time_after_s':float(root.getTime())})
        return result
    env.sofa_simulation.animate=traced_animate
    started=time.perf_counter(); terminal=None
    try:
        for step in range(1,int(args.end_step)+1):
            action=np.asarray(model.predict(observation,deterministic=True)[0],dtype=np.float32).reshape(3)
            actions.append(action.copy())
            observation,_,terminated,truncated,info=env.step(action)
            if step in (1,100,200,300,400,500,600,700,args.end_step):
                print(f'[B02_CAPTURE] step={step}/{args.end_step}',flush=True)
            if terminated or truncated:
                terminal=str(info.get('terminal_reason') or ('terminated' if terminated else 'truncated')); break
    finally:
        env.sofa_simulation.animate=original
        try: env.close()
        except Exception: pass
    if len(actions)!=args.end_step: raise RuntimeError(f'ended at {len(actions)} steps, terminal={terminal}')
    action_array=np.asarray(actions,dtype=np.float32).reshape((-1,3)); action_sha=hashlib.sha256(action_array.tobytes()).hexdigest()
    if len(captures)!=4: raise RuntimeError(f'expected 4 captured physics substeps, got {len(captures)}')
    manifest={'test':'B02 epoch74 target_04 deterministic raw PPO action prefix','checkpoint':str(checkpoint),'seed':int(args.seed),
        'shape':list(action_array.shape),'dtype':str(action_array.dtype),'sha256':action_sha,'action_count':len(actions),
        'action_sequence':action_array.tolist(),'dt_s':0.005,'physics_substeps':2,'rl_control_period_s':0.01,
        'wall_stiffness_n_per_m':float(args.wall_stiffness),'terminal_reason':terminal}
    state={'test':'B02 target_04 actual Beam/CollisionDOF dangerous-window capture','diagnostic_only':True,'training_started':False,
        'checkpoint':str(checkpoint),'seed':int(args.seed),'action_sha256':action_sha,'capture_end_step':int(args.end_step),
        'wall_s':time.perf_counter()-started,'physics_dt_s':0.005,'physics_substeps':2,'rl_control_period_s':0.01,
        'vessel_collision':'Triangle-only; vessel Point/Line OFF','unilateral_constructed':False,'hard_constraint_constructed':False,
        'captures':captures}
    RESULTS.mkdir(parents=True,exist_ok=True)
    (RESULTS/'beam_b02_action_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    (RESULTS/'beam_b02_dangerous_state.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'CAPTURED','action_sha256':action_sha,'action_count':len(actions),'capture_rows':len(captures),'wall_s':state['wall_s']}))
if __name__=='__main__': main()
