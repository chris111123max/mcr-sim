#!/usr/bin/env python3
"""Replay the exact stored B02 action prefix and capture real Beam/interpolation state."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import numpy as np
from mcr_sim.mcr_rl_env import EnvType, MCREnv
from mcr_sim.rl_core.base import RenderMode
ROOT=Path(__file__).resolve().parents[3]
RESULTS=ROOT/'testing/beam_feasible_domain/_runtime/results'
DIAG=ROOT.parent/'training_runs/ppo_v15_2b_physics_wall_10Nm_20260922_215738/diagnostics'
REF=DIAG/'v15_2c_sdf_unilateral_targeted_step775_776_seed15204.json'
def arr(d):
    try:return np.asarray(d.array(),dtype=np.float64).copy()
    except Exception:return np.asarray(d.value,dtype=np.float64).copy()
def clean(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,(np.floating,np.integer)): return v.item()
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    try:
        if hasattr(v,'tolist'): return v.tolist()
    except Exception: pass
    try:
        if np.isscalar(v): return v.item() if hasattr(v,'item') else v
    except Exception: pass
    return str(v)
def field(obj,name):
    try:
        d=obj.findData(name)
        try:return clean(d.array())
        except Exception:return clean(d.value)
    except Exception as exc:return {'read_error':repr(exc)}
def main():
    manifest=json.loads((RESULTS/'beam_b02_action_manifest.json').read_text())
    ref=json.loads(REF.read_text())
    ref_sha=ref['A_reference']['action_sha256']
    actions=np.asarray(manifest['action_sequence'],dtype=np.float32).reshape((-1,3))
    sha=hashlib.sha256(actions.tobytes()).hexdigest()
    if sha!=manifest['sha256'] or sha!=ref_sha or len(actions)!=776:
        raise RuntimeError('FAIL: ACTION PREFIX MISMATCH')
    os.environ['MCR_CONSTRAINT_SOLVER']='generic'; os.environ['MCR_SOFA_DT']='0.005'
    env=MCREnv(create_scene_kwargs={'force_model':'B02','centerline_file':'target_04_centerline.vtk','verbose_scene':False,
       'training_curriculum_enabled':False,'vessel_scale_min':1.0,'vessel_scale_max':1.0,
       'start_window_distance_m':0.0,'target_window_distance_m':0.0,'initial_orientation_max_angle_deg':0.0,
       'sdf_physics_wall_enabled':True,'sdf_wall_stiffness_n_per_m':10.0,
       'diagnostic_intersection_method':'local_min_distance','use_vessel_line_point_collision':False,
       'use_vessel_point_collision':False,'use_vessel_line_collision':False,'sdf_hard_constraint_construct':False,
       'sdf_hard_constraint_enabled':False,'sdf_unilateral_constraint_construct':False,'sdf_unilateral_constraint_enabled':False},
       env_type=EnvType.AORTIC,render_mode=RenderMode.NONE,max_episode_steps=2048,time_step=.005,frame_skip=1,physics_substeps=2)
    obs=None; data=None; step_id=0; substeps=0; captured=None
    try:
        obs,_=env.reset(seed=15204)
        inst=env.mcr_controller_sofa.instrument.InstrumentCombined
        beam=inst.getObject('DOFs'); interp=inst.getObject('InterpolGuide'); controller=inst.getObject('m_ircontroller')
        coll=inst.getChild('mcr_collis').getObject('CollisionDOFs')
        topology=inst.getObject('meshLinesCombined')
        original=env.sofa_simulation.animate
        def traced(root,dt):
            nonlocal substeps,captured
            q_prev=arr(beam.position)
            result=original(root,dt)
            substeps+=1
            if step_id==776 and substeps==1:
                q_free=arr(beam.free_position); q_native=arr(beam.position)
                c_free=arr(coll.free_position); c_native=arr(coll.position)
                captured={'rl_step':776,'physics_substep':1,'dt_s':float(dt),
                  'inserted_length_m':float(env.mcr_controller_sofa._getXTipValue()),
                  'raw_action':actions[775].astype(float).tolist(),
                  'q_prev':q_prev,'q_free':q_free,'q_native':q_native,
                  'collision_free_diagnostic_only':c_free,'collision_native_diagnostic_only':c_native,
                  'interpolation':{n:field(interp,n) for n in ('edgeList','lengthList','curvAbsList','DOF0TransformNode0','DOF1TransformNode1','dofsAndBeamsAligned','straight','radius','vecID')},
                  'topology':{n:field(topology,n) for n in ('edges','position','nx','xmin','xmax')},
                  'controller':{n:field(controller,n) for n in ('xtip','CurvAbs','indexFirstNode','instruments','controlledInstrument')},
                  'sdf_metadata':{'path':str(env.sdf_grid.source_path),'source_scalar':env.sdf_grid.scalar_name,
                    'origin_source':env.sdf_grid.origin.tolist(),'spacing_source':env.sdf_grid.spacing.tolist(),
                    'shape_zyx':list(env.sdf_grid.values.shape),'asset_source_to_sim_scale':float(env.asset_source_to_sim_scale),
                    'asset_T_env_sim':np.asarray(env.asset_T_env_sim,dtype=np.float64).reshape(-1).tolist(),
                    'asset_offset_sim':np.asarray(env.asset_offset_sim,dtype=np.float64).reshape(-1).tolist(),
                    'catheter_radius_m':float(env.catheter_radius)}}
            return result
        env.sofa_simulation.animate=traced
        for j,action in enumerate(actions,1):
            step_id=j; substeps=0
            obs,_,terminated,truncated,_=env.step(action)
            if j==776 and (terminated or truncated): raise RuntimeError('episode ended at target step')
            if j in (1,776): print('REPLAY',j,flush=True)
        env.sofa_simulation.animate=original
        if captured is None: raise RuntimeError('target substep capture missing')
        arrays={k:captured.pop(k) for k in ('q_prev','q_free','q_native','collision_free_diagnostic_only','collision_native_diagnostic_only')}
        npz=RESULTS/'b02_step776_states.npz'; np.savez_compressed(npz,**arrays)
        hashes={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()}
        def summary(x):
            x=np.asarray(x); trans=x[:,:3]; quat=x[:,3:7]
            return {'shape':list(x.shape),'translation_min_m':np.min(trans,axis=0).tolist(),'translation_max_m':np.max(trans,axis=0).tolist(),
                    'quaternion_norm_min':float(np.min(np.linalg.norm(quat,axis=1))),'quaternion_norm_max':float(np.max(np.linalg.norm(quat,axis=1)))}
        out={'test':'B02 step776 substep1 real Beam state and BeamAdapter interpolation capture','diagnostic_only':True,
          'seed':15204,'model':'B02','target':'target_04_centerline.vtk','checkpoint':manifest['checkpoint'],
          'action_count':len(actions),'action_shape':list(actions.shape),'action_dtype':str(actions.dtype),'action_sha256':sha,
          'historical_reference_sha256':ref_sha,'action_prefix_matches':sha==ref_sha,'raw_action_step776':captured['raw_action'],
          'rl_step':776,'physics_substep':1,'dt_s':captured['dt_s'],'inserted_length_m':captured['inserted_length_m'],
          'state_npz':str(npz),'state_npz_sha256':hashlib.sha256(npz.read_bytes()).hexdigest(),'state_sha256':hashes,
          'states':{k:summary(arrays[k]) for k in ('q_prev','q_free','q_native')},
          'interpolation':captured['interpolation'],'topology':captured['topology'],'controller':captured['controller'],
          'sdf_metadata':captured['sdf_metadata'],'collision_dofs_recorded_for_diagnostic_only':True,
          'uses_collision_dofs_as_constraint_source':False,'used_native_post_contact_state_as_solver_input':False,
          'production_files_modified':False}
        (RESULTS/'b02_step776_capture.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'capture':'PASS','action_sha256':sha,'qfree_shape':out['states']['q_free']['shape'],'interp_edge_count':len(out['interpolation']['edgeList']),'out':str(RESULTS/'b02_step776_capture.json')}),flush=True)
    finally:
        try:env.close()
        except Exception:pass
if __name__=='__main__':main()
