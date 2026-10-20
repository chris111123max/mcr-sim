"""Budgeted geometric-feedback contact-seeking test; only production actions."""
import sys,os,json,time,math,re,hashlib,subprocess,traceback,argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from run_wall_challenge import TEST,PYTHON,geometry,solver_metrics,object_data,dump,digest
from run_acceptance import jsonable,_log_messages

LIMITS={'max_rl_actions':48,'max_physics_substeps':1056,'max_candidates_per_action':75,'max_total_candidates':3600,'max_wall_seconds':1800,'max_insertion_m':.100,'max_post_contact_actions':4,'max_post_contact_advance_m':.001,'max_post_contact_substeps':88,'stop_penetration_mm':.2,'stop_consecutive_positive_depth':5,'stop_tip_jump_per_substep_m':.005,'stagnation_window_actions':6,'stagnation_gap_gain_mm':.05,'stagnation_min_tip_travel_mm':.1,'mesh_coordinate_margin_mm':.000020,'contact_distance_mm':.3}

class StopTest(BaseException):pass

def unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v)
    if n<1e-12:raise ValueError('Undefined geometric direction')
    return v/n

def contact_objects(root):
    found=[]; pending=[root]; visited=0
    while pending and visited<80:
        node=pending.pop();visited+=1;pending.extend(list(node.children))
        for obj in node.objects:
            cls=obj.getClassName()
            if 'UnilateralInteractionConstraint' in cls or cls.startswith('FrictionContact'):
                found.append({'path':obj.getPathName(),'class':cls,'data':object_data(obj)})
    if pending:raise RuntimeError('Contact scene traversal exceeded declared 80-node limit')
    return found

class Policy:
    def __init__(self,controller):self.controller=controller;self.direction=None;self.history=[];self.candidates=0
    def choose(self,state,wall,loading,rl):
        tangent=unit(wall['tip_tangent_world']); probe=np.array(wall['forward_probe_m']); target=np.array(wall['forward_probe_wall']['closest_wall_m']); direction=target-probe;direction-=np.dot(direction,tangent)*tangent
        direction=unit(direction)
        if self.direction is None:self.direction=direction
        elif np.dot(direction,self.direction)>0:self.direction=unit(.9*self.direction+.1*direction)
        desired=unit(self.direction+.15*tangent)
        gap=wall['minimum_entity_gap_lower_bound_mm']
        push=.0625 if loading or gap<=.4 else (.125 if gap<=1 else (.25 if gap<=5 else 1.))
        if rl<=2:push=min(push,.25)
        max_turn=.25 if rl<=2 else (.5 if rl<=4 else 1.)
        mag=[(0.,0.)]+[(rr*max_turn*math.cos(k*math.pi/4),rr*max_turn*math.sin(k*math.pi/4)) for rr in (.25,.5,1.) for k in range(8)]
        candidates=[]
        coef=None
        if len(self.history)>=4:
            recent=self.history[-12:]; X=np.array([[1.,*h['action']] for h in recent]);Y=np.array([h['wallward_tip_motion_mm'] for h in recent]);coef=np.linalg.solve(X.T@X+np.eye(4)*.2,X.T@Y)
        # Search uses production plan() only. No candidate executes SOFA or changes state.
        for a,b in mag:
            for z in (push,push/2,0.):
                action=np.array([a,b,z]); plan=self.controller.plan(action,state.magnetic_field_world_t,state.tip_frame_world)
                align=float(np.dot(unit(plan.target_field_world_t),desired))
                predicted=0. if coef is None else float(np.array([1.,a,b,z])@coef)
                score=5*align+.20*z+.08*math.tanh(predicted)-.01*(a*a+b*b)
                candidates.append({'action':action.tolist(),'score':score,'field_alignment':align,'history_wallward_prediction_mm':predicted})
        self.candidates+=len(candidates)
        best=max(candidates,key=lambda x:x['score'])
        return np.array(best['action'],np.float32),{'candidate_count':len(candidates),'total_candidates':self.candidates,'desired_field_direction':desired.tolist(),'wallward_direction':self.direction.tolist(),'current_tip_m':state.tip_m.tolist(),'tip_tangent_world':tangent.tolist(),'whole_body_gap_lower_mm':gap,'history_size':len(self.history),'selected':best,'candidates':candidates,'physical_candidate_rollouts':0}
    def update(self,action,before,after,wall0,wall1):
        self.history.append({'action':list(action),'wallward_tip_motion_mm':float(np.dot(after.tip_m-before.tip_m,self.direction))*1000,'tip_travel_mm':float(np.linalg.norm(after.tip_m-before.tip_m))*1000,'gap_gain_mm':wall0['minimum_entity_gap_lower_bound_mm']-wall1['minimum_entity_gap_lower_bound_mm']})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-rl-actions',type=int,default=48,choices=range(1,49));args=ap.parse_args()
    limits=dict(LIMITS);limits['max_rl_actions']=args.max_rl_actions;limits['max_physics_substeps']=args.max_rl_actions*22
    calibration=json.loads((TEST/'reports/adaptive_detector_calibration.json').read_text())
    if calibration['status']!='PASS':raise ValueError('Adaptive detector synthetic calibration did not PASS')
    reference=json.loads((PYTHON/'stEVE_anatomical_vessels/tmp_seed101_geometry_r3TeZG/geometry_repair_report.json').read_text())['runs']['refined_03'];vessel=Path(reference['directory'])
    if digest(vessel/'collision_inner.stl')!=reference['sha256']['collision_inner.stl']:raise ValueError('Repaired STL hash mismatch')
    out=TEST/'reports'/('adaptive_contact_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();print('OUTPUT',out,flush=True)
    protected={str(p.resolve()):digest(p) for p in (PYTHON/'mcr_sim').glob('*.py')};protected.update({str(p.resolve()):digest(p) for p in vessel.rglob('*') if p.is_file()});dump(out/'protected_sha256.json',protected)
    report={'status':'NOT_VERIFIED','contact_status':'NOT_REACHED','penetration_status':'NOT_VERIFIED','vessel':str(vessel),'stl_sha256':digest(vessel/'collision_inner.stl'),'limits':limits,'steps':[],'events':[],'search_candidates':0,'physical_candidate_rollouts':0,'physics_substeps':0,'errors':[],'direct_contact_count':'NOT_OBSERVED','direct_contact_force':'NOT_OBSERVED','lcp_iterations':'NOT_OBSERVED','source_sha256':{str(p):digest(p) for p in Path(__file__).parent.glob('*.py')}};dump(out/'predeclared_limits.json',limits)
    started=time.monotonic(); worker=None; backend=None; trace=None; rows=[]; context={}; contact_at=None; contact_length=None; consecutive=0; active_streak=0; wall=None
    def stop(reason):raise StopTest(reason)
    try:
        worker=subprocess.Popen([str(PYTHON/'stEVE_anatomical_vessels/.venv/bin/python'),'-B','-u',str(Path(__file__).with_name('adaptive_wall_geometry.py')),str(vessel)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(out/'geometry.stderr').open('w'),text=True,bufsize=1)
        report['detector']=json.loads(worker.stdout.readline())['ready'];dump(out/'detector_validation.json',report['detector'])
        if report['detector']['status']!='PASS':raise RuntimeError('Detector failed before SOFA')
        from dataclasses import replace
        from mcr_sim.config import DEFAULT_CONFIG
        from mcr_sim.sofa_simulation import SofaBackend
        from mcr_sim.vessel_bundle import load_vessel_bundle
        from mcr_sim.rl_env import MagneticNavigationEnv
        cfg=replace(DEFAULT_CONFIG,vessel_root=vessel.parent);backend=SofaBackend(cfg);env=MagneticNavigationEnv(backend,config=cfg,bundles=[load_vessel_bundle(vessel)])
        obs,info=env.reset(seed=101,options={'vessel_seed':101,'target_branch':'lcca'});policy=Policy(env.action_controller)
        report['reset']={'observation':obs.tolist(),'info':info,'stages':backend.stages,'dt_s':float(backend._root.dt.value),'solver':solver_metrics(backend),'actual_loaded_stl':backend._handles.vessel.environment_stl}
        def measure():
            raw=geometry(backend);worker.stdin.write(json.dumps(raw)+'\n');worker.stdin.flush();w=json.loads(worker.stdout.readline())
            if w['status']!='PASS':raise RuntimeError('Measurement failure: '+str(w))
            return raw,w
        raw,wall=measure();report['initial_wall']=wall;report['initial_contact_constraint_objects']=contact_objects(backend._root);dump(out/'initial_geometry.json',raw)
        trace=(out/'physical_substeps.jsonl').open('w');original=backend.physics_substep
        def observed(field,delta):
            nonlocal wall,contact_at,contact_length,consecutive,active_streak
            if backend.physics_substeps_executed>=limits['max_physics_substeps'] or time.monotonic()-started>=limits['max_wall_seconds']:stop('Hard compute budget reached')
            before=backend.state()
            if contact_at is not None and before.inserted_length_m+float(delta)-contact_length>limits['max_post_contact_advance_m']+1e-12:stop('Next insertion would exceed post-contact advance cap')
            if before.inserted_length_m+float(delta)>limits['max_insertion_m']:stop('Next insertion would exceed 100 mm cap')
            t0=time.monotonic();diagnostic=original(field,delta)
            row={'physical_substep':backend.physics_substeps_executed,**context,'commanded_field_t':np.asarray(field).tolist(),'requested_insertion_delta_m':float(delta),'integration_seconds':time.monotonic()-t0,'simulation_error':diagnostic.simulation_error}
            try:
                state=backend.state();state.validate();raw,w=measure();metrics=solver_metrics(backend);objects=contact_objects(backend._root);logs=_log_messages(backend)
                graph=str(metrics['lcp']['data'].get('graph',''));numbers=re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?',graph)
                near=w['minimum_entity_gap_lower_bound_mm']-limits['mesh_coordinate_margin_mm']<=limits['contact_distance_mm']
                active=bool(objects and numbers and near);active_streak=active_streak+1 if active else 0
                movement=state.tip_m-before.tip_m;normal=np.array(w['closest_wall_witness']['closest_wall_m'])-np.array(w['material_witness_m']);normal=unit(normal) if np.linalg.norm(normal)>1e-12 else np.zeros(3);normal_move=float(np.dot(movement,normal));tangent_move=float(np.linalg.norm(movement-normal_move*normal))
                row.update({'tip_m':state.tip_m.tolist(),'tip_displacement_m':movement.tolist(),'inserted_length_m':state.inserted_length_m,'actual_insertion_delta_m':state.inserted_length_m-before.inserted_length_m,'magnetic_field_t':state.magnetic_field_world_t.tolist(),'geometry':raw,'wall':w,'solver':metrics,'observed_contact_constraint_objects':objects,'lcp_graph_numeric_samples':numbers,'contact_envelope_witness':near,'constraint_contact_evidence':active,'contact_evidence_consecutive_steps':active_streak,'tip_normal_motion_mm':normal_move*1000,'tip_tangential_motion_mm':tangent_move*1000,'sofa_logs':logs})
                if active_streak>=2 and contact_at is None:
                    contact_at=backend.physics_substeps_executed;contact_length=state.inserted_length_m;report['contact_status']='CONTACT_REACHED';report['events'].append({'type':'CONTACT_REACHED','physical_substep':contact_at,'rl_action':context['rl_action'],'evidence':'Actual Unilateral/FrictionContact constraint objects plus numeric LCP graph plus geometric gap interval reaching configured 0.3 mm contact envelope for two substeps; does not assert zero material gap','recorded_material_witness_m':w['material_witness_m']});print('CONTACT_REACHED',contact_at,'gap_bounds',w['minimum_entity_gap_lower_bound_mm'],w['minimum_entity_gap_sampled_upper_bound_mm'],flush=True)
                positive=w['max_penetration_depth_mm']>w['numeric_tolerance_mm'];consecutive=consecutive+1 if positive else 0;row['consecutive_geometric_exceedance_steps']=consecutive
                if positive:
                    report['penetration_status']='PENETRATION_DETECTED'
                    if not any(e['type']=='PENETRATION_DETECTED' for e in report['events']):report['events'].append({'type':'PENETRATION_DETECTED','physical_substep':backend.physics_substeps_executed,'rl_action':context['rl_action'],'action':context['action'],'depth_mm':w['max_penetration_depth_mm'],'position_m':w['material_witness_m']});print('PENETRATION_DETECTED',backend.physics_substeps_executed,w['max_penetration_depth_mm'],flush=True)
                row['record_seconds']=time.monotonic()-t0;rows.append(row);trace.write(json.dumps(jsonable(row),allow_nan=False)+'\n');trace.flush();wall=w
                if diagnostic.simulation_error or not np.all(np.isfinite(np.asarray(backend._handles.dofs.velocity.value))):stop('SOFA numerical failure/nonfinite velocity')
                if np.linalg.norm(movement)>limits['stop_tip_jump_per_substep_m']:stop('Safety stop: >5 mm tip movement in one 0.006 s substep')
                serious=[m for m in logs['warnings'] if re.search('not converg|max.*iter|singular|diverg|NaN',m,re.I)]
                if logs['errors'] or serious:stop('SOFA error or solver failure warning')
                if state.inserted_length_m>limits['max_insertion_m']:stop('100 mm insertion safety cap')
                if w['max_penetration_depth_mm']>=limits['stop_penetration_mm'] or consecutive>=limits['stop_consecutive_positive_depth']:stop('Penetration safety stop; not an acceptance tolerance')
                if contact_at is not None and (backend.physics_substeps_executed-contact_at>=limits['max_post_contact_substeps'] or state.inserted_length_m-contact_length>=limits['max_post_contact_advance_m']):stop('Controlled post-contact loading budget completed')
                return diagnostic
            except StopTest:raise
            except Exception:
                row['measurement_or_state_error']=traceback.format_exc();rows.append(row);trace.write(json.dumps(jsonable(row))+'\n');trace.flush();stop('Invalid state/measurement, no more integration')
        backend.physics_substep=observed
        loading_action_count=0
        try:
            for index in range(1,limits['max_rl_actions']+1):
                before=backend.state();old_wall=wall
                if index==1:action=np.zeros(3,np.float32);search={'candidate_count':0,'total_candidates':0,'reason':'Zero action baseline'}
                else:action,search=policy.choose(before,wall,contact_at is not None,index)
                report['search_candidates']=policy.candidates
                if policy.candidates>limits['max_total_candidates']:stop('Candidate budget reached')
                if contact_at is not None:
                    loading_action_count+=1
                    if loading_action_count>limits['max_post_contact_actions']:stop('Controlled post-contact action budget completed')
                context.update({'rl_action':index,'action':action.tolist(),'phase':'post_contact_load' if contact_at is not None else 'adaptive_approach'})
                dump(out/('search_%02d.json'%index),search);count0=backend.physics_substeps_executed;t0=time.monotonic()
                obs,reward,term,trunc,info=env.step(action)
                after=backend.state();policy.update(action,before,after,old_wall,wall) if index>1 else None
                step={**context,'observation':obs.tolist(),'observation_shape':list(obs.shape),'observation_finite':bool(np.all(np.isfinite(obs))),'reward':reward,'info':info,'substeps':backend.physics_substeps_executed-count0,'seconds':time.monotonic()-t0,'gap_lower_mm':wall['minimum_entity_gap_lower_bound_mm'],'gap_sampled_upper_mm':wall['minimum_entity_gap_sampled_upper_bound_mm'],'tip_m':after.tip_m.tolist(),'insertion_mm':after.inserted_length_m*1000};report['steps'].append(step)
                print('ACTION',index,'a',action.tolist(),'insert_mm',step['insertion_mm'],'gap_mm',step['gap_lower_mm'],step['gap_sampled_upper_mm'],'depth',wall['max_penetration_depth_mm'],flush=True)
                if term or trunc or not info.get('transition_valid') or not step['observation_finite'] or step['substeps']!=22:stop('Production episode ended or invalid transition')
                if index>=10 and contact_at is None and len(policy.history)>=6:
                    h=policy.history[-6:]
                    if sum(x['gap_gain_mm'] for x in h)<limits['stagnation_gap_gain_mm'] and sum(x['tip_travel_mm'] for x in h)<limits['stagnation_min_tip_travel_mm']:stop('History-confirmed approach stagnation')
            report['stop_reason']='RL action budget exhausted'
        except StopTest as exc:report['stop_reason']=str(exc)
        report['status']='PENETRATION_DETECTED' if report['penetration_status']=='PENETRATION_DETECTED' else ('CONTACT_REACHED' if contact_at is not None else 'NOT_REACHED')
        if any('measurement_or_state_error'in r or r.get('simulation_error') for r in rows) or re.search('numerical|nonfinite|Invalid|SOFA error|Safety stop',report.get('stop_reason',''),re.I):report['status']='NOT_VERIFIED'
        valid=[r for r in rows if 'wall'in r]
        report['physics_substeps']=backend.physics_substeps_executed
        if valid:
            peak=max(valid,key=lambda r:r['wall']['max_penetration_depth_mm']);report['maximum_penetration']={k:peak[k] for k in ('physical_substep','rl_action','action','wall')};report['maximum_penetration']['wall']=dict(report['maximum_penetration']['wall']);report['maximum_penetration']['wall'].pop('sampled_curve_m',None)
            report['minimum_gap_lower_mm']=min(r['wall']['minimum_entity_gap_lower_bound_mm'] for r in valid);report['minimum_gap_sampled_upper_mm']=min(r['wall']['minimum_entity_gap_sampled_upper_bound_mm'] for r in valid);report['contact_constraint_substeps']=sum(r['constraint_contact_evidence'] for r in valid);report['post_contact_substeps']=0 if contact_at is None else backend.physics_substeps_executed-contact_at
            report['max_consecutive_geometric_exceedance_steps']=max(r['consecutive_geometric_exceedance_steps'] for r in valid)
        report['anti_penetration_effectiveness']='NOT_VERIFIED: see loaded-contact evidence and geometry bounds; no global safety claim'
    except BaseException:report['errors'].append(traceback.format_exc());report['status']='NOT_VERIFIED'
    finally:
        if backend is not None:
            report['physics_substeps']=backend.physics_substeps_executed;report['final_sofa_logs']=_log_messages(backend)
            if backend._handles is not None:report['final_solver']=solver_metrics(backend)
            backend.close()
        if trace:trace.close()
        if worker is not None:
            try:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=15)
            except Exception:worker.terminate()
        report['preservation']={'status':'PASS' if all(Path(p).is_file() and digest(p)==h for p,h in protected.items()) else 'FAIL','protected_files':len(protected)};report['wall_seconds']=time.monotonic()-started;dump(out/'report.json',report)
        print('FINAL',report['status'],'physics',report['physics_substeps'],'candidates',report['search_candidates'],'REPORT',out/'report.json',flush=True)
    return 0 if report['status']=='CONTACT_REACHED' else 2
if __name__=='__main__':raise SystemExit(main())
