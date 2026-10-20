"""Bounded real SOFA wall challenge. Test-only observation; no production edits."""
import os,sys,json,time,hashlib,subprocess,traceback,argparse
from pathlib import Path
from datetime import datetime
import numpy as np
HERE=Path(__file__).resolve().parent; TEST=HERE.parent; REPO=TEST.parents[2]; PYTHON=REPO/'python'
sys.path.insert(0,str(PYTHON)); sys.path.insert(0,str(TEST))
from run_acceptance import jsonable,_log_messages

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj): p.write_text(json.dumps(jsonable(obj),indent=2,allow_nan=False)+'\n')
def read_data(obj,name): return getattr(obj,name).value

def object_data(obj):
    result={}
    for d in obj.getDataFields():
        name=d.getName()
        try:
            value=jsonable(d.value)
            result[name]=value if len(str(value))<12000 else {'status':'NOT_VERIFIED','reason':'Data exceeds snapshot limit','size_chars':len(str(value))}
        except Exception as exc: result[name]={'status':'NOT_VERIFIED','reason':str(exc)}
    return result

def geometry(backend):
    h=backend._handles; ir=h.ir_controller; ic=h.instrument.IC
    pos=np.asarray(h.dofs.position.value,float); first=int(np.asarray(ir.indexFirstNode.value).reshape(-1)[0]); curv=np.asarray(ir.CurvAbs.value,float).reshape(-1)
    ids=np.asarray(ic.edgeList.value,int).reshape(-1); lengths=np.asarray(ic.lengthList.value,float).reshape(-1)
    top=np.asarray(h.instrument.RG.edges.value,int).reshape(-1,2); edges=top[ids]
    expected=np.c_[np.arange(first,len(pos)-1),np.arange(first+1,len(pos))]
    if not np.array_equal(edges,expected) or not np.allclose(lengths,np.diff(curv),rtol=1e-6,atol=1e-8): raise RuntimeError('Actual interpolation edge/length list does not match IRC active range')
    bezier_nodes=[obj for obj in ic.getSlaves() if obj.getName()=='bezierNodes']
    if len(bezier_nodes)!=1: raise RuntimeError('Authoritative BeamAdapter bezierNodes slave unavailable')
    controls=np.asarray(bezier_nodes[0].position.value,float).reshape(-1,4,3)
    if len(controls)!=len(edges) or not np.all(np.isfinite(controls)): raise RuntimeError('Invalid current Bezier control points')
    if not np.allclose(controls[:,0],pos[edges[:,0],:3],rtol=0,atol=1e-12) or not np.allclose(controls[:,3],pos[edges[:,1],:3],rtol=0,atol=1e-12): raise RuntimeError('Bezier controls stale or have unverified positional DOF-to-beam transforms')
    return {'dofs':pos.tolist(),'first':first,'edges':edges.tolist(),'lengths':lengths.tolist(),'aligned':bool(ic.dofsAndBeamsAligned.value),'bezier_control_points_m':controls.tolist(),'curv_abs_m':curv.tolist(),'irc_xtip_m':np.asarray(ir.xtip.value).tolist(),'mapped_collision_positions_m':np.asarray(h.instrument.collision_dofs.position.value).tolist(),'geometry_source':'Authoritative current WireBeamInterpolation slave bezierNodes.position from AdaptiveBeamMapping.apply; includes actual DOF-to-beam transforms; active edge/length/endpoint correspondence checked against IRC and Rigid3d DOFs.'}

def solver_metrics(backend):
    h=backend._handles; result={}
    for key,obj in [('lcp',h.simulator.lcp_solver),('contact_manager',h.simulator.response),('narrow_phase',h.simulator.narrow_phase)]:
        item={'data':object_data(obj),'getters':{}}
        for name in ('getNumberOfContacts','getNumberOfConstraints','getNumConstraints','getContacts'):
            try: method=getattr(obj,name) if name in dir(obj) else None
            except Exception: method=None
            if callable(method):
                try: item['getters'][name]={'status':'OBSERVED','value':jsonable(method())}
                except Exception as exc: item['getters'][name]={'status':'NOT_VERIFIED','reason':str(exc)}
            else: item['getters'][name]={'status':'NOT_VERIFIED','reason':'Not exposed by SofaPython3'}
        result[key]=item
    return result

class AcceptanceStop(BaseException): pass

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--vessel',type=Path,required=True); ap.add_argument('--max-rl-steps',type=int,default=23,choices=range(1,24)); ap.add_argument('--init-only',action='store_true'); args=ap.parse_args()
    vessel=args.vessel.resolve(); reference=json.loads((PYTHON/'stEVE_anatomical_vessels/tmp_seed101_geometry_r3TeZG/geometry_repair_report.json').read_text())['runs']['refined_03']
    if vessel!=Path(reference['directory']) or digest(vessel/'collision_inner.stl')!=reference['sha256']['collision_inner.stl']: raise ValueError('Refusing wrong/old STL')
    out=TEST/'reports'/('wall_challenge_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')); out.mkdir()
    print('REPORT_DIRECTORY',out,flush=True)
    report={'status':'NOT_VERIFIED','seed':101,'scale':.85,'vessel':str(vessel),'stl_sha256':digest(vessel/'collision_inner.stl'),'source_sha256':{str(p):digest(p) for p in (PYTHON/'mcr_sim').glob('*.py')},'steps':[],'physics_substeps':0,'failures':[],'numerical_failure':False,'production_modified':False}
    geom_python=PYTHON/'stEVE_anatomical_vessels/.venv/bin/python'; worker=None; backend=None; env=None; stream=None
    started=time.monotonic(); records=[]
    try:
        err=(out/'geometry_worker.stderr').open('w')
        worker=subprocess.Popen([str(geom_python),'-B','-u',str(HERE/'penetration_geometry.py'),str(vessel)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
        response=worker.stdout.readline()
        if not response: raise RuntimeError('Detector worker did not return calibration: '+str(worker.poll()))
        report['detector']=json.loads(response)['ready']; dump(out/'detector_validation.json',report['detector'])
        print('DETECTOR',report['detector']['status'],flush=True)
        if report['detector']['status']!='PASS': raise RuntimeError('Detector validation failed; no SOFA initialized')
        from dataclasses import replace
        from mcr_sim.config import DEFAULT_CONFIG
        from mcr_sim.vessel_bundle import load_vessel_bundle
        from mcr_sim.sofa_simulation import SofaBackend
        from mcr_sim.rl_env import MagneticNavigationEnv
        bundle=load_vessel_bundle(vessel); config=replace(DEFAULT_CONFIG,vessel_root=vessel.parent)
        backend=SofaBackend(config); env=MagneticNavigationEnv(backend,config=config,bundles=[bundle]); obs,info=env.reset(seed=101,options={'vessel_seed':101,'target_branch':'lcca'})
        report['reset']={'observation':obs.tolist(),'info':jsonable(info),'stages':backend.stages,'dt':float(backend._root.dt.value),'solver':solver_metrics(backend),'interpolation_data':object_data(backend._handles.instrument.IC)}
        raw=geometry(backend); dump(out/'initial_sofa_geometry.json',raw)
        print('RESET_PASS',obs.shape,'active_beams',len(raw['edges']),flush=True)
        def measure(raw):
            worker.stdin.write(json.dumps(jsonable(raw),allow_nan=False)+'\n'); worker.stdin.flush()
            text=worker.stdout.readline()
            if not text: raise RuntimeError('Geometry worker exited')
            result=json.loads(text)
            if result['status']!='PASS': raise RuntimeError('Untrusted geometry diagnostic: '+str(result))
            return result
        report['initial_wall']=measure(raw)
        if args.init_only:
            report['reason']='Initialization and geometry inspection only; no animate'; return_code=0
        else:
            stream=(out/'physical_substeps.jsonl').open('w'); original=backend.physics_substep
            context={}; consecutive=0
            def observed(field,delta):
                nonlocal consecutive
                before=backend.state(); t0=time.monotonic(); diag=original(field,delta)
                record={'physical_substep':backend.physics_substeps_executed,**context,'commanded_field_t':np.asarray(field).tolist(),'commanded_insertion_delta_m':float(delta),'integration_seconds':time.monotonic()-t0,'simulation_error':diag.simulation_error}
                try:
                    state=backend.state(); state.validate(); raw=geometry(backend); wall=measure(raw)
                    record.update({'tip_m':state.tip_m.tolist(),'tip_displacement_m':(state.tip_m-before.tip_m).tolist(),'inserted_length_m':state.inserted_length_m,'actual_insertion_delta_m':state.inserted_length_m-before.inserted_length_m,'magnetic_field_t':state.magnetic_field_world_t.tolist(),'geometry':raw,'wall':wall,'solver':solver_metrics(backend),'sofa_logs':_log_messages(backend)})
                    all_pos=np.asarray(raw['dofs']); vel=np.asarray(backend._handles.dofs.velocity.value,float)
                    if not np.all(np.isfinite(all_pos)) or not np.all(np.isfinite(vel)) or diag.simulation_error: raise FloatingPointError('SOFA nonfinite state or reported numerical failure')
                    # Gross mechanical impossibility, not an acceptance safety tolerance.
                    if np.linalg.norm(state.tip_m-bundle.insertion_point_m)>2*state.inserted_length_m+1e-5: raise FloatingPointError('Tip farther than twice inserted rest length from entrance')
                    consecutive=consecutive+1 if wall['max_sampled_surface_penetration_mm']>wall['numeric_tolerance_mm'] else 0
                    record['consecutive_outside_substeps']=consecutive
                    record['total_record_seconds']=time.monotonic()-t0
                    records.append(record); stream.write(json.dumps(jsonable(record),allow_nan=False)+'\n'); stream.flush()
                    if consecutive>=5: raise AcceptanceStop('Five consecutive substeps with measured surface outside: stop without correcting positions')
                    return diag
                except AcceptanceStop: raise
                except Exception:
                    record['capture_or_numeric_error']=traceback.format_exc(); records.append(record); stream.write(json.dumps(jsonable(record),allow_nan=False)+'\n'); stream.flush()
                    raise AcceptanceStop('State/measurement invalid; stop immediately')
            backend.physics_substep=observed
            plan=[('zero',(0,0,0))]+[('low_advance',(0,0,.25))]*4+[('low_turn',(.25,0,0))]*2+[('low_turn_advance',(.25,.25,.25))]*4+[('medium_turn_advance',(.5,.5,.5))]*6+[('high_turn_advance',(1,0,1))]*6
            report['planned_actions']=plan[:args.max_rl_steps]
            for i,(name,action) in enumerate(plan[:args.max_rl_steps],1):
                context.update({'rl_step':i,'pattern':name,'action':list(action)})
                start_count=backend.physics_substeps_executed; t0=time.monotonic()
                try: obs,reward,term,trunc,info=env.step(np.array(action,np.float32))
                except AcceptanceStop as exc:
                    report['stopped_during_action']={**context,'reason':str(exc),'substeps_completed':backend.physics_substeps_executed-start_count}; break
                step={**context,'observation':obs.tolist(),'observation_shape':list(obs.shape),'observation_finite':bool(np.all(np.isfinite(obs))),'reward':reward,'info':info,'terminated':term,'truncated':trunc,'substeps':backend.physics_substeps_executed-start_count,'duration_seconds':time.monotonic()-t0}; report['steps'].append(step)
                print('ACTION',i,name,'substeps',step['substeps'],'insertion_mm',backend.state().inserted_length_m*1000,'max_surface_outside_mm',max(x['wall']['max_sampled_surface_penetration_mm'] for x in records[-22:] if 'wall' in x),flush=True)
                if term or trunc or not info.get('transition_valid',True) or not step['observation_finite'] or step['substeps']!=22: report['stop_reason']='Episode ended or transition invalid; no additional actions'; break
            return_code=0
        report['physics_substeps']=backend.physics_substeps_executed
        report['sofa_logs']=_log_messages(backend)
        valid=[x for x in records if 'wall' in x]
        report['numerical_failure']=bool(any(x.get('simulation_error') or x.get('capture_or_numeric_error') for x in records))
        if valid:
            worst=max(valid,key=lambda x:x['wall']['max_sampled_surface_penetration_mm'])
            report['maximum_penetration']={k:worst[k] for k in ('physical_substep','rl_step','pattern','action','wall','solver')}
            report['maximum_penetration']['wall'].pop('sampled_curve_m',None)
            report['outside_substep_count']=sum(x['wall']['max_sampled_surface_penetration_mm']>x['wall']['numeric_tolerance_mm'] for x in valid)
            report['maximum_consecutive_outside_substeps']=max(x.get('consecutive_outside_substeps',0) for x in valid)
            report['status']='FAIL' if report['outside_substep_count'] or report['numerical_failure'] else 'NOT_VERIFIED'
            report['anti_penetration_conclusion']='Measured sampled entity breach' if report['outside_substep_count'] else 'No sampled entity breach; effectiveness after wall contact requires separate observed contact evidence.'
        else: report['anti_penetration_conclusion']='NOT_VERIFIED: no physical substeps'
    except BaseException:
        report['failures'].append(traceback.format_exc()); report['status']='FAIL'; return_code=1
    finally:
        if backend is not None:
            report['physics_substeps']=backend.physics_substeps_executed
            if backend._handles is not None:
                try:
                    report['final_solver']=solver_metrics(backend); report['final_sofa_logs']=_log_messages(backend)
                    report['final_rigid_dofs']=np.asarray(backend._handles.dofs.position.value).tolist()
                except Exception: report['final_capture_error']=traceback.format_exc()
            backend.close()
        if stream: stream.close()
        if worker is not None:
            try: worker.stdin.write('{"stop":true}\n'); worker.stdin.flush(); worker.wait(timeout=30)
            except Exception: worker.terminate()
        manifest=json.loads((TEST/'reports/wall_challenge_protected_sha256.json').read_text()); changed=[p for p,h in manifest.items() if not Path(p).exists() or digest(p)!=h]
        report['preservation']={'status':'PASS' if not changed else 'FAIL','count':len(manifest),'changed':changed}
        report['duration_seconds']=time.monotonic()-started
        dump(out/'report.json',report)
        print('RESULT',report['status'],'substeps',report['physics_substeps'],'REPORT',out/'report.json',flush=True)
    return return_code if report['status']!='FAIL' else 1
if __name__=='__main__': raise SystemExit(main())
