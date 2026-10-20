"""Read-only real-SOFA collision coverage audit for an existing 303 replay.

Uses the same 15 real actions as the earlier decimate99 comparison, records
actual dynamic collision edge connectivity at *every* integration endpoint,
and probes contact constraint metadata without changing physics. Neither
collision counts nor pairing are inferred from the mere presence of models.
Python 3.8 compatible. No training.
"""
import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
TEST=HERE.parent
PYTHON=TEST.parents[1]
sys.path[:0]=[str(PYTHON),str(TEST),str(HERE)]

from mcr_sim.config import DEFAULT_CONFIG
from mcr_sim.sofa_simulation import SofaBackend
from mcr_sim.vessel_bundle import load_vessel_bundle
from mcr_sim.rl_env import MagneticNavigationEnv
from run_wall_challenge import geometry,object_data
from run_adaptive_contact import contact_objects
from run_acceptance import jsonable,_log_messages


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def save(path,obj):
    Path(path).write_text(json.dumps(jsonable(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def production_hashes():
    return {str(p):sha(p) for p in sorted((PYTHON/'mcr_sim').rglob('*.py'))}


def snapshot_models(backend):
    h=backend._handles
    rows={}
    for category,node in [('catheter',h.instrument.CollisionModel),('vessel',h.vessel.CollisionModel)]:
        rows[category]=[]
        for obj in node.objects:
            cls=obj.getClassName()
            if cls in ('TriangleCollisionModel','LineCollisionModel','PointCollisionModel'):
                data=object_data(obj)
                rows[category].append({
                    'class':cls,'path':obj.getPathName(),
                    'data':{k:data.get(k,'NOT_EXPOSED') for k in
                            ('proximity','group','moving','simulated','bothSide','selfCollision','active')},
                })
    rows['solver']={
        'intersection':{'class':h.simulator.intersection.getClassName(),
                        'data':{k:object_data(h.simulator.intersection).get(k,'NOT_EXPOSED')
                                for k in ('contactDistance','alarmDistance','angleCone')}},
        'narrow_phase':h.simulator.narrow_phase.getClassName(),
        'contact_manager':{'class':h.simulator.response.getClassName(),
                           'response':object_data(h.simulator.response).get('response','NOT_EXPOSED')},
        'lcp':{'class':h.simulator.lcp_solver.getClassName(),
               'data':{k:object_data(h.simulator.lcp_solver).get(k,'NOT_EXPOSED')
                       for k in ('mu','tolerance','maxIt','build_lcp')}},
    }
    return rows


def limited_contact_objects(root):
    rows=[]
    for item in contact_objects(root):
        data=item.get('data',{})
        candidates={}
        for name,value in data.items():
            if any(t in name.lower() for t in ('model','collision','contact','pair','element','constraint')):
                s=str(value)
                candidates[name]=value if len(s)<=1500 else 'TOO_LARGE_TO_CAPTURE'
        rows.append({'class':item['class'],'path':item['path'],
                     'possible_pair_fields':candidates,
                     'exact_collision_model_pair':'NOT_OBSERVED_UNLESS_EXPLICIT_FIELDS',
                     'note':'Constraint object existence alone does not identify primitive pair or contact force'})
    return rows


def lcp_snapshot(backend):
    obj=backend._handles.simulator.lcp_solver
    data=object_data(obj)
    graph=data.get('graph','NOT_EXPOSED')
    sweep=None
    if isinstance(graph,str) and graph.startswith('Error '):
        try:
            values=[float(x) for x in graph[6:].split()]
            if all(math.isfinite(x) for x in values):
                sweep=len(values)
        except ValueError:
            pass
    return {'mu':data.get('mu','NOT_EXPOSED'),
            'build_lcp':data.get('build_lcp','NOT_EXPOSED'),
            'residual_vector_length_not_direct_contact_count':sweep}


def line_distance_bound_mm(raw,edges,points,number_samples=17):
    """Conservative true cubic BEZIER CENTERLINE-to-runtime-LINE bound, NOT tube."""
    control=np.asarray(raw['bezier_control_points_m'],float)
    if control.ndim!=3 or control.shape[1:]!=(4,3):
        raise ValueError('Missing authoritative current Beam Bezier nodes')
    points=np.asarray(points,float).reshape(-1,3)
    edges=np.asarray(edges,int).reshape(-1,2)
    if not len(control) or not len(edges) or (edges.min()<0 or edges.max()>=len(points)):
        raise ValueError('Invalid active beams or collision connectivity')
    a=points[edges[:,0]]
    b=points[edges[:,1]]
    v=b-a
    length2=np.einsum('ij,ij->i',v,v)
    mask=length2>1e-24
    if not np.any(mask):
        return {'status':'NOT_VERIFIED_NO_NONZERO_LINE','nonzero_proxy_edges':0}
    a=a[mask];v=v[mask];length2=length2[mask]
    grid=np.linspace(0.,1.,number_samples)
    t=grid[None,:,None]
    p=((1-t)**3*control[:,0,None,:] +
       3*(1-t)**2*t*control[:,1,None,:] +
       3*(1-t)*t*t*control[:,2,None,:] +
       t**3*control[:,3,None,:])
    queries=p.reshape(-1,3)
    minimum=np.empty(len(queries),float)
    for i in range(0,len(queries),128):
        x=queries[i:i+128,None,:]
        diff=x-a[None,:,:]
        proj=np.clip(np.einsum('ijk,jk->ij',diff,v)/length2[None,:],0.,1.)
        closest=a[None,:,:]+proj[:,:,None]*v[None,:,:]
        minimum[i:i+128]=np.linalg.norm(x-closest,axis=2).min(axis=1)
    per_beam=minimum.reshape(len(control),number_samples)
    speed=3*np.linalg.norm(np.diff(control,axis=1),axis=2).max(axis=1)
    cover=speed/(2*(number_samples-1))
    upper=float(np.max(per_beam.max(axis=1)+cover))*1000
    lower=float(np.max(minimum))*1000
    return {
        'status':'MEASURED_CENTERLINE_ONLY',
        'observed_nonzero_proxy_edges':int(mask.sum()),
        'bezier_beams':len(control),
        'samples_per_beam':number_samples,
        'max_sampled_centerline_to_collision_line_distance_mm':lower,
        'max_entire_cubic_centerline_to_collision_line_upper_bound_mm':upper,
        'bound_method':'Closest distance to actual runtime nonzero line segments is 1-Lipschitz; cubic derivative <=3*max control-edge length; sample-half-interval addition',
        'tube_surface_coverage':'NOT_VERIFIED',
        'detected_contact_pair':'NOT_VERIFIED',
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--comparison',type=Path,required=True)
    ap.add_argument('--kind',choices=('fine','candidate'),required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--uncertain-frames',type=Path,required=False)
    args=ap.parse_args()
    comparison=args.comparison.resolve()
    gate=json.loads((comparison/'wrapper_retry/geometry_report.json').read_text())
    seed=int(gate['seed'])
    if seed!=303:raise ValueError('This targeted runtime replay is restricted to seed 303')
    if gate['stage_a_status']!='PASS' or gate['preservation']['status']!='PASS':
        raise ValueError('Prior candidate Stage A failed')
    vessel=Path(gate['source_directory'] if args.kind=='fine' else gate['candidate_directory']).resolve()
    known=gate['source_stl_sha256'] if args.kind=='fine' else gate['candidate_stl_sha256']
    if sha(vessel/'collision_inner.stl')!=known:raise ValueError('Input vessel STL was changed')
    # Keep new diagnostic files inside testing reports (Python 3.8 compatible).
    import os
    expected_root=(TEST/'reports').resolve()
    result_path=args.output.resolve()
    if os.path.commonpath([str(result_path),str(expected_root)]) != str(expected_root):
        raise ValueError('Diagnostic output must remain in testing reports')
    plan=json.loads((comparison/'replay_plan.json').read_text())
    actions=plan['full_rl_actions']
    if len(actions)!=15:raise ValueError('Expected previous 15 action replay')
    previous=comparison/('sofa_'+args.kind)/'physical_substeps.jsonl'
    if not previous.is_file():raise FileNotFoundError(previous)
    selected={22*i for i in range(1,16)}
    initial=json.loads((comparison/('sofa_'+args.kind)/'report.json').read_text())
    onset=initial.get('first_constraint_object_substep')
    if onset is not None:selected.update(range(max(1,int(onset)-5),min(330,int(onset)+5)+1))
    if args.uncertain_frames and args.uncertain_frames.exists():
        critical=json.loads(args.uncertain_frames.read_text())
        selected.update(int(x) for x in critical.get('physical_substeps',[]))
    output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    started=time.monotonic()
    protected=production_hashes()
    status={'status':'FAIL','seed':seed,'kind':args.kind,
            'vessel':str(vessel),'stl_sha256':known,'saved_trace_sha256':sha(previous),
            'steps':0,'selected_coverage_frames':sorted(selected),
            'errors':[],'runtime_model_catalog':None,
            'trajectory_replay_identity':'NOT_VERIFIED',
            'direct_contact_force':'NOT_OBSERVED','direct_contact_count':'NOT_OBSERVED',
            'exact_primitive_pair':'NOT_VERIFIED',
            'continuous_anti_penetration':'NOT_VERIFIED'}
    backend=None;env=None;stream=None;old_stream=None
    try:
        cfg=replace(DEFAULT_CONFIG,vessel_root=vessel.parent,vessel_seeds=(seed,))
        backend=SofaBackend(cfg)
        env=MagneticNavigationEnv(backend,config=cfg,bundles=[load_vessel_bundle(vessel)])
        env.reset(seed=seed,options={'vessel_seed':seed,'target_branch':'lcca'})
        # Replicate the original timing-warmup then full reset.
        for a in ([0.,0.,.125],[0.,0.,0.]):
            env.step(np.asarray(a,np.float32))
        env.reset(seed=seed,options={'vessel_seed':seed,'target_branch':'lcca'})
        status['runtime_model_catalog']=snapshot_models(backend)
        status['reset_physics_dt_s']=float(backend._root.dt.value)
        stream=(output/'runtime_substeps.jsonl').open('w')
        old_stream=previous.open('r')
        original=backend.physics_substep
        gap_max=0.;bad=0;events=0;nonzero_min=None;nonzero_max=0
        def audited(field,delta):
            nonlocal gap_max,bad,events,nonzero_min,nonzero_max
            diagnostic=original(field,delta)
            if diagnostic.simulation_error:raise RuntimeError('Real SOFA numerical failure')
            i=backend.physics_substeps_executed
            old_line=old_stream.readline()
            if not old_line:raise ValueError('Original saved replay ended before physical step %d'%i)
            old=json.loads(old_line)
            if int(old['physical_substep'])!=i:raise ValueError('Saved step index mismatch')
            h=backend._handles
            raw=geometry(backend)
            state=backend.state();state.validate()
            tip_error=float(np.linalg.norm(np.asarray(old['tip_m'])-state.tip_m)*1000)
            gap_max=max(gap_max,tip_error)
            if tip_error>.01:bad+=1
            mapped=np.asarray(raw['mapped_collision_positions_m'],float).reshape(-1,3)
            topo=np.asarray(h.instrument.CollisionModel.getObject('collisEdgeSet').edges.value,int).reshape(-1,2)
            if len(topo)!=int(old['collision_topology_edges']) or len(mapped)!=int(old['collision_points']):
                raise ValueError('Dynamic collision topology count differs from saved source')
            if len(topo) and (topo.min()<0 or topo.max()>=len(mapped)):
                raise ValueError('Out-of-bounds actual SOFA collision graph')
            lengths=np.linalg.norm(mapped[topo[:,1]]-mapped[topo[:,0]],axis=1) if len(topo) else np.array([])
            positive=int(np.count_nonzero(lengths>1e-12))
            if positive!=int(old['geometrically_nonzero_edges_gt_1e_12_m']):
                bad+=1
            nonzero_min=positive if nonzero_min is None else min(positive,nonzero_min)
            nonzero_max=max(positive,nonzero_max)
            objs=limited_contact_objects(backend._root)
            if objs:events+=1
            row={'physical_substep':i,'rl_action':old.get('rl_action'),
                 'seed':seed,'kind':args.kind,
                 'actual_edges':len(topo),'actual_points':len(mapped),
                 'actual_nonzero_edges':positive,
                 'topology_sha256':hashlib.sha256(topo.astype(np.int64).tobytes()).hexdigest(),
                 'tip_replay_error_mm':tip_error,
                 'model_constraints':objs,
                 'lcp':lcp_snapshot(backend),
                 'geometry_sanity':{'bezier_beams':len(raw['bezier_control_points_m']),
                                    'irc_first':raw['first']}}
            if i in selected:
                row['runtime_collision_edges']=topo.tolist()
                row['active_line_coverage']=line_distance_bound_mm(raw,topo,mapped)
            stream.write(json.dumps(jsonable(row),allow_nan=False)+'\n')
            if i%22==0:
                stream.flush()
                print('RUNTIME_COVERAGE',args.kind,i,'nonzero',positive,'tip_diff_mm',round(tip_error,5),
                      'constraints',len(objs),flush=True)
            return diagnostic
        backend.physics_substep=audited
        for item in actions:
            pre=backend.physics_substeps_executed
            obs,reward,term,trunc,info=env.step(np.asarray(item['action'],np.float32))
            if backend.physics_substeps_executed-pre!=22 or term or trunc or not info.get('transition_valid') or not np.isfinite(obs).all():
                raise RuntimeError('Production RL step invalid during controlled replay')
        if old_stream.readline():raise RuntimeError('Historical trace contains extra physical frames')
        status['steps']=backend.physics_substeps_executed
        status['max_tip_replay_error_mm']=gap_max
        status['frames_above_0_01mm_tip_difference']=bad
        status['actual_nonzero_line_edges_min_max']=[nonzero_min,nonzero_max]
        status['constraint_object_frames']=events
        status['trajectory_replay_identity']='MATCHED_WITHIN_0_01MM' if bad==0 else 'NOT_VERIFIED_REPLAY_DIFFERED'
        status['status']='CAPTURED' if bad==0 and status['steps']==330 else 'CAPTURED_WITH_REPLAY_DISCREPANCY'
    except BaseException:
        status['errors'].append(traceback.format_exc())
        print(status['errors'][-1],flush=True)
    finally:
        if old_stream:old_stream.close()
        if stream:stream.close()
        if env:env.close()
        elif backend:backend.close()
        status['production_source_unchanged']=production_hashes()==protected
        status['input_stl_unchanged']=sha(vessel/'collision_inner.stl')==known
        if not status['production_source_unchanged'] or not status['input_stl_unchanged']:
            status['status']='FAIL_SOURCE_CHANGED'
        status['seconds']=time.monotonic()-started
        save(output/'report.json',status)
        print('RUNTIME_COVERAGE_FINAL',args.kind,status['status'],status['seconds'],flush=True)
    return 0 if status['status']=='CAPTURED' else 2


if __name__=='__main__':
    raise SystemExit(main())
