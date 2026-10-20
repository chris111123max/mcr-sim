"""Seed 303 collision-coverage and uncertain-contact audit (READ-ONLY).

Runs denser OFFLINE endpoint tube witnesses first, then exact-action REAL SOFA
replays in candidate and fine meshes to observe each dynamic collision graph
and constraint object. Does not modify vessel or production code. No RL train.
Python 3.8 compatible. Original report files are read-only.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
TEST=HERE.parent
PYTHON=TEST.parents[1]
REPORTS=TEST/'reports'
COMPARISON=REPORTS/'official_decimate99_seed202_303'/'seed_303'
GEOM_PYTHON=PYTHON/'stEVE_anatomical_vessels'/'.venv'/'bin'/'python'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def production_sources():
    return {str(p):sha(p) for p in sorted((PYTHON/'mcr_sim').rglob('*.py'))}


def verify_saved():
    cross=load(REPORTS/'official_decimate99_seed202_303'/'CROSS_SEED_SUMMARY.json')
    seed303=cross['seeds']['303']
    if seed303.get('geometry_status')!='PASS' or seed303.get('physics_status')!='PASS_CAPTURE_COMPLETE' or seed303.get('offline_status')!='COMPLETED_ENDPOINT_ONLY':
        raise ValueError('Seed303 prior geometry, SOFA and offline gate are not complete')
    gate=load(COMPARISON/'wrapper_retry'/'geometry_report.json')
    if int(gate.get('seed',-1))!=303 or gate['stage_a_status']!='PASS' or gate['preservation']['status']!='PASS':
        raise ValueError('Seed303 previous geometry Stage A not PASS')
    assets={}
    for kind in ('fine','candidate'):
        vessel=Path(gate['source_directory'] if kind=='fine' else gate['candidate_directory'])
        digest=gate['source_stl_sha256'] if kind=='fine' else gate['candidate_stl_sha256']
        if not vessel.is_dir() or sha(vessel/'collision_inner.stl')!=digest:
            raise ValueError('Previously accepted seed303 asset was changed: '+kind)
        previous=load(COMPARISON/('sofa_'+kind)/'report.json')
        if previous.get('status')!='PASS_CAPTURE_COMPLETE' or previous.get('seed')!=303:
            raise ValueError('Saved real SOFA trace is not accepted for '+kind)
        expected_frames=sum(1 for _ in (COMPARISON/('sofa_'+kind)/'physical_substeps.jsonl').open())
        if expected_frames!=330:raise ValueError('Expected exactly 330 saved frames per mesh')
        assets[kind]={'vessel':str(vessel),'stl_sha256':digest,
                      'saved_physical_substeps':expected_frames,
                      'original_sofa_report':str(COMPARISON/('sofa_'+kind)/'report.json')}
    for kind in ('fine','candidate'):
        if not (COMPARISON/(kind+'_geometry_substeps.jsonl')).is_file():
            raise FileNotFoundError('Missing original geometry substeps for '+kind)
    return assets


def execute(name,command,out,timeout):
    log=out/(name+'.log')
    began=time.monotonic()
    print('START',name,' '.join(map(str,command)),flush=True)
    with log.open('w') as stream:
        try:
            proc=subprocess.run([str(x) for x in command],cwd=str(PYTHON),
                                stdout=stream,stderr=subprocess.STDOUT,
                                timeout=timeout,check=False,env=os.environ.copy())
            exitcode=proc.returncode
            timed_out=False
        except subprocess.TimeoutExpired:
            exitcode=124;timed_out=True
            stream.write('\nTIMEOUT: %d sec (subprocess terminated)\n'%timeout)
    result={'command':list(map(str,command)),'exit_code':exitcode,
            'seconds':time.monotonic()-began,'timed_out':timed_out,'log':str(log)}
    print('FINISH',name,'exit',exitcode,'seconds',round(result['seconds'],1),flush=True)
    return result


def summarize_replay(source):
    report=load(source/'report.json')
    measures=[]
    first_contact=None
    last_contact=None
    topologies=set()
    active_min=None;active_max=None
    frames=0;count_constraint=0
    primitive_fields=[]
    warnings=[]
    with (source/'runtime_substeps.jsonl').open() as f:
        for line in f:
            row=json.loads(line);frames+=1
            if int(row['physical_substep'])!=frames:raise ValueError('Runtime recording has missing frames')
            topologies.add(row['topology_sha256'])
            active=int(row['actual_nonzero_edges'])
            active_min=active if active_min is None else min(active_min,active)
            active_max=active if active_max is None else max(active_max,active)
            if row['model_constraints']:
                count_constraint+=1
                if first_contact is None:first_contact=frames
                last_contact=frames
                for obj in row['model_constraints']:
                    keys=obj.get('possible_pair_fields',{})
                    if keys and len(primitive_fields)<10:
                        primitive_fields.append({'frame':frames,'class':obj['class'],
                                                 'path':obj['path'],'fields':keys})
            if 'active_line_coverage' in row:
                measures.append({
                    'physical_substep':frames,
                    'actual_nonzero_edges':active,
                    'coverage':row['active_line_coverage'],
                })
    if frames!=330:raise ValueError('Runtime replay captured only %d frames'%frames)
    model_catalog=report['runtime_model_catalog']
    source_classes={kind:[x['class'] for x in model_catalog[kind]]
                    for kind in ('catheter','vessel')}
    expected=(set(source_classes['catheter'])>={'LineCollisionModel','PointCollisionModel'} and
              set(source_classes['vessel'])>={'TriangleCollisionModel','LineCollisionModel'})
    return {
        'runtime_report_status':report['status'],
        'trajectory_replay_identity':report['trajectory_replay_identity'],
        'max_tip_replay_error_mm':report.get('max_tip_replay_error_mm'),
        'real_physical_substeps':frames,
        'runtime_collision_graph_topology_versions':len(topologies),
        'runtime_nonzero_line_edges_min_max':[active_min,active_max],
        'collision_models':source_classes,
        'expected_models_present':expected,
        'observed_constraint_object_frames':count_constraint,
        'first_constraint_object_frame':first_contact,
        'last_constraint_object_frame':last_contact,
        'candidate_pair_fields':primitive_fields,
        'explicit_contact_pair_classification':'NOT_VERIFIED',
        'direct_contact_force':'NOT_OBSERVED',
        'direct_contact_count':'NOT_OBSERVED',
        'proxy_centerline_coverage':measures,
        'whole_catheter_surface_collision_coverage':'NOT_VERIFIED',
        'proxy_topology_changes_not_assumed_static':True,
    }


def summarize_geometry(source):
    rep=load(source/'critical303_report.json')
    cases=[{'trajectory':c['trajectory'],'physical_substep':c['physical_substep'],
            'wall':c['wall'],'prior_upper_mm':c['old_continuous_swept_ball_upper_mm'],
            'new_dense_sampled_depth_mm':c.get('refined_sampled_tube',{}).get('maximum_actual_sampled_tube_surface_depth_mm'),
            'num_surface_samples':c.get('refined_sampled_tube',{}).get('actual_sample_points'),
            'measurement_status':c['status'],
            'continuous_time':'NOT_VERIFIED'} for c in rep['cases']]
    return {'status':rep['status'],'uncertain_frames':rep['uncertain_frames'],
            'sampled_positive_cases':rep.get('sampled_positive_cases'),
            'cases':cases,'global_anti_penetration':'NOT_VERIFIED'}


def report_markdown(report):
    lines=[
        '# Seed 303 collision coverage: production-scene evidence and critical geometry',
        '',
        '**Overall:** '+report['status'],
        '',
        '## Prior source comparison',
        '',
        '- Original 0.3 mm fine mesh and official 99% candidate are immutable inputs.',
        '- Original comparison used 15 RL actions ×22 real SOFA substeps (330 frames per mesh).',
        '- Previous 202/303 study passed geometry Stage A; this audit is **only seed303**.',
        '',
        '## Geometry at previously unresolved 303 frames',
        '',
        '| Trajectory | Substep | Against wall | Old conservative swept-ball upper (mm) | New sampled tube depth (mm) | Result |',
        '|---|---:|---|---:|---:|---|',
    ]
    geom=report.get('critical_geometry',{})
    for c in geom.get('cases',[]):
        def fmt(x):
            return '-' if x is None else ('%.6f'%float(x))
        lines.append('| %s | %d | %s | %s | %s | %s |'%(
            c['trajectory'],c['physical_substep'],c['wall'],
            fmt(c['prior_upper_mm']),fmt(c['new_dense_sampled_depth_mm']),c['measurement_status']))
    lines.extend(['','','## Real SOFA collision topology and contact constraints','','| Mesh | 330 physics frames | Nonzero dynamic collision lines | Distinct topology hashes | First constraint object | Trace identity |',
                  '|---|---|---|---:|---|---|'])
    for kind in ('candidate','fine'):
        r=report.get('runtime_replays',{}).get(kind,{})
        n=r.get('runtime_nonzero_line_edges_min_max')
        lines.append('| %s | %s | %s | %s | %s | %s |'%(
            kind,r.get('real_physical_substeps','NOT_RUN'),n if n is not None else '-',
            r.get('runtime_collision_graph_topology_versions','-'),
            r.get('first_constraint_object_frame','NOT_OBSERVED'),
            r.get('trajectory_replay_identity','NOT_RUN')))
    lines.extend(['','## Interpretation','',
       '- Configured models: Catheter Line+Point, vessel Triangle+Line. Their presence does not establish which primitive pairing generated constraint events.',
       '- The runtime edge arrays and mapped collision vertices are captured at each SOFA endpoint, not reconstructed from nominal edge counts.',
       '- Bezier-to-actual-collision-line bounds cover CENTERLINE only, NOT finite-radius tube or every wall-triangle pairing.',
       '- SOFA constraint objects are observed but cannot be assumed to expose contact forces or complete contact-element count.',
       '- Zero dense sampled penetration is NOT a continuous tube surface or integrator-between-substeps certificate.',
       '- Strict global vessel mesh self-intersection, whole-body contact coverage and response equivalence remain NOT_VERIFIED.',
       '- No physics parameter edits; no training, publication, or production asset replacement.',
       ''])
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=REPORTS/(
        'contact_coverage_seed303_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    ap.add_argument('--only-offline',action='store_true',
                    help='Do only saved-geometry refinement, no new real SOFA replay')
    args=ap.parse_args()
    out=args.output.resolve()
    if os.path.commonpath([str(out),str(REPORTS.resolve())])!=str(REPORTS.resolve()):
        ap.error('Output must remain within testing/mcr_sim_wall_acceptance/reports')
    if out.exists():ap.error('Output exists; choose a fresh --output to preserve previous reports')
    if not GEOM_PYTHON.is_file():ap.error('Geometry virtualenv not available')
    out.mkdir(parents=True)
    start=time.monotonic();protected=production_sources()
    rep={'status':'FAIL','comparison_root':str(COMPARISON),
         'output':str(out),'production_hashes_before':protected,
         'training_started':False,'physics_modified':False,
         'runtime_replays':{},'critical_geometry':{},'processes':{},'errors':[],
         'strict_collision_equivalence':'NOT_VERIFIED'}
    try:
        rep['source_assets']=verify_saved()
        geometry_output=out/'critical303'
        rep['processes']['critical303']=execute(
            'critical303',
            [GEOM_PYTHON,'-B','-u',HERE/'contact_coverage_refine303.py',
             '--comparison',COMPARISON,'--output',geometry_output],out,1200)
        if (geometry_output/'critical303_report.json').is_file():
            rep['critical_geometry']=summarize_geometry(geometry_output)
        outcome=rep['critical_geometry'].get('status')
        if outcome=='PENETRATION_DETECTED':
            raise RuntimeError('Saved real physics has confirmed positive material witness; no further SOFA replay')
        if outcome not in ('DENSE_SAMPLES_COMPLETE_NOT_CONTINUOUSLY_CERTIFIED',
                         'COMPLETED_WITH_UNCERTAIN_CASES'):
            raise RuntimeError('Critical geometry worker failed: see original log')
        if args.only_offline:
            rep['status']='OFFLINE_COMPLETE_RUNTIME_NOT_RUN'
        else:
            for kind in ('candidate','fine'):
                child=out/('runtime_'+kind)
                rep['processes']['runtime_'+kind]=execute(
                    'runtime_'+kind,
                    [sys.executable,'-B','-u',HERE/'contact_coverage_runtime.py',
                     '--comparison',COMPARISON,'--kind',kind,'--output',child,
                     '--uncertain-frames',geometry_output/'critical_frames.json'],out,1800)
                if (child/'report.json').is_file() and (child/'runtime_substeps.jsonl').is_file():
                    status=load(child/'report.json')
                    if status.get('status') in ('CAPTURED','CAPTURED_WITH_REPLAY_DISCREPANCY'):
                        rep['runtime_replays'][kind]=summarize_replay(child)
                if rep['processes']['runtime_'+kind]['exit_code']!=0:
                    raise RuntimeError('Actual SOFA replay '+kind+' failed or differed; no false acceptance')
            rep['status']='COMPLETED_WITH_COLLISION_EQUIVALENCE_NOT_VERIFIED'
    except BaseException:
        rep['errors'].append(traceback.format_exc())
        if rep['critical_geometry'].get('status')=='PENETRATION_DETECTED':
            rep['status']='PENETRATION_DETECTED_STOP'
        else:rep['status']='STOPPED_WITH_FAILURE'
        print(rep['errors'][-1],flush=True)
    finally:
        rep['seconds']=time.monotonic()-start
        rep['production_sources_unchanged']=protected==production_sources()
        rep.pop('production_hashes_before',None)
        if not rep['production_sources_unchanged']:rep['status']='FAIL_PRODUCTION_CHANGED'
        write(out/'CONTACT_COVERAGE_REPORT.json',rep)
        (out/'CONTACT_COVERAGE_REPORT.md').write_text(report_markdown(rep))
        print('CONTACT_COVERAGE_FINAL',rep['status'],'report',out/'CONTACT_COVERAGE_REPORT.md',flush=True)
    return 0 if rep['status'] in ('OFFLINE_COMPLETE_RUNTIME_NOT_RUN',
                                  'COMPLETED_WITH_COLLISION_EQUIVALENCE_NOT_VERIFIED') else 2


if __name__=='__main__':
    raise SystemExit(main())
