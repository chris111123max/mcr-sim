"""Generate the ten-vessel cohort first, then geometry gates; never interleave SOFA."""
import sys,os,time,json,subprocess,hashlib,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;TEST=HERE.parent;PYTHON=TEST.parents[1]
SEEDS=(101,202,303,404,505,606,707,808,909,1010)
GEOM=PYTHON/'stEVE_anatomical_vessels/.venv/bin/python'
BASE=PYTHON/'stEVE_anatomical_vessels/pretrain_20261020_d085_vox03'
REPORT=TEST/'reports/pretrain_final_20261020'
APPROVED=PYTHON/'stEVE_anatomical_vessels/tmp_seed101_geometry_r3TeZG/refined_03/arch_type_I_seed_101/parameters.json'

def dump(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
def main():
    started=time.monotonic();approved=json.loads(APPROVED.read_text());report={'status':'GENERATING','train_seeds':SEEDS,'eval_seeds_not_in_this_batch':[1101,1202],'max_generator_processes':4,'max_stage_a_processes':2,'generation_timeout_seconds_per_seed':1200,'sofa_started_before_all_assets':False,'assets':{},'errors':[]};jobs={};pending=[];failed=False
    def save():report['seconds']=time.monotonic()-started;dump(REPORT/'ten_cohort_generation.json',report)
    def verify(seed,path):
        m=json.loads((path/'metadata.json').read_text());params=json.loads((path/'parameters.json').read_text())
        if params['effective_profile']!=approved['effective_profile'] or params['source_sha256']!=approved['source_sha256'] or params['diameter_scale']!=.85:raise ValueError('Frozen generator config/source differs')
        if not m['validation']['passed']:raise ValueError('Generator geometry FAIL: '+str(m['validation']['failure_reasons']))
        report['assets'][str(seed)].update({'status':'PASS_GENERATOR','metadata_sha256':hashlib.sha256((path/'metadata.json').read_bytes()).hexdigest(),'stl_sha256':hashlib.sha256((path/'collision_inner.stl').read_bytes()).hexdigest(),'generated_at_utc':m['generated_at_utc']})
    try:
        for seed in SEEDS:
            path=BASE/('arch_type_I_seed_%d'%seed) if seed in (202,303) else BASE/('raw_seed_%d'%seed)/('arch_type_I_seed_%d'%seed)
            report['assets'][str(seed)]={'directory':str(path),'status':'PENDING'}
            if (path/'metadata.json').is_file():verify(seed,path)
            else:
                if path.exists():raise ValueError('Partial/existing asset is never overwritten: '+str(path))
                pending.append(seed)
        save()
        while pending or jobs:
            while pending and len(jobs)<4 and not failed:
                seed=pending.pop(0);path=Path(report['assets'][str(seed)]['directory']);log=REPORT/('seed%d_cohort_generation.log'%seed);f=log.open('w')
                cmd=[str(GEOM),'-B','-u',str(PYTHON/'stEVE_anatomical_vessels/src/generate.py'),'--seed',str(seed),'--profile',approved['source_profile_path'],'--diameter-scale','0.85','--voxel-spacing-mm','0.3','--clean-tolerance-mm','0.001','--simplification-error-mm','0.01','--decimate-target-reduction','0.25','--output-root',str(path.parent)]
                proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);jobs[seed]=(proc,f,time.monotonic());report['assets'][str(seed)].update({'status':'GENERATING','command':cmd,'pid':proc.pid,'log':str(log)})
                print('GENERATING',seed,'pid',proc.pid,flush=True);save()
            for seed,(proc,f,t0) in list(jobs.items()):
                if proc.poll() is None and time.monotonic()-t0>1200:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    report['assets'][str(seed)]['timeout']=True
                if proc.poll() is None:continue
                f.close();del jobs[seed];item=report['assets'][str(seed)];item['process_exit_code']=proc.returncode;item['generation_seconds']=time.monotonic()-t0
                try:
                    if proc.returncode:raise RuntimeError('Generator exit'+str(proc.returncode))
                    verify(seed,Path(item['directory']));print('GENERATOR_PASS',seed,flush=True)
                except BaseException:item['status']='FAIL';item['error']=traceback.format_exc();failed=True;print('GENERATOR_FAIL',seed,flush=True)
                save()
            if failed and not jobs:break
            time.sleep(2)
        if failed:raise RuntimeError('Geometry/generation failure: no additional seeds or SOFA started; active generation jobs retained until completion')
        report['status']='ALL_TEN_GENERATED_GEOMETRY_GATES';save();print('ALL_TEN_GENERATED',flush=True)
        pending=list(SEEDS);jobs={}
        while pending or jobs:
            while pending and len(jobs)<2:
                seed=pending.pop(0);item=report['assets'][str(seed)];log=REPORT/('seed%d_cohort_stage_a.log'%seed);f=log.open('w');out=REPORT/('seed%d_cohort_stage_a.json'%seed)
                cmd=[str(GEOM),'-B','-u',str(HERE/'pretrain_check_asset.py'),item['directory'],str(seed),str(out)]
                proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);jobs[seed]=(proc,f,time.monotonic(),out);item['stage_a_report']=str(out)
            for seed,(proc,f,t0,out) in list(jobs.items()):
                if proc.poll() is None:continue
                f.close();del jobs[seed];item=report['assets'][str(seed)];item['stage_a_exit_code']=proc.returncode;item['independent_stage_a']='PASS' if proc.returncode==0 and out.is_file() and json.loads(out.read_text())['status']=='PASS' else 'FAIL';print('COHORT_STAGE_A',seed,item['independent_stage_a'],flush=True);save()
                if item['independent_stage_a']!='PASS':failed=True
            if failed:pending=[]
            time.sleep(1)
        if failed:raise RuntimeError('Independent Stage A failed; no real SOFA test started')
        report['status']='ALL_TEN_STAGE_A_PASS';save();print('ALL_TEN_STAGE_A_PASS_READY_FOR_TEN_ENVS',flush=True)
    except BaseException:report['status']='FAIL';report['errors'].append(traceback.format_exc());save();print(report['errors'][-1],flush=True);return 2
    return 0
if __name__=='__main__':raise SystemExit(main())
