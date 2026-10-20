"""Unmodified independent Stage A plus formal generator acceptance, no SOFA."""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_acceptance import inspect_asset,jsonable

def main(vessel,seed,output):
    started=time.monotonic();vessel=Path(vessel);result={}
    try:
        result,_,_=inspect_asset(vessel,int(seed))
        metadata=json.loads((vessel/'metadata.json').read_text())
        result['independent_stage_a_status']=result['status'];result['generator_validation_passed']=metadata['validation']['passed']
        result['generator_failure_reasons']=metadata['validation']['failure_reasons']
        result['effective_meshing']=metadata['meshing']
        if not result['generator_validation_passed']:result['status']='FAIL'
    except BaseException:result.update({'status':'FAIL','traceback':traceback.format_exc()})
    result['seconds']=time.monotonic()-started;Path(output).write_text(json.dumps(jsonable(result),indent=2)+'\n')
    print('STAGE_A',seed,result['status'],result.get('generator_failure_reasons'),flush=True)
    return 0 if result['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main(*sys.argv[1:]))
