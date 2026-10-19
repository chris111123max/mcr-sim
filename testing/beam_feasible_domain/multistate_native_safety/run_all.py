import json,subprocess,sys,time
from pathlib import Path
base=Path(__file__).resolve().parent
cases=json.loads((base/'cases.json').read_text())
status=[]
for case in cases:
 print('START',case['id'],flush=True)
 with (base/'results'/f"{case['id']}.log").open('w') as log:
  p=subprocess.run([sys.executable,str(base/'run_case.py'),'--case-id',case['id']],stdout=log,stderr=subprocess.STDOUT)
 status.append({'id':case['id'],'returncode':p.returncode})
 (base/'results/run_status.json').write_text(json.dumps(status,indent=2))
 print('DONE',case['id'],p.returncode,flush=True)
 # State mismatches are excluded; do not invent missing context or rerun seeds.
 if p.returncode not in (0,3): break
print('ALL_DONE',flush=True)
