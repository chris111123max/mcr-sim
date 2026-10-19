from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures,base64
P=Path(__file__).resolve().parent
D=P/'research_sources';D.mkdir(exist_ok=True)
SOFA='d300cc9af550afbd5c2167470676a5a7d47381a1'
COMP='1d0211c66053d11a70497a94b600db7e72588839'
paths=['modules/SofaConstraint/src/SofaConstraint/'+f for f in ['FreeMotionAnimationLoop.cpp','GenericConstraintSolver.cpp','GenericConstraintCorrection.cpp','LinearSolverConstraintCorrection.inl','FrictionContact.inl','FrictionContact.h','UnilateralInteractionConstraint.inl','UnilateralInteractionConstraint.h','ConstraintSolverImpl.cpp']]
paths+=['SofaKernel/modules/SofaCore/src/sofa/core/'+f for f in ['ConstraintParams.h','Mapping.h','Mapping.inl','behavior/MechanicalState.h','behavior/ConstraintCorrection.inl']]
paths+=['applications/plugins/PersistentContact/'+f for f in ['PersistentFrictionContact.inl','PersistentUnilateralInteractionConstraint.inl']]
paths+=['examples/Components/constraint/FrictionContact.scn']
items=[('sofa-framework/sofa',SOFA,p) for p in paths]
items += [('sofa-framework/Compliant',COMP,p) for p in ['odesolver/CompliantImplicitSolver.cpp','numericalsolver/IterativeSolver.cpp','Compliant_test/CompliantImplicitSolver_test.cpp']]
trees={}
for repo,sha in [('sofa-framework/sofa',SOFA),('sofa-framework/Compliant',COMP)]:
 j=json.load(urllib.request.urlopen('https://api.github.com/repos/'+repo+'/git/trees/'+sha+'?recursive=1',timeout=30))
 (D/(repo.rsplit('/',1)[1]+'_tree.json')).write_text(json.dumps(j))
 trees[repo]={x['path']:x['sha'] for x in j['tree']}
def fetch(it):
 repo,sha,path=it;url=f'https://raw.githubusercontent.com/{repo}/{sha}/{path}'
 try:
  installed=Path('/data/home/3220251075/mcr_sim/mcr_env/sofa/src/sofa')/path
  expected=trees[repo][path]
  b=installed.read_bytes() if repo.endswith('/sofa') and installed.is_file() else b''
  attested=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==expected
  if not attested:
   u=f'https://api.github.com/repos/{repo}/git/blobs/{expected}'
   j=json.load(urllib.request.urlopen(u,timeout=25));b=base64.b64decode(j['content'])
  assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==expected
  name=repo.rsplit('/',1)[1]+'__'+path.replace('/','__');(D/name).write_bytes(b)
  return dict(repo=repo,commit=sha,path=path,url=f'https://github.com/{repo}/blob/{sha}/{path}',file=name,sha256=hashlib.sha256(b).hexdigest(),github_blob_sha=expected,installed_equal=installed.read_bytes()==b if installed.is_file() else None,transport='github_tree_attested_installed_bytes' if attested else 'github_blob_api',status='OK')
 except Exception as e:return dict(repo=repo,commit=sha,path=path,status='ERROR',error=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e: manifest=list(e.map(fetch,items))
(D/'manifest.json').write_text(json.dumps(manifest,indent=2))
for x in manifest:print(x['status'],x['path'],'installed_equal',x.get('installed_equal'),x.get('error',''))
