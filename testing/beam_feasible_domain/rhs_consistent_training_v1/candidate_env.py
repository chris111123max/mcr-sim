"""Independent experiment adapter; production sources are not modified."""
from pathlib import Path
import os,json,hashlib,ctypes,time,shutil,traceback
import numpy as np
import Sofa
from mcr_sim.beam_safety_env import BeamSafetyMCREnv
from mcr_sim import beam_sdf_unilateral_fast as fast
from world_rows import convert_rows
from dense_geometry import dense_check
HERE=Path(__file__).resolve().parent
EXP=Path(os.environ['MCR_CANDIDATE_RUN_DIR'])
PLUGIN=HERE/'native/libMCRContactReferenceConsistency.so'
ORIGINAL_INIT=Sofa.Simulation.init
RAW_BUILDER=fast.build_rows_fast_local_fd
PLUGIN_HANDLE=None
INSTANCE=0
NATIVE_OBJECTS={}

def create_memfd():
 libc=ctypes.CDLL(None,use_errno=True);fn=libc.memfd_create
 fn.argtypes=[ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
 fd=fn(b'mcr_native_frame',0)
 if fd<0:raise OSError(ctypes.get_errno(),'native memory journal unavailable')
 return fd

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return str(x)
 return x

def save(p,x):p.write_text(json.dumps(clean(x),indent=2,allow_nan=False))
def world_builder(*a,**kw):return convert_rows(RAW_BUILDER(*a,**kw),kw['q_free'])

def read_rows(c):
 if not bool(c.enabled.value):
  return {'rowOffsets':np.array([0],dtype=np.int64),'dofIndices':np.array([],dtype=np.int64),'linearJacobian':np.empty((0,3)),
   'angularJacobian':np.empty((0,3)),'freeViolations':np.array([]),'sourceClearances':np.array([])}
 return {k:np.asarray(getattr(c,k).value).copy() for k in ('rowOffsets','dofIndices','linearJacobian','angularJacobian','freeViolations','sourceClearances')}

class StageController(Sofa.Core.Controller):
 def __init__(self,**kw):
  kw['listening']=True;super().__init__(**kw);self.env=None
 def onScriptEvent(self,event):
  if self.env is None or not isinstance(event,dict):return
  name=str(event.get('event_name',''));prefix='MCR_NONLINEAR_RELINEARIZATION_STAGE_'
  if not name.startswith(prefix):return
  e=self.env;s=e.native;n=int(name[len(prefix):])
  try:
   q=fast._as_array(e._beam_dofs.position).copy();pts,cl=fast._profile(e._beam_adapter,q)
   record={'stage':n,'q_linearization':e.linearization.copy(),'q_stage':q,'rows':e.rows,
    'dense_min_mm':float(cl.min()*1000),'activeCount':int(e._beam_constraint.activeCount.value),
    'solver_error':float(s.currentError.value),'solver_iterations':int(s.currentIterations.value),
    'global_rows':int(s.currentNumConstraints.value)}
   e.stages.append(record)
   if record['activeCount']!=len(e.rows['freeViolations']):raise RuntimeError('CANDIDATE_ACTIVE_COUNT_MISMATCH')
   if n<int(s.requestedPasses.value):
    rows=fast.build_rows_fast_local_fd(adapter=e._beam_adapter,q_prev=q,q_free=q,dense_points=pts,dense_clearance=cl,requested_margin_m=fast.MARGIN_M)
    fast.write_rows(e._beam_constraint,rows)
    r=e._beam_planner._record;r['row_count']=len(rows['free_violations']);r['rows_armed']=r['rows_required']=bool(r['row_count'])
    e.linearization=q.copy();e.rows=read_rows(e._beam_constraint)
    s.stageEventStatus.value=f'ROWS_REBUILT_FOR_STAGE_{n+1}'
   else:s.stageEventStatus.value=f'CAPTURED_STAGE_{n}'
  except Exception as exc:
   e.native_error=traceback.format_exc();s.stageEventStatus.value='ERROR:'+str(exc)

def native_init(root,*a,**kw):
 old=[o for o in root.objects if o.getClassName()=='GenericConstraintSolver']
 if len(old)!=1:raise RuntimeError('EXPECTED_ONE_GENERIC_SOLVER')
 o=old[0];settings={k:getattr(o,k).value for k in ['tolerance','maxIterations','printLog','scaleTolerance','allVerified']}
 root.removeObject(o)
 solver=root.addObject('TestOnlyRelinearizingGenericConstraintSolver',name='RHSConsistentNativeSolver',relinearizationEnabled=True,
  requestedPasses=2,consistentContactReference=True,**settings)
 controller=StageController(name='RHSConsistentStages');root.addObject(controller)
 NATIVE_OBJECTS[id(root)]={'solver':solver,'controller':controller}
 root.getObject('Response').response.value='ReferenceFrictionContactConstraint'
 return ORIGINAL_INIT(root,*a,**kw)

class CandidateEnv(BeamSafetyMCREnv):
 def __init__(self,*a,**kw):
  global PLUGIN_HANDLE,INSTANCE
  if PLUGIN_HANDLE is None:PLUGIN_HANDLE=ctypes.CDLL(str(PLUGIN),mode=ctypes.RTLD_GLOBAL)
  Sofa.Simulation.init=native_init;fast.build_rows_fast_local_fd=world_builder
  INSTANCE+=1;self.worker_dir=EXP/'beam_records'/f'worker_{os.getpid()}_{INSTANCE}';self.worker_dir.mkdir(parents=True)
  self.fd=create_memfd();self.frame=None;self.stages=[];self.rows=None;self.native_error=None
  self.actions=[];self.episode_index=0;self.seed_value=None;self.completed_episodes=0
  self.totals={'physics_substeps':0,'free_negative':0,'committed_negative':0,'min_committed_mm':None,'min_free_mm':None}
  self.trace=(self.worker_dir/'substeps.jsonl').open('a',buffering=1)
  kw['beam_safety_validation_mode']='debug'
  super().__init__(*a,**kw)
 def _validate_solver(self):
  objs=[o for o in self._sofa_root_node.objects if o.getClassName() in ('GenericConstraintSolver','LCPConstraintSolver','TestOnlyRelinearizingGenericConstraintSolver')]
  if len(objs)!=1 or objs[0].getClassName()!='TestOnlyRelinearizingGenericConstraintSolver':raise RuntimeError('CANDIDATE_NATIVE_SOLVER_NOT_ACTIVE')
 def _attach_beam_safety(self):
  if self._beam_attached:return
  super()._attach_beam_safety()
  objects=NATIVE_OBJECTS[id(self._sofa_root_node)];self.native=objects['solver'];objects['controller'].env=self
  self.native.auditPath.value=f'/proc/self/fd/{self.fd}'
  original=self._beam_planner._collision_begin
  def hook(source):
   os.ftruncate(self.fd,0);self.stages=[];self.native_error=None
   self.frame={'q_prev':fast._as_array(self._beam_dofs.position).copy(),'q_free':fast._as_array(self._beam_dofs.free_position).copy()}
   original(source)
   self.frame['free']=dense_check(self,self.frame['q_free']);self.frame['geometry']=self._beam_adapter.solver_geometry()
   t=self.totals;free=self.frame['free']['clearance_mm'];t['physics_substeps']+=1;t['free_negative']+=int(free<0)
   t['min_free_mm']=free if t['min_free_mm'] is None else min(t['min_free_mm'],free)
   self.linearization=self.frame['q_free'].copy();self.rows=read_rows(self._beam_constraint)
   self.frame['initial_rows']={k:v.copy() for k,v in self.rows.items()}
  self._beam_planner._collision_begin=hook
  self.write_status('READY')
  print('CANDIDATE_WORKER_ENABLED',json.dumps({'pid':os.getpid(),'worker_dir':str(self.worker_dir),'solver':self.native.getClassName(),
   'world_rotation_J':True,'consistent_contact_reference':bool(self.native.consistentContactReference.value),
   'passes':int(self.native.requestedPasses.value),'dt':float(self._sofa_root_node.dt.value),'independent_check':'every physics substep, active Beam <=10um, radius subtracted'}),flush=True)
 def reset(self,*a,**kw):
  self.episode_index+=1;self.actions=[];self.frame=None
  if kw.get('seed') is not None:self.seed_value=int(kw['seed'])
  return super().reset(*a,**kw)
 def step(self,action):
  self.actions.append(np.asarray(action,dtype=np.float32).copy())
  result=super().step(action)
  if result[2] or result[3]:self.completed_episodes+=1;self.write_status('RUNNING')
  return result
 def context(self):
  return {'pid':os.getpid(),'seed':self.seed_value,'episode':self.episode_index,'rl_step':int(self._beam_rl_step),'substep':int(self._beam_substep),
   'vessel':getattr(self,'current_sampling_model',None),'target':getattr(self,'current_target_route_id',None)}
 def write_status(self,status):
  x=dict(self.context(),status=status,completed_episodes=self.completed_episodes,**self.totals)
  p=self.worker_dir/'status.json';t=p.with_suffix('.tmp');save(t,x);t.replace(p)
 def native_records(self):
  os.lseek(self.fd,0,os.SEEK_SET);parts=[]
  while True:
   b=os.read(self.fd,1024*1024)
   if not b:break
   parts.append(b)
  raw=b''.join(parts).decode();return raw,[json.loads(x) for x in raw.splitlines() if x]
 def _audit_committed(self,reason):
  q=fast._as_array(self._beam_dofs.position).copy();check=dense_check(self,q)
  self._beam_audit_count+=1;self._beam_worst_audited_clearance_m=min(self._beam_worst_audited_clearance_m,check['clearance_mm']/1000)
  self.last_independent=check
  if check['clearance_mm']<0:
   self.totals['committed_negative']+=1;self.totals['min_committed_mm']=check['clearance_mm']
   self.capture_failure('STRICT_COMMITTED_PENETRATION',q,check)
   raise RuntimeError('STRICT_COMMITTED_PENETRATION:'+repr(check['clearance_mm'])+' mm')
  return check['clearance_mm']/1000

 def capture_failure(self,reason,q=None,check=None):
  d=self.worker_dir/'failure'
  if d.exists():return
  d.mkdir();q=fast._as_array(self._beam_dofs.position).copy() if q is None else q
  raw,records=self.native_records();(d/'native_frame.jsonl').write_text(raw)
  arrays={'q_committed':q,'raw_actions':np.asarray(self.actions,dtype=np.float32),'smoothed_action':np.asarray(getattr(self,'_last_smoothed_action',[]))}
  if self.frame is not None:
   arrays.update(q_prev=self.frame['q_prev'],q_free=self.frame['q_free'],**self.frame['initial_rows'])
  np.savez_compressed(d/'frame.npz',**arrays)
  for st in self.stages:
   np.savez_compressed(d/f'stage{st["stage"]}.npz',q_linearization=st['q_linearization'],q_committed=st['q_stage'],**st['rows'])
  sdf=Path(self.sdf_grid.source_path);shutil.copy2(sdf,d/sdf.name)
  meta={'reason':reason,'context':self.context(),'committed':check,'free':self.frame['free'] if self.frame else None,
   'geometry':self._beam_adapter.solver_geometry(),'asset_T_env_sim':np.asarray(self.asset_T_env_sim),
   'asset_offset_sim':np.asarray(self.asset_offset_sim),'asset_source_to_sim_scale':self.asset_source_to_sim_scale,
   'catheter_radius_m':self.catheter_radius,'sdf_source':str(sdf),'sdf_sha256':hashlib.sha256(sdf.read_bytes()).hexdigest(),
   'action_sha256':hashlib.sha256(arrays['raw_actions'].tobytes()).hexdigest(),'checkpoint_source':os.environ['MCR_CANDIDATE_RESUME'],
   'scene_kwargs':getattr(self,'_create_scene_kwargs',None),'native_error':self.native_error}
  save(d/'metadata.json',meta)
  cfg=EXP/'logs/run_config.json'
  if cfg.exists():shutil.copy2(cfg,d/'run_config.json')
  self.write_status('FAIL')
  try:
   fd=os.open(str(EXP/'FAILURE.json'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
   os.write(fd,json.dumps(clean({'artifact':str(d),**meta})).encode());os.close(fd)
  except FileExistsError:pass
  print('CANDIDATE_FAIL_FAST',reason,'artifact',d,flush=True)
 def _beam_traced_animate(self,root,dt):
  if (EXP/'FAILURE.json').exists():raise RuntimeError('CANDIDATE_PEER_FAILURE_STOP')
  if abs(float(dt)-0.005)>1e-12:raise RuntimeError('CANDIDATE_DT_MISMATCH')
  try:
   result=super()._beam_traced_animate(root,dt)
   if self.native_error:raise RuntimeError('CANDIDATE_ROW_REBUILD_FAILED:'+self.native_error)
   if int(self.native.nativeSolveCalls.value)!=2 or int(self.native.completedStages.value)!=2 or int(self.native.mappingRefreshCount.value)!=1:
    raise RuntimeError('CANDIDATE_NATIVE_EXECUTION_MISMATCH')
   if str(self.native.relinearizationStatus.value)!='PASS_NATIVE_SAME_SUBSTEP_RELINEARIZATION':raise RuntimeError('CANDIDATE_NATIVE_STATUS_FAILED')
   raw,audit=self.native_records();built=[x for x in audit if x['phase']=='built'];solved=[x for x in audit if x['phase']=='solved']
   if len(built)!=2 or len(solved)!=2:raise RuntimeError('CANDIDATE_NATIVE_AUDIT_INCOMPLETE')
   refs=[];contacts=[]
   for i,x in enumerate(built):
    if (not x['reference_aliases_free'] if i==0 else not x['reference_aliases_position']):raise RuntimeError('CANDIDATE_REFERENCE_VECTOR_MISMATCH')
    geo=[c for group in x['contact_geometry'] for c in group];contacts.append(len(geo))
    for c in geo:
     refs.extend([float(np.max(np.abs(np.asarray(c['P'])-c['cparams_P']))),float(np.max(np.abs(np.asarray(c['Q'])-c['cparams_Q'])))])
     if not c['selected_reference']:raise RuntimeError('CANDIDATE_OLD_CONTACT_RHS_USED')
   referr=max(refs or [0])
   if referr>1e-12:raise RuntimeError('CANDIDATE_CONTACT_REFERENCE_MISMATCH')
   cl=self.last_independent['clearance_mm'];free=self.frame['free']['clearance_mm'];t=self.totals
   t['min_committed_mm']=cl if t['min_committed_mm'] is None else min(t['min_committed_mm'],cl)
   t['min_free_mm']=free if t['min_free_mm'] is None else min(t['min_free_mm'],free)
   row={'context':self.context(),'free':self.frame['free'],'committed':self.last_independent,
    'contact_rows':[n*3 for n in contacts],'beam_rows':[len(s['rows']['freeViolations']) for s in self.stages],
    'activeCounts':[s['activeCount'] for s in self.stages],'native_solve_calls':2,'reference_consistency_error_m':referr,
    'solver_errors':[s['solver_error'] for s in self.stages],'solver_iterations':[s['solver_iterations'] for s in self.stages]}
   self.trace.write(json.dumps(clean(row),allow_nan=False)+'\n');self.write_status('RUNNING')
   return result
  except Exception as exc:
   self.capture_failure(str(exc));raise
 def close(self):
  try:super().close()
  finally:
   self.trace.close();os.close(self.fd)
