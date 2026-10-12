"""Independent experiment adapter; production sources are not modified."""
from pathlib import Path
import os,json,hashlib,ctypes,time,shutil,traceback
import numpy as np
import Sofa
from mcr_sim.beam_safety_env import BeamSafetyMCREnv
from mcr_sim import beam_sdf_unilateral_fast as fast
from world_rows import convert_rows
from dense_geometry import dense_check, query_clearance_only
from fast_guard import coarse_beam_clearance_guard
HERE=Path(__file__).resolve().parent
EXP=Path(os.environ['MCR_CANDIDATE_RUN_DIR'])
PENETRATION_REPORT_THRESHOLD_MM=0.100
NATIVE_AUDIT_NEAR_MM=0.150
TRACE_INTERVAL_SUBSTEPS=16
COARSE_SPACING_M=1.0e-4
COARSE_SWITCH_MARGIN_M=2.5e-4
PLUGIN=Path(os.environ.get('MCR_RHS_PLUGIN',str(HERE/'native/_build/lib/libMCRContactReferenceConsistency.so')))
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
   # A certified far-wall coarse pass has no Beam rows.  Ack the native
   # single-solve stage without copying all DOFs and Jacobian records.
   if e.frame is not None and e.frame.get('coarse_safe',False):
    if n!=1 or int(e._beam_constraint.activeCount.value)!=0:
     raise RuntimeError('COARSE_FAST_PATH_NATIVE_ROWS_MISMATCH')
    if not np.isfinite(float(s.currentError.value)):
     raise RuntimeError('CANDIDATE_NATIVE_NON_FINITE')
    s.requestedPasses.value=1
    s.stageEventStatus.value='CAPTURED_STAGE_1'
    return
   q=fast._as_array(e._beam_dofs.position).copy()
   pts=cl=None
   if n==1:
    free_mm=e.frame['free']['clearance_mm']
    risk=free_mm is not None and free_mm<=NATIVE_AUDIT_NEAR_MM
    if not risk and not e.step_invalid_sdf and int(s.currentNumConstraints.value)>0:
     before=e.frame['q_free'];move=float(np.linalg.norm(q[:,:3]-before[:,:3],axis=1).max())
     dots=np.clip(np.abs(np.sum(q[:,3:7]*before[:,3:7],axis=1)),0,1)
     span=2*float(np.linalg.norm(np.diff(before[:,:3],axis=0),axis=1).sum())
     bound=move+span*float((2*np.arccos(dots)).max())
     if free_mm-bound*1000<=NATIVE_AUDIT_NEAR_MM:
      try:
       pts,cl=fast._profile(e._beam_adapter,q)
       risk=float(cl.min()*1000)<=NATIVE_AUDIT_NEAR_MM
       if risk:e.totals['post_first_risk_triggers']+=1
      except RuntimeError:
       probe=e._beam_adapter.query(e._beam_adapter.sample(q))
       if np.isfinite(probe).all():raise
       e.step_invalid_sdf=True
       risk=False
    if risk and cl is None:
     try:pts,cl=fast._profile(e._beam_adapter,q)
     except RuntimeError:
      probe=e._beam_adapter.query(e._beam_adapter.sample(q))
      if np.isfinite(probe).all():raise
      e.step_invalid_sdf=True
      risk=False
    s.requestedPasses.value=2 if risk else 1
   record={'stage':n,'q_linearization':e.linearization.copy(),'q_stage':q,'rows':e.rows,
    'dense_min_mm':float(cl.min()*1000) if cl is not None else None,'activeCount':int(e._beam_constraint.activeCount.value),
    'solver_error':float(s.currentError.value),'solver_iterations':int(s.currentIterations.value),
    'global_rows':int(s.currentNumConstraints.value)}
   if not np.isfinite(record['solver_error']) or not np.isfinite(q).all():raise RuntimeError('CANDIDATE_NATIVE_NON_FINITE')
   e.stages.append(record)
   if record['activeCount']!=len(e.rows['freeViolations']):raise RuntimeError('CANDIDATE_ACTIVE_COUNT_MISMATCH')
   if n<int(s.requestedPasses.value):
    try:
     rows=fast.build_rows_fast_local_fd(adapter=e._beam_adapter,q_prev=q,q_free=q,dense_points=pts,dense_clearance=cl,requested_margin_m=fast.MARGIN_M)
    except RuntimeError as exc:
     if 'Invalid batched local FD SDF query' not in str(exc):raise
     # Near the finite SDF grid edge, even FD perturbations may be invalid.
     # Do not invent a Jacobian or repeat the native correction.
     e.step_invalid_sdf=True
     s.requestedPasses.value=n
     s.stageEventStatus.value=f'CAPTURED_STAGE_{n}'
    else:
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
  requestedPasses=1,consistentContactReference=True,auditDetail=False,**settings)
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
  self.last_status_wall=0.0
  self.actions=[];self.episode_index=0;self.seed_value=None;self.completed_episodes=0
  self.totals={'physics_substeps':0,'free_negative':0,'committed_negative':0,'committed_over_0p100_mm':0,'single_solves':0,'double_solves':0,'audited_substeps':0,'skipped_substeps':0,'post_first_risk_triggers':0,'tip_negative':0,'body_negative':0,'min_committed_mm':None,'min_free_mm':None,'invalid_sdf_substeps':0,'invalid_sdf_episodes':0,'severe_penetration_episodes':0,'coarse_safe_substeps':0,'fine_fallback_substeps':0,'coarse_invalid_substeps':0}
  self.trace=(self.worker_dir/'substeps.jsonl').open('a',buffering=65536)
  kw['beam_safety_validation_mode']='audit'
  kw['beam_safety_committed_penetration_limit_m']=PENETRATION_REPORT_THRESHOLD_MM/1000
  super().__init__(*a,**kw)
 def _validate_solver(self):
  objs=[o for o in self._sofa_root_node.objects if o.getClassName() in ('GenericConstraintSolver','LCPConstraintSolver','TestOnlyRelinearizingGenericConstraintSolver')]
  if len(objs)!=1 or objs[0].getClassName()!='TestOnlyRelinearizingGenericConstraintSolver':raise RuntimeError('CANDIDATE_NATIVE_SOLVER_NOT_ACTIVE')
 def _attach_beam_safety(self):
  if self._beam_attached:return
  super()._attach_beam_safety()
  # The Beam planner needs only signed clearance.  The default adapter
  # unnecessarily computes gradients at every 10-um point (six extra SDF
  # interpolation passes).  Override only this candidate env instance.
  self._beam_adapter.query=lambda points:query_clearance_only(self,points)
  objects=NATIVE_OBJECTS[id(self._sofa_root_node)];self.native=objects['solver'];objects['controller'].env=self
  self.native.auditPath.value=f'/proc/self/fd/{self.fd}'
  original=self._beam_planner._collision_begin
  def hook(source):
   # SOFA may dispatch the same collision-begin event through two Python
   # listener interfaces; never run the coarse planner twice per substep.
   key=(int(self._beam_planner.current_rl_step),int(self._beam_planner.current_substep))
   if key==self._beam_planner._last_key:
    return
   os.ftruncate(self.fd,0);self.stages=[];self.native_error=None
   self.frame={'q_prev':fast._as_array(self._beam_dofs.position),
    'q_free':fast._as_array(self._beam_dofs.free_position)}
   probe=coarse_beam_clearance_guard(
    self._beam_adapter,self.frame['q_free'],
    spacing_m=COARSE_SPACING_M,switch_margin_m=COARSE_SWITCH_MARGIN_M)
   self.frame['coarse_safe']=bool(probe['safe'])
   self.frame['coarse']=probe
   if probe['safe']:
    self._beam_planner._last_key=key
    self._beam_constraint.enabled.value=False
    self._beam_planner._record={
     'rl_step':key[0],'substep':key[1], 'event_source':source,
     'planning_status':'SAFE_FREE_NO_ROWS','reason':'COARSE_GUARD_FAR_WALL',
     'rows_required':False,'rows_armed':False,'row_count':0,
     'q_free_clearance_m':float(probe['minimum_clearance_m']),
     'q_free_worst_index':int(probe['worst_index']),
     'dense_sample_count':0,'coarse_sample_count':int(probe['sample_count']),
     'total_q_free_full_profile_count':0}
   else:
    original(source)
   rec=self._beam_planner._record
   # An out-of-grid free state cannot be linearized from the SDF.  Leave
   # native Contact active, disarm the Beam rows, and terminate this episode
   # after the original physics substep, rather than crashing all PPO workers.
   reason=str(rec.get('reason',''))
   free_outside=str(rec.get('planning_status'))=='FAIL' and reason.startswith('DENSE_BEAM_MEASUREMENT_FAILED:')
   fd_outside=str(rec.get('planning_status'))=='FAIL' and (
    reason.startswith('FAST_BEAM_UNILATERAL_ROW_BUILD_FAILED:') and
    'Invalid batched local FD SDF query' in reason)
   if free_outside or fd_outside:
    if free_outside:
     probe=self._beam_adapter.query(self._beam_adapter.sample(self.frame['q_free']))
     if np.isfinite(probe).all():raise RuntimeError('UNEXPECTED_BEAM_PLANNER_FAILURE:'+reason)
    self.step_invalid_sdf=True
    self._beam_constraint.enabled.value=False
    rec.update(planning_status='SAFE_FREE_NO_ROWS',reason='SDF_GRID_UNAVAILABLE_EPISODE_TERMINAL',
     rows_required=False,rows_armed=False,row_count=0,q_free_clearance_m=float('nan'))
   value=float(rec['q_free_clearance_m'])
   self.frame['free']={'clearance_mm':value*1000 if np.isfinite(value) else None,
    'sdf_out_of_grid':bool(self.step_invalid_sdf),
    'coarse_sample_count':int(probe['sample_count']),
    'coarse_lower_bound_mm':probe['lower_bound_m']*1000 if probe['lower_bound_m'] is not None else None,
    'source':'conservative 100um coarse safe bound' if probe['safe'] else
    'production 10um fine Beam profile'}
   self.native.auditPath.value=f'/proc/self/fd/{self.fd}' if (
    self.step_invalid_sdf or (self.frame['free']['clearance_mm'] is not None and
    self.frame['free']['clearance_mm']<=NATIVE_AUDIT_NEAR_MM)) else ''
   self.native.requestedPasses.value=1
   self.last_independent=None
   t=self.totals;free=self.frame['free']['clearance_mm'];t['physics_substeps']+=1
   t['coarse_safe_substeps' if probe['safe'] else 'fine_fallback_substeps']+=1
   t['coarse_invalid_substeps']+=int(probe['invalid_sdf'])
   if free is not None:
    t['free_negative']+=int(free<0)
    t['min_free_mm']=free if t['min_free_mm'] is None else min(t['min_free_mm'],free)
   self.linearization=None if probe['safe'] else self.frame['q_free'].copy()
   self.rows=read_rows(self._beam_constraint)
   if not probe['safe']:
    self.frame['initial_rows']={k:v.copy() for k,v in self.rows.items()}
  self._beam_planner._collision_begin=hook
  self.write_status('READY')
  print('CANDIDATE_WORKER_ENABLED',json.dumps({'pid':os.getpid(),'worker_dir':str(self.worker_dir),'solver':self.native.getClassName(),
   'world_rotation_J':True,'consistent_contact_reference':bool(self.native.consistentContactReference.value),
   'penetration_policy':'episode_terminate_gt_0p100mm_or_invalid_sdf','penetration_report_threshold_mm':PENETRATION_REPORT_THRESHOLD_MM,'compact_audit':True,'coarse_guard_um':COARSE_SPACING_M*1e6,
   'coarse_switch_margin_mm':COARSE_SWITCH_MARGIN_M*1000,
   'passes':int(self.native.requestedPasses.value),'dt':float(self._sofa_root_node.dt.value),'independent_check':'near-wall and periodic audit; skipped frames unknown'}),flush=True)
 def reset(self,*a,**kw):
  self.episode_index+=1;self.actions=[];self.frame=None;self.step_invalid_sdf=False;self.safety_failure_reason=None
  if kw.get('seed') is not None:self.seed_value=int(kw['seed'])
  return super().reset(*a,**kw)
 def step(self,action):
  self.step_min_clearance_mm=float('inf')
  self.step_invalid_sdf=False
  self.safety_failure_reason=None
  self.actions.append(np.asarray(action,dtype=np.float32).copy())
  result=super().step(action)
  if result[2] or result[3]:self.completed_episodes+=1;self.write_status('RUNNING')
  obs,reward,terminated,truncated,info=result;info=dict(info)
  if self.safety_failure_reason:
   terminated=True;truncated=False
   info['beam_surface_failure']=True
   info['beam_surface_failure_reason']=self.safety_failure_reason
   info['is_success']=False
  info['beam_penetration']=dict(self.totals)
  return obs,reward,terminated,truncated,info
 def _get_reward_features(self):
  features=super()._get_reward_features()
  severe=getattr(self,'step_min_clearance_mm',float('inf')) < -PENETRATION_REPORT_THRESHOLD_MM
  invalid=bool(self.step_invalid_sdf)
  if severe or invalid:
   self.safety_failure_reason='SDF_OUT_OF_GRID' if invalid else 'BEAM_SURFACE_PENETRATION_GT_0P100MM'
   self.current_out_of_vessel=True
   self.out_of_vessel_failure=True
   self.out_of_vessel_this_episode=True
   self.episode_success=False
   self.episode_success_2mm=False
   self.episode_safe_success=False
   self.episode_contact_free_success=False
   features=dict(features)
   features['successful_task']=0.0
   features['out_of_vessel_penalty']=1.0
   counter='invalid_sdf_episodes' if invalid else 'severe_penetration_episodes'
   self.totals[counter]+=1
  return features
 def context(self):
  return {'pid':os.getpid(),'seed':self.seed_value,'episode':self.episode_index,'rl_step':int(self._beam_rl_step),'substep':int(self._beam_substep),
   'vessel':getattr(self,'current_sampling_model',None),'target':getattr(self,'current_target_route_id',None)}
 def write_status(self,status):
  now=time.monotonic()
  if status=='RUNNING' and now-self.last_status_wall<5.0:return
  self.last_status_wall=now
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
  q=fast._as_array(self._beam_dofs.position).copy()
  check=dense_check(self,q)
  self._beam_audit_count+=1
  self.last_independent=check
  if check['sdf_out_of_grid']:
   self.step_invalid_sdf=True
   return float('nan')
  value=check['clearance_mm']
  self._beam_worst_audited_clearance_m=min(self._beam_worst_audited_clearance_m,value/1000)
  self.totals['tip_negative']+=int(check['tip_clearance_mm'] is not None and check['tip_clearance_mm']<0)
  self.totals['body_negative']+=int(check['body_clearance_mm'] is not None and check['body_clearance_mm']<0)
  self.step_min_clearance_mm=min(getattr(self,'step_min_clearance_mm',float('inf')),value)
  if value<0:
   self.totals['committed_negative']+=1
   prior=self.totals['min_committed_mm']
   self.totals['min_committed_mm']=value if prior is None else min(prior,value)
  self.totals['committed_over_0p100_mm']+=int(value < -PENETRATION_REPORT_THRESHOLD_MM)
  # Count every measured negative; only >0.100 mm terminates the episode.
  return value/1000

 def _should_audit_substep(self,record):
  if self.step_invalid_sdf:return True,'free_sdf_out_of_grid'
  if int(self.native.nativeSolveCalls.value)==2:return True,'adaptive_second_solve'
  return super()._should_audit_substep(record)
 def capture_failure(self,reason,q=None,check=None):
  d=self.worker_dir/'failure'
  if d.exists():return
  self.trace.flush()
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
   'action_sha256':hashlib.sha256(arrays['raw_actions'].tobytes()).hexdigest(),'checkpoint_source':os.environ.get('MCR_CANDIDATE_RESUME','FRESH_TRAINING_NO_CHECKPOINT'),
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
   calls=int(self.native.nativeSolveCalls.value)
   if calls not in (1,2) or int(self.native.completedStages.value)!=calls or int(self.native.mappingRefreshCount.value)!=calls-1:
    raise RuntimeError('CANDIDATE_NATIVE_EXECUTION_MISMATCH')
   if str(self.native.relinearizationStatus.value)!='PASS_NATIVE_SAME_SUBSTEP_RELINEARIZATION':raise RuntimeError('CANDIDATE_NATIVE_STATUS_FAILED')
   referr=None;contacts=[]
   if str(self.native.auditPath.value):
    raw,audit=self.native_records()
    built=[x for x in audit if x['phase']=='built']
    solved=[x for x in audit if x['phase']=='solved']
    if len(built)!=calls or len(solved)!=calls:raise RuntimeError('CANDIDATE_NATIVE_AUDIT_INCOMPLETE')
    refs=[]
    for i,x in enumerate(built):
     if (not x['reference_aliases_free'] if i==0 else not x['reference_aliases_position']):raise RuntimeError('CANDIDATE_REFERENCE_VECTOR_MISMATCH')
     geo=[c for group in x['contact_geometry'] for c in group];contacts.append(len(geo))
     for contact in geo:
      refs.extend([float(np.max(np.abs(np.asarray(contact['P'])-contact['cparams_P']))),
                   float(np.max(np.abs(np.asarray(contact['Q'])-contact['cparams_Q'])))])
      if not contact['selected_reference']:raise RuntimeError('CANDIDATE_OLD_CONTACT_RHS_USED')
    referr=max(refs or [0])
    if referr>1e-12:raise RuntimeError('CANDIDATE_CONTACT_REFERENCE_MISMATCH')
   checked=bool(self._beam_last_substep_audited);t=self.totals
   t['audited_substeps' if checked else 'skipped_substeps']+=1
   if self.step_invalid_sdf:t['invalid_sdf_substeps']+=1
   t['single_solves' if calls==1 else 'double_solves']+=1
   cl=self.last_independent['clearance_mm'] if checked else None
   if checked and cl is not None:
    t['min_committed_mm']=cl if t['min_committed_mm'] is None else min(t['min_committed_mm'],cl)
   row={'context':self.context(),'free':self.frame['free'],'committed':self.last_independent,
    'coarse_safe':bool(self.frame.get('coarse_safe',False)),
    'contact_rows':[n*3 for n in contacts],'beam_rows':[len(s['rows']['freeViolations']) for s in self.stages],
    'activeCounts':[s['activeCount'] for s in self.stages],'native_solve_calls':calls,'committed_audited':checked,'penetration_policy':'episode_terminate_gt_0p100mm_or_invalid_sdf','max_penetration_mm':max(0.0,-cl) if cl is not None else None,'reference_consistency_error_m':referr,
    'native_solve_ms':float(self.native.nativeSolveTotalWallMs.value),
    'native_map_ms':float(self.native.mappingTotalWallMs.value),
    'solver_errors':[s['solver_error'] for s in self.stages],'solver_iterations':[s['solver_iterations'] for s in self.stages]}
   if calls==2 or self.step_invalid_sdf or (checked and cl is not None and cl<0) or t['physics_substeps']%TRACE_INTERVAL_SUBSTEPS==0:
    self.trace.write(json.dumps(clean(row),allow_nan=False)+'\n')
   self.write_status('RUNNING')
   return result
  except Exception as exc:
   self.capture_failure(str(exc));raise
 def close(self):
  try:super().close()
  finally:
   self.write_status('CLOSED');self.trace.close();os.close(self.fd)
