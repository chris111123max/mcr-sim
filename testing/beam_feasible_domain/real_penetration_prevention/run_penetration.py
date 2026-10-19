"""Bounded real prefix and exact live native fork; no position writes."""
from pathlib import Path
import importlib.util,json,hashlib,ctypes,os,time,traceback
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('validated_case',HERE/'validated_case.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
base,fast,Sofa,factory=v.base,v.fast,v.Sofa,v.factory
SOURCE=next(c for c in v.CASES if c['worker']==23)
with np.load(SOURCE['snapshot'],allow_pickle=False) as z:RAW=np.asarray(z['raw_actions'],dtype=np.float32)
assert hashlib.sha256(RAW.tobytes()).hexdigest()==SOURCE['action_sha256']
WORLD=False
MAX_PREFIX=1000
MAX_CASES=3
CAPTURE_CLEARANCE_MM=-0.05
class Done(BaseException):pass
def dump(p,x):v.save_json(p,x)
def builder(*args,**kw):
 rows=v.RAW_BUILDER(*args,**kw)
 return v.convert_rows(rows,kw['q_free']) if WORLD else rows

def dense_check(env,q):
 adapter=env._beam_adapter;geometry=adapter._geometry();parts=[];tags=[];locations=[];bounds=[]
 elements=geometry['active_elements'];tip_first=max(0,len(elements)-3)
 for k,e in enumerate(elements):
  a,b=q[np.asarray(e['nodes'],dtype=int)];L=e['rest_length_m'];p0=a[:3];p3=b[:3]
  p1=p0+Rotation.from_quat(a[3:]).apply([L/3,0,0]);p2=p3+Rotation.from_quat(b[3:]).apply([-L/3,0,0])
  linear=np.dot(p1-p0,p2-p1)<0 and np.linalg.norm(p3-p0)<0.4*L
  bound=np.linalg.norm(p3-p0) if linear else 3*max(np.linalg.norm(p1-p0),np.linalg.norm(p2-p1),np.linalg.norm(p3-p2))
  n=max(1,int(np.ceil(max(L,bound)/1e-5)));t=np.linspace(0,1,n+1);u=1-t
  pts=u[:,None]*p0+t[:,None]*p3 if linear else u[:,None]**3*p0+3*(u*u*t)[:,None]*p1+3*(u*t*t)[:,None]*p2+t[:,None]**3*p3
  parts.append(pts);tags.extend([k>=tip_first]*len(pts));locations.extend([(k,float(x)) for x in t]);bounds.append(bound/n)
 points=np.concatenate(parts);clearance=np.asarray(adapter.query(points),dtype=float)
 if not np.isfinite(clearance).all():raise RuntimeError('INDEPENDENT_SDF_NONFINITE_NO_IGNORING')
 from mcr_sim.sdf_hard_constraint import sample_sdf_clearance_and_outward
 _,_,valid=sample_sdf_clearance_and_outward(env,points)
 mask=np.asarray(tags,dtype=bool);worst=int(np.argmin(clearance));i,t=locations[worst]
 return {'clearance_mm':float(clearance.min()*1000),'body_clearance_mm':float(clearance[~mask].min()*1000) if (~mask).any() else None,
  'tip_clearance_mm':float(clearance[mask].min()*1000),'negative_samples':int((clearance<0).sum()),
  'body_negative_samples':int(((clearance<0)&~mask).sum()),'tip_negative_samples':int(((clearance<0)&mask).sum()),
  'sample_count':len(clearance),'invalid_normal_count':int((~np.asarray(valid,dtype=bool)).sum()),
  'negative_invalid_normal_count':int(((clearance<0)&~np.asarray(valid,dtype=bool)).sum()),
  'max_spacing_bound_mm':float(max(bounds)*1000),'covered_active_elements':len(elements),
  'coverage':'all physically deployed Beam elements; inactive tail outside inlet excluded',
  'radius_subtracted_m':float(env.catheter_radius),'worst_element':elements[i],'worst_t':t,'worst_point_m':points[worst].tolist(),
  'strict_nonpenetration':bool(clearance.min()>=0),'aux_margin_0p100':bool(clearance.min()>=1e-4)}

class CaptureEnv(v.MultiEnv):
 def __init__(self,*a,**kw):
  self.active_case=False;self.capture_count=0;self.case_dir=None;self.saved=None;self.completed_cases=[]
  self.free_negative_count=0;self.worst_free_mm=1e9;self.last_capture_step=-100;self.guard_errors=[]
  super().__init__(*a,**kw)
 def _attach_beam_safety(self):
  if self._beam_attached:return
  base.BeamSafetyMCREnv._attach_beam_safety(self)
  reg=base.REGISTRY[id(self._sofa_root_node)];self.native=reg['solver'];self.ctrl=reg['controller'];self.ctrl.bind(self,self.native)
  original=self._beam_planner._collision_begin
  def hook(source):
   global WORLD
   WORLD=False
   original(source)
   qprev=fast._as_array(self._beam_dofs.position).copy();qfree=fast._as_array(self._beam_dofs.free_position).copy()
   points,profile=fast._profile(self._beam_adapter,qfree);cl=float(profile.min()*1000)
   if not self.active_case:
    self.free_negative_count+=int(cl<0);self.worst_free_mm=min(self.worst_free_mm,cl)
    if cl>CAPTURE_CLEARANCE_MM or self._beam_rl_step-self.last_capture_step<5:return
    self.capture_count+=1;self.last_capture_step=self._beam_rl_step
    self.case_id=f'state{self.capture_count}_step{self._beam_rl_step}_sub{self._beam_substep}'
    self.case_dir=HERE/'results'/self.case_id;self.case_dir.mkdir(exist_ok=False)
    self.first_sub=int(self._beam_substep);self.active_case=True;self.substeps=[];self.target_animation_calls=1;self.continuity=None;self.guard_errors=[]
    self.saved={'q_prev':qprev.copy(),'q_free':qfree.copy()}
    actions=RAW[:self._beam_rl_step]
    np.savez_compressed(self.case_dir/'real_free_snapshot.npz',q_prev=qprev,q_free=qfree,raw_actions=actions,free_points=points,free_clearance=profile)
    sdfpath=Path(self.sdf_grid.source_path)
    metadata={'source_saved_actions':SOURCE,'action_sha256':hashlib.sha256(actions.tobytes()).hexdigest(),
     'rl_step':int(self._beam_rl_step),'substep':self.first_sub,'seed':int(SOURCE['case']['seed']),
     'geometry':self._beam_adapter.solver_geometry(),'asset_T_env_sim':np.asarray(self.asset_T_env_sim).tolist(),
     'asset_offset_sim':np.asarray(self.asset_offset_sim).tolist(),'sdf_path':str(sdfpath),
     'sdf_sha256':hashlib.sha256(sdfpath.read_bytes()).hexdigest(),'previous':dense_check(self,qprev),'free':dense_check(self,qfree),
     'capture':'actual CollisionBeginEvent, normal physical prefix; process fork preserves native context'}
    dump(self.case_dir/'metadata.json',metadata)
    fd=v.verify_fd(self._beam_adapter,qfree,v.RAW_BUILDER(adapter=self._beam_adapter,q_prev=qprev,q_free=qfree,dense_points=points,dense_clearance=profile))
    dump(self.case_dir/'independent_world_fd.json',fd)
    if not fd['pass']:raise RuntimeError('WORLD_FD_GATE_FAILED')
    for branch,passes in [('B',1),('C',2)]:
     pid=os.fork()
     if pid==0:
      self.child=True;self.branch=branch;self.passes=passes;break
     _,status=os.waitpid(pid,0)
     if not os.WIFEXITED(status) or os.WEXITSTATUS(status)!=0:dump(self.case_dir/f'{branch}_process_error.json',{'wait_status':status})
    if not self.child:self.branch='A';self.passes=1
    self.state_match={'q_prev':v.compare_state(fast._as_array(self._beam_dofs.position),self.saved['q_prev']),
     'q_free':v.compare_state(fast._as_array(self._beam_dofs.free_position),self.saved['q_free'])}
    self.state_match['pass']=self.state_match['q_prev']['pass'] and self.state_match['q_free']['pass']
    if not self.state_match['pass']:raise RuntimeError('LIVE_FORK_STATE_MISMATCH')
   WORLD=self.branch!='A';v.WORLD_MODE=WORLD;v.OUT=self.case_dir
   v.CASE=dict(SOURCE,id=self.case_id,step=int(self._beam_rl_step),substep=self.first_sub)
   if WORLD and bool(self._beam_constraint.enabled.value):
    raw=base._read_rows(self._beam_constraint,fast._select_rows(profile))
    fast.write_rows(self._beam_constraint,v.convert_rows(raw,qfree))
   if self.substeps:
    self.continuity=v.compare_state(qprev,self.last_committed)
    if not self.continuity['pass']:raise RuntimeError('SUBSTEP2_CONTINUITY_FAILED')
   self.current={'substep':int(self._beam_substep),'q_prev':qprev,'q_free':qfree,'free_points':points,'free_profile':profile,
    'geometry':self._beam_adapter.solver_geometry(),'free_clearance_mm':cl}
   self.native.auditPath.value=str(self.case_dir/f'{self.branch}_sub{self._beam_substep}_native.jsonl')
   self.native.requestedPasses.value=self.passes;self.native.relinearizationEnabled.value=True
   self.native.consistentContactReference.value=self.branch=='C'
   required=bool(self._beam_planner._record.get('rows_required'))
   self.ctrl.arm(qprev,qfree,profile,self.branch,initial_row_build_wall_ms=base.LAST_ROW_BUILD_WALL_MS if required else None)
  self._beam_planner._collision_begin=hook
 def _audit_committed(self,reason):
  try:return super()._audit_committed(reason)
  except RuntimeError as e:
   if not self.active_case or 'BEAM_SAFETY_COMMITTED_PENETRATION' not in str(e):raise
   self.guard_errors.append(str(e))
   return float(fast._profile(self._beam_adapter,fast._as_array(self._beam_dofs.position))[1].min())
 def branch_result(self):
  return {'branch':self.branch,'case_id':self.case_id,'first_substep':self.first_sub,'state_match':self.state_match,'substeps':self.substeps,
   'jacobian_frame':'world' if self.branch!='A' else 'production_local','passes_per_physics_substep':self.passes,
   'consistent_contact_reference':self.branch=='C','target_animate_calls':self.target_animation_calls,
   'expected_animate_calls':3-self.first_sub,'extra_animate_calls':max(0,self.target_animation_calls-(3-self.first_sub)),
   'production_modified':False,'ppo_started':False,'direct_position_write':False,'rollback':False,'projection':False,'action_shielding':False,
   'diagnostic_committed_guard_continuation':self.guard_errors}
 def _beam_traced_animate(self,root,dt):
  global WORLD
  start=time.perf_counter();was_active=self.active_case
  if was_active:self.target_animation_calls+=1
  if abs(float(dt)-0.005)>1e-12:raise RuntimeError('PHYSICS_DT_CHANGED')
  error=None
  try:result=base.BeamSafetyMCREnv._beam_traced_animate(self,root,dt)
  except Exception as e:
   if not self.active_case:raise
   error=f'{type(e).__name__}:{e}';result=None
  if not self.active_case:return result
  v.OUT=self.case_dir;v.WORLD_MODE=self.branch!='A'
  summary=v.MultiEnv.capture_substep(self,(time.perf_counter()-start)*1000,error)
  q=fast._as_array(self._beam_dofs.position)
  summary['entire_beam_committed']=dense_check(self,q);summary['entire_beam_free']=dense_check(self,self.current['q_free'])
  for st in summary['stages']:
   z=np.load(self.case_dir/st['npz']);st['entire_beam']=dense_check(self,z['q_committed'])
  summary['guard_failures']=list(self.guard_errors)
  dump(self.case_dir/f'branch_{self.branch}.json',self.branch_result())
  if int(self._beam_substep)==2 or error:
   if self.child:raise Done()
   self.completed_cases.append(self.case_id)
   cpath=self.case_dir/'branch_C.json';c=json.loads(cpath.read_text()) if cpath.exists() else None
   cfail=c is None or any(not s['entire_beam_committed']['strict_nonpenetration'] or s['error'] for s in c['substeps'])
   self.native.relinearizationEnabled.value=False;self.native.auditPath.value='';self.ctrl.active=False;WORLD=False;v.WORLD_MODE=False
   self.active_case=False
   if cfail or self.capture_count>=MAX_CASES:raise Done()
  return result

def main():
 plugin=HERE/'results/native_build/lib/libMCRContactReferenceConsistency.so';ctypes.CDLL(str(plugin),mode=ctypes.RTLD_GLOBAL)
 base.ORIGINAL_ROW_BUILDER=builder;fast.build_rows_fast_local_fd=base._timed_row_builder
 def init(root,*a,**kw):
  response=root.getObject('Response')
  if response is None:raise RuntimeError('CONTACT_MANAGER_NOT_FOUND')
  response.response.value='ReferenceFrictionContactConstraint'
  return base._patched_sim_init(root,*a,**kw)
 Sofa.Simulation.init=init;factory.BeamSafetyMCREnv=CaptureEnv
 cfg=json.loads((base.RUN/'logs/run_config.json').read_text());args=SimpleNamespace(**cfg)
 args.local_n_envs=1;args.n_envs=1;args.distributed_rank=SOURCE['worker'];args.render='headless';args.force_model='';args.beam_safety_plugin=str(base.PROD_PLUGIN)
 vec=None;env=None;started=time.perf_counter();out={};steps=0
 try:
  vec=factory.build_env(args);env=vec.envs[0].unwrapped
  vessel=SOURCE['case']['vessel'];route=int(SOURCE['case']['target'].rsplit('_',1)[1])
  env.set_training_model_sampling_weights({vessel:1.0});env.set_training_route_sampling_weights({vessel:{route:1.0}})
  vec.seed(int(SOURCE['case']['seed']));vec.reset()
  try:
   for index,action in enumerate(RAW[:MAX_PREFIX]):
    steps=index+1;_,_,done,_=vec.step(action[None,:])
    if done[0]:break
    if steps%100==0:print('PREFIX_PROGRESS',steps,'captures',env.capture_count,'worst_free_mm',env.worst_free_mm,flush=True)
  except Done:pass
  out={'status':'BOUNDED_CAPTURE_FINISHED','rl_steps_executed':steps,'captured_states':env.capture_count,
   'free_negative_substeps':env.free_negative_count,'worst_free_clearance_mm':env.worst_free_mm,'completed_cases':env.completed_cases}
 except Exception as e:
  out={'status':'ERROR','error':str(e),'traceback':traceback.format_exc(),'rl_steps_executed':steps};traceback.print_exc()
 finally:
  if vec:vec.close()
  Sofa.Simulation.init=base.ORIGINAL_SIM_INIT
 if env is not None and env.child:os._exit(0 if out.get('status')=='BOUNDED_CAPTURE_FINISHED' else 1)
 out['wall_s']=time.perf_counter()-started;dump(HERE/'results/run_summary.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
