"""One recorded-action prefix; POSIX process fork at real CollisionBeginEvent."""
from capture_failure import *
from scipy.spatial.transform import Rotation
class TargetDone(BaseException):pass
META=json.loads((OUT/'final_stress_metadata.json').read_text())
SNAP=np.load(OUT/'final_stress_snapshot.npz')
TARGET=int(META['rl_step'])
class ForkEnv(BeamSafetyMCREnv):
    def __init__(self,*a,**kw):
        self.forked=False;self.children=[];self.branch='baseline';self.child=False
        super().__init__(*a,**kw)
    def _attach_beam_safety(self):
        super()._attach_beam_safety()
        planner=self._beam_planner; original=planner._collision_begin
        def hook(source):
            original(source)
            if self._beam_rl_step!=TARGET or self._beam_substep!=1 or self.forked:return
            qp=fast._as_array(self._beam_dofs.position).copy()
            qf=fast._as_array(self._beam_dofs.free_position).copy()
            errors={'q_prev_max_abs_m':float(np.max(abs(qp-SNAP['q_prev']))),
                    'q_free_max_abs_m':float(np.max(abs(qf-SNAP['q_free'])))}
            savejson(OUT/'native_fork_state_match.json',errors)
            if max(errors.values())>1e-10:
                self.branch='STATE_MISMATCH';self.forked=True;return
            self.forked=True
            for branch in ('world_angular_basis','solver_tolerance_1e9'):
                pid=os.fork()
                if pid==0:
                    self.child=True;self.branch=branch;self.children=[]
                    if branch=='world_angular_basis':
                        c=self._beam_constraint
                        indices=np.array(c.dofIndices.value,dtype=int)
                        j=np.array(c.angularJacobian.value,dtype=float)
                        c.angularJacobian.value=Rotation.from_quat(qf[indices,3:7]).apply(j).tolist()
                    else:
                        solver=next(o for o in self._sofa_root_node.objects if o.getClassName()=='GenericConstraintSolver')
                        solver.tolerance.value=1e-9
                    return
                self.children.append(pid)
                os.waitpid(pid,0)
        planner._collision_begin=hook
    def _beam_traced_animate(self,root,dt):
        try: result=super()._beam_traced_animate(root,dt)
        except RuntimeError as exc:
            if self._beam_rl_step==TARGET and self._beam_substep==1:
                self.record(str(exc));raise TargetDone()
            raise
        if self._beam_rl_step==TARGET and self._beam_substep==1:
            self.record(None);raise TargetDone()
        return result
    def record(self,error):
        q=fast._as_array(self._beam_dofs.position).copy()
        points,clear=fast._profile(self._beam_adapter,q)
        c=self._beam_constraint
        qf=SNAP['q_free'];dt=q[:,:3]-qf[:,:3]
        dw=(Rotation.from_quat(q[:,3:7])*Rotation.from_quat(qf[:,3:7]).inv()).as_rotvec()
        off=np.array(c.rowOffsets.value,dtype=int);ids=np.array(c.dofIndices.value,dtype=int)
        jt=np.array(c.linearJacobian.value);ja=np.array(c.angularJacobian.value)
        gf=np.array(c.freeViolations.value);pred=gf.copy()
        for row in range(len(gf)):
            for b in range(off[row],off[row+1]):pred[row]+=jt[b].dot(dt[ids[b]])+ja[b].dot(dw[ids[b]])
        solver=next(o for o in self._sofa_root_node.objects if o.getClassName()=='GenericConstraintSolver')
        d={'branch':self.branch,'rl_step':TARGET,'substep':1,'substep2_executed':False,
           'committed_clearance_mm':float(clear.min()*1000),'linear_row_min_gap_mm':float(pred.min()*1000),
           'solver_iterations':int(solver.currentIterations.value),'solver_error':float(solver.currentError.value),
           'solver_tolerance':float(solver.tolerance.value),'row_count':len(gf),'activeCount':int(c.activeCount.value),
           'error':error,'NaN_Inf':bool(not np.isfinite(q).all()),'direct_position_writes':False,
           'rollback_projection_shielding':False,'faithful_fork_method':'POSIX os.fork at CollisionBeginEvent'}
        savejson(OUT/('native_ab_'+self.branch+'.json'),d)
        np.savez_compressed(OUT/('native_ab_'+self.branch+'.npz'),q_committed=q,clearance=clear,linear_gap=pred)

def main():
    cfg=json.loads((RUN/'logs/run_config.json').read_text());args=SimpleNamespace(**cfg)
    case=META['case'];worker=int(case['seed'])-int(cfg['seed'])
    args.local_n_envs=1;args.n_envs=1;args.distributed_rank=worker;args.render='headless';args.force_model=''
    args.beam_safety_plugin=str(ROOT/'mcr_sim/native/beam_unilateral/_build/lib/libMCRBeamLinearizedUnilateral.so')
    factory.BeamSafetyMCREnv=ForkEnv
    vec=factory.build_env(args);env=vec.envs[0].unwrapped
    env.set_training_model_sampling_weights({case['vessel']:1.0})
    env.set_training_route_sampling_weights({case['vessel']:{int(case['target'].split('_')[-1]):1.0}})
    vec.seed(case['seed']);vec.reset()
    actions=SNAP['raw_actions'];assert len(actions)==TARGET
    try:
        for i,action in enumerate(actions):
            _,_,done,_=vec.step(action[None,:])
            if done[0]:raise RuntimeError('PREFIX_TERMINATED_BEFORE_TARGET')
            if (i+1)%500==0:print('PREFIX',i+1,flush=True)
        raise RuntimeError('TARGET_STOP_NOT_OBSERVED')
    except TargetDone:
        print('TARGET_DONE',env.branch,flush=True)
    finally:
        vec.close()
    if env.child:os._exit(0)
    print('NATIVE_AB_COMPLETE; no substep2',flush=True)
if __name__=='__main__':main()
