"""Method B: at most one TEST-ONLY epoch; stop all workers on first failure."""
from capture_failure import *
class ContinuationCaptureEnv(CaptureEnv):
    def __init__(self,*a,**kw):
        seed=int(np.random.get_state()[1][0])
        self.case={'seed':seed,'vessel':'sampled_stage0','target':'sampled_stage0','method':'test_only_continuation'}
        self.capture_dir=OUT/('worker_'+str(seed-42))
        super().__init__(*a,**kw)
    def reset(self,*a,**kw):
        self.actions=[];self.capture=None;self.saved=False
        self.diag_episode=getattr(self,'diag_episode',0)+1
        return super().reset(*a,**kw)
    def snapshot(self,*a,**kw):
        self.case.update(vessel=getattr(self,'current_sampling_model',None),target=getattr(self,'current_target_route_id',None),episode=self.diag_episode)
        super().snapshot(*a,**kw)

def main():
    from mcr_sim.distributed.context import initialize_distributed
    context=initialize_distributed(requested_device='npu',enabled=False,requested_backend='')
    cfg=json.loads((RUN/'logs/run_config.json').read_text());args=SimpleNamespace(**cfg)
    from mcr_sim.distributed import configure_npu_execution
    configure_npu_execution(context.device.accelerator,enabled=bool(cfg['npu_fast_execution']))
    args.local_n_envs=32;args.n_envs=32;args.distributed_rank=0;args.render='headless'
    args.beam_safety_plugin=str(ROOT/'mcr_sim/native/beam_unilateral/_build/lib/libMCRBeamLinearizedUnilateral.so')
    factory.BeamSafetyMCREnv=ContinuationCaptureEnv
    vec=factory.build_env(args);model=DistributedPPO.load(str(CKPT),env=vec,device='npu:0')
    model.set_distributed_context(context)
    from mcr_sim.distributed import convert_to_npu_fused_adam
    if cfg['npu_fused_adam']:
        optimizer,status=convert_to_npu_fused_adam(model.policy.optimizer)
        assert status in ('enabled','already_enabled'),status
        model.policy.optimizer=optimizer
    model.tensorboard_log=str(OUT/'continuation_tb')
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(str(OUT/'continuation_metrics'),['csv','tensorboard']))
    assert model.n_steps==256 and model.batch_size==1024 and model.n_epochs==5
    print('TEST_ONLY_ONE_EPOCH',model.n_steps,model.batch_size,model.n_epochs,flush=True)
    started=time.monotonic();result={}
    try:
        model.learn(total_timesteps=100*2048,reset_num_timesteps=False)
        result={'status':'NO_FAILURE_WITHIN_ONE_TEST_EPOCH','num_timesteps':model.num_timesteps}
    except Exception as exc:
        result={'status':'STOPPED_FIRST_WORKER_FAILURE','error':str(exc),'num_timesteps':model.num_timesteps}
        traceback.print_exc()
    finally:
        for proc in vec.processes:
            if proc.is_alive(): proc.terminate()
        for proc in vec.processes:proc.join(timeout=5)
        result['runtime_s']=time.monotonic()-started
        result['saved_failure_metadata']=[str(p) for p in OUT.glob('worker_*/failure_metadata.json')]
        savejson(OUT/'continuation_result.json',result)
    print('END',result,flush=True)
if __name__=='__main__':main()
