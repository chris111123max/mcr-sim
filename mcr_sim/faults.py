"""Typed collector failures; communication errors never imply a physics cause."""
from dataclasses import dataclass
import traceback

class SamplingStop(RuntimeError):
    """Sampling is drained and learner state is eligible for a fault snapshot."""

@dataclass
class CollectedStep:
    env_id: int
    result: object = None
    error: object = None
    worker_pid: object = None
    exitcode: object = None
    worker_log: object = None

    def error_info(self):
        return dict(env_id=self.env_id,worker_pid=self.worker_pid,exitcode=self.exitcode,
                    worker_log=self.worker_log,error=repr(self.error))

def valid_transition(result,shape=(41,),substeps=22):
    import numpy as np
    if not isinstance(result,(tuple,list)) or len(result)!=5:return False
    obs,reward,terminated,truncated,info=result
    if not isinstance(info,dict):return False
    try:
        return bool(info.get('transition_valid',False) and info.get('observation_valid',False)
                    and not info.get('simulation_error',False) and info.get('state_error') is None
                    and info.get('termination_reason')!='simulation_error'
                    and isinstance(terminated,(bool,np.bool_)) and isinstance(truncated,(bool,np.bool_))
                    and np.asarray(obs).shape==shape and np.isfinite(obs).all() and np.isfinite(reward)
                    and info.get('physics_substeps_attempted')==substeps
                    and info.get('physics_substeps_validated',0)==substeps)
    except (TypeError,ValueError,OverflowError):return False


def result_info(result):
    if isinstance(result,(tuple,list)) and len(result)==5 and isinstance(result[4],dict):
        return result[4]
    return {}


def isolated_lcp_failure(info):
    attempted=info.get('physics_substeps_attempted')
    validated=info.get('physics_substeps_validated')
    return (isinstance(info.get('failure_kind'),str) and info.get('failure_kind')=='lcp_nonconvergence' and info.get('simulation_error') is True
            and info.get('transition_valid') is False and isinstance(attempted,int)
            and isinstance(validated,int) and 1<=attempted<=22 and validated==attempted-1
            and info.get('failed_physics_substep')==attempted)


def diagnostic_value(value):
    """Preserve bad numeric diagnostics as explicit strings, never invent zeros."""
    import math
    import numpy as np
    from dataclasses import asdict,is_dataclass
    if is_dataclass(value):return diagnostic_value(asdict(value))
    if isinstance(value,np.ndarray):return diagnostic_value(value.tolist())
    if isinstance(value,np.generic):return diagnostic_value(value.item())
    if isinstance(value,float) and not math.isfinite(value):return 'NONFINITE:'+repr(value)
    if isinstance(value,dict):return {str(k):diagnostic_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [diagnostic_value(v) for v in value]
    if value is None or isinstance(value,(int,float,str,bool)):return value
    return repr(value)
