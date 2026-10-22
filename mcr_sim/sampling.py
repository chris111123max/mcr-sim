"""Optional process-isolated SOFA collectors. No distributed learner or DDP."""
import multiprocessing as mp
import os
from pathlib import Path
import traceback
from multiprocessing.connection import wait
from .native_threads import limit_worker_environment
import numpy as np
from gymnasium import spaces


def _worker(connection, config, mode, log_dir=None):
    limit_worker_environment()
    if log_dir is not None:
        folder=Path(log_dir);folder.mkdir(parents=True,exist_ok=True)
        stream=(folder / ("worker_%d.log"%os.getpid())).open("a",buffering=1)
        os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
    env = None
    try:
        from .rl_env import MagneticNavigationEnv
        from .sofa_simulation import SofaBackend
        env = MagneticNavigationEnv(SofaBackend(config), config=config, mode=mode)
        connection.send((True, None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                result = env.reset(**payload)
                from .runtime_audit import audit
                result[1]["physics_runtime_audit"] = audit(env.backend, initial=True)
            elif command == "rng_state":
                import random
                result = dict(env=env.np_random.bit_generator.state,action=env.action_space.np_random.bit_generator.state,python=random.getstate(),numpy=np.random.get_state())
            elif command == "restore_rng":
                import random
                env.np_random.bit_generator.state=payload['env']
                env.action_space.np_random.bit_generator.state=payload['action']
                random.setstate(payload['python']);np.random.set_state(payload['numpy'])
                result = None
            elif command == "runtime_audit":
                from .runtime_audit import audit
                result = audit(env.backend, initial=True)
            elif command == "step":
                result = env.step(payload)
            else:
                raise ValueError("Unknown collector command: " + command)
            connection.send((True, result))
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        try:
            if env is not None:env.close()
        finally:connection.close()


class ProcessEnvironment:
    def __init__(self, config, mode="train", log_dir=None):
        limit_worker_environment()
        self.config = config
        self.mode=mode;self.log_dir=log_dir
        self.reset_count=0;self.recycles=0
        self._pending = None
        self._start()

    def _start(self):
        self._closed=False
        self._scene_initialized=False
        config=self.config;mode=self.mode;log_dir=self.log_dir
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        args=(child,config,mode) if log_dir is None else (child,config,mode,str(log_dir))
        self.process = context.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        child.close()
        try:
            self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if not self.connection.poll(120):
            raise TimeoutError("SOFA collector did not respond within 120 seconds")
        try:
            success, result = self.connection.recv()
        except EOFError as exc:
            from .memory_guard import snapshot
            self.process.join(timeout=.1)
            raise RuntimeError('SOFA worker died: pid=%s exitcode=%s cgroup=%r; inspect worker log and OOM counters' % (self.process.pid,self.process.exitcode,snapshot())) from exc
        if not success:
            raise RuntimeError("SOFA collector failed:\n" + result)
        return result

    def reset_async(self, **kwargs):
        if self._pending is not None:
            raise RuntimeError("Collect the pending operation before resetting this environment")
        # Native SOFA allocations accumulate across scene rebuilds. Reclaim only
        # at explicit episode resets, preserving RNG across process boundaries.
        if self.mode=='train' and self._scene_initialized and self.reset_count and self.reset_count%16==0:
            self.connection.send(('rng_state',None));state=self._receive()
            parent_action_rng=self.action_space.np_random.bit_generator.state
            cleanup=self.close()
            if not cleanup['stopped']:raise RuntimeError('Periodic recycle could not stop worker')
            self._start()
            self.action_space.np_random.bit_generator.state=parent_action_rng
            self.connection.send(('restore_rng',state));self._receive()
            self.recycles+=1
        self.reset_count+=1
        self.connection.send(("reset", kwargs))
        self._pending = "reset"

    def reset_wait(self):
        if self._pending != "reset":
            raise RuntimeError("No outstanding reset for this environment")
        try:
            result=self._receive()
            self._scene_initialized=True
            return result
        finally:
            self._pending = None

    def reset(self, **kwargs):
        self.reset_async(**kwargs)
        return self.reset_wait()

    def step_async(self, action):
        if self._pending is not None:
            raise RuntimeError("This environment already has an outstanding step")
        self.connection.send(("step", np.asarray(action).copy()))
        self._pending = "step"

    def step_wait(self):
        if self._pending != "step":
            raise RuntimeError("No outstanding step for this environment")
        try:
            return self._receive()
        finally:
            self._pending = None

    def step(self, action):
        self.step_async(action)
        return self.step_wait()

    def runtime_audit(self):
        if self._pending is not None:
            raise RuntimeError("Collect pending operation before audit")
        self.connection.send(("runtime_audit", None))
        return self._receive()

    def close(self):
        if getattr(self,'_closed',False):return self.cleanup_result
        process=self.process;pid=process.pid;errors=[]
        try:
            if process.is_alive():
                try:self.connection.send(('close',None))
                except (OSError,EOFError) as exc:errors.append(repr(exc))
                process.join(timeout=5)
            if process.is_alive():process.terminate();process.join(timeout=5)
            if process.is_alive():process.kill();process.join(timeout=5)
            stopped=not process.is_alive();exitcode=process.exitcode
        finally:
            self.connection.close();self._pending=None
        self.cleanup_result=dict(pid=pid,exitcode=exitcode,stopped=stopped,clean_exit=exitcode==0,errors=errors,worker_log=None if self.log_dir is None else str(Path(self.log_dir)/('worker_%d.log'%pid)))
        if stopped:
            process.close()
            self.process=ClosedProcess(pid,exitcode)
            self._closed=True
        return self.cleanup_result

    def rebuild(self):
        # Capture environment RNG before closing only a responsive numerical-fault
        # worker. Do not infer the cause of a dead worker from failed communication.
        if self._pending is not None:raise RuntimeError('Drain step before rebuild')
        self.connection.send(('rng_state',None));state=self._receive()
        parent_rng=self.action_space.np_random.bit_generator.state
        cleanup=self.close()
        if not cleanup['stopped']:raise RuntimeError('Old worker did not exit')
        from .memory_guard import safe,snapshot
        if not safe(snapshot()):raise MemoryError('Insufficient memory for replacement')
        self._start()
        self.action_space.np_random.bit_generator.state=parent_rng
        self.connection.send(('restore_rng',state));self._receive()
        self.recycles+=1
        return cleanup


class ClosedProcess:
    """Read-only terminal process metadata after OS handles have been closed."""
    def __init__(self,pid,exitcode):self.pid=pid;self.exitcode=exitcode
    def is_alive(self):return False
    def join(self,timeout=None):return None


def step_environments(environments, actions):
    """Dispatch every worker first, then collect in stable environment order."""
    if len(environments) != len(actions) or not environments:
        raise ValueError("One action is required for each selected environment")
    if len(environments) == 1 and not hasattr(environments[0], "step_async"):
        return [environments[0].step(actions[0])]
    if any(not hasattr(env, "step_async") or not hasattr(env, "step_wait") for env in environments):
        raise TypeError("Multiple environments require asynchronous process collectors")
    for env, action in zip(environments, actions):
        env.step_async(action)
    return _collect_ready(environments, "step_wait")


def reset_environments(environments, reset_kwargs=None):
    """Dispatch independent resets before waiting, preserving environment order."""
    if reset_kwargs is None:
        reset_kwargs = [{} for _ in environments]
    if len(environments) != len(reset_kwargs):
        raise ValueError("One reset argument dict is required per environment")
    if not environments:
        return []
    if all(hasattr(env, "reset_async") and hasattr(env, "reset_wait") for env in environments):
        for env, kwargs in zip(environments, reset_kwargs):
            env.reset_async(**kwargs)
        return _collect_ready(environments, "reset_wait")
    if len(environments) != 1:
        raise TypeError("Multiple environments require asynchronous resets")
    return [environments[0].reset(**reset_kwargs[0])]


def _collect_ready(environments, method):
    """Drain ready pipes immediately; return stable environment-index order."""
    if not all(hasattr(env, "connection") for env in environments):
        return [getattr(env, method)() for env in environments]
    pending = {env.connection: (i, env) for i, env in enumerate(environments)}
    results = [None] * len(environments)
    while pending:
        ready = wait(list(pending), timeout=120)
        if not ready:
            raise TimeoutError("SOFA collectors did not respond within 120 seconds")
        for connection in ready:
            index, env = pending.pop(connection)
            results[index] = getattr(env, method)()
    return results


def collect_step_outcomes(environments,actions,env_ids):
    """Drain every dispatched result, including peers of failed workers."""
    from .faults import CollectedStep
    if not(len(environments)==len(actions)==len(env_ids)):
        raise ValueError('One action and stable ID required per worker')
    outcomes=[None]*len(environments);pending={}
    def failure(j,exc):
        env=environments[j];env.process.join(timeout=.01)
        path=None if env.log_dir is None else str(Path(env.log_dir)/('worker_%d.log'%env.process.pid))
        outcomes[j]=CollectedStep(env_ids[j],error=exc,worker_pid=env.process.pid,
                                  exitcode=env.process.exitcode,worker_log=path)
    for j,(env,action) in enumerate(zip(environments,actions)):
        try:
            env.step_async(action);pending[env.connection]=j
        except Exception as exc:failure(j,exc)
    while pending:
        ready=wait(list(pending),timeout=120)
        if not ready:
            for conn,j in list(pending.items()):
                failure(j,TimeoutError('Worker response exceeded 120 seconds'))
            pending.clear();break
        for conn in ready:
            j=pending.pop(conn);env=environments[j]
            try:
                result=env.step_wait()
                path=None if env.log_dir is None else str(Path(env.log_dir)/('worker_%d.log'%env.process.pid))
                outcomes[j]=CollectedStep(env_ids[j],result=result,worker_pid=env.process.pid,
                                          exitcode=env.process.exitcode,worker_log=path)
            except Exception as exc:failure(j,exc)
    return outcomes


def close_environments(environments):
    """Always attempt all owned workers; report cleanup failures independently."""
    results=[]
    for env in environments:
        try:results.append(env.close())
        except Exception as exc:
            results.append(dict(stopped=False,pid=getattr(env.process,'pid',None),error=repr(exc)))
    return results
