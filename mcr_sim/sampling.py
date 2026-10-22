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
        if env is not None:
            env.close()
        connection.close()


class ProcessEnvironment:
    def __init__(self, config, mode="train", log_dir=None):
        limit_worker_environment()
        self.config = config
        self._pending = None
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
        success, result = self.connection.recv()
        if not success:
            raise RuntimeError("SOFA collector failed:\n" + result)
        return result

    def reset_async(self, **kwargs):
        if self._pending is not None:
            raise RuntimeError("Collect the pending operation before resetting this environment")
        self.connection.send(("reset", kwargs))
        self._pending = "reset"

    def reset_wait(self):
        if self._pending != "reset":
            raise RuntimeError("No outstanding reset for this environment")
        try:
            return self._receive()
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
        if self.process.is_alive():
            try:
                self.connection.send(("close", None))
            except (EOFError, BrokenPipeError):
                pass
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        self.connection.close()


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
