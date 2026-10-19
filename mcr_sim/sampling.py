"""Optional process-isolated SOFA collectors. No distributed learner or DDP."""
import multiprocessing as mp
import traceback
import numpy as np
from gymnasium import spaces


def _worker(connection, config, mode):
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
    def __init__(self, config, mode="train"):
        self.config = config
        self._pending = None
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, config, mode), daemon=True)
        self.process.start()
        child.close()
        self._receive()

    def _receive(self):
        if not self.connection.poll(120):
            raise TimeoutError("SOFA collector did not respond within 120 seconds")
        success, result = self.connection.recv()
        if not success:
            raise RuntimeError("SOFA collector failed:\n" + result)
        return result

    def reset(self, **kwargs):
        if self._pending is not None:
            raise RuntimeError("Collect the pending step before resetting this environment")
        self.connection.send(("reset", kwargs))
        return self._receive()

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
    return [env.step_wait() for env in environments]
