"""Process-local native thread limits; no solver or physical parameter changes."""
import os
import ctypes
from pathlib import Path

_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
         "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

def limit_worker_environment():
    # Must be inherited before spawn imports numpy/scipy, not just in _worker.
    for key in _KEYS:
        os.environ[key] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"


def limit_sofa_scheduler():
    """SOFA21.12 exported C++ API, guarded by actual DefaultTaskScheduler RTTI.

    FreeMotionAnimationLoop::init calls getInstance unconditionally, which calls
    init(0), creating hardware_concurrency()/2 threads even in serial mode.
    create("") does not init. Bootstrap it with init(1) before scene init.
    Never access guessed vtable slots or mutate native object memory.
    """
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if "libSofa.SimulationCore.so" in line}
    if len(paths) != 1:
        raise RuntimeError("Cannot identify the loaded SOFA21 scheduler library")
    lib = ctypes.CDLL(paths.pop())
    prefix = "_ZN4sofa10simulation"
    current = ctypes.c_void_p.in_dll(lib, prefix + "13TaskScheduler17_currentSchedulerE")
    ptr = current.value
    if not ptr:
        create = getattr(lib, prefix + "13TaskScheduler6createEPKc")
        create.argtypes = [ctypes.c_char_p]
        create.restype = ctypes.c_void_p
        ptr = create(b"")
    class DlInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_void_p) for name in ("file", "base", "symbol", "address")]
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    info = DlInfo()
    vtable = ctypes.c_void_p.from_address(ptr).value
    if not dladdr(vtable, ctypes.byref(info)) or not info.symbol:
        raise RuntimeError("Cannot verify SOFA scheduler concrete type")
    symbol = ctypes.string_at(info.symbol).decode()
    if symbol != "_ZTVN4sofa10simulation20DefaultTaskSchedulerE":
        raise RuntimeError("Refusing to change an unknown task scheduler: " + symbol)
    init = getattr(lib, prefix + "20DefaultTaskScheduler4initEj")
    init.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    init.restype = None
    init(ptr, 1)
    count = getattr(lib, "_ZNK4sofa10simulation20DefaultTaskScheduler14getThreadCountEv")
    count.argtypes = [ctypes.c_void_p]
    count.restype = ctypes.c_uint
    if count(ptr) != 1:
        raise RuntimeError("SOFA scheduler did not honor one compute thread")
    return {"scheduler_threads": 1, "library": lib._name, "type": symbol}
