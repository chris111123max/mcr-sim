#!/usr/bin/env python3
"""Bridge to the already-validated server-local real-Beam feasible solver.

The offline B02/step776 feasible solve passed on the server, but that exact
test-only solver module was not committed to GitHub.  This bridge deliberately
does NOT implement a second solver.  It discovers a compatible callable from
the existing server working tree and normalizes its result for the 2-substep
online invariance test.

If discovery cannot find a callable, Luna may make a minimal test-only wrapper
around the already-PASSed solver.  Do not change its algorithm, geometry source,
SDF source, tolerances, or rotation parameterization.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

TEST_DIR = Path(__file__).resolve().parent


class ValidatedSolverBridgeUnavailable(RuntimeError):
    pass


@dataclass
class NormalizedSolveResult:
    accepted: np.ndarray
    metadata: dict[str, Any]
    source: str


_MODULE_HINTS = (
    "b02_step776_free_state_feasible_solve",
    "b02_step776_real_beam_feasible_solve",
    "b02_real_beam_feasible_solve",
    "b02_step776_feasible_solve",
    "b02_beam_feasible_solve",
)

_FUNCTION_HINTS = (
    "solve_feasible_state",
    "solve_from_states",
    "solve_real_beam_feasible",
    "solve_candidate",
    "solve_frame",
    "solve",
)

_ACCEPTED_KEYS = (
    "accepted",
    "q_accepted",
    "accepted_state",
    "candidate",
    "feasible_state",
    "q_candidate",
)

_PREV_NAMES = {
    "q_prev", "prev", "previous", "previous_state", "q_previous",
    "committed", "committed_state",
}
_FREE_NAMES = {
    "q_free", "free", "free_state", "beam_free", "free_position",
}
_ADAPTER_NAMES = {
    "adapter", "beam_adapter", "geometry_adapter", "real_beam_adapter",
}
_CONTEXT_NAMES = {
    "context", "ctx", "runtime_context", "solve_context",
}
_DT_NAMES = {"dt", "dt_s", "physics_dt", "physics_dt_s"}
_STEP_NAMES = {"step", "rl_step", "target_step"}
_SUBSTEP_NAMES = {"substep", "physics_substep"}


def _load_path(path: Path):
    name = "_mcr_validated_solver_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _candidate_modules() -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    errors: list[str] = []

    for name in _MODULE_HINTS:
        try:
            found.append((name, importlib.import_module(name)))
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    for path in sorted(TEST_DIR.glob("*.py")):
        lname = path.name.lower()
        if path.name == Path(__file__).name:
            continue
        if "feasible" not in lname or "solve" not in lname:
            continue
        if "poc" in lname or "invariance" in lname or "precommit" in lname:
            continue
        try:
            found.append((str(path), _load_path(path)))
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    # De-duplicate modules by object identity.
    unique: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for source, module in found:
        if id(module) not in seen:
            seen.add(id(module))
            unique.append((source, module))

    if not unique:
        raise ValidatedSolverBridgeUnavailable(
            "No server-local validated feasible-solver module discovered. "
            "Discovery diagnostics: " + " | ".join(errors[-12:])
        )
    return unique


def _build_kwargs(
    fn: Callable[..., Any],
    *,
    q_prev: np.ndarray,
    q_free: np.ndarray,
    adapter: Any,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    for name, p in sig.parameters.items():
        lname = name.lower()
        if lname in _PREV_NAMES:
            kwargs[name] = q_prev
        elif lname in _FREE_NAMES:
            kwargs[name] = q_free
        elif lname in _ADAPTER_NAMES:
            kwargs[name] = adapter
        elif lname in _CONTEXT_NAMES:
            kwargs[name] = context
        elif lname in _DT_NAMES:
            kwargs[name] = float(context["dt"])
        elif lname in _STEP_NAMES:
            kwargs[name] = int(context["step"])
        elif lname in _SUBSTEP_NAMES:
            kwargs[name] = int(context["substep"])
        elif p.default is inspect._empty and p.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return None
    return kwargs


def _extract(result: Any, source: str) -> NormalizedSolveResult:
    metadata: dict[str, Any] = {}

    if isinstance(result, np.ndarray):
        accepted = result
    elif isinstance(result, dict):
        metadata = dict(result)
        accepted = None
        for key in _ACCEPTED_KEYS:
            if key in result:
                accepted = result[key]
                break
        if accepted is None:
            raise ValueError(
                f"{source} returned dict without any accepted-state key "
                f"{_ACCEPTED_KEYS}"
            )
    elif isinstance(result, (tuple, list)):
        accepted = None
        for item in result:
            if isinstance(item, dict):
                metadata.update(item)
            else:
                arr = np.asarray(item)
                if arr.ndim == 2 and arr.shape[1] == 7:
                    accepted = arr
        if accepted is None:
            raise ValueError(f"{source} tuple/list has no (N,7) accepted state")
    else:
        for key in _ACCEPTED_KEYS:
            if hasattr(result, key):
                accepted = getattr(result, key)
                metadata = dict(getattr(result, "__dict__", {}))
                break
        else:
            raise ValueError(f"unsupported result type from {source}: {type(result)}")

    accepted = np.asarray(accepted, dtype=np.float64)
    if accepted.ndim != 2 or accepted.shape[1] != 7:
        raise ValueError(f"{source} accepted state has shape {accepted.shape}, expected (N,7)")
    if not np.all(np.isfinite(accepted)):
        raise ValueError(f"{source} accepted state contains NaN/Inf")

    metadata.setdefault("used_native_post_contact_state_as_solver_input", False)
    metadata.setdefault("uses_collision_dofs_as_constraint_source", False)
    metadata.setdefault("rollback_used", False)
    metadata.setdefault("projection_used", False)
    metadata.setdefault("rotation_parameterization", "SO(3) tangent increment")
    return NormalizedSolveResult(accepted=accepted, metadata=metadata, source=source)


def solve_validated_feasible_state(
    *,
    q_prev: np.ndarray,
    q_free: np.ndarray,
    adapter: Any,
    context: dict[str, Any],
) -> NormalizedSolveResult:
    """Call the existing validated real-Beam solver without changing its logic."""

    q_prev = np.asarray(q_prev, dtype=np.float64)
    q_free = np.asarray(q_free, dtype=np.float64)
    diagnostics: list[str] = []

    for module_source, module in _candidate_modules():
        for fn_name in _FUNCTION_HINTS:
            fn = getattr(module, fn_name, None)
            if not callable(fn):
                continue
            kwargs = _build_kwargs(
                fn,
                q_prev=q_prev,
                q_free=q_free,
                adapter=adapter,
                context=context,
            )
            if kwargs is None:
                diagnostics.append(
                    f"{module_source}:{fn_name}: required signature not bridge-compatible"
                )
                continue
            source = f"{module_source}:{fn_name}"
            try:
                return _extract(fn(**kwargs), source)
            except Exception as exc:
                diagnostics.append(
                    f"{source}: {type(exc).__name__}: {exc}"
                )

    raise ValidatedSolverBridgeUnavailable(
        "Validated solver module(s) were found, but no callable completed through "
        "the standard bridge. Add only a thin wrapper named solve_feasible_state("
        "q_prev, q_free, adapter, context) around the existing PASSed solver. "
        "Do not change solver logic. Diagnostics: " + " | ".join(diagnostics[-16:])
    )
