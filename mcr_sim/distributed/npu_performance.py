"""Version-tolerant Ascend performance helpers for SB3 training."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch as th


_FUSED_ADAM_PROBE_RESULTS = {}


def _probe_fused_adam(fused_adam, reference_parameter: th.Tensor) -> None:
    """Run one isolated optimizer step before touching the real model."""

    cached = _FUSED_ADAM_PROBE_RESULTS.get(fused_adam)
    if cached is not None:
        if isinstance(cached, Exception):
            raise cached
        return

    try:
        probe_a = th.nn.Parameter(
            th.zeros(8, dtype=reference_parameter.dtype, device=reference_parameter.device)
        )
        probe_b = th.nn.Parameter(
            th.zeros((2, 3), dtype=reference_parameter.dtype, device=reference_parameter.device)
        )
        probe_optimizer = fused_adam([probe_a, probe_b], lr=1e-3)
        zero_optimizer_grad(probe_optimizer)
        (probe_a.square().sum() + probe_b.square().sum()).backward()
        probe_optimizer.step()
        zero_optimizer_grad(probe_optimizer)
        npu_api = getattr(th, "npu", None)
        if npu_api is not None and callable(getattr(npu_api, "synchronize", None)):
            npu_api.synchronize()
        _FUSED_ADAM_PROBE_RESULTS[fused_adam] = True
    except Exception as exc:
        _FUSED_ADAM_PROBE_RESULTS[fused_adam] = exc
        raise


def configure_npu_execution(accelerator: str, enabled: bool = True) -> Dict[str, Any]:
    """Enable safe eager/Linear execution options exposed by this torch_npu.

    The deployment uses an older torch_npu stack, so every optional API is
    detected at runtime.  Unsupported options are reported rather than making
    CPU/CUDA training or an older Ascend installation fail at import time.
    """

    status = {
        "requested": bool(enabled),
        "jit_compile_disabled": False,
        "mm_bmm_nd_enabled": False,
        "errors": [],
    }
    if not enabled or str(accelerator) != "npu":
        return status

    try:
        import torch_npu
    except Exception as exc:  # pragma: no cover - only available on Ascend
        status["errors"].append(f"torch_npu import failed: {exc}")
        return status

    npu_apis = [getattr(torch_npu, "npu", None), getattr(th, "npu", None)]
    set_compile_mode = next(
        (
            candidate
            for candidate in (
                getattr(api, "set_compile_mode", None) for api in npu_apis if api is not None
            )
            if callable(candidate)
        ),
        None,
    )
    if callable(set_compile_mode):
        try:
            # Precompiled eager operators avoid repeated online compilation for
            # SAC's many small, fixed-shape MLP operations.
            set_compile_mode(jit_compile=False)
            status["jit_compile_disabled"] = True
        except Exception as exc:  # pragma: no cover - Ascend-version specific
            status["errors"].append(f"set_compile_mode failed: {exc}")

    set_mm_bmm_format_nd = next(
        (
            candidate
            for candidate in (
                getattr(api, "set_mm_bmm_format_nd", None)
                for api in npu_apis
                if api is not None
            )
            if callable(candidate)
        ),
        None,
    )
    if callable(set_mm_bmm_format_nd):
        try:
            # Let Linear layers keep matrix operands in the NPU-native ND
            # layout instead of inserting format conversions around mm/bmm.
            set_mm_bmm_format_nd(True)
            status["mm_bmm_nd_enabled"] = True
        except Exception as exc:  # pragma: no cover - Ascend-version specific
            status["errors"].append(f"set_mm_bmm_format_nd failed: {exc}")

    return status


def is_npu_fused_optimizer(optimizer: th.optim.Optimizer) -> bool:
    optimizer_type = type(optimizer)
    return (
        optimizer_type.__name__.startswith("NpuFused")
        or optimizer_type.__module__.startswith("torch_npu.optim")
    )


def zero_optimizer_grad(optimizer: th.optim.Optimizer) -> None:
    """Clear gradients using the fastest mode supported by the optimizer."""

    # Older NpuFusedAdam releases reject set_to_none=True.  Standard Adam and
    # current PyTorch optimizers benefit from avoiding a full gradient memset.
    optimizer.zero_grad(set_to_none=not is_npu_fused_optimizer(optimizer))


def convert_to_npu_fused_adam(
    optimizer: th.optim.Optimizer,
) -> Tuple[th.optim.Optimizer, str]:
    """Convert a torch Adam optimizer while preserving groups and state.

    Returns the original optimizer and a diagnostic reason when the installed
    torch_npu does not expose a compatible fused Adam implementation.
    """

    if is_npu_fused_optimizer(optimizer):
        return optimizer, "already_enabled"
    if not isinstance(optimizer, th.optim.Adam):
        return optimizer, f"unsupported_optimizer:{type(optimizer).__name__}"

    try:
        import torch_npu

        fused_adam = getattr(getattr(torch_npu, "optim", None), "NpuFusedAdam", None)
        if fused_adam is None:
            # Older Ascend PyTorch releases shipped fused optimizers through
            # the matching Ascend Apex package instead of torch_npu.optim.
            try:
                import apex

                fused_adam = getattr(
                    getattr(apex, "optimizers", None), "NpuFusedAdam", None
                )
            except Exception:
                fused_adam = None
        if fused_adam is None:
            return optimizer, "NpuFusedAdam_unavailable"

        reference_parameter = next(
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        _probe_fused_adam(fused_adam, reference_parameter)

        supported_group_keys = {
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
        }
        parameter_groups = []
        for group in optimizer.param_groups:
            converted_group = {
                key: value for key, value in group.items() if key in supported_group_keys
            }
            converted_group["params"] = list(group["params"])
            parameter_groups.append(converted_group)

        converted = fused_adam(parameter_groups)
        state = optimizer.state_dict()
        if state.get("state"):
            converted.load_state_dict(state)
        return converted, "enabled"
    except Exception as exc:  # pragma: no cover - Ascend-version specific
        return optimizer, f"fallback:{type(exc).__name__}:{exc}"
