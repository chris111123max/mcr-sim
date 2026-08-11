"""CUDA/Ascend/CPU device selection without hard-coding CANN versions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch as th


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    resolved: str
    accelerator: str
    local_rank: int


def _import_torch_npu(required: bool = False) -> bool:
    try:
        import torch_npu  # noqa: F401

        return True
    except ImportError as exc:
        if required:
            raise RuntimeError(
                "Ascend NPU was requested but torch_npu is not installed. "
                "Install the torch/torch_npu pair matching the server CANN release."
            ) from exc
        return False


def _npu_available() -> bool:
    if not _import_torch_npu(required=False):
        return False
    npu_api = getattr(th, "npu", None)
    if npu_api is None:
        return False
    try:
        return bool(npu_api.is_available())
    except Exception:
        return False


def _indexed_device(kind: str, requested: str, distributed: bool, local_rank: int) -> str:
    if ":" in requested:
        if distributed:
            raise ValueError(
                f"Use --device {kind} (without an index) in distributed mode; "
                f"torchrun maps each rank to {kind}:LOCAL_RANK."
            )
        return requested
    return f"{kind}:{local_rank if distributed else 0}"


def resolve_device(requested: str, distributed: bool, local_rank: int = 0) -> DeviceSelection:
    """Resolve a logical training device and activate the local accelerator."""
    requested = str(requested or "auto").strip().lower()
    local_rank = int(local_rank)

    if requested == "auto":
        if th.cuda.is_available():
            requested = "cuda"
        elif _npu_available():
            requested = "npu"
        else:
            requested = "cpu"

    if requested == "cpu":
        resolved = "cpu"
        accelerator = "cpu"
    elif requested == "cuda" or requested.startswith("cuda:"):
        if not th.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        resolved = _indexed_device("cuda", requested, distributed, local_rank)
        th.cuda.set_device(int(resolved.split(":", 1)[1]))
        accelerator = "cuda"
    elif requested == "npu" or requested.startswith("npu:"):
        _import_torch_npu(required=True)
        if not _npu_available():
            raise RuntimeError("Ascend NPU was requested but torch.npu.is_available() is False.")
        resolved = _indexed_device("npu", requested, distributed, local_rank)
        th.npu.set_device(int(resolved.split(":", 1)[1]))
        accelerator = "npu"
    else:
        raise ValueError(
            f"Unsupported --device {requested!r}. Use auto, cpu, cuda, cuda:N, npu, or npu:N."
        )

    # Let PyTorch validate that the selected private-use device is registered.
    th.device(resolved)
    return DeviceSelection(requested=requested, resolved=resolved, accelerator=accelerator, local_rank=local_rank)
