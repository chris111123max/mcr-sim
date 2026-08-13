"""Small torch.distributed context shared by the launcher and DistributedSAC."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import torch as th
import torch.distributed as dist

from .device import DeviceSelection, resolve_device


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: DeviceSelection
    communication_seconds: float = 0.0
    communication_calls: int = 0
    update_seconds: float = 0.0
    _performance_window_started: float = 0.0
    _metric_sums: dict = field(default_factory=dict, repr=False)
    _metric_counts: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._performance_window_started = time.perf_counter()

    def _record_communication(self, started: float) -> None:
        self.communication_seconds += time.perf_counter() - started
        self.communication_calls += 1

    def all_reduce(self, tensor: th.Tensor, op=dist.ReduceOp.SUM) -> None:
        if not self.enabled:
            return
        started = time.perf_counter()
        dist.all_reduce(tensor, op=op)
        self._record_communication(started)

    def all_gather(self, outputs, tensor: th.Tensor) -> None:
        if not self.enabled:
            return
        started = time.perf_counter()
        dist.all_gather(outputs, tensor)
        self._record_communication(started)

    def record_update_time(self, seconds: float) -> None:
        self.update_seconds += float(seconds)

    def performance_snapshot(self, reset: bool = True):
        now = time.perf_counter()
        elapsed = now - self._performance_window_started
        result = {
            "elapsed_seconds": elapsed,
            "update_seconds": self.update_seconds,
            "communication_seconds": self.communication_seconds,
            "communication_calls": self.communication_calls,
            "rollout_other_seconds": max(0.0, elapsed - self.update_seconds),
        }
        if reset:
            self._performance_window_started = now
            self.update_seconds = 0.0
            self.communication_seconds = 0.0
            self.communication_calls = 0
        return result

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            started = time.perf_counter()
            dist.barrier()
            self._record_communication(started)

    def broadcast_module(self, module: th.nn.Module, source: int = 0) -> None:
        if not self.enabled:
            return
        with th.no_grad():
            for parameter in module.parameters():
                dist.broadcast(parameter.data, src=source)
            for buffer in module.buffers():
                dist.broadcast(buffer.data, src=source)

    def broadcast_tensor(self, tensor: Optional[th.Tensor], source: int = 0) -> None:
        if self.enabled and tensor is not None:
            dist.broadcast(tensor.data, src=source)

    def average_gradients(self, parameters: Iterable[th.nn.Parameter]) -> None:
        """Average trainable gradients in a fixed collective order.

        Do not manually concatenate NPU gradients here.  On the deployed
        torch_npu/HCCL stack, the extra device-side concatenate and copy-back
        kernels cost more than the small collectives they replace.  Algorithms
        that are genuinely wrapped by DDP should use DDP's native buckets.
        """
        if not self.enabled:
            return
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                parameter.grad = th.zeros_like(parameter)
            self.all_reduce(parameter.grad)
            parameter.grad.div_(float(self.world_size))

    def average_tensor_gradient(self, tensor: Optional[th.Tensor]) -> None:
        if not self.enabled or tensor is None:
            return
        if tensor.grad is None:
            tensor.grad = th.zeros_like(tensor)
        self.all_reduce(tensor.grad)
        tensor.grad.div_(float(self.world_size))

    def average_metrics(self, values: Sequence[float]) -> Sequence[float]:
        if not self.enabled:
            return list(values)
        metrics = th.tensor(list(values), dtype=th.float32, device=self.device.resolved)
        self.all_reduce(metrics)
        metrics.div_(float(self.world_size))
        return metrics.detach().cpu().tolist()

    def average_metric_tensor(self, metrics: th.Tensor) -> th.Tensor:
        """Average an already device-resident metric vector in place."""
        if self.enabled:
            self.all_reduce(metrics)
            metrics.div_(float(self.world_size))
        return metrics

    def reduce_metrics_periodically(
        self,
        key: str,
        metrics: th.Tensor,
        interval: int,
    ):
        """Locally accumulate metrics and communicate only at log intervals."""

        key = str(key)
        interval = max(1, int(interval))
        detached = metrics.detach()
        if key not in self._metric_sums:
            self._metric_sums[key] = th.zeros_like(detached)
            self._metric_counts[key] = 0
        self._metric_sums[key].add_(detached)
        self._metric_counts[key] += 1
        if self._metric_counts[key] < interval:
            return None

        reduced = self._metric_sums[key] / float(self._metric_counts[key])
        reduced = self.average_metric_tensor(reduced)
        values = reduced.detach().cpu().tolist()
        self._metric_sums[key].zero_()
        self._metric_counts[key] = 0
        return values

    def broadcast_text(self, value: str, source: int = 0) -> str:
        if not self.enabled:
            return str(value)
        encoded = str(value).encode("utf-8") if self.rank == source else b""
        length = th.tensor(
            [len(encoded)], dtype=th.int32, device=self.device.resolved
        )
        dist.broadcast(length, src=source)
        payload_size = int(length.detach().cpu().item())
        if self.rank == source:
            payload = th.tensor(list(encoded), dtype=th.uint8, device=self.device.resolved)
        else:
            payload = th.zeros(payload_size, dtype=th.uint8, device=self.device.resolved)
        dist.broadcast(payload, src=source)
        return bytes(payload.detach().cpu().tolist()).decode("utf-8")

    def close(self) -> None:
        if self.enabled and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else int(default)


def initialize_distributed(
    enabled: bool,
    requested_device: str,
    requested_world_size: int = 0,
    cli_local_rank: int = 0,
    requested_backend: str = "",
) -> DistributedContext:
    """Initialize torchrun/HCCL, NCCL, or Gloo using logical device indices."""
    enabled = bool(enabled)
    if not enabled:
        device = resolve_device(requested_device, distributed=False, local_rank=0)
        return DistributedContext(False, 0, 0, 1, "none", device)

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this PyTorch build.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "--distributed must be launched through torchrun. "
            "Use training/sh/run_train_sac.sh, which adds torchrun automatically."
        )

    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", cli_local_rank)
    world_size = _env_int("WORLD_SIZE", requested_world_size or 1)
    if requested_world_size not in (0, world_size):
        raise ValueError(
            f"--world-size={requested_world_size} does not match torchrun WORLD_SIZE={world_size}."
        )
    if world_size <= 1:
        raise ValueError("--distributed requires WORLD_SIZE greater than one.")

    device = resolve_device(requested_device, distributed=True, local_rank=local_rank)
    backend = str(requested_backend or "").strip().lower()
    if not backend:
        backend = {"npu": "hccl", "cuda": "nccl", "cpu": "gloo"}[device.accelerator]

    expected_backend = {"npu": "hccl", "cuda": "nccl", "cpu": "gloo"}[device.accelerator]
    if backend != expected_backend:
        raise ValueError(
            f"Backend {backend!r} is incompatible with {device.accelerator}; "
            f"expected {expected_backend!r}."
        )

    dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(True, rank, local_rank, world_size, backend, device)
