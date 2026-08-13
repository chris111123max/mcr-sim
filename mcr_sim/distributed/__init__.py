"""Project-side distributed training support for SB3 SAC and PPO."""

from .context import DistributedContext, initialize_distributed
from .device import DeviceSelection, resolve_device
from .npu_performance import (
    configure_npu_execution,
    convert_to_npu_fused_adam,
    zero_optimizer_grad,
)
from .sac import DistributedSAC
from .ppo import DistributedPPO

__all__ = [
    "DeviceSelection",
    "DistributedContext",
    "DistributedSAC",
    "DistributedPPO",
    "configure_npu_execution",
    "convert_to_npu_fused_adam",
    "initialize_distributed",
    "resolve_device",
    "zero_optimizer_grad",
]
