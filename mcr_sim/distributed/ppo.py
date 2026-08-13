"""Stable-Baselines3 PPO with synchronized cross-rank policy gradients."""

from __future__ import annotations

from typing import Optional

import torch as th
import torch.distributed as dist
from stable_baselines3 import PPO

from .context import DistributedContext


class DistributedPPO(PPO):
    """One on-policy PPO learner over rank-local rollout buffers.

    Every rank owns an equal share of environments and an independent on-policy
    rollout buffer.  Parameter hooks average gradients during ``backward()``,
    before SB3 applies PPO's global gradient clipping and optimizer step.
    """

    def __init__(self, *args, distributed_context: Optional[DistributedContext] = None, **kwargs):
        self.distributed_context = distributed_context
        self._distributed_gradient_hooks = []
        super().__init__(*args, **kwargs)
        if distributed_context is not None:
            self.set_distributed_context(distributed_context)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "distributed_context",
            "_distributed_gradient_hooks",
        ]

    def set_distributed_context(self, context: DistributedContext) -> None:
        self.distributed_context = context
        for handle in self._distributed_gradient_hooks:
            handle.remove()
        self._distributed_gradient_hooks = []
        if not context.enabled:
            return

        def average_gradient(gradient: th.Tensor) -> th.Tensor:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(float(context.world_size))
            return gradient

        for parameter in self.policy.parameters():
            if parameter.requires_grad:
                self._distributed_gradient_hooks.append(
                    parameter.register_hook(average_gradient)
                )

    def synchronize_parameters(self) -> None:
        context = self.distributed_context
        if context is None:
            raise RuntimeError("DistributedPPO requires a DistributedContext.")
        context.broadcast_module(self.policy)
        context.barrier()
