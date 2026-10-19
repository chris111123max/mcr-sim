"""Reusable device utilities; no legacy PPO, SAC, HER or distributed learner."""
from .device import DeviceSelection, resolve_device
__all__ = ["DeviceSelection", "resolve_device"]
