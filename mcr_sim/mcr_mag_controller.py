"""Nelson uniform-field cross-product torque applied through ConstantForceField.

A single 4 mm magnet is integrated over four terminal 1 mm elements. The sum
of overlap weights is one; the full magnet moment is never applied four times.
"""
import Sofa
import numpy as np
from .magnetics import distributed_magnetic_torques


class MagController(Sofa.Core.Controller):
    def __init__(self, instrument, initial_field_t):
        super().__init__(listening=True)
        self.instrument = instrument
        self.field_world_t = np.asarray(initial_field_t, dtype=float).copy()
        self.last_total_torque_n_m = None

    def onAnimateBeginEvent(self, event):
        positions = np.asarray(self.instrument.MO.position.value, dtype=float)
        quaternions = [positions[item.node_index, 3:7] for item in self.instrument.layout.magnet_overlaps]
        torques = distributed_magnetic_torques(self.field_world_t, quaternions, self.instrument.layout)
        self.instrument.CFF.forces.value = [[0.0, 0.0, 0.0] + item.torque_world_n_m.tolist() for item in torques]
        self.last_total_torque_n_m = np.sum([item.torque_world_n_m for item in torques], axis=0)
