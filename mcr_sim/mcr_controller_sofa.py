import Sofa
import Sofa.Core
import numpy as np

from mcr_sim import mcr_mag_controller
from scipy.spatial.transform import Rotation as R

# Increment field angle in rad
DFIELD_ANGLE = 3.0 * np.pi / 180.0


class ControllerSofa(Sofa.Core.Controller):
    """
    A class that interfaces with the SOFA controller and with the magnetic
    field controller.
    On keyboard events, the desired magnetic field and insertion inputs are
    sent to the controllers.

    :param root_node: The sofa root node
    :param e_mns: The object defining the eMNS
    :param instrument: The object defining the instrument
    :param environment: The object defining the environment
    :param T_sim_mns: The transform defining the pose of the sofa_sim frame center in Navion frame [x, y, z, qx, qy, qz, qw]
    :type T_sim_mns: list[float]
    :param T_sim_mns: The inital magnetic field direction and magnitude (T)
    :type T_sim_mns: ndarray
    """

    def __init__(
            self,
            root_node,
            e_mns,
            instrument,
            T_sim_mns,
            mag_field_init=np.array([0.01, 0.01, 0.0]),
            *args, **kwargs):

        # ====== 核心修复：强制开启键盘监听 ======
        kwargs["listening"] = True

        # These are needed (and the normal way to override from a python class)
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root_node = root_node
        self.e_mns = e_mns
        self.instrument = instrument
        self.T_sim_mns = T_sim_mns
        self.mag_field_init = mag_field_init

        self.dfield_angle = 0.0

        self.mag_controller = mcr_mag_controller.MagController(
            root_node=self.root_node,
            e_mns=self.e_mns,
            instrument=self.instrument,
            T_sim_mns=self.T_sim_mns,
            listening=True  # ====== 核心修复：让底层磁场控制器也开启监听 ======
        )
        self.root_node.addObject(self.mag_controller)

        self.mag_controller.field_des = self.mag_field_init
        self.invalid_action = False

        # Net insertion command per RL step remains 0.6 mm at |action_insert|=1.
        # To improve collision detection without reducing macroscopic insertion speed,
        # the command is buffered and then applied in smaller chunks over SOFA substeps.
        # Recommended runtime: --time-step 0.05 --frame-skip 2.
        self.insert_step_per_action = 0.0002
        self.insert_substep_max = 0.0002
        self.pending_insert_delta = 0.0

    def onKeypressedEvent(self, event):
        """Send magnetic field and insertion inputs when keys are pressed."""
        key = event['key']
        # J key : z-rotation +
        if ord(key) == 76:
            self.rotateZ(-1)

        # L key : z-rotation -
        if ord(key) == 74:
            self.rotateZ(1)

        # I key : x-rotation +
        if ord(key) == 73:
            self.rotateX(-1)

        # K key : x-rotation -
        if ord(key) == 75:
            self.rotateX(1)

    def rotateZ(self, val):
        r = R.from_rotvec(val * DFIELD_ANGLE * np.array([0, 0, 1]))
        self.mag_controller.field_des = r.apply(self.mag_controller.field_des)

    def rotateX(self, val):
        r = R.from_rotvec(val * DFIELD_ANGLE * np.array([1, 0, 0]))
        self.mag_controller.field_des = r.apply(self.mag_controller.field_des)

    def insertRetract(self, val):
        """Buffer insertion/retraction command.

        The RL action still represents up to 0.6 mm insertion per RL step, but the
        actual xtip update is split into smaller SOFA substeps in onAnimateBeginEvent.
        With --time-step 0.05 --frame-skip 2, this preserves the old macroscopic
        speed while giving collision/constraint solving two chances to react.
        """
        val = float(np.clip(val, -1.0, 1.0))
        delta = val * float(self.insert_step_per_action)

        # Prevent the buffer from requesting impossible insertion beyond the catheter limit.
        current_xtip = float(self._getXTipValue())
        target_xtip = current_xtip + float(self.pending_insert_delta) + delta
        if target_xtip > 0.51:
            self.invalid_action = True
            delta = max(0.0, 0.51 - current_xtip - float(self.pending_insert_delta))
        else:
            self.invalid_action = False

        self.pending_insert_delta += float(delta)

    def onAnimateBeginEvent(self, event):
        """Apply pending insertion in small chunks at every SOFA physics step."""
        pending = float(getattr(self, "pending_insert_delta", 0.0))
        if abs(pending) < 1e-12:
            return

        max_chunk = abs(float(getattr(self, "insert_substep_max", 0.0003)))
        if max_chunk <= 0.0:
            max_chunk = abs(float(getattr(self, "insert_step_per_action", 0.0006)))

        chunk = float(np.clip(pending, -max_chunk, max_chunk))
        current_xtip = float(self._getXTipValue())
        new_xtip = current_xtip + chunk

        if new_xtip > 0.51:
            self.instrument.IRC.xtip[0] = 0.51
            self.pending_insert_delta = 0.0
            self.invalid_action = True
        elif new_xtip < 0.0:
            self.instrument.IRC.xtip[0] = 0.0
            self.pending_insert_delta = 0.0
            self.invalid_action = True
        else:
            self.instrument.IRC.xtip[0] = new_xtip
            self.pending_insert_delta = pending - chunk
            self.invalid_action = False

    def _getXTipValue(self):
        return self.instrument.IRC.xtip[0]

    def reset(self) -> None:
        """Reset magnetic field."""
        super().reset()
        self.mag_controller.field_des = self.mag_field_init
        self.dfield_angle = 0.0

        # Reset insertRetract state
        self.instrument.IRC.xtip[0] = 0.0
        self.pending_insert_delta = 0.0
        self.invalid_action = False

    def get_mag_field_des(self):
        return self.mag_controller.field_des


    def get_pos_catheter(self, num_points):
        positions = self.instrument.MO.position.array()
        n = len(positions)
        if n <= 0 or num_points <= 0:
            return ()

        idxs = np.linspace(0, n - 1, int(num_points)).astype(int)
        pos_catheter = ()
        for idx in idxs:
            pos_catheter = np.append(pos_catheter, positions[int(idx)][:3])
        return pos_catheter


    def get_pos_quat_catheter_tip(self):
        positions = self.instrument.MO.position.array()
        return positions[-1]
