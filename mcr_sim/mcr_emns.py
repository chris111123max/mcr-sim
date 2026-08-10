import numpy as np
from mag_manip import mag_manip


class EMNS():
    '''
    A class used to build an eMNS object.

    :param name: The name of the eMNS object
    :type name: str
    :param calibration_path: The path to the eMNS calibartion file
    :type name: str
    '''

    def __init__(
           self,
           name='emns',
           calibration_path='../calib/Navion_2_Calibration_24-02-2020.yaml',
           ):

        self.name = name
        self.calibration_path = calibration_path
        self.forward_model = mag_manip.ForwardModelMPEM()
        self.forward_model.setCalibrationFile(calibration_path)

    def currents_to_field(
            self,
            currents=np.array([0., 0., 0.]),
            position=np.array([0., 0., 0.]),
            ):
        '''
        Apply forward model to compute the magnetic field at a given position.
        '''

        bg_jac = self.forward_model.getFieldActuationMatrix(position)
        field = bg_jac.dot(currents)

        return field

    def field_to_currents(
            self,
            field=np.array([0., 0., 0.]),
            position=np.array([0., 0., 0.])):
        '''
        Apply backward model to compute the currents needed to generate a
        magnetic field at a given position.
        '''

        bg_jac = np.asarray(self.forward_model.getFieldActuationMatrix(position), dtype=np.float64)
        field = np.asarray(field, dtype=np.float64)

        if not np.all(np.isfinite(bg_jac)) or not np.all(np.isfinite(field)):
            return np.zeros(3, dtype=np.float64)

        # Use a robust solver for near-singular actuation matrices.
        try:
            currents = np.linalg.solve(bg_jac, field)
        except np.linalg.LinAlgError:
            currents = np.linalg.pinv(bg_jac, rcond=1e-6).dot(field)

        if not np.all(np.isfinite(currents)):
            currents = np.zeros(3, dtype=np.float64)

        return currents
