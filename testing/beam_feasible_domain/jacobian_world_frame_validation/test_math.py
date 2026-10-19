import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from world_rows import convert_rows
class MathTests(unittest.TestCase):
 def test_world_row_covector(self):
  r=Rotation.from_euler('xyz',[.8,-1.2,.3]);q=np.r_[np.zeros(3),r.as_quat()][None,:]
  j=np.array([[.2,-.4,.7]]);rows={'dof_indices':[0],'angular_jacobian':j,'linear_jacobian':[[1,2,3]]}
  out=convert_rows(rows,q);world=np.array(out['angular_jacobian']);axis=np.array([.8,.3,-.2])
  self.assertAlmostEqual(float(world[0]@axis),float(j[0]@r.inv().apply(axis)),places=14)
  np.testing.assert_array_equal(out['linear_jacobian'],rows['linear_jacobian'])
  np.testing.assert_array_equal(rows['angular_jacobian'],j)
  with self.assertRaises(RuntimeError):convert_rows(out,q)
 def test_identity(self):
  q=np.array([[0,0,0,0,0,0,1.]])
  np.testing.assert_array_equal(convert_rows({'dof_indices':[0],'angular_jacobian':[[1,2,3]]},q)['angular_jacobian'],[[1,2,3]])
if __name__=='__main__':unittest.main()
