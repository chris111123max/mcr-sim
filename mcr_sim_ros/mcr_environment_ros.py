import Sofa
from scipy.spatial.transform import Rotation as R
import numpy as np


class Environment(Sofa.Core.Controller):
    """
    A class used to define environment objects and to build the SOFA collision
    and visualization model of the environment.
    """

    def __init__(
        self,
        root_node,
        environment_stl,
        name="environment",
        T_env_sim=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        flip_normals=False,
        color=[1.0, 0.0, 0.0, 0.3],
        *args,
        **kwargs,
    ):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root_node = root_node
        self.environment_stl = environment_stl
        self.name_env = name
        self.color = color
        self.T_env_sim = T_env_sim

        r = R.from_quat(self.T_env_sim[3:7])
        rot_env_sim = (r.as_euler("xyz", degrees=True)).tolist()

        # Collision model environment
        self.CollisionModel = root_node.addChild("CollisionModel")

        # 关键修复：保存 meshLoader 引用。
        # 之前 get_vessel_tree_positions() 返回的是 self.MO.position.array()，而 self.MO 只有一个 [0,0,0] 点，
        # 会导致 bbox_diag=0，cartesian_scaling_factor=inf，最终 reward 出现 NaN。
        self.meshLoader = self.CollisionModel.addObject(
            "MeshSTLLoader",
            filename=self.environment_stl,
            flipNormals=flip_normals,
            triangulate=True,
            name="meshLoader",
            rotation=rot_env_sim,
            translation=self.T_env_sim[0:3],
            scale="0.0005",
        )

        self.CollisionModel.addObject(
            "Mesh",
            position="@meshLoader.position",
            triangles="@meshLoader.triangles",
            drawTriangles="0",
        )

        self.MO = self.CollisionModel.addObject(
            "MechanicalObject",
            position=[0, 0, 0],
            scale=1,
            name="DOFs1",
        )

        self.CollisionModel.addObject(
            "TriangleCollisionModel",
            moving=False,
            simulated=False,
        )
        self.CollisionModel.addObject(
            "LineCollisionModel",
            moving=False,
            simulated=False,
        )
        self.CollisionModel.addObject(
            "PointCollisionModel",
            moving=False,
            simulated=False,
        )

        # Visual model environment
        VisuModel = self.CollisionModel.addChild("VisuModel")
        VisuModel.addObject(
            "OglModel",
            name="VisualOgl_model",
            src="@../meshLoader",
            color=self.color,
        )

    def get_vessel_tree_positions(self):
        """
        Return the transformed STL mesh vertices, not the dummy MechanicalObject point.

        This is used by MCREnv to compute cartesian_scaling_factor. It must return a
        non-degenerate point cloud; otherwise the bbox diagonal becomes zero and the reward
        can become inf / NaN.
        """
        try:
            points = np.asarray(self.meshLoader.position.array(), dtype=np.float32)
        except Exception:
            points = np.asarray(self.CollisionModel.meshLoader.position.array(), dtype=np.float32)

        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] <= 1:
            raise RuntimeError(
                f"Invalid vessel mesh positions from MeshSTLLoader: shape={getattr(points, 'shape', None)}"
            )

        if not np.all(np.isfinite(points)):
            raise RuntimeError("Invalid vessel mesh positions: contains NaN or Inf.")

        return points
