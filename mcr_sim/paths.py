"""Portable paths for the non-ROS simulator checkout.

The server layout recorded in ``mcr_sim_structure.txt`` is::

    mCR_simulator-master/
    |-- calib/
    |-- mesh/
    |   |-- train/B01..B05
    |   `-- train/C01..C05
    |-- training_runs/
    `-- python/                 # this Git repository

All paths are derived from this file.  Nothing depends on the current working
directory or on a user-specific absolute path.
"""

from pathlib import Path


MCR_SIM_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = MCR_SIM_DIR.parent
PROJECT_ROOT = PYTHON_ROOT.parent

SCENE_DIR = PYTHON_ROOT / "scene"
MESH_ROOT = PROJECT_ROOT / "mesh"
TRAIN_MESH_DIR = MESH_ROOT / "train"
TEST_MESH_DIR = MESH_ROOT / "test"
CALIB_DIR = PROJECT_ROOT / "calib"
DEFAULT_CALIBRATION_PATH = CALIB_DIR / "Navion_2_Calibration_24-02-2020.yaml"
TRAINING_RUNS_DIR = PROJECT_ROOT / "training_runs"

