"""runSofa facade for the single production Nelson scene (no training)."""
from mcr_sim.config import DEFAULT_CONFIG
from mcr_sim.sofa_simulation import create_sofa_scene
from mcr_sim.vessel_bundle import load_vessel_bundle


def createScene(rootNode):
    bundle = load_vessel_bundle(DEFAULT_CONFIG.vessel_root / "arch_type_I_seed_101")
    create_sofa_scene(rootNode, bundle)
    return rootNode
