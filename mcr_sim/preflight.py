"""Non-integrating preflight: config, local caches and magnetic geometry only."""
from dataclasses import asdict
import json
from .config import DEFAULT_CONFIG
from .instrument_layout import NelsonInstrumentLayout
from .magnetics import nelson_dipole_moment_a_m2
from .sofa_simulation import component_parameters
from .vessel_bundle import coordinate_consistency, discover_vessel_bundles


def build_report():
    cfg = DEFAULT_CONFIG
    layout = NelsonInstrumentLayout(cfg.instrument)
    try:
        bundles = discover_vessel_bundles(cfg.vessel_root)
        vessels = [{"seed": b.seed, **coordinate_consistency(b)} for b in bundles]
        cache_status = "PRESENT"
    except FileNotFoundError:
        vessels, cache_status = [], "NOT_GENERATED"
    return {"sofa": component_parameters(cfg), "vessel_root": str(cfg.vessel_root),
            "cache_status": cache_status, "vessels": vessels,
            "train_seeds": cfg.vessel_seeds, "eval_seeds": cfg.eval_vessel_seeds,
            "observation_dimension": cfg.observation.dimension, "action_dimension": 3,
            "magnet_overlap_nodes": layout.magnetic_node_indices,
            "magnet_overlap_weights": [x.weight for x in layout.magnet_overlaps],
            "magnet_dipole_moment_a_m2": nelson_dipole_moment_a_m2(cfg.instrument),
            "formal_training_started": False, "physics_substeps_executed": 0}


def main():
    print(json.dumps(build_report(), indent=2))


if __name__ == "__main__":
    main()
