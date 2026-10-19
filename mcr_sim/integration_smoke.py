"""Luna-only single-action SOFA21 probe. Never imports or launches a learner."""
import argparse
from dataclasses import replace
import json
import time
import traceback
import numpy as np
from .config import DEFAULT_CONFIG
from .rl_env import MagneticNavigationEnv
from .sofa_simulation import SofaBackend
from .vessel_bundle import load_vessel_bundle


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def observed(handles):
    def data(obj, field):
        try:
            value = getattr(obj, field).value
            return np.asarray(value).tolist()
        except Exception:
            return "NOT_OBSERVED"
    s, i, v = handles.simulator, handles.instrument, handles.vessel
    return {
        "dt_s": data(handles.root, "dt"),
        "lcp_class": s.lcp_solver.getClassName(),
        "lcp": {name: data(s.lcp_solver, name) for name in ("mu", "tolerance", "maxIt", "build_lcp")},
        "intersection": {name: data(s.intersection, name) for name in ("contactDistance", "alarmDistance", "angleCone")},
        "beam": {name: data(i.rest_shape, name) for name in ("straightLength", "length", "densityOfBeams", "numEdgesCollis", "numEdges", "youngModulus", "youngModulusExtremity", "radius")},
        "rigid_dofs_shape": list(np.asarray(i.MO.position.value).shape),
        "rigid_positions_m": np.asarray(i.MO.position.value)[:, :3].tolist(),
        "collision_dofs_shape": list(np.asarray(i.collision_dofs.position.value).shape),
        "magnetic_force_indices": data(i.CFF, "indices"),
        "magnetic_forces": data(i.CFF, "forces"),
        "magnet_weights": [m.weight for m in i.layout.magnet_overlaps],
        "magnetic_field_world_t": handles.torque_controller.field_world_t.tolist(),
        "vessel_mesh_bounds_m": {"min": np.asarray(v.MO.position.value).min(axis=0).tolist(), "max": np.asarray(v.MO.position.value).max(axis=0).tolist()},
        "vessel_vertex_count": len(v.MO.position.value),
        "vessel_first_three_vertices_m": np.asarray(v.MO.position.value)[:3].tolist(),
        "vessel_collision_classes": [obj.getClassName() for obj in v.CollisionModel.getObjects() if "CollisionModel" in obj.getClassName()],
        "catheter_collision_classes": [i.line_model.getClassName(), i.point_model.getClassName()],
        "CONTACT_RESPONSE": "NOT_OBSERVED",
        "PHYSICAL_PENETRATION": "NOT_OBSERVED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, choices=(1,), default=1)
    parser.add_argument("--action", type=float, nargs=3, default=(0, 0, 0))
    parser.add_argument("--vessel-root", default=str(DEFAULT_CONFIG.vessel_root))
    args = parser.parse_args()
    from pathlib import Path
    cfg = replace(DEFAULT_CONFIG, vessel_root=Path(args.vessel_root), vessel_seeds=(), eval_vessel_seeds=(args.seed,))
    bundle = load_vessel_bundle(cfg.vessel_root / ("arch_type_I_seed_%d" % args.seed))
    backend = SofaBackend(cfg)
    env = MagneticNavigationEnv(backend, config=cfg, mode="eval", bundles=[bundle])
    report = {"seed": args.seed, "action": args.action, "formal_training_started": False}
    start = time.perf_counter()
    try:
        observation, reset_info = env.reset(seed=args.seed, options={"vessel_seed": args.seed})
        report["reset"] = reset_info
        report["before"] = observed(backend._handles)
        integration_start = time.perf_counter()
        observation, reward, terminated, truncated, info = env.step(np.asarray(args.action, dtype=np.float32))
        report.update({"integration_seconds": time.perf_counter()-integration_start,
                       "observation": observation, "observation_shape": list(observation.shape),
                       "reward": reward, "terminated": terminated, "truncated": truncated, "info": info,
                       "after": observed(backend._handles), "physics_substeps_executed": backend.physics_substeps_executed})
        if info["simulation_error"] or backend.physics_substeps_executed != 22:
            raise RuntimeError("Single action did not finish 22 valid SOFA integrations")
    except Exception:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-start
        report["stages"] = backend.stages
        report["sofa_messages"] = backend._messages.messages if backend._messages is not None else backend.last_messages
        print(json.dumps(report, indent=2, default=json_value), flush=True)
        env.close()


if __name__ == "__main__":
    main()
