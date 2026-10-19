"""Local official stEVE AorticArchRandom adapter. No network access or fallback.

The adult-diameter 1.0 caches are intentionally not used for ArchVariety 0.85.
Mesh construction/export reuses the already-present server adapter.
"""
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tempfile
import numpy as np
from .config import DEFAULT_CONFIG, DEFAULT_VESSEL_ROOT, PYTHON_DIR

SOURCE_ROOT = PYTHON_DIR / "stEVE_anatomical_vessels"
UPSTREAM = SOURCE_ROOT / "upstream" / "stEVE"


@dataclass(frozen=True)
class GenerationConfig:
    diameter_scale: float = 0.85
    width_min: float = 0.7
    width_max: float = 1.3
    width_count: int = 61
    height_min: float = 0.7
    height_max: float = 1.3
    height_count: int = 61
    voxel_spacing_mm: tuple = (0.6, 0.6, 0.6)
    gaussian_sigma: float = 1.0
    gaussian_passes: int = 2
    marching_cubes_level: float = 0.5
    decimate_target_reduction: float = 0.25
    padding_layers: int = 5


def _official():
    required = UPSTREAM / "eve/intervention/vesseltree/aorticarchrandom.py"
    if not required.is_file():
        raise FileNotFoundError("Missing local official AorticArchRandom: " + str(required))
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import eve
    if UPSTREAM.resolve() not in Path(eve.__file__).resolve().parents:
        raise RuntimeError("A different eve package is already imported; refusing non-local generator")
    from eve.intervention.vesseltree import AorticArchRandom, ArchType
    return AorticArchRandom, ArchType


def sample_geometry(seed, cfg=GenerationConfig()):
    AorticArchRandom, ArchType = _official()
    tree = AorticArchRandom(
        scale_width_array=np.linspace(cfg.width_min, cfg.width_max, cfg.width_count).tolist(),
        scale_heigth_array=np.linspace(cfg.height_min, cfg.height_max, cfg.height_count).tolist(),
        scale_diameter_array=[cfg.diameter_scale], arch_types_filter=[ArchType.I],
        seeds_vessel=[int(seed)], episodes_between_change=1)
    tree.reset(episode_nr=0, seed=int(seed))
    return tree


def geometry_hash(branches):
    digest = hashlib.sha256()
    for b in branches:
        digest.update(b.name.encode())
        digest.update(np.asarray(b.coordinates, dtype="<f8").tobytes())
        digest.update(np.asarray(b.radii, dtype="<f8").tobytes())
    return digest.hexdigest()


def _exports():
    path = SOURCE_ROOT / "src/export_assets.py"
    spec = importlib.util.spec_from_file_location("mcr_local_vessel_exports", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_one(seed, root=DEFAULT_VESSEL_ROOT, cfg=GenerationConfig()):
    from .vessel_bundle import load_vessel_bundle, TYPE_I_PARENTS
    root = Path(root).resolve()
    output = root / ("arch_type_I_seed_%d" % int(seed))
    signature = asdict(cfg)
    # JSON canonicalizes tuples as arrays.
    signature = json.loads(json.dumps(signature))
    if output.exists():
        bundle = load_vessel_bundle(output)
        if bundle.metadata.get("generation_config") != signature:
            raise ValueError("Refusing mismatched cached geometry: " + str(output))
        return bundle
    tree = sample_geometry(seed, cfg)
    branches = tree.branches
    fingerprint = geometry_hash(branches)
    if fingerprint != geometry_hash(sample_geometry(seed, cfg).branches):
        raise RuntimeError("Official geometry is not reproducible")
    exports = _exports()
    root.mkdir(parents=True, exist_ok=True)
    # Failed generation directories are retained for diagnosis, never overwritten.
    stage = Path(tempfile.mkdtemp(prefix="pending_seed_%d_" % int(seed), dir=str(root)))
    meshing = {k: signature[k] for k in ("voxel_spacing_mm", "gaussian_sigma", "gaussian_passes", "marching_cubes_level", "decimate_target_reduction", "padding_layers")}
    poly, raw = exports.generate_lumen_mesh(branches, stage, meshing)
    exports.export_centerlines(branches, stage)
    quality = exports.mesh_quality(poly)
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    cloud = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.vstack([b.coordinates for b in branches]).astype(float), deep=True))
    cloud.SetPoints(points)
    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetInputData(cloud)
    enclosed.SetSurfaceData(poly)
    enclosed.SetTolerance(1e-7)
    enclosed.Update()
    inside = vtk_to_numpy(enclosed.GetOutput().GetPointData().GetArray("SelectedPoints"))
    passed = quality["connected_components"] == 1 and quality["degenerate_triangles"] == 0 and bool(np.all(inside))
    metadata = {
        "generator": "official-local stEVE AorticArchRandom", "seed": int(seed),
        "arch_type": "I", "length_unit": "mm", "scale_diameter_array": [cfg.diameter_scale],
        "generation_config": signature, "geometry_sha256": fingerprint,
        "sampled_scaling_xyzd": list(tree._aortic_arch.scaling_xyzd),
        "sampled_rotation_yzx_deg": list(tree._aortic_arch.rotation_yzx_deg),
        "local_upstream": str(UPSTREAM), "centerline_length_mm": exports.branch_lengths(branches),
        "parents": TYPE_I_PARENTS,
        "insertion": {"point_mm": np.asarray(tree.insertion.position).tolist(), "direction": np.asarray(tree.insertion.direction).tolist()},
        "branching_positions_mm": {b.name: np.asarray(b.coordinates[0]).tolist() for b in branches if b.name != "aorta"},
        "target_paths": {n: "target_%s_centerline.vtk" % n for n in ("lcca", "rcca", "lsa", "rsa")},
        "raw_marching_cubes": raw,
        "validation": {"passed": passed, "mesh_quality": quality,
                       "centerline_mesh_enclosed": bool(np.all(inside)), "outside_centerline_points": int(np.sum(inside == 0))},
        "asset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.iterdir()) if p.suffix in (".stl", ".vtk")},
    }
    (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not passed:
        raise RuntimeError("Generated geometry validation failed; retained at " + str(stage))
    stage.rename(output)
    return load_vessel_bundle(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--split", choices=("train", "eval", "all"))
    parser.add_argument("--root", type=Path, default=DEFAULT_VESSEL_ROOT)
    args = parser.parse_args()
    if not args.seed and not args.split:
        parser.error("Choose explicit --seed or --split; no implicit bulk generation")
    seeds = args.seed or []
    if args.split in ("train", "all"):
        seeds += list(DEFAULT_CONFIG.vessel_seeds)
    if args.split in ("eval", "all"):
        seeds += list(DEFAULT_CONFIG.eval_vessel_seeds)
    for seed in dict.fromkeys(seeds):
        bundle = generate_one(seed, args.root)
        print(json.dumps({"seed": seed, "directory": str(bundle.directory), "validation": bundle.metadata["validation"]}, indent=2))


if __name__ == "__main__":
    main()
