"""Read-only adapter for the already-generated stEVE Type-I vessel assets."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple
import json
import hashlib
import numpy as np

from .config import DEFAULT_VESSEL_ROOT, MM_TO_M

TYPE_I_PARENTS = {"bct": "aorta", "lcca": "aorta", "lsa": "aorta", "rcca": "bct", "rsa": "bct"}
REQUIRED_BRANCHES = ("aorta", "bct", "rcca", "rsa", "lcca", "lsa")


@dataclass(frozen=True)
class BranchData:
    name: str
    coordinates_mm: np.ndarray
    radii_mm: np.ndarray
    coordinates_m: np.ndarray
    radii_m: np.ndarray


@dataclass(frozen=True)
class VesselBundle:
    seed: int
    directory: Path
    collision_stl: Path
    visual_stl: Path
    centerline_graph_vtk: Path
    metadata_path: Path
    metadata: Mapping
    branches: Mapping[str, BranchData]
    parents: Mapping[str, str]
    insertion_point_mm: np.ndarray
    insertion_point_m: np.ndarray
    insertion_direction: np.ndarray
    target_centerlines: Mapping[str, Path]
    sdf_path: Optional[Path]

    @property
    def branch_names(self) -> Tuple[str, ...]:
        return tuple(self.branches.keys())

    def available_target_branches(self, requested) -> Tuple[str, ...]:
        return tuple(name for name in requested if name in self.branches)


class LegacyVTKPolyline:
    """Minimal strict reader for the legacy ASCII POLYDATA written here."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.points, self.lines, self.point_data = self._read()

    def _read(self):
        text = self.path.read_text(encoding="utf-8")
        if "ASCII" not in text[:200] or "DATASET POLYDATA" not in text:
            raise ValueError(f"Expected legacy ASCII POLYDATA: {self.path}")
        tokens = text.replace("\r", " ").replace("\n", " ").split()
        point_idx = tokens.index("POINTS")
        n_points = int(tokens[point_idx + 1])
        start = point_idx + 3
        points = np.asarray(tokens[start:start + 3 * n_points], dtype=np.float64).reshape(n_points, 3)
        line_idx = tokens.index("LINES")
        n_lines = int(tokens[line_idx + 1])
        cursor = line_idx + 3
        lines: List[np.ndarray] = []
        for _ in range(n_lines):
            count = int(tokens[cursor]); cursor += 1
            lines.append(np.asarray(tokens[cursor:cursor + count], dtype=np.int64))
            cursor += count
        point_data = {}
        if "POINT_DATA" in tokens:
            cursor = tokens.index("POINT_DATA") + 2
            while cursor < len(tokens):
                if tokens[cursor] != "SCALARS":
                    cursor += 1; continue
                name = tokens[cursor + 1]
                components = 1
                lookup_at = cursor + 3
                if tokens[lookup_at] != "LOOKUP_TABLE":
                    components = int(tokens[lookup_at]); lookup_at += 1
                data_start = lookup_at + 2
                count = n_points * components
                values = np.asarray(tokens[data_start:data_start + count], dtype=np.float64)
                point_data[name] = values.reshape(n_points, components).squeeze()
                cursor = data_start + count
        return points, lines, point_data


def _branch_names(metadata: Mapping, n_lines: int) -> Tuple[str, ...]:
    names = tuple(metadata.get("centerline_length_mm", {}).keys())
    if len(names) != n_lines:
        raise ValueError(f"metadata branch count {len(names)} != VTK line count {n_lines}")
    return names


def load_vessel_bundle(directory: Path) -> VesselBundle:
    directory = Path(directory).expanduser().resolve()
    required = [directory / name for name in (
        "collision_inner.stl", "visual_wall.stl", "centerline_graph.vtk", "metadata.json"
    )]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing vessel assets: " + ", ".join(missing))
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("arch_type") != "I" or metadata.get("length_unit") != "mm":
        raise ValueError("Only generated Type-I assets with millimetre units are supported")
    from .dataset_policy import verify_dataset_asset
    verify_dataset_asset(directory, metadata)
    for filename, expected in metadata.get("asset_sha256", {}).items():
        path = directory / filename
        if path.parent.resolve() != directory or not path.is_file():
            raise ValueError("Invalid bundle asset path: " + str(path))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Mismatched vessel mesh/centerline asset: " + str(path))
    vtk = LegacyVTKPolyline(directory / "centerline_graph.vtk")
    if "radius_mm" not in vtk.point_data:
        raise ValueError("centerline_graph.vtk has no radius_mm point data")
    names = _branch_names(metadata, len(vtk.lines))
    branches: Dict[str, BranchData] = {}
    for name, ids in zip(names, vtk.lines):
        coordinates_mm = np.asarray(vtk.points[ids], dtype=np.float64)
        radii_mm = np.asarray(vtk.point_data["radius_mm"][ids], dtype=np.float64)
        if len(coordinates_mm) < 2 or not np.all(np.isfinite(coordinates_mm)) or not np.all(np.isfinite(radii_mm)) or np.any(radii_mm <= 0.0):
            raise ValueError(f"Invalid branch geometry: {name}")
        branches[name] = BranchData(
            name, coordinates_mm, radii_mm,
            coordinates_mm * MM_TO_M, radii_mm * MM_TO_M,
        )
    if set(REQUIRED_BRANCHES) - set(branches):
        raise ValueError("Generated vessel lacks required Type-I branches")
    insertion = metadata["insertion"]
    point_mm = np.asarray(insertion["point_mm"], dtype=np.float64)
    direction = np.asarray(insertion["direction"], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-15)
    target_paths = {
        name: directory / filename
        for name, filename in metadata.get("target_paths", {}).items()
        if (directory / filename).is_file()
    }
    sdf = directory / "vessel_sdf.vti"
    return VesselBundle(
        seed=int(metadata["seed"]), directory=directory,
        collision_stl=directory / "collision_inner.stl",
        visual_stl=directory / "visual_wall.stl",
        centerline_graph_vtk=directory / "centerline_graph.vtk",
        metadata_path=directory / "metadata.json", metadata=metadata,
        branches=branches, parents=dict(TYPE_I_PARENTS),
        insertion_point_mm=point_mm, insertion_point_m=point_mm * MM_TO_M,
        insertion_direction=direction, target_centerlines=target_paths,
        sdf_path=sdf if sdf.is_file() else None,
    )


def discover_vessel_bundles(root: Path = DEFAULT_VESSEL_ROOT, split=None) -> Tuple[VesselBundle, ...]:
    root = Path(root).expanduser().resolve()
    if (root / "train").is_dir() and (root / "test").is_dir():
        names = ("train", "test") if split is None else (("test" if split == "eval" else split),)
        if any(name not in ("train", "test") for name in names):raise ValueError("Invalid dataset split")
        return tuple(bundle for name in names for bundle in discover_vessel_bundles(root / name))
    paths = list(root.glob("arch_type_I_seed_*")) + list(root.glob("seed_*"))
    bundles = tuple(
        load_vessel_bundle(path)
        for path in sorted(set(paths)) if path.is_dir()
    )
    if not bundles:
        raise FileNotFoundError(f"No generated Type-I vessels under {root}")
    return bundles


def coordinate_consistency(bundle: VesselBundle) -> dict:
    """Check explicit mm->m conversion and topology attachment without SOFA."""
    conversion = all(
        np.array_equal(branch.coordinates_m, branch.coordinates_mm * MM_TO_M)
        and np.array_equal(branch.radii_m, branch.radii_mm * MM_TO_M)
        for branch in bundle.branches.values()
    )
    gaps = {}
    for child, parent in bundle.parents.items():
        origin = bundle.branches[child].coordinates_mm[0]
        distance = np.linalg.norm(bundle.branches[parent].coordinates_mm - origin, axis=1)
        gaps[child] = float(np.min(distance))
    quality = bundle.metadata["validation"]["mesh_quality"]
    low = np.asarray(quality["bbox_min_mm"], dtype=float)
    high = np.asarray(quality["bbox_max_mm"], dtype=float)
    all_points = np.vstack([branch.coordinates_mm for branch in bundle.branches.values()])
    return {
        "mm_to_m_exact": bool(conversion),
        "branch_gap_mm": gaps,
        "all_branches_connected": bool(all(value < 1e-6 for value in gaps.values())),
        "centerlines_inside_stl_bbox": bool(np.all(all_points >= low) and np.all(all_points <= high)),
        "sdf_available": bundle.sdf_path is not None,
    }
