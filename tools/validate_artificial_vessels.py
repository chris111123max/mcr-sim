#!/usr/bin/env python3
"""Validate generated mCR artificial-vessel assets without SOFA/VTK dependencies."""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

# This repository is the server's mCR_simulator-master/python directory.
PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from mcr_sim.paths import TRAIN_MESH_DIR


MODEL_IDS = [f"C{i:02d}" for i in range(1, 6)] + [f"B{i:02d}" for i in range(1, 6)]
REQUIRED = (
    "collision_inner.stl",
    "Segmentation.stl",
    "visual_wall.stl",
    "Centerline model.vtk",
    "centerline.vtk",
    "vessel_sdf.vti",
    "metadata.json",
    "preview.png",
)


def validate_stl(path: Path, expected_openings: int | None = None) -> int:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"truncated binary STL: {path}")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * count:
        raise ValueError(f"binary STL size/count mismatch: {path}")
    record_dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    triangles = np.frombuffer(data, dtype=record_dtype, offset=84, count=count)[
        "vertices"
    ].astype(np.float64)
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if not np.isfinite(triangles).all() or np.any(areas <= 1e-8):
        raise ValueError(f"non-finite or degenerate triangles: {path}")

    if expected_openings is not None:
        _, inverse = np.unique(
            np.round(triangles.reshape((-1, 3)), decimals=4), axis=0, return_inverse=True
        )
        triangle_ids = inverse.reshape((-1, 3))
        edges = np.sort(
            np.concatenate(
                (
                    triangle_ids[:, [0, 1]],
                    triangle_ids[:, [1, 2]],
                    triangle_ids[:, [2, 0]],
                ),
                axis=0,
            ),
            axis=1,
        )
        unique_edges, incidence = np.unique(edges, axis=0, return_counts=True)
        if np.any(incidence > 2):
            raise ValueError(f"non-manifold collision edges: {path}")
        boundary = unique_edges[incidence == 1]
        adjacency: dict[int, set[int]] = {}
        for a, b in boundary:
            adjacency.setdefault(int(a), set()).add(int(b))
            adjacency.setdefault(int(b), set()).add(int(a))
        if any(len(neighbours) != 2 for neighbours in adjacency.values()):
            raise ValueError(f"open or branching boundary loop: {path}")
        unseen = set(adjacency)
        opening_count = 0
        while unseen:
            opening_count += 1
            stack = [unseen.pop()]
            while stack:
                node = stack.pop()
                for neighbour in adjacency[node]:
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        stack.append(neighbour)
        if opening_count != expected_openings:
            raise ValueError(
                f"unexpected openings in {path}: expected {expected_openings}, got {opening_count}"
            )
    return count


def validate_vti(path: Path) -> tuple[int, int, int]:
    text = path.read_text(encoding="utf-8")
    extent_match = re.search(r'WholeExtent="([^"]+)"', text)
    data_match = re.search(r"<DataArray[^>]*>([^<]+)</DataArray>", text)
    if extent_match is None or data_match is None:
        raise ValueError(f"invalid VTI XML: {path}")
    extent = [int(value) for value in extent_match.group(1).split()]
    dims = (extent[1] + 1, extent[3] + 1, extent[5] + 1)
    payload = base64.b64decode(data_match.group(1))
    blocks, block_size, last_size, compressed_size = struct.unpack_from("<4I", payload, 0)
    if blocks != 1 or block_size != last_size:
        raise ValueError(f"unexpected VTI compression header: {path}")
    raw = zlib.decompress(payload[16:16 + compressed_size])
    if len(raw) != 4 * dims[0] * dims[1] * dims[2]:
        raise ValueError(f"VTI scalar length mismatch: {path}")
    return dims


def validate_model(root: Path, model_id: str) -> dict:
    model_dir = root / model_id
    missing = [name for name in REQUIRED if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{model_id}: missing {missing}")

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    expected_openings = 7 if model_id.startswith("B") else 2
    triangles = validate_stl(
        model_dir / "collision_inner.stl", expected_openings=expected_openings
    )
    validate_stl(model_dir / "visual_wall.stl")
    if (model_dir / "collision_inner.stl").read_bytes() != (model_dir / "Segmentation.stl").read_bytes():
        raise ValueError(f"{model_id}: Segmentation.stl is not the collision alias")
    if triangles != int(metadata["collision_validation"]["triangle_count"]):
        raise ValueError(f"{model_id}: metadata triangle count mismatch")
    if int(metadata["collision_validation"]["degenerate_triangles"]) != 0:
        raise ValueError(f"{model_id}: collision STL contains degenerate triangles")

    targets = sorted(model_dir.glob("target_*_centerline.vtk"))
    expected_targets = 6 if model_id.startswith("B") else 0
    if len(targets) != expected_targets:
        raise ValueError(f"{model_id}: expected {expected_targets} targets, got {len(targets)}")
    dims = validate_vti(model_dir / "vessel_sdf.vti")
    return {"model": model_id, "triangles": triangles, "targets": len(targets), "sdf_dims": dims}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=TRAIN_MESH_DIR)
    args = parser.parse_args()

    rows = [validate_model(args.root.resolve(), model_id) for model_id in MODEL_IDS]
    for row in rows:
        print(
            f"[OK] {row['model']}: triangles={row['triangles']} "
            f"targets={row['targets']} sdf={list(row['sdf_dims'])}"
        )
    print(f"[DONE] validated {len(rows)} models in {args.root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
