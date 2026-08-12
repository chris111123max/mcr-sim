"""Runtime helpers for generated vessel assets.

The artificial-vessel generator writes a compact VTK ImageData (``.vti``)
signed-distance field using one or more zlib-compressed blocks.  The training
workers should not need the heavyweight VTK Python package merely to query
that regular grid, so this module implements the small subset of VTI that the
generator emits.

All values stored in the VTI are in the source asset coordinate system
(millimetres for the generated B/C vessels).  Coordinate conversion into and
out of SOFA's metre-based simulation frame remains the environment's
responsibility because it owns the per-episode vessel scale and rigid pose.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Union
from xml.etree import ElementTree

import numpy as np


PathLike = Union[str, Path]


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _find_first(root, local_name: str):
    for element in root.iter():
        if _local_name(element.tag) == local_name:
            return element
    return None


def _decode_vtk_zlib_payload(encoded_text: str, byte_order: str) -> bytes:
    payload = base64.b64decode("".join(str(encoded_text or "").split()))
    endian = "<" if str(byte_order).lower() == "littleendian" else ">"
    if len(payload) < 12:
        raise ValueError("VTI compressed payload is shorter than its block header.")

    num_blocks, block_size, last_block_size = struct.unpack_from(endian + "3I", payload, 0)
    if num_blocks < 1:
        raise ValueError("VTI compressed payload contains no blocks.")
    header_size = 12 + 4 * int(num_blocks)
    if len(payload) < header_size:
        raise ValueError("VTI compressed payload has a truncated block-size table.")
    compressed_sizes = struct.unpack_from(endian + f"{int(num_blocks)}I", payload, 12)

    cursor = header_size
    raw_blocks = []
    for block_index, compressed_size in enumerate(compressed_sizes):
        end = cursor + int(compressed_size)
        if end > len(payload):
            raise ValueError("VTI compressed payload has a truncated data block.")
        raw = zlib.decompress(payload[cursor:end])
        expected = int(last_block_size if block_index == int(num_blocks) - 1 else block_size)
        if len(raw) != expected:
            raise ValueError(
                "VTI decompressed block length mismatch: "
                f"block={block_index} actual={len(raw)} expected={expected}"
            )
        raw_blocks.append(raw)
        cursor = end
    return b"".join(raw_blocks)


@dataclass(frozen=True)
class SignedDistanceGrid:
    """Regular signed-distance grid in source-asset coordinates."""

    values: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    scalar_name: str
    source_path: str

    def sample(self, points_source) -> np.ndarray:
        """Trilinearly sample one or more ``(x, y, z)`` source-space points.

        Points outside the VTI extent return positive infinity, which is the
        conservative outside-lumen result for the generated SDF convention.
        """

        points = np.asarray(points_source, dtype=np.float64)
        scalar_input = points.ndim == 1
        points = points.reshape((-1, 3))

        nz, ny, nx = self.values.shape
        dims_xyz = np.asarray([nx, ny, nz], dtype=np.int64)
        grid_xyz = (points - self.origin[None, :]) / self.spacing[None, :]
        valid = np.all(np.isfinite(grid_xyz), axis=1)
        valid &= np.all(grid_xyz >= 0.0, axis=1)
        valid &= np.all(grid_xyz <= (dims_xyz - 1)[None, :], axis=1)

        output = np.full(len(points), np.inf, dtype=np.float64)
        if np.any(valid):
            q = grid_xyz[valid]
            lower = np.floor(q).astype(np.int64)
            lower = np.minimum(lower, (dims_xyz - 2)[None, :])
            lower = np.maximum(lower, 0)
            fraction = np.clip(q - lower, 0.0, 1.0)

            x0, y0, z0 = lower[:, 0], lower[:, 1], lower[:, 2]
            x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
            fx, fy, fz = fraction[:, 0], fraction[:, 1], fraction[:, 2]
            field = self.values

            c000 = field[z0, y0, x0]
            c100 = field[z0, y0, x1]
            c010 = field[z0, y1, x0]
            c110 = field[z0, y1, x1]
            c001 = field[z1, y0, x0]
            c101 = field[z1, y0, x1]
            c011 = field[z1, y1, x0]
            c111 = field[z1, y1, x1]

            c00 = c000 * (1.0 - fx) + c100 * fx
            c10 = c010 * (1.0 - fx) + c110 * fx
            c01 = c001 * (1.0 - fx) + c101 * fx
            c11 = c011 * (1.0 - fx) + c111 * fx
            c0 = c00 * (1.0 - fy) + c10 * fy
            c1 = c01 * (1.0 - fy) + c11 * fy
            output[valid] = c0 * (1.0 - fz) + c1 * fz

        return output[0] if scalar_input else output

    def gradient(self, points_source) -> np.ndarray:
        """Return the source-space SDF gradient at one or more points.

        The generator stores distance and coordinates in the same units, so a
        well-resolved signed-distance gradient has approximately unit length.
        Central differences use one grid spacing on each axis.  Samples too
        close to the VTI boundary return a zero vector rather than propagating
        infinities into the policy observation.
        """

        points = np.asarray(points_source, dtype=np.float64)
        scalar_input = points.ndim == 1
        points = points.reshape((-1, 3))
        result = np.zeros((len(points), 3), dtype=np.float64)
        for axis in range(3):
            offset = np.zeros(3, dtype=np.float64)
            offset[axis] = float(self.spacing[axis])
            lower = np.asarray(self.sample(points - offset[None, :]), dtype=np.float64)
            upper = np.asarray(self.sample(points + offset[None, :]), dtype=np.float64)
            valid = np.isfinite(lower) & np.isfinite(upper)
            result[valid, axis] = (
                upper[valid] - lower[valid]
            ) / (2.0 * float(self.spacing[axis]))
        return result[0] if scalar_input else result


@lru_cache(maxsize=32)
def load_signed_distance_grid(path: PathLike) -> SignedDistanceGrid:
    """Load and cache the generator's compressed ``vessel_sdf.vti`` file."""

    source_path = Path(path).expanduser().resolve()
    root = ElementTree.parse(str(source_path)).getroot()
    image = _find_first(root, "ImageData")
    if image is None:
        raise ValueError(f"VTI ImageData element is missing: {source_path}")

    extent = [int(v) for v in str(image.attrib["WholeExtent"]).split()]
    if len(extent) != 6:
        raise ValueError(f"Invalid VTI WholeExtent: {image.attrib.get('WholeExtent')}")
    nx = extent[1] - extent[0] + 1
    ny = extent[3] - extent[2] + 1
    nz = extent[5] - extent[4] + 1
    if min(nx, ny, nz) < 2:
        raise ValueError(f"VTI grid must have at least two samples per axis: {(nx, ny, nz)}")

    origin = np.asarray([float(v) for v in str(image.attrib["Origin"]).split()], dtype=np.float64)
    spacing = np.asarray([float(v) for v in str(image.attrib["Spacing"]).split()], dtype=np.float64)
    if origin.shape != (3,) or spacing.shape != (3,) or np.any(spacing <= 0.0):
        raise ValueError(f"Invalid VTI origin/spacing in {source_path}")

    data_array = None
    for candidate in root.iter():
        if _local_name(candidate.tag) == "DataArray" and candidate.attrib.get("Name") == "signed_distance_mm":
            data_array = candidate
            break
    if data_array is None:
        raise ValueError(f"VTI signed_distance_mm DataArray is missing: {source_path}")
    if str(data_array.attrib.get("format", "")).lower() != "binary":
        raise ValueError("Only the generator's binary VTI DataArray format is supported.")
    if str(data_array.attrib.get("type", "")).lower() != "float32":
        raise ValueError("Only Float32 signed-distance VTI arrays are supported.")

    byte_order = root.attrib.get("byte_order", "LittleEndian")
    raw = _decode_vtk_zlib_payload(data_array.text or "", byte_order=byte_order)
    dtype = np.dtype("<f4" if str(byte_order).lower() == "littleendian" else ">f4")
    values = np.frombuffer(raw, dtype=dtype)
    expected = int(nx) * int(ny) * int(nz)
    if values.size != expected:
        raise ValueError(
            f"VTI scalar count mismatch: actual={values.size} expected={expected} path={source_path}"
        )
    values = np.asarray(values.reshape((nz, ny, nx)), dtype=np.float32)
    values.setflags(write=False)
    origin.setflags(write=False)
    spacing.setflags(write=False)
    return SignedDistanceGrid(
        values=values,
        origin=origin,
        spacing=spacing,
        scalar_name="signed_distance_mm",
        source_path=str(source_path),
    )


@lru_cache(maxsize=32)
def load_vessel_metadata(path: PathLike) -> Dict[str, Any]:
    """Load immutable generation metadata once per worker process."""

    source_path = Path(path).expanduser().resolve()
    with source_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Vessel metadata must contain a JSON object: {source_path}")
    return data
