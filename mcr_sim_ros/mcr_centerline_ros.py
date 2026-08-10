import os
import struct
import warnings
from dataclasses import dataclass

from scipy.spatial.transform import Rotation as R


@dataclass
class CenterlineData:
    points_raw: list
    points_sim: list
    edges: list
    vtk_path: str = ''
    start_index: int = -1
    target_index: int = -1

    def to_python(self):
        endpoints = get_start_target_by_y(self)
        return {
            'vtk_path': self.vtk_path,
            'points_raw': self.points_raw,
            'points_sim': self.points_sim,
            'edges': self.edges,
            'start_index': endpoints['start_index'],
            'target_index': endpoints['target_index'],
            'start_point_sim': endpoints['start_point_sim'],
            'target_point_sim': endpoints['target_point_sim'],
        }


def _find_curve_endpoints(points, edges):
    if len(points) == 0:
        return []

    if len(edges) == 0:
        if len(points) == 1:
            return [0]
        return [0, len(points) - 1]

    degree = [0 for _ in range(len(points))]
    for edge in edges:
        if len(edge) < 2:
            continue
        a, b = int(edge[0]), int(edge[1])
        if 0 <= a < len(points):
            degree[a] += 1
        if 0 <= b < len(points):
            degree[b] += 1

    endpoint_candidates = [idx for idx, d in enumerate(degree) if d == 1]
    return endpoint_candidates


def get_start_target_by_y(centerline_data):
    points = centerline_data.points_sim if centerline_data is not None else []
    edges = centerline_data.edges if centerline_data is not None else []
    if len(points) == 0:
        return {
            'start_index': -1,
            'target_index': -1,
            'start_point_sim': None,
            'target_point_sim': None,
        }

    endpoints = _find_curve_endpoints(points, edges)
    if len(endpoints) == 2:
        a, b = endpoints[0], endpoints[1]
        if points[a][1] <= points[b][1]:
            start_index, target_index = a, b
        else:
            start_index, target_index = b, a
    else:
        warnings.warn(
            (
                'Centerline endpoint detection expected exactly 2 topology endpoints '
                f'(degree==1), but got {len(endpoints)}. '
                'Falling back to legacy y-extrema selection over all points.'
            ),
            RuntimeWarning,
        )
        y_values = [point[1] for point in points]
        start_index = min(range(len(points)), key=lambda idx: y_values[idx])
        target_index = max(range(len(points)), key=lambda idx: y_values[idx])

    if centerline_data is not None:
        centerline_data.start_index = start_index
        centerline_data.target_index = target_index

    return {
        'start_index': start_index,
        'target_index': target_index,
        'start_point_sim': points[start_index],
        'target_point_sim': points[target_index],
    }


def _build_edges_from_polyline_cells(cells):
    edges = []
    for ids in cells:
        if len(ids) < 2:
            continue
        for i in range(len(ids) - 1):
            edges.append([ids[i], ids[i + 1]])
    return edges


def _read_ascii_vtk_polydata(data_text):
    """Read ASCII VTK POLYDATA centerline.

    Supports both legacy VTK line-cell format and newer VTK format with
    OFFSETS / CONNECTIVITY blocks.
    """
    lines = data_text.splitlines()
    points = []
    polyline_cells = []

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue

        tokens = stripped.split()
        if len(tokens) == 0:
            index += 1
            continue

        key = tokens[0].upper()

        if key == 'POINTS' and len(tokens) >= 3:
            num_points = int(tokens[1])
            values = []
            index += 1
            while index < len(lines) and len(values) < 3 * num_points:
                values.extend(lines[index].strip().split())
                index += 1
            if len(values) < 3 * num_points:
                raise ValueError('Invalid VTK file: not enough point coordinates.')
            points = []
            for point_id in range(num_points):
                base = 3 * point_id
                points.append([
                    float(values[base]),
                    float(values[base + 1]),
                    float(values[base + 2]),
                ])
            continue

        if key == 'LINES' and len(tokens) >= 3:
            num_lines = int(tokens[1])
            index += 1

            while index < len(lines) and not lines[index].strip():
                index += 1

            # Newer VTK POLYDATA layout:
            # LINES <num_lines> <total_size>
            # OFFSETS vtktypeint64
            # 0 ...
            # CONNECTIVITY vtktypeint64
            # ...
            if index < len(lines) and lines[index].strip().upper().startswith('OFFSETS'):
                index += 1
                offsets = []

                while index < len(lines):
                    line_strip = lines[index].strip()
                    if not line_strip:
                        index += 1
                        continue
                    if line_strip.upper().startswith('CONNECTIVITY'):
                        break
                    offsets.extend([int(x) for x in line_strip.split()])
                    index += 1

                if index >= len(lines) or not lines[index].strip().upper().startswith('CONNECTIVITY'):
                    raise ValueError('Invalid VTK LINES OFFSETS block: CONNECTIVITY not found.')

                index += 1
                connectivity = []
                while index < len(lines):
                    line_strip = lines[index].strip()
                    if not line_strip:
                        index += 1
                        continue
                    first = line_strip.split()[0]
                    if not first.replace('-', '').isdigit():
                        break
                    connectivity.extend([int(x) for x in line_strip.split()])
                    index += 1

                if len(offsets) == 0:
                    raise ValueError('Invalid VTK LINES OFFSETS block: no offsets found.')

                if offsets[0] == 0 and len(offsets) >= num_lines + 1:
                    for i in range(num_lines):
                        start = int(offsets[i])
                        end = int(offsets[i + 1])
                        ids = connectivity[start:end]
                        if len(ids) >= 2:
                            polyline_cells.append(ids)
                else:
                    start = 0
                    for i in range(min(num_lines, len(offsets))):
                        end = int(offsets[i])
                        ids = connectivity[start:end]
                        if len(ids) >= 2:
                            polyline_cells.append(ids)
                        start = end
                continue

            # Legacy VTK layout:
            # LINES <num_lines> <total_size>
            # <count> <id0> <id1> ...
            parsed_lines = 0
            while index < len(lines) and parsed_lines < num_lines:
                line_tokens = lines[index].strip().split()
                index += 1
                if not line_tokens:
                    continue
                count = int(line_tokens[0])
                ids = [int(value) for value in line_tokens[1:1 + count]]
                polyline_cells.append(ids)
                parsed_lines += 1
            continue

        index += 1

    if len(points) == 0:
        raise ValueError('Invalid or unsupported ASCII VTK file: no POINTS found.')

    edges = _build_edges_from_polyline_cells(polyline_cells)
    if len(edges) == 0 and len(points) >= 2:
        edges = [[i, i + 1] for i in range(len(points) - 1)]

    return points, edges


def _read_binary_vtk_polydata(vtk_path):
    with open(vtk_path, 'rb') as file:
        _ = file.readline()
        _ = file.readline()
        format_line = file.readline().decode('latin1', errors='ignore').strip().upper()
        if 'BINARY' not in format_line:
            raise ValueError('VTK format is not binary.')

        dataset_line = file.readline().decode('latin1', errors='ignore').strip().upper()
        if 'POLYDATA' not in dataset_line:
            raise ValueError('Only POLYDATA VTK is supported.')

        points = []
        polyline_cells = []

        while True:
            line_bytes = file.readline()
            if not line_bytes:
                break

            line = line_bytes.decode('latin1', errors='ignore').strip()
            if not line:
                continue

            tokens = line.split()
            if len(tokens) == 0:
                continue

            key = tokens[0].upper()

            if key == 'POINTS' and len(tokens) >= 3:
                num_points = int(tokens[1])
                dtype = tokens[2].lower()
                if dtype == 'float':
                    scalar_size = 4
                    scalar_fmt = 'f'
                elif dtype == 'double':
                    scalar_size = 8
                    scalar_fmt = 'd'
                else:
                    raise ValueError(f'Unsupported binary POINTS dtype: {dtype}')

                num_scalars = 3 * num_points
                raw = file.read(num_scalars * scalar_size)
                if len(raw) < num_scalars * scalar_size:
                    raise ValueError('Binary VTK points block is truncated.')

                values = struct.unpack('>' + scalar_fmt * num_scalars, raw)
                points = []
                for point_id in range(num_points):
                    base = 3 * point_id
                    points.append([
                        float(values[base]),
                        float(values[base + 1]),
                        float(values[base + 2]),
                    ])
                continue

            if key == 'LINES' and len(tokens) >= 3:
                num_lines = int(tokens[1])
                total_values = int(tokens[2])
                raw = file.read(total_values * 4)
                if len(raw) < total_values * 4:
                    raise ValueError('Binary VTK LINES block is truncated.')

                values = struct.unpack('>' + 'i' * total_values, raw)
                cursor = 0
                polyline_cells = []
                for _line_id in range(num_lines):
                    if cursor >= len(values):
                        break
                    count = int(values[cursor])
                    cursor += 1
                    ids = [int(v) for v in values[cursor:cursor + count]]
                    cursor += count
                    polyline_cells.append(ids)
                continue

        if len(points) == 0:
            raise ValueError('Invalid or unsupported binary VTK file: no POINTS found.')

        edges = _build_edges_from_polyline_cells(polyline_cells)
        if len(edges) == 0 and len(points) >= 2:
            edges = [[i, i + 1] for i in range(len(points) - 1)]

        return points, edges


def _read_legacy_vtk_polydata(vtk_path):
    if not os.path.isfile(vtk_path):
        raise FileNotFoundError(f'Centerline VTK file not found: {vtk_path}')

    with open(vtk_path, 'rb') as file:
        file_bytes = file.read()

    header_text = file_bytes[:512].decode('latin1', errors='ignore').upper()
    if 'BINARY' in header_text:
        return _read_binary_vtk_polydata(vtk_path)

    data_text = file_bytes.decode('utf-8', errors='ignore')
    return _read_ascii_vtk_polydata(data_text)


def _transform_points_to_sim(
        points,
        T_env_sim=None,
        point_frame='env',
        scale=1.0,
        offset_sim=(0.0, 0.0, 0.0)):
    transformed = []

    if point_frame not in ('env', 'sim'):
        raise ValueError("point_frame must be 'env' or 'sim'.")

    if point_frame == 'env' and T_env_sim is None:
        raise ValueError('T_env_sim is required when point_frame is env.')

    rotation = None
    if point_frame == 'env':
        rotation = R.from_quat(T_env_sim[3:7])

    for point in points:
        p = [point[0] * scale, point[1] * scale, point[2] * scale]

        if point_frame == 'env':
            p_rot = rotation.apply(p)
            p_sim = [
                p_rot[0] + T_env_sim[0] + offset_sim[0],
                p_rot[1] + T_env_sim[1] + offset_sim[1],
                p_rot[2] + T_env_sim[2] + offset_sim[2]
            ]
        else:
            p_sim = [
                p[0] + offset_sim[0],
                p[1] + offset_sim[1],
                p[2] + offset_sim[2]
            ]

        transformed.append(p_sim)

    return transformed


def load_centerline_data(
        vtk_path,
        T_env_sim=None,
        point_frame='env',
        scale=1.0,
        offset_sim=(0.0, 0.0, 0.0)):
    points_raw, edges = _read_legacy_vtk_polydata(vtk_path)
    points_sim = _transform_points_to_sim(
        points=points_raw,
        T_env_sim=T_env_sim,
        point_frame=point_frame,
        scale=scale,
        offset_sim=offset_sim)

    return CenterlineData(
        points_raw=points_raw,
        points_sim=points_sim,
        edges=edges,
        vtk_path=vtk_path)


def add_centerline_to_sofa(
        root_node,
        centerline_data,
        node_name='Centerline',
        line_color=(0.0, 1.0, 0.0, 1.0),
        target_color=(1.0, 1.0, 0.0, 1.0),
        target_point_sim=None,
        show_target=True,
        target_scale=0.01):
    if centerline_data is None:
        return None

    points = centerline_data.points_sim
    edges = centerline_data.edges

    if len(points) == 0:
        return None

    centerline_node = root_node.addChild(node_name)
    centerline_node.addObject(
        'MechanicalObject',
        name='CenterlineMO',
        template='Vec3d',
        position=points)
    centerline_node.addObject(
        'EdgeSetTopologyContainer',
        name='CenterlineTopo',
        edges=edges)
    centerline_node.addObject(
        'EdgeSetTopologyModifier',
        name='CenterlineTopoMod')

    centerline_visu = centerline_node.addChild('Visu')
    centerline_visu.addObject(
        'OglModel',
        name='CenterlineVisual',
        position='@../CenterlineMO.position',
        edges='@../CenterlineTopo.edges',
        color=list(line_color))
    centerline_visu.addObject(
        'IdentityMapping',
        input='@../CenterlineMO',
        output='@CenterlineVisual')

    target_node = None
    if show_target:
        if target_point_sim is None:
            endpoints = get_start_target_by_y(centerline_data)
            target_point_sim = endpoints['target_point_sim']
        if target_point_sim is None:
            target_point_sim = points[-1]

        target_node = root_node.addChild('TargetPoint')

        radius = max(float(target_scale) * 0.1, 0.0002)
        tx, ty, tz = float(target_point_sim[0]), float(target_point_sim[1]), float(target_point_sim[2])
        target_marker_points = [
            [tx - radius, ty, tz], [tx + radius, ty, tz],
            [tx, ty - radius, tz], [tx, ty + radius, tz],
            [tx, ty, tz - radius], [tx, ty, tz + radius],
        ]
        target_marker_edges = [[0, 1], [2, 3], [4, 5]]

        target_node.addObject(
            'MechanicalObject',
            name='TargetMarkerMO',
            template='Vec3d',
            position=target_marker_points)
        target_node.addObject(
            'EdgeSetTopologyContainer',
            name='TargetMarkerTopo',
            edges=target_marker_edges)
        target_node.addObject(
            'EdgeSetTopologyModifier',
            name='TargetMarkerTopoMod')

        target_visu = target_node.addChild('Visu')
        target_visu.addObject(
            'OglModel',
            name='TargetVisual',
            position='@../TargetMarkerMO.position',
            edges='@../TargetMarkerTopo.edges',
            color=list(target_color),
            lineWidth=1)
        target_visu.addObject(
            'IdentityMapping',
            input='@../TargetMarkerMO',
            output='@TargetVisual')

    return {
        'centerline_node': centerline_node,
        'target_node': target_node,
        'centerline_data': centerline_data,
    }
