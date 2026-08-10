import os
import struct
import warnings
import math
import base64
import zlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class CenterlineData:
	points_raw: list
	points_sim: list
	edges: list
	radius_raw: list = None
	radius_sim: list = None
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
			'radius_raw': self.radius_raw,
			'radius_sim': self.radius_sim,
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
	lines = data_text.splitlines()
	points = []
	polyline_cells = []
	point_arrays = {}
	point_data_count = None

	def _is_section_header(token):
		return token.upper() in {
			'POINTS',
			'LINES',
			'POLYGONS',
			'VERTICES',
			'TRIANGLE_STRIPS',
			'POINT_DATA',
			'CELL_DATA',
			'SCALARS',
			'VECTORS',
			'TENSORS',
			'NORMALS',
			'FIELD',
			'TEXTURE_COORDINATES',
			'COLOR_SCALARS',
			'LOOKUP_TABLE',
			'OFFSETS',
			'CONNECTIVITY',
		}

	def _parse_numeric_tokens(tokens):
		try:
			return [float(t) for t in tokens], True
		except Exception:
			return [], False

	def _consume_numeric_values(start_index, max_values=None):
		values = []
		index = start_index
		while index < len(lines):
			ln = lines[index].strip()
			if not ln:
				index += 1
				continue
			parts = ln.split()
			if _is_section_header(parts[0]):
				break
			parsed, ok = _parse_numeric_tokens(parts)
			if not ok:
				break
			values.extend(parsed)
			index += 1
			if max_values is not None and len(values) >= max_values:
				break
		return values, index

	index = 0
	while index < len(lines):
		stripped = lines[index].strip()
		if not stripped:
			index += 1
			continue

		tokens = stripped.split()
		key = tokens[0].upper()

		if key == 'POINTS' and len(tokens) >= 3:
			num_points = int(tokens[1])
			point_data_count = num_points
			values = []
			index += 1
			while index < len(lines) and len(values) < 3 * num_points:
				values.extend(lines[index].strip().split())
				index += 1
			if len(values) < 3 * num_points:
				raise ValueError('Invalid VTK file: not enough point coordinates.')
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
			parsed_lines = 0
			if index < len(lines) and lines[index].strip().startswith('OFFSETS'):
				index += 1
				offsets = []
				while index < len(lines) and not lines[index].strip().startswith('CONNECTIVITY'):
					offsets.extend([int(x) for x in lines[index].strip().split()])
					index += 1
				if index < len(lines) and lines[index].strip().startswith('CONNECTIVITY'):
					index += 1
					connectivity = []
					while index < len(lines):
						line_strip = lines[index].strip()
						if not line_strip:
							index += 1
							continue
						if not line_strip.split()[0].replace('-','').isdigit():
							break
						connectivity.extend([int(x) for x in line_strip.split()])
						index += 1
					for i in range(num_lines):
						start = offsets[i]
						end = offsets[i+1] if i+1 < len(offsets) else len(connectivity)
						polyline_cells.append(connectivity[start:end])
				continue

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

		# Parse POINT_DATA / SCALARS / FIELD arrays if present
		if key == 'POINT_DATA':
			point_data_count = int(tokens[1]) if len(tokens) >= 2 else point_data_count
			index += 1
			while index < len(lines):
				ln = lines[index].strip()
				if not ln:
					index += 1
					continue
				parts = ln.split()
				key_pd = parts[0].upper()
				if key_pd == 'SCALARS' and len(parts) >= 2:
					scalar_name = parts[1]
					num_components = int(parts[3]) if len(parts) >= 4 else 1
					expected = int(point_data_count) * int(num_components) if point_data_count is not None else None
					index += 1
					if index < len(lines) and lines[index].strip().upper().startswith('LOOKUP_TABLE'):
						index += 1
					vals, index = _consume_numeric_values(index, max_values=expected)
					if expected is not None and len(vals) >= expected:
						vals = vals[:expected]
						if num_components == 1 and point_data_count is not None and len(vals) == int(point_data_count):
							point_arrays[scalar_name] = vals
					continue

				if key_pd == 'FIELD' and len(parts) >= 3:
					try:
						num_fields = int(parts[2])
					except Exception:
						num_fields = 0
					index += 1
					for _ in range(num_fields):
						while index < len(lines) and not lines[index].strip():
							index += 1
						if index >= len(lines):
							break
						field_parts = lines[index].strip().split()
						if len(field_parts) < 4:
							index += 1
							continue
						field_name = field_parts[0]
						try:
							num_components = int(field_parts[1])
							num_tuples = int(field_parts[2])
						except Exception:
							index += 1
							continue
						total_values = int(num_components) * int(num_tuples)
						index += 1
						vals, index = _consume_numeric_values(index, max_values=total_values)
						if total_values > 0 and len(vals) >= total_values:
							vals = vals[:total_values]
							if num_components == 1 and point_data_count is not None and num_tuples == int(point_data_count):
								point_arrays[field_name] = vals
					continue

				if key_pd in ('POINT_DATA', 'CELL_DATA', 'LINES', 'POLYGONS', 'VERTICES', 'TRIANGLE_STRIPS'):
					break

				index += 1
			continue

		index += 1

	if len(points) == 0:
		raise ValueError('Invalid or unsupported ASCII VTK file: no POINTS found.')

	edges = _build_edges_from_polyline_cells(polyline_cells)
	if len(edges) == 0 and len(points) >= 2:
		edges = [[i, i + 1] for i in range(len(points) - 1)]

	return points, edges, point_arrays



def _legacy_vtk_binary_numpy_dtype(dtype_name):
	"""Return a big-endian numpy dtype for legacy binary VTK scalar types."""
	name = str(dtype_name or '').strip().lower()
	name = name.replace(' ', '_')

	dtype_map = {
		'char': 'i1',
		'signed_char': 'i1',
		'unsigned_char': 'u1',
		'uchar': 'u1',
		'uint8': 'u1',
		'int8': 'i1',
		'short': 'i2',
		'unsigned_short': 'u2',
		'ushort': 'u2',
		'int16': 'i2',
		'uint16': 'u2',
		'int': 'i4',
		'unsigned_int': 'u4',
		'uint': 'u4',
		'int32': 'i4',
		'uint32': 'u4',
		'long': 'i8',
		'unsigned_long': 'u8',
		'ulong': 'u8',
		'int64': 'i8',
		'uint64': 'u8',
		'float': 'f4',
		'float32': 'f4',
		'double': 'f8',
		'float64': 'f8',
	}

	if name not in dtype_map:
		raise ValueError(f'Unsupported legacy binary VTK dtype: {dtype_name}')

	code = dtype_map[name]
	if code.endswith('1'):
		return np.dtype(code)
	return np.dtype('>' + code)


def _read_legacy_vtk_binary_values(file, num_values, dtype_name, context='VTK binary data'):
	"""Read big-endian numeric values from a legacy binary VTK stream."""
	num_values = int(num_values)
	if num_values < 0:
		raise ValueError(f'Invalid value count for {context}: {num_values}')

	dtype = _legacy_vtk_binary_numpy_dtype(dtype_name)
	num_bytes = int(num_values) * int(dtype.itemsize)
	raw = file.read(num_bytes)
	if len(raw) < num_bytes:
		raise ValueError(f'Binary VTK block is truncated while reading {context}.')

	return np.frombuffer(raw, dtype=dtype).copy()


def _read_next_nonempty_legacy_vtk_line(file):
	"""Read the next non-empty ASCII header line from a legacy binary VTK file."""
	while True:
		line_bytes = file.readline()
		if not line_bytes:
			return None
		line = line_bytes.decode('latin1', errors='ignore').strip()
		if line:
			return line

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
		point_arrays = {}
		point_data_count = None

		while True:
			line = _read_next_nonempty_legacy_vtk_line(file)
			if line is None:
				break

			tokens = line.split()
			if len(tokens) == 0:
				continue

			key = tokens[0].upper()

			if key == 'POINTS' and len(tokens) >= 3:
				num_points = int(tokens[1])
				dtype = tokens[2].lower()
				values = _read_legacy_vtk_binary_values(
					file,
					3 * num_points,
					dtype,
					context='POINTS',
				)
				if values.size < 3 * num_points:
					raise ValueError('Binary VTK points block is truncated.')

				points = []
				for point_id in range(num_points):
					base = 3 * point_id
					points.append([
						float(values[base]),
						float(values[base + 1]),
						float(values[base + 2]),
					])
				point_data_count = num_points
				continue

			if key == 'LINES' and len(tokens) >= 3:
				num_lines = int(tokens[1])
				total_values = int(tokens[2])
				values = _read_legacy_vtk_binary_values(
					file,
					total_values,
					'int',
					context='LINES',
				).astype(np.int64, copy=False)

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

			if key == 'POINT_DATA' and len(tokens) >= 2:
				point_data_count = int(tokens[1])

				while True:
					pd_line = _read_next_nonempty_legacy_vtk_line(file)
					if pd_line is None:
						break

					pd_tokens = pd_line.split()
					if len(pd_tokens) == 0:
						continue

					pd_key = pd_tokens[0].upper()

					if pd_key in {
						'CELL_DATA',
						'POINTS',
						'LINES',
						'POLYGONS',
						'VERTICES',
						'TRIANGLE_STRIPS',
					}:
						# Centerline use only needs POINT_DATA.  If a later section
						# appears, stop parsing point arrays and ignore the remaining
						# non-point data.
						break

					if pd_key == 'SCALARS' and len(pd_tokens) >= 3:
						scalar_name = pd_tokens[1]
						scalar_dtype = pd_tokens[2]
						num_components = int(pd_tokens[3]) if len(pd_tokens) >= 4 else 1
						expected = int(point_data_count) * int(num_components)

						lookup_line = _read_next_nonempty_legacy_vtk_line(file)
						if lookup_line is None:
							break
						if not lookup_line.upper().startswith('LOOKUP_TABLE'):
							raise ValueError(
								f'Invalid binary VTK SCALARS block for {scalar_name}: '
								'LOOKUP_TABLE line is missing.'
							)

						values = _read_legacy_vtk_binary_values(
							file,
							expected,
							scalar_dtype,
							context=f'SCALARS {scalar_name}',
						)
						if (
							num_components == 1
							and point_data_count is not None
							and values.size == int(point_data_count)
						):
							point_arrays[scalar_name] = [float(v) for v in values.tolist()]
						continue

					if pd_key == 'FIELD' and len(pd_tokens) >= 3:
						try:
							num_fields = int(pd_tokens[2])
						except Exception:
							num_fields = 0

						for _field_idx in range(num_fields):
							field_line = _read_next_nonempty_legacy_vtk_line(file)
							if field_line is None:
								break

							field_tokens = field_line.split()
							if len(field_tokens) < 4:
								continue

							field_name = field_tokens[0]
							try:
								num_components = int(field_tokens[1])
								num_tuples = int(field_tokens[2])
							except Exception:
								continue
							field_dtype = field_tokens[3]

							total_values = int(num_components) * int(num_tuples)
							values = _read_legacy_vtk_binary_values(
								file,
								total_values,
								field_dtype,
								context=f'FIELD {field_name}',
							)

							if (
								total_values > 0
								and num_components == 1
								and point_data_count is not None
								and num_tuples == int(point_data_count)
								and values.size == total_values
							):
								point_arrays[field_name] = [float(v) for v in values.tolist()]
						continue

					# Unknown POINT_DATA array type.  Keep the parser conservative:
					# skip its header line and continue looking for supported arrays.
					continue

				break

		if len(points) == 0:
			raise ValueError('Invalid or unsupported binary VTK file: no POINTS found.')

		edges = _build_edges_from_polyline_cells(polyline_cells)
		if len(edges) == 0 and len(points) >= 2:
			edges = [[i, i + 1] for i in range(len(points) - 1)]

		return points, edges, point_arrays


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



def _xml_find_child(parent, tag_name):
	"""Find a direct child by tag name without caring about XML namespaces."""
	if parent is None:
		return None
	for child in list(parent):
		if str(child.tag).split('}')[-1] == tag_name:
			return child
	return None


def _xml_find_children(parent, tag_name):
	"""Find direct children by tag name without caring about XML namespaces."""
	if parent is None:
		return []
	return [child for child in list(parent) if str(child.tag).split('}')[-1] == tag_name]


def _xml_find_descendant(parent, tag_name):
	"""Find the first descendant by tag name without caring about XML namespaces."""
	if parent is None:
		return None
	for elem in parent.iter():
		if str(elem.tag).split('}')[-1] == tag_name:
			return elem
	return None


def _base64_padded_decode(text):
	"""Decode base64 text, adding padding if needed."""
	text = ''.join(str(text).split())
	if not text:
		return b''
	missing = (-len(text)) % 4
	if missing:
		text += '=' * missing
	return base64.b64decode(text)


def _base64_char_len_for_bytes(num_bytes):
	return 4 * ((int(num_bytes) + 2) // 3)


def _decode_vtp_appended_segment(segment_text, compressor_name=''):
	"""Decode one VTK XML appended DataArray segment.

	The carotid bifurcation dataset stores .vtp arrays as base64-encoded,
	zlib-compressed appended data. VTK writes the compression header and each
	compressed block as separate base64 chunks, so decoding the whole segment in
	one call fails when padding appears inside the segment. This helper decodes
	the VTK zlib-compressed form block by block.
	"""
	segment_text = ''.join(str(segment_text).split())
	if not segment_text:
		return b''

	compressor_name = str(compressor_name or '')
	if 'vtkZLibDataCompressor' not in compressor_name:
		# Fallback for uncompressed appended/base64 data.
		try:
			decoded = _base64_padded_decode(segment_text)
		except Exception:
			return b''
		# VTK XML uncompressed appended arrays often start with a UInt32 byte count.
		if len(decoded) >= 4:
			try:
				byte_count = struct.unpack('<I', decoded[:4])[0]
				if 0 <= byte_count <= len(decoded) - 4:
					return decoded[4:4 + byte_count]
			except Exception:
				pass
		return decoded

	# First decode the compression header. We need the first 16 bytes to get the
	# number of blocks. The base64 representation of 16 bytes is 24 characters.
	first_header = _base64_padded_decode(segment_text[:24])
	if len(first_header) < 12:
		return b''

	num_blocks = int(struct.unpack('<I', first_header[:4])[0])
	if num_blocks <= 0:
		return b''

	# Header layout for vtkZLibDataCompressor:
	#   UInt32 numBlocks
	#   UInt32 blockSize
	#   UInt32 lastBlockSize
	#   UInt32 compressedBlockSize[numBlocks]
	header_num_bytes = int(3 + num_blocks) * 4
	header_num_chars = _base64_char_len_for_bytes(header_num_bytes)
	header = _base64_padded_decode(segment_text[:header_num_chars])
	if len(header) < header_num_bytes:
		raise ValueError('Invalid VTP zlib header: truncated header.')

	header_values = struct.unpack('<' + 'I' * (3 + num_blocks), header[:header_num_bytes])
	compressed_sizes = [int(v) for v in header_values[3:3 + num_blocks]]

	compressed_blob = _base64_padded_decode(segment_text[header_num_chars:])
	out = bytearray()
	cursor = 0
	for compressed_size in compressed_sizes:
		if compressed_size <= 0:
			continue
		block = compressed_blob[cursor:cursor + compressed_size]
		cursor += compressed_size
		if len(block) != compressed_size:
			raise ValueError('Invalid VTP zlib data: compressed block is truncated.')
		out.extend(zlib.decompress(block))

	return bytes(out)


def _vtk_xml_dtype_to_numpy(vtk_type, byte_order='LittleEndian'):
	vtk_type = str(vtk_type or '').strip()
	endian = '<' if str(byte_order).lower().startswith('little') else '>'
	dtype_map = {
		'Float32': 'f4',
		'Float64': 'f8',
		'Double': 'f8',
		'Int8': 'i1',
		'UInt8': 'u1',
		'Int16': 'i2',
		'UInt16': 'u2',
		'Int32': 'i4',
		'UInt32': 'u4',
		'Int64': 'i8',
		'UInt64': 'u8',
	}
	if vtk_type not in dtype_map:
		raise ValueError(f'Unsupported VTP DataArray type: {vtk_type}')
	code = dtype_map[vtk_type]
	if code.endswith('1'):
		return np.dtype(code)
	return np.dtype(endian + code)


def _read_vtp_data_array(data_array_elem, appended_text, offset_to_next, byte_order, compressor_name):
	"""Read one VTP DataArray element into a numpy array."""
	fmt = str(data_array_elem.attrib.get('format', '')).lower()
	vtk_type = data_array_elem.attrib.get('type', '')
	num_components = int(data_array_elem.attrib.get('NumberOfComponents', '1'))
	dtype = _vtk_xml_dtype_to_numpy(vtk_type, byte_order=byte_order)

	if fmt == 'ascii':
		text = ''.join((data_array_elem.text or '').split())
		if not text:
			return np.asarray([], dtype=dtype)
		values = np.fromstring(data_array_elem.text or '', sep=' ', dtype=dtype)
	elif fmt == 'appended':
		offset = int(data_array_elem.attrib.get('offset', '0'))
		end_offset = int(offset_to_next.get(offset, len(appended_text)))
		segment = appended_text[offset:end_offset]
		raw = _decode_vtp_appended_segment(segment, compressor_name=compressor_name)
		values = np.frombuffer(raw, dtype=dtype).copy()
	elif fmt == 'binary':
		text = ''.join((data_array_elem.text or '').split())
		raw = _base64_padded_decode(text)
		values = np.frombuffer(raw, dtype=dtype).copy()
	else:
		raise ValueError(f'Unsupported VTP DataArray format: {fmt}')

	if num_components > 1 and values.size > 0:
		usable = (values.size // num_components) * num_components
		values = values[:usable].reshape((-1, num_components))

	return values


def _read_vtp_polydata(vtp_path):
	"""Read XML VTK PolyData (.vtp) centerline files.

	This supports the carotid bifurcation centerline files whose arrays are stored
	as appended, base64-encoded zlib-compressed DataArray blocks. It preserves
	polyline topology and scalar point arrays such as MaximumInscribedSphereRadius.
	"""
	if not os.path.isfile(vtp_path):
		raise FileNotFoundError(f'Centerline VTP file not found: {vtp_path}')

	with open(vtp_path, 'rb') as file:
		file_bytes = file.read()
	text = file_bytes.decode('utf-8', errors='ignore')

	try:
		root = ET.fromstring(text)
	except Exception as exc:
		raise ValueError(f'Invalid VTP XML file: {vtp_path}') from exc

	byte_order = root.attrib.get('byte_order', 'LittleEndian')
	compressor_name = root.attrib.get('compressor', '')
	piece = _xml_find_descendant(root, 'Piece')
	if piece is None:
		raise ValueError(f'Invalid VTP file: no Piece element found: {vtp_path}')

	appended_elem = _xml_find_descendant(root, 'AppendedData')
	appended_text = ''
	if appended_elem is not None and appended_elem.text is not None:
		appended_text = ''.join(appended_elem.text.split())
		if appended_text.startswith('_'):
			appended_text = appended_text[1:]

	# Build DataArray offset map. In these files offsets are character offsets in
	# the whitespace-stripped appended base64 text after the leading underscore.
	all_data_arrays = [elem for elem in root.iter() if str(elem.tag).split('}')[-1] == 'DataArray']
	offsets = []
	for elem in all_data_arrays:
		if str(elem.attrib.get('format', '')).lower() == 'appended' and 'offset' in elem.attrib:
			offsets.append(int(elem.attrib['offset']))
	offsets = sorted(set(offsets))
	offset_to_next = {}
	for i, off in enumerate(offsets):
		offset_to_next[off] = offsets[i + 1] if i + 1 < len(offsets) else len(appended_text)

	# Points
	points_elem = _xml_find_child(piece, 'Points')
	points_array_elem = _xml_find_child(points_elem, 'DataArray')
	if points_array_elem is None:
		raise ValueError(f'Invalid VTP file: no Points/DataArray found: {vtp_path}')

	points_np = _read_vtp_data_array(
		points_array_elem,
		appended_text=appended_text,
		offset_to_next=offset_to_next,
		byte_order=byte_order,
		compressor_name=compressor_name,
	)
	points_np = np.asarray(points_np, dtype=np.float64)
	if points_np.ndim == 1:
		if points_np.size % 3 != 0:
			raise ValueError(f'Invalid VTP points array length: {points_np.size}')
		points_np = points_np.reshape((-1, 3))
	if points_np.ndim != 2 or points_np.shape[1] != 3:
		raise ValueError(f'Invalid VTP points shape: {points_np.shape}')
	points = points_np.astype(float).tolist()

	# Lines
	polyline_cells = []
	lines_elem = _xml_find_child(piece, 'Lines')
	if lines_elem is not None:
		connectivity_elem = None
		offsets_elem = None
		for elem in _xml_find_children(lines_elem, 'DataArray'):
			name = elem.attrib.get('Name', '')
			if name == 'connectivity':
				connectivity_elem = elem
			elif name == 'offsets':
				offsets_elem = elem

		if connectivity_elem is not None and offsets_elem is not None:
			connectivity = _read_vtp_data_array(
				connectivity_elem,
				appended_text=appended_text,
				offset_to_next=offset_to_next,
				byte_order=byte_order,
				compressor_name=compressor_name,
			).astype(np.int64).reshape(-1)
			line_offsets = _read_vtp_data_array(
				offsets_elem,
				appended_text=appended_text,
				offset_to_next=offset_to_next,
				byte_order=byte_order,
				compressor_name=compressor_name,
			).astype(np.int64).reshape(-1)

			start = 0
			for end in line_offsets.tolist():
				end = int(end)
				ids = [int(v) for v in connectivity[start:end].tolist()]
				if len(ids) >= 2:
					polyline_cells.append(ids)
				start = end

	edges = _build_edges_from_polyline_cells(polyline_cells)
	if len(edges) == 0 and len(points) >= 2:
		edges = [[i, i + 1] for i in range(len(points) - 1)]

	# Point data arrays
	point_arrays = {}
	point_data_elem = _xml_find_child(piece, 'PointData')
	for elem in _xml_find_children(point_data_elem, 'DataArray'):
		name = elem.attrib.get('Name', '') or 'unnamed'
		try:
			arr = _read_vtp_data_array(
				elem,
				appended_text=appended_text,
				offset_to_next=offset_to_next,
				byte_order=byte_order,
				compressor_name=compressor_name,
			)
		except Exception as exc:
			print(f'[CENTERLINE_LOAD] warning: failed to read VTP point array {name}: {exc}')
			continue

		arr = np.asarray(arr)
		if arr.ndim == 1:
			point_arrays[name] = [float(v) for v in arr.tolist()]
		elif arr.ndim == 2 and arr.shape[1] == 1:
			point_arrays[name] = [float(v[0]) for v in arr.tolist()]

	return points, edges, point_arrays


def _read_centerline_polydata(centerline_path):
	"""Read either legacy .vtk or XML .vtp centerline PolyData."""
	ext = os.path.splitext(str(centerline_path))[1].lower()
	if ext == '.vtp':
		return _read_vtp_polydata(centerline_path)
	return _read_legacy_vtk_polydata(centerline_path)


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
	points_raw, edges, point_arrays = _read_centerline_polydata(vtk_path)
	print(f"[CENTERLINE_LOAD] vtk_path = {vtk_path}")
	print(f"[CENTERLINE_LOAD] point_arrays keys = {sorted((point_arrays or {}).keys())}")
	points_sim = _transform_points_to_sim(
		points=points_raw,
		T_env_sim=T_env_sim,
		point_frame=point_frame,
		scale=scale,
		offset_sim=offset_sim)

	def _safe_stats(values):
		cleaned = []
		for value in values:
			try:
				v = float(value)
			except Exception:
				continue
			if math.isfinite(v):
				cleaned.append(v)
		if len(cleaned) == 0:
			return None
		vmin = min(cleaned)
		vmax = max(cleaned)
		vmean = float(sum(cleaned) / len(cleaned))
		return vmin, vmean, vmax

	radius_raw = None
	selected_radius_name = None
	for radius_name in (
		'Radius',
		'MaximumInscribedSphereRadius',
		'MaximumInscribedSphereRadius_',
		'InscribedSphereRadius',
	):
		candidate = (point_arrays or {}).get(radius_name)
		if candidate is None:
			continue
		if len(candidate) != len(points_raw):
			print(
				f"[CENTERLINE_LOAD] {radius_name} length mismatch: "
				f"{len(candidate)} vs {len(points_raw)}"
			)
			continue
		radius_raw = candidate
		selected_radius_name = radius_name
		break

	if radius_raw is None:
		print("[CENTERLINE_LOAD] No Radius-like array found in VTK.")
	else:
		print(f"[CENTERLINE_LOAD] selected radius array = {selected_radius_name}")
		raw_stats = _safe_stats(radius_raw)
		if raw_stats is not None:
			print(
				"[CENTERLINE_LOAD] radius_raw min/mean/max =",
				float(raw_stats[0]),
				float(raw_stats[1]),
				float(raw_stats[2]),
			)
		else:
			print("[CENTERLINE_LOAD] radius_raw min/mean/max = nan nan nan")

	radius_sim = None
	if radius_raw is not None and len(radius_raw) == len(points_raw):
		try:
			radius_sim = [float(r) * float(scale) for r in radius_raw]
		except Exception:
			radius_sim = None

	if radius_sim is not None:
		sim_stats = _safe_stats(radius_sim)
		if sim_stats is not None:
			print(
				"[CENTERLINE_LOAD] radius_sim min/mean/max mm =",
				float(sim_stats[0]) * 1000.0,
				float(sim_stats[1]) * 1000.0,
				float(sim_stats[2]) * 1000.0,
			)
		else:
			print("[CENTERLINE_LOAD] radius_sim min/mean/max mm = nan nan nan")

	return CenterlineData(
		points_raw=points_raw,
		points_sim=points_sim,
		edges=edges,
		radius_raw=radius_raw,
		radius_sim=radius_sim,
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