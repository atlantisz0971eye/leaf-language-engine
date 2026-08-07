from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


LOGICAL_WIDTH = 1000.0
LOGICAL_HEIGHT = 760.0
LIGHT_GREEN = "#6b997c"
DARK_GREEN = "#286947"
LOCATOR_GREEN = "#3f7958"
SCAFFOLD_GREEN = "#326c4c"


def _hex_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


def _on_white(value: str, opacity: float = 1.0) -> tuple[int, int, int]:
    color = _hex_bgr(value)
    alpha = max(0.0, min(1.0, float(opacity)))
    return tuple(round(255 * (1.0 - alpha) + channel * alpha) for channel in color)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(_distance(a, b) for a, b in zip(points, points[1:]))


def _point_on_polyline(
    points: list[tuple[float, float]], fraction: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not points:
        return (0.0, 0.0), (1.0, 0.0)
    total = max(1.0, _polyline_length(points))
    target = max(0.0, min(1.0, fraction)) * total
    for index in range(1, len(points)):
        start, end = points[index - 1], points[index]
        section = _distance(start, end)
        if target <= section or index == len(points) - 1:
            ratio = target / section if section else 0.0
            tangent_length = max(1.0, section)
            return (
                (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                ),
                (
                    (end[0] - start[0]) / tangent_length,
                    (end[1] - start[1]) / tangent_length,
                ),
            )
        target -= section
    return points[-1], (1.0, 0.0)


def _polyline_slice(
    points: list[tuple[float, float]], start_fraction: float, end_fraction: float
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    total = max(1.0, _polyline_length(points))
    cumulative = [0.0]
    for start, end in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + _distance(start, end))
    start_fraction = max(0.0, min(1.0, start_fraction))
    end_fraction = max(start_fraction, min(1.0, end_fraction))
    output = [_point_on_polyline(points, start_fraction)[0]]
    for index in range(1, len(points) - 1):
        fraction = cumulative[index] / total
        if start_fraction < fraction < end_fraction:
            output.append(points[index])
    output.append(_point_on_polyline(points, end_fraction)[0])
    return output


def _protocol_boundaries(points: list[tuple[float, float]], count: int) -> list[float]:
    if count <= 0:
        return [0.0, 1.0]
    total = max(1.0, _polyline_length(points))
    cumulative = [0.0]
    for start, end in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + _distance(start, end))
    natural = [value / total for value in cumulative[1:-1]]
    boundaries = [0.0]
    for index in range(1, count):
        target = index / count
        tolerance = 0.28 / count
        nearest = min(natural, key=lambda value: abs(value - target), default=target)
        previous = boundaries[-1]
        boundaries.append(
            nearest
            if abs(nearest - target) <= tolerance and nearest > previous + 0.35 / count
            else target
        )
    boundaries.append(1.0)
    return boundaries


def _graph_transform(graph: dict[str, Any]) -> Callable[[tuple[float, float]], tuple[float, float]]:
    nodes = {int(node["id"]): node for node in graph["nodes"]}
    main_ids = [int(item) for item in graph["main_path"]["node_ids"]]
    if len(main_ids) < 2:
        raise ValueError("The specimen graph has no usable primary vein.")
    start = nodes[main_ids[0]]
    end = nodes[main_ids[-1]]
    angle = -math.atan2(float(end["y"]) - float(start["y"]), float(end["x"]) - float(start["x"]))
    cosine, sine = math.cos(angle), math.sin(angle)

    def rotate(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return x * cosine - y * sine, x * sine + y * cosine

    rotated = [rotate((float(node["x"]), float(node["y"]))) for node in graph["nodes"]]
    xs = [point[0] for point in rotated]
    ys = [point[1] for point in rotated]
    minimum_x, maximum_x = min(xs), max(xs)
    minimum_y, maximum_y = min(ys), max(ys)
    scale = min(900.0 / max(1.0, maximum_x - minimum_x), 620.0 / max(1.0, maximum_y - minimum_y))
    offset_x = 500.0 - (minimum_x + maximum_x) * scale / 2.0
    offset_y = 380.0 - (minimum_y + maximum_y) * scale / 2.0

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        rotated_x, rotated_y = rotate(point)
        return rotated_x * scale + offset_x, rotated_y * scale + offset_y

    return transform


def _pixel_points(
    points: list[tuple[float, float]], output_scale: float
) -> np.ndarray:
    return np.asarray(
        [[round(x * output_scale), round(y * output_scale)] for x, y in points],
        dtype=np.int32,
    ).reshape((-1, 1, 2))


def _draw_path(
    canvas: np.ndarray,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    logical_width: float,
    output_scale: float,
) -> None:
    if len(points) < 2:
        return
    cv2.polylines(
        canvas,
        [_pixel_points(points, output_scale)],
        False,
        color,
        max(1, round(logical_width * output_scale)),
        cv2.LINE_AA,
    )


def _payload_states(packet: dict[str, Any]) -> dict[int, int]:
    payload_symbols = [
        int(symbol)
        for block in packet["payload_codec"]["blocks"]
        for symbol in block["symbols"]
    ]
    plan = packet["render_plan"]["carrier_plan"]
    branches = {int(branch["branch_id"]): branch for branch in plan.get("branches", [])}
    states: dict[int, int] = {}
    for assignment in packet["render_plan"].get("branch_assignments", []):
        branch = branches.get(int(assignment["branch_id"]))
        if branch is None:
            continue
        carriers = branch.get("carrier_edges", [])
        start = int(assignment["start_symbol"])
        count = min(int(assignment["symbol_count"]), len(carriers))
        for index in range(count):
            states[int(carriers[index]["edge_id"])] = payload_symbols[start + index]
    return states


def render_encoding_png(packet: dict[str, Any]) -> bytes:
    graph = packet.get("surface_graph")
    if not graph or not graph.get("edges") or not graph.get("main_path", {}).get("node_ids"):
        raise ValueError("A prepared specimen graph is required for PNG export.")
    resolution = max(2048, min(16384, int(packet["summary"]["target_resolution"])))
    output_scale = resolution / LOGICAL_WIDTH
    output_height = round(LOGICAL_HEIGHT * output_scale)
    canvas = np.full((output_height, resolution, 3), 255, dtype=np.uint8)
    transform = _graph_transform(graph)
    payload_states = _payload_states(packet)

    for edge in graph["edges"]:
        points = [transform((float(x), float(y))) for x, y in edge["path"]]
        state = payload_states.get(int(edge["id"]))
        if state is None:
            order = int(edge.get("order", 6))
            logical_width = 2.2 if order <= 1 else 1.25 if order == 2 else 0.72
            opacity = 0.34 if order <= 1 else 0.32 if order == 2 else 0.24
            color = _on_white(SCAFFOLD_GREEN, opacity)
        else:
            logical_width = 1.85 if state >= 2 else 0.8
            opacity = (0.84, 0.98, 0.90, 1.0)[state]
            color = _on_white(DARK_GREEN if state % 2 else LIGHT_GREEN, opacity)
        _draw_path(canvas, points, color, logical_width, output_scale)

    nodes = {int(node["id"]): node for node in graph["nodes"]}
    main_points = [
        transform((float(nodes[int(node_id)]["x"]), float(nodes[int(node_id)]["y"])))
        for node_id in graph["main_path"]["node_ids"]
    ]
    protocol_symbols = [int(symbol) for symbol in packet["protocol_codec"]["symbols"]]
    boundaries = _protocol_boundaries(main_points, len(protocol_symbols))
    for index, state in enumerate(protocol_symbols):
        midpoint = (boundaries[index] + boundaries[index + 1]) / 2.0
        natural_width = 6.4 - 4.35 * midpoint
        logical_width = natural_width * (1.14 if state >= 2 else 0.86)
        color = _hex_bgr(DARK_GREEN if state % 2 else LIGHT_GREEN)
        segment = _polyline_slice(main_points, boundaries[index], boundaries[index + 1])
        _draw_path(canvas, segment, color, logical_width, output_scale)

    locator_color = _hex_bgr(LOCATOR_GREEN)
    start_point, _ = _point_on_polyline(main_points, 0.0)
    cv2.circle(
        canvas,
        (round(start_point[0] * output_scale), round(start_point[1] * output_scale)),
        max(2, round(6.0 * output_scale)),
        (255, 255, 255),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(
        canvas,
        (round(start_point[0] * output_scale), round(start_point[1] * output_scale)),
        max(2, round(6.0 * output_scale)),
        locator_color,
        thickness=max(1, round(1.5 * output_scale)),
        lineType=cv2.LINE_AA,
    )

    end_point, tangent = _point_on_polyline(main_points, 1.0)
    normal = (-tangent[1], tangent[0])
    for along, normal_extent in ((-3.0, 5.0), (2.0, 4.0)):
        center = (
            end_point[0] + tangent[0] * along,
            end_point[1] + tangent[1] * along,
        )
        line = [
            (
                center[0] - normal[0] * normal_extent,
                center[1] - normal[1] * normal_extent,
            ),
            (
                center[0] + normal[0] * normal_extent,
                center[1] + normal[1] * normal_extent,
            ),
        ]
        _draw_path(canvas, line, locator_color, 1.5, output_scale)

    ok, encoded = cv2.imencode(
        ".png",
        canvas,
        [cv2.IMWRITE_PNG_COMPRESSION, 4],
    )
    if not ok:
        raise RuntimeError("Could not encode the generated leaf PNG.")
    return encoded.tobytes()


def encoding_export_filename(packet: dict[str, Any]) -> str:
    sample = Path(str(packet.get("style_profile", {}).get("sample") or "generic-leaf")).stem
    sample = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample).strip("._") or "leaf"
    checksum = int(packet.get("payload_codec", {}).get("payload_crc16", 0))
    resolution = int(packet.get("summary", {}).get("target_resolution", 8192))
    return f"{sample}_encoding_{resolution}px_{checksum:04X}.png"
