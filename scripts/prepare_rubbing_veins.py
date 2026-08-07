#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _kernel(size: int) -> np.ndarray:
    size = max(3, int(size) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _remove_small_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    output = np.zeros_like(mask)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum:
            output[labels == index] = 255
    return output


def _normalize_inside(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sample = values[mask > 0]
    positive = sample[sample > 0]
    if not positive.size:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(positive, [12, 99.3])
    normalized = np.clip((values - low) / max(1e-6, high - low), 0.0, 1.0)
    normalized[mask == 0] = 0.0
    return normalized.astype(np.float32)


def _hessian_vesselness(
    image: np.ndarray,
    mask: np.ndarray,
    scales: tuple[float, ...] = (2.0, 3.2, 4.8, 7.0, 9.0),
) -> np.ndarray:
    source = image.astype(np.float32)
    maximum = np.zeros_like(source, dtype=np.float32)
    beta = 0.52
    for sigma in scales:
        blurred = cv2.GaussianBlur(source, (0, 0), sigma)
        dxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3) * sigma * sigma
        dyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3) * sigma * sigma
        dxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3) * sigma * sigma
        discriminant = np.sqrt(np.maximum(0.0, (dxx - dyy) ** 2 + 4.0 * dxy**2))
        lambda_a = 0.5 * (dxx + dyy + discriminant)
        lambda_b = 0.5 * (dxx + dyy - discriminant)
        swap = np.abs(lambda_a) > np.abs(lambda_b)
        lambda_1 = np.where(swap, lambda_b, lambda_a)
        lambda_2 = np.where(swap, lambda_a, lambda_b)
        ratio = np.abs(lambda_1) / (np.abs(lambda_2) + 1e-6)
        structure = np.sqrt(lambda_1**2 + lambda_2**2)
        structure_values = structure[mask > 0]
        c = max(0.015, float(np.percentile(structure_values, 88)) * 0.55) if structure_values.size else 0.1
        response = np.exp(-(ratio**2) / (2.0 * beta * beta))
        response *= 1.0 - np.exp(-(structure**2) / (2.0 * c * c))
        response[lambda_2 >= 0] = 0.0  # Bright ridges have negative transverse curvature.
        response[mask == 0] = 0.0
        maximum = np.maximum(maximum, response)
    return maximum


def add_axis_support(
    vein_binary: np.ndarray,
    leaf: np.ndarray,
    maximum_connectors: int = 34,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    points_yx = np.argwhere(leaf > 0)
    if len(points_yx) < 32:
        empty = np.zeros_like(vein_binary)
        return vein_binary, empty, empty, {"applied": False}
    points_xy = points_yx[:, ::-1].astype(np.float32)
    center = points_xy.mean(axis=0)
    covariance = np.cov(points_xy, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))].astype(np.float32)
    axis /= max(1e-6, float(np.linalg.norm(axis)))
    normal = np.asarray((-axis[1], axis[0]), dtype=np.float32)
    projections = (points_xy - center) @ axis
    cross_projections = (points_xy - center) @ normal
    start_projection, end_projection = np.percentile(projections, [3.0, 97.0])
    minor_span = float(np.percentile(cross_projections, 97) - np.percentile(cross_projections, 3))
    start = center + axis * start_projection
    end = center + axis * end_projection

    axis_mask = np.zeros_like(vein_binary)
    cv2.line(
        axis_mask,
        (round(float(start[0])), round(float(start[1]))),
        (round(float(end[0])), round(float(end[1]))),
        255,
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    axis_mask[leaf == 0] = 0
    supported = cv2.bitwise_or(vein_binary, axis_mask)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(vein_binary, 8)
    connector_candidates = []
    maximum_distance = max(12.0, minor_span * 0.16)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 14 or max(width, height) < 10:
            continue
        component = np.argwhere(labels == label)
        component_xy = component[:, ::-1].astype(np.float32)
        relative = component_xy - center
        distances = np.abs(relative @ normal)
        closest_index = int(np.argmin(distances))
        distance = float(distances[closest_index])
        if distance > maximum_distance:
            continue
        source = component_xy[closest_index]
        axis_projection = float((source - center) @ axis)
        target = center + axis * axis_projection
        connector_candidates.append(
            (
                area * max(width, height) / max(1.0, distance + 2.0),
                source,
                target,
                distance,
            )
        )
    connector_candidates.sort(key=lambda item: item[0], reverse=True)
    connector_mask = np.zeros_like(vein_binary)
    connector_count = 0
    for _, source, target, distance in connector_candidates[:maximum_connectors]:
        line_probe = np.zeros_like(vein_binary)
        cv2.line(
            line_probe,
            (round(float(source[0])), round(float(source[1]))),
            (round(float(target[0])), round(float(target[1]))),
            255,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        probe_pixels = line_probe > 0
        if not np.any(probe_pixels):
            continue
        inside_ratio = float(np.mean(leaf[probe_pixels] > 0))
        if inside_ratio < 0.92:
            continue
        connector_mask[probe_pixels] = 255
        connector_count += 1
    support_mask = cv2.bitwise_or(axis_mask, connector_mask)
    supported = cv2.bitwise_or(supported, connector_mask)
    supported[leaf == 0] = 0
    return supported, support_mask, axis_mask, {
        "applied": True,
        "axis": [round(float(axis[0]), 6), round(float(axis[1]), 6)],
        "minor_span": round(minor_span, 3),
        "maximum_connection_distance": round(maximum_distance, 3),
        "connector_count": connector_count,
    }


def suppress_textile_and_extract(
    clean: np.ndarray,
    silhouette: np.ndarray,
    ink_mask: np.ndarray,
    maximum_side: int = 1100,
) -> dict[str, np.ndarray | float]:
    height, width = silhouette.shape
    scale = min(1.0, maximum_side / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    work = cv2.resize(clean, size, interpolation=cv2.INTER_AREA)
    leaf = cv2.resize(silhouette, size, interpolation=cv2.INTER_NEAREST)
    ink = cv2.resize(ink_mask, size, interpolation=cv2.INTER_AREA)
    leaf = np.where(leaf > 127, 255, 0).astype(np.uint8)
    core = cv2.erode(leaf, _kernel(9), iterations=1)

    # Closing at this scale fills the narrow textile grid while retaining the
    # wider uninked channels made by primary and secondary leaf veins.
    ink_closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, _kernel(7))
    ink_closed = cv2.GaussianBlur(ink_closed, (0, 0), 1.45).astype(np.float32) / 255.0
    neighbourhood_ink = cv2.GaussianBlur(ink_closed, (0, 0), 11.0)
    gap_from_ink = np.clip((1.0 - ink_closed) * neighbourhood_ink, 0.0, 1.0)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), 1.25)
    local_bright = np.zeros_like(gray, dtype=np.float32)
    for sigma, weight in ((5.0, 0.48), (11.0, 0.34), (22.0, 0.18)):
        local_bright += np.maximum(gray - cv2.GaussianBlur(gray, (0, 0), sigma), 0.0) * weight
    local_bright = _normalize_inside(local_bright, core)

    gap_score = np.clip(0.58 * gap_from_ink + 0.42 * local_bright, 0.0, 1.0)
    gap_score[core == 0] = 0.0
    gap_score = cv2.GaussianBlur(gap_score, (0, 0), 0.75)
    vesselness = _hessian_vesselness(gap_score, core)
    vesselness *= np.clip(cv2.GaussianBlur(neighbourhood_ink, (0, 0), 4.0) * 1.3, 0.0, 1.0)
    vesselness = _normalize_inside(vesselness, core)

    positive = vesselness[core > 0]
    positive = positive[positive > 0]
    threshold = float(np.percentile(positive, 82.0)) if positive.size else 1.0
    threshold = max(0.14, min(0.52, threshold))
    binary = np.where((vesselness >= threshold) & (core > 0), 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _kernel(3))
    binary = _remove_small_components(binary, max(14, round(binary.size * 0.000022)))

    # A restrained close joins short print interruptions without turning broad
    # missing-ink patches into fabricated veins.
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _kernel(5))
    connected = _remove_small_components(connected, max(18, round(binary.size * 0.000035)))
    connected, support_mask, main_axis_mask, support_metadata = add_axis_support(connected, leaf)
    skeleton = cv2.ximgproc.thinning(connected, cv2.ximgproc.THINNING_ZHANGSUEN)
    skeleton = _remove_small_components(skeleton, max(7, round(binary.size * 0.000012)))

    return {
        "scale": scale,
        "leaf": leaf,
        "fabric_suppressed": np.clip(gap_score * 255.0, 0, 255).astype(np.uint8),
        "vesselness": np.clip(vesselness * 255.0, 0, 255).astype(np.uint8),
        "vein_binary": connected,
        "vein_skeleton": skeleton,
        "support_mask": support_mask,
        "main_axis_mask": main_axis_mask,
        "support_metadata": support_metadata,
        "threshold": threshold,
    }


def synthetic_analysis_input(
    leaf: np.ndarray,
    vein_binary: np.ndarray,
    main_axis_mask: np.ndarray,
) -> np.ndarray:
    height, width = leaf.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = (255, 255, 255)
    active = leaf > 0
    rgba[active, :3] = (214, 231, 217)
    veins = cv2.dilate(vein_binary, _kernel(3)) > 0
    rgba[veins, :3] = (69, 105, 76)
    main_axis = cv2.dilate(main_axis_mask, _kernel(7)) > 0
    rgba[main_axis, :3] = (38, 67, 45)
    rgba[:, :, 3] = leaf
    return rgba


def align_graph_main_path(
    graph: dict[str, Any],
    main_axis_mask: np.ndarray,
    axis_vector: list[float] | tuple[float, float] | None,
) -> dict[str, Any]:
    nodes = {int(node["id"]): node for node in graph.get("nodes", [])}
    edges = {int(edge["id"]): edge for edge in graph.get("edges", [])}
    if not nodes or not edges or not np.any(main_axis_mask > 0):
        return {"applied": False, "reason": "missing graph or axis hint"}
    axis = np.asarray(axis_vector or (1.0, 0.0), dtype=np.float64)
    axis /= max(1e-9, float(np.linalg.norm(axis)))
    hint_points_yx = np.argwhere(main_axis_mask > 0)
    hint_points_xy = hint_points_yx[:, ::-1].astype(np.float64)
    center = hint_points_xy.mean(axis=0)
    hint_projections = (hint_points_xy - center) @ axis
    start_hint = center + axis * float(np.percentile(hint_projections, 1.0))
    end_hint = center + axis * float(np.percentile(hint_projections, 99.0))
    distance_to_hint = cv2.distanceTransform(
        np.where(main_axis_mask > 0, 0, 255).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )

    def node_point(node_id: int) -> np.ndarray:
        node = nodes[node_id]
        return np.asarray((float(node["x"]), float(node["y"])), dtype=np.float64)

    def endpoint_node(hint: np.ndarray) -> int:
        return min(
            nodes,
            key=lambda node_id: float(np.linalg.norm(node_point(node_id) - hint))
            + 2.2
            * float(
                distance_to_hint[
                    max(0, min(distance_to_hint.shape[0] - 1, round(float(nodes[node_id]["y"])))),
                    max(0, min(distance_to_hint.shape[1] - 1, round(float(nodes[node_id]["x"])))),
                ]
            ),
        )

    start_node = endpoint_node(start_hint)
    end_node = endpoint_node(end_hint)
    adjacency: dict[int, list[tuple[int, int]]] = {node_id: [] for node_id in nodes}
    for edge_id, edge in edges.items():
        source, target = int(edge["source"]), int(edge["target"])
        adjacency[source].append((target, edge_id))
        adjacency[target].append((source, edge_id))

    # Pick both main endpoints from the same graph component. Faint rubbings
    # can lose a few axis pixels during the legacy re-thresholding pass; this
    # avoids selecting two visually correct but topologically disconnected tips.
    components: list[list[int]] = []
    unseen = set(nodes)
    while unseen:
        seed = unseen.pop()
        component = [seed]
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbour, _ in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    component.append(neighbour)
                    stack.append(neighbour)
        components.append(component)

    def hint_distance(node_id: int) -> float:
        node = nodes[node_id]
        return float(
            distance_to_hint[
                max(0, min(distance_to_hint.shape[0] - 1, round(float(node["y"])))),
                max(0, min(distance_to_hint.shape[1] - 1, round(float(node["x"])))),
            ]
        )

    def projection(node_id: int) -> float:
        return float((node_point(node_id) - center) @ axis)

    viable_components = [component for component in components if len(component) >= 2]
    if viable_components:
        def component_score(component: list[int]) -> float:
            ordered = sorted(component, key=hint_distance)
            near = [node_id for node_id in ordered if hint_distance(node_id) <= 18.0]
            selected = near if len(near) >= 2 else ordered[: min(14, len(ordered))]
            span = max(projection(node_id) for node_id in selected) - min(
                projection(node_id) for node_id in selected
            )
            median_distance = float(np.median([hint_distance(node_id) for node_id in selected]))
            return span * math.sqrt(len(selected)) / (1.0 + median_distance / 5.0)

        main_component = max(viable_components, key=component_score)
        ordered = sorted(main_component, key=hint_distance)
        near = [node_id for node_id in ordered if hint_distance(node_id) <= 18.0]
        selected = near if len(near) >= 2 else ordered[: min(18, len(ordered))]
        start_node = min(selected, key=projection)
        end_node = max(selected, key=projection)

    def edge_cost(edge: dict[str, Any]) -> float:
        path = edge.get("path") or []
        distances = []
        for x, y in path:
            px = max(0, min(distance_to_hint.shape[1] - 1, round(float(x))))
            py = max(0, min(distance_to_hint.shape[0] - 1, round(float(y))))
            distances.append(float(distance_to_hint[py, px]))
        mean_distance = float(np.mean(distances)) if distances else 30.0
        source = node_point(int(edge["source"]))
        target = node_point(int(edge["target"]))
        direction = target - source
        direction /= max(1e-9, float(np.linalg.norm(direction)))
        alignment = abs(float(np.dot(direction, axis)))
        hint_factor = 0.055 + min(1.8, mean_distance / 10.0)
        direction_factor = 0.58 + 0.42 * (1.0 - alignment)
        return (
            float(edge.get("length_px", 1.0))
            * hint_factor
            * direction_factor
            / max(1.0, float(edge.get("mean_width_px", 1.0)) ** 0.72)
        )

    distances = {node_id: float("inf") for node_id in nodes}
    previous: dict[int, tuple[int, int]] = {}
    distances[start_node] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start_node)]
    while queue:
        distance, node_id = heapq.heappop(queue)
        if distance != distances[node_id]:
            continue
        if node_id == end_node:
            break
        for neighbour, edge_id in adjacency[node_id]:
            candidate = distance + edge_cost(edges[edge_id])
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                previous[neighbour] = (node_id, edge_id)
                heapq.heappush(queue, (candidate, neighbour))
    if end_node not in previous:
        return {"applied": False, "reason": "axis endpoints are disconnected"}

    main_nodes = [end_node]
    main_edges = []
    cursor = end_node
    while cursor != start_node and cursor in previous:
        cursor, edge_id = previous[cursor]
        main_edges.append(edge_id)
        main_nodes.append(cursor)
    main_nodes.reverse()
    main_edges.reverse()
    main_edge_set = set(main_edges)
    main_node_set = set(main_nodes)
    for edge in edges.values():
        edge["order"] = 1 if int(edge["id"]) in main_edge_set else None
    breadth_queue: list[tuple[int, int]] = []
    for node_id in main_nodes:
        for neighbour, edge_id in adjacency[node_id]:
            if edge_id not in main_edge_set and edges[edge_id]["order"] is None:
                edges[edge_id]["order"] = 2
                breadth_queue.append((neighbour, 2))
    visited = set(main_edges)
    cursor_index = 0
    while cursor_index < len(breadth_queue):
        node_id, order = breadth_queue[cursor_index]
        cursor_index += 1
        for neighbour, edge_id in adjacency[node_id]:
            if edge_id in visited or edges[edge_id]["order"] is not None:
                continue
            edges[edge_id]["order"] = min(6, order + 1)
            visited.add(edge_id)
            breadth_queue.append((neighbour, min(6, order + 1)))
    for edge in edges.values():
        if edge["order"] is None:
            edge["order"] = 6
    for node in nodes.values():
        node["role"] = "main_vein" if int(node["id"]) in main_node_set else node.get("type", "endpoint")
    anchors = []
    for position, node_id in enumerate(main_nodes):
        branch_edges = [
            edge_id for _, edge_id in adjacency[node_id] if edge_id not in main_edge_set
        ]
        if branch_edges:
            anchors.append(
                {
                    "node_id": node_id,
                    "main_position": round(position / max(1, len(main_nodes) - 1), 4),
                    "branch_edge_ids": branch_edges,
                }
            )
    graph["main_path"] = {"node_ids": main_nodes, "edge_ids": main_edges}
    graph["secondary_anchors"] = anchors
    graph.setdefault("stats", {}).update(
        {
            "main_edge_count": len(main_edges),
            "secondary_anchor_count": len(anchors),
        }
    )
    return {
        "applied": True,
        "start_node": start_node,
        "end_node": end_node,
        "main_edge_count": len(main_edges),
        "secondary_anchor_count": len(anchors),
    }


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32)[:, :, None] / 255.0
        image = (image[:, :, :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    ratio = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def review_card(
    stem: str,
    clean: np.ndarray,
    fabric: np.ndarray,
    analysis_input: np.ndarray,
    overlay: np.ndarray | None,
    status: str,
    graph_stats: dict[str, Any],
) -> np.ndarray:
    card = np.full((300, 920, 3), 255, dtype=np.uint8)
    images = (clean, fabric, analysis_input, overlay if overlay is not None else analysis_input)
    for index, image in enumerate(images):
        card[48:286, 8 + index * 228 : 220 + index * 228] = _fit(image, 212, 238)
    color = (48, 128, 55) if status == "graph-ready" else (40, 40, 205)
    label = (
        f"{stem}  {status.upper()}  edges {graph_stats.get('edge_count', 0)}  "
        f"main {graph_stats.get('main_edge_count', 0)}  branches {graph_stats.get('secondary_anchor_count', 0)}"
    )
    cv2.putText(card, label, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
    return card


def contact_sheet(cards: list[np.ndarray], columns: int = 2) -> np.ndarray:
    if not cards:
        return np.full((200, 900, 3), 255, dtype=np.uint8)
    rows = math.ceil(len(cards) / columns)
    height, width = cards[0].shape[:2]
    sheet = np.full((rows * height, columns * width, 3), 242, dtype=np.uint8)
    for index, card in enumerate(cards):
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = card
    return sheet


def load_analyzer(app_directory: Path):
    sys.path.insert(0, str(app_directory.resolve()))
    from leaf_analysis import AnalysisSettings, analyze_image, export_result
    from document_engine import _branch_carrier_plan

    return AnalysisSettings, analyze_image, export_result, _branch_carrier_plan


def process_folder(processed_root: Path, app_directory: Path) -> list[dict[str, Any]]:
    AnalysisSettings, analyze_image, export_result, branch_carrier_plan = load_analyzer(app_directory)
    results: list[dict[str, Any]] = []
    cards: list[np.ndarray] = []
    for directory in sorted(path for path in processed_root.iterdir() if path.is_dir()):
        clean = cv2.imread(str(directory / "rubbing_clean.png"), cv2.IMREAD_COLOR)
        silhouette = cv2.imread(str(directory / "silhouette.png"), cv2.IMREAD_GRAYSCALE)
        ink_mask = cv2.imread(str(directory / "ink_mask.png"), cv2.IMREAD_GRAYSCALE)
        if clean is None or silhouette is None or ink_mask is None:
            continue
        prepared = suppress_textile_and_extract(clean, silhouette, ink_mask)
        analysis_input = synthetic_analysis_input(
            prepared["leaf"], prepared["vein_binary"], prepared["main_axis_mask"]
        )
        _write(directory / "fabric_suppressed.png", prepared["fabric_suppressed"])
        _write(directory / "vein_response.png", prepared["vesselness"])
        _write(directory / "vein_clean.png", prepared["vein_binary"])
        _write(directory / "vein_skeleton.png", prepared["vein_skeleton"])
        _write(directory / "axis_support.png", prepared["support_mask"])
        _write(directory / "main_axis_support.png", prepared["main_axis_mask"])
        _write(directory / "analysis_input.png", analysis_input)

        analysis = analyze_image(
            directory / "analysis_input.png",
            AnalysisSettings(max_side=1600, sensitivity=70.0, contrast_scale=8.0, min_component=7),
        )
        main_alignment = align_graph_main_path(
            analysis["graph"],
            prepared["main_axis_mask"],
            prepared["support_metadata"].get("axis"),
        )
        analysis_directory = export_result(analysis, directory / "analysis")
        graph_stats = analysis["graph"].get("stats", {})
        carrier_plan = branch_carrier_plan(analysis["graph"], 8192, 14.0)
        main_capacity = int(carrier_plan.get("main_capacity", 0))
        payload_capacity = int(carrier_plan.get("payload_capacity", 0))
        graph_ready = bool(
            int(graph_stats.get("edge_count", 0)) >= 8
            and main_capacity >= 72
            and payload_capacity >= 80
            and int(graph_stats.get("edge_count", 0)) <= 3500
        )
        status = "graph-ready" if graph_ready else "review"
        record = {
            "sample": directory.name,
            "status": status,
            "extraction_scale": round(float(prepared["scale"]), 6),
            "vessel_threshold": round(float(prepared["threshold"]), 5),
            "graph_stats": graph_stats,
            "axis_support": prepared["support_metadata"],
            "main_path_alignment": main_alignment,
            "capacity_8192": {
                "protocol_cells": main_capacity,
                "payload_edges": payload_capacity,
                "branch_count": len(carrier_plan.get("branches", [])),
            },
            "analysis_metrics": analysis["metrics"],
            "analysis_directory": str(analysis_directory.relative_to(processed_root)),
            "outputs": {
                "fabric_suppressed": "fabric_suppressed.png",
                "vein_response": "vein_response.png",
                "vein_clean": "vein_clean.png",
                "vein_skeleton": "vein_skeleton.png",
                "axis_support": "axis_support.png",
                "main_axis_support": "main_axis_support.png",
                "analysis_input": "analysis_input.png",
            },
        }
        (directory / "vein_metadata.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results.append(record)
        cards.append(
            review_card(
                directory.name,
                clean,
                prepared["fabric_suppressed"],
                analysis_input,
                analysis["images"].get("overlay"),
                status,
                graph_stats,
            )
        )
        print(
            f"{directory.name}: {status} · edges {graph_stats.get('edge_count', 0)} · "
            f"main {graph_stats.get('main_edge_count', 0)} · "
            f"branches {graph_stats.get('secondary_anchor_count', 0)}"
        )

    (processed_root / "graph_manifest.json").write_text(
        json.dumps({"version": "rubbing-vein-preparation-0.1", "items": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (processed_root / "graph_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample", "status", "edges", "main_edges", "branch_anchors", "junctions", "endpoints"))
        for item in results:
            stats = item["graph_stats"]
            metrics = item["analysis_metrics"]
            writer.writerow(
                (
                    item["sample"],
                    item["status"],
                    stats.get("edge_count", 0),
                    stats.get("main_edge_count", 0),
                    stats.get("secondary_anchor_count", 0),
                    metrics.get("junction_count", 0),
                    metrics.get("endpoint_count", 0),
                )
            )
    _write(processed_root / "vein_review_sheet.png", contact_sheet(cards))
    return results


def main() -> None:
    script_app_directory = Path(__file__).resolve().parents[1]
    project_directory = script_app_directory.parent
    parser = argparse.ArgumentParser(
        description="Suppress textile texture, reconstruct rubbing veins, and run the leaf graph analyzer."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=project_directory / "Real_shoot_leaves_processed",
    )
    parser.add_argument(
        "--app-dir",
        type=Path,
        default=script_app_directory,
    )
    args = parser.parse_args()
    if not args.processed_dir.is_dir():
        raise FileNotFoundError(args.processed_dir)
    results = process_folder(args.processed_dir, args.app_dir)
    ready = sum(item["status"] == "graph-ready" for item in results)
    print(
        f"Completed {len(results)} samples · {ready} graph-ready · "
        f"review sheet: {args.processed_dir / 'vein_review_sheet.png'}"
    )


if __name__ == "__main__":
    main()
