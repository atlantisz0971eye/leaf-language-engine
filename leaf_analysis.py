from __future__ import annotations

import base64
import heapq
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class AnalysisSettings:
    max_side: int = 1600
    sensitivity: float = 74.0
    contrast_scale: float = 12.0
    min_component: int = 6

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AnalysisSettings":
        data = data or {}
        return cls(
            max_side=max(700, min(2400, int(data.get("max_side", 1600)))),
            sensitivity=max(55.0, min(98.5, float(data.get("sensitivity", 74.0)))),
            contrast_scale=max(3.0, min(35.0, float(data.get("contrast_scale", 12.0)))),
            min_component=max(0, min(500, int(data.get("min_component", 6)))),
        )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _remove_small_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    if minimum <= 1:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    output = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum:
            output[labels == index] = 255
    return output


def _encode_png(image: np.ndarray) -> str:
    ok, payload = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode preview image")
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values.astype(np.float64) / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum() / math.log2(len(values)))


def _bounded(value: float, scale: float) -> float:
    return 1.0 - math.exp(-max(0.0, value) / scale)


def _infer_opaque_leaf_mask(bgr: np.ndarray) -> np.ndarray:
    """Infer a centered cleared-leaf silhouette from a flat black/white matte."""
    height, width = bgr.shape[:2]
    border_width = max(2, min(height, width) // 80)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate(
        [
            lab[:border_width].reshape(-1, 3),
            lab[-border_width:].reshape(-1, 3),
            lab[:, :border_width].reshape(-1, 3),
            lab[:, -border_width:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    border_distance = np.linalg.norm(border - background, axis=1)
    threshold = max(8.0, float(np.percentile(border_distance, 99)) * 2.5)
    candidate = np.where(distance > threshold, 255, 0).astype(np.uint8)
    kernel_size = max(7, 2 * round(min(height, width) / 320) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    candidate = _largest_component(candidate)
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full((height, width), 255, np.uint8)
    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / max(1.0, float(height * width))
    if area_ratio < 0.01 or area_ratio > 0.96:
        return np.full((height, width), 255, np.uint8)
    mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def _prepare_image(path: Path, max_side: int) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
    original = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if original is None:
        raise ValueError(f"Cannot read image: {path.name}")

    original_height, original_width = original.shape[:2]
    if original.ndim == 2:
        bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        alpha = np.full(original.shape, 255, np.uint8)
    elif original.shape[2] == 4:
        bgr = original[:, :, :3]
        alpha = original[:, :, 3]
    else:
        bgr = original[:, :, :3]
        alpha = np.full(original.shape[:2], 255, np.uint8)

    ratio = min(1.0, max_side / max(original_width, original_height))
    if ratio < 1.0:
        size = (max(1, round(original_width * ratio)), max(1, round(original_height * ratio)))
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)

    meaningful_alpha = bool(np.min(alpha) < 245 and np.mean(alpha > 245) < 0.985)
    leaf_mask = (
        np.where(alpha > 18, 255, 0).astype(np.uint8)
        if meaningful_alpha
        else _infer_opaque_leaf_mask(bgr)
    )
    leaf_mask = cv2.morphologyEx(
        leaf_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    leaf_mask = _largest_component(leaf_mask)

    composite = np.full_like(bgr, 255)
    composite[leaf_mask > 0] = bgr[leaf_mask > 0]
    return composite, leaf_mask, ratio, (original_width, original_height)


def _extract_veins(
    image: np.ndarray, leaf_mask: np.ndarray, settings: AnalysisSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    green = image[:, :, 1].astype(np.float32)
    red = image[:, :, 2].astype(np.float32)
    valid = leaf_mask > 0

    sigma_small = max(1.0, settings.contrast_scale * 0.35)
    sigma_large = max(2.0, settings.contrast_scale)
    local_small = cv2.GaussianBlur(green, (0, 0), sigma_small) - green
    local_large = cv2.GaussianBlur(green, (0, 0), sigma_large) - green

    redness = red - green
    red_local = redness - cv2.GaussianBlur(redness, (0, 0), sigma_large)
    vessel_score = np.maximum(local_small, 0) * 0.60
    vessel_score += np.maximum(local_large, 0) * 0.30
    vessel_score += np.maximum(red_local, 0) * 0.10
    vessel_score[~valid] = 0

    core = cv2.erode(
        leaf_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    sample = vessel_score[core > 0]
    positive = sample[sample > 0]
    threshold = float(np.percentile(positive, settings.sensitivity)) if positive.size else 0.0
    binary = np.where((vessel_score >= threshold) & (core > 0), 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    binary = _remove_small_components(binary, settings.min_component)

    skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    skeleton = _remove_small_components(skeleton, max(3, settings.min_component // 2))

    score_preview = np.zeros_like(vessel_score, dtype=np.uint8)
    if positive.size:
        low, high = np.percentile(positive, [10, 99.5])
        score_preview = np.clip((vessel_score - low) * 255.0 / max(1e-6, high - low), 0, 255).astype(
            np.uint8
        )
    score_preview[~valid] = 0
    return binary, skeleton, score_preview


def _network_metrics(
    skeleton: np.ndarray, binary: np.ndarray, leaf_mask: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    skel = skeleton > 0
    leaf = leaf_mask > 0
    leaf_area = int(leaf.sum())
    vertex_count = int(skel.sum())

    padded = np.pad(skel.astype(np.uint8), 1)
    ring = [
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
        padded[:-2, :-2],
    ]
    transitions = np.zeros_like(ring[0], dtype=np.uint8)
    for index in range(8):
        transitions += ((ring[index] == 0) & (ring[(index + 1) % 8] == 1)).astype(np.uint8)
    endpoint_mask = (skel & (transitions == 1)).astype(np.uint8) * 255
    junction_mask = (skel & (transitions >= 3)).astype(np.uint8) * 255
    endpoints = max(0, cv2.connectedComponents(endpoint_mask, 8)[0] - 1)
    junction_groups = cv2.dilate(
        junction_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    junctions = max(0, cv2.connectedComponents(junction_groups, 8)[0] - 1)

    horizontal = int(np.logical_and(skel[:, :-1], skel[:, 1:]).sum())
    vertical = int(np.logical_and(skel[:-1, :], skel[1:, :]).sum())
    diagonal_a = int(np.logical_and(skel[:-1, :-1], skel[1:, 1:]).sum())
    diagonal_b = int(np.logical_and(skel[:-1, 1:], skel[1:, :-1]).sum())
    vein_length = horizontal + vertical + math.sqrt(2.0) * (diagonal_a + diagonal_b)

    direction_counts = np.array([horizontal, diagonal_a, vertical, diagonal_b], dtype=np.float64)
    orientation_entropy = _entropy(direction_counts)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mesh_areas: list[float] = []
    if hierarchy is not None:
        for index, contour in enumerate(contours):
            if hierarchy[0][index][3] >= 0:
                area = float(cv2.contourArea(contour))
                if area >= 6.0:
                    mesh_areas.append(area)

    normalized_length = vein_length / max(1.0, math.sqrt(leaf_area))
    junction_density = junctions * 100_000.0 / max(1, leaf_area)
    endpoint_density = endpoints * 100_000.0 / max(1, leaf_area)
    mesh_density = len(mesh_areas) * 100_000.0 / max(1, leaf_area)

    complexity = 100.0 * (
        0.28 * _bounded(normalized_length, 120.0)
        + 0.24 * _bounded(junction_density, 180.0)
        + 0.20 * _bounded(mesh_density, 70.0)
        + 0.18 * orientation_entropy
        + 0.10 * _bounded(endpoint_density, 220.0)
    )

    metrics = {
        "experimental_complexity": round(complexity, 2),
        "leaf_area_px": leaf_area,
        "skeleton_pixels": vertex_count,
        "vein_length_px": round(vein_length, 2),
        "normalized_vein_length": round(normalized_length, 3),
        "junction_count": int(junctions),
        "junction_density_per_100k": round(junction_density, 3),
        "endpoint_count": int(endpoints),
        "endpoint_density_per_100k": round(endpoint_density, 3),
        "mesh_count": int(len(mesh_areas)),
        "mesh_density_per_100k": round(mesh_density, 3),
        "mean_mesh_area_px": round(float(np.mean(mesh_areas)), 3) if mesh_areas else 0.0,
        "median_mesh_area_px": round(float(np.median(mesh_areas)), 3) if mesh_areas else 0.0,
        "orientation_entropy": round(orientation_entropy, 4),
    }
    return metrics, endpoint_mask, junction_mask


def _make_overlay(
    image: np.ndarray,
    leaf_mask: np.ndarray,
    skeleton: np.ndarray,
    junction_mask: np.ndarray,
) -> np.ndarray:
    overlay = image.copy()
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255
    skel_dilated = cv2.dilate(skeleton, np.ones((2, 2), np.uint8))
    active = skel_dilated > 0
    overlay[active] = cv2.addWeighted(overlay[active], 0.30, green[active], 0.70, 0)

    junctions = cv2.dilate(junction_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    overlay[junctions > 0] = (30, 40, 230)

    alpha = leaf_mask.copy()
    return np.dstack([overlay, alpha])


def _make_heatmap(
    image: np.ndarray, leaf_mask: np.ndarray, skeleton: np.ndarray, junction_mask: np.ndarray
) -> np.ndarray:
    height, width = leaf_mask.shape
    window = max(21, int(max(height, width) * 0.045) | 1)
    skel_float = (skeleton > 0).astype(np.float32)
    junction_float = (junction_mask > 0).astype(np.float32)
    local_lines = cv2.GaussianBlur(skel_float, (0, 0), window / 4)
    local_nodes = cv2.GaussianBlur(junction_float, (0, 0), window / 3)
    score = local_lines + local_nodes * 4.0
    values = score[leaf_mask > 0]
    if values.size:
        ceiling = max(1e-6, float(np.percentile(values, 99)))
        normalized = np.clip(score * 255.0 / ceiling, 0, 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(leaf_mask)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    blended = cv2.addWeighted(image, 0.45, colored, 0.55, 0)
    alpha = leaf_mask.copy()
    return np.dstack([blended, alpha])


def _graph_payload(
    skeleton: np.ndarray,
    binary: np.ndarray,
    leaf_mask: np.ndarray,
    endpoint_mask: np.ndarray,
    junction_mask: np.ndarray,
) -> dict[str, Any]:
    """Convert the pixel skeleton into a compact, navigable vein graph."""
    height, width = skeleton.shape
    skel = skeleton > 0
    node_seed = cv2.dilate(
        junction_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    node_seed = np.where((node_seed > 0) | (endpoint_mask > 0), 255, 0).astype(np.uint8)
    node_seed[~skel] = 0

    count, node_labels, stats, centroids = cv2.connectedComponentsWithStats(node_seed, 8)
    nodes: list[dict[str, Any]] = []
    label_to_node: dict[int, int] = {}
    for label in range(1, count):
        ys, xs = np.where(node_labels == label)
        if not len(xs):
            continue
        node_id = len(nodes)
        label_to_node[label] = node_id
        is_junction = bool(np.any(junction_mask[ys, xs] > 0))
        nodes.append(
            {
                "id": node_id,
                "type": "junction" if is_junction else "endpoint",
                "x": round(float(centroids[label][0]), 2),
                "y": round(float(centroids[label][1]), 2),
                "pixel_area": int(stats[label, cv2.CC_STAT_AREA]),
            }
        )

    chain_mask = np.where(skel & (node_labels == 0), 255, 0).astype(np.uint8)
    chain_count, chain_labels, chain_stats, _ = cv2.connectedComponentsWithStats(chain_mask, 8)
    distance = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)
    edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    neighbours = (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    )

    def trace_chain(component: np.ndarray, source_label: int, target_label: int) -> list[tuple[int, int]]:
        pixels = {tuple(item) for item in np.argwhere(component)}
        if not pixels:
            return []
        source_region = node_labels == source_label
        target_region = node_labels == target_label
        source_touch = cv2.dilate(source_region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        target_touch = cv2.dilate(target_region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        starts = [pixel for pixel in pixels if source_touch[pixel]]
        targets = {pixel for pixel in pixels if target_touch[pixel]}
        if not starts or not targets:
            return []
        start = starts[0]
        queue = deque([start])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        finish: tuple[int, int] | None = None
        while queue:
            current = queue.popleft()
            if current in targets:
                finish = current
                break
            cy, cx = current
            for dy, dx in neighbours:
                candidate = (cy + dy, cx + dx)
                if candidate in pixels and candidate not in parent:
                    parent[candidate] = current
                    queue.append(candidate)
        if finish is None:
            return []
        path: list[tuple[int, int]] = []
        cursor: tuple[int, int] | None = finish
        while cursor is not None:
            path.append(cursor)
            cursor = parent[cursor]
        return list(reversed(path))

    for chain_label in range(1, chain_count):
        if chain_stats[chain_label, cv2.CC_STAT_AREA] < 2:
            continue
        component = chain_labels == chain_label
        halo = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        touching = sorted(int(item) for item in np.unique(node_labels[halo]) if item > 0)
        if len(touching) < 2:
            continue
        if len(touching) > 2:
            touching = list(
                max(
                    (
                        (a, b)
                        for index, a in enumerate(touching)
                        for b in touching[index + 1 :]
                    ),
                    key=lambda pair: float(
                        np.linalg.norm(centroids[pair[0]] - centroids[pair[1]])
                    ),
                )
            )
        source_label, target_label = touching[:2]
        source = label_to_node.get(source_label)
        target = label_to_node.get(target_label)
        if source is None or target is None or source == target:
            continue
        pair = tuple(sorted((source, target)))
        if pair in seen_pairs:
            continue
        path_yx = trace_chain(component, source_label, target_label)
        if len(path_yx) < 2:
            continue
        seen_pairs.add(pair)
        source_xy = (nodes[source]["x"], nodes[source]["y"])
        target_xy = (nodes[target]["x"], nodes[target]["y"])
        path_xy = [source_xy] + [(float(x), float(y)) for y, x in path_yx] + [target_xy]
        length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(path_xy, path_xy[1:])
        )
        sample_indices = np.linspace(0, len(path_xy) - 1, min(28, len(path_xy))).astype(int)
        sampled = [[round(path_xy[i][0], 2), round(path_xy[i][1], 2)] for i in sample_indices]
        widths = [float(distance[y, x] * 2.0) for y, x in path_yx]
        edges.append(
            {
                "id": len(edges),
                "source": source,
                "target": target,
                "length_px": round(length, 2),
                "mean_width_px": round(float(np.mean(widths)) if widths else 1.0, 2),
                "path": sampled,
                "order": None,
            }
        )

    adjacency: dict[int, list[tuple[int, int]]] = {node["id"]: [] for node in nodes}
    for edge in edges:
        adjacency[edge["source"]].append((edge["target"], edge["id"]))
        adjacency[edge["target"]].append((edge["source"], edge["id"]))
    for node in nodes:
        node["degree"] = len(adjacency[node["id"]])

    endpoint_ids = [node["id"] for node in nodes if node["degree"] == 1]
    main_nodes: list[int] = []
    main_edges: list[int] = []
    if edges and len(endpoint_ids) >= 2:
        leaf_yx = np.argwhere(leaf_mask > 0)
        xy = leaf_yx[:, ::-1].astype(np.float64)
        axis = np.linalg.eigh(np.cov(xy, rowvar=False))[1][:, -1]
        projections = {node["id"]: float(np.dot([node["x"], node["y"]], axis)) for node in nodes}
        ordered_endpoints = sorted(endpoint_ids, key=lambda item: projections[item])
        extreme_count = max(1, len(ordered_endpoints) // 6)
        extreme_ids = ordered_endpoints[:extreme_count] + ordered_endpoints[-extreme_count:]

        def incident_width(node_id: int) -> float:
            return max((edges[edge_id]["mean_width_px"] for _, edge_id in adjacency[node_id]), default=0)

        petiole = max(extreme_ids, key=incident_width)
        petiole_projection = projections[petiole]
        apex = max(
            endpoint_ids,
            key=lambda item: abs(projections[item] - petiole_projection)
            + 0.15
            * math.hypot(
                nodes[item]["x"] - nodes[petiole]["x"],
                nodes[item]["y"] - nodes[petiole]["y"],
            ),
        )

        distances = {node["id"]: float("inf") for node in nodes}
        previous: dict[int, tuple[int, int]] = {}
        distances[petiole] = 0.0
        heap: list[tuple[float, int]] = [(0.0, petiole)]
        while heap:
            current_distance, current = heapq.heappop(heap)
            if current_distance != distances[current]:
                continue
            for neighbour, edge_id in adjacency[current]:
                edge = edges[edge_id]
                cost = edge["length_px"] / max(1.0, edge["mean_width_px"] ** 1.35)
                candidate = current_distance + cost
                if candidate < distances[neighbour]:
                    distances[neighbour] = candidate
                    previous[neighbour] = (current, edge_id)
                    heapq.heappush(heap, (candidate, neighbour))
        if math.isinf(distances[apex]):
            reachable = [item for item in endpoint_ids if not math.isinf(distances[item])]
            apex = max(reachable, key=lambda item: distances[item], default=petiole)
        cursor = apex
        main_nodes = [cursor]
        while cursor != petiole and cursor in previous:
            cursor, edge_id = previous[cursor]
            main_edges.append(edge_id)
            main_nodes.append(cursor)
        main_nodes.reverse()
        main_edges.reverse()

        main_edge_set = set(main_edges)
        main_node_set = set(main_nodes)
        for edge_id in main_edges:
            edges[edge_id]["order"] = 1
        queue: deque[tuple[int, int]] = deque()
        for node_id in main_nodes:
            for neighbour, edge_id in adjacency[node_id]:
                if edge_id not in main_edge_set:
                    edges[edge_id]["order"] = 2
                    queue.append((neighbour, 2))
        visited_edges = set(main_edges)
        while queue:
            node_id, vein_order = queue.popleft()
            for neighbour, edge_id in adjacency[node_id]:
                if edge_id in visited_edges or edges[edge_id]["order"] is not None:
                    continue
                edges[edge_id]["order"] = min(6, vein_order + 1)
                visited_edges.add(edge_id)
                queue.append((neighbour, min(6, vein_order + 1)))
        for edge in edges:
            if edge["order"] is None:
                edge["order"] = 6
        for node in nodes:
            node["role"] = "main_vein" if node["id"] in main_node_set else node["type"]

    anchors = []
    main_edge_set = set(main_edges)
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

    return {
        "version": "leaf-surface-graph-0.5",
        "coordinate_system": "analysis_pixels_top_left_origin",
        "width": int(width),
        "height": int(height),
        "nodes": nodes,
        "edges": edges,
        "main_path": {"node_ids": main_nodes, "edge_ids": main_edges},
        "secondary_anchors": anchors,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "main_edge_count": len(main_edges),
            "secondary_anchor_count": len(anchors),
        },
    }


def analyze_image(path: Path, settings: AnalysisSettings) -> dict[str, Any]:
    image, leaf_mask, resize_ratio, original_size = _prepare_image(path, settings.max_side)
    binary, skeleton, score_preview = _extract_veins(image, leaf_mask, settings)
    metrics, endpoint_mask, junction_mask = _network_metrics(skeleton, binary, leaf_mask)
    overlay = _make_overlay(image, leaf_mask, skeleton, junction_mask)
    heatmap = _make_heatmap(image, leaf_mask, skeleton, junction_mask)

    original_rgba = np.dstack([image, leaf_mask])
    mask_rgba = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGRA)
    mask_rgba[:, :, 3] = leaf_mask
    skeleton_rgba = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGRA)
    skeleton_rgba[:, :, 3] = leaf_mask
    score_rgba = cv2.applyColorMap(score_preview, cv2.COLORMAP_VIRIDIS)
    score_rgba = np.dstack([score_rgba, leaf_mask])

    metrics.update(
        {
            "file_name": path.name,
            "original_width": original_size[0],
            "original_height": original_size[1],
            "analysis_width": int(image.shape[1]),
            "analysis_height": int(image.shape[0]),
            "resize_ratio": round(resize_ratio, 6),
        }
    )

    return {
        "metrics": metrics,
        "graph": _graph_payload(skeleton, binary, leaf_mask, endpoint_mask, junction_mask),
        "images": {
            "original": original_rgba,
            "vessel_score": score_rgba,
            "mask": mask_rgba,
            "skeleton": skeleton_rgba,
            "overlay": overlay,
            "heatmap": heatmap,
        },
    }


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "metrics": result["metrics"],
        "graph": result["graph"],
        "previews": {name: _encode_png(image) for name, image in result["images"].items()},
    }


def export_result(result: dict[str, Any], output_root: Path) -> Path:
    stem = Path(result["metrics"]["file_name"]).stem
    destination = output_root / stem
    destination.mkdir(parents=True, exist_ok=True)

    for name, image in result["images"].items():
        cv2.imwrite(str(destination / f"{name}.png"), image)

    (destination / "metrics.json").write_text(
        json.dumps(result["metrics"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "graph.json").write_text(
        json.dumps(result["graph"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
