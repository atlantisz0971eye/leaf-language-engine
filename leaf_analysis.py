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
    bgr = image.astype(np.float32)
    green = bgr[:, :, 1]
    red = bgr[:, :, 2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    luminance = lab[:, :, 0]
    dye = lab[:, :, 1]
    valid = leaf_mask > 0

    scales = (
        max(0.8, settings.contrast_scale * 0.18),
        max(1.6, settings.contrast_scale * 0.48),
        max(3.0, settings.contrast_scale * 1.15),
    )
    dark_responses = []
    dye_responses = []
    for sigma in scales:
        dark_responses.append(np.maximum(cv2.GaussianBlur(luminance, (0, 0), sigma) - luminance, 0))
        dye_responses.append(np.abs(dye - cv2.GaussianBlur(dye, (0, 0), sigma)))
    darkness = np.maximum.reduce(dark_responses)
    dye_ridge = np.maximum.reduce(dye_responses)
    green_ridge = np.maximum(
        cv2.GaussianBlur(green, (0, 0), scales[1]) - green,
        cv2.GaussianBlur(red, (0, 0), scales[1]) - red,
    )
    # Multichannel, multiscale response works for red/pink NMNS dyes as well
    # as the original green transparent specimens.
    vessel_score = darkness * 0.56 + dye_ridge * 0.29 + np.maximum(green_ridge, 0) * 0.15
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


def _trace_response_ridge(
    root: tuple[float, float],
    direction: tuple[float, float],
    vessel_score: np.ndarray,
    binary: np.ndarray,
    leaf_mask: np.ndarray,
) -> list[list[float]]:
    """Follow a primary ridge through small threshold gaps in image space."""
    height, width = leaf_mask.shape
    dx, dy = direction
    norm = max(1e-6, math.hypot(dx, dy))
    dx, dy = dx / norm, dy / norm
    nx, ny = -dy, dx
    ys, xs = np.where(leaf_mask > 0)
    if not len(xs):
        return [[round(root[0], 2), round(root[1], 2)]]
    along = (xs - root[0]) * dx + (ys - root[1]) * dy
    across = (xs - root[0]) * nx + (ys - root[1]) * ny
    corridor = max(18.0, min(height, width) * 0.115)
    forward = along[(along >= 0) & (np.abs(across) <= corridor)]
    if not len(forward):
        return [[round(root[0], 2), round(root[1], 2)]]
    maximum_distance = float(np.percentile(forward, 99.4))
    step = max(2.0, min(height, width) / 430.0)
    lateral_step = max(1.5, step * 0.8)
    distances = np.arange(0.0, maximum_distance + step, step, dtype=np.float32)
    offsets = np.arange(-corridor, corridor + lateral_step, lateral_step, dtype=np.float32)
    if len(distances) < 3 or len(offsets) < 3:
        return [[round(root[0], 2), round(root[1], 2)]]

    previous = np.full(len(offsets), np.inf, dtype=np.float64)
    origin_index = int(np.argmin(np.abs(offsets)))
    previous[origin_index] = 0.0
    parents = np.full((len(distances), len(offsets)), -1, dtype=np.int16)
    transition_radius = max(2, round(8.0 / lateral_step))

    for row, distance_value in enumerate(distances):
        sample_x = np.rint(root[0] + dx * distance_value + nx * offsets).astype(np.int32)
        sample_y = np.rint(root[1] + dy * distance_value + ny * offsets).astype(np.int32)
        valid = (
            (sample_x >= 0)
            & (sample_x < width)
            & (sample_y >= 0)
            & (sample_y < height)
        )
        clipped_x = np.clip(sample_x, 0, width - 1)
        clipped_y = np.clip(sample_y, 0, height - 1)
        inside = valid & (leaf_mask[clipped_y, clipped_x] > 0)
        response = vessel_score[clipped_y, clipped_x].astype(np.float64) / 255.0
        on_mask = (binary[clipped_y, clipped_x] > 0).astype(np.float64)
        emission = 1.55 * (1.0 - response) ** 2 - on_mask * 0.34
        emission += 0.10 * (np.abs(offsets) / max(1.0, corridor)) ** 1.6
        emission[~inside] += 7.0
        current = np.full(len(offsets), np.inf, dtype=np.float64)
        for index in range(len(offsets)):
            low = max(0, index - transition_radius)
            high = min(len(offsets), index + transition_radius + 1)
            candidates = previous[low:high]
            if not np.isfinite(candidates).any():
                continue
            local = int(np.argmin(candidates + 0.045 * (np.arange(low, high) - index) ** 2))
            parent = low + local
            current[index] = candidates[local] + 0.045 * (parent - index) ** 2 + emission[index]
            parents[row, index] = parent
        previous = current

    cursor = int(np.argmin(previous))
    chosen = np.zeros(len(distances), dtype=np.int32)
    for row in range(len(distances) - 1, -1, -1):
        chosen[row] = cursor
        parent = int(parents[row, cursor])
        if parent >= 0:
            cursor = parent
    points = []
    stride = max(1, round(5.0 / step))
    for index in range(0, len(distances), stride):
        offset = float(offsets[chosen[index]])
        x = root[0] + dx * float(distances[index]) + nx * offset
        y = root[1] + dy * float(distances[index]) + ny * offset
        if 0 <= round(x) < width and 0 <= round(y) < height and leaf_mask[round(y), round(x)] > 0:
            points.append([round(float(x), 2), round(float(y), 2)])
    if points and points[-1] != [round(float(root[0]), 2), round(float(root[1]), 2)]:
        return points
    return [[round(root[0], 2), round(root[1], 2)]]


def _response_guided_primary_paths(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    adjacency: dict[int, list[tuple[int, int]]],
    endpoint_ids: list[int],
    component: set[int],
    leaf_mask: np.ndarray,
    vessel_score: np.ndarray,
    binary: np.ndarray,
) -> dict[str, Any] | None:
    """Trace primary veins while remaining on response-supported graph edges.

    A cleared leaf already supplies a strong ridge image.  The graph therefore
    acts as an address structure, not as permission to invent a smoother path.
    The petiole is followed to its first basal junction and every strong radial
    continuation is traced independently.  The most collinear continuation is
    the central primary vein.
    """
    component_endpoints = [item for item in endpoint_ids if item in component]
    if len(component_endpoints) < 2:
        return None

    leaf_yx = np.argwhere(leaf_mask > 0)
    if not len(leaf_yx):
        return None
    leaf_xy = leaf_yx[:, ::-1].astype(np.float64)
    leaf_centre = np.mean(leaf_xy, axis=0)
    leaf_diagonal = math.hypot(leaf_mask.shape[1], leaf_mask.shape[0])

    component_edge_ids = {
        edge_id
        for node_id in component
        for _, edge_id in adjacency.get(node_id, [])
    }
    component_widths = [float(edges[item].get("mean_width_px", 1.0)) for item in component_edge_ids]
    median_width = float(np.median(component_widths)) if component_widths else 1.0

    def edge_quality(edge_id: int) -> float:
        edge = edges[edge_id]
        response = (
            float(edge.get("response_mean", 0.0)) * 0.46
            + float(edge.get("response_p75", 0.0)) * 0.22
            + float(edge.get("response_support", 0.0)) * 0.20
        )
        width_score = min(1.0, float(edge.get("mean_width_px", 1.0)) / max(2.0, median_width * 3.5))
        return response + width_score * 0.12

    def other_node(edge_id: int, node_id: int) -> int:
        edge = edges[edge_id]
        return int(edge["target"]) if int(edge["source"]) == node_id else int(edge["source"])

    def unit_direction(source_id: int, target_id: int) -> tuple[float, float]:
        dx = float(nodes[target_id]["x"]) - float(nodes[source_id]["x"])
        dy = float(nodes[target_id]["y"]) - float(nodes[source_id]["y"])
        norm = max(1e-6, math.hypot(dx, dy))
        return dx / norm, dy / norm

    def trace_terminal_chain(endpoint: int) -> tuple[list[int], list[int]]:
        chain_nodes = [endpoint]
        chain_edges: list[int] = []
        previous_node: int | None = None
        current_node = endpoint
        visited = {endpoint}
        for _ in range(96):
            choices = [
                (neighbour, edge_id)
                for neighbour, edge_id in adjacency[current_node]
                if neighbour in component and neighbour != previous_node
            ]
            if not choices:
                break
            if current_node != endpoint and len(choices) >= 2:
                maximum_quality = max(edge_quality(edge_id) for _, edge_id in choices)
                maximum_width = max(
                    float(edges[edge_id].get("mean_width_px", 1.0))
                    for _, edge_id in choices
                )
                significant = [
                    item
                    for item in choices
                    if edge_quality(item[1]) >= max(0.12, maximum_quality * 0.38)
                    and float(edges[item[1]].get("mean_width_px", 1.0))
                    >= max(1.0, maximum_width * 0.22)
                ]
                if len(significant) >= 2:
                    break
                if significant:
                    choices = significant
            neighbour, edge_id = max(
                choices,
                key=lambda item: (
                    edge_quality(item[1]),
                    float(edges[item[1]].get("mean_width_px", 1.0)),
                    float(edges[item[1]].get("length_px", 0.0)),
                ),
            )
            if neighbour in visited:
                break
            chain_edges.append(edge_id)
            chain_nodes.append(neighbour)
            visited.add(neighbour)
            previous_node, current_node = current_node, neighbour
        return chain_nodes, chain_edges

    terminal_candidates = []
    for endpoint in component_endpoints:
        chain_nodes, chain_edges = trace_terminal_chain(endpoint)
        if not chain_edges:
            continue
        root_candidate = chain_nodes[-1]
        incoming = chain_edges[-1]
        outgoing = [
            edge_id
            for neighbour, edge_id in adjacency[root_candidate]
            if neighbour in component and edge_id != incoming
        ]
        strong_outgoing = [
            edge_id
            for edge_id in outgoing
            if edge_quality(edge_id) >= 0.16
            and float(edges[edge_id].get("mean_width_px", 1.0)) >= max(1.0, median_width * 0.65)
        ]
        length = sum(float(edges[item].get("length_px", 0.0)) for item in chain_edges)
        mean_width = sum(
            float(edges[item].get("mean_width_px", 1.0))
            * float(edges[item].get("length_px", 0.0))
            for item in chain_edges
        ) / max(1.0, length)
        mean_quality = sum(
            edge_quality(item) * float(edges[item].get("length_px", 0.0))
            for item in chain_edges
        ) / max(1.0, length)
        radial = math.hypot(
            float(nodes[endpoint]["x"]) - leaf_centre[0],
            float(nodes[endpoint]["y"]) - leaf_centre[1],
        ) / max(1.0, leaf_diagonal)
        score = (
            max(1.0, length) ** 0.72
            * max(1.0, mean_width) ** 0.92
            * (0.45 + mean_quality)
            * (1.0 + min(6, len(strong_outgoing)) * 0.65)
            * (0.55 + radial)
        )
        terminal_candidates.append(
            (score, endpoint, chain_nodes, chain_edges, root_candidate)
        )
    if not terminal_candidates:
        return None
    _, petiole, petiole_nodes, petiole_edges, root = max(terminal_candidates)
    if len(petiole_nodes) < 2 or len(adjacency[root]) < 3:
        return None
    incoming_direction = unit_direction(petiole_nodes[-2], root)
    incoming_edge = petiole_edges[-1]

    root_choices = [
        (neighbour, edge_id)
        for neighbour, edge_id in adjacency[root]
        if edge_id != incoming_edge and neighbour in component
    ]
    if not root_choices:
        return None
    best_root_quality = max(edge_quality(edge_id) for _, edge_id in root_choices)
    best_root_width = max(float(edges[edge_id].get("mean_width_px", 1.0)) for _, edge_id in root_choices)
    root_choices = [
        item
        for item in root_choices
        if (
            edge_quality(item[1]) >= max(0.08, best_root_quality * 0.28)
            or float(edges[item[1]].get("mean_width_px", 1.0)) >= best_root_width * 0.30
        )
        and float(edges[item[1]].get("mean_width_px", 1.0)) >= max(1.0, best_root_width * 0.14)
    ]

    def trace_branch(first_neighbour: int, first_edge: int) -> dict[str, Any]:
        path_nodes = [root, first_neighbour]
        path_edges = [first_edge]
        used_nodes = {root, first_neighbour}
        previous = root
        current_node = first_neighbour
        previous_direction = unit_direction(root, first_neighbour)
        initial_direction = previous_direction
        cumulative_length = float(edges[first_edge].get("length_px", 0.0))
        for _ in range(180):
            choices = []
            for neighbour, edge_id in adjacency[current_node]:
                if neighbour == previous or neighbour in used_nodes or neighbour not in component:
                    continue
                direction = unit_direction(current_node, neighbour)
                alignment = max(0.0, direction[0] * previous_direction[0] + direction[1] * previous_direction[1])
                radial_alignment = max(
                    0.0,
                    direction[0] * initial_direction[0]
                    + direction[1] * initial_direction[1],
                )
                width_score = min(
                    1.0,
                    float(edges[edge_id].get("mean_width_px", 1.0))
                    / max(1.0, float(edges[first_edge].get("mean_width_px", 1.0))),
                )
                score = (
                    edge_quality(edge_id) * 0.50
                    + alignment * 0.22
                    + radial_alignment * 0.20
                    + width_score * 0.08
                )
                choices.append((score, alignment, edge_quality(edge_id), neighbour, edge_id, direction))
            if not choices:
                break
            score, alignment, quality, neighbour, edge_id, direction = max(choices)
            if cumulative_length > leaf_diagonal * 0.12 and alignment < 0.05 and quality < 0.18:
                break
            if score < 0.11:
                break
            path_edges.append(edge_id)
            path_nodes.append(neighbour)
            used_nodes.add(neighbour)
            cumulative_length += float(edges[edge_id].get("length_px", 0.0))
            previous, current_node = current_node, neighbour
            previous_direction = direction
            if cumulative_length > leaf_diagonal * 1.35:
                break

        length = sum(float(edges[item].get("length_px", 0.0)) for item in path_edges)
        weighted_support = sum(
            float(edges[item].get("response_support", 0.0))
            * float(edges[item].get("length_px", 0.0))
            for item in path_edges
        ) / max(1.0, length)
        weighted_response = sum(
            float(edges[item].get("response_mean", 0.0))
            * float(edges[item].get("length_px", 0.0))
            for item in path_edges
        ) / max(1.0, length)
        first_direction = unit_direction(root, first_neighbour)
        continuation = max(
            0.0,
            incoming_direction[0] * first_direction[0]
            + incoming_direction[1] * first_direction[1],
        )
        endpoint = nodes[path_nodes[-1]]
        extent = math.hypot(
            float(endpoint["x"]) - float(nodes[root]["x"]),
            float(endpoint["y"]) - float(nodes[root]["y"]),
        ) / max(1.0, leaf_diagonal)
        angle = math.atan2(first_direction[1], first_direction[0])
        central_score = continuation * 0.56 + weighted_response * 0.20 + weighted_support * 0.14 + min(1.0, extent * 2.0) * 0.10
        return {
            "node_ids": path_nodes,
            "edge_ids": path_edges,
            "angle": angle,
            "length_px": length,
            "response_support": weighted_support,
            "response_mean": weighted_response,
            "central_score": central_score,
        }

    primary_paths = [trace_branch(neighbour, edge_id) for neighbour, edge_id in root_choices]
    primary_paths = [
        item
        for item in primary_paths
        if item["length_px"] >= leaf_diagonal * 0.01
    ]
    if not primary_paths:
        return None
    central = max(primary_paths, key=lambda item: item["central_score"])
    ordered_paths = sorted(primary_paths, key=lambda item: item["angle"])
    serialised_paths = []
    for index, item in enumerate(ordered_paths):
        serialised_paths.append(
            {
                "id": index,
                "role": "central" if item is central else "radial_primary",
                "root_node_id": root,
                "node_ids": item["node_ids"],
                "edge_ids": item["edge_ids"],
                "angle_radians": round(float(item["angle"]), 5),
                "length_px": round(float(item["length_px"]), 2),
                "response_support": round(float(item["response_support"]), 4),
                "response_mean": round(float(item["response_mean"]), 4),
            }
        )

    def oriented_edge_points(edge_id: int, source_id: int) -> list[list[float]]:
        points = [list(map(float, point)) for point in edges[edge_id].get("path", [])]
        if len(points) < 2:
            return points
        source_xy = (float(nodes[source_id]["x"]), float(nodes[source_id]["y"]))
        if math.hypot(points[-1][0] - source_xy[0], points[-1][1] - source_xy[1]) < math.hypot(
            points[0][0] - source_xy[0], points[0][1] - source_xy[1]
        ):
            points.reverse()
        return points

    petiole_pixel_path: list[list[float]] = []
    for index, edge_id in enumerate(petiole_edges):
        segment = oriented_edge_points(edge_id, petiole_nodes[index])
        if petiole_pixel_path and segment:
            segment = segment[1:]
        petiole_pixel_path.extend(segment)
    ridge_path = _trace_response_ridge(
        (float(nodes[root]["x"]), float(nodes[root]["y"])),
        incoming_direction,
        vessel_score,
        binary,
        leaf_mask,
    )
    main_pixel_path = petiole_pixel_path + (ridge_path[1:] if petiole_pixel_path else ridge_path)

    return {
        "petiole_node_id": petiole,
        "root_node_id": root,
        "petiole_nodes": petiole_nodes,
        "petiole_edges": petiole_edges,
        "main_nodes": petiole_nodes + central["node_ids"][1:],
        "main_edges": petiole_edges + central["edge_ids"],
        "main_pixel_path": main_pixel_path,
        "primary_paths": serialised_paths,
    }


def _graph_payload(
    skeleton: np.ndarray,
    binary: np.ndarray,
    leaf_mask: np.ndarray,
    vessel_score: np.ndarray,
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
        response_values = np.asarray(
            [float(vessel_score[y, x]) for y, x in path_yx], dtype=np.float32
        )
        response_mean = float(np.mean(response_values)) if response_values.size else 0.0
        response_p75 = float(np.percentile(response_values, 75)) if response_values.size else 0.0
        response_support = (
            float(np.mean(response_values >= 72.0)) if response_values.size else 0.0
        )
        edges.append(
            {
                "id": len(edges),
                "source": source,
                "target": target,
                "length_px": round(length, 2),
                "mean_width_px": round(float(np.mean(widths)) if widths else 1.0, 2),
                "response_mean": round(response_mean / 255.0, 4),
                "response_p75": round(response_p75 / 255.0, 4),
                "response_support": round(response_support, 4),
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
    main_pixel_path: list[list[float]] = []
    primary_paths: list[dict[str, Any]] = []
    primary_root_node_id: int | None = None
    petiole_node_id: int | None = None
    specimen_component_edge_ids: list[int] = []
    path_source = "graph_fallback"
    regularized = False
    if edges and len(endpoint_ids) >= 2:
        # Skeleton thresholding can leave several disconnected vein islands.
        # Select the component that spans the specimen before looking for a
        # primary path; otherwise a thick pigment patch can win by width alone.
        components: list[set[int]] = []
        unseen = set(adjacency)
        while unseen:
            start = unseen.pop()
            component = {start}
            stack = [start]
            while stack:
                current = stack.pop()
                for neighbour, _ in adjacency[current]:
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        component.add(neighbour)
                        stack.append(neighbour)
            if len(component) >= 2:
                components.append(component)

        def component_score(component: set[int]) -> float:
            component_edges = {
                edge_id
                for node_id in component
                for _, edge_id in adjacency[node_id]
            }
            xs = [nodes[item]["x"] for item in component]
            ys = [nodes[item]["y"] for item in component]
            extent_x = max(xs) - min(xs)
            extent_y = max(ys) - min(ys)
            major_extent = max(1.0, extent_x, extent_y)
            minor_extent = max(1.0, min(extent_x, extent_y))
            spatial_area = math.sqrt(max(1.0, extent_x * extent_y))
            total = sum(
                float(edges[item]["length_px"])
                * (0.35 + float(edges[item].get("response_mean", 0.0)))
                for item in component_edges
            )
            # A leaf network occupies a two-dimensional field. Labels, rulers
            # and frame fragments are long but nearly one-dimensional and must
            # not win merely because one stroke is thick.
            two_dimensionality = minor_extent / major_extent
            return (
                math.log1p(total)
                * math.sqrt(max(1.0, len(component)))
                * spatial_area
                * (0.22 + two_dimensionality)
            )

        primary_component = max(components, key=component_score, default=set(endpoint_ids))
        specimen_component_edge_ids = sorted(
            {
                int(edge_id)
                for node_id in primary_component
                for _, edge_id in adjacency[node_id]
            }
        )
        component_endpoints = [item for item in endpoint_ids if item in primary_component]
        if len(component_endpoints) >= 2:
            endpoint_ids = component_endpoints

        leaf_yx = np.argwhere(leaf_mask > 0)
        xy = leaf_yx[:, ::-1].astype(np.float64)
        axis = np.linalg.eigh(np.cov(xy, rowvar=False))[1][:, -1]
        projections = {node["id"]: float(np.dot([node["x"], node["y"]], axis)) for node in nodes}
        leaf_centre = np.mean(xy, axis=0)
        leaf_diagonal = math.hypot(width, height)

        def incident_width(node_id: int) -> float:
            return max((edges[edge_id]["mean_width_px"] for _, edge_id in adjacency[node_id]), default=0)

        def petiole_score(node_id: int) -> float:
            incident = [edges[edge_id] for _, edge_id in adjacency[node_id]]
            longest = max((float(edge["length_px"]) for edge in incident), default=0.0)
            radial = math.hypot(nodes[node_id]["x"] - leaf_centre[0], nodes[node_id]["y"] - leaf_centre[1])
            return incident_width(node_id) ** 1.45 * math.sqrt(max(1.0, longest)) * (0.6 + radial / max(1.0, leaf_diagonal))

        petiole = max(endpoint_ids, key=petiole_score)
        apex = max(
            endpoint_ids,
            key=lambda item: math.hypot(
                nodes[item]["x"] - nodes[petiole]["x"],
                nodes[item]["y"] - nodes[petiole]["y"],
            ) + 0.08 * abs(projections[item] - projections[petiole]),
        )

        distances = {node["id"]: float("inf") for node in nodes}
        previous: dict[int, tuple[int, int]] = {}
        distances[petiole] = 0.0
        heap: list[tuple[float, int]] = [(0.0, petiole)]
        target_dx = nodes[apex]["x"] - nodes[petiole]["x"]
        target_dy = nodes[apex]["y"] - nodes[petiole]["y"]
        target_norm = max(1e-6, math.hypot(target_dx, target_dy))
        while heap:
            current_distance, current = heapq.heappop(heap)
            if current_distance != distances[current]:
                continue
            for neighbour, edge_id in adjacency[current]:
                edge = edges[edge_id]
                edge_dx = nodes[neighbour]["x"] - nodes[current]["x"]
                edge_dy = nodes[neighbour]["y"] - nodes[current]["y"]
                edge_norm = max(1e-6, math.hypot(edge_dx, edge_dy))
                alignment = abs((edge_dx * target_dx + edge_dy * target_dy) / (edge_norm * target_norm))
                direction_penalty = 1.0 + 2.4 * (1.0 - alignment) ** 2
                cost = (
                    edge["length_px"]
                    / max(1.0, edge["mean_width_px"] ** 1.48)
                    * direction_penalty
                )
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

        guided = _response_guided_primary_paths(
            nodes,
            edges,
            adjacency,
            endpoint_ids,
            primary_component,
            leaf_mask,
            vessel_score,
            binary,
        )
        if guided:
            main_nodes = [int(item) for item in guided["main_nodes"]]
            main_edges = [int(item) for item in guided["main_edges"]]
            main_pixel_path = list(guided["main_pixel_path"])
            primary_paths = list(guided["primary_paths"])
            primary_root_node_id = int(guided["root_node_id"])
            petiole_node_id = int(guided["petiole_node_id"])
            path_source = "vessel_response_graph"

            # Preserve every measured pixel path.  Orders are rebuilt from the
            # response-supported radial primaries, never from a fitted curve.
            primary_edge_set = {
                int(edge_id)
                for path in primary_paths
                for edge_id in path.get("edge_ids", [])
            } | set(int(item) for item in guided["petiole_edges"])
            primary_node_set = {
                int(node_id)
                for path in primary_paths
                for node_id in path.get("node_ids", [])
            } | set(int(item) for item in guided["petiole_nodes"])
            for edge in edges:
                edge["order"] = 1 if int(edge["id"]) in primary_edge_set else None
            visited_edges = set(primary_edge_set)
            queue = deque()
            for node_id in primary_node_set:
                for neighbour, edge_id in adjacency[node_id]:
                    if edge_id in visited_edges:
                        continue
                    edges[edge_id]["order"] = 2
                    visited_edges.add(edge_id)
                    queue.append((neighbour, 2))
            while queue:
                node_id, vein_order = queue.popleft()
                for neighbour, edge_id in adjacency[node_id]:
                    if edge_id in visited_edges:
                        continue
                    edges[edge_id]["order"] = min(6, vein_order + 1)
                    visited_edges.add(edge_id)
                    queue.append((neighbour, min(6, vein_order + 1)))
            for edge in edges:
                if edge["order"] is None:
                    edge["order"] = 6
            main_node_set = set(main_nodes)
            for node in nodes:
                if int(node["id"]) == primary_root_node_id:
                    node["role"] = "primary_root"
                elif int(node["id"]) in main_node_set:
                    node["role"] = "main_vein"
                elif int(node["id"]) in primary_node_set:
                    node["role"] = "primary_vein"
                else:
                    node["role"] = node["type"]
        elif len(main_nodes) >= 2 and main_edges:
            # Complex lobed leaves can lack a clean radial graph junction even
            # though their ridge image is strong.  Keep the graph-selected
            # petiole/root, but replace its jagged continuation with a response
            # trace rather than a fitted geometric curve.
            root_index = next(
                (
                    index
                    for index, node_id in enumerate(main_nodes[1:], start=1)
                    if int(nodes[node_id].get("degree", 0)) >= 3
                ),
                min(len(main_nodes) - 1, max(1, len(main_nodes) // 4)),
            )
            fallback_petiole_nodes = main_nodes[: root_index + 1]
            fallback_petiole_edges = main_edges[:root_index]
            fallback_root = fallback_petiole_nodes[-1]
            direction = (
                float(nodes[fallback_root]["x"]) - float(nodes[fallback_petiole_nodes[0]]["x"]),
                float(nodes[fallback_root]["y"]) - float(nodes[fallback_petiole_nodes[0]]["y"]),
            )
            petiole_points: list[list[float]] = []
            for index, edge_id in enumerate(fallback_petiole_edges):
                points = [list(map(float, point)) for point in edges[edge_id].get("path", [])]
                source = nodes[fallback_petiole_nodes[index]]
                if len(points) >= 2 and math.hypot(
                    points[-1][0] - float(source["x"]),
                    points[-1][1] - float(source["y"]),
                ) < math.hypot(
                    points[0][0] - float(source["x"]),
                    points[0][1] - float(source["y"]),
                ):
                    points.reverse()
                if petiole_points and points:
                    points = points[1:]
                petiole_points.extend(points)
            ridge = _trace_response_ridge(
                (float(nodes[fallback_root]["x"]), float(nodes[fallback_root]["y"])),
                direction,
                vessel_score,
                binary,
                leaf_mask,
            )
            if len(ridge) >= 2:
                main_pixel_path = petiole_points + (ridge[1:] if petiole_points else ridge)
                primary_root_node_id = fallback_root
                petiole_node_id = fallback_petiole_nodes[0]
                path_source = "vessel_response_fallback"

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

    if len(main_pixel_path) >= 2:
        path_length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(main_pixel_path, main_pixel_path[1:])
        )
        chord = math.hypot(
            main_pixel_path[-1][0] - main_pixel_path[0][0],
            main_pixel_path[-1][1] - main_pixel_path[0][1],
        )
    else:
        path_length = sum(float(edges[edge_id]["length_px"]) for edge_id in main_edges)
        if len(main_nodes) >= 2:
            first, last = nodes[main_nodes[0]], nodes[main_nodes[-1]]
            chord = math.hypot(last["x"] - first["x"], last["y"] - first["y"])
        else:
            chord = 0.0
    straightness = chord / max(1.0, path_length)
    mean_main_width = float(np.mean([edges[item]["mean_width_px"] for item in main_edges])) if main_edges else 0.0
    main_response = (
        sum(
            float(edges[item].get("response_mean", 0.0))
            * float(edges[item].get("length_px", 0.0))
            for item in main_edges
        )
        / max(1.0, path_length)
    )
    main_support = (
        sum(
            float(edges[item].get("response_support", 0.0))
            * float(edges[item].get("length_px", 0.0))
            for item in main_edges
        )
        / max(1.0, path_length)
    )
    if main_pixel_path:
        ridge_values = np.asarray(
            [
                float(vessel_score[
                    max(0, min(height - 1, round(point[1]))),
                    max(0, min(width - 1, round(point[0]))),
                ])
                for point in main_pixel_path
            ],
            dtype=np.float32,
        )
        main_response = float(np.mean(ridge_values) / 255.0) if ridge_values.size else 0.0
        main_support = float(np.mean(ridge_values >= 72.0)) if ridge_values.size else 0.0
    early_anchors = [item for item in anchors if item["main_position"] <= 0.52]
    basal_branches = len(primary_paths) if primary_paths else max(
        (len(item["branch_edge_ids"]) for item in early_anchors), default=0
    )
    venation_type = "palmate" if basal_branches >= 3 else "pinnate"
    width_confidence = min(1.0, mean_main_width / 9.0)
    confidence = float(
        np.clip(
            100.0
            * (
                main_response * 0.46
                + main_support * 0.30
                + width_confidence * 0.14
                + min(1.0, straightness) * 0.10
            ),
            0,
            99,
        )
    )

    return {
        "version": "leaf-surface-graph-0.8",
        "coordinate_system": "analysis_pixels_top_left_origin",
        "width": int(width),
        "height": int(height),
        "nodes": nodes,
        "edges": edges,
        "main_path": {
            "node_ids": main_nodes,
            "edge_ids": main_edges,
            "pixel_path": main_pixel_path,
        },
        "primary_root_node_id": primary_root_node_id,
        "petiole_node_id": petiole_node_id,
        "primary_paths": primary_paths,
        "specimen_component_edge_ids": specimen_component_edge_ids,
        "venation": {
            "type": venation_type,
            "basal_branch_count": int(basal_branches),
            "main_path_confidence": round(confidence, 1),
            "main_path_straightness": round(straightness, 4),
            "main_path_response": round(main_response, 4),
            "main_path_pixel_support": round(main_support, 4),
            "regularized": regularized,
            "path_source": path_source,
        },
        "secondary_anchors": anchors,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "main_edge_count": len(main_edges),
            "secondary_anchor_count": len(anchors),
            "venation_type": venation_type,
            "main_path_confidence": round(confidence, 1),
        },
    }


def _make_primary_overlay(image: np.ndarray, leaf_mask: np.ndarray, graph: dict[str, Any]) -> np.ndarray:
    overlay = image.copy()
    pixel_path = graph.get("main_path", {}).get("pixel_path", [])
    if len(pixel_path) >= 2:
        points = np.asarray(pixel_path, dtype=np.float32).round().astype(np.int32)
        mean_width = float(
            np.mean(
                [
                    graph["edges"][edge_id].get("mean_width_px", 2.0)
                    for edge_id in graph.get("main_path", {}).get("edge_ids", [])
                ]
                or [2.0]
            )
        )
        cv2.polylines(
            overlay,
            [points],
            False,
            (56, 106, 68),
            max(2, round(mean_width * 0.58)),
            cv2.LINE_AA,
        )
    else:
        for edge_id in graph.get("main_path", {}).get("edge_ids", []):
            edge = graph["edges"][edge_id]
            points = np.asarray(edge.get("path", []), dtype=np.float32).round().astype(np.int32)
            if len(points) >= 2:
                cv2.polylines(overlay, [points], False, (56, 106, 68), max(2, round(edge["mean_width_px"] * 0.75)), cv2.LINE_AA)
    root_id = graph.get("primary_root_node_id")
    if root_id is not None:
        node = graph["nodes"][int(root_id)]
        cv2.circle(overlay, (round(node["x"]), round(node["y"])), 5, (142, 191, 142), -1, cv2.LINE_AA)
    return np.dstack([overlay, leaf_mask])


def analyze_image(path: Path, settings: AnalysisSettings) -> dict[str, Any]:
    image, leaf_mask, resize_ratio, original_size = _prepare_image(path, settings.max_side)
    binary, skeleton, score_preview = _extract_veins(image, leaf_mask, settings)
    metrics, endpoint_mask, junction_mask = _network_metrics(skeleton, binary, leaf_mask)
    graph = _graph_payload(
        skeleton,
        binary,
        leaf_mask,
        score_preview,
        endpoint_mask,
        junction_mask,
    )
    overlay = _make_overlay(image, leaf_mask, skeleton, junction_mask)
    heatmap = _make_heatmap(image, leaf_mask, skeleton, junction_mask)
    primary = _make_primary_overlay(image, leaf_mask, graph)

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
        "graph": graph,
        "images": {
            "original": original_rgba,
            "vessel_score": score_rgba,
            "mask": mask_rgba,
            "skeleton": skeleton_rgba,
            "overlay": overlay,
            "primary": primary,
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
