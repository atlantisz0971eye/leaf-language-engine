from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from contour_color_codec import (
    _cell_geometry,
    _cell_masks,
    _jpeg_leaf_mask,
    _read_colour_bit,
    _read_header,
    decode_contour_colour,
    encode_contour_colour,
)


def _bounded(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _source_files(specimen_dir: Path) -> tuple[Path, Path | None]:
    choices = (
        specimen_dir / "rubbing_clean.png",
        specimen_dir / "analysis_input.png",
        specimen_dir / "rubbing_rgba.png",
    )
    source = next((item for item in choices if item.exists()), None)
    if source is None:
        raise FileNotFoundError(f"No rubbing carrier was found in {specimen_dir}")
    silhouette = specimen_dir / "silhouette.png"
    return source, silhouette if silhouette.exists() else None


def _carrier_axes(cells: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray([cell["center"] for cell in cells], np.float32)
    centroid = centers.mean(axis=0)
    centered = centers - centroid
    values, vectors = np.linalg.eigh(np.cov(centered.T))
    major = vectors[:, int(np.argmax(values))]
    if major[0] < 0:
        major = -major
    minor = np.asarray((-major[1], major[0]), np.float32)
    projection = centered @ major
    secondary = centered @ minor
    return centroid, major, projection, secondary


def _audio_field(
    cells: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    levels: int,
    config: dict[str, Any],
) -> np.ndarray:
    if not frames:
        return np.zeros(len(cells), np.float32)
    maximum = max(1, int(levels) - 1)
    centroid, major, projection, secondary = _carrier_axes(cells)
    projection_min = float(projection.min(initial=0.0))
    projection_span = max(1.0, float(projection.max(initial=1.0)) - projection_min)
    secondary_span = max(1.0, float(np.max(np.abs(secondary), initial=1.0)))
    normalized_position = np.clip((projection - projection_min) / projection_span, 0.0, 1.0)
    strength = _bounded(config.get("strength"), 0.25)
    clustering = _bounded(config.get("clustering"), 0.55)
    severity = np.zeros(len(cells), np.float32)

    for index, position in enumerate(normalized_position):
        frame = frames[min(len(frames) - 1, int(round(position * (len(frames) - 1))))]
        energy = _bounded(float(frame.get("energy", 0)) / maximum)
        peak = _bounded(float(frame.get("peak", 0)) / maximum)
        noise = _bounded(float(frame.get("noise", 0)) / maximum)
        severity[index] = strength * (0.06 + energy * 0.38 + peak * 0.16 + noise * 0.12)

    impact_frames = [(index, frame) for index, frame in enumerate(frames) if int(frame.get("impact", 0)) > 0]
    if len(impact_frames) > 160:
        impact_frames = sorted(impact_frames, key=lambda item: int(item[1].get("impact", 0)), reverse=True)[:160]
    centers = np.asarray([cell["center"] for cell in cells], np.float32)
    spacing = float(np.median([cell["radius"] for cell in cells])) / 0.43
    for frame_index, frame in impact_frames:
        time_position = frame_index / max(1, len(frames) - 1)
        peak = _bounded(float(frame.get("peak", 0)) / maximum)
        impact = _bounded(float(frame.get("impact", 0)) / maximum)
        brightness = _bounded(float(frame.get("brightness", 0)) / maximum)
        noise = _bounded(float(frame.get("noise", 0)) / maximum)
        primary = projection_min + time_position * projection_span
        secondary_position = (brightness - 0.5) * secondary_span * 1.15
        center = centroid + major * primary + np.asarray((-major[1], major[0])) * secondary_position
        radius = spacing * (2.0 + peak * 6.0 + noise * 2.2) * (0.55 + clustering * 0.75)
        distance_squared = np.sum((centers - center[None, :]) ** 2, axis=1)
        pulse = np.exp(-distance_squared / max(1.0, 2.0 * radius * radius))
        severity = np.maximum(severity, strength * impact * (0.42 + clustering * 0.95) * pulse)
    return np.clip(severity, 0.0, 1.0)


def _modify_cell(
    lab: np.ndarray,
    cell: dict[str, Any],
    severity: float,
    operation: str,
) -> None:
    y0, y1, x0, x1 = cell["roi"]
    region = lab[y0:y1, x0:x1]
    left, right, _ = _cell_masks(cell)
    channel = region[..., 1]
    left_median = float(np.median(channel[left]))
    right_median = float(np.median(channel[right]))
    midpoint = (left_median + right_median) * 0.5
    if operation == "erasure":
        factor = max(0.015, 0.12 * (1.0 - severity))
        channel[left] = np.clip(midpoint + (channel[left] - midpoint) * factor, 0, 255)
        channel[right] = np.clip(midpoint + (channel[right] - midpoint) * factor, 0, 255)
    elif operation == "flip":
        difference = right_median - left_median
        displacement = max(8.0, abs(difference) * (0.58 + severity * 0.72))
        sign = 1.0 if difference >= 0 else -1.0
        channel[left] = np.clip(channel[left] + sign * displacement, 0, 255)
        channel[right] = np.clip(channel[right] - sign * displacement, 0, 255)
    else:
        factor = max(0.06, 1.0 - severity * 0.92)
        channel[left] = np.clip(midpoint + (channel[left] - midpoint) * factor, 0, 255)
        channel[right] = np.clip(midpoint + (channel[right] - midpoint) * factor, 0, 255)


def _diagnostic_images(
    clean: np.ndarray,
    disturbed: np.ndarray,
    cells: list[dict[str, Any]],
    operations: list[str],
    confidences: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    difference = np.clip(clean.astype(np.float32) * 0.18, 0, 255).astype(np.uint8)
    confidence = difference.copy()
    operation_colours = {
        "shift": np.asarray((46, 126, 218), np.uint8),
        "erasure": np.asarray((73, 75, 224), np.uint8),
        "flip": np.asarray((116, 63, 223), np.uint8),
        "none": np.asarray((74, 126, 74), np.uint8),
    }
    for cell, operation, score in zip(cells, operations, confidences):
        y0, y1, x0, x1 = cell["roi"]
        left, right, _ = _cell_masks(cell)
        active = left | right
        if operation != "none":
            region = difference[y0:y1, x0:x1]
            colour = operation_colours[operation]
            region[active] = np.clip(region[active].astype(np.float32) * 0.25 + colour * 0.75, 0, 255).astype(np.uint8)
        confidence_region = confidence[y0:y1, x0:x1]
        if score >= 0.66:
            colour = np.asarray((82, 160, 82), np.uint8)
        elif score >= 0.32:
            colour = np.asarray((48, 176, 218), np.uint8)
        else:
            colour = np.asarray((74, 72, 224), np.uint8)
        confidence_region[active] = np.clip(confidence_region[active].astype(np.float32) * 0.2 + colour * 0.8, 0, 255).astype(np.uint8)
    return difference, confidence


def run_leaf_experiment(
    specimen_dir: Path,
    text: str,
    frames: list[dict[str, Any]],
    levels: int,
    config: dict[str, Any],
    output_dir: Path,
    target_long_side: int = 2048,
    clean_carrier_path: Path | None = None,
) -> dict[str, Any]:
    source, silhouette = _source_files(specimen_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_output = output_dir / "original.png"
    clean_output = output_dir / "clean.png"
    disturbed_output = output_dir / "disturbed.png"
    difference_output = output_dir / "difference.png"
    confidence_output = output_dir / "confidence.png"
    shutil.copy2(source, original_output)

    resolution = max(1536, min(8192, int(target_long_side)))
    if clean_carrier_path is not None:
        clean = cv2.imread(str(clean_carrier_path), cv2.IMREAD_COLOR)
        if clean is None:
            raise OSError("The prepared colour carrier could not be read.")
        if not cv2.imwrite(str(clean_output), clean, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
            raise OSError("The clean carrier monitor could not be saved.")
        clean_metadata = {
            "payload": {"text_bytes": len(text.encode("utf-8"))},
            "protocol": {"source": "prepared-main-site-carrier"},
        }
    else:
        while True:
            try:
                clean_metadata = encode_contour_colour(
                    source_path=source,
                    text=text,
                    output_path=clean_output,
                    silhouette_path=silhouette,
                    target_long_side=resolution,
                    jpeg_quality=96,
                    cell_spacing=14,
                    chroma_delta=28.0,
                )
                break
            except ValueError as exc:
                match = re.search(r"Use at least (\d+)px", str(exc))
                recommended = int(match.group(1)) if match else 0
                if recommended <= resolution or recommended > 8192:
                    raise
                resolution = recommended
        clean = cv2.imread(str(clean_output), cv2.IMREAD_COLOR)
    if clean is None:
        raise OSError("The clean colour carrier was not generated.")
    mask = _jpeg_leaf_mask(clean)
    header, sync = _read_header(clean, mask)
    lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
    seed = (int(header["body_length"]) << 16) ^ int(header["encoded_bits"]) ^ 0xC0110A
    cells = _cell_geometry(mask, lab, int(header["cell_spacing"]), seed)
    severity = _audio_field(cells, frames, levels, config)
    random_seed = int(hashlib.sha256(f"{seed}|{len(frames)}|ADCL-0.2".encode()).hexdigest()[:8], 16)
    random = np.random.default_rng(random_seed)
    erasure_weight = _bounded(config.get("erasure"), 0.12)
    flip_weight = _bounded(config.get("flip"), 0.05)
    shift_weight = _bounded(config.get("shift"), 0.4)
    operations: list[str] = []
    for cell, value in zip(cells, severity):
        roll = float(random.random())
        erasure_probability = float(value) * erasure_weight * 0.55
        flip_probability = float(value) * flip_weight * 0.42
        shift_probability = float(value) * shift_weight * 0.86
        if roll < erasure_probability:
            operation = "erasure"
        elif roll < erasure_probability + flip_probability:
            operation = "flip"
        elif roll < erasure_probability + flip_probability + shift_probability:
            operation = "shift"
        else:
            operation = "none"
        operations.append(operation)
        if operation != "none":
            _modify_cell(lab, cell, float(value), operation)
    disturbed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    disturbed[mask == 0] = clean[mask == 0]
    if not cv2.imwrite(str(disturbed_output), disturbed, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise OSError("Could not save the disturbed leaf carrier.")

    disturbed_lab = cv2.cvtColor(disturbed, cv2.COLOR_BGR2LAB)
    confidences = [_read_colour_bit(disturbed_lab, cell)[1] for cell in cells]
    difference, confidence = _diagnostic_images(clean, disturbed, cells, operations, confidences)
    cv2.imwrite(str(difference_output), difference, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    cv2.imwrite(str(confidence_output), confidence, [cv2.IMWRITE_PNG_COMPRESSION, 4])

    clean_decode = decode_contour_colour(clean_output)
    disturbed_decode: dict[str, Any] | None = None
    decode_error: str | None = None
    try:
        disturbed_decode = decode_contour_colour(disturbed_output)
    except Exception as exc:  # diagnostic tool: surface decoder failure verbatim
        decode_error = f"{type(exc).__name__}: {exc}"

    counts = {name: operations.count(name) for name in ("none", "shift", "erasure", "flip")}
    result = {
        "version": "ADCL-0.2",
        "specimen": specimen_dir.name,
        "source": source.name,
        "image_size": [int(clean.shape[1]), int(clean.shape[0])],
        "carrier": {
            "recognized_cells": len(cells),
            "changed_cells": len(cells) - counts["none"],
            "operations": counts,
            "mean_confidence": round(float(np.mean(confidences)) * 100.0, 2),
            "low_confidence_cells": int(sum(value < 0.32 for value in confidences)),
            "protocol_sync": sync,
        },
        "clean_decode": clean_decode,
        "disturbed_decode": disturbed_decode,
        "decode_error": decode_error,
        "clean_metadata": {
            "payload": clean_metadata.get("payload", {}),
            "protocol": clean_metadata.get("protocol", {}),
        },
        "outputs": {
            "original": "original.png",
            "clean": "clean.png",
            "disturbed": "disturbed.png",
            "difference": "difference.png",
            "confidence": "confidence.png",
        },
    }
    (output_dir / "experiment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
