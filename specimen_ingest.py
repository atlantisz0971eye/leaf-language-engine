from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class LeafCandidate:
    contour: np.ndarray
    area: float
    box: tuple[int, int, int, int]
    border_sides: int
    foreground_density: float
    mean_saturation: float
    solidity: float
    score: float


def _kernel(size: int) -> np.ndarray:
    size = max(3, int(size) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == selected, 255, 0).astype(np.uint8)


def _filled_mask(shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    return mask


def _meaningful_alpha(image: np.ndarray) -> bool:
    if image.ndim != 3 or image.shape[2] != 4:
        return False
    alpha = image[..., 3]
    return bool(np.min(alpha) < 245 and np.mean(alpha > 245) < 0.995)


def _border_pixels(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    edge = max(3, min(height, width) // 90)
    return np.concatenate(
        [
            image[:edge].reshape(-1, image.shape[2]),
            image[-edge:].reshape(-1, image.shape[2]),
            image[:, :edge].reshape(-1, image.shape[2]),
            image[:, -edge:].reshape(-1, image.shape[2]),
        ],
        axis=0,
    )


def _opaque_candidates(bgr: np.ndarray) -> tuple[list[LeafCandidate], dict[str, float]]:
    height, width = bgr.shape[:2]
    scale = min(1.0, 1400.0 / max(height, width))
    small = cv2.resize(
        bgr,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    blurred = cv2.GaussianBlur(small, (0, 0), 2.4)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    border = _border_pixels(lab)
    background = np.median(border, axis=0)
    border_distance = np.linalg.norm(border - background, axis=1)
    distance = np.linalg.norm(lab - background, axis=2)
    distance_threshold = max(7.0, float(np.percentile(border_distance, 99.2)) * 2.15)

    saturation = hsv[..., 1]
    value = hsv[..., 2]
    foreground = np.where(
        (distance > distance_threshold)
        | ((saturation > 20) & (value < 250))
        | (gray < max(145, int(np.median(gray) - 38))),
        255,
        0,
    ).astype(np.uint8)
    base = max(3, round(min(small.shape[:2]) / 310))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, _kernel(base))
    merged = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, _kernel(base * 7))
    merged = cv2.dilate(merged, _kernel(base * 3))

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(small.shape[0] * small.shape[1])
    candidates: list[LeafCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.006 or area > image_area * 0.94:
            continue
        x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
        if min(candidate_width, candidate_height) < 45:
            continue
        local_mask = _filled_mask(small.shape[:2], contour) > 0
        foreground_density = float(np.mean(foreground[local_mask] > 0))
        mean_saturation = float(np.mean(saturation[local_mask]))
        hull = cv2.convexHull(contour)
        solidity = area / max(1.0, float(cv2.contourArea(hull)))
        border_sides = sum(
            (
                x <= 3,
                y <= 3,
                x + candidate_width >= small.shape[1] - 3,
                y + candidate_height >= small.shape[0] - 3,
            )
        )
        area_ratio = area / image_area
        shape_score = min(1.0, solidity / 0.72)
        size_score = min(1.0, area_ratio / 0.10)
        border_score = 1.0 - min(1.0, border_sides / 2.0)
        score = 1.7 * shape_score + 1.2 * size_score + 0.8 * border_score
        candidates.append(
            LeafCandidate(
                contour=contour,
                area=area,
                box=(x, y, candidate_width, candidate_height),
                border_sides=border_sides,
                foreground_density=foreground_density,
                mean_saturation=mean_saturation,
                solidity=solidity,
                score=score,
            )
        )
    candidates.sort(key=lambda item: (item.score, item.area), reverse=True)
    return candidates, {
        "analysis_scale": scale,
        "border_variation": round(float(np.median(border_distance)), 3),
        "border_noise_p99": round(float(np.percentile(border_distance, 99)), 3),
        "background_distance_threshold": round(distance_threshold, 3),
    }


def _select_opaque_candidate(
    candidates: list[LeafCandidate], source_hint: str = "auto"
) -> tuple[LeafCandidate, str]:
    if not candidates:
        raise ValueError(
            "No closed leaf-like region was found. Use a clearer background or crop closer to one leaf."
        )
    usable = [item for item in candidates if item.border_sides <= 1] or candidates
    if source_hint == "nmns":
        return max(usable, key=lambda item: (item.area, item.score)), "nmns-largest-leaf"
    major = [item for item in usable if item.area >= max(candidate.area for candidate in usable) * 0.22]
    if len(major) >= 2:
        # Paired rubbing photographs usually contain one solid reference leaf
        # and one lighter, discontinuous print. Prefer the latter while still
        # requiring a plausible closed leaf shape.
        selected = max(
            major,
            key=lambda item: (
                item.score
                + 1.25 * (1.0 - min(1.0, item.foreground_density))
                + 0.35 * (1.0 - item.mean_saturation / 255.0)
            ),
        )
        return selected, "paired-rubbing-texture"
    return usable[0], "largest-leaf-region"


def _crop_bounds(mask: np.ndarray, padding_fraction: float = 0.055) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("The detected leaf mask is empty.")
    x, y, width, height = cv2.boundingRect(points)
    padding = max(12, round(max(width, height) * padding_fraction))
    return (
        max(0, x - padding),
        max(0, y - padding),
        min(mask.shape[1], x + width + padding),
        min(mask.shape[0], y + height + padding),
    )


def _texture_metrics(clean: np.ndarray, mask: np.ndarray) -> dict[str, float | int | bool]:
    valid = mask > 0
    area = int(np.count_nonzero(valid))
    if area <= 0:
        raise ValueError("The detected leaf mask is empty.")
    lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
    texture = valid & ((lab[..., 0] < 238) | (hsv[..., 1] > 12))
    texture_pixels = int(np.count_nonzero(texture))
    texture_coverage = texture_pixels / area
    leaf_values = lab[..., 0][valid].astype(np.float32)
    luminance_spread = float(np.percentile(leaf_values, 90) - np.percentile(leaf_values, 10))
    contour = max(
        cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        key=cv2.contourArea,
    )
    perimeter = float(cv2.arcLength(contour, True))
    contour_area = float(cv2.contourArea(contour))
    compactness = 4.0 * math.pi * contour_area / max(1.0, perimeter * perimeter)
    minimum_dimension = min(clean.shape[:2])
    ready = bool(
        area >= 12_000
        and minimum_dimension >= 128
        and texture_coverage >= 0.025
        and perimeter >= 220
    )
    texture_score = min(1.0, texture_coverage / 0.28)
    resolution_score = min(1.0, minimum_dimension / 650.0)
    geometry_score = min(1.0, max(0.0, compactness) / 0.42)
    quality = round(100.0 * (0.50 * texture_score + 0.30 * resolution_score + 0.20 * geometry_score), 1)
    return {
        "leaf_pixels": area,
        "texture_pixels": texture_pixels,
        "texture_coverage": round(texture_coverage, 4),
        "luminance_spread": round(luminance_spread, 2),
        "contour_perimeter": round(perimeter, 2),
        "compactness": round(compactness, 4),
        "quality_score": quality,
        "encode_ready": ready,
        "review_recommended": bool(not ready or quality < 48.0),
    }


def ingest_specimen(
    raw: bytes,
    filename: str,
    output_root: Path,
    source_hint: str = "auto",
) -> dict[str, Any]:
    source_hint = source_hint if source_hint in {"auto", "rubbing", "nmns"} else "auto"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Upload a JPG, JPEG, or PNG leaf image.")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in {2, 3}:
        raise ValueError("The uploaded image could not be decoded.")
    if image.shape[0] < 128 or image.shape[1] < 128:
        raise ValueError("The leaf image must be at least 128 × 128 pixels.")

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image[..., :3].copy()
    source_type: str
    selection_method: str
    analysis: dict[str, Any] = {}

    if _meaningful_alpha(image):
        alpha = image[..., 3]
        mask = np.where(alpha > 14, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(max(5, min(mask.shape) // 240)))
        mask = _largest_component(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("The PNG alpha channel does not contain a leaf silhouette.")
        mask = _filled_mask(mask.shape, max(contours, key=cv2.contourArea))
        source_type = "nmns-transparent" if suffix == ".png" else "transparent-leaf"
        selection_method = "alpha-silhouette"
    else:
        candidates, analysis = _opaque_candidates(bgr)
        selected, selection_method = _select_opaque_candidate(candidates, source_hint)
        scale = float(analysis["analysis_scale"])
        contour = np.rint(selected.contour.astype(np.float32) / max(scale, 1e-9)).astype(np.int32)
        mask = _filled_mask(bgr.shape[:2], contour)
        background_is_flat = float(analysis["border_noise_p99"]) < 8.0
        if selection_method == "paired-rubbing-texture" or not background_is_flat:
            source_type = "rubbing-photo"
        else:
            source_type = "opaque-cleared-leaf"
        analysis.update(
            {
                "candidate_count": len(candidates),
                "selected_box_small": list(selected.box),
                "selected_foreground_density": round(selected.foreground_density, 4),
                "selected_mean_saturation": round(selected.mean_saturation, 3),
                "selected_solidity": round(selected.solidity, 4),
            }
        )

    x0, y0, x1, y1 = _crop_bounds(mask)
    crop = bgr[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1].copy()
    clean = np.full_like(crop, 255)
    clean[crop_mask > 0] = crop[crop_mask > 0]
    metrics = _texture_metrics(clean, crop_mask)

    digest = hashlib.sha256(raw).hexdigest()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_") or "leaf"
    identifier = f"upload_{stem[:32]}_{digest[:10]}"
    destination = output_root / identifier
    destination.mkdir(parents=True, exist_ok=True)
    original_name = f"source_original{suffix}"
    (destination / original_name).write_bytes(raw)
    if not cv2.imwrite(str(destination / "rubbing_clean.png"), clean):
        raise OSError("Could not save the preserved colour carrier.")
    if not cv2.imwrite(str(destination / "silhouette.png"), crop_mask):
        raise OSError("Could not save the leaf silhouette.")

    metadata = {
        "version": "unified-specimen-ingest-0.1",
        "id": identifier,
        "display_name": Path(filename).stem,
        "source_filename": Path(filename).name,
        "source_sha256": digest,
        "source_type": source_type,
        "source_hint": source_hint,
        "selection_method": selection_method,
        "source_size": [int(bgr.shape[1]), int(bgr.shape[0])],
        "crop_box": [x0, y0, x1, y1],
        "carrier_size": [int(clean.shape[1]), int(clean.shape[0])],
        "analysis": analysis,
        **metrics,
        "outputs": {
            "original": original_name,
            "carrier": "rubbing_clean.png",
            "silhouette": "silhouette.png",
        },
        "texture_policy": {
            "interior_geometry_added": False,
            "source_pixels_preserved_inside_mask": True,
            "outside_leaf_replaced_with_white": True,
        },
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
