from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class NmnsIsolation:
    image: np.ndarray
    mask: np.ndarray
    crop_box: tuple[int, int, int, int]
    confidence: float
    method: str


def _largest_scored_component(candidate: np.ndarray) -> tuple[np.ndarray, float] | None:
    height, width = candidate.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, 8)
    best_label = 0
    best_score = float("-inf")
    best_confidence = 0.0
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        if area < height * width * 0.006 or w < width * 0.08 or h < height * 0.08:
            continue
        box_area = max(1, w * h)
        fill = area / box_area
        area_ratio = area / max(1, height * width)
        touches = int(x <= 2) + int(y <= 2) + int(x + w >= width - 2) + int(y + h >= height - 2)
        cx, cy = centroids[label]
        centre_distance = ((cx / width - 0.5) ** 2 + (cy / height - 0.55) ** 2) ** 0.5
        aspect = max(w / max(1, h), h / max(1, w))
        score = area_ratio * 12.0 + min(fill, 0.72) * 1.8 - touches * 1.8
        score -= centre_distance * 0.7 + max(0.0, aspect - 7.0) * 0.18
        if score > best_score:
            best_score = score
            best_label = label
            best_confidence = float(np.clip(45 + area_ratio * 110 + fill * 28 - touches * 25, 0, 99))
    if not best_label:
        return None
    return np.where(labels == best_label, 255, 0).astype(np.uint8), best_confidence


def isolate_nmns_scan(raw: bytes, max_side: int = 1800) -> NmnsIsolation:
    """Isolate a dyed cleared leaf from an NMNS plate or a clean specimen image.

    Black plate borders, rulers and labels are rejected before the vein extractor sees
    the image. The returned crop keeps the source colour and adds a real alpha mask.
    """
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("The NMNS scan could not be decoded.")
    if decoded.ndim == 2:
        bgr = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
        alpha = np.full(decoded.shape, 255, np.uint8)
    elif decoded.shape[2] == 4:
        bgr = decoded[:, :, :3]
        alpha = decoded[:, :, 3]
    else:
        bgr = decoded[:, :, :3]
        alpha = np.full(decoded.shape[:2], 255, np.uint8)

    original_height, original_width = bgr.shape[:2]
    ratio = min(1.0, max_side / max(original_height, original_width))
    if ratio < 1.0:
        size = (max(1, round(original_width * ratio)), max(1, round(original_height * ratio)))
        work = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        work_alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    else:
        work = bgr.copy()
        work_alpha = alpha.copy()

    meaningful_alpha = bool(np.min(work_alpha) < 245 and np.mean(work_alpha > 245) < 0.985)
    method = "alpha"
    confidence = 98.0
    if meaningful_alpha:
        candidate = np.where(work_alpha > 20, 255, 0).astype(np.uint8)
        selected = _largest_scored_component(candidate)
        mask = selected[0] if selected else candidate
    else:
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(work.astype(np.int16))
        maximum = np.maximum(np.maximum(r, g), b)
        minimum = np.minimum(np.minimum(r, g), b)
        chroma = maximum - minimum
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        red_green_bias = np.maximum(r - (g + b) / 2.0, g - (r + b) / 2.0)

        # Cleared leaves in the NMNS collection are usually dyed red, pink,
        # green or brown. Neutral black typography and the ruler are excluded.
        colour = (
            (value > 78)
            & (saturation > 25)
            & (chroma > 22)
            & ((red_green_bias > 17) | (saturation > 48))
        ).astype(np.uint8) * 255
        scale = max(5, round(min(work.shape[:2]) * 0.009))
        if scale % 2 == 0:
            scale += 1
        colour = cv2.morphologyEx(
            colour,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale)),
        )
        selected = _largest_scored_component(colour)
        if selected is not None:
            seed, confidence = selected
            method = "chromatic-plate"
            # The coloured tissue/veins locate the specimen. Filling its outer
            # contour restores transparent intercostal areas without importing text.
            seed = cv2.morphologyEx(
                seed,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale * 2 + 1, scale * 2 + 1)),
            )
            contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros(seed.shape, np.uint8)
            if contours:
                cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
            else:
                mask = seed
        else:
            # Fallback for an already-clean neutral scan: estimate the matte from
            # the image border and reject implausibly tiny or full-page regions.
            lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
            margin = max(3, min(work.shape[:2]) // 70)
            border = np.concatenate(
                (lab[:margin].reshape(-1, 3), lab[-margin:].reshape(-1, 3),
                 lab[:, :margin].reshape(-1, 3), lab[:, -margin:].reshape(-1, 3)), axis=0
            )
            background = np.median(border, axis=0)
            distance = np.linalg.norm(lab - background, axis=2)
            threshold = max(10.0, float(np.percentile(np.linalg.norm(border - background, axis=1), 97)) * 1.7)
            matte = np.where(distance > threshold, 255, 0).astype(np.uint8)
            matte = cv2.morphologyEx(matte, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
            selected = _largest_scored_component(matte)
            if selected is None:
                raise ValueError("The leaf could not be separated from the NMNS plate.")
            mask, confidence = selected
            method = "matte-fallback"
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
            mask = filled

    if ratio < 1.0:
        mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # Remove ruler/frame slivers that may touch a leaf edge. The thick lamina
    # defines the safe horizontal span; a long petiole is retained because it
    # remains inside that span.
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    core = np.where(distance > max(7.0, min(original_width, original_height) * 0.006), 255, 0).astype(np.uint8)
    core_selected = _largest_scored_component(core)
    if core_selected is not None:
        core_ys, core_xs = np.where(core_selected[0] > 0)
        horizontal_pad = max(20, round(original_width * 0.020))
        left = max(0, int(core_xs.min()) - horizontal_pad)
        right = min(original_width, int(core_xs.max()) + horizontal_pad + 1)
        mask[:, :left] = 0
        mask[:, right:] = 0

    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("The leaf mask is empty.")
    padding = max(12, round(max(original_width, original_height) * 0.018))
    x0, x1 = max(0, int(xs.min()) - padding), min(original_width, int(xs.max()) + padding + 1)
    y0, y1 = max(0, int(ys.min()) - padding), min(original_height, int(ys.max()) + padding + 1)
    crop_mask = mask[y0:y1, x0:x1]
    crop_bgr = bgr[y0:y1, x0:x1]
    rgba = np.dstack((crop_bgr, crop_mask))
    return NmnsIsolation(rgba, crop_mask, (x0, y0, x1, y1), round(confidence, 1), method)
