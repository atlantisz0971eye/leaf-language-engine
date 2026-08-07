from __future__ import annotations

import argparse
import itertools
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from contour_texture_codec import (
    _bits_to_bytes,
    _bytes_to_bits,
    _crc16,
    _hamming_decode,
    _hamming_encode,
    _largest_contour,
    _mask_from_image,
    _resample_closed,
)


VERSION = 3
HEADER_BYTES = 5
HEADER_REPETITIONS = 7
HEADER_RAW_BITS = HEADER_BYTES * 8
HEADER_CODE_BITS = HEADER_RAW_BITS

# The outline uses two natural leaf colours with similar brightness. The
# payload uses the source pixels themselves and changes only their Lab a-axis.
PROTOCOL_ZERO = np.asarray((72, 126, 69), dtype=np.float32)   # cool green, BGR
PROTOCOL_ONE = np.asarray((62, 139, 146), dtype=np.float32)  # warm olive, BGR


def _pack_header(
    compressed: bool,
    body_length: int,
    cell_spacing: int,
    address_salt: int = 0,
    version: int = VERSION,
) -> bytes:
    if not 0 < body_length <= 0xFFF:
        raise ValueError("The colour payload body must fit in 12 header bits.")
    spacing_base = 12 if version >= 3 else 16
    if not spacing_base <= cell_spacing <= spacing_base + 63:
        raise ValueError("Colour-cell spacing is outside the protocol range.")
    value = (
        (body_length << 12)
        | ((cell_spacing - spacing_base) << 6)
        | ((version & 0x07) << 3)
        | ((1 if compressed else 0) << 2)
        | (int(address_salt) & 0x03)
    )
    prefix = value.to_bytes(3, "big")
    return prefix + struct.pack(">H", _crc16(prefix))


def _unpack_header(raw: bytes) -> dict[str, int | bool]:
    if len(raw) != 5:
        raise ValueError("The colour protocol header has the wrong length.")
    if _crc16(raw[:3]) != struct.unpack(">H", raw[3:])[0]:
        raise ValueError("The contour header failed its CRC check.")
    value = int.from_bytes(raw[:3], "big")
    body_length = (value >> 12) & 0xFFF
    version = (value >> 3) & 0x07
    if version not in {2, 3}:
        raise ValueError("No supported contour-colour protocol was found.")
    spacing = ((value >> 6) & 0x3F) + (12 if version >= 3 else 16)
    compressed = bool((value >> 2) & 1)
    address_salt = value & 0x03
    if body_length <= 0:
        raise ValueError("No supported contour-colour protocol was found.")
    encoded_bits = len(_hamming_encode([0] * (body_length * 8)))
    return {
        "version": int(version),
        "compressed": compressed,
        "body_length": int(body_length),
        "encoded_bits": int(encoded_bits),
        "cell_spacing": int(spacing),
        "address_salt": int(address_salt),
        "header_crc_valid": True,
    }


def _repair_header(raw: bytes) -> tuple[dict[str, int | bool], int]:
    try:
        return _unpack_header(raw), 0
    except ValueError:
        pass
    bits = _bytes_to_bits(raw)
    # JPG boundary errors normally alter the compact metadata prefix while the
    # repeated CRC bytes remain stable. Searching three prefix flips gives the
    # outline a small list-decoding margin without weakening CRC validation.
    for count in (1, 2, 3):
        for positions in itertools.combinations(range(24), count):
            candidate = bits.copy()
            for position in positions:
                candidate[position] ^= 1
            try:
                return _unpack_header(_bits_to_bytes(candidate, HEADER_RAW_BITS)), count
            except ValueError:
                continue
    raise ValueError("The compact contour header could not be repaired.")


def _source_mask(image: np.ndarray, silhouette: np.ndarray | None) -> np.ndarray:
    return _mask_from_image(image, silhouette)


def _resize_preserving_texture(
    image: np.ndarray, mask: np.ndarray, target_long_side: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    scale = target_long_side / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image[..., :3]
        if image.shape[2] == 4:
            alpha = image[..., 3:4].astype(np.float32) / 255.0
            bgr = np.clip(bgr.astype(np.float32) * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    resized = cv2.resize(bgr, size, interpolation=cv2.INTER_CUBIC)
    resized[resized_mask == 0] = 255
    return resized, resized_mask


def _jpeg_leaf_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coloured = np.where((hsv[..., 1] > 18) | (gray < 238), 255, 0).astype(np.uint8)
    coloured = cv2.morphologyEx(coloured, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(coloured, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("The JPG does not contain a closed leaf protocol boundary.")
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def _protocol_points(mask: np.ndarray, count: int) -> np.ndarray:
    return _resample_closed(_largest_contour(mask), count)


def _draw_protocol(image: np.ndarray, mask: np.ndarray, header_bits: list[int]) -> dict[str, int]:
    # Consecutive repetition turns every header bit into one long colour band.
    # This is substantially more stable than repeating complete bitstreams when
    # the contour recovered from JPG has a slightly different arc-length map.
    repeated = [bit for bit in header_bits for _ in range(HEADER_REPETITIONS)]
    points = _protocol_points(mask, len(repeated))
    width = max(3, round(max(mask.shape) / 430.0))
    for index, bit in enumerate(repeated):
        start = tuple(np.rint(points[index]).astype(int))
        end = tuple(np.rint(points[(index + 1) % len(points)]).astype(int))
        color = PROTOCOL_ONE if bit else PROTOCOL_ZERO
        cv2.line(image, start, end, tuple(int(value) for value in color), width, cv2.LINE_AA)

    # Three short transverse ticks form a visible start locator. They overwrite
    # only one header copy; the other two copies preserve the affected bits.
    centroid = points.mean(axis=0)
    for index, multiplier in zip((0, 2, 5), (1.7, 1.2, 0.8)):
        point = points[index]
        inward = centroid - point
        inward /= max(1e-6, float(np.linalg.norm(inward)))
        tangent = np.asarray((-inward[1], inward[0]), np.float32)
        half = tangent * width * multiplier
        cv2.line(
            image,
            tuple(np.rint(point - half).astype(int)),
            tuple(np.rint(point + half).astype(int)),
            tuple(int(value) for value in PROTOCOL_ONE),
            max(1, width // 2),
            cv2.LINE_AA,
        )
    return {"header_cells": len(repeated), "line_width": width}


def _patch_median(image: np.ndarray, point: np.ndarray, radius: int) -> np.ndarray:
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return np.asarray((255, 255, 255), np.float32)
    return np.median(patch.reshape(-1, patch.shape[-1]), axis=0)


def _protocol_sample(image: np.ndarray, point: np.ndarray, radius: int) -> tuple[int, float]:
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
    pixels = image[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if not len(pixels):
        return 0, 0.0
    distance_zero = np.linalg.norm(pixels - PROTOCOL_ZERO[None, :], axis=1)
    distance_one = np.linalg.norm(pixels - PROTOCOL_ONE[None, :], axis=1)
    zero = float(np.min(distance_zero))
    one = float(np.min(distance_one))
    return int(one < zero), abs(one - zero) / max(1.0, one + zero)


def _read_header(image: np.ndarray, mask: np.ndarray) -> tuple[dict[str, int | bool], dict[str, Any]]:
    header_bits = HEADER_CODE_BITS
    total = header_bits * HEADER_REPETITIONS
    oversampling = 4
    points = _protocol_points(mask, total * oversampling)
    width = max(3, round(max(mask.shape) / 430.0))
    centroid = points.mean(axis=0)
    inset = width * 0.62
    for phase in range(oversampling):
        samples: list[int] = []
        confidences: list[float] = []
        for index in range(total):
            point = points[(index * oversampling + phase) % len(points)].copy()
            inward = centroid - point
            inward /= max(1e-6, float(np.linalg.norm(inward)))
            bit, confidence = _protocol_sample(image, point + inward * inset, width)
            samples.append(bit)
            confidences.append(confidence)

        # The mask recovered from a lossy JPG can change the contour's first
        # point or direction. Magic + CRC search every circular alignment.
        for reverse in (False, True):
            sequence = samples[::-1] if reverse else samples
            for shift in range(total):
                aligned = sequence[shift:] + sequence[:shift]
                voted = [
                    int(
                        sum(
                            aligned[index * HEADER_REPETITIONS + copy]
                            for copy in range(HEADER_REPETITIONS)
                        )
                        >= (HEADER_REPETITIONS // 2 + 1)
                    )
                    for index in range(header_bits)
                ]
                try:
                    header, repaired = _repair_header(_bits_to_bytes(voted, HEADER_RAW_BITS))
                except (ValueError, struct.error):
                    continue
                theoretical_cells = int(
                    np.count_nonzero(mask)
                    * 2.0
                    / max(1, int(header["cell_spacing"]) ** 2)
                )
                if int(header["encoded_bits"]) > theoretical_cells:
                    continue
                return header, {
                    "circular_shift": shift,
                    "reversed": reverse,
                    "sample_phase": phase,
                    "corrected_header_words": 0,
                    "repaired_header_bits": repaired,
                    "confidence": round(float(np.mean(confidences)) * 100.0, 1),
                }
    raise ValueError("The JPG contour was found, but its protocol header could not be synchronized.")


def _cell_geometry(
    mask: np.ndarray,
    lab: np.ndarray,
    spacing: int,
    seed: int,
) -> list[dict[str, Any]]:
    bounds = cv2.boundingRect(mask)
    x0, y0, width, height = (int(value) for value in bounds)
    if width <= 0 or height <= 0:
        return []
    x1, y1 = x0 + width - 1, y0 + height - 1
    # Distance only needs to exist inside the leaf bounding box. Cropping it
    # prevents a mostly-white 8K canvas from allocating another full-size
    # float32 image.
    cropped_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    padded_mask = cv2.copyMakeBorder(cropped_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)[1:-1, 1:-1]
    luminance = lab[..., 0]
    texture = luminance < 236
    radius = spacing * 0.43
    margin = max(radius + 3.0, spacing * 0.72)
    minimum_pixels = max(10, round(spacing * spacing * 0.035))
    candidates: list[dict[str, Any]] = []
    row = 0
    for y in range(y0 + spacing, y1 - spacing + 1, spacing):
        offset = spacing // 2 if row % 2 else 0
        column = 0
        for x in range(x0 + spacing + offset, x1 - spacing + 1, spacing):
            value = (x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)
            jx = ((value & 255) / 255.0 - 0.5) * spacing * 0.26
            jy = (((value >> 8) & 255) / 255.0 - 0.5) * spacing * 0.26
            cx, cy = int(round(x + jx)), int(round(y + jy))
            if not (0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0]) or distance[cy - y0, cx - x0] < margin:
                column += 1
                continue
            angle = (((value >> 16) & 0xFFFF) / 65535.0) * math.pi
            direction = np.asarray((math.cos(angle), math.sin(angle)), np.float32)
            r = int(math.ceil(radius))
            yy0, yy1 = max(0, cy - r), min(mask.shape[0], cy + r + 1)
            xx0, xx1 = max(0, cx - r), min(mask.shape[1], cx + r + 1)
            gy, gx = np.mgrid[yy0:yy1, xx0:xx1]
            dx, dy = gx - cx, gy - cy
            circular = dx * dx + dy * dy <= radius * radius
            active = circular & texture[yy0:yy1, xx0:xx1]
            projection = dx * direction[0] + dy * direction[1]
            left = active & (projection < -spacing * 0.055)
            right = active & (projection > spacing * 0.055)
            if int(left.sum()) >= minimum_pixels and int(right.sum()) >= minimum_pixels:
                order = ((value >> 7) ^ (value * 2654435761)) & 0xFFFFFFFF
                candidates.append(
                    {
                        "order": order,
                        "grid_order": row * 100000 + column,
                        "roi": (yy0, yy1, xx0, xx1),
                        "center": (cx, cy),
                        "direction": (float(direction[0]), float(direction[1])),
                        "radius": radius,
                        "active_shape": active.shape,
                        "active_bits": np.packbits(active.reshape(-1)),
                    }
                )
            column += 1
        row += 1
    candidates.sort(key=lambda cell: (cell["order"], cell["grid_order"]))
    return candidates


def _cell_masks(cell: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild one small cell mask on demand instead of retaining thousands."""
    y0, y1, x0, x1 = cell["roi"]
    cx, cy = cell["center"]
    direction_x, direction_y = cell["direction"]
    gy, gx = np.mgrid[y0:y1, x0:x1]
    dx, dy = gx - cx, gy - cy
    distance_squared = dx * dx + dy * dy
    active = np.unpackbits(
        cell["active_bits"], count=int(np.prod(cell["active_shape"]))
    ).reshape(cell["active_shape"]).astype(bool)
    projection = dx * direction_x + dy * direction_y
    threshold = math.sqrt(float(cell["radius"]) ** 2) * (0.055 / 0.43)
    return active & (projection < -threshold), active & (projection > threshold), distance_squared


def _apply_colour_bit(lab: np.ndarray, cell: dict[str, Any], bit: int, delta: float) -> None:
    y0, y1, x0, x1 = cell["roi"]
    region = lab[y0:y1, x0:x1]
    left, right, distance_squared = _cell_masks(cell)
    radial = np.clip(1.0 - np.sqrt(distance_squared) / max(1.0, cell["radius"]), 0.0, 1.0)
    weight = 0.58 + 0.42 * radial
    first_sign, second_sign = (-1.0, 1.0) if bit == 0 else (1.0, -1.0)
    a = region[..., 1]
    a[left] = np.clip(a[left] + first_sign * delta * weight[left], 0, 255)
    a[right] = np.clip(a[right] + second_sign * delta * weight[right], 0, 255)


def _read_colour_bit(lab: np.ndarray, cell: dict[str, Any]) -> tuple[int, float]:
    y0, y1, x0, x1 = cell["roi"]
    a = lab[y0:y1, x0:x1, 1]
    left_mask, right_mask, _ = _cell_masks(cell)
    left = float(np.median(a[left_mask]))
    right = float(np.median(a[right_mask]))
    difference = right - left
    return int(difference < 0), min(1.0, abs(difference) / 36.0)


def _address_bucket(grid_order: int, seed: int, salt: int, bit_count: int) -> int:
    value = (
        int(grid_order) * 0x9E3779B1
        + int(seed) * 0x85EBCA6B
        + (int(salt) + 1) * 0xC2B2AE35
    ) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value % max(1, int(bit_count))


def _address_coverage(
    cells: list[dict[str, Any]], seed: int, salt: int, bit_count: int
) -> list[int]:
    coverage = [0] * bit_count
    for cell in cells:
        coverage[_address_bucket(int(cell["grid_order"]), seed, salt, bit_count)] += 1
    return coverage


def _select_address_salt(
    cells: list[dict[str, Any]], seed: int, bit_count: int
) -> tuple[int, list[int]]:
    choices = []
    for salt in range(4):
        coverage = _address_coverage(cells, seed, salt, bit_count)
        choices.append(
            (
                -sum(count == 0 for count in coverage),
                min(coverage, default=0),
                sum(min(count, 5) for count in coverage),
                -salt,
                salt,
                coverage,
            )
        )
    best = max(choices, key=lambda item: item[:4])
    return int(best[4]), list(best[5])


def _payload(text: str) -> tuple[bytes, list[int], dict[str, int | bool]]:
    raw = text.encode("utf-8")
    if not raw:
        raise ValueError("Enter at least one character.")
    compressed = zlib.compress(raw, 9)
    use_compressed = len(compressed) + 2 < len(raw)
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    body = struct.pack(">I", crc) + (compressed if use_compressed else raw)
    bits = _hamming_encode(_bytes_to_bits(body))
    return body, bits, {
        "compressed": use_compressed,
        "raw_length": len(raw),
        "body_length": len(body),
        "encoded_bits": len(bits),
        "text_crc32": crc,
        "seed": crc,
    }


def estimate_contour_colour_capacity(
    source_path: Path,
    text: str,
    silhouette_path: Path | None = None,
    resolutions: tuple[int, ...] = (2048, 3072, 4096, 6144, 8192),
    cell_spacing: int = 14,
    votes_per_bit: float = 8.0,
) -> dict[str, Any]:
    source = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise FileNotFoundError(source_path)
    silhouette = cv2.imread(str(silhouette_path), cv2.IMREAD_UNCHANGED) if silhouette_path else None
    if silhouette_path and silhouette is None:
        raise FileNotFoundError(silhouette_path)
    _, payload_bits, packet = _payload(text)
    seed = (int(packet["body_length"]) << 16) ^ int(packet["encoded_bits"]) ^ 0xC0110A

    calibration_resolution = 1536
    source_mask = _source_mask(source, silhouette)
    canvas, mask = _resize_preserving_texture(source, source_mask, calibration_resolution)
    header = _pack_header(
        bool(packet["compressed"]), int(packet["body_length"]), cell_spacing, 0
    )
    _draw_protocol(canvas, mask, _bytes_to_bits(header))
    ok, buffer = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 96])
    if not ok:
        raise OSError("Could not prepare the capacity calibration surface.")
    visible = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    visible_mask = _jpeg_leaf_mask(visible)
    visible_lab = cv2.cvtColor(visible, cv2.COLOR_BGR2LAB)
    calibration_cells = len(
        _cell_geometry(visible_mask, visible_lab, int(cell_spacing), seed)
    )
    contour = _largest_contour(visible_mask)
    closed = np.vstack([contour, contour[0]])
    calibration_perimeter = float(
        np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1)).sum()
    )

    raw_length = int(packet["raw_length"])
    content_length = max(1, int(packet["body_length"]) - 4)
    compression_ratio = content_length / max(1, raw_length)
    required_payload_votes = int(math.ceil(len(payload_bits) * votes_per_bit))
    rows = []
    for resolution in sorted({max(1536, min(8192, int(value))) for value in resolutions}):
        scale = resolution / calibration_resolution
        available_cells = max(1, int(round(calibration_cells * scale * scale)))
        protocol_available = max(1, int(math.floor(calibration_perimeter * scale / 12.0)))
        mean_votes = available_cells / max(1, len(payload_bits))
        max_encoded_bits = int(available_cells / votes_per_bit)
        max_body_bytes = max(0, int(math.floor(max_encoded_bits / 15.0) * 11 // 8))
        max_text_bytes = max(
            0,
            int((max_body_bytes - 4) / max(0.08, compression_ratio)),
        )
        protocol_pass = protocol_available >= HEADER_RAW_BITS * HEADER_REPETITIONS
        payload_pass = available_cells >= required_payload_votes
        rows.append(
            {
                "resolution": resolution,
                "exportable": bool(protocol_pass and payload_pass),
                "protocol": {
                    "required": HEADER_RAW_BITS * HEADER_REPETITIONS,
                    "available": protocol_available,
                    "pass": protocol_pass,
                },
                "payload": {
                    "required": required_payload_votes,
                    "available": available_cells,
                    "encoded_bits": len(payload_bits),
                    "mean_votes_per_bit": round(mean_votes, 3),
                    "safe_text_bytes": max_text_bytes,
                    "pass": payload_pass,
                },
            }
        )
    safe = next((row for row in rows if row["exportable"]), None)
    return {
        "codec": "contour-colour-codec",
        "version": VERSION,
        "text": {
            "utf8_bytes": raw_length,
            "compressed_body_bytes": int(packet["body_length"]),
            "encoded_bits": len(payload_bits),
            "required_carrier_votes": required_payload_votes,
            "compression_ratio": round(compression_ratio, 4),
        },
        "carrier": {
            "calibration_resolution": calibration_resolution,
            "calibration_cells": calibration_cells,
            "cell_spacing": int(cell_spacing),
            "target_votes_per_bit": votes_per_bit,
        },
        "resolutions": rows,
        "minimum_safe_resolution": safe["resolution"] if safe else None,
        "recommendation": (
            f"Use at least {safe['resolution']}px for this text and specimen."
            if safe
            else "The text exceeds the safe colour capacity up to 8192px. Shorten it or reduce redundancy."
        ),
    }


def encode_contour_colour(
    source_path: Path,
    text: str,
    output_path: Path,
    silhouette_path: Path | None = None,
    target_long_side: int = 3072,
    cell_spacing: int | None = None,
    jpeg_quality: int = 96,
    chroma_delta: float = 28.0,
) -> dict[str, Any]:
    source = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise FileNotFoundError(source_path)
    silhouette = cv2.imread(str(silhouette_path), cv2.IMREAD_UNCHANGED) if silhouette_path else None
    if silhouette_path and silhouette is None:
        raise FileNotFoundError(silhouette_path)
    source_mask = _source_mask(source, silhouette)
    canvas, mask = _resize_preserving_texture(
        source, source_mask, max(1536, min(8192, int(target_long_side)))
    )
    spacing = int(cell_spacing or 14)
    spacing = max(12, min(75, spacing))

    body, payload_bits, packet = _payload(text)
    suffix = output_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Colour-carrier output must be JPG or PNG.")
    quality = max(75, min(100, int(jpeg_quality)))
    seed = (int(packet["body_length"]) << 16) ^ int(packet["encoded_bits"]) ^ 0xC0110A
    packet["seed"] = seed

    def jpg_surface(surface: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if suffix not in {".jpg", ".jpeg"}:
            return surface.copy(), None
        ok, buffer = cv2.imencode(".jpg", surface, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise OSError("Could not prepare the JPG carrier surface.")
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR), buffer

    # A provisional boundary provides the exact visible coordinate system used
    # to choose the best two-bit address salt. The final header then records it.
    provisional_header = _pack_header(
        bool(packet["compressed"]), int(packet["body_length"]), spacing, 0
    )
    provisional = canvas.copy()
    _draw_protocol(provisional, mask, _bytes_to_bits(provisional_header))
    provisional_surface, _ = jpg_surface(provisional)
    provisional_mask = _jpeg_leaf_mask(provisional_surface)
    provisional_lab = cv2.cvtColor(provisional_surface, cv2.COLOR_BGR2LAB)
    provisional_cells = _cell_geometry(provisional_mask, provisional_lab, spacing, seed)
    address_salt, initial_coverage = _select_address_salt(
        provisional_cells, seed, len(payload_bits)
    )
    if not initial_coverage or min(initial_coverage) <= 0:
        recommended = int(math.ceil(max(mask.shape) * 1.28 / 256.0) * 256)
        raise ValueError(
            "The preserved texture cannot address every payload bit at this resolution. "
            f"Use at least {recommended}px, reduce cell spacing, or shorten the text."
        )
    del provisional, provisional_surface, provisional_mask, provisional_lab, provisional_cells
    header = _pack_header(
        bool(packet["compressed"]),
        int(packet["body_length"]),
        spacing,
        address_salt,
    )
    header_bits = _bytes_to_bits(header)

    # Establish the visible outline first, then pass it through the requested
    # JPG profile once. Payload positions are derived from this visible result,
    # not from an alpha mask or an unpublished source image.
    _draw_protocol(canvas, mask, header_bits)
    carrier, _ = jpg_surface(canvas)
    base_carrier = carrier
    reference = carrier
    encoded = carrier
    image_size = [int(carrier.shape[1]), int(carrier.shape[0])]
    del canvas
    cells: list[dict[str, Any]] = []
    final_buffer: np.ndarray | None = None
    stabilized = False
    stabilization_passes = 0
    protocol: dict[str, int] = {}
    final_coverage = initial_coverage
    for pass_index in range(4):
        stabilization_passes = pass_index + 1
        reference_mask = _jpeg_leaf_mask(reference)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
        cells = _cell_geometry(reference_mask, reference_lab, spacing, seed)
        del reference_mask, reference_lab
        final_coverage = _address_coverage(cells, seed, address_salt, len(payload_bits))
        if not final_coverage or min(final_coverage) <= 0:
            recommended = int(math.ceil(max(mask.shape) * 1.25 / 256.0) * 256)
            raise ValueError(
                "Some payload addresses have no usable source texture. "
                f"Use at least {recommended}px, reduce cell spacing, or shorten the text."
            )
        lab = cv2.cvtColor(base_carrier, cv2.COLOR_BGR2LAB)
        for cell in cells:
            bit_index = _address_bucket(
                int(cell["grid_order"]), seed, address_salt, len(payload_bits)
            )
            _apply_colour_bit(lab, cell, payload_bits[bit_index], float(chroma_delta))
        encoded = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        del lab
        encoded[mask == 0] = 255
        protocol = _draw_protocol(encoded, mask, header_bits)

        if suffix in {".jpg", ".jpeg"}:
            ok, output_buffer = cv2.imencode(".jpg", encoded, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise OSError("Could not encode the JPG colour carrier.")
            decoded_output = cv2.imdecode(output_buffer, cv2.IMREAD_COLOR)
            final_buffer = output_buffer
            encoded = None
        else:
            decoded_output = encoded.copy()

        next_mask = _jpeg_leaf_mask(decoded_output)
        next_lab = cv2.cvtColor(decoded_output, cv2.COLOR_BGR2LAB)
        next_cells = _cell_geometry(next_mask, next_lab, spacing, seed)
        del next_mask, next_lab
        current_signature = [(cell["grid_order"], cell["roi"]) for cell in cells]
        next_signature = [(cell["grid_order"], cell["roi"]) for cell in next_cells]
        reference = decoded_output
        if current_signature == next_signature:
            stabilized = True
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".jpg", ".jpeg"}:
        if final_buffer is None:
            raise OSError("No JPG output was generated.")
        output_path.write_bytes(final_buffer.tobytes())
    elif not cv2.imwrite(str(output_path), encoded, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise OSError(f"Could not write {output_path}")

    metadata = {
        "codec": "contour-colour-codec",
        "version": VERSION,
        "mode": "jpg-colour-carrier",
        "source": str(source_path),
        "output": str(output_path),
        "image_size": image_size,
        "jpeg_quality": int(jpeg_quality),
        "protocol": {
            **protocol,
            "header_bytes": HEADER_BYTES,
            "repetitions": HEADER_REPETITIONS,
            "carrier": "continuous colour-modulated leaf contour",
        },
        "payload": {
            **packet,
            "available_cells": len(cells),
            "used_cells": len(cells),
            "cell_spacing": spacing,
            "address_salt": address_salt,
            "minimum_votes_per_bit": min(final_coverage, default=0),
            "mean_votes_per_bit": round(float(np.mean(final_coverage)), 3),
            "maximum_votes_per_bit": max(final_coverage, default=0),
            "chroma_delta": float(chroma_delta),
            "position_stabilized": stabilized,
            "stabilization_passes": stabilization_passes,
            "carrier_redundancy": round(len(cells) / max(1, len(payload_bits)), 4),
            "fec": "Hamming(15,11)",
            "carrier": "Lab chroma shifts on source-texture pixels only",
        },
        "texture_policy": {
            "new_interior_marks": False,
            "luminance_geometry_preserved": True,
            "changed_channel": "Lab a chroma",
        },
        "text": text,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def decode_contour_colour(image_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    mask = _jpeg_leaf_mask(image)
    header, sync = _read_header(image, mask)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    seed = (int(header["body_length"]) << 16) ^ int(header["encoded_bits"]) ^ 0xC0110A
    cells = _cell_geometry(mask, lab, int(header["cell_spacing"]), seed)
    encoded_count = int(header["encoded_bits"])
    if int(header["version"]) < 3 and encoded_count > len(cells):
        raise ValueError("The JPG is missing part of its colour payload.")
    bits: list[int] = []
    confidences: list[float] = []
    address_votes: list[int] = []
    if int(header["version"]) >= 3:
        scores = np.zeros(encoded_count, np.float64)
        weights = np.zeros(encoded_count, np.float64)
        votes = np.zeros(encoded_count, np.int32)
        salt = int(header.get("address_salt", 0))
        for cell in cells:
            bit, confidence = _read_colour_bit(lab, cell)
            weight = max(0.0, confidence - 0.08)
            if weight <= 0.0:
                continue
            index = _address_bucket(int(cell["grid_order"]), seed, salt, encoded_count)
            scores[index] += weight if bit else -weight
            weights[index] += weight
            votes[index] += 1
            confidences.append(confidence)
        bits = [int(score > 0.0) for score in scores]
        address_votes = [int(value) for value in votes]
    else:
        for cell in cells[:encoded_count]:
            bit, confidence = _read_colour_bit(lab, cell)
            bits.append(bit)
            confidences.append(confidence)
    body_length = int(header["body_length"])
    decoded_bits, corrected = _hamming_decode(bits, body_length * 8)
    body = _bits_to_bytes(decoded_bits, body_length * 8)[:body_length]
    if len(body) < 4:
        raise ValueError("The recovered colour payload is incomplete.")
    recorded_crc = struct.unpack(">I", body[:4])[0]
    content = body[4:]
    try:
        raw = zlib.decompress(content) if header["compressed"] else content
    except zlib.error as exc:
        raise ValueError("The colour payload contains too many errors to decompress.") from exc
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    valid = crc == recorded_crc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("The recovered bytes are not valid UTF-8 text.") from exc
    return {
        "codec": "contour-colour-codec",
        "version": int(header["version"]),
        "valid": valid,
        "text": text,
        "header": header,
        "recorded_text_crc32": recorded_crc,
        "calculated_text_crc32": crc,
        "protocol_sync": sync,
        "corrected_hamming_words": corrected,
        "payload_confidence": round(float(np.mean(confidences)) * 100.0, 1),
        "minimum_address_votes": min(address_votes, default=1),
        "mean_address_votes": round(float(np.mean(address_votes)), 3) if address_votes else 1.0,
        "used_cells": len(cells) if address_votes else encoded_count,
        "available_cells": len(cells),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode text by recolouring existing leaf-rubbing texture.")
    commands = parser.add_subparsers(dest="command", required=True)
    encoder = commands.add_parser("encode")
    encoder.add_argument("source", type=Path)
    encoder.add_argument("output", type=Path)
    encoder.add_argument("--silhouette", type=Path)
    encoder.add_argument("--text", required=True)
    encoder.add_argument("--resolution", type=int, default=3072)
    encoder.add_argument("--cell-spacing", type=int)
    encoder.add_argument("--quality", type=int, default=96)
    encoder.add_argument("--chroma-delta", type=float, default=28.0)
    decoder = commands.add_parser("decode")
    decoder.add_argument("image", type=Path)
    args = parser.parse_args()
    if args.command == "encode":
        result = encode_contour_colour(
            args.source,
            args.text,
            args.output,
            silhouette_path=args.silhouette,
            target_long_side=args.resolution,
            cell_spacing=args.cell_spacing,
            jpeg_quality=args.quality,
            chroma_delta=args.chroma_delta,
        )
    else:
        result = decode_contour_colour(args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
