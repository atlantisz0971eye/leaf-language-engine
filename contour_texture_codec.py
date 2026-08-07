from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MAGIC = b"CT"
VERSION = 1
HEADER_WITHOUT_CRC = ">2sBBHHHHI"
HEADER_FORMAT = HEADER_WITHOUT_CRC + "H"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)
HEADER_REPETITIONS = 3

# These are deliberately close to colours already present in leaf rubbings.
# Protocol marks are olive; payload marks remain green.
PROTOCOL_ZERO = np.asarray((151, 190, 169), dtype=np.float32)  # BGR
PROTOCOL_ONE = np.asarray((67, 115, 91), dtype=np.float32)
PAYLOAD_LIGHT = np.asarray((171, 207, 183), dtype=np.float32)
PAYLOAD_DARK = np.asarray((48, 104, 67), dtype=np.float32)


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _bytes_to_bits(data: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def _bits_to_bytes(bits: list[int], bit_length: int | None = None) -> bytes:
    selected = list(bits[: len(bits) if bit_length is None else bit_length])
    selected.extend([0] * ((8 - len(selected) % 8) % 8))
    return bytes(
        sum(selected[index + offset] << (7 - offset) for offset in range(8))
        for index in range(0, len(selected), 8)
    )


def _hamming_encode(bits: list[int]) -> list[int]:
    padded = bits + [0] * ((11 - len(bits) % 11) % 11)
    encoded: list[int] = []
    data_positions = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
    for offset in range(0, len(padded), 11):
        word = [0] * 16
        for position, bit in zip(data_positions, padded[offset : offset + 11]):
            word[position] = bit
        for parity in (1, 2, 4, 8):
            word[parity] = sum(
                word[position]
                for position in range(1, 16)
                if position & parity and position != parity
            ) % 2
        encoded.extend(word[1:])
    return encoded


def _hamming_decode(bits: list[int], raw_bit_length: int) -> tuple[list[int], int]:
    if len(bits) % 15:
        raise ValueError("The texture payload does not contain complete Hamming words.")
    output: list[int] = []
    corrected = 0
    data_positions = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
    for offset in range(0, len(bits), 15):
        word = [0] + bits[offset : offset + 15]
        syndrome = 0
        for parity in (1, 2, 4, 8):
            if sum(word[position] for position in range(1, 16) if position & parity) % 2:
                syndrome |= parity
        if syndrome:
            word[syndrome] ^= 1
            corrected += 1
        output.extend(word[position] for position in data_positions)
    return output[:raw_bit_length], corrected


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No leaf silhouette was found.")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(contour) < 32:
        raise ValueError("The leaf boundary is too small to carry a protocol.")
    # Fix traversal direction, then fix the first point with the PCA major axis.
    if cv2.contourArea(contour.reshape(-1, 1, 2), oriented=True) > 0:
        contour = contour[::-1]
    centered = contour - contour.mean(axis=0)
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
        axis = -axis
    start = int(np.argmin(centered @ axis))
    return np.roll(contour, -start, axis=0)


def _resample_closed(contour: np.ndarray, count: int) -> np.ndarray:
    closed = np.vstack([contour, contour[0]])
    lengths = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if total <= 1.0:
        raise ValueError("The leaf boundary has no measurable perimeter.")
    targets = np.linspace(0.0, total, count, endpoint=False)
    result = np.empty((count, 2), dtype=np.float32)
    segment = 0
    for index, target in enumerate(targets):
        while segment + 1 < len(cumulative) and cumulative[segment + 1] <= target:
            segment += 1
        span = max(1e-6, cumulative[segment + 1] - cumulative[segment])
        ratio = (target - cumulative[segment]) / span
        result[index] = closed[segment] * (1.0 - ratio) + closed[segment + 1] * ratio
    return result


def _mask_from_image(image: np.ndarray, silhouette: np.ndarray | None = None) -> np.ndarray:
    if silhouette is not None:
        if silhouette.ndim == 3:
            silhouette = cv2.cvtColor(silhouette, cv2.COLOR_BGR2GRAY)
        return np.where(silhouette > 127, 255, 0).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 4 and int(image[..., 3].max()) > 0:
        return np.where(image[..., 3] > 127, 255, 0).astype(np.uint8)
    bgr = image[..., :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1]
    candidate = np.where((gray < 246) | (saturation > 16), 255, 0).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(candidate.shape, np.uint8)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return mask


def _resize_source(image: np.ndarray, mask: np.ndarray, target_long_side: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    scale = target_long_side / max(height, width)
    output_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized_mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
    bgr = image[..., :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(bgr, output_size, interpolation=cv2.INTER_CUBIC)
    # Keep the source rubbing, but give blank paper inside the silhouette a very
    # slight green cast so that light carrier states remain measurable.
    pale = np.full_like(resized, (239, 246, 235))
    ink = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) < 242
    normalized = pale.copy()
    normalized[ink] = resized[ink]
    normalized[resized_mask == 0] = 255
    return normalized, resized_mask


def _protocol_geometry(mask: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, float]:
    contour = _largest_contour(mask)
    points = _resample_closed(contour, count)
    centroid = contour.mean(axis=0)
    height, width = mask.shape
    inset = max(5.0, min(height, width) * 0.0075)
    inward = centroid[None, :] - points
    inward /= np.maximum(1e-6, np.linalg.norm(inward, axis=1, keepdims=True))
    points = points + inward * inset
    return points, centroid, inset


def _sample_patch(image: np.ndarray, point: np.ndarray, radius: int) -> np.ndarray:
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
    patch = image[y0:y1, x0:x1, :3]
    if patch.size == 0:
        return np.asarray((255, 255, 255), np.float32)
    return np.median(patch.reshape(-1, 3), axis=0)


def _draw_mark(image: np.ndarray, point: np.ndarray, color: np.ndarray, radius: float, angle: float = 0.0) -> None:
    center = (int(round(float(point[0]))), int(round(float(point[1]))))
    axes = (max(1, round(radius * 1.45)), max(1, round(radius * 0.62)))
    cv2.ellipse(
        image,
        center,
        axes,
        math.degrees(angle),
        0,
        360,
        tuple(int(value) for value in color),
        -1,
        cv2.LINE_AA,
    )


def _pack_header(
    flags: int,
    body_length: int,
    encoded_bits: int,
    raw_length: int,
    cell_spacing: int,
    text_crc32: int,
) -> bytes:
    prefix = struct.pack(
        HEADER_WITHOUT_CRC,
        MAGIC,
        VERSION,
        flags,
        body_length,
        encoded_bits,
        raw_length,
        cell_spacing,
        text_crc32,
    )
    return prefix + struct.pack(">H", _crc16(prefix))


def _unpack_header(raw: bytes) -> dict[str, int | bool]:
    if len(raw) != HEADER_BYTES:
        raise ValueError("The contour protocol header has the wrong length.")
    magic, version, flags, body_length, encoded_bits, raw_length, spacing, crc32, header_crc = struct.unpack(
        HEADER_FORMAT, raw
    )
    if magic != MAGIC or version != VERSION:
        raise ValueError("This PNG does not contain a supported contour-texture protocol.")
    if _crc16(raw[:-2]) != header_crc:
        raise ValueError("The contour protocol header failed its CRC check.")
    return {
        "version": version,
        "compressed": bool(flags & 1),
        "flags": flags,
        "body_length": body_length,
        "encoded_bits": encoded_bits,
        "raw_length": raw_length,
        "cell_spacing": spacing,
        "text_crc32": crc32,
        "header_crc_valid": True,
    }


def _payload_centers(mask: np.ndarray, spacing: int, seed: int) -> list[np.ndarray]:
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return []
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    margin = max(7.0, spacing * 0.85)
    candidates: list[tuple[int, int, np.ndarray]] = []
    row = 0
    for y in range(y0 + spacing, y1 - spacing + 1, spacing):
        offset = spacing // 2 if row % 2 else 0
        column = 0
        for x in range(x0 + spacing + offset, x1 - spacing + 1, spacing):
            # Coordinate hashing gives a stable, non-periodic jitter without
            # making positions depend on how many boundary cells survived.
            value = (x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)
            jx = ((value & 255) / 255.0 - 0.5) * spacing * 0.42
            jy = (((value >> 8) & 255) / 255.0 - 0.5) * spacing * 0.42
            px, py = int(round(x + jx)), int(round(y + jy))
            if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0] and distance[py, px] >= margin:
                order = ((value >> 16) ^ (value * 2654435761)) & 0xFFFFFFFF
                candidates.append((order, row * 100000 + column, np.asarray((px, py), np.float32)))
            column += 1
        row += 1
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in candidates]


def _pair_geometry(center: np.ndarray, spacing: int, seed: int, index: int) -> tuple[np.ndarray, np.ndarray, float]:
    value = ((index + 1) * 1103515245 + seed * 12345) & 0xFFFFFFFF
    angle = (value / 0xFFFFFFFF) * math.pi
    direction = np.asarray((math.cos(angle), math.sin(angle)), np.float32)
    separation = spacing * 0.235
    return center - direction * separation, center + direction * separation, angle


def _encode_payload(text: str) -> tuple[bytes, list[int], dict[str, int | bool]]:
    raw = text.encode("utf-8")
    if not raw:
        raise ValueError("Enter at least one character.")
    compressed = zlib.compress(raw, 9)
    use_compressed = len(compressed) + 2 < len(raw)
    body = compressed if use_compressed else raw
    encoded = _hamming_encode(_bytes_to_bits(body))
    crc32 = zlib.crc32(raw) & 0xFFFFFFFF
    return body, encoded, {
        "compressed": use_compressed,
        "raw_length": len(raw),
        "body_length": len(body),
        "encoded_bits": len(encoded),
        "text_crc32": crc32,
        "seed": crc32,
    }


def encode_contour_texture(
    source_path: Path,
    text: str,
    output_path: Path,
    silhouette_path: Path | None = None,
    target_long_side: int = 2048,
    cell_spacing: int | None = None,
) -> dict[str, Any]:
    source = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise FileNotFoundError(source_path)
    silhouette = None
    if silhouette_path is not None:
        silhouette = cv2.imread(str(silhouette_path), cv2.IMREAD_UNCHANGED)
        if silhouette is None:
            raise FileNotFoundError(silhouette_path)
    source_mask = _mask_from_image(source, silhouette)
    canvas, mask = _resize_source(source, source_mask, max(1024, min(8192, int(target_long_side))))
    spacing = int(cell_spacing or max(12, round(max(mask.shape) / 148)))
    spacing = max(10, min(64, spacing))

    body, payload_bits, packet = _encode_payload(text)
    header = _pack_header(
        1 if packet["compressed"] else 0,
        int(packet["body_length"]),
        int(packet["encoded_bits"]),
        int(packet["raw_length"]),
        spacing,
        int(packet["text_crc32"]),
    )
    header_bits = _bytes_to_bits(header)
    repeated_header = header_bits * HEADER_REPETITIONS

    protocol_points, centroid, inset = _protocol_geometry(mask, len(repeated_header))
    protocol_radius = max(1.6, inset * 0.24)
    for index, bit in enumerate(repeated_header):
        previous = protocol_points[index - 1]
        following = protocol_points[(index + 1) % len(protocol_points)]
        tangent = following - previous
        angle = math.atan2(float(tangent[1]), float(tangent[0]))
        _draw_mark(
            canvas,
            protocol_points[index],
            PROTOCOL_ONE if bit else PROTOCOL_ZERO,
            protocol_radius,
            angle,
        )

    centers = _payload_centers(mask, spacing, int(packet["seed"]))
    if len(payload_bits) > len(centers):
        ratio = math.sqrt(len(payload_bits) / max(1, len(centers)))
        recommended = int(math.ceil(max(mask.shape) * ratio / 256.0) * 256)
        raise ValueError(
            f"This leaf provides {len(centers)} reliable texture cells but the text needs "
            f"{len(payload_bits)}. Use a target resolution of at least {recommended}px or shorten the text."
        )
    payload_radius = max(1.5, spacing * 0.135)
    for index, bit in enumerate(payload_bits):
        left, right, angle = _pair_geometry(centers[index], spacing, int(packet["seed"]), index)
        first, second = (PAYLOAD_DARK, PAYLOAD_LIGHT) if bit == 0 else (PAYLOAD_LIGHT, PAYLOAD_DARK)
        _draw_mark(canvas, left, first, payload_radius, angle)
        _draw_mark(canvas, right, second, payload_radius, angle)

    # A small three-part locator identifies the protocol start without changing
    # the silhouette. It is decorative in v1 and will support rectification in
    # the later print/scan profile.
    start = protocol_points[0]
    inward = centroid - start
    inward /= max(1e-6, float(np.linalg.norm(inward)))
    normal = np.asarray((-inward[1], inward[0]), np.float32)
    locator_center = start + inward * inset * 1.8
    locator_color = tuple(int(value) for value in PROTOCOL_ONE)
    cv2.circle(canvas, tuple(np.rint(locator_center).astype(int)), round(protocol_radius * 2.2), locator_color, 1, cv2.LINE_AA)
    for sign in (-1.0, 1.0):
        point = locator_center + normal * sign * protocol_radius * 2.8
        cv2.circle(canvas, tuple(np.rint(point).astype(int)), max(1, round(protocol_radius * 0.72)), locator_color, -1, cv2.LINE_AA)

    rgba = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rgba):
        raise OSError(f"Could not write {output_path}")

    metadata = {
        "codec": "contour-registered-texture-codec",
        "version": VERSION,
        "mode": "digital-png",
        "source": str(source_path),
        "output": str(output_path),
        "image_size": [int(rgba.shape[1]), int(rgba.shape[0])],
        "protocol": {
            "location": "inner contour band",
            "header_bytes": HEADER_BYTES,
            "repetitions": HEADER_REPETITIONS,
            "header_cells": len(repeated_header),
            "locator": "ring plus two satellites",
        },
        "payload": {
            **packet,
            "available_cells": len(centers),
            "used_cells": len(payload_bits),
            "cell_spacing": spacing,
            "utilization": round(len(payload_bits) / max(1, len(centers)), 4),
            "fec": "Hamming(15,11)",
            "symbol": "paired local optical density",
        },
        "text": text,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def decode_contour_texture(image_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(image_path)
    mask = _mask_from_image(image)
    bgr = image[..., :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    header_bit_count = HEADER_BYTES * 8
    repeated_count = header_bit_count * HEADER_REPETITIONS
    protocol_points, _, inset = _protocol_geometry(mask, repeated_count)
    sample_radius = max(1, round(inset * 0.11))
    repeated: list[int] = []
    protocol_confidences: list[float] = []
    for point in protocol_points:
        color = _sample_patch(bgr, point, sample_radius)
        zero_distance = float(np.linalg.norm(color - PROTOCOL_ZERO))
        one_distance = float(np.linalg.norm(color - PROTOCOL_ONE))
        repeated.append(int(one_distance < zero_distance))
        protocol_confidences.append(abs(one_distance - zero_distance) / max(1.0, one_distance + zero_distance))
    header_bits = [
        int(sum(repeated[copy * header_bit_count + index] for copy in range(HEADER_REPETITIONS)) >= 2)
        for index in range(header_bit_count)
    ]
    header = _unpack_header(_bits_to_bytes(header_bits, header_bit_count))

    encoded_count = int(header["encoded_bits"])
    body_length = int(header["body_length"])
    spacing = int(header["cell_spacing"])
    seed = int(header["text_crc32"])
    centers = _payload_centers(mask, spacing, seed)
    if encoded_count > len(centers):
        raise ValueError("The PNG is missing part of its texture payload.")
    bits: list[int] = []
    payload_confidences: list[float] = []
    payload_radius = max(1, round(spacing * 0.09))
    for index in range(encoded_count):
        left, right, _ = _pair_geometry(centers[index], spacing, seed, index)
        left_color = _sample_patch(bgr, left, payload_radius)
        right_color = _sample_patch(bgr, right, payload_radius)
        # Green-channel optical density is stable for this monochrome carrier.
        left_density = 255.0 - float(left_color[1])
        right_density = 255.0 - float(right_color[1])
        bits.append(int(right_density > left_density))
        payload_confidences.append(abs(left_density - right_density) / 255.0)

    decoded_bits, corrected = _hamming_decode(bits, body_length * 8)
    body = _bits_to_bytes(decoded_bits, body_length * 8)[:body_length]
    raw = zlib.decompress(body) if header["compressed"] else body
    crc32 = zlib.crc32(raw) & 0xFFFFFFFF
    valid = crc32 == int(header["text_crc32"]) and len(raw) == int(header["raw_length"])
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("The payload was recovered but is not valid UTF-8 text.") from exc
    return {
        "codec": "contour-registered-texture-codec",
        "version": int(header["version"]),
        "valid": valid,
        "text": text,
        "header": header,
        "corrected_hamming_words": corrected,
        "protocol_confidence": round(float(np.mean(protocol_confidences)) * 100.0, 1),
        "payload_confidence": round(float(np.mean(payload_confidences)) * 100.0, 1),
        "used_cells": encoded_count,
        "available_cells": len(centers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode or decode text in a leaf contour and rubbing texture.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    encoder = subparsers.add_parser("encode")
    encoder.add_argument("source", type=Path)
    encoder.add_argument("output", type=Path)
    encoder.add_argument("--silhouette", type=Path)
    encoder.add_argument("--text", required=True)
    encoder.add_argument("--resolution", type=int, default=2048)
    encoder.add_argument("--cell-spacing", type=int)
    decoder = subparsers.add_parser("decode")
    decoder.add_argument("image", type=Path)
    args = parser.parse_args()
    if args.command == "encode":
        result = encode_contour_texture(
            args.source,
            args.text,
            args.output,
            silhouette_path=args.silhouette,
            target_long_side=args.resolution,
            cell_spacing=args.cell_spacing,
        )
    else:
        result = decode_contour_texture(args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
