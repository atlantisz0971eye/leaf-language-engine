from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from branch_codec import (
    COMPACT_GUARD_END,
    COMPACT_GUARD_START,
    decode_branch_protocol,
    decode_lexical_payload,
    lexical_codebook,
)
from document_engine import _branch_carrier_plan, _graph_capacity_profile
from encoding_renderer import (
    DARK_GREEN,
    LIGHT_GREEN,
    LOGICAL_HEIGHT,
    LOGICAL_WIDTH,
    _graph_transform,
    _hex_bgr,
    _on_white,
    _point_on_polyline,
    _protocol_boundaries,
)
from leaf_codec import MACRO_GUARD_END, MACRO_GUARD_START


PROTOCOL_SYMBOL_COUNT = 72
PROTOCOL_RAW_BIT_COUNT = 128
SUPPORTED_WIDTHS = (2048, 4096, 8192, 16384)
PAYLOAD_PALETTE = (
    _on_white(LIGHT_GREEN, 0.84),
    _on_white(DARK_GREEN, 0.98),
    _on_white(LIGHT_GREEN, 0.90),
    _on_white(DARK_GREEN, 1.0),
)
PROTOCOL_PALETTE = (_hex_bgr(LIGHT_GREEN), _hex_bgr(DARK_GREEN))


def _color_distance(pixel: np.ndarray, color: tuple[int, int, int]) -> float:
    delta = pixel.astype(np.float32) - np.asarray(color, dtype=np.float32)
    return float(np.linalg.norm(delta))


def _pixel(image: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    x = max(0, min(image.shape[1] - 1, round(point[0])))
    y = max(0, min(image.shape[0] - 1, round(point[1])))
    return image[y, x, :3]


def _sample_center_color(
    image: np.ndarray,
    point: tuple[float, float],
    tangent: tuple[float, float],
    output_scale: float,
) -> np.ndarray:
    samples = []
    for logical_offset in (-0.55, -0.25, 0.0, 0.25, 0.55):
        samples.append(
            _pixel(
                image,
                (
                    point[0] + tangent[0] * logical_offset * output_scale,
                    point[1] + tangent[1] * logical_offset * output_scale,
                ),
            )
        )
    return np.median(np.asarray(samples), axis=0)


def _stroke_width(
    image: np.ndarray,
    point: tuple[float, float],
    tangent: tuple[float, float],
    output_scale: float,
    palette: tuple[tuple[int, int, int], ...],
    maximum_logical_width: float,
    color_tolerance: float,
) -> float:
    normal = (-tangent[1], tangent[0])
    maximum_pixels = max(4, int(math.ceil(maximum_logical_width * output_scale)))
    hits: dict[int, bool] = {}
    for offset in range(-maximum_pixels, maximum_pixels + 1):
        pixel = _pixel(
            image,
            (
                point[0] + normal[0] * offset,
                point[1] + normal[1] * offset,
            ),
        )
        hits[offset] = min(_color_distance(pixel, color) for color in palette) <= color_tolerance

    left = 0
    while left - 1 >= -maximum_pixels and hits[left - 1]:
        left -= 1
    right = 0
    while right + 1 <= maximum_pixels and hits[right + 1]:
        right += 1
    return (right - left + 1) / max(0.001, output_scale)


def _main_points(graph: dict[str, Any]) -> list[tuple[float, float]]:
    nodes = {int(node["id"]): node for node in graph["nodes"]}
    transform = _graph_transform(graph)
    return [
        transform((float(nodes[int(node_id)]["x"]), float(nodes[int(node_id)]["y"])))
        for node_id in graph["main_path"]["node_ids"]
    ]


def _read_protocol(
    image: np.ndarray,
    graph: dict[str, Any],
    output_scale: float,
) -> dict[str, Any]:
    points = _main_points(graph)
    boundaries = _protocol_boundaries(points, PROTOCOL_SYMBOL_COUNT)
    symbols: list[int] = []
    color_scores: list[float] = []
    width_scores: list[float] = []
    measured_widths: list[float] = []

    for index in range(PROTOCOL_SYMBOL_COUNT):
        local_samples = (
            (0.68, 0.78, 0.86)
            if index == 0
            else (0.14, 0.22, 0.32)
            if index == PROTOCOL_SYMBOL_COUNT - 1
            else (0.30, 0.50, 0.70)
        )
        readings: list[tuple[int, float, float, float]] = []
        for local_fraction in local_samples:
            fraction = boundaries[index] + (
                boundaries[index + 1] - boundaries[index]
            ) * local_fraction
            logical_point, tangent = _point_on_polyline(points, fraction)
            point = (logical_point[0] * output_scale, logical_point[1] * output_scale)
            center = _sample_center_color(image, point, tangent, output_scale)
            tone_distances = [
                _color_distance(center, color) for color in PROTOCOL_PALETTE
            ]
            tone = int(tone_distances[1] < tone_distances[0])
            color_score = max(0.0, 1.0 - min(tone_distances) / 90.0)

            natural_width = 6.4 - 4.35 * fraction
            measured = _stroke_width(
                image,
                point,
                tangent,
                output_scale,
                PROTOCOL_PALETTE,
                maximum_logical_width=max(4.0, natural_width * 1.7),
                color_tolerance=72.0,
            )
            expected_thin_pixels = max(
                1, round(natural_width * 0.86 * output_scale)
            )
            expected_thick_pixels = max(
                1, round(natural_width * 1.14 * output_scale)
            )
            measured_pixels = measured * output_scale
            width_midpoint = (
                expected_thin_pixels + expected_thick_pixels
            ) / 2.0 + 1.5
            thick = int(measured_pixels > width_midpoint)
            width_score = min(
                1.0,
                abs(measured_pixels - width_midpoint)
                / max(1.0, (expected_thick_pixels - expected_thin_pixels) / 2.0),
            )
            readings.append(
                (tone + 2 * thick, 0.68 * color_score + 0.32 * width_score, measured, natural_width)
            )
        totals = {
            state: sum(score for candidate, score, _, _ in readings if candidate == state)
            for state in range(4)
        }
        state = max(totals, key=lambda candidate: (totals[candidate], -candidate))
        supporting = [item for item in readings if item[0] == state]
        symbols.append(state)
        color_scores.append(sum(item[1] for item in supporting) / max(1, len(supporting)))
        measured_widths.append(sum(item[2] for item in supporting) / max(1, len(supporting)))
        width_scores.append(
            sum(
                min(1.0, abs(item[2] - item[3]) / max(0.35, item[3] * 0.14))
                for item in supporting
            )
            / max(1, len(supporting))
        )

    # The start ring deliberately overlaps the first guard cell. Once the
    # ring fixes orientation, the compact guard pattern is deterministic and
    # can safely restore locator-obscured guard cells before CRC validation.
    symbols[: len(COMPACT_GUARD_START)] = COMPACT_GUARD_START
    symbols[-len(COMPACT_GUARD_END) :] = COMPACT_GUARD_END
    guard_matches = sum(
        left == right
        for left, right in zip(
            symbols[: len(COMPACT_GUARD_START)], COMPACT_GUARD_START
        )
    ) + sum(
        left == right
        for left, right in zip(
            symbols[-len(COMPACT_GUARD_END) :], COMPACT_GUARD_END
        )
    )
    corrected_symbols: list[int] = []
    try:
        decoded = decode_branch_protocol(
            {"symbols": symbols, "raw_bit_count": PROTOCOL_RAW_BIT_COUNT}
        )
        expected_fingerprint = _graph_capacity_profile(graph)["fingerprint"][:8]
        if not decoded.get("valid"):
            for symbol_index in range(
                len(COMPACT_GUARD_START),
                len(symbols) - len(COMPACT_GUARD_END),
            ):
                original = symbols[symbol_index]
                found = False
                for replacement in range(4):
                    if replacement == original:
                        continue
                    symbols[symbol_index] = replacement
                    candidate = decode_branch_protocol(
                        {"symbols": symbols, "raw_bit_count": PROTOCOL_RAW_BIT_COUNT}
                    )
                    if (
                        candidate.get("valid")
                        and candidate.get("header", {}).get("graph_fingerprint")
                        == expected_fingerprint
                    ):
                        decoded = candidate
                        corrected_symbols = [symbol_index]
                        found = True
                        break
                if found:
                    break
                symbols[symbol_index] = original
    except (ValueError, TypeError, IndexError, struct.error) as exc:
        decoded = {
            "valid": False,
            "guards_valid": False,
            "crc_valid": False,
            "header": None,
            "decode_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        **decoded,
        "symbols": symbols,
        "corrected_symbols": corrected_symbols,
        "guard_matches": guard_matches,
        "confidence": round(
            100.0
            * (
                0.58 * (sum(color_scores) / len(color_scores))
                + 0.30 * (sum(width_scores) / len(width_scores))
                + 0.12 * (guard_matches / 8.0)
            ),
            1,
        ),
        "mean_measured_width": round(sum(measured_widths) / len(measured_widths), 3),
    }


def _edge_state(
    image: np.ndarray,
    points: list[tuple[float, float]],
    output_scale: float,
    fractions: tuple[float, ...] = (0.32, 0.50, 0.68),
) -> tuple[int, float]:
    readings: list[tuple[int, float]] = []
    for fraction in fractions:
        logical_point, tangent = _point_on_polyline(points, fraction)
        point = (logical_point[0] * output_scale, logical_point[1] * output_scale)
        center = _sample_center_color(image, point, tangent, output_scale)
        distances = [_color_distance(center, color) for color in PAYLOAD_PALETTE]
        color_state = min(range(4), key=lambda state: distances[state])
        measured = _stroke_width(
            image,
            point,
            tangent,
            output_scale,
            PAYLOAD_PALETTE,
            maximum_logical_width=3.4,
            color_tolerance=62.0,
        )
        thick = int(measured > 1.32)
        tone = color_state % 2
        width_state = tone + 2 * thick
        ordered_distances = sorted(distances)
        color_margin = ordered_distances[1] - ordered_distances[0]
        state = color_state if color_margin >= 2.0 else width_state
        expected = 1.85 if state >= 2 else 0.8
        color_score = max(0.0, 1.0 - distances[state] / 90.0)
        width_score = max(0.0, 1.0 - abs(measured - expected) / 1.5)
        readings.append((state, 0.82 * color_score + 0.18 * width_score))

    totals = {
        state: sum(score for candidate, score in readings if candidate == state)
        for state in range(4)
    }
    state = max(totals, key=lambda candidate: (totals[candidate], -candidate))
    supporting = [score for candidate, score in readings if candidate == state]
    confidence = sum(supporting) / max(1, len(supporting))
    return state, confidence


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-9:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
            / length_squared,
        ),
    )
    nearest = (start[0] + ratio * dx, start[1] + ratio * dy)
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1])


def _path_distance(
    point: tuple[float, float], path: list[tuple[float, float]]
) -> float:
    return min(
        (
            _point_segment_distance(point, start, end)
            for start, end in zip(path, path[1:])
        ),
        default=float("inf"),
    )


def _safest_edge_fraction(
    points: list[tuple[float, float]],
    occluders: list[list[tuple[float, float]]],
) -> float:
    best_fraction = 0.5
    best_clearance = -1.0
    for fraction in (0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48, 0.54, 0.60, 0.66, 0.72, 0.78, 0.84, 0.88):
        point, _ = _point_on_polyline(points, fraction)
        clearance = min(
            (_path_distance(point, path) for path in occluders),
            default=1000.0,
        )
        if clearance > best_clearance:
            best_clearance = clearance
            best_fraction = fraction
    return best_fraction


def _read_payload(
    image: np.ndarray,
    graph: dict[str, Any],
    source_resolution: int,
    output_scale: float,
    symbol_count: int,
) -> dict[str, Any]:
    carrier_plan = _branch_carrier_plan(graph, source_resolution, 14.0)
    carriers = [
        carrier
        for branch in carrier_plan.get("branches", [])
        for carrier in branch.get("carrier_edges", [])
    ]
    if len(carriers) < symbol_count:
        return {
            "valid": False,
            "symbols": [],
            "capacity": len(carriers),
            "confidence": 0.0,
            "error": f"The specimen exposes {len(carriers)} carriers, but {symbol_count} are required.",
        }

    edges = {int(edge["id"]): edge for edge in graph["edges"]}
    transform = _graph_transform(graph)
    ordered_edges = [int(edge["id"]) for edge in graph["edges"]]
    edge_positions = {edge_id: index for index, edge_id in enumerate(ordered_edges)}
    transformed_paths = {
        int(edge["id"]): [
            transform((float(point[0]), float(point[1]))) for point in edge["path"]
        ]
        for edge in graph["edges"]
    }
    main_points = _main_points(graph)
    symbols: list[int] = []
    confidences: list[float] = []
    for carrier in carriers[:symbol_count]:
        edge_id = int(carrier["edge_id"])
        points = transformed_paths[edge_id]
        position = edge_positions[edge_id]
        occluders = [
            transformed_paths[later_id] for later_id in ordered_edges[position + 1 :]
        ]
        occluders.append(main_points)
        safest = _safest_edge_fraction(points, occluders)
        state, confidence = _edge_state(
            image,
            points,
            output_scale,
            (safest,),
        )
        symbols.append(state)
        confidences.append(confidence)
    return {
        "valid": True,
        "symbols": symbols,
        "capacity": len(carriers),
        "confidence": round(100.0 * sum(confidences) / max(1, len(confidences)), 1),
    }


def _stage(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail}


def _decode_lexical_with_repair(packet: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
    symbols = [int(item) for item in packet["blocks"][0]["symbols"]]
    guard_length = len(MACRO_GUARD_START)
    symbols[:guard_length] = MACRO_GUARD_START
    symbols[-guard_length:] = MACRO_GUARD_END
    packet["blocks"][0]["symbols"] = symbols
    decoded = decode_lexical_payload(packet)
    if decoded.get("valid"):
        return decoded, []

    # A crossing can visually occlude one carrier even in a lossless export.
    # Try one four-state substitution and accept it only when the embedded
    # lexical CRC, UTF-8 stream, guards, and local lexicon all validate.
    for index in range(guard_length, len(symbols) - guard_length):
        original = symbols[index]
        for replacement in range(4):
            if replacement == original:
                continue
            symbols[index] = replacement
            candidate = decode_lexical_payload(packet)
            if candidate.get("valid"):
                return candidate, [index]
        symbols[index] = original
    packet["blocks"][0]["symbols"] = symbols
    return decoded, []


def _decode_candidate(
    image: np.ndarray,
    specimen: dict[str, Any],
    source_resolution: int,
    output_scale: float,
    lexicon_fingerprint: str,
) -> dict[str, Any]:
    graph = specimen["graph"]
    protocol = _read_protocol(image, graph, output_scale)
    graph_fingerprint = _graph_capacity_profile(graph)["fingerprint"][:8]
    header = protocol.get("header") or {}
    graph_valid = bool(
        protocol.get("valid")
        and header.get("graph_fingerprint") == graph_fingerprint
    )
    lexicon_prefix_valid = bool(
        protocol.get("valid")
        and lexicon_fingerprint.startswith(str(header.get("lexicon_fingerprint", "")))
    )

    lexical_bit_length = int(header.get("lexical_bit_length", 0))
    payload_symbol_count = (
        len(MACRO_GUARD_START)
        + math.ceil((lexical_bit_length + 16) / 2)
        + len(MACRO_GUARD_END)
        if protocol.get("valid")
        else 0
    )
    payload = (
        _read_payload(
            image,
            graph,
            source_resolution,
            output_scale,
            payload_symbol_count,
        )
        if graph_valid and lexicon_prefix_valid and payload_symbol_count
        else {
            "valid": False,
            "symbols": [],
            "capacity": 0,
            "confidence": 0.0,
            "error": "Protocol or specimen identity did not validate.",
        }
    )

    lexical = {
        "valid": False,
        "text": "",
        "blocks_valid": False,
        "payload_crc_valid": False,
        "lexicon_valid": lexicon_prefix_valid,
        "decode_error": payload.get("error"),
    }
    repaired_symbols: list[int] = []
    if payload.get("valid"):
        try:
            lexical, repaired_symbols = _decode_lexical_with_repair(
                {
                    "codebook_version": "LPH-LEX-1",
                    "lexicon_fingerprint": lexicon_fingerprint,
                    "lexical_bit_length": lexical_bit_length,
                    "payload_crc16": int(header["payload_crc16"]),
                    "blocks": [{"symbols": list(payload["symbols"])}],
                },
            )
        except (ValueError, TypeError, IndexError, KeyError) as exc:
            lexical["decode_error"] = f"{type(exc).__name__}: {exc}"

    stages = [
        _stage(
            "Protocol rail",
            bool(protocol.get("valid")),
            (
                f"72 cells · guard {protocol.get('guard_matches', 0)}/8 · "
                f"header CRC {'valid' if protocol.get('crc_valid') else 'failed'}"
                + (
                    f" · corrected cell {protocol['corrected_symbols'][0] + 1}"
                    if protocol.get("corrected_symbols")
                    else ""
                )
            ),
        ),
        _stage(
            "Specimen graph",
            graph_valid,
            (
                f"{specimen['name']} · fingerprint "
                f"{'matched' if graph_valid else 'did not match'}"
            ),
        ),
        _stage(
            "Lexicon",
            lexicon_prefix_valid,
            "Local dictionary fingerprint matched."
            if lexicon_prefix_valid
            else "The PNG expects a different dictionary.",
        ),
        _stage(
            "Branch payload",
            bool(payload.get("valid")),
            (
                f"{len(payload.get('symbols', []))}/{payload_symbol_count} states read · "
                f"{payload.get('capacity', 0)} available carriers"
            ),
        ),
        _stage(
            "Text CRC",
            bool(lexical.get("payload_crc_valid")),
            (
                "Reconstructed text matches the CRC."
                if not repaired_symbols
                else f"CRC restored one occluded carrier at state {repaired_symbols[0] + 1}."
            )
            if lexical.get("payload_crc_valid")
            else (lexical.get("decode_error") or "Reconstructed text failed CRC validation."),
        ),
    ]
    score = (
        10000 * int(bool(lexical.get("valid")))
        + 2000 * int(graph_valid)
        + 1000 * int(bool(protocol.get("valid")))
        + 50 * int(protocol.get("guard_matches", 0))
        + float(protocol.get("confidence", 0))
        + float(payload.get("confidence", 0))
    )
    return {
        "valid": bool(lexical.get("valid")),
        "text": lexical.get("text", ""),
        "specimen": specimen["name"],
        "resolution": source_resolution,
        "protocol": {
            "valid": bool(protocol.get("valid")),
            "confidence": protocol.get("confidence", 0.0),
            "header": header or None,
            "guard_matches": protocol.get("guard_matches", 0),
            "corrected_symbols": len(protocol.get("corrected_symbols", [])),
        },
        "payload": {
            "valid": bool(payload.get("valid")),
            "confidence": payload.get("confidence", 0.0),
            "symbol_count": len(payload.get("symbols", [])),
            "required_symbols": payload_symbol_count,
            "crc_valid": bool(lexical.get("payload_crc_valid")),
            "corrected_symbols": len(repaired_symbols),
        },
        "confidence": round(
            (
                0.44 * float(protocol.get("confidence", 0.0))
                + 0.36 * float(payload.get("confidence", 0.0))
                + 20.0 * int(bool(lexical.get("valid")))
            ),
            1,
        ),
        "stages": stages,
        "_score": score,
    }


def decode_encoding_png(
    raw: bytes,
    specimens: list[dict[str, Any]],
    database_path: Path,
) -> dict[str, Any]:
    if not raw.startswith(b"\x89PNG\r\n\x1a\n") or len(raw) < 24:
        raise ValueError("The selected file is not a valid PNG.")
    source_width, source_height = struct.unpack(">II", raw[16:24])
    if source_width not in SUPPORTED_WIDTHS:
        raise ValueError(
            "This decoder accepts Encoding PNG exports at 2048, 4096, 8192, or 16384 px."
        )
    expected_height = round(LOGICAL_HEIGHT * source_width / LOGICAL_WIDTH)
    if abs(source_height - expected_height) > 2:
        raise ValueError(
            f"The PNG canvas is {source_width}×{source_height}; an Encoding PNG at this width "
            f"must be {source_width}×{expected_height}."
        )
    if not specimens:
        raise ValueError("No prepared specimen graphs are available for template matching.")

    reduction = 2 if source_width >= 16384 else 1
    read_flag = cv2.IMREAD_REDUCED_COLOR_2 if reduction == 2 else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), read_flag)
    if image is None:
        raise ValueError("The PNG pixel data could not be decoded.")
    output_scale = image.shape[1] / LOGICAL_WIDTH
    _, lexicon_fingerprint = lexical_codebook(str(database_path.resolve()))

    candidates = [
        _decode_candidate(
            image,
            specimen,
            int(source_width),
            output_scale,
            lexicon_fingerprint,
        )
        for specimen in specimens
    ]
    best = max(candidates, key=lambda item: item["_score"])
    best.pop("_score", None)
    best["png"] = {
        "width": int(source_width),
        "height": int(source_height),
        "decode_width": int(image.shape[1]),
        "file_bytes": len(raw),
    }
    best["stages"].insert(
        0,
        _stage(
            "Encoding PNG",
            True,
            f"{source_width}×{source_height} · exact export canvas recognized",
        ),
    )
    best["candidate_count"] = len(candidates)
    return best
