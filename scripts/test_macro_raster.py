"""Synthetic raster-degradation check for the MVG-4 visual alphabet."""

from __future__ import annotations

import random

import cv2
import numpy as np


PITCH = 18
SHORT = 8
LONG = 15
BASELINE = 48


def render(symbols: list[int]) -> np.ndarray:
    width = (len(symbols) + 2) * PITCH
    image = np.full((96, width), 255, dtype=np.uint8)
    cv2.line(image, (PITCH // 2, BASELINE), (width - PITCH // 2, BASELINE), 35, 3)
    for index, symbol in enumerate(symbols):
        x = (index + 1) * PITCH
        side = -1 if symbol < 2 else 1
        extent = LONG if symbol % 2 else SHORT
        cv2.line(
            image,
            (x, BASELINE),
            (x + 3, BASELINE + side * extent),
            35,
            3,
        )
    return image


def degrade(image: np.ndarray) -> np.ndarray:
    reduced = cv2.resize(
        image,
        None,
        fx=0.72,
        fy=0.72,
        interpolation=cv2.INTER_AREA,
    )
    restored = cv2.resize(
        reduced,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    restored = cv2.GaussianBlur(restored, (3, 3), 0.65)
    ok, encoded = cv2.imencode(".jpg", restored, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        raise RuntimeError("JPEG degradation failed.")
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def decode(image: np.ndarray, count: int) -> list[int]:
    output = []
    for index in range(count):
        x = (index + 1) * PITCH
        region = image[:, max(0, x - 2) : x + 7]
        dark = region < 150
        rows = np.where(dark)[0]
        above = rows[rows < BASELINE - 3]
        below = rows[rows > BASELINE + 3]
        if len(above) >= len(below):
            side = 0
            extent = BASELINE - int(above.min()) if len(above) else 0
        else:
            side = 2
            extent = int(below.max()) - BASELINE if len(below) else 0
        output.append(side + (1 if extent >= 12 else 0))
    return output


def render_edge_states(states: list[int]) -> np.ndarray:
    cell_width = 28
    image = np.full((52, (len(states) + 2) * cell_width), 255, dtype=np.uint8)
    for index, state in enumerate(states):
        x = (index + 1) * cell_width
        tone = 105 if state % 2 == 0 else 38
        width = 2 if state < 2 else 5
        cv2.line(image, (x - 9, 31), (x + 9, 20), tone, width, cv2.LINE_AA)
    return image


def decode_edge_states(image: np.ndarray, count: int) -> list[int]:
    cell_width = 28
    output = []
    for index in range(count):
        x = (index + 1) * cell_width
        region = image[14:38, x - 12 : x + 13]
        dark = region < 205
        area = int(dark.sum())
        tone_pixels = region[region < 180]
        tone = float(np.median(tone_pixels)) if len(tone_pixels) else 255
        width_state = 2 if area >= 115 else 0
        tone_threshold = 75 if width_state else 112
        tone_state = 1 if tone < tone_threshold else 0
        output.append(width_state + tone_state)
    return output


def render_protocol_rail(states: list[int]) -> np.ndarray:
    cell_width = 30
    image = np.full((100, (len(states) + 2) * cell_width), 255, dtype=np.uint8)
    for index, state in enumerate(states):
        midpoint = (index + 0.5) / len(states)
        natural_width = 14 - 7 * midpoint
        width = max(2, round(natural_width * (1.14 if state >= 2 else 0.86)))
        tone = 112 if state % 2 == 0 else 48
        x1 = (index + 1) * cell_width
        x2 = x1 + cell_width + 1
        cv2.line(image, (x1, 50), (x2, 50), tone, width, cv2.LINE_AA)
    return image


def decode_protocol_rail(image: np.ndarray, count: int) -> list[int]:
    cell_width = 30
    output = []
    for index in range(count):
        midpoint = (index + 0.5) / count
        expected_width = 14 - 7 * midpoint
        x = (index + 1) * cell_width + cell_width // 2
        column = image[:, x - 1 : x + 2]
        dark_rows = np.where((column < 205).any(axis=1))[0]
        measured_width = int(dark_rows.max() - dark_rows.min() + 1) if len(dark_rows) else 0
        region = image[35:66, x - 8 : x + 9]
        tone_pixels = region[region < 190]
        tone = float(np.median(tone_pixels)) if len(tone_pixels) else 255
        width_state = 2 if measured_width >= expected_width + 3.2 else 0
        tone_state = 1 if tone < 88 else 0
        output.append(width_state + tone_state)
    return output


def main() -> None:
    random.seed(8064)
    symbols = [random.randrange(4) for _ in range(60)]
    raster = degrade(render(symbols))
    decoded = decode(raster, len(symbols))
    accuracy = sum(a == b for a, b in zip(symbols, decoded, strict=True)) / len(symbols)
    print(f"MVG-4 raster accuracy: {accuracy:.1%} ({len(symbols)} symbols)")
    if accuracy < 0.98:
        raise SystemExit("Raster-safe macro alphabet test failed.")
    states = [random.randrange(4) for _ in range(60)]
    state_raster = degrade(render_edge_states(states))
    decoded_states = decode_edge_states(state_raster, len(states))
    state_accuracy = sum(
        a == b for a, b in zip(states, decoded_states, strict=True)
    ) / len(states)
    print(
        f"Four-state branch-edge accuracy: {state_accuracy:.1%} "
        f"({len(states)} edge states)"
    )
    if state_accuracy < 0.98:
        raise SystemExit("Branch edge-state raster test failed.")
    rail_states = [random.randrange(4) for _ in range(60)]
    rail_raster = degrade(render_protocol_rail(rail_states))
    decoded_rail = decode_protocol_rail(rail_raster, len(rail_states))
    rail_accuracy = sum(
        a == b for a, b in zip(rail_states, decoded_rail, strict=True)
    ) / len(rail_states)
    print(
        f"Tapered protocol-rail accuracy: {rail_accuracy:.1%} "
        f"({len(rail_states)} cells)"
    )
    if rail_accuracy < 0.95:
        raise SystemExit("Tapered protocol-rail raster test failed.")


if __name__ == "__main__":
    main()
