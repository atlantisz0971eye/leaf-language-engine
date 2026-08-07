#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# These photographs are the first curated study set. The rule records where
# the rubbing is relative to its real-leaf reference. Candidate extraction is
# still automatic; only the final pair choice is curated for this batch.
CURATED_SELECTION = {
    "1551": "lower",
    "1552": "lower",
    "1553": "right",
    "1554": "right",
    "1555": "right",
    "1556": "right",
    "1557": "right",
    "1558": "lower",
    "1559": "upper",
    "1560": "left",
    "1561": "left",
    "1562": "left",
    "1563": "right",
    "1564": "right",
    "1565": "left",
    "1566": "left",
    "1567": "left",
    "1568": "left",
    "1569": "left",
    "1570": "upper",
}

REVIEW_RECOMMENDED = {"1558", "1559", "1561", "1567"}


@dataclass
class Candidate:
    label: int
    x: int
    y: int
    width: int
    height: int
    area: int
    center_x: float
    center_y: float
    foreground_density: float
    mean_saturation: float
    border_touch: bool


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _kernel(size: int) -> np.ndarray:
    size = max(3, int(size) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def detect_candidates(
    image: np.ndarray,
    maximum_side: int = 1100,
) -> tuple[np.ndarray, np.ndarray, list[Candidate], float]:
    height, width = image.shape[:2]
    scale = min(1.0, maximum_side / max(height, width))
    small = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    blurred = cv2.GaussianBlur(small, (0, 0), 3.0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Saturated green/brown ink and dark neutral material are retained. The
    # initial blur suppresses most single-thread textile marks.
    foreground = (
        (((saturation > 18) & (value < 248)) | (value < 175)).astype(np.uint8)
        * 255
    )
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, _kernel(5))
    merged = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, _kernel(23))
    merged = cv2.dilate(merged, _kernel(13))

    label_count, labels, stats, centers = cv2.connectedComponentsWithStats(merged)
    image_area = small.shape[0] * small.shape[1]
    candidates: list[Candidate] = []
    for label in range(1, label_count):
        x, y, candidate_width, candidate_height, area = [
            int(item) for item in stats[label]
        ]
        if area < image_area * 0.008 or candidate_width < 40 or candidate_height < 40:
            continue
        local_foreground = foreground[
            y : y + candidate_height, x : x + candidate_width
        ] > 0
        local_saturation = saturation[
            y : y + candidate_height, x : x + candidate_width
        ][local_foreground]
        border_touch = bool(
            x <= 2
            or y <= 2
            or x + candidate_width >= small.shape[1] - 2
            or y + candidate_height >= small.shape[0] - 2
        )
        candidates.append(
            Candidate(
                label=label,
                x=x,
                y=y,
                width=candidate_width,
                height=candidate_height,
                area=area,
                center_x=float(centers[label][0]),
                center_y=float(centers[label][1]),
                foreground_density=float(local_foreground.mean()),
                mean_saturation=float(local_saturation.mean())
                if local_saturation.size
                else 0.0,
                border_touch=border_touch,
            )
        )
    candidates.sort(key=lambda item: item.area, reverse=True)
    return small, labels, candidates, scale


def choose_candidate(
    stem: str,
    candidates: list[Candidate],
    image_shape: tuple[int, int, int],
) -> tuple[Candidate, str]:
    if not candidates:
        raise ValueError("No leaf-like foreground candidates were found.")
    usable = [
        item
        for item in candidates
        if not (item.border_touch and item.area < image_shape[0] * image_shape[1] * 0.05)
    ] or candidates
    usable = sorted(usable, key=lambda item: item.area, reverse=True)[:4]
    direction = CURATED_SELECTION.get(stem)
    if direction == "left":
        return min(usable, key=lambda item: item.center_x), "curated-left"
    if direction == "right":
        return max(usable, key=lambda item: item.center_x), "curated-right"
    if direction == "upper":
        return min(usable, key=lambda item: item.center_y), "curated-upper"
    if direction == "lower":
        return max(usable, key=lambda item: item.center_y), "curated-lower"

    # Generic fallback: rubbings tend to have lower solid coverage and lower
    # saturation than the physical reference leaf in the same photograph.
    return max(
        usable,
        key=lambda item: (
            (1.0 - item.foreground_density) * 2.0
            + (1.0 - item.mean_saturation / 255.0),
            item.area,
        ),
    ), "automatic-texture-score"


def component_silhouette(
    labels: np.ndarray,
    candidate: Candidate,
    full_shape: tuple[int, int],
) -> np.ndarray:
    component = (labels == candidate.label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(component)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, _kernel(19))
    return cv2.resize(
        filled,
        (full_shape[1], full_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def crop_bounds(mask: np.ndarray, padding_fraction: float = 0.075) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("The selected rubbing silhouette is empty.")
    x, y, width, height = cv2.boundingRect(points)
    padding = max(24, round(max(width, height) * padding_fraction))
    return (
        max(0, x - padding),
        max(0, y - padding),
        min(mask.shape[1], x + width + padding),
        min(mask.shape[0], y + height + padding),
    )


def ink_alpha(image: np.ndarray, silhouette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.35)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

    outside = silhouette == 0
    if int(outside.sum()) > 100:
        background_l = float(np.median(lab[:, :, 0][outside]))
    else:
        background_l = float(np.percentile(lab[:, :, 0], 90))
    saturation_score = np.clip((hsv[:, :, 1] - 7.0) / 90.0, 0.0, 1.0)
    darkness_score = np.clip((background_l - lab[:, :, 0] - 2.0) / 70.0, 0.0, 1.0)
    score = np.maximum(saturation_score, darkness_score)
    score[silhouette == 0] = 0.0

    values = score[silhouette > 0]
    if values.size:
        otsu_input = np.clip(values * 255.0, 0, 255).astype(np.uint8)
        threshold, _ = cv2.threshold(
            otsu_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        normalized_threshold = max(0.10, min(0.34, threshold / 255.0 * 0.78))
    else:
        normalized_threshold = 0.16

    binary = ((score >= normalized_threshold) & (silhouette > 0)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _kernel(3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _kernel(3))
    alpha = np.clip(
        (score - normalized_threshold * 0.72) / max(0.08, 0.58 - normalized_threshold),
        0.0,
        1.0,
    )
    alpha = (alpha * 255.0).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha[silhouette == 0] = 0
    return alpha, binary


def preliminary_vein_map(
    image: np.ndarray,
    silhouette: np.ndarray,
    ink_mask: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.8)
    responses = []
    for size in (13, 23, 37):
        responses.append(cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, _kernel(size)))
    response = np.maximum.reduce(responses)
    response[silhouette == 0] = 0
    valid = response[(silhouette > 0) & (ink_mask > 0)]
    threshold = float(np.percentile(valid, 74)) if valid.size else 255.0
    vein = ((response >= max(5.0, threshold)) & (silhouette > 0)).astype(np.uint8) * 255
    vein = cv2.morphologyEx(vein, cv2.MORPH_OPEN, _kernel(3))
    return vein


def quality_metrics(
    silhouette: np.ndarray,
    ink_mask: np.ndarray,
    vein_map: np.ndarray,
    stem: str,
) -> dict[str, Any]:
    leaf_area = max(1, int(np.count_nonzero(silhouette)))
    coverage = np.count_nonzero(ink_mask) / leaf_area
    vein_density = np.count_nonzero(vein_map) / leaf_area
    coverage_score = max(0.0, 1.0 - abs(coverage - 0.48) / 0.48)
    vein_score = min(1.0, vein_density / 0.12)
    score = round(100.0 * (0.7 * coverage_score + 0.3 * vein_score), 1)
    review = stem in REVIEW_RECOMMENDED or coverage < 0.12 or coverage > 0.82
    return {
        "ink_coverage": round(coverage, 4),
        "preliminary_vein_density": round(vein_density, 4),
        "quality_score": score,
        "review_recommended": review,
    }


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def process_image(source: Path, output_root: Path) -> tuple[dict[str, Any], np.ndarray]:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {source.name}")
    small, labels, candidates, scale = detect_candidates(image)
    selected, method = choose_candidate(source.stem, candidates, small.shape)
    silhouette = component_silhouette(labels, selected, image.shape[:2])
    x0, y0, x1, y1 = crop_bounds(silhouette)
    crop = image[y0:y1, x0:x1].copy()
    crop_silhouette = silhouette[y0:y1, x0:x1].copy()
    alpha, ink_mask = ink_alpha(crop, crop_silhouette)
    vein_map = preliminary_vein_map(crop, crop_silhouette, ink_mask)

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    white = np.full_like(crop, 255)
    opacity = alpha.astype(np.float32)[:, :, None] / 255.0
    clean = (crop * opacity + white * (1.0 - opacity)).astype(np.uint8)

    destination = output_root / source.stem
    _write(destination / "rubbing_rgba.png", rgba)
    _write(destination / "rubbing_clean.png", clean)
    _write(destination / "silhouette.png", crop_silhouette)
    _write(destination / "ink_mask.png", ink_mask)
    _write(destination / "vein_candidate.png", vein_map)

    metrics = quality_metrics(crop_silhouette, ink_mask, vein_map, source.stem)
    inverse_scale = 1.0 / max(scale, 1e-9)
    selected_bbox = [
        round(selected.x * inverse_scale),
        round(selected.y * inverse_scale),
        round(selected.width * inverse_scale),
        round(selected.height * inverse_scale),
    ]
    metadata = {
        "source": source.name,
        "source_size": [int(image.shape[1]), int(image.shape[0])],
        "crop_box": [x0, y0, x1, y1],
        "selected_candidate_box": selected_bbox,
        "selection_method": method,
        "candidate_count": len(candidates),
        "candidate_features": [asdict(item) for item in candidates[:4]],
        **metrics,
        "outputs": {
            "rgba": "rubbing_rgba.png",
            "clean": "rubbing_clean.png",
            "silhouette": "silhouette.png",
            "ink_mask": "ink_mask.png",
            "vein_candidate": "vein_candidate.png",
        },
        "notes": (
            "vein_candidate.png is a preliminary visual layer; textile-frequency "
            "suppression and graph validation belong to the next algorithm stage."
        ),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    annotated = small.copy()
    for candidate in candidates[:4]:
        selected_candidate = candidate.label == selected.label
        color = (38, 38, 210) if selected_candidate else (55, 150, 65)
        cv2.rectangle(
            annotated,
            (candidate.x, candidate.y),
            (candidate.x + candidate.width, candidate.y + candidate.height),
            color,
            3,
        )
    return metadata, make_review_card(source.stem, annotated, clean, metrics)


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def make_review_card(
    stem: str,
    annotated: np.ndarray,
    clean: np.ndarray,
    metrics: dict[str, Any],
) -> np.ndarray:
    card = np.full((360, 620, 3), 255, dtype=np.uint8)
    card[48:338, 12:302] = _fit(annotated, 290, 290)
    card[48:338, 318:608] = _fit(clean, 290, 290)
    status = "REVIEW" if metrics["review_recommended"] else "READY"
    color = (35, 35, 205) if status == "REVIEW" else (55, 125, 55)
    cv2.putText(
        card,
        f"{stem}  {status}  Q{metrics['quality_score']:.0f}",
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )
    return card


def make_contact_sheet(cards: list[np.ndarray], columns: int = 3) -> np.ndarray:
    if not cards:
        return np.full((200, 600, 3), 255, dtype=np.uint8)
    rows = math.ceil(len(cards) / columns)
    card_height, card_width = cards[0].shape[:2]
    sheet = np.full((rows * card_height, columns * card_width, 3), 242, dtype=np.uint8)
    for index, card in enumerate(cards):
        row, column = divmod(index, columns)
        sheet[
            row * card_height : (row + 1) * card_height,
            column * card_width : (column + 1) * card_width,
        ] = card
    return sheet


def write_manifest(output_root: Path, results: list[dict[str, Any]]) -> None:
    (output_root / "manifest.json").write_text(
        json.dumps({"version": "rubbing-preprocessor-0.1", "items": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "selection_method",
                "ink_coverage",
                "preliminary_vein_density",
                "quality_score",
                "review_recommended",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item["source"],
                    item["selection_method"],
                    item["ink_coverage"],
                    item["preliminary_vein_density"],
                    item["quality_score"],
                    item["review_recommended"],
                ]
            )
    (output_root / "README.md").write_text(
        """# Processed leaf rubbings

The source photographs remain unchanged in `Real_shoot_leaves`.

Each numbered folder contains:

- `rubbing_rgba.png` — original-color rubbing with a transparent background.
- `rubbing_clean.png` — the isolated rubbing composited on white.
- `silhouette.png` — a filled leaf-region mask for crop and orientation analysis.
- `ink_mask.png` — detected rubbing material without the white textile background.
- `vein_candidate.png` — preliminary bright-ridge response for later algorithm work.
- `metadata.json` — crop coordinates, selection method, quality metrics, and review flag.

`review_sheet.png` is the batch visual check. Red rectangles indicate the selected
rubbing; green rectangles indicate the physical reference leaf. `vein_candidate.png`
is not yet a production vein graph because periodic textile suppression remains part
of the next development stage.
""",
        encoding="utf-8",
    )


def main() -> None:
    app_directory = Path(__file__).resolve().parents[1]
    project_directory = app_directory.parent
    parser = argparse.ArgumentParser(
        description="Separate paired physical leaves and leaf rubbings into reusable PNG layers."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_directory / "Real_shoot_leaves",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_directory / "Real_shoot_leaves_processed",
    )
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    cards: list[np.ndarray] = []
    failures: list[dict[str, str]] = []
    for source in image_files(args.input_dir):
        try:
            metadata, card = process_image(source, args.output_dir)
            results.append(metadata)
            cards.append(card)
            print(
                f"{source.name}: {metadata['selection_method']} · "
                f"Q{metadata['quality_score']:.1f} · "
                f"{'review' if metadata['review_recommended'] else 'ready'}"
            )
        except Exception as exc:  # Keep a complete batch report.
            failures.append({"source": source.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{source.name}: ERROR · {type(exc).__name__}: {exc}")

    write_manifest(args.output_dir, results)
    _write(args.output_dir / "review_sheet.png", make_contact_sheet(cards))
    (args.output_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Completed {len(results)}/{len(results) + len(failures)} images · "
        f"review sheet: {args.output_dir / 'review_sheet.png'}"
    )


if __name__ == "__main__":
    main()
