#!/usr/bin/env python3
"""Collect the compact curated dataset bundled into the deployable image."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
SEED_DIR = APP_DIR / "seed_data"


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_nmns_samples() -> int:
    source_root = PROJECT_DIR / "leaves_origin_images"
    destination_root = SEED_DIR / "leaves_origin_images"
    copied = 0
    for source in sorted(source_root.glob("*")):
        if source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if source.stem.startswith("text_jp_database"):
            continue
        copy_file(source, destination_root / source.name)
        analysis_root = PROJECT_DIR / "analysis_output" / source.stem
        for name in ("metrics.json", "graph.json"):
            analysis_file = analysis_root / name
            if analysis_file.is_file():
                copy_file(
                    analysis_file,
                    SEED_DIR / "analysis_output" / source.stem / name,
                )
        copied += 1
    return copied


def copy_rubbing_samples() -> int:
    source_root = PROJECT_DIR / "Real_shoot_leaves_processed"
    destination_root = SEED_DIR / "Real_shoot_leaves_processed"
    copied = 0
    for directory in sorted(source_root.iterdir() if source_root.is_dir() else []):
        if not directory.is_dir():
            continue
        required = (directory / "rubbing_clean.png", directory / "silhouette.png")
        if not all(path.is_file() for path in required):
            continue
        for name in ("rubbing_clean.png", "silhouette.png", "metadata.json"):
            source = directory / name
            if source.is_file():
                copy_file(source, destination_root / directory.name / name)
        copied += 1
    return copied


def main() -> None:
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    nmns_count = copy_nmns_samples()
    rubbing_count = copy_rubbing_samples()
    manifest = {
        "format": "leaf-language-engine-seed-v1",
        "nmns_samples": nmns_count,
        "rubbing_samples": rubbing_count,
    }
    (SEED_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {nmns_count} NMNS samples and {rubbing_count} rubbing samples.")


if __name__ == "__main__":
    main()
