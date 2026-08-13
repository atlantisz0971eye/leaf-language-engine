"""Build lightweight public exhibition assets from a poetry leaf archive."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
ARCHIVE_ROOT = WORKSPACE_DIR / "poetry_leaf_archive"
SOUND_ROOT = WORKSPACE_DIR / "rubbing_sounds"
OUTPUT_ROOT = PROJECT_DIR / "static" / "archive"
ASSET_ROOT = OUTPUT_ROOT / "assets"

IMAGE_KEYS = {
    "original": "original",
    "clean_encoding": "clean",
    "disturbed": "disturbed",
    "difference": "difference",
    "confidence": "confidence",
    "waveform": "waveform",
}


def latest_archive() -> Path:
    candidates = sorted(
        directory
        for directory in ARCHIVE_ROOT.iterdir()
        if directory.is_dir() and (directory / "00_manifest" / "archive_manifest.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No poetry archive was found under {ARCHIVE_ROOT}")
    return candidates[-1]


def convert_image(source: Path, destination: Path, max_side: int, quality: int = 86) -> None:
    with Image.open(source) as image:
        image.load()
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGBA" if "transparency" in image.info else "RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, "WEBP", quality=quality, method=6)


def convert_audio(source: Path, destination: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-b:a",
            "72k",
            str(destination),
        ],
        check=True,
    )
    return True


def public_record(record: dict, entry_directory: Path, identifier: str) -> dict:
    destination = ASSET_ROOT / identifier
    files = record.get("files", {})
    public_images: dict[str, str] = {}
    for source_key, public_key in IMAGE_KEYS.items():
        filename = files.get(source_key)
        if not filename:
            continue
        source = entry_directory / filename
        if not source.is_file():
            continue
        max_side = 1500 if public_key not in {"waveform"} else 1100
        target = destination / f"{public_key}.webp"
        convert_image(source, target, max_side, 88 if public_key == "original" else 84)
        public_images[public_key] = f"/static/archive/assets/{identifier}/{target.name}"

    original_source = entry_directory / files.get("original", "")
    if original_source.is_file():
        thumb_target = destination / "thumb.webp"
        convert_image(original_source, thumb_target, 430, 82)
        public_images["thumb"] = f"/static/archive/assets/{identifier}/{thumb_target.name}"

    sound_name = str(record.get("sound", ""))
    audio_target = destination / "rubbing.mp3"
    has_audio = convert_audio(SOUND_ROOT / sound_name, audio_target)

    semantic = record.get("semantic", {})
    carrier = record.get("carrier", {})
    audio = record.get("audio", {})
    return {
        "id": identifier,
        "index": int(record.get("entry_index", 0)),
        "label": str(record.get("label", "Untitled")),
        "sourceText": str(record.get("source_text", "")),
        "decodedText": str(record.get("hard_decode_text", "")),
        "leaf": str(record.get("leaf", "")),
        "sound": sound_name,
        "audio": f"/static/archive/assets/{identifier}/rubbing.mp3" if has_audio else None,
        "images": public_images,
        "metrics": {
            "fidelity": round(float(semantic.get("fidelity", 0)), 1),
            "affectedWords": int(semantic.get("affected_words", 0)),
            "changedCells": int(carrier.get("changed_cells", 0)),
            "confidence": round(float(carrier.get("mean_confidence", 0)), 1),
            "duration": round(float(audio.get("duration_seconds", 0)), 1),
            "impacts": int(audio.get("impacts", 0)),
        },
    }


def main() -> None:
    archive = latest_archive()
    manifest_path = archive / "00_manifest" / "archive_manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = source_manifest.get("records", [])
    if not records:
        raise ValueError(f"No archive records were found in {manifest_path}")

    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    public_records = []
    for record in sorted(records, key=lambda item: int(item.get("entry_index", 0))):
        identifier = f"leaf-{int(record['entry_index']):02d}"
        entry_directory = archive / str(record["folder"])
        public_records.append(public_record(record, entry_directory, identifier))
        print(f"Prepared {identifier}: {record.get('label', 'Untitled')}")

    manifest = {
        "version": "Ga1a-web-archive-1.0",
        "sourceArchive": archive.name,
        "count": len(public_records),
        "entries": public_records,
    }
    (OUTPUT_ROOT / "archive.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(public_records)} entries to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
