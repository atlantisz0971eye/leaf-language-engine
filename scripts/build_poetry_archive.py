#!/usr/bin/env python3
"""Build a reproducible leaf / sound poetry archive from an RTF or text file."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from contour_color_codec import (  # noqa: E402
    _jpeg_leaf_mask,
    encode_contour_colour,
    estimate_contour_colour_capacity,
)
from leaf_disturbance import run_leaf_experiment  # noqa: E402
from semantic_disturbance import disturb_text  # noqa: E402


RESOLUTIONS = (3072, 4096, 6144, 8192)
LEVELS = 16
SEMANTIC_CONFIG = {
    "semantic_mode": "breakdown",
    "semantic_loss": 0.60,
    "protect_first_line": True,
}
PIXEL_PROFILES = (
    {"strength": 0.86, "shift": 0.40, "erasure": 0.12, "flip": 0.05, "clustering": 0.90},
    {"strength": 0.72, "shift": 0.34, "erasure": 0.09, "flip": 0.04, "clustering": 0.82},
    {"strength": 0.56, "shift": 0.28, "erasure": 0.06, "flip": 0.025, "clustering": 0.72},
    {"strength": 0.40, "shift": 0.22, "erasure": 0.04, "flip": 0.015, "clustering": 0.62},
)


@dataclass(frozen=True)
class Entry:
    index: int
    label: str
    source_text: str


@dataclass(frozen=True)
class AudioAnalysis:
    source: Path
    frames: list[dict[str, Any]]
    normalized_frames: list[dict[str, float]]
    duration: float
    sample_rate: int
    impacts: int


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".rtf":
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(path)],
            check=True,
            capture_output=True,
        )
        return result.stdout.decode("utf-8", errors="replace")
    return path.read_text(encoding="utf-8-sig")


def _clean_line(line: str) -> str:
    return re.sub(r"^[\s\u00b7\u2022]+", "", line).rstrip()


def parse_entries(path: Path) -> list[Entry]:
    raw = _read_document(path).replace("\r\n", "\n").replace("\r", "\n")
    raw = raw.replace("\u2028", "\n").replace("\u2029", "\n")
    blocks = re.split(r"\n[\t ]*\n+", raw.strip())
    split_blocks: list[list[str]] = []
    for block in blocks:
        lines = [_clean_line(line) for line in block.splitlines() if line.strip()]
        current: list[str] = []
        quote_seen = False
        for line in lines:
            starts_quote = bool(re.match(r"^[\"\u201c\u2018]", line))
            if current and quote_seen and starts_quote:
                split_blocks.append(current)
                current = []
                quote_seen = False
            current.append(line)
            quote_seen = quote_seen or starts_quote
        if current:
            split_blocks.append(current)

    entries: list[Entry] = []
    for lines in split_blocks:
        text = "\n".join(lines).strip()
        if not text:
            continue
        first = lines[0]
        if re.match(r"^[\"\u201c\u2018]", first):
            words = re.findall(r"[\w'\u2019-]+", first, re.UNICODE)[:7]
            label = "Anonymous — " + " ".join(words)
        else:
            label = first.rstrip(" ,")
        entries.append(Entry(len(entries) + 1, label, text))
    return entries


def _decode_pcm(path: Path) -> tuple[np.ndarray, int, float]:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        rate = reader.getframerate()
        count = reader.getnframes()
        raw = reader.readframes(count)
    if width == 1:
        samples = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 3:
        packed = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        values = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        samples = values.astype(np.float32) / 8388608.0
    elif width == 4:
        samples = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width}")
    samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate, count / max(1, rate)


def _percentile(values: Iterable[float], ratio: float) -> float:
    array = np.asarray(list(values), np.float32)
    if not array.size:
        return 0.0
    return float(np.quantile(array, ratio))


def _normalize(frames: list[dict[str, float]], key: str) -> None:
    low = _percentile((frame[key] for frame in frames), 0.06)
    high = _percentile((frame[key] for frame in frames), 0.96)
    span = max(1e-8, high - low)
    for frame in frames:
        frame[key] = max(0.0, min(1.0, (frame[key] - low) / span))


def analyze_wav(path: Path, levels: int = LEVELS, max_frames: int = 2400) -> AudioAnalysis:
    samples, sample_rate, duration = _decode_pcm(path)
    stride = max(1, sample_rate // 24000)
    samples = np.ascontiguousarray(samples[::stride], dtype=np.float32)
    effective_rate = sample_rate / stride
    frame_size, hop = 1024, 512
    if len(samples) < frame_size:
        samples = np.pad(samples, (0, frame_size - len(samples)))
    bass_window = max(5, int(round(effective_rate / 220.0)))
    if bass_window % 2 == 0:
        bass_window += 1
    low_signal = cv2.blur(samples.reshape(-1, 1), (1, bass_window)).reshape(-1)
    high_signal = samples - low_signal
    frame_count = max(1, math.floor((len(samples) - frame_size) / hop) + 1)
    frames: list[dict[str, float]] = []
    last_energy = last_noise = 0.0
    for index in range(frame_count):
        start = index * hop
        sample = samples[start : start + frame_size]
        low = low_signal[start : start + frame_size]
        high = high_signal[start : start + frame_size]
        energy = float(np.sqrt(np.mean(sample * sample)))
        noise = float(np.sqrt(np.mean(high * high)))
        peak = float(np.max(np.abs(sample), initial=0.0))
        crossings = int(np.count_nonzero((sample[1:] >= 0) != (sample[:-1] >= 0)))
        flux = max(0.0, energy - last_energy) + max(0.0, noise - last_noise) * 0.55
        frames.append(
            {
                "time": index * hop / effective_rate,
                "energy": energy,
                "peak": peak,
                "bass": float(np.sqrt(np.mean(low * low))),
                "brightness": min(1.0, crossings / frame_size * 17.0),
                "noise": noise,
                "flux": flux,
                "impact": 0.0,
            }
        )
        last_energy, last_noise = energy, noise
    for key in ("energy", "peak", "bass", "brightness", "noise", "flux"):
        _normalize(frames, key)
    flux_gate = _percentile((frame["flux"] for frame in frames), 0.90)
    energy_gate = _percentile((frame["energy"] for frame in frames), 0.40)
    gap = max(2, math.floor(0.075 * effective_rate / hop))
    last_impact = -gap
    for index in range(1, len(frames) - 1):
        frame = frames[index]
        if (
            frame["flux"] >= flux_gate
            and frame["energy"] >= energy_gate
            and frame["flux"] >= frames[index - 1]["flux"]
            and frame["flux"] >= frames[index + 1]["flux"]
            and index - last_impact >= gap
        ):
            frame["impact"] = max(frame["flux"], frame["peak"] * 0.82)
            last_impact = index

    group = max(1, math.ceil(len(frames) / max_frames))
    compact: list[dict[str, float]] = []
    for start in range(0, len(frames), group):
        section = frames[start : start + group]
        compact.append(
            {
                "time": section[0]["time"],
                "energy": float(np.mean([frame["energy"] for frame in section])),
                "peak": max(frame["peak"] for frame in section),
                "bass": float(np.mean([frame["bass"] for frame in section])),
                "brightness": float(np.mean([frame["brightness"] for frame in section])),
                "noise": float(np.mean([frame["noise"] for frame in section])),
                "impact": max(frame["impact"] for frame in section),
            }
        )
    maximum = levels - 1
    quantized = [
        {
            "time": frame["time"],
            "energy": round(frame["energy"] * maximum),
            "peak": round(frame["peak"] * maximum),
            "bass": round(frame["bass"] * maximum),
            "brightness": round(frame["brightness"] * maximum),
            "noise": round(frame["noise"] * maximum),
            "impact": max(1, round(frame["impact"] * maximum)) if frame["impact"] > 0 else 0,
        }
        for frame in compact
    ]
    return AudioAnalysis(
        source=path,
        frames=quantized,
        normalized_frames=frames,
        duration=duration,
        sample_rate=sample_rate,
        impacts=sum(1 for frame in frames if frame["impact"] > 0),
    )


def _slug(value: str, fallback: str, limit: int = 70) -> str:
    value = unicodedata.normalize("NFKC", value).strip().lower()
    value = re.sub(r"[^\w\-]+", "-", value, flags=re.UNICODE)
    value = re.sub(r"-+", "-", value).strip("-_")
    return (value or fallback)[:limit].rstrip("-_")


def _rgba(path: Path, mask: np.ndarray | None = None) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read {path}")
    if mask is None:
        mask = _jpeg_leaf_mask(image)
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise OSError(f"Could not write transparent PNG {path}")


def _source_mask(silhouette_path: Path, shape: tuple[int, int]) -> np.ndarray:
    silhouette = cv2.imread(str(silhouette_path), cv2.IMREAD_UNCHANGED)
    if silhouette is None:
        raise OSError(f"Could not read {silhouette_path}")
    if silhouette.ndim == 3:
        silhouette = cv2.cvtColor(silhouette[..., :3], cv2.COLOR_BGR2GRAY)
    mask = np.where(silhouette > 20, 255, 0).astype(np.uint8)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def make_waveform_png(analysis: AudioAnalysis, destination: Path) -> None:
    width, height = 1800, 520
    canvas = np.zeros((height, width, 4), np.uint8)
    frames = analysis.normalized_frames
    palette = {
        "energy": (115, 200, 149, 235),
        "peak": (210, 173, 82, 235),
        "noise": (119, 169, 183, 235),
    }
    for key, colour in palette.items():
        points = np.asarray(
            [
                (
                    round(index / max(1, len(frames) - 1) * (width - 1)),
                    round(height - 18 - frame[key] * (height - 36)),
                )
                for index, frame in enumerate(frames)
            ],
            np.int32,
        )
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, colour, 2, cv2.LINE_AA)
    for index, frame in enumerate(frames):
        if frame["impact"] <= 0:
            continue
        x = round(index / max(1, len(frames) - 1) * (width - 1))
        y = min(height - 1, round(18 + frame["impact"] * 130))
        cv2.line(canvas, (x, 0), (x, y), (198, 82, 95, 225), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{analysis.source.name}  {analysis.duration:.1f}s  {analysis.impacts} impacts",
        (28, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (64, 117, 83, 230),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise OSError(f"Could not save {destination}")


def _balanced_assignments(items: list[Path], count: int, rng: random.Random) -> list[Path]:
    assigned: list[Path] = []
    while len(assigned) < count:
        cycle = list(items)
        rng.shuffle(cycle)
        assigned.extend(cycle)
    return assigned[:count]


def _select_resolution(specimen: Path, text: str) -> tuple[int, dict[str, Any]]:
    source = specimen / "rubbing_clean.png"
    silhouette = specimen / "silhouette.png"
    estimate = estimate_contour_colour_capacity(
        source_path=source,
        text=text,
        silhouette_path=silhouette,
        resolutions=RESOLUTIONS,
    )
    selected = next((row for row in estimate["resolutions"] if row["exportable"]), None)
    if selected is None:
        raise ValueError("Text exceeds safe capacity through 8192px")
    return int(selected["resolution"]), estimate


def _archive_one(
    entry: Entry,
    specimen: Path,
    audio: AudioAnalysis,
    output_root: Path,
) -> dict[str, Any]:
    semantic = disturb_text(entry.source_text, audio.frames, SEMANTIC_CONFIG)
    carrier_text = str(semantic["disturbed_text"])
    resolution, capacity = _select_resolution(specimen, carrier_text)
    leaf_slug = _slug(specimen.name, "leaf")
    sound_slug = _slug(audio.source.stem, "sound")
    entry_slug = _slug(entry.label, f"entry-{entry.index:03d}")
    folder = output_root / "entries" / f"{entry.index:03d}__{entry_slug}__leaf-{leaf_slug}__sound-{sound_slug}"
    folder.mkdir(parents=True, exist_ok=False)
    image_files = {
        "original": "01_original.png",
        "clean": "02_clean_encoding.png",
        "disturbed": "03_disturbed_hard_decode.png",
        "difference": "04_difference.png",
        "confidence": "05_confidence.png",
    }
    waveform_file = f"06_waveform__{sound_slug}.png"
    text_file = "07_source_and_decoded.txt"
    record_file = "08_record.json"
    clean_carrier = folder / "_prepared_clean.jpg"
    encode_meta = encode_contour_colour(
        source_path=specimen / "rubbing_clean.png",
        text=carrier_text,
        output_path=clean_carrier,
        silhouette_path=specimen / "silhouette.png",
        target_long_side=resolution,
        jpeg_quality=96,
        cell_spacing=14,
        chroma_delta=28.0,
    )

    selected_result: dict[str, Any] | None = None
    selected_profile: dict[str, float] | None = None
    with tempfile.TemporaryDirectory(prefix=f"leaf-archive-{entry.index:03d}-") as temp_name:
        temp_root = Path(temp_name)
        for attempt, profile in enumerate(PIXEL_PROFILES, 1):
            attempt_dir = temp_root / f"attempt-{attempt}"
            result = run_leaf_experiment(
                specimen_dir=specimen,
                text=carrier_text,
                frames=audio.frames,
                levels=LEVELS,
                config={**profile, **SEMANTIC_CONFIG},
                output_dir=attempt_dir,
                target_long_side=resolution,
                clean_carrier_path=clean_carrier,
            )
            if selected_result is None or bool((result.get("disturbed_decode") or {}).get("valid")):
                selected_result = result
                selected_profile = profile
                for name in ("original", "clean", "disturbed", "difference", "confidence"):
                    shutil.copy2(attempt_dir / result["outputs"][name], folder / image_files[name])
            if bool((result.get("disturbed_decode") or {}).get("valid")):
                break
    clean_carrier.unlink(missing_ok=True)
    clean_carrier.with_suffix(".json").unlink(missing_ok=True)
    if selected_result is None or selected_profile is None:
        raise RuntimeError("No disturbance result was produced")

    original = cv2.imread(str(folder / image_files["original"]), cv2.IMREAD_COLOR)
    if original is None:
        raise OSError("Original archive image could not be read")
    _rgba(folder / image_files["original"], _source_mask(specimen / "silhouette.png", original.shape[:2]))
    clean_image = cv2.imread(str(folder / image_files["clean"]), cv2.IMREAD_COLOR)
    if clean_image is None:
        raise OSError("Clean archive image could not be read")
    carrier_mask = _jpeg_leaf_mask(clean_image)
    for name in ("clean", "disturbed", "difference", "confidence"):
        _rgba(folder / image_files[name], carrier_mask)
    make_waveform_png(audio, folder / waveform_file)

    decoded_transport = selected_result.get("disturbed_decode") or {}
    decoded_text = str(decoded_transport.get("text") or carrier_text)
    source_record = (
        "SOURCE\n"
        + entry.source_text
        + "\n\nHARD-DECODE TARGET\n"
        + carrier_text
        + "\n\nDECODED FROM DISTURBED PNG\n"
        + decoded_text
        + "\n"
    )
    (folder / text_file).write_text(source_record, encoding="utf-8")
    record = {
        "archive_version": "Ga1a-poetry-archive-1.0",
        "entry_index": entry.index,
        "label": entry.label,
        "source_text": entry.source_text,
        "hard_decode_text": carrier_text,
        "decoded_from_disturbed_png": decoded_text,
        "transport_valid": bool(decoded_transport.get("valid")),
        "leaf": specimen.name,
        "leaf_source": str(specimen / "rubbing_clean.png"),
        "sound": audio.source.name,
        "sound_source": str(audio.source),
        "audio": {
            "duration_seconds": round(audio.duration, 3),
            "sample_rate": audio.sample_rate,
            "feature_frames": len(audio.frames),
            "impacts": audio.impacts,
        },
        "resolution": resolution,
        "semantic": semantic,
        "pixel_profile": selected_profile,
        "carrier": selected_result.get("carrier", {}),
        "capacity": capacity,
        "encoding": encode_meta,
        "files": {
            "original": image_files["original"],
            "clean_encoding": image_files["clean"],
            "disturbed": image_files["disturbed"],
            "difference": image_files["difference"],
            "confidence": image_files["confidence"],
            "waveform": waveform_file,
            "text": text_file,
            "record": record_file,
        },
    }
    (folder / record_file).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"folder": str(folder.relative_to(output_root)), **record}


def _find_text_file(root: Path) -> Path:
    candidates = sorted(root.glob("text_demo_copy*"))
    if not candidates:
        raise FileNotFoundError(f"No text_demo_copy file was found in {root}")
    return candidates[0]


def _write_manifest(output_root: Path, records: list[dict[str, Any]], errors: list[dict[str, Any]], seed: int) -> None:
    manifest_dir = output_root / "00_manifest"
    manifest_dir.mkdir(exist_ok=True)
    manifest = {
        "archive_version": "Ga1a-poetry-archive-1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": seed,
        "entries_complete": len(records),
        "entries_failed": len(errors),
        "records": records,
        "errors": errors,
    }
    (manifest_dir / "archive_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = (
        "entry_index",
        "label",
        "leaf",
        "sound",
        "resolution",
        "transport_valid",
        "source_text",
        "hard_decode_text",
        "decoded_from_disturbed_png",
        "folder",
    )
    with (manifest_dir / "archive_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fields})
    (manifest_dir / "README.txt").write_text(
        "Ga1a Poetry Leaf Archive\n\n"
        "Each numbered folder contains six transparent PNG files, the source and decoded text, "
        "and a machine-readable record.json. Clean encoding is the Hard-decode target before "
        "audio pixel disturbance. Disturbed is the same carrier after WAV-derived pixel disturbance.\n",
        encoding="utf-8",
    )


def build_archive(args: argparse.Namespace) -> Path:
    text_path = args.text_file or _find_text_file(ROOT_DIR)
    entries = parse_entries(text_path)
    if args.limit:
        entries = entries[: args.limit]
    specimens = sorted(path.parent for path in args.rubbing_dir.glob("*/rubbing_clean.png") if (path.parent / "silhouette.png").exists())
    sounds = sorted(args.sound_dir.glob("*.wav")) + sorted(args.sound_dir.glob("*.WAV"))
    if not entries:
        raise ValueError("The text document does not contain any entries")
    if not specimens:
        raise FileNotFoundError(f"No processed rubbing leaves were found in {args.rubbing_dir}")
    if not sounds:
        raise FileNotFoundError(f"No WAV files were found in {args.sound_dir}")

    rng = random.Random(args.seed)
    leaf_assignments = _balanced_assignments(specimens, len(entries), rng)
    sound_assignments = _balanced_assignments(sounds, len(entries), rng)
    output_root = args.output_dir or ROOT_DIR / "poetry_leaf_archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "entries").mkdir()
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    audio_cache: dict[Path, AudioAnalysis] = {}
    print(f"Archive: {output_root}", flush=True)
    print(f"Entries: {len(entries)} | Leaves: {len(specimens)} | Sounds: {len(sounds)}", flush=True)
    for entry, specimen, sound in zip(entries, leaf_assignments, sound_assignments):
        print(f"[{entry.index:03d}/{len(entries):03d}] {entry.label} | leaf {specimen.name} | {sound.name}", flush=True)
        try:
            audio = audio_cache.get(sound)
            if audio is None:
                audio = analyze_wav(sound)
                audio_cache[sound] = audio
            record = _archive_one(entry, specimen, audio, output_root)
            records.append(record)
            print(
                f"  done: {record['resolution']}px | fidelity {record['semantic']['fidelity']}% | "
                f"transport {'valid' if record['transport_valid'] else 'failed'}",
                flush=True,
            )
        except Exception as exc:
            error = {
                "entry_index": entry.index,
                "label": entry.label,
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            print(f"  ERROR: {error['error']}", flush=True)
        _write_manifest(output_root, records, errors, args.seed)
    print(f"Complete: {len(records)} archived, {len(errors)} failed", flush=True)
    return output_root


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, help="RTF or UTF-8 text file; default: text_demo_copy*")
    parser.add_argument("--rubbing-dir", type=Path, default=ROOT_DIR / "Real_shoot_leaves_processed")
    parser.add_argument("--sound-dir", type=Path, default=ROOT_DIR / "rubbing_sounds")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260807, help="Reproducible random assignment seed")
    parser.add_argument("--limit", type=int, default=0, help="Archive only the first N entries for testing")
    return parser


if __name__ == "__main__":
    build_archive(argument_parser().parse_args())
