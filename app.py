from __future__ import annotations

import base64
import csv
import gc
import hashlib
import json
import mimetypes
import os
import re
import shutil
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse
from uuid import uuid4

import cv2
import numpy as np

from leaf_analysis import AnalysisSettings, analyze_image, export_result, public_result
from document_engine import analyze_document, decode_document
from encoding_renderer import encoding_export_filename, render_encoding_png
from language_engine import analyze_word
from language_database import LanguageDatabase
from leaf_codec import decode_packet, encode_text
from raster_decoder import decode_encoding_png
from leaf_disturbance import run_leaf_experiment
from semantic_disturbance import disturb_text
from contour_color_codec import (
    decode_contour_colour,
    encode_contour_colour,
    estimate_contour_colour_capacity,
)
from specimen_ingest import ingest_specimen
from nmns_ingest import isolate_nmns_scan


APP_DIR = Path(__file__).resolve().parent
LOCAL_PROJECT_DIR = APP_DIR.parent
DATA_DIR = Path(os.environ.get("LEAF_DATA_DIR", str(LOCAL_PROJECT_DIR))).expanduser().resolve()
SEED_DATA_DIR = APP_DIR / "seed_data"
IMAGE_DIR = DATA_DIR / "leaves_origin_images"
OUTPUT_DIR = DATA_DIR / "analysis_output"
RUBBING_DIR = DATA_DIR / "Real_shoot_leaves_processed"
UPLOADED_SPECIMEN_DIR = DATA_DIR / "uploaded_leaf_specimens"
COLOUR_OUTPUT_DIR = OUTPUT_DIR / "colour_codec"
DOCUMENT_OUTPUT_DIR = OUTPUT_DIR / "document_codec"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio_disturbance"
STATIC_DIR = APP_DIR / "static"
LANGUAGE_LEXICON = APP_DIR / "language_data" / "root_lexicon.json"
LANGUAGE_DATABASE = APP_DIR / "language_data" / "language.db"
HOST = os.environ.get("LEAF_TOOL_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", os.environ.get("LEAF_TOOL_PORT", "8765")))
OPEN_BROWSER = os.environ.get(
    "LEAF_OPEN_BROWSER",
    "1" if HOST in {"127.0.0.1", "localhost"} else "0",
).lower() in {"1", "true", "yes"}
MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("LEAF_MAX_CONCURRENT_JOBS", "1")))
RUBBING_RESOLUTIONS = (2048, 3072, 4096, 6144, 8192)
MAX_RUBBING_RESOLUTION = max(
    2048,
    min(8192, int(os.environ.get("LEAF_MAX_RUBBING_RESOLUTION", "8192"))),
)
STYLE_ANALYSIS_CACHE: dict[str, dict] = {}
BACKGROUND_JOBS: dict[str, dict] = {}
BACKGROUND_JOB_LOCK = threading.Lock()
BACKGROUND_JOB_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_UPLOAD_PIXELS = 120_000_000


def prepare_runtime_data() -> None:
    """Create writable runtime folders and copy bundled examples once."""
    for directory in (
        IMAGE_DIR,
        OUTPUT_DIR,
        RUBBING_DIR,
        UPLOADED_SPECIMEN_DIR,
        COLOUR_OUTPUT_DIR,
        DOCUMENT_OUTPUT_DIR,
        AUDIO_OUTPUT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if not SEED_DATA_DIR.is_dir() or DATA_DIR == LOCAL_PROJECT_DIR:
        return

    for relative in (
        "leaves_origin_images",
        "analysis_output",
        "Real_shoot_leaves_processed",
    ):
        source = SEED_DATA_DIR / relative
        destination = DATA_DIR / relative
        if not source.is_dir():
            continue
        outdated_analysis_dirs: set[str] = set()
        if relative == "analysis_output":
            for seed_graph in source.glob("*/graph.json"):
                target_graph = destination / seed_graph.relative_to(source)
                try:
                    seed_version = json.loads(seed_graph.read_text(encoding="utf-8")).get("version")
                    target_version = json.loads(target_graph.read_text(encoding="utf-8")).get("version")
                except (OSError, ValueError, AttributeError):
                    seed_version, target_version = "seed", None
                if seed_version != target_version:
                    outdated_analysis_dirs.add(seed_graph.parent.name)
        for item in source.rglob("*"):
            if not item.is_file():
                continue
            target = destination / item.relative_to(source)
            if target.exists():
                if relative != "analysis_output" or item.parent.name not in outdated_analysis_dirs:
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def create_background_job(kind: str) -> str:
    identifier = uuid4().hex[:16]
    with BACKGROUND_JOB_LOCK:
        BACKGROUND_JOBS[identifier] = {
            "id": identifier,
            "kind": kind,
            "status": "queued",
            "progress": 2,
            "stage": "Queued",
            "created_at": time.time(),
        }
        if len(BACKGROUND_JOBS) > 80:
            oldest = sorted(
                BACKGROUND_JOBS,
                key=lambda key: float(BACKGROUND_JOBS[key].get("created_at", 0)),
            )[:-60]
            for key in oldest:
                BACKGROUND_JOBS.pop(key, None)
    return identifier


def update_background_job(identifier: str, **changes: object) -> None:
    with BACKGROUND_JOB_LOCK:
        if identifier in BACKGROUND_JOBS:
            BACKGROUND_JOBS[identifier].update(changes)


def public_background_job(identifier: str) -> dict:
    with BACKGROUND_JOB_LOCK:
        job = BACKGROUND_JOBS.get(identifier)
        if job is None:
            raise FileNotFoundError(identifier)
        return dict(job)


def launch_background_job(
    identifier: str,
    target: Callable[..., None],
    *args: object,
) -> None:
    def execute() -> None:
        update_background_job(identifier, stage="Waiting for analysis slot", progress=3)
        with BACKGROUND_JOB_SLOTS:
            target(identifier, *args)

    threading.Thread(target=execute, daemon=True).start()


def colour_specimen_directories() -> list[tuple[Path, str]]:
    directories: list[tuple[Path, str]] = []
    if RUBBING_DIR.is_dir():
        directories.extend(
            (path, "curated")
            for path in RUBBING_DIR.iterdir()
            if path.is_dir() and (path / "rubbing_clean.png").is_file()
        )
    if UPLOADED_SPECIMEN_DIR.is_dir():
        directories.extend(
            (path, "uploaded")
            for path in UPLOADED_SPECIMEN_DIR.iterdir()
            if path.is_dir() and (path / "rubbing_clean.png").is_file()
        )
    return sorted(directories, key=lambda item: (item[1], item[0].name))


def colour_specimen_record(directory: Path, library: str) -> dict:
    source = directory / "rubbing_clean.png"
    silhouette = directory / "silhouette.png"
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2] if image is not None else (0, 0)
    metadata = {}
    metadata_path = directory / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            metadata = {}
    uploaded = library == "uploaded"
    return {
        "id": directory.name,
        "name": (
            str(metadata.get("display_name") or directory.name)
            if uploaded
            else f"Real rubbing {directory.name}"
        ),
        "width": width,
        "height": height,
        "library": library,
        "source_type": str(metadata.get("source_type") or ("curated-rubbing" if not uploaded else "uploaded-leaf")),
        "quality_score": metadata.get("quality_score"),
        "texture_coverage": metadata.get("texture_coverage"),
        "ready": bool(metadata.get("encode_ready", True)),
        "review_recommended": bool(metadata.get("review_recommended", False)),
        "source_url": f"/colour-source/{directory.name}",
        "silhouette_ready": silhouette.is_file(),
    }


def colour_specimens() -> list[dict]:
    items = []
    for directory, library in colour_specimen_directories():
        source = directory / "rubbing_clean.png"
        silhouette = directory / "silhouette.png"
        if not source.is_file() or not silhouette.is_file():
            continue
        record = colour_specimen_record(directory, library)
        if record["source_type"] in {
            "nmns-transparent",
            "transparent-leaf",
            "opaque-cleared-leaf",
        }:
            continue
        items.append(record)
    return items


def safe_colour_specimen(sample: str) -> tuple[Path, Path]:
    identifier = Path(str(sample)).name
    for root in (RUBBING_DIR, UPLOADED_SPECIMEN_DIR):
        directory = (root / identifier).resolve()
        if directory.parent != root.resolve():
            continue
        source, silhouette = directory / "rubbing_clean.png", directory / "silhouette.png"
        if source.is_file() and silhouette.is_file():
            return source, silhouette
    raise FileNotFoundError(sample)


def decode_specimen_source_upload(body: dict) -> tuple[bytes, str]:
    encoded = str(body.get("data", ""))
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("The uploaded leaf image data is invalid.") from exc
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Leaf image files must be between 1 byte and 32 MB.")
    filename = Path(str(body.get("filename", "uploaded-leaf"))).name
    suffix = Path(filename).suffix.lower()
    detected = ".png" if raw.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg" if raw.startswith(b"\xff\xd8") else ""
    if suffix not in {".jpg", ".jpeg", ".png"} or not detected:
        raise ValueError("Upload a valid JPG, JPEG, or PNG leaf image.")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("The uploaded leaf image could not be decoded.")
    height, width = image.shape[:2]
    if min(width, height) < 128:
        raise ValueError("The leaf image must be at least 128 × 128 pixels.")
    if width * height > MAX_UPLOAD_PIXELS:
        raise ValueError("The leaf image exceeds the 120-megapixel safety limit.")
    return raw, filename


def decode_colour_upload(body: dict) -> tuple[bytes, str]:
    encoded = str(body.get("data", ""))
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("The uploaded image data is invalid.") from exc
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Image files must be between 1 byte and 32 MB.")
    suffix = ".png" if raw.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg" if raw.startswith(b"\xff\xd8") else ""
    if not suffix:
        raise ValueError("Only JPG and PNG colour-carrier files are supported.")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded colour-carrier image could not be decoded.")
    if image.shape[0] * image.shape[1] > MAX_UPLOAD_PIXELS:
        raise ValueError("The image exceeds the 120-megapixel safety limit.")
    return raw, suffix


def colour_cache_path(sample: str, text: str, resolution: int) -> Path:
    digest = hashlib.sha256(f"v3\0{sample}\0{resolution}\0{text}".encode("utf-8")).hexdigest()[:16]
    return COLOUR_OUTPUT_DIR / f"colour_{Path(sample).name}_{resolution}px_{digest}.jpg"


def prepare_colour_document(
    sample: str,
    text: str,
    resolution: int,
    progress: Callable[[int, str], None] | None = None,
) -> tuple[dict, dict, Path | None]:
    if progress:
        progress(12, "Reading leaf")
    source, silhouette = safe_colour_specimen(sample)
    resolutions = RUBBING_RESOLUTIONS
    if progress:
        progress(28, "Measuring capacity")
    capacity = estimate_contour_colour_capacity(
        source,
        text,
        silhouette,
        resolutions=resolutions,
    )
    specimen_metadata = {}
    specimen_metadata_path = source.parent / "metadata.json"
    if specimen_metadata_path.is_file():
        try:
            specimen_metadata = json.loads(specimen_metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            specimen_metadata = {}
    uploaded_quality = str(specimen_metadata.get("version", "")).startswith("unified-specimen-ingest")
    quality_pass = bool(specimen_metadata.get("encode_ready", True)) if uploaded_quality else True
    capacity["source_quality"] = {
        "pass": quality_pass,
        "score": specimen_metadata.get("quality_score"),
        "source_type": specimen_metadata.get("source_type", "curated-rubbing"),
        "texture_coverage": specimen_metadata.get("texture_coverage"),
        "review_recommended": bool(specimen_metadata.get("review_recommended", False)),
    }
    capacity["runtime_limit"] = {
        "max_resolution": MAX_RUBBING_RESOLUTION,
        "safe_mode": MAX_RUBBING_RESOLUTION < max(RUBBING_RESOLUTIONS),
        "message": (
            f"This server safely supports leaf-rubbing exports up to {MAX_RUBBING_RESOLUTION}px. "
            "NMNS vein-graph exports keep their full resolution range."
        ),
    }
    for row in capacity["resolutions"]:
        row["capacity_exportable"] = bool(row["exportable"])
        row["runtime_safe"] = int(row["resolution"]) <= MAX_RUBBING_RESOLUTION
        row["resource_limited"] = not row["runtime_safe"]
        row["exportable"] = bool(row["exportable"] and row["runtime_safe"])
    if not quality_pass:
        for row in capacity["resolutions"]:
            row["exportable"] = False
            row["source_quality_pass"] = False
        capacity["minimum_safe_resolution"] = None
        capacity["recommendation"] = (
            "The uploaded leaf needs a clearer crop or stronger visible texture before it can carry text."
        )
    elif not any(row["exportable"] for row in capacity["resolutions"]):
        capacity["minimum_safe_resolution"] = None
        capacity["recommendation"] = (
            f"This text needs more space than the current server-safe limit of "
            f"{MAX_RUBBING_RESOLUTION}px. Shorten the text or use an NMNS scan."
        )
    else:
        safe = next(row for row in capacity["resolutions"] if row["exportable"])
        capacity["minimum_safe_resolution"] = safe["resolution"]
        capacity["recommendation"] = (
            f"Use at least {safe['resolution']}px for this text and specimen."
        )
    selected = next(
        (row for row in capacity["resolutions"] if int(row["resolution"]) == int(resolution)),
        None,
    )
    if selected is None:
        raise ValueError("Choose one of the supported colour-carrier resolutions.")
    destination = None
    metadata = None
    if selected["exportable"]:
        if progress:
            progress(54, "Building preview")
        destination = colour_cache_path(sample, text, resolution)
        if not destination.is_file():
            metadata = encode_contour_colour(
                source,
                text,
                destination,
                silhouette_path=silhouette,
                target_long_side=resolution,
                jpeg_quality=96,
            )
        else:
            metadata_path = destination.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
    if progress:
        progress(94, "Final check")
    return capacity, selected, destination


def run_specimen_job(
    identifier: str,
    raw: bytes,
    filename: str,
    source_hint: str,
) -> None:
    if source_hint == "nmns":
        run_nmns_specimen_job(identifier, raw, filename)
        return
    try:
        update_background_job(
            identifier,
            status="running",
            progress=12,
            stage="Reading image",
        )
        update_background_job(identifier, progress=34, stage="Finding leaf")
        metadata = ingest_specimen(
            raw,
            filename,
            UPLOADED_SPECIMEN_DIR,
            source_hint=source_hint,
        )
        update_background_job(identifier, progress=78, stage="Measuring texture")
        directory = UPLOADED_SPECIMEN_DIR / str(metadata["id"])
        specimen = colour_specimen_record(directory, "uploaded")
        update_background_job(
            identifier,
            status="complete",
            progress=100,
            stage="Ready",
            result={
                "ok": True,
                "specimen": specimen,
                "analysis": metadata,
                "message": (
                    "Ready for text"
                    if specimen["ready"]
                    else "Try a cleaner crop"
                ),
            },
        )
    except Exception as exc:
        update_background_job(
            identifier,
            status="error",
            progress=100,
            stage="Could not analyze",
            error=f"{type(exc).__name__}: {exc}",
        )


def nmns_specimen_record(image_path: Path) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2] if image is not None else (0, 0)
    status = specimen_status(image_path)
    preview_names = ("original", "vessel_score", "mask", "skeleton", "overlay", "primary", "heatmap")
    preview_directory = OUTPUT_DIR / image_path.stem
    previews = {
        name: f"/analysis-preview/{image_path.stem}/{name}.png"
        for name in preview_names
        if (preview_directory / f"{name}.png").is_file()
    }
    source_plate = next(
        (item for item in (preview_directory / "source.jpg", preview_directory / "source.png") if item.is_file()),
        None,
    )
    if source_plate is not None:
        previews["source"] = f"/analysis-preview/{image_path.stem}/{source_plate.name}"
    return {
        "id": image_path.name,
        "name": image_path.stem.replace("_", " "),
        "filename": image_path.name,
        "width": width,
        "height": height,
        "library": "nmns",
        "source_type": "nmns-vein-graph",
        "source_url": f"/source/{image_path.name}",
        "analysis_urls": previews,
        **status,
    }


def run_nmns_specimen_job(
    identifier: str,
    raw: bytes,
    filename: str,
) -> None:
    destination = None
    created = False
    try:
        update_background_job(identifier, status="running", progress=10, stage="Reading scan")
        isolation = isolate_nmns_scan(raw)
        update_background_job(identifier, progress=24, stage="Isolating leaf from plate")
        ok, encoded = cv2.imencode(".png", isolation.image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise ValueError("The NMNS scan could not be normalized.")
        normalized = encoded.tobytes()
        upload_name = normalized_upload_name(f"{Path(filename).stem}.png", normalized)
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        destination = IMAGE_DIR / upload_name
        created = not destination.exists()
        if created:
            destination.write_bytes(normalized)

        update_background_job(identifier, progress=38, stage="Extracting multiscale veins")
        safe_settings = AnalysisSettings(1100, 82.0, 14.0, 10)
        analyzed, selected_settings = adaptive_analysis(destination, safe_settings)
        update_background_job(identifier, progress=78, stage="Building vein graph")
        export_destination = export_result(analyzed, OUTPUT_DIR)
        source_suffix = ".png" if str(filename).lower().endswith(".png") else ".jpg"
        (export_destination / f"source{source_suffix}").write_bytes(raw)
        (export_destination / "isolation.json").write_text(
            json.dumps(
                {
                    "method": isolation.method,
                    "confidence": isolation.confidence,
                    "crop_box": list(isolation.crop_box),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        STYLE_ANALYSIS_CACHE.clear()
        specimen = nmns_specimen_record(destination)
        if not specimen["ready"]:
            raise ValueError("No stable primary vein and branch network could be found in this scan.")
        update_background_job(
            identifier,
            status="complete",
            progress=100,
            stage="Vein graph ready",
            result={
                "ok": True,
                "pipeline": "nmns",
                "specimen": specimen,
                "analysis": {
                    "graph_edges": specimen.get("graph_edges", 0),
                    "main_edges": specimen.get("main_edges", 0),
                    "branch_anchors": specimen.get("branch_anchors", 0),
                    "venation_type": analyzed["graph"].get("venation", {}).get("type", "unknown"),
                    "main_path_confidence": analyzed["graph"].get("venation", {}).get("main_path_confidence", 0),
                    "isolation_confidence": isolation.confidence,
                    "previews": specimen.get("analysis_urls", {}),
                    "selected_settings": {
                        "max_side": selected_settings.max_side,
                        "sensitivity": selected_settings.sensitivity,
                        "contrast_scale": selected_settings.contrast_scale,
                        "min_component": selected_settings.min_component,
                    },
                },
                "message": "Leaf isolated and vein graph ready",
            },
        )
    except Exception as exc:
        if created and destination is not None:
            destination.unlink(missing_ok=True)
        update_background_job(
            identifier,
            status="error",
            progress=100,
            stage="Could not analyze veins",
            error=f"{type(exc).__name__}: {exc}",
        )


def run_colour_job(identifier: str, sample: str, text: str, resolution: int) -> None:
    try:
        update_background_job(identifier, status="running", progress=6, stage="Preparing text")

        def report(value: int, stage: str) -> None:
            update_background_job(identifier, progress=value, stage=stage)

        capacity, selected, destination = prepare_colour_document(
            sample,
            text,
            resolution,
            progress=report,
        )
        update_background_job(
            identifier,
            status="complete",
            progress=100,
            stage="Ready" if selected["exportable"] else "Capacity result",
            result={
                "mode": "contour-colour-v3",
                "sample": sample,
                "text": text,
                "capacity": capacity,
                "selected": selected,
                "exportable": bool(selected["exportable"] and destination),
                "preview_url": (
                    f"/colour-preview/{destination.name}"
                    if destination is not None
                    else f"/colour-source/{sample}"
                ),
            },
        )
    except Exception as exc:
        update_background_job(
            identifier,
            status="error",
            progress=100,
            stage="Could not calculate",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        # Release large OpenCV arrays promptly on memory-limited cloud containers.
        gc.collect()


def run_audio_disturbance_job(
    identifier: str,
    sample: str,
    text: str,
    resolution: int,
    frames: list[dict],
    levels: int,
    config: dict,
    audio_meta: dict,
) -> None:
    try:
        update_background_job(identifier, status="running", progress=6, stage="Preparing carrier")

        def report(value: int, stage: str) -> None:
            update_background_job(
                identifier,
                progress=min(46, max(6, int(value * 0.46))),
                stage=stage,
            )

        semantic = disturb_text(text, frames, config)
        carrier_text = str(semantic["disturbed_text"])
        capacity, selected, clean_carrier = prepare_colour_document(
            sample,
            carrier_text,
            resolution,
            progress=report,
        )
        requested_resolution = resolution
        if not selected.get("exportable") or clean_carrier is None:
            fallback = next(
                (
                    row
                    for row in capacity.get("resolutions", [])
                    if row.get("exportable")
                    and int(row.get("resolution", 0)) > int(resolution)
                ),
                None,
            )
            if fallback is not None:
                resolution = int(fallback["resolution"])
                _, selected, clean_carrier = prepare_colour_document(
                    sample,
                    carrier_text,
                    resolution,
                    progress=report,
                )
        if not selected.get("exportable") or clean_carrier is None:
            raise ValueError("This leaf and resolution cannot carry the complete text packet.")

        source, _ = safe_colour_specimen(sample)
        update_background_job(identifier, progress=52, stage="Mapping sound to leaf pixels")
        output_dir = AUDIO_OUTPUT_DIR / identifier
        result = run_leaf_experiment(
            specimen_dir=source.parent,
            text=carrier_text,
            frames=frames,
            levels=levels,
            config=config,
            output_dir=output_dir,
            target_long_side=resolution,
            clean_carrier_path=clean_carrier,
        )
        update_background_job(identifier, progress=92, stage="Reading disturbed PNG")
        result["audio"] = {
            "name": str(audio_meta.get("name", "Uploaded WAV")),
            "duration": round(float(audio_meta.get("duration", 0.0)), 3),
            "impacts": int(audio_meta.get("impacts", 0)),
            "frames": len(frames),
            "levels": levels,
            "config": config,
        }
        decoded_transport = result.get("disturbed_decode") or {}
        result["semantic"] = semantic
        result["resolution"] = {
            "requested": requested_resolution,
            "used": resolution,
            "upgraded": resolution != requested_resolution,
        }
        result["transport"] = {
            "valid": bool(decoded_transport.get("valid")),
            "decoded_text": str(decoded_transport.get("text", "")),
            "source_altered": bool(semantic["source_altered"]),
        }
        result["output_urls"] = {
            name: f"/audio-output/{identifier}/{filename}"
            for name, filename in result["outputs"].items()
        }
        result["experiment_url"] = f"/audio-output/{identifier}/experiment.json"
        (output_dir / "experiment.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        update_background_job(
            identifier,
            status="complete",
            progress=100,
            stage="Decode complete",
            result=result,
        )
    except Exception as exc:
        update_background_job(
            identifier,
            status="error",
            progress=100,
            stage="Could not apply sound",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        gc.collect()


def document_cache_path(sample: str, text: str, resolution: int) -> Path:
    digest = hashlib.sha256(
        f"nmns-v1\0{sample}\0{resolution}\0{text}".encode("utf-8")
    ).hexdigest()[:16]
    return DOCUMENT_OUTPUT_DIR / f"vein_{Path(sample).stem}_{resolution}px_{digest}.png"


def prepare_nmns_document(
    sample: str,
    text: str,
    resolution: int,
    progress: Callable[[int, str], None] | None = None,
) -> tuple[dict, dict, Path | None]:
    if progress:
        progress(12, "Reading vein graph")
    image_path = safe_image(sample)
    sample_analysis = cached_analysis(image_path)
    if not analysis_ready(sample_analysis):
        raise ValueError("Analyze this NMNS scan before adding text.")
    if progress:
        progress(34, "Compiling text")
    packet = analyze_document(
        text,
        LANGUAGE_LEXICON,
        LANGUAGE_DATABASE,
        sample_analysis["metrics"],
        image_path.name,
        sample_analysis["graph"],
        resolution,
    )
    preflight = packet["export_preflight"]
    selected = next(
        row for row in preflight["resolutions"]
        if int(row["resolution"]) == int(resolution)
    )
    text_bytes = len(packet["canonical_text"].encode("utf-8"))
    normalized_rows = []
    for row in preflight["resolutions"]:
        payload = dict(row["payload"])
        required = max(1, int(payload["required"]))
        payload["safe_text_bytes"] = max(
            0, int(text_bytes * int(payload["available"]) / required)
        )
        normalized_rows.append({**row, "payload": payload})
    selected = next(
        row for row in normalized_rows
        if int(row["resolution"]) == int(resolution)
    )
    capacity = {
        "text": {"utf8_bytes": text_bytes},
        "minimum_safe_resolution": preflight["minimum_safe_resolution"],
        "recommendation": preflight["recommendation"],
        "resolutions": normalized_rows,
    }
    destination = None
    if selected["exportable"]:
        if progress:
            progress(62, "Filling natural veins")
        DOCUMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        destination = document_cache_path(sample, text, resolution)
        if not destination.is_file():
            destination.write_bytes(render_encoding_png(packet))
    if progress:
        progress(95, "Verifying packet")
    return {
        "pipeline": "nmns",
        "mode": "hierarchical-branch-document-0.9",
        "sample": sample,
        "text": text,
        "packet": packet,
        "capacity": capacity,
        "selected": selected,
        "exportable": bool(selected["exportable"] and destination),
        "preview_url": (
            f"/document-preview/{destination.name}"
            if destination is not None
            else f"/source/{image_path.name}"
        ),
        "carrier_stats": {
            "used_branches": int(packet["summary"]["used_branch_count"]),
            "branches": int(packet["summary"]["branch_count"]),
            "payload_symbols": int(packet["summary"]["payload_symbol_count"]),
        },
    }, packet, destination


def run_nmns_document_job(identifier: str, sample: str, text: str, resolution: int) -> None:
    try:
        update_background_job(identifier, status="running", progress=5, stage="Preparing text")

        def report(value: int, stage: str) -> None:
            update_background_job(identifier, progress=value, stage=stage)

        result, _, _ = prepare_nmns_document(sample, text, resolution, progress=report)
        update_background_job(
            identifier,
            status="complete",
            progress=100,
            stage="Ready" if result["exportable"] else "Capacity result",
            result=result,
        )
    except Exception as exc:
        update_background_job(
            identifier,
            status="error",
            progress=100,
            stage="Could not calculate",
            error=f"{type(exc).__name__}: {exc}",
        )


def image_files() -> list[Path]:
    if not IMAGE_DIR.exists():
        return []
    return sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def safe_image(name: str) -> Path:
    candidate = (IMAGE_DIR / Path(name).name).resolve()
    if candidate.parent != IMAGE_DIR.resolve() or not candidate.is_file():
        raise FileNotFoundError(name)
    return candidate


def analysis_paths(image_path: Path) -> tuple[Path, Path]:
    destination = OUTPUT_DIR / image_path.stem
    return destination / "metrics.json", destination / "graph.json"


def specimen_status(image_path: Path) -> dict:
    metrics_path, graph_path = analysis_paths(image_path)
    if not metrics_path.is_file() or not graph_path.is_file():
        return {"prepared": False, "ready": False}
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        stats = graph.get("stats", {})
        ready = bool(
            graph.get("version") == "leaf-surface-graph-0.8"
            and graph.get("main_path", {}).get("edge_ids")
            and int(stats.get("edge_count", 0)) >= 8
        )
        return {
            "prepared": True,
            "ready": ready,
            "graph_edges": int(stats.get("edge_count", 0)),
            "main_edges": int(stats.get("main_edge_count", 0)),
            "branch_anchors": int(stats.get("secondary_anchor_count", 0)),
            "venation_type": str(stats.get("venation_type", "unknown")),
            "main_path_confidence": float(stats.get("main_path_confidence", 0)),
        }
    except (OSError, ValueError, TypeError):
        return {"prepared": False, "ready": False}


def cached_analysis(
    image_path: Path,
    settings: AnalysisSettings | None = None,
) -> dict:
    metrics_path, graph_path = analysis_paths(image_path)
    use_exported = settings is None and metrics_path.is_file() and graph_path.is_file()
    if use_exported:
        try:
            exported = {
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
                "graph": json.loads(graph_path.read_text(encoding="utf-8")),
            }
        except (OSError, ValueError, TypeError):
            exported = None
        if exported is not None and analysis_ready(exported):
            cache_key = (
                f"export:{image_path}:{image_path.stat().st_mtime_ns}:"
                f"{metrics_path.stat().st_mtime_ns}:{graph_path.stat().st_mtime_ns}"
            )
            cached = STYLE_ANALYSIS_CACHE.get(cache_key)
            if cached is None:
                cached = exported
                STYLE_ANALYSIS_CACHE[cache_key] = cached
            return cached

    active_settings = settings or AnalysisSettings()
    settings_key = (
        active_settings.max_side,
        active_settings.sensitivity,
        active_settings.contrast_scale,
        active_settings.min_component,
    )
    cache_key = f"live:{image_path}:{image_path.stat().st_mtime_ns}:{settings_key}"
    cached = STYLE_ANALYSIS_CACHE.get(cache_key)
    if cached is None:
        analyzed = analyze_image(image_path, active_settings)
        cached = {"metrics": analyzed["metrics"], "graph": analyzed["graph"]}
        if settings is None:
            export_result(analyzed, OUTPUT_DIR)
        STYLE_ANALYSIS_CACHE[cache_key] = cached
    return cached


def analysis_ready(result: dict) -> bool:
    graph = result.get("graph", {})
    confidence = float(graph.get("venation", {}).get("main_path_confidence", graph.get("stats", {}).get("main_path_confidence", 0)))
    return bool(
        graph.get("version") == "leaf-surface-graph-0.8"
        and graph.get("main_path", {}).get("edge_ids")
        and int(graph.get("stats", {}).get("edge_count", 0)) >= 8
        and confidence >= 28
    )


def adaptive_analysis(image_path: Path, settings: AnalysisSettings) -> tuple[dict, AnalysisSettings]:
    candidates = [settings]
    if settings == AnalysisSettings():
        candidates.extend(
            [
                AnalysisSettings(settings.max_side, 68.0, 9.0, 4),
                AnalysisSettings(settings.max_side, 80.0, 16.0, 6),
                AnalysisSettings(settings.max_side, 62.0, 20.0, 3),
            ]
        )
    best_result = None
    best_settings = settings
    best_score = float("-inf")
    for candidate in candidates:
        result = analyze_image(image_path, candidate)
        stats = result["graph"].get("stats", {})
        edges = int(stats.get("edge_count", 0))
        main_edges = int(stats.get("main_edge_count", 0))
        anchors = int(stats.get("secondary_anchor_count", 0))
        confidence = float(result["graph"].get("venation", {}).get("main_path_confidence", 0))
        score = confidence * 85 + anchors * 8 + min(edges, 2200)
        score -= max(0, main_edges - 55) * 35
        if not result["graph"].get("main_path", {}).get("edge_ids"):
            score -= 100_000
        if edges > 7000:
            score -= (edges - 7000) * 2
        if score > best_score:
            best_result = result
            best_settings = candidate
            best_score = score
        if candidate == settings and analysis_ready(result):
            break
    if best_result is None:
        raise ValueError("The specimen could not be analyzed.")
    return best_result, best_settings


def normalized_upload_name(name: str, raw: bytes) -> str:
    supplied = Path(name).name
    if Path(supplied).suffix.lower() != ".png":
        raise ValueError("Only PNG specimen files are supported.")
    stem = re.sub(r"[^\w.-]+", "_", Path(supplied).stem, flags=re.UNICODE).strip("._")
    if not stem:
        stem = f"leaf_{abs(hash(raw)) & 0xFFFFFFFF:08x}"
    candidate = IMAGE_DIR / f"{stem}.png"
    if candidate.is_file() and candidate.read_bytes() == raw:
        return candidate.name
    index = 2
    while candidate.exists():
        candidate = IMAGE_DIR / f"{stem}_{index}.png"
        index += 1
    return candidate.name


def decode_upload(body: dict) -> tuple[str, bytes, np.ndarray]:
    encoded = str(body.get("data", ""))
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("The uploaded PNG data is invalid.") from exc
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("PNG files must be between 1 byte and 32 MB.")
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("The uploaded file is not a valid PNG.")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in {2, 3}:
        raise ValueError("The uploaded PNG could not be decoded.")
    height, width = image.shape[:2]
    if width < 128 or height < 128:
        raise ValueError("The specimen must be at least 128 × 128 pixels.")
    if width * height > MAX_UPLOAD_PIXELS:
        raise ValueError("The specimen exceeds the 120-megapixel safety limit.")
    return normalized_upload_name(str(body.get("filename", "leaf.png")), raw), raw, image


def decode_png_payload(body: dict) -> bytes:
    encoded = str(body.get("data", ""))
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("The uploaded PNG data is invalid.") from exc
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("PNG files must be between 1 byte and 32 MB.")
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("The uploaded file is not a valid PNG.")
    return raw


def prepared_specimen_graphs() -> list[dict]:
    specimens = []
    for image_path in image_files():
        status = specimen_status(image_path)
        if not status.get("ready"):
            continue
        _, graph_path = analysis_paths(image_path)
        try:
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        specimens.append({"name": image_path.name, "graph": graph})
    return specimens


class LeafToolHandler(BaseHTTPRequestHandler):
    server_version = "LeafAnalysisTool/0.1"

    def log_message(self, format: str, *args: object) -> None:
        print(f"[leaf-tool] {self.address_string()} - {format % args}")

    def _json(self, payload: object, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            # The browser can cancel a long analysis request when the page is
            # reloaded. That is a normal disconnect, not an analysis failure.
            return

    def _file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        raw = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {
            "application/javascript",
            "application/json",
        }:
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _binary(self, payload: bytes, content_type: str, filename: str) -> None:
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        if length > MAX_REQUEST_BYTES:
            raise ValueError("The request exceeds the 64 MB safety limit.")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._json(
                {
                    "status": "ok",
                    "service": "leaf-language-engine",
                    "revision": os.environ.get("RAILWAY_GIT_COMMIT_SHA", "local")[:12],
                    "features": ["rubbing", "nmns", "audio-hard-decode", "png-readback"],
                    "images": len(image_files()),
                    "data_persistent": DATA_DIR != LOCAL_PROJECT_DIR,
                }
            )
            return
        if path == "/":
            self._file(STATIC_DIR / "language.html")
            return
        if path in {"/language", "/carrier"}:
            self._file(STATIC_DIR / "language.html")
            return
        if path == "/analysis":
            self.send_response(302)
            self.send_header("Location", "/?panel=analysis")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path.startswith("/api/jobs/"):
            try:
                self._json(public_background_job(Path(path).name))
            except FileNotFoundError:
                self.send_error(404)
            return
        if path == "/api/images":
            items = []
            for image_path in image_files():
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                height, width = image.shape[:2] if image is not None else (0, 0)
                channels = (
                    1 if image is not None and image.ndim == 2
                    else int(image.shape[2]) if image is not None
                    else 0
                )
                item = nmns_specimen_record(image_path)
                item.update({"channels": channels, "display_name": item["name"]})
                items.append(item)
            self._json({"images": items, "image_dir": str(IMAGE_DIR), "output_dir": str(OUTPUT_DIR)})
            return
        if path == "/api/language/status":
            self._json(LanguageDatabase(LANGUAGE_DATABASE).status())
            return
        if path == "/api/runtime":
            self._json(
                {
                    "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
                    "rubbing": {
                        "resolutions": list(RUBBING_RESOLUTIONS),
                        "max_resolution": MAX_RUBBING_RESOLUTION,
                        "safe_mode": MAX_RUBBING_RESOLUTION < max(RUBBING_RESOLUTIONS),
                    },
                    "nmns": {
                        "resolutions": [2048, 4096, 8192, 16384],
                    },
                }
            )
            return
        if path == "/api/colour/specimens":
            self._json(
                {
                    "specimens": colour_specimens(),
                    "curated_directory": str(RUBBING_DIR),
                    "uploaded_directory": str(UPLOADED_SPECIMEN_DIR),
                }
            )
            return
        if path.startswith("/audio-output/"):
            relative = unquote(path.removeprefix("/audio-output/")).lstrip("/")
            candidate = (AUDIO_OUTPUT_DIR / relative).resolve()
            try:
                candidate.relative_to(AUDIO_OUTPUT_DIR.resolve())
            except ValueError:
                self.send_error(403)
                return
            if candidate.is_file():
                self._file(candidate)
            else:
                self.send_error(404)
            return
        if path.startswith("/colour-source/"):
            try:
                source, _ = safe_colour_specimen(unquote(path.removeprefix("/colour-source/")))
                self._file(source)
            except FileNotFoundError:
                self.send_error(404)
            return
        if path.startswith("/colour-preview/"):
            candidate = (COLOUR_OUTPUT_DIR / Path(unquote(path.removeprefix("/colour-preview/"))).name).resolve()
            if candidate.parent == COLOUR_OUTPUT_DIR.resolve() and candidate.is_file():
                self._file(candidate)
            else:
                self.send_error(404)
            return
        if path.startswith("/document-preview/"):
            candidate = (
                DOCUMENT_OUTPUT_DIR
                / Path(unquote(path.removeprefix("/document-preview/"))).name
            ).resolve()
            if candidate.parent == DOCUMENT_OUTPUT_DIR.resolve() and candidate.is_file():
                self._file(candidate)
            else:
                self.send_error(404)
            return
        if path.startswith("/source/"):
            try:
                self._file(safe_image(unquote(path.removeprefix("/source/"))))
            except FileNotFoundError:
                self.send_error(404)
            return
        if path.startswith("/analysis-preview/"):
            relative = unquote(path.removeprefix("/analysis-preview/")).lstrip("/")
            candidate = (OUTPUT_DIR / relative).resolve()
            try:
                candidate.relative_to(OUTPUT_DIR.resolve())
            except ValueError:
                self.send_error(403)
                return
            if candidate.is_file():
                self._file(candidate)
            else:
                self.send_error(404)
            return
        if path.startswith("/static/"):
            candidate = (STATIC_DIR / path.removeprefix("/static/")).resolve()
            if candidate.parent == STATIC_DIR.resolve():
                self._file(candidate)
            else:
                self.send_error(403)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            body = self._body()
            if self.path == "/api/jobs/specimen":
                raw, filename = decode_specimen_source_upload(body)
                source_hint = str(body.get("source_hint", "auto"))
                identifier = create_background_job("specimen")
                launch_background_job(
                    identifier,
                    run_specimen_job,
                    raw,
                    filename,
                    source_hint,
                )
                self._json({"job_id": identifier}, 202)
                return

            if self.path == "/api/jobs/colour":
                sample = str(body.get("sample", "")).strip()
                text = str(body.get("text", "")).strip()
                resolution = int(body.get("target_resolution", 4096))
                if not sample:
                    raise ValueError("Choose a leaf first.")
                if not text:
                    raise ValueError("Enter some text first.")
                safe_colour_specimen(sample)
                identifier = create_background_job("colour")
                launch_background_job(identifier, run_colour_job, sample, text, resolution)
                self._json({"job_id": identifier}, 202)
                return

            if self.path == "/api/jobs/carrier":
                pipeline = str(body.get("pipeline", "rubbing")).strip().lower()
                sample = str(body.get("sample", "")).strip()
                text = str(body.get("text", "")).strip()
                resolution = int(body.get("target_resolution", 4096))
                if not sample:
                    raise ValueError("Choose a leaf first.")
                if not text:
                    raise ValueError("Enter some text first.")
                if pipeline == "nmns":
                    safe_image(sample)
                    identifier = create_background_job("nmns-document")
                    target = run_nmns_document_job
                elif pipeline == "rubbing":
                    safe_colour_specimen(sample)
                    identifier = create_background_job("rubbing-document")
                    target = run_colour_job
                else:
                    raise ValueError("Choose either the leaf-rubbing or NMNS pipeline.")
                launch_background_job(identifier, target, sample, text, resolution)
                self._json({"job_id": identifier}, 202)
                return

            if self.path == "/api/jobs/audio-disturbance":
                sample = str(body.get("sample", "")).strip()
                text = str(body.get("text", "")).strip()
                resolution = int(body.get("target_resolution", 4096))
                raw_frames = body.get("audio_array")
                if not sample:
                    raise ValueError("Choose a leaf first.")
                if not text:
                    raise ValueError("Enter some text first.")
                safe_colour_specimen(sample)
                if not isinstance(raw_frames, list) or not raw_frames:
                    raise ValueError("Choose and analyze a WAV file first.")
                if len(raw_frames) > 6000:
                    raise ValueError("The WAV array is too long. Use a shorter recording.")
                levels = max(4, min(32, int(body.get("levels", 16))))
                frame_keys = ("time", "energy", "peak", "bass", "brightness", "noise", "impact")
                frames = []
                for raw_frame in raw_frames:
                    if not isinstance(raw_frame, dict):
                        continue
                    frame = {}
                    for key in frame_keys:
                        try:
                            value = float(raw_frame.get(key, 0))
                        except (TypeError, ValueError):
                            value = 0.0
                        frame[key] = max(0.0, min(float(levels - 1), value))
                    frames.append(frame)
                if not frames:
                    raise ValueError("The WAV array does not contain usable feature frames.")
                defaults = {
                    "strength": 0.24,
                    "shift": 0.40,
                    "erasure": 0.12,
                    "flip": 0.05,
                    "clustering": 0.55,
                }
                raw_config = body.get("config") if isinstance(body.get("config"), dict) else {}
                config = {}
                for key, default in defaults.items():
                    try:
                        value = float(raw_config.get(key, default))
                    except (TypeError, ValueError):
                        value = default
                    config[key] = max(0.0, min(1.0, value))
                semantic_mode = str(raw_config.get("semantic_mode", "trace")).strip().lower()
                config["semantic_mode"] = (
                    semantic_mode
                    if semantic_mode in {"preserve", "trace", "breakdown"}
                    else "trace"
                )
                try:
                    semantic_loss = float(raw_config.get("semantic_loss", 0.06))
                except (TypeError, ValueError):
                    semantic_loss = 0.06
                config["semantic_loss"] = max(0.0, min(0.80, semantic_loss))
                audio_meta = body.get("audio") if isinstance(body.get("audio"), dict) else {}
                identifier = create_background_job("audio-disturbance")
                launch_background_job(
                    identifier,
                    run_audio_disturbance_job,
                    sample,
                    text,
                    resolution,
                    frames,
                    levels,
                    config,
                    audio_meta,
                )
                self._json({"job_id": identifier}, 202)
                return

            if self.path == "/api/specimens/upload":
                filename, raw, decoded_image = decode_upload(body)
                destination = IMAGE_DIR / filename
                created = not destination.exists()
                if created:
                    destination.write_bytes(raw)
                settings = AnalysisSettings.from_dict(body.get("settings"))
                try:
                    analyzed, selected_settings = adaptive_analysis(destination, settings)
                    export_destination = export_result(analyzed, OUTPUT_DIR)
                    STYLE_ANALYSIS_CACHE.clear()
                except Exception:
                    if created:
                        destination.unlink(missing_ok=True)
                    raise
                channels = 1 if decoded_image.ndim == 2 else int(decoded_image.shape[2])
                response = {
                    "ok": True,
                    "image": {
                        "name": filename,
                        "width": int(decoded_image.shape[1]),
                        "height": int(decoded_image.shape[0]),
                        "channels": channels,
                        **specimen_status(destination),
                    },
                    "selected_settings": {
                        "max_side": selected_settings.max_side,
                        "sensitivity": selected_settings.sensitivity,
                        "contrast_scale": selected_settings.contrast_scale,
                        "min_component": selected_settings.min_component,
                    },
                    "destination": str(export_destination),
                }
                if body.get("include_previews"):
                    response["result"] = public_result(analyzed)
                self._json(response, 201)
                return

            if self.path == "/api/analyze":
                image_path = safe_image(str(body.get("filename", "")))
                settings = AnalysisSettings.from_dict(body.get("settings"))
                result = analyze_image(image_path, settings)
                self._json(public_result(result))
                return

            if self.path == "/api/language/analyze":
                result = analyze_word(
                    str(body.get("word", "")),
                    LANGUAGE_LEXICON,
                    LANGUAGE_DATABASE,
                )
                self._json(result)
                return

            if self.path == "/api/language/colour/analyze":
                sample = str(body.get("sample", "")).strip()
                text = str(body.get("text", ""))
                resolution = int(body.get("target_resolution", 4096))
                capacity, selected, destination = prepare_colour_document(
                    sample, text, resolution
                )
                self._json(
                    {
                        "mode": "contour-colour-v3",
                        "sample": sample,
                        "text": text,
                        "capacity": capacity,
                        "selected": selected,
                        "exportable": bool(selected["exportable"] and destination),
                        "preview_url": (
                            f"/colour-preview/{destination.name}"
                            if destination is not None
                            else f"/colour-source/{sample}"
                        ),
                    }
                )
                return

            if self.path == "/api/colour/specimens/upload":
                raw, filename = decode_specimen_source_upload(body)
                metadata = ingest_specimen(
                    raw,
                    filename,
                    UPLOADED_SPECIMEN_DIR,
                    source_hint=str(body.get("source_hint", "auto")),
                )
                directory = UPLOADED_SPECIMEN_DIR / str(metadata["id"])
                specimen = colour_specimen_record(directory, "uploaded")
                self._json(
                    {
                        "ok": True,
                        "specimen": specimen,
                        "analysis": metadata,
                        "message": (
                            "Leaf source is ready for text capacity analysis."
                            if specimen["ready"]
                            else "Leaf source was isolated, but image quality needs review."
                        ),
                    },
                    201,
                )
                return

            if self.path == "/api/language/colour/export":
                sample = str(body.get("sample", "")).strip()
                text = str(body.get("text", ""))
                resolution = int(body.get("target_resolution", 4096))
                _, selected, destination = prepare_colour_document(sample, text, resolution)
                if not selected["exportable"] or destination is None:
                    raise ValueError("JPG export requires a resolution that passes both contour and texture capacity checks.")
                filename = f"leaf_colour_{Path(sample).name}_{resolution}px.jpg"
                self._binary(destination.read_bytes(), "image/jpeg", filename)
                return

            if self.path == "/api/language/colour/decode":
                raw, suffix = decode_colour_upload(body)
                COLOUR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(raw).hexdigest()[:16]
                temporary = COLOUR_OUTPUT_DIR / f"decode_{digest}{suffix}"
                temporary.write_bytes(raw)
                try:
                    result = decode_contour_colour(temporary)
                finally:
                    temporary.unlink(missing_ok=True)
                self._json(result)
                return

            if self.path == "/api/language/document":
                sample_filename = str(body.get("sample", "")).strip()
                sample_metrics = None
                sample_graph = None
                if sample_filename:
                    image_path = safe_image(sample_filename)
                    sample_analysis = cached_analysis(image_path)
                    sample_metrics = sample_analysis["metrics"]
                    sample_graph = sample_analysis["graph"]
                result = analyze_document(
                    str(body.get("text", "")),
                    LANGUAGE_LEXICON,
                    LANGUAGE_DATABASE,
                    sample_metrics,
                    sample_filename or None,
                    sample_graph,
                    int(body.get("target_resolution", 8192)),
                )
                self._json(result)
                return

            if self.path == "/api/language/document/decode":
                packet = body.get("packet", body)
                if not isinstance(packet, dict):
                    raise ValueError("The document packet must be a JSON object.")
                self._json(decode_document(packet))
                return

            if self.path == "/api/language/document/decode-png":
                result = decode_encoding_png(
                    decode_png_payload(body),
                    prepared_specimen_graphs(),
                    LANGUAGE_DATABASE,
                )
                self._json(result)
                return

            if self.path == "/api/language/document/export":
                packet = body.get("packet", body)
                if not isinstance(packet, dict):
                    raise ValueError("The document packet must be a JSON object.")
                verification = decode_document(packet)
                if not verification.get("valid"):
                    raise ValueError(
                        "PNG export requires a complete packet with valid CRC and sufficient specimen capacity."
                    )
                png = render_encoding_png(packet)
                self._binary(png, "image/png", encoding_export_filename(packet))
                return

            if self.path == "/api/codec/encode":
                self._json(encode_text(str(body.get("text", ""))))
                return

            if self.path == "/api/codec/decode":
                packet = body.get("packet", body)
                if not isinstance(packet, dict):
                    raise ValueError("The packet must be a JSON object.")
                self._json(decode_packet(packet))
                return

            if self.path == "/api/export":
                image_path = safe_image(str(body.get("filename", "")))
                settings = AnalysisSettings.from_dict(body.get("settings"))
                result = analyze_image(image_path, settings)
                destination = export_result(result, OUTPUT_DIR)
                STYLE_ANALYSIS_CACHE.clear()
                self._json({"ok": True, "destination": str(destination), "result": public_result(result)})
                return

            if self.path == "/api/export-all":
                settings = AnalysisSettings.from_dict(body.get("settings"))
                rows = []
                for image_path in image_files():
                    result = analyze_image(image_path, settings)
                    export_result(result, OUTPUT_DIR)
                    rows.append(result["metrics"])
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                summary = OUTPUT_DIR / "summary.csv"
                if rows:
                    with summary.open("w", newline="", encoding="utf-8-sig") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(rows)
                self._json(
                    {
                        "ok": True,
                        "count": len(rows),
                        "destination": str(OUTPUT_DIR),
                        "summary": str(summary),
                    }
                )
                return
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            return
        except FileNotFoundError:
            self._json({"error": "The selected image could not be found."}, 404)
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def open_browser() -> None:
    time.sleep(0.8)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    prepare_runtime_data()
    if OPEN_BROWSER:
        threading.Thread(target=open_browser, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), LeafToolHandler)
    print(f"Leaf analysis tool: http://{HOST}:{PORT}")
    print(f"Runtime data: {DATA_DIR}")
    print(f"Images: {IMAGE_DIR}")
    print(f"Outputs: {OUTPUT_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
